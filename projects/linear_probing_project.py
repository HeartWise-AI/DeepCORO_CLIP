import os
import torch
from typing import Any, Optional, Dict
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import itertools

from runners.typing import Runner
from models.video_encoder import VideoEncoder
from models.multi_instance_linear_probing import MultiInstanceLinearProbing
from projects.base_project import BaseProject
from utils.registry import (
    ProjectRegistry, 
    RunnerRegistry, 
    ModelRegistry,
    LossRegistry
)

from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.enums import RunMode, LossType
from utils.schedulers import get_scheduler
from utils.wandb_wrapper import WandbWrapper
from utils.video_project import calculate_dataset_statistics_ddp
from utils.config.linear_probing_config import LinearProbingConfig
from dataloaders.video_dataset import get_distributed_video_dataloader

class VideoMILWrapper(torch.nn.Module):
    def __init__(self, video_encoder, mil_model, num_videos: int):
        """Wrapper around *VideoEncoder* and *MultiInstanceLinearProbing*.

        Args
        ----
        video_encoder: Backbone that outputs either per-video embeddings
            ``[B, N, D]`` or per-patch embeddings ``[B, N_tokens, D]`` where
            tokens are ordered consecutively for each video.
        mil_model:  Multi-Instance head (e.g. *MultiInstanceLinearProbing*).
        num_videos: Expected number of videos/segments per sample (*N*).
            This is needed to reshape flat per-patch tokens into
            ``[B, N, L, D]`` so that the downstream MIL model can perform
            hierarchical pooling.
        """
        super().__init__()
        self.video_encoder = video_encoder
        self.mil_model = mil_model
        self.num_videos: int = num_videos

    def forward(
        self,
        x: torch.Tensor,
        video_indices: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Wrapper forward pass.

        This helper guarantees that the **MultiInstanceLinearProbing** module
        always receives a 3-D tensor of shape ``[B, N, D]`` where *N* is the
        number of video segments associated with each sample.

        Args
        ----
        x:  Input videos tensor.
        video_indices:  Optional mapping from videos to batch items.
        video_mask: Optional ``[B, N]`` boolean tensor where ``True`` marks
            a real video and ``False`` marks a padded/skipped slot.
        view_ids:  Optional ``[B, N]`` long tensor of per-video view class IDs
            (EchoJEPA-style angle embeddings).  Forwarded to the MIL model.
        """

        # ------------------------------------------------------------------
        # 1) Run the backbone / encoder
        # ------------------------------------------------------------------
        embeddings: torch.Tensor = self.video_encoder(x)
        index_mask: Optional[torch.Tensor] = None
        if video_indices is not None and x.dim() == 5:
            embeddings, index_mask, view_ids = self._regroup_flat_embeddings(
                embeddings=embeddings,
                video_indices=video_indices,
                video_mask=video_mask,
                view_ids=view_ids,
            )

        # ------------------------------------------------------------------
        # 2) Reshape so that the MIL module always sees *either*:
        #    • 3-D tensor  [B, N, D]  (per-video embeddings)
        #    • 4-D tensor  [B, N, L, D] (per-patch tokens, hierarchical)
        # ------------------------------------------------------------------

        if embeddings.ndim == 2:
            # Encoder returned [B, D] → add singleton video dimension
            embeddings = embeddings.unsqueeze(1)  # [B, 1, D]

        elif embeddings.ndim == 3 and embeddings.shape[1] > self.num_videos:
            # Received flat patch tokens [B, N*L, D].  Reshape into
            # hierarchical layout [B, N, L, D] so that the MIL head can
            # perform two-level attention.
            B, NL, D = embeddings.shape  # noqa: N806
            if NL % self.num_videos != 0:
                raise ValueError(
                    f"Number of tokens (NL={NL}) is not divisible by the "
                    f"expected num_videos={self.num_videos}. Cannot "
                    "infer tokens per video for hierarchical pooling."
                )
            L = NL // self.num_videos
            embeddings = embeddings.view(B, self.num_videos, L, D)  # [B,N,L,D]

        # ------------------------------------------------------------------
        # 3) Build attention mask (video-level only for now)
        # ------------------------------------------------------------------
        if embeddings.ndim == 4:
            B, N, _, _ = embeddings.shape
        else:
            B, N, _ = embeddings.shape  # type: ignore[misc]

        if video_mask is not None:
            attention_mask = self._coerce_video_mask(
                video_mask=video_mask,
                batch_size=B,
                num_instances=N,
                device=embeddings.device,
            )
            if index_mask is not None:
                attention_mask = attention_mask & self._coerce_video_mask(
                    video_mask=index_mask,
                    batch_size=B,
                    num_instances=N,
                    device=embeddings.device,
                )
        elif index_mask is not None:
            attention_mask = self._coerce_video_mask(
                video_mask=index_mask,
                batch_size=B,
                num_instances=N,
                device=embeddings.device,
            )
        elif x.dim() == 6:  # multi-video input [B, N, F, H, W, C]
            with torch.no_grad():
                inferred_mask = x.reshape(x.shape[0], x.shape[1], -1).abs().amax(dim=-1) > 0
            attention_mask = self._coerce_video_mask(
                video_mask=inferred_mask,
                batch_size=B,
                num_instances=N,
                device=embeddings.device,
            )
        else:
            attention_mask = torch.ones((B, N), dtype=torch.bool, device=embeddings.device)

        view_ids = self._coerce_view_ids(
            view_ids=view_ids,
            batch_size=B,
            num_instances=N,
            device=embeddings.device,
        )

        # ------------------------------------------------------------------
        # 4) Forward through the MIL head(s)
        # ------------------------------------------------------------------
        return self.mil_model(embeddings, mask=attention_mask, view_ids=view_ids)

    @staticmethod
    def _coerce_video_mask(
        video_mask: torch.Tensor,
        batch_size: int,
        num_instances: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Align a dataloader video mask with the tensor shape seen by MIL."""
        attention_mask = video_mask.to(device=device, dtype=torch.bool)
        expected_shape = (batch_size, num_instances)
        if tuple(attention_mask.shape) == expected_shape:
            return attention_mask

        if (
            attention_mask.ndim == 2
            and attention_mask.shape[0] == batch_size
            and num_instances == 1
        ):
            return attention_mask.any(dim=1, keepdim=True)

        raise ValueError(
            f"video_mask shape {tuple(attention_mask.shape)} does not match "
            f"MIL instance shape {expected_shape}"
        )

    def _regroup_flat_embeddings(
        self,
        embeddings: torch.Tensor,
        video_indices: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Group flat per-video encoder outputs into [B, N, ...] MIL layout."""
        if video_indices.ndim != 1:
            raise ValueError(
                f"video_indices must be a 1D tensor, got shape {tuple(video_indices.shape)}"
            )
        if embeddings.shape[0] != video_indices.numel():
            raise ValueError(
                f"video_indices length {video_indices.numel()} does not match "
                f"flat embedding batch size {embeddings.shape[0]}"
            )
        if video_indices.numel() == 0:
            raise ValueError("video_indices cannot be empty for flat multi-video inputs")

        video_indices = video_indices.to(device=embeddings.device, dtype=torch.long)
        if video_mask is not None:
            if video_mask.ndim != 2:
                raise ValueError(
                    f"video_mask must be 2D when grouping flat embeddings, got {tuple(video_mask.shape)}"
                )
            batch_size, num_instances = video_mask.shape
        else:
            batch_size = int(video_indices.max().item()) + 1
            num_instances = self.num_videos

        slot_indices = self._slot_indices_for_flat_videos(
            video_indices=video_indices,
            batch_size=batch_size,
            num_instances=num_instances,
        )

        grouped = embeddings.new_zeros((batch_size, num_instances, *embeddings.shape[1:]))
        grouped[video_indices, slot_indices] = embeddings

        index_mask = torch.zeros(
            (batch_size, num_instances),
            dtype=torch.bool,
            device=embeddings.device,
        )
        index_mask[video_indices, slot_indices] = True

        grouped_view_ids = self._regroup_flat_view_ids(
            view_ids=view_ids,
            video_indices=video_indices,
            slot_indices=slot_indices,
            batch_size=batch_size,
            num_instances=num_instances,
            device=embeddings.device,
        )
        return grouped, index_mask, grouped_view_ids

    def _regroup_flat_view_ids(
        self,
        view_ids: Optional[torch.Tensor],
        video_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        batch_size: int,
        num_instances: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Group flat per-video view IDs into [B, N], preserving shaped inputs."""
        if view_ids is None:
            return None

        view_ids = view_ids.to(device=device, dtype=torch.long)
        if view_ids.ndim == 2:
            return view_ids
        if view_ids.ndim != 1:
            raise ValueError(f"view_ids must be 1D or 2D, got shape {tuple(view_ids.shape)}")
        if view_ids.numel() != video_indices.numel():
            raise ValueError(
                f"view_ids length {view_ids.numel()} does not match "
                f"flat video count {video_indices.numel()}"
            )

        pad_id = int(getattr(self.mil_model, "view_pad_id", 0))
        grouped_view_ids = torch.full(
            (batch_size, num_instances),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        grouped_view_ids[video_indices, slot_indices] = view_ids
        return grouped_view_ids

    @staticmethod
    def _coerce_view_ids(
        view_ids: Optional[torch.Tensor],
        batch_size: int,
        num_instances: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Align optional per-video view IDs with the MIL instance layout."""
        if view_ids is None:
            return None

        view_ids = view_ids.to(device=device, dtype=torch.long)
        expected_shape = (batch_size, num_instances)
        if tuple(view_ids.shape) == expected_shape:
            return view_ids

        if view_ids.ndim == 2 and view_ids.shape[0] == batch_size and num_instances == 1:
            # The encoder has already collapsed multiple videos into one
            # embedding, so per-video view IDs no longer have a valid axis.
            return None

        raise ValueError(
            f"view_ids shape {tuple(view_ids.shape)} does not match "
            f"MIL instance shape {expected_shape}"
        )

    @staticmethod
    def _slot_indices_for_flat_videos(
        video_indices: torch.Tensor,
        batch_size: int,
        num_instances: int,
    ) -> torch.Tensor:
        """Assign each flat video to its per-study slot by encounter order."""
        counts = torch.zeros(batch_size, dtype=torch.long, device=video_indices.device)
        slot_indices = torch.empty_like(video_indices)
        for flat_idx, batch_idx in enumerate(video_indices.tolist()):
            if batch_idx < 0 or batch_idx >= batch_size:
                raise ValueError(
                    f"video_indices contains batch index {batch_idx}, "
                    f"outside expected range [0, {batch_size})"
                )
            slot_idx = int(counts[batch_idx].item())
            if slot_idx >= num_instances:
                raise ValueError(
                    f"Sample {batch_idx} has more than {num_instances} videos in flat batch"
                )
            slot_indices[flat_idx] = slot_idx
            counts[batch_idx] += 1
        return slot_indices

@ProjectRegistry.register("DeepCORO_video_linear_probing")
@ProjectRegistry.register("DeepCORO_video_linear_probing_cardio_syntax")
class LinearProbingProject(BaseProject):
    def __init__(
        self, 
        config: LinearProbingConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)

    def _disable_encoder_aggregation_for_mil(self) -> None:
        """Linear probing must preserve per-instance encoder outputs for MIL."""
        if not getattr(self.config, "aggregate_videos_tokens", False):
            return

        if self.config.is_ref_device:
            print(
                "[WARNING] aggregate_videos_tokens=True detected but "
                "should be False for linear-probing. This is only used for CLIP. "
                "Overriding to False so that the VideoEncoder preserves "
                "per-instance tokens. This override is logged to wandb."
            )
        self.config.aggregate_videos_tokens = False
        if self.wandb_wrapper.is_initialized():
            self.wandb_wrapper.log({"config/aggregate_videos_tokens_override": True})

    def _setup_training_objects(
        self
    )->dict[str, Any]:
        self._disable_encoder_aggregation_for_mil()
                
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)        
        
        # View embedding parameters — only enable when view_column is set in the YAML
        view_column = getattr(self.config, 'view_column', None)
        view_labels_map = getattr(self.config, 'view_labels_map', None)
        if view_column:
            num_view_classes = getattr(self.config, 'num_view_classes', 0)
            print(f"View embeddings ENABLED: view_column='{view_column}', {num_view_classes} view classes + 1 PAD")
        else:
            num_view_classes = 0
            print("View embeddings DISABLED (no view_column in config)")

        # Get dataloaders with multi-video parameters
        train_loader: DataLoader = get_distributed_video_dataloader(
            config=self.config,
            split=RunMode.TRAIN,
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=True,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
            multi_video=self.config.multi_video,
            groupby_column=self.config.groupby_column,
            num_videos=self.config.num_videos,
            shuffle_videos=self.config.shuffle_videos,
            labels_map=getattr(self.config, 'labels_map', None),
            view_column=view_column,
            view_labels_map=view_labels_map,
            num_view_classes=num_view_classes,
        )
        val_loader: DataLoader = get_distributed_video_dataloader(
            config=self.config,
            split=RunMode.VALIDATE,
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
            multi_video=self.config.multi_video,
            groupby_column=self.config.groupby_column,
            num_videos=self.config.num_videos,
            shuffle_videos=False,  # Don't shuffle validation videos
            labels_map=getattr(self.config, 'labels_map', None),
            view_column=view_column,
            view_labels_map=view_labels_map,
            num_view_classes=num_view_classes,
        )        
        
        # Initialize video encoder backbone for linear probing
        video_encoder: VideoEncoder = ModelRegistry.get("video_encoder")(
            backbone=self.config.model_name,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
            aggregate_videos_tokens=self.config.aggregate_videos_tokens,
            per_video_pool=self.config.per_video_pool,
        )        
        
        # Get embedding dimension from encoder
        embedding_dim = video_encoder.embedding_dim

        # Load video encoder checkpoint 
        video_encoder = video_encoder.to(self.config.device).float()
        checkpoint: Dict[str, Any] = self._load_checkpoint(self.config.video_encoder_checkpoint_path)       
        video_encoder.load_state_dict(checkpoint["video_encoder"])

        # Freeze video encoder if specified
        if self.config.video_freeze_ratio == 1.0:
            for param in video_encoder.parameters():
                param.requires_grad = False
            video_encoder.eval() # Set to eval mode if fully frozen

        # Initialize Multi-Instance Linear Probing model
        print(f"Initializing Multi-Instance Linear Probing model with pooling mode: {self.config.pooling_mode}")
        mil_model: MultiInstanceLinearProbing = ModelRegistry.get("multi_instance_linear_probing")(
            embedding_dim=embedding_dim,
            head_structure=self.config.head_structure,
            pooling_mode=self.config.pooling_mode,
            attention_hidden=self.config.attention_hidden,
            dropout=self.config.dropout_attention,
            num_view_classes=num_view_classes,
        )
        mil_model = mil_model.to(self.config.device).float()

        # Wrap both models
        linear_probing = VideoMILWrapper(video_encoder, mil_model, self.config.num_videos)
                
        # Initialize optimizer with separate learning rates
        param_groups = []

        # Add video encoder parameters if not fully frozen
        if self.config.video_freeze_ratio < 1.0:
            param_groups.append({
                'params': video_encoder.parameters(), 
                'lr': self.config.video_encoder_lr, 
                'name': 'video_encoder',
                'weight_decay': self.config.video_encoder_weight_decay
            })

        # Add MIL model parameters (pooling layers, heads)
        # Add head parameters
        for head_name in self.config.head_structure:
            param_groups.append({
                'params': mil_model.heads[head_name].parameters(),
                'lr': self.config.head_lr[head_name],
                'name': head_name,
                'weight_decay': self.config.head_weight_decay[head_name]
            })
            
        # Add attention parameters if applicable (potentially different LR/WD)
        if "attention" in self.config.pooling_mode and self.config.train_pooling_params:
            # Combine all attention-specific parameters (V, U, w) into one group
            attention_params = itertools.chain(
                mil_model.attention_V.parameters(),
                mil_model.attention_U.parameters(),
                mil_model.attention_w.parameters(),
            )
            param_groups.append({
                'params': attention_params,
                'lr': self.config.attention_lr,
                'name': 'attention_pooling',
                'weight_decay': self.config.attention_weight_decay,
            })

        # Add CLS token parameters if applicable
        if "cls_token" in self.config.pooling_mode and self.config.train_pooling_params:
            cls_params = [mil_model.cls_token]
            if hasattr(mil_model, 'cls_attention_within'):
                cls_params_iter = itertools.chain(
                    cls_params,
                    mil_model.cls_attention_within.parameters(),
                    mil_model.cls_attention_across.parameters(),
                    mil_model.cls_norm_within.parameters(),
                    mil_model.cls_norm_across.parameters(),
                )
            else:
                cls_params_iter = itertools.chain(
                    cls_params,
                    mil_model.cls_attention.parameters(),
                    mil_model.cls_norm.parameters(),
                )
            param_groups.append({
                'params': cls_params_iter,
                'lr': self.config.attention_across_lr,
                'name': 'cls_token_pooling',
                'weight_decay': self.config.attention_across_weight_decay,
            })

        if not self.config.train_pooling_params:
            print("NOTE: train_pooling_params=False — attention/cls_token params are FROZEN (old behaviour)")

        # Add view embedding parameters if applicable
        if num_view_classes > 0 and hasattr(mil_model, 'view_embedding'):
            ve_lr = self.config.view_embedding_lr if self.config.view_embedding_lr is not None else self.config.attention_lr
            ve_wd = self.config.view_embedding_weight_decay if self.config.view_embedding_weight_decay is not None else self.config.attention_weight_decay
            param_groups.append({
                'params': mil_model.view_embedding.parameters(),
                'lr': ve_lr,
                'name': 'view_embedding',
                'weight_decay': ve_wd,
            })

        # Initialize optimizer
        optimizer_class = getattr(torch.optim, self.config.optimizer)
        optimizer = optimizer_class(param_groups)

        # Initialize scheduler
        scheduler = get_scheduler(
            scheduler_name=self.config.scheduler_name,
            optimizer=optimizer,
            num_epochs=self.config.epochs,
            train_dataloader=train_loader,
            factor=self.config.factor,
            step_size=self.config.lr_step_period,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_warmup_percent=self.config.num_warmup_percent,
            num_hard_restarts_cycles=self.config.num_hard_restarts_cycles,
            warm_restart_tmult=self.config.warm_restart_tmult,
        )

        # Initialize scaler
        print(f"Using AMP: {self.config.use_amp}")
        scaler = GradScaler('cuda') if self.config.use_amp else None
        
        # Create loss function
        loss_fn = Loss(
            loss_type=LossRegistry.get(LossType.MULTI_HEAD)(
                head_structure=self.config.head_structure,
                loss_structure=self.config.loss_structure,
                head_weights=self.config.head_weights,
            )
        )

        # Distribute the full model, not just the MIL head. When the video
        # encoder is partially trainable, its gradients must be synchronized too.
        linear_probing = DistributedUtils.DDP(
            linear_probing,
            device_ids=[self.config.device],
            find_unused_parameters=True
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "linear_probing": linear_probing,
            "optimizer": optimizer,
            "scaler": scaler,
            "lr_scheduler": scheduler,
            "loss_fn": loss_fn,
            "output_dir": self.config.output_dir if self.config.is_ref_device else None,
        }            
            
    def _setup_validation_objects(self) -> dict[str, Any]:
        """Setup objects for model validation/evaluation."""
        self._disable_encoder_aggregation_for_mil()

        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)

        # View embedding parameters — only enable when view_column is set in the YAML
        view_column = getattr(self.config, 'view_column', None)
        view_labels_map = getattr(self.config, 'view_labels_map', None)
        if view_column:
            num_view_classes = getattr(self.config, 'num_view_classes', 0)
            print(f"View embeddings ENABLED: view_column='{view_column}', {num_view_classes} view classes + 1 PAD")
        else:
            num_view_classes = 0
            print("View embeddings DISABLED (no view_column in config)")

        val_loader: DataLoader = get_distributed_video_dataloader(
            config=self.config,
            split=self.config.run_mode,
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
            multi_video=self.config.multi_video,
            groupby_column=self.config.groupby_column,
            num_videos=self.config.num_videos,
            shuffle_videos=False,  # Don't shuffle validation videos
            labels_map=getattr(self.config, 'labels_map', None),
            view_column=view_column,
            view_labels_map=view_labels_map,
            num_view_classes=num_view_classes,
        )

        # Initialize video encoder backbone for linear probing
        video_encoder: VideoEncoder = ModelRegistry.get("video_encoder")(
            backbone=self.config.model_name,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
            aggregate_videos_tokens=self.config.aggregate_videos_tokens,
            per_video_pool=self.config.per_video_pool,
        )
        video_encoder = video_encoder.to(self.config.device)

        # Get embedding dimension from encoder
        embedding_dim = video_encoder.embedding_dim

        # Initialize Multi-Instance Linear Probing model
        mil_model: MultiInstanceLinearProbing = ModelRegistry.get("multi_instance_linear_probing")(
            embedding_dim=embedding_dim,
            head_structure=self.config.head_structure,
            pooling_mode=self.config.pooling_mode,
            attention_hidden=self.config.attention_hidden,
            dropout=self.config.dropout_attention,
            num_view_classes=num_view_classes,
        )
        mil_model = mil_model.to(self.config.device)

        # Wrap both models
        linear_probing = VideoMILWrapper(video_encoder, mil_model, self.config.num_videos)
                
        # Distribute linear probing model
        linear_probing = DistributedUtils.DDP(
            linear_probing,
            device_ids=[self.config.device],
            find_unused_parameters=True
        )
        
        # Load checkpoint with fixed keys
        checkpoint: dict[str, Any] = self._load_and_fix_checkpoint(self.config.inference_model_path)
        linear_probing.load_state_dict(checkpoint["linear_probing"])
        
        # Set to eval mode
        linear_probing.eval()
                
        # Create loss function
        loss_fn = Loss(
            loss_type=LossRegistry.get(LossType.MULTI_HEAD)(
                head_structure=self.config.head_structure,
                loss_structure=self.config.loss_structure,
                head_weights=self.config.head_weights,
            )
        )                
                
        return {
            "loss_fn": loss_fn,
            "val_loader": val_loader,
            "linear_probing": linear_probing,
            "output_dir": self.config.output_dir if self.config.is_ref_device else None,
        }

    def _setup_test_objects(self) -> dict[str, Any]:
        return self._setup_validation_objects() # Diff. is self.config.run_mode

    def _setup_inference_objects(self) -> dict[str, Any]:
        return self._setup_validation_objects() # Diff. is self.config.run_mode

    def run(self):
        
        if self.config.is_ref_device:
            self._setup_project()
        
        runner_args = {
            "config": self.config,
            "wandb_wrapper": self.wandb_wrapper
        }
        
        if self.config.run_mode == RunMode.TRAIN:
            runner_args.update(self._setup_training_objects())
        elif self.config.run_mode == RunMode.VALIDATE:
            runner_args.update(self._setup_validation_objects())
        elif self.config.run_mode == RunMode.TEST:
            runner_args.update(self._setup_test_objects())
        elif self.config.run_mode == RunMode.INFERENCE:
            runner_args.update(self._setup_inference_objects())
        
        # Create runner instance
        runner: Runner = RunnerRegistry.get(
            name=self.config.pipeline_project
        )(**runner_args)

        # Train the model
        if self.config.run_mode == RunMode.TRAIN:
            start_epoch = 0
            resume_path = getattr(self.config, "resume_checkpoint_path", None)
            if resume_path:
                ck = self._load_and_fix_checkpoint(resume_path)
                runner.linear_probing.load_state_dict(ck["linear_probing"])
                if ck.get("optimizer") is not None and runner.optimizer is not None:
                    runner.optimizer.load_state_dict(ck["optimizer"])
                if ck.get("scheduler") is not None and getattr(runner, "lr_scheduler", None) is not None:
                    runner.lr_scheduler.load_state_dict(ck["scheduler"])
                if ck.get("scaler") is not None and getattr(runner, "scaler", None) is not None:
                    runner.scaler.load_state_dict(ck["scaler"])
                start_epoch = int(ck.get("epoch", -1)) + 1
                print(f"[RESUME] loaded {resume_path}; resuming at epoch {start_epoch}", flush=True)
            runner.train(start_epoch=start_epoch, end_epoch=self.config.epochs)
        elif self.config.run_mode == RunMode.TEST:
            runner.test()
        elif self.config.run_mode == RunMode.VALIDATE:
            runner.validate()
        elif self.config.run_mode == RunMode.INFERENCE:
            runner.inference()

        # Final cleanup
        if self.config.is_ref_device:
            self.wandb_wrapper.finish()

    def _load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load checkpoint from path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        device = self.config.device
        if isinstance(device, int):
            device = f"cuda:{device}"
        checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
        return checkpoint

    def _load_and_fix_checkpoint(self, path: str) -> dict[str, Any]:
        """Load checkpoint and fix DDP key mismatches."""
        checkpoint = self._load_checkpoint(path)
        
        if "linear_probing" not in checkpoint:
            return checkpoint
        
        state_dict = checkpoint["linear_probing"]
        fixed_state_dict = {}
        
        for key, value in state_dict.items():
            # Fix video_encoder keys: add missing "module." prefix
            if key.startswith("video_encoder."):
                new_key = f"module.{key}"
                fixed_state_dict[new_key] = value

            # Fix mil_model keys: remove extra "module." and add top-level "module."
            elif key.startswith("mil_model.module."):
                # Remove the middle "module." and add top-level "module."
                inner_key = key.replace("mil_model.module.", "mil_model.", 1)
                new_key = f"module.{inner_key}"
                fixed_state_dict[new_key] = value

            # Current checkpoints save the unwrapped VideoMILWrapper state_dict,
            # but validation/resume loads into DDP(VideoMILWrapper).
            elif key.startswith("mil_model."):
                fixed_state_dict[f"module.{key}"] = value

            # Handle any other keys normally
            else:
                fixed_state_dict[key] = value

        # Fix old cls_attention -> new cls_attention_within/across split
        keys_to_add = {}
        keys_to_remove = []
        for key in list(fixed_state_dict.keys()):
            # Map old single cls_attention to both within and across
            if ".cls_attention." in key and ".cls_attention_within." not in key and ".cls_attention_across." not in key:
                within_key = key.replace(".cls_attention.", ".cls_attention_within.")
                across_key = key.replace(".cls_attention.", ".cls_attention_across.")
                keys_to_add[within_key] = fixed_state_dict[key]
                keys_to_add[across_key] = fixed_state_dict[key].clone()
                keys_to_remove.append(key)
            # Map old single cls_norm to both within and across
            elif ".cls_norm." in key and ".cls_norm_within." not in key and ".cls_norm_across." not in key:
                within_key = key.replace(".cls_norm.", ".cls_norm_within.")
                across_key = key.replace(".cls_norm.", ".cls_norm_across.")
                keys_to_add[within_key] = fixed_state_dict[key]
                keys_to_add[across_key] = fixed_state_dict[key].clone()
                keys_to_remove.append(key)
            # Drop old pre_agg_norm (removed from new architecture)
            elif ".pre_agg_norm." in key:
                keys_to_remove.append(key)

        for k in keys_to_remove:
            del fixed_state_dict[k]
        fixed_state_dict.update(keys_to_add)
        
        checkpoint["linear_probing"] = fixed_state_dict
        return checkpoint
