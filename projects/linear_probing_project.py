import os
import torch
from typing import Any, Optional, Dict
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import itertools

from runners.typing import Runner
from models.video_encoder import VideoEncoder
from models.linear_probing import LinearProbing
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

    def forward(self, x: torch.Tensor, video_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Wrapper forward pass.

        This helper guarantees that the **MultiInstanceLinearProbing** module
        always receives a 3-D tensor of shape ``[B, N, D]`` where *N* is the
        number of video segments associated with each sample.

        In the typical *single-video* case, ``video_indices`` will be ``None``
        and the underlying ``VideoEncoder`` already returns a tensor of shape
        ``[B, D]`` (aggregated representation).  We therefore unsqueeze a
        singleton *N* dimension so that it becomes ``[B, 1, D]``.

        When ``multi_video=True`` the dataloader supplies inputs of shape
        ``[B, N, C, F, H, W]``.  If the encoder is configured with
        ``aggregate_videos_tokens=False`` it will naturally produce the
        expected ``[B, N, D]`` tensor.  However, if
        ``aggregate_videos_tokens=True`` the encoder will *already* pool over
        the *N* dimension and output ``[B, D]``.  In this scenario we again
        expand the aggregated vector so that downstream MIL logic continues
        to work (with ``N = 1``).
        """

        # ------------------------------------------------------------------
        # 1) Run the backbone / encoder
        # ------------------------------------------------------------------
        embeddings: torch.Tensor = self.video_encoder(x)

        # ------------------------------------------------------------------
        # 2) Reshape so that the MIL module always sees *either*:
        #    • 3-D tensor  [B, N, D]  (per-video embeddings)
        #    • 4-D tensor  [B, N, L, D] (per-patch tokens, hierarchical)
        # ------------------------------------------------------------------

        if embeddings.ndim == 2:
            # Encoder returned [B, D] → add singleton video dimension
            embeddings = embeddings.unsqueeze(1)  # [B, 1, D]

        elif embeddings.ndim == 3 and embeddings.shape[1] > self.num_videos:
            #print(f"Received flat patch tokens [B, N*L, D]. Reshaping into [B, N, L, D]")
            # Received flat patch tokens [B, N*L, D].  Reshape into
            # hierarchical layout [B, N, L, D] so that the MIL head can
            # perform two-level attention.
            B, NL, D = embeddings.shape  # noqa: N806
            #print(f"embeddings.shape: {embeddings.shape}")
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

        # Build a simple boolean mask that marks every video as valid.  In the
        # future we could incorporate ``video_indices`` to create selective
        # masks, e.g. when some videos are padded.
        attention_mask = torch.ones((B, N), dtype=torch.bool, device=embeddings.device)

        # ------------------------------------------------------------------
        # 4) Forward through the MIL head(s)
        # ------------------------------------------------------------------
        return self.mil_model(embeddings, mask=attention_mask)

@ProjectRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingProject(BaseProject):
    def __init__(
        self, 
        config: LinearProbingConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)

    def _setup_training_objects(
        self
    )->dict[str, Any]:
                
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)        
        
        # Get dataloaders with multi-video parameters
        train_loader: DataLoader = get_distributed_video_dataloader(
            config=self.config, 
            split="train", 
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
        )
        val_loader: DataLoader = get_distributed_video_dataloader(
            config=self.config, 
            split="val", 
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
        elif self.config.video_freeze_ratio > 0:
             # Partial freezing logic might be needed here if supported by VideoEncoder
             pass # Assuming freeze_ratio in VideoEncoder init handles partial freezing

        # Initialize Multi-Instance Linear Probing model
        mil_model: MultiInstanceLinearProbing = ModelRegistry.get("multi_instance_linear_probing")(
            embedding_dim=embedding_dim, 
            head_structure=self.config.head_structure,
            pooling_mode=self.config.pooling_mode, 
            attention_hidden=self.config.attention_hidden, 
            dropout=self.config.dropout_attention, 
        )
        mil_model = mil_model.to(self.config.device).float()

        # Distribute MIL model
        mil_model = DistributedUtils.DDP(
            mil_model, 
            device_ids=[self.config.device]
        )
        
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
                'params': mil_model.module.heads[head_name].parameters(),
                'lr': self.config.head_lr[head_name],
                'name': head_name,
                'weight_decay': self.config.head_weight_decay[head_name]
            })
            
        # Add attention parameters if applicable (potentially different LR/WD)
        if self.config.pooling_mode == "attention":
            # Combine all attention-specific parameters (V, U, w) into one group
            attention_params = itertools.chain(
                mil_model.module.attention_V.parameters(),
                mil_model.module.attention_U.parameters(),
                mil_model.module.attention_w.parameters(),
            )
            param_groups.append({
                'params': attention_params,
                'lr': self.config.attention_lr,
                'name': 'attention_pooling',
                'weight_decay': self.config.attention_weight_decay,
            })

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
        scaler = GradScaler() if self.config.use_amp else None
        
        # Create loss function
        loss_fn = Loss(
            loss_type=LossRegistry.get(LossType.MULTI_HEAD)(
                head_structure=self.config.head_structure,
                loss_structure=self.config.loss_structure,
                head_weights=self.config.head_weights,
            )
        )

        # --------------------------------------------------------------
        # Linear-probing *must* receive per-video (or per-patch) tokens so
        # that the downstream MIL module can do its own aggregation.  If the
        # YAML accidentally sets ``aggregate_videos_tokens=True`` we disable
        # it and emit a warning instead of failing later with shape errors.
        # --------------------------------------------------------------
        if getattr(self.config, "aggregate_videos_tokens", False):
            if self.config.is_ref_device:
                print(
                    "[WARNING] aggregate_videos_tokens=True detected but "
                    "should be False for linear-probing. This is only used for CLIP. Overriding to "
                    "False so that the VideoEncoder preserves per-instance "
                    "tokens. This override is logged to wandb."
                )
            # Mutate in-place so every subsequent consumer (e.g. VideoEncoder)
            # sees the corrected value.
            self.config.aggregate_videos_tokens = False
            if self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.log({"config/aggregate_videos_tokens_override": True})

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
        

    def _setup_inference_objects(
        self
    )->dict[str, Any]:
        raise NotImplementedError("Inference is not implemented for this project")

    def run(self):
        
        if self.config.is_ref_device:
            self._setup_project()
        
        training_setup: dict[str, Any] = self._setup_training_objects()
        
        # Create runner instance
        runner: Runner = RunnerRegistry.get(
            name=self.config.pipeline_project
        )(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            **training_setup
        )

        # Train the model
        runner.train(start_epoch=0, end_epoch=self.config.epochs)

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
        checkpoint: dict[str, Any] = torch.load(path, map_location=device)
        return checkpoint

    def _load_runner_checkpoint(self, runner: Runner, path: str) -> int:
        """Load checkpoint for the entire runner."""
        checkpoint: dict[str, Any] = self._load_checkpoint(path)
        # Load state for the wrapper model (video_encoder + mil_model)
        runner.linear_probing.video_encoder.load_state_dict(checkpoint["video_encoder"])
        runner.linear_probing.mil_model.module.load_state_dict(checkpoint["mil_model"])
        # Load optimizer, scaler, and scheduler states
        runner.optimizer.load_state_dict(checkpoint["optimizer"])
        if runner.scaler and checkpoint["scaler"]:
            runner.scaler.load_state_dict(checkpoint["scaler"])
        if runner.lr_scheduler and checkpoint["scheduler"]:
            runner.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        # Return the epoch to resume from
        return checkpoint.get("epoch", 0) + 1 # Start from next epoch
