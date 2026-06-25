
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from models.text_encoder import TextEncoder
from models.video_encoder import VideoEncoder
from projects.base_project import BaseProject
from runners.typing import Runner
from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.enums import RunMode
from utils.config.clip_config import ClipConfig
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry, 
    RunnerRegistry, 
    ProjectRegistry, 
    LossRegistry
)
from utils.wandb_wrapper import WandbWrapper
from utils.video_project import calculate_dataset_statistics_ddp
from dataloaders.video_clip_dataset import get_distributed_video_clip_dataloader

@ProjectRegistry.register('DeepCORO_clip')
class ContrastivePretrainingProject(BaseProject):
    def __init__(
        self, 
        config: ClipConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)
        
    def _build_loss(self) -> Loss:
        """
        Construct the loss function.

        For SigLIP-family losses, wire the YAML-configurable knobs from
        ``ClipConfig`` into the loss constructor. ``getattr`` with defaults
        keeps this robust if the loss ``__init__`` signature differs.
        """
        loss_name: str = self.config.loss_name
        loss_cls = LossRegistry.get(loss_name)

        if "siglip" in str(loss_name).lower():
            siglip_kwargs: dict[str, Any] = dict(
                bias_init=getattr(self.config, "siglip_bias_init", -10.0),
                learnable_bias=getattr(self.config, "siglip_learnable_bias", True),
                positive_weight=getattr(self.config, "siglip_positive_loss_weight", 1.0),
                negative_weight=getattr(self.config, "siglip_negative_loss_weight", 1.0),
                use_severity_weights=getattr(
                    self.config, "siglip_enable_severity_weighting", True
                ),
                auto_balance=getattr(
                    self.config, "siglip_auto_positive_loss_weight", False
                ),
                entropy_regularization=getattr(
                    self.config, "siglip_entropy_regularization", False
                ),
                entropy_weight=getattr(self.config, "siglip_entropy_weight", 0.1),
                min_entropy_threshold=getattr(
                    self.config, "siglip_min_entropy_threshold", 2.0
                ),
                gather_in_ddp=getattr(self.config, "siglip_gather_in_ddp", False),
            )
            try:
                return Loss(loss_type=loss_cls(**siglip_kwargs))
            except TypeError as exc:
                # Fall back gracefully if the loss signature does not accept
                # one or more of the wired kwargs.
                print(
                    f"[WARN] SigLIP loss '{loss_name}' rejected configured "
                    f"kwargs ({exc}); falling back to default construction."
                )
                return Loss(loss_type=loss_cls())

        return Loss(loss_type=loss_cls())

    def _setup_training_objects(
        self,
    )->dict:
        """
        Load training objects.

        Args:
            config (HeartWiseConfig): Configuration object

        Returns:
            dict: Dictionary containing training objects
        """
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)

        train_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="train", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=True,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=True,
        )
        val_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="val", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,  # IMPORTANT: Never shuffle validation
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,  # IMPORTANT: Keep all validation samples
        )

        # Create models
        video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone=self.config.model_name,
            input_channels=3,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            output_dim=512,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
            token_pooling_mode=getattr(self.config, 'video_pooling_mode', 'mean'),
            attention_pool_heads=getattr(self.config, 'attention_pool_heads', 8),
            attention_pool_dropout=getattr(self.config, 'attention_pool_dropout', 0.1),
            use_cls_token=getattr(self.config, 'use_cls_token', False),
            # RoPE parameters
            use_rope=getattr(self.config, 'use_rope', False),
            rope_base=getattr(self.config, 'rope_base', 10000.0),
            rope_temporal_scale=getattr(self.config, 'rope_temporal_scale', 1.0),
            rope_normalize_mode=getattr(self.config, 'rope_normalize_mode', 'separate'),
        )
        video_encoder = video_encoder.to(self.config.device).float()

        text_encoder: TextEncoder = ModelRegistry.get(
            name="text_encoder"
        )(
            freeze_ratio=self.config.text_freeze_ratio,
            dropout=self.config.dropout,
        )
        text_encoder = text_encoder.to(self.config.device).float()

        video_encoder = DistributedUtils.DDP(
            video_encoder, 
            device_ids=[self.config.device], 
        )
        text_encoder = DistributedUtils.DDP(
            text_encoder, 
            device_ids=[self.config.device], 
        )

        # Make temperature a trainable parameter directly on the device
        log_temperature: nn.Parameter = nn.Parameter(
            torch.log(
                torch.tensor(
                    [self.config.temperature], 
                    dtype=torch.float32, 
                    device=self.config.device
                )
            )
        )

        # Different learning rates for different components
        param_groups = [
            {
                'params': video_encoder.module.model.parameters(),  # Main video backbone
                'lr': self.config.lr,
                'name': 'video_backbone',
                'weight_decay': self.config.video_weight_decay
            },
            {
                'params': text_encoder.module.parameters(),  # Entire text encoder
                'lr': 0.00002,  # Lower learning rate for text encoder
                'name': 'text_encoder',
                'weight_decay': self.config.text_weight_decay
            },
            {
                'params': [log_temperature],  # Temperature parameter
                'lr': self.config.lr,
                'name': 'temperature'
            }
        ]

        insert_idx = 1

        attention_pool_params = [
            p for p in getattr(video_encoder.module, 'attention_pool', nn.Identity()).parameters()
            if p.requires_grad
        ]
        if attention_pool_params:
            param_groups.insert(
                insert_idx,
                {
                    'params': attention_pool_params,
                    'lr': self.config.lr * 2.0,
                    'name': 'video_attention_pool',
                    'weight_decay': self.config.video_weight_decay
                }
            )
            insert_idx += 1

        aggregator_params = [
            p for p in getattr(video_encoder.module, 'aggregator', nn.Identity()).parameters()
            if p.requires_grad
        ]
        if aggregator_params:
            param_groups.insert(
                insert_idx,
                {
                    'params': aggregator_params,
                    'lr': self.config.lr * 2.0,  # Higher learning rate for aggregator
                    'name': 'video_aggregator',
                    'weight_decay': self.config.video_weight_decay
                }
            )

        # Include the temperature parameter in the optimizer
        optimizer_class: torch.optim.Optimizer = getattr(torch.optim, self.config.optimizer)
        optimizer: torch.optim.Optimizer = optimizer_class(
            param_groups,
            lr=self.config.lr # act as a default learning rate for unset learning rates in param_groups
        )

        scheduler: LRScheduler = get_scheduler(
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

        scaler: GradScaler = GradScaler('cuda') if self.config.use_amp else None

        # Create loss function
        loss_fn: Loss = self._build_loss()

        if self.config.is_ref_device:
            if self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.config_update(
                    {
                        "train_dataset_size": len(train_loader),
                        "val_dataset_size": len(val_loader),
                    },
                )        
            print("\n=== Dataset Information ===")
            print(f"Training:   {len(train_loader):,} batches per GPU")
            print(f"Validation: {len(val_loader):,} batches per GPU")
            print(f"Total:      {(len(train_loader) + len(val_loader)):,} batches per GPU")
            print(f"\nBatch Size: {self.config.batch_size}")
            print(f"Training: {len(train_loader) * self.config.batch_size:,} videos per GPU")
            print(f"Validation: {len(val_loader) * self.config.batch_size:,} videos per GPU")
            print(f"Total: {(len(train_loader) + len(val_loader)) * self.config.batch_size:,} videos per GPU")
            print("===========================\n")

        return {
            "video_encoder": video_encoder,
            "text_encoder": text_encoder,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "scaler": scaler,
            "log_temp": log_temperature,
            "loss_fn": loss_fn,
            "output_dir": self.config.output_dir if self.config.is_ref_device else None,
        }    

    def _setup_inference_objects(
        self,
    )->dict[str, Any]:
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)
        
        val_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="inference", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
        )
        
        # Create models
        video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone=self.config.model_name,
            input_channels=3,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            output_dim=512,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
            token_pooling_mode=getattr(self.config, 'video_pooling_mode', 'mean'),
            attention_pool_heads=getattr(self.config, 'attention_pool_heads', 8),
            attention_pool_dropout=getattr(self.config, 'attention_pool_dropout', 0.1),
            # RoPE parameters
            use_rope=getattr(self.config, 'use_rope', False),
            rope_base=getattr(self.config, 'rope_base', 10000.0),
            rope_temporal_scale=getattr(self.config, 'rope_temporal_scale', 1.0),
            rope_normalize_mode=getattr(self.config, 'rope_normalize_mode', 'separate'),
        )        
        video_encoder = video_encoder.to(self.config.device).float()
        
        video_encoder = DistributedUtils.DDP(
            video_encoder, 
            device_ids=[self.config.device], 
        )
        
        checkpoint: dict[str, Any] = self._load_checkpoint(self.config.checkpoint)
        video_encoder.module.load_state_dict(checkpoint["video_encoder"])

        # Recover log-temperature without corruption (see resume path). Prefer
        # the raw "log_temp"; fall back to log(temperature) for old checkpoints.
        if "log_temp" in checkpoint and checkpoint["log_temp"] is not None:
            log_temp = torch.as_tensor(
                checkpoint["log_temp"], dtype=torch.float32
            )
        else:
            temp_value = checkpoint.get(
                "temperature",
                checkpoint.get("train/temperature", self.config.temperature),
            )
            log_temp = torch.log(
                torch.as_tensor(temp_value, dtype=torch.float32)
            )

        return {
            "val_loader": val_loader,
            "video_encoder": video_encoder,
            "log_temp": log_temp,
            "output_dir": self.config.inference_results_path,
        }

    def _update_training_setup_with_checkpoint(
        self, 
        training_setup: dict[str, Any], 
        checkpoint: dict[str, Any]
    )->dict[str, Any]:
        print(f"Resuming from checkpoint: {checkpoint.keys()}")
        training_setup["video_encoder"].module.load_state_dict(checkpoint["video_encoder"])
        training_setup["text_encoder"].module.load_state_dict(checkpoint["text_encoder"])
        training_setup["optimizer"].load_state_dict(checkpoint["optimizer"])
        training_setup["lr_scheduler"].load_state_dict(checkpoint["scheduler"])
        training_setup["scaler"].load_state_dict(checkpoint["scaler"])

        # Restore log-temperature WITHOUT corruption. The checkpoint logs
        # ``train/temperature`` as exp(log_temp); copying that onto log_temp
        # would double-exponentiate (T -> exp(T)). Prefer the raw "log_temp"
        # tensor; fall back to log(temperature) for old checkpoints.
        target_log_temp = training_setup["log_temp"]
        if "log_temp" in checkpoint and checkpoint["log_temp"] is not None:
            log_temp_value = torch.as_tensor(
                checkpoint["log_temp"],
                dtype=target_log_temp.dtype,
                device=target_log_temp.device,
            )
        else:
            temp_value = checkpoint.get(
                "temperature",
                checkpoint.get("train/temperature", self.config.temperature),
            )
            log_temp_value = torch.log(
                torch.as_tensor(
                    temp_value,
                    dtype=target_log_temp.dtype,
                    device=target_log_temp.device,
                )
            )
        target_log_temp.data.copy_(log_temp_value.reshape(target_log_temp.shape))
        return training_setup
        
    def run(self):
        self._setup_project()
        
        if self.config.run_mode == RunMode.TRAIN:
            training_setup: dict[str, Any] = self._setup_training_objects()
            start_epoch = 0
            if self.config.resume_training:
                checkpoint = self._load_checkpoint(self.config.checkpoint)
                training_setup = self._update_training_setup_with_checkpoint(training_setup, checkpoint)
                start_epoch = checkpoint["epoch"]
                print(f"Resuming from epoch: {start_epoch}")
            runner: Runner = Runner(
                runner_type=RunnerRegistry.get(
                    name=self.config.pipeline_project
                )(
                    config=self.config,
                    wandb_wrapper=self.wandb_wrapper,
                    **training_setup,
                )
            )
            end_epoch = start_epoch + self.config.epochs
            runner.train(start_epoch=start_epoch, end_epoch=end_epoch)
        elif self.config.run_mode == RunMode.INFERENCE:
            inference_setup: dict[str, Any] = self._setup_inference_objects()
            runner: Runner = Runner(
                runner_type=RunnerRegistry.get(
                    name=self.config.pipeline_project
                )(
                    config=self.config,
                    wandb_wrapper=self.wandb_wrapper,
                    **inference_setup,
                )
            )
            runner.inference()
        else:
            raise ValueError(
                f"Invalid run mode: {self.config.run_mode}, must be one of {RunMode.TRAIN} or {RunMode.INFERENCE}"
            )
