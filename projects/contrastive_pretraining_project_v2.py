"""Contrastive pretraining project with two-optimizer setup and LLRD."""

import os
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from typing import Any, Optional
import math

from models.text_encoder import TextEncoder
from models.video_encoder import VideoEncoder
from utils.config.clip_config import ClipConfig
from utils.loss.typing import Loss
from utils.registry import LossRegistry, ModelRegistry, ProjectRegistry, RunnerRegistry
from runners.typing import Runner
from dataloaders.video_clip_dataset import get_distributed_video_clip_dataloader
from utils.video_project import calculate_dataset_statistics_ddp
from utils.enums import RunMode
from utils.wandb_logger import WandbWrapper
from utils.schedulers import get_scheduler
from torch.optim.lr_scheduler import LRScheduler
from utils.ddp import DistributedUtils
from utils.optimizer_utils import (
    create_two_optimizer_setup,
    PhaseConfig,
    PhasedTrainingScheduler,
    initialize_logit_scale,
    clamp_logit_scale,
)
from projects.base_project import BaseProject


@ProjectRegistry.register("DeepCORO_clip_v2")
class ContrastivePretrainingProjectV2(BaseProject):
    """Enhanced contrastive pretraining with two-optimizer setup and LLRD."""
    
    def __init__(
        self,
        config: ClipConfig,
        wandb_wrapper: Optional[WandbWrapper] = None,
    ):
        """Initialize project.
        
        Args:
            config: Configuration object
            wandb_wrapper: Wandb wrapper for logging
        """
        super().__init__(config=config, wandb_wrapper=wandb_wrapper)
        self.config: ClipConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper

    def _setup_training_objects(self) -> dict[str, Any]:
        """Setup all training objects with two-optimizer configuration."""
        
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)
        
        # Create dataloaders
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
            shuffle=False,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
        )
        
        # Create video encoder with 3D RoPE
        video_encoder: VideoEncoder = ModelRegistry.get("video_encoder")(
            backbone=self.config.model_name,
            input_channels=3,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            output_dim=512,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
            aggregate_videos_tokens=not self.config.per_video_pool,
            per_video_pool=self.config.per_video_pool,
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
        
        # Create text encoder
        text_encoder: TextEncoder = ModelRegistry.get("text_encoder")(
            freeze_ratio=self.config.text_freeze_ratio,
            dropout=self.config.dropout,
        )
        text_encoder = text_encoder.to(self.config.device).float()
        
        # Create projection heads (2-layer MLPs)
        # Video: 512 -> 768 -> 512
        video_proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(768, 512),
        ).to(self.config.device)
        
        # Text: 512 -> 768 -> 512 (text encoder already outputs 512)
        text_proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(768, 512),
        ).to(self.config.device)
        
        # Initialize logit scale with proper value
        logit_scale = initialize_logit_scale(
            initial_value=self.config.logit_scale_init if hasattr(self.config, 'logit_scale_init') else 2.659,
            device=self.config.device
        )
        
        # Wrap models in DDP
        # Use find_unused_parameters=True for models that will have dynamic parameter freezing
        video_encoder = DistributedUtils.DDP(video_encoder, device_ids=[self.config.device])
        text_encoder = DistributedUtils.DDP(
            text_encoder, 
            device_ids=[self.config.device],
            find_unused_parameters=True  # Required for phased training with partial freezing
        )
        video_proj = DistributedUtils.DDP(video_proj, device_ids=[self.config.device])
        text_proj = DistributedUtils.DDP(text_proj, device_ids=[self.config.device])
        
        # Create optimizers based on configuration
        if self.config.use_two_optimizers:
            # Two-optimizer setup with LLRD
            video_optimizer, text_optimizer = create_two_optimizer_setup(
                video_encoder=video_encoder.module,
                text_encoder=text_encoder.module,
                video_proj=video_proj.module,
                text_proj=text_proj.module,
                logit_scale=logit_scale,
                config=self.config,
            )
            
            # Create separate schedulers
            video_scheduler = get_scheduler(
                scheduler_name=self.config.scheduler_name,
                optimizer=video_optimizer,
                num_epochs=self.config.epochs,
                train_dataloader=train_loader,
                factor=self.config.factor,
                step_size=self.config.lr_step_period,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_warmup_percent=self.config.num_warmup_percent,
                num_hard_restarts_cycles=self.config.num_hard_restarts_cycles,
                warm_restart_tmult=self.config.warm_restart_tmult,
            )
            
            text_scheduler = get_scheduler(
                scheduler_name=self.config.scheduler_name,
                optimizer=text_optimizer,
                num_epochs=self.config.epochs,
                train_dataloader=train_loader,
                factor=self.config.factor,
                step_size=self.config.lr_step_period,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_warmup_percent=0.1,  # More warmup for text
                num_hard_restarts_cycles=self.config.num_hard_restarts_cycles,
                warm_restart_tmult=self.config.warm_restart_tmult,
            )
            
            # Combine for compatibility
            optimizer = video_optimizer  # Primary optimizer
            scheduler = video_scheduler  # Primary scheduler
            
        else:
            # Original single optimizer setup (fallback)
            param_groups = [
                {
                    'params': video_encoder.module.model.parameters(),
                    'lr': self.config.lr,
                    'name': 'video_backbone',
                    'weight_decay': self.config.video_weight_decay
                },
                {
                    'params': video_encoder.module.aggregator.parameters(),
                    'lr': self.config.lr * 2.0,
                    'name': 'video_aggregator',
                    'weight_decay': self.config.video_weight_decay
                },
                {
                    'params': video_proj.module.parameters(),
                    'lr': self.config.lr * 5.0,
                    'name': 'video_projection',
                    'weight_decay': self.config.video_weight_decay
                },
                {
                    'params': text_encoder.module.parameters(),
                    'lr': 0.00002,
                    'name': 'text_encoder',
                    'weight_decay': self.config.text_weight_decay
                },
                {
                    'params': text_proj.module.parameters(),
                    'lr': 0.0001,
                    'name': 'text_projection',
                    'weight_decay': self.config.text_weight_decay
                },
                {
                    'params': [logit_scale],
                    'lr': self.config.lr,
                    'name': 'temperature'
                }
            ]
            
            optimizer_class = getattr(torch.optim, self.config.optimizer)
            optimizer = optimizer_class(param_groups, lr=self.config.lr)
            
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
            
            video_optimizer = None
            text_optimizer = None
            video_scheduler = None
            text_scheduler = None
        
        # Create phased training scheduler if enabled
        phased_scheduler = None
        if self.config.use_phased_training and self.config.use_two_optimizers:
            phases = [
                PhaseConfig(
                    name="warm_start",
                    epochs=self.config.phase_a_epochs,
                    text_freeze_layers=None if self.config.phase_a_text_layers == 0 else self.config.phase_a_text_layers,
                    video_freeze_ratio=0.0,
                    logit_scale_trainable=False,
                    text_lr_multiplier=0.0 if self.config.phase_a_text_layers == 0 else 0.5,
                    video_lr_multiplier=1.0,
                ),
                PhaseConfig(
                    name="alignment",
                    epochs=self.config.phase_b_epochs,
                    text_freeze_layers=self.config.phase_b_text_layers,
                    video_freeze_ratio=0.0,
                    logit_scale_trainable=True,
                    text_lr_multiplier=0.5,
                    video_lr_multiplier=1.0,
                ),
                PhaseConfig(
                    name="fine_tune",
                    epochs=self.config.phase_c_epochs,
                    text_freeze_layers=-1 if self.config.phase_c_text_layers == 12 else self.config.phase_c_text_layers,
                    video_freeze_ratio=0.0,
                    logit_scale_trainable=True,
                    text_lr_multiplier=1.0,
                    video_lr_multiplier=1.0,
                ),
            ]
            
            total_steps = len(train_loader) * self.config.epochs
            phased_scheduler = PhasedTrainingScheduler(
                phases=phases,
                total_steps=total_steps,
                video_encoder=video_encoder.module,
                text_encoder=text_encoder.module,
                video_optimizer=video_optimizer,
                text_optimizer=text_optimizer,
                logit_scale=logit_scale,
            )
        
        # Create scalers for mixed precision
        if self.config.use_amp:
            # Use a single scaler for both optimizers to avoid DDP issues
            if self.config.video_amp_enabled or self.config.text_amp_enabled:
                primary_scaler = GradScaler('cuda')
                video_scaler = primary_scaler
                text_scaler = primary_scaler  # Use same scaler
                scaler = primary_scaler  # Primary scaler for compatibility
            else:
                video_scaler = None
                text_scaler = None
                scaler = None
        else:
            video_scaler = None
            text_scaler = None
            scaler = None
        
        # Create loss function
        loss_fn = Loss(loss_type=LossRegistry.get(self.config.loss_name)())
        
        # Log dataset information
        if self.config.is_ref_device:
            if self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.config_update({
                    "train_dataset_size": len(train_loader),
                    "val_dataset_size": len(val_loader),
                    "use_two_optimizers": self.config.use_two_optimizers,
                    "use_phased_training": self.config.use_phased_training,
                    "use_llrd": self.config.use_two_optimizers,
                })
            
            print("\n=== Training Configuration ===")
            print(f"Two-Optimizer Setup: {self.config.use_two_optimizers}")
            print(f"Phased Training: {self.config.use_phased_training}")
            print(f"LLRD Enabled: {self.config.use_two_optimizers}")
            if self.config.use_two_optimizers:
                print(f"Text LR: {self.config.text_lr} (decay: {self.config.text_llrd_factor})")
                print(f"Video LR: {self.config.video_lr} (decay: {self.config.video_llrd_factor})")
                print(f"Projection LR: {self.config.proj_lr}")
            print(f"Initial Logit Scale: {math.exp(self.config.logit_scale_init):.3f}")
            print("==============================\n")
        
        return {
            "video_encoder": video_encoder,
            "text_encoder": text_encoder,
            "video_proj": video_proj,
            "text_proj": text_proj,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "video_optimizer": video_optimizer,
            "text_optimizer": text_optimizer,
            "video_scheduler": video_scheduler,
            "text_scheduler": text_scheduler,
            "phased_scheduler": phased_scheduler,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "scaler": scaler,
            "video_scaler": video_scaler,
            "text_scaler": text_scaler,
            "log_temp": logit_scale,
            "loss_fn": loss_fn,
            "output_dir": self.config.output_dir if self.config.is_ref_device else None,
        }
    
    def _load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Load checkpoint from disk."""
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.config.device}")
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def _update_training_setup_with_checkpoint(
        self,
        training_setup: dict[str, Any],
        checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        """Update training setup with checkpoint data."""
        print(f"Resuming from checkpoint: {checkpoint.keys()}")
        
        # Load model states
        training_setup["video_encoder"].module.load_state_dict(checkpoint["video_encoder"])
        training_setup["text_encoder"].module.load_state_dict(checkpoint["text_encoder"])
        
        if "video_proj" in checkpoint:
            training_setup["video_proj"].module.load_state_dict(checkpoint["video_proj"])
        if "text_proj" in checkpoint:
            training_setup["text_proj"].module.load_state_dict(checkpoint["text_proj"])
        
        # Load optimizer states
        if self.config.use_two_optimizers:
            if "video_optimizer" in checkpoint:
                training_setup["video_optimizer"].load_state_dict(checkpoint["video_optimizer"])
            if "text_optimizer" in checkpoint:
                training_setup["text_optimizer"].load_state_dict(checkpoint["text_optimizer"])
        else:
            training_setup["optimizer"].load_state_dict(checkpoint["optimizer"])
        
        # Load scheduler states
        if "scheduler" in checkpoint and checkpoint["scheduler"]:
            training_setup["lr_scheduler"].load_state_dict(checkpoint["scheduler"])
        
        # Load scaler states
        if "scaler" in checkpoint and checkpoint["scaler"]:
            if training_setup["scaler"] is not None:
                training_setup["scaler"].load_state_dict(checkpoint["scaler"])
        
        # Load temperature
        if "train/temperature" in checkpoint:
            training_setup["log_temp"].data.copy_(checkpoint["train/temperature"])
        
        return training_setup
    
    def run(self):
        """Run the training or inference pipeline."""
        self._setup_project()
        
        if self.config.run_mode == RunMode.TRAIN:
            training_setup = self._setup_training_objects()
            start_epoch = 0
            
            if self.config.resume_training:
                checkpoint = self._load_checkpoint(self.config.checkpoint)
                training_setup = self._update_training_setup_with_checkpoint(training_setup, checkpoint)
                start_epoch = checkpoint.get("epoch", 0)
                print(f"Resuming from epoch: {start_epoch}")
            
            # Create runner with enhanced configuration
            runner = Runner(
                runner_type=RunnerRegistry.get("DeepCORO_clip_v2")(
                    config=self.config,
                    wandb_wrapper=self.wandb_wrapper,
                    **training_setup,
                )
            )
            
            end_epoch = start_epoch + self.config.epochs
            runner.train(start_epoch=start_epoch, end_epoch=end_epoch)
            
        elif self.config.run_mode == RunMode.INFERENCE:
            raise NotImplementedError("Inference mode not yet implemented for V2")
        else:
            raise ValueError(f"Invalid run mode: {self.config.run_mode}")
    
    def _setup_inference_objects(self) -> dict[str, Any]:
        """Setup inference objects (not implemented for v2 yet)."""
        raise NotImplementedError("Inference mode not yet implemented for V2")