import os
import torch
from typing import Any
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from models.video_encoder import VideoEncoder
from models.linear_probing import LinearProbing
from projects.base_project import BaseProject
from utils.registry import (
    ProjectRegistry, 
    ModelRegistry,
    LossRegistry
)

from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.enums import LossType
from utils.schedulers import get_scheduler
from utils.wandb_wrapper import WandbWrapper
from utils.video_project import calculate_dataset_statistics_ddp
from utils.config.linear_probing_config import LinearProbingConfig
from utils.files_handler import generate_output_dir_name, backup_config
from dataloaders.video_dataset import get_distributed_video_dataloader

@ProjectRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingProject(BaseProject):
    def __init__(
        self, 
        config: LinearProbingConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: LinearProbingConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper

    def _setup_project(self):
        # Generate the output directory name
        self.config.output_dir = generate_output_dir_name(
            config=self.config, 
            run_id=self.wandb_wrapper.get_run_id() if self.wandb_wrapper.is_initialized() else None
        )
        
        print(f"Output directory: {self.config.output_dir}")        
        
        # Create the output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Backup the configuration file
        backup_config(
            config=self.config,
            output_dir=self.config.output_dir
        )
        
    def _setup_training_objects(
        self
    )->dict[str, Any]:
        
        if self.config.is_ref_device:
            self._setup_project()
        
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)        
        
        # Get dataloaders
        train_loader: DataLoader = get_distributed_video_dataloader(
            self.config, 
            split="train", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=True,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
        )
        val_loader: DataLoader = get_distributed_video_dataloader(
            self.config, 
            split="val", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
        )        
        
        # Initialize video encoder backbone for linear probing
        video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone=self.config.model_name,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
        )        

        # Load video encoder checkpoint
        video_encoder = video_encoder.to(self.config.device).float()        
        checkpoint: dict[str, Any] = self._load_checkpoint(self.config.video_encoder_checkpoint_path)       
        video_encoder.load_state_dict(checkpoint["video_encoder"])

        # Initialize linear probing model
        linear_probing: LinearProbing = ModelRegistry.get(
            name=self.config.pipeline_project
        )(
            backbone=video_encoder,
            head_linear_probing=self.config.head_linear_probing,
            head_structure=self.config.head_structure,
            dropout=self.config.dropout,
            freeze_backbone_ratio=self.config.video_freeze_ratio,
        )
        linear_probing = linear_probing.to(self.config.device).float()

        # Distribute linear probing model
        linear_probing = DistributedUtils.DDP(
            linear_probing, 
            device_ids=[self.config.device]
        )
        
        # Initialize optimizer with separate learning rates for backbone and heads
        param_groups = [
            {
                'params': linear_probing.module.backbone.parameters(),  # Backbone parameters
                'lr': self.config.video_encoder_lr,  # Lower learning rate for backbone
                'name': 'backbone',
                'weight_decay': self.config.video_encoder_weight_decay
            }
        ]
        for head_name in self.config.head_structure:
            param_groups.append({
                'params': linear_probing.module.heads[head_name].parameters(),
                'lr': self.config.head_lr[head_name],
                'name': head_name,
                'weight_decay': self.config.head_weight_decay[head_name]
            })
            
        optimizer_class: torch.optim.Optimizer = getattr(torch.optim, self.config.optimizer)
        optimizer: torch.optim.Optimizer = optimizer_class(param_groups)
        # Initialize scheduler
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

        # Initialize scaler
        print(f"Using AMP: {self.config.use_amp}")
        scaler: GradScaler = GradScaler() if self.config.use_amp else None
        
        # Create loss function
        loss_fn: Loss = Loss(
            loss_type=LossRegistry.get(
                name=LossType.MULTI_HEAD
            )(
                head_structure=self.config.head_structure,
                loss_structure=self.config.loss_structure,
                head_weights=self.config.head_weights,
            )
        )

        return {
            "linear_probing": linear_probing,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "scaler": scaler,
            "output_dir": self.config.output_dir if self.config.is_ref_device else None,
            "loss_fn": loss_fn,
        }            
        