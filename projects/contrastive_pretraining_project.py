import os

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from typing import Any
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
from utils.files_handler import generate_output_dir_name
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
        if self.config.is_ref_device:
            self._setup_project()
                        
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
                'params': video_encoder.module.aggregator.parameters(),  # Multihead attention aggregator
                'lr': self.config.lr * 2.0,  # Higher learning rate for aggregator
                'name': 'video_aggregator',
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

        scaler: GradScaler = GradScaler() if self.config.use_amp else None

        # Create loss function
        loss_fn: Loss = Loss(
            loss_type=LossRegistry.get(self.config.loss_name)()
        )

        # Setup output directory (fix for NameError)
        full_output_path = None
        if self.config.is_ref_device:
            run_id = self.wandb_wrapper.get_run_id() if self.wandb_wrapper.is_initialized() else None
            output_subdir = generate_output_dir_name(self.config, run_id)
            full_output_path = os.path.join(self.config.output_dir, output_subdir) if hasattr(self.config, 'output_dir') else output_subdir
            os.makedirs(full_output_path, exist_ok=True)

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
            "output_dir": full_output_path,
            "loss_fn": loss_fn,
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
        )        
        video_encoder = video_encoder.to(self.config.device).float()
        
        video_encoder = DistributedUtils.DDP(
            video_encoder, 
            device_ids=[self.config.device], 
        )
        
        checkpoint: dict[str, Any] = self._load_checkpoint(self.config.checkpoint)
        video_encoder.module.load_state_dict(checkpoint["video_encoder"], weight_only=True)
        log_temp: float = checkpoint["train/temperature"]

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
        training_setup["log_temp"].data.copy_(checkpoint["train/temperature"])
        return training_setup
        
    def run(self):
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

