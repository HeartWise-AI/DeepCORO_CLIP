import os
import wandb

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import GradScaler
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
from utils.files_handler import generate_output_dir_name
from utils.video_project import calculate_dataset_statistics_ddp
from dataloaders.video_clip_dataset import get_distributed_video_clip_dataloader
from dataloaders.multi_video_dataset import get_distributed_multi_video_dataloader

def stats_collate_fn(batch):
    """Collate function for StatsDataset that stacks video tensors."""
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        raise RuntimeError("No valid samples in batch")
    videos = torch.stack([torch.from_numpy(sample[0]) for sample in valid_samples])
    return videos

@ProjectRegistry.register('DeepCORO_clip')
class ContrastivePretrainingProject(BaseProject):
    def __init__(
        self, 
        config: ClipConfig,
    ):
        self.config: ClipConfig = config
        
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
        full_output_path = None
        if self.config.is_ref_device:
            # Generate output directory using wandb run ID that was already created
            run_id = wandb.run.id if wandb.run is not None else ""
            output_subdir = generate_output_dir_name(self.config, run_id)
            full_output_path = os.path.join(self.config.output_dir, output_subdir)        
            os.makedirs(full_output_path, exist_ok=True)
                        
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)

        if self.config.multi_video:
            train_loader: DataLoader = get_distributed_multi_video_dataloader(
                self.config, 
                split="train",
                mean=mean.tolist(),
                std=std.tolist(),
                shuffle=True,
                num_replicas=self.config.world_size,
                rank=self.config.device,
                drop_last=True,
            )
            val_loader: DataLoader = get_distributed_multi_video_dataloader(
                self.config, 
                split="val", 
                mean=mean.tolist(),
                std=std.tolist(),
                shuffle=False,
                num_replicas=self.config.world_size,
                rank=self.config.device,
                drop_last=False,
            )
        else:
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
            find_unused_parameters=True
        )
        text_encoder = DistributedUtils.DDP(
            text_encoder, 
            device_ids=[self.config.device], 
            find_unused_parameters=True
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
                'lr': 0.00001,  # Lower learning rate for text encoder
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

        if self.config.is_ref_device:
            wandb.config.update(
                {
                    "train_dataset_size": len(train_loader),
                    "val_dataset_size": len(val_loader),
                },
                allow_val_change=True,
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
            "scheduler": scheduler,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "device": self.config.device,
            "scaler": scaler,
            "log_temp": log_temperature,
            "full_output_path": full_output_path,
            "loss_fn": loss_fn,
        }    

    def _setup_inference_objects(
        self,
    )->dict:
        raise NotImplementedError("Inference is not implemented for this project")
    
    def _save_texts_csv(
        self, 
        output_dir: str, 
        texts: list[str]
    ):
        import csv
        csv_path = os.path.join(output_dir, "val_texts.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Text"])
            for idx, txt in enumerate(texts):
                writer.writerow([idx, txt])
        print(f"Saved {len(texts)} val texts to {csv_path}")    

    def _update_training_setup_with_checkpoint(
        self, 
        training_setup: dict[str, Any], 
        checkpoint: dict[str, Any]
    )->dict[str, Any]:
        print(f"Resuming from checkpoint: {checkpoint.keys()}")
        training_setup["video_encoder"].module.load_state_dict(checkpoint["video_encoder"])
        training_setup["text_encoder"].module.load_state_dict(checkpoint["text_encoder"])
        training_setup["optimizer"].load_state_dict(checkpoint["optimizer"])
        training_setup["scheduler"].load_state_dict(checkpoint["scheduler"])
        training_setup["scaler"].load_state_dict(checkpoint["scaler"])
        training_setup["log_temp"].data.copy_(checkpoint["train/temperature"])
        return training_setup
        
    def run(self):
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
                device=self.config.device,
                world_size=self.config.world_size,
                train_loader=training_setup["train_loader"],
                val_loader=training_setup["val_loader"],
                video_encoder=training_setup["video_encoder"],
                text_encoder=training_setup["text_encoder"],
                optimizer=training_setup["optimizer"],
                scaler=training_setup["scaler"],
                log_temp=training_setup["log_temp"],
                lr_scheduler=training_setup["scheduler"],
                loss_fn=training_setup["loss_fn"],
                output_dir=training_setup["full_output_path"],
            )
        )
        if self.config.run_mode == RunMode.TRAIN:
            end_epoch = start_epoch + self.config.epochs
            runner.train(
                start_epoch=start_epoch, 
                end_epoch=end_epoch
            ) 
        elif self.config.run_mode == RunMode.INFERENCE:
            runner.inference()
        else:
            raise ValueError(f"Invalid run mode: {self.config.run_mode}, must be one of {RunMode.TRAIN} or {RunMode.INFERENCE}")

