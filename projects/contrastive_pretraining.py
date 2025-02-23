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

from runners.video_constrative_learning import VideoContrastiveLearningRunner

from utils.ddp import DDP
from utils.enums import RunMode
from utils.losses import get_loss_fn
from utils.config import HeartWiseConfig
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry, 
    RunnerRegistry, 
    ProjectRegistry, 
)
from utils.files_handler import generate_output_dir_name
from dataloaders.stats_dataset import get_stats_dataloader
from dataloaders.video_dataset import get_distributed_video_dataloader
from dataloaders.multi_video_dataset import get_distributed_multi_video_dataloader

def stats_collate_fn(batch):
    """Collate function for StatsDataset that stacks video tensors."""
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        raise RuntimeError("No valid samples in batch")
    videos = torch.stack([torch.from_numpy(sample[0]) for sample in valid_samples])
    return videos


def load_train_objs(
    config: HeartWiseConfig,
)->dict:
    """
    Load training objects.

    Args:
        config (HeartWiseConfig): Configuration object

    Returns:
        dict: Dictionary containing training objects
    """
    full_output_path = None
    if config.is_ref_device:
        # Generate output directory using wandb run ID that was already created
        run_id = wandb.run.id if wandb.run is not None else ""
        output_subdir = generate_output_dir_name(config, run_id)
        full_output_path = os.path.join(config.output_dir, output_subdir)        
        os.makedirs(full_output_path, exist_ok=True)
                    
    # Calculate dataset statistics (only on rank 0)
    mean, std = None, None
    if config.is_ref_device:
        print("\n=== Calculating Dataset Statistics ===")

        stats_loader: DataLoader = get_stats_dataloader(config)

        print(f"Frame count per video: {config.frames}")

        mean_sum, squared_sum, pixel_count = 0.0, 0.0, 0
        for batch in tqdm(stats_loader, desc="Calculating statistics"):
            batch = batch.float()
            batch = batch.reshape(-1, batch.shape[-1])
            mean_sum += batch.sum(dim=0)
            squared_sum += (batch**2).sum(dim=0)
            pixel_count += batch.shape[0]

        mean: torch.Tensor = mean_sum / pixel_count
        std: torch.Tensor = torch.sqrt((squared_sum / pixel_count) - (mean**2))

        print("\nDataset Statistics:")
        print(f"Mean: {mean.tolist()}")
        print(f"Std:  {std.tolist()}")
        print("===========================\n")                    
    
    # Broadcast stats if distributed
    if torch.distributed.is_initialized():
        if mean is not None:
            mean = mean.cuda()
            std = std.cuda()
        mean_tensor = torch.zeros(3, device="cuda")
        std_tensor = torch.zeros(3, device="cuda")
        if config.is_ref_device:
            mean_tensor.copy_(mean)
            std_tensor.copy_(std)
        torch.distributed.broadcast(mean_tensor, 0)
        torch.distributed.broadcast(std_tensor, 0)
        mean = mean_tensor.cpu()
        std = std_tensor.cpu()    
    
    print(f"Rank: {config.gpu} - mean: {mean} - std: {std}")


    if config.multi_video:
        train_loader: DataLoader = get_distributed_multi_video_dataloader(
            config, 
            split="train",
            mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
            std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
            shuffle=True,
            num_replicas=config.world_size,
            rank=config.gpu,
            drop_last=True,
        )
        val_loader: DataLoader = get_distributed_multi_video_dataloader(
            config, 
            split="val", 
            mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
            std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
            shuffle=False,
            num_replicas=config.world_size,
            rank=config.gpu,
            drop_last=False,
        )
    else:
        print("DEBUG single video")
        train_loader: DataLoader = get_distributed_video_dataloader(
            config, 
            split="train", 
            mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
            std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
            shuffle=True,
            num_replicas=config.world_size,
            rank=config.gpu,
            drop_last=True,
        )
        val_loader: DataLoader = get_distributed_video_dataloader(
            config, 
            split="val", 
            mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
            std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
            shuffle=False,
            num_replicas=config.world_size,
            rank=config.gpu,
            drop_last=False,
        )

    # Create models
    video_encoder: VideoEncoder = ModelRegistry.get(
        name="video_encoder"
    )(
        backbone=config.model_name,
        input_channels=3,
        num_frames=config.frames,
        pretrained=config.pretrained,
        output_dim=512,
        freeze_ratio=config.video_freeze_ratio,
        dropout=config.dropout,
        num_heads=config.num_heads,
        aggregator_depth=config.aggregator_depth,
    )
    video_encoder = video_encoder.to(config.gpu).float()

    text_encoder: TextEncoder = ModelRegistry.get(
        name="text_encoder"
    )(
        freeze_ratio=config.text_freeze_ratio,
        dropout=config.dropout,
    )
    text_encoder = text_encoder.to(config.gpu).float()

    video_encoder = DDP(
        video_encoder, 
        device_ids=[config.gpu], 
        find_unused_parameters=True
    )
    text_encoder = DDP(
        text_encoder, 
        device_ids=[config.gpu], 
        find_unused_parameters=True
    )

    # Make temperature a trainable parameter directly on the device
    log_temperature: nn.Parameter = nn.Parameter(
        torch.log(torch.tensor([config.temperature], dtype=torch.float32, device=config.gpu))
    )

    # Different learning rates for different components
    param_groups = [
        {
            'params': video_encoder.module.model.parameters(),  # Main video backbone
            'lr': config.lr,
            'name': 'video_backbone',
            'weight_decay': config.video_weight_decay
        },
        {
            'params': video_encoder.module.aggregator.parameters(),  # Multihead attention aggregator
            'lr': config.lr * 5.0,  # Higher learning rate for aggregator
            'name': 'video_aggregator',
            'weight_decay': config.video_weight_decay
        },
        {
            'params': video_encoder.module.proj.parameters(),  # Video projection
            'lr': config.lr,
            'name': 'video_proj',
            'weight_decay': config.video_weight_decay
        },
        {
            'params': text_encoder.parameters(),  # Entire text encoder
            'lr': 0.000009,  # Lower learning rate for text encoder
            'name': 'text_encoder',
            'weight_decay': config.text_weight_decay
        },
        {
            'params': [log_temperature],  # Temperature parameter
            'lr': config.lr,
            'name': 'temperature'
        }
    ]

    # Include the temperature parameter in the optimizer
    optimizer_class: torch.optim.Optimizer = getattr(torch.optim, config.optimizer)
    optimizer: torch.optim.Optimizer = optimizer_class(
        param_groups,
        lr=config.lr
    )

    scheduler: LRScheduler = get_scheduler(
        scheduler_name=config.scheduler_type,
        optimizer=optimizer,
        num_epochs=config.epochs,
        train_dataloader=train_loader,
        factor=config.factor,
        step_size=config.lr_step_period,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_warmup_percent=config.num_warmup_percent,
    )

    scaler: GradScaler = GradScaler() if config.use_amp else None

    if config.is_ref_device:
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
        print(f"\nBatch Size: {config.batch_size}")
        print(f"Training: {len(train_loader) * config.batch_size:,} videos per GPU")
        print(f"Validation: {len(val_loader) * config.batch_size:,} videos per GPU")
        print(f"Total: {(len(train_loader) + len(val_loader)) * config.batch_size:,} videos per GPU")
        print("===========================\n")

    return {
        "video_encoder": video_encoder,
        "text_encoder": text_encoder,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": config.gpu,
        "scaler": scaler,
        "log_temp": log_temperature,
        "full_output_path": full_output_path,
    }


@ProjectRegistry.register('contrastive_pretraining')
class ContrastivePretraining:
    def __init__(
        self, 
        config: HeartWiseConfig,
    ):
        self.config: HeartWiseConfig = config
    
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
    
    def _load_checkpoint(
        self, 
        checkpoint_path: str
    )->dict[str, Any]:
        if not self.config.resume_training:
            raise ValueError("Flag 'resume_training' is set, but no checkpoint provided")

        if not os.path.exists(self.config.checkpoint):
            raise ValueError(f"Checkpoint file does not exist: {self.config.checkpoint}")
        
        print(
            f"[VideoContrastiveLearningRunner] Resuming from checkpoint: {self.config.checkpoint}"
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    
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
        training_setup: dict[str, Any] = load_train_objs(
            config=self.config
        )

        start_epoch = 0
        if self.config.resume_training:
            checkpoint = self._load_checkpoint(self.config.checkpoint)
            training_setup = self._update_training_setup_with_checkpoint(training_setup, checkpoint)
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch: {start_epoch}")

        runner: VideoContrastiveLearningRunner = RunnerRegistry.get(
            name="video_contrastive_learning"
        )(
            config=self.config,
            device=self.config.gpu,
            world_size=self.config.world_size,
            train_loader=training_setup["train_loader"],
            val_loader=training_setup["val_loader"],
            video_encoder=training_setup["video_encoder"],
            text_encoder=training_setup["text_encoder"],
            optimizer=training_setup["optimizer"],
            scaler=training_setup["scaler"],
            log_temp=training_setup["log_temp"],
            lr_scheduler=training_setup["scheduler"],
            loss_fn=get_loss_fn(self.config.loss_name),
            output_dir=training_setup["full_output_path"],
        )
        if self.config.mode == RunMode.TRAIN:
            end_epoch = start_epoch + self.config.epochs
            runner.train(
                start_epoch=start_epoch, 
                end_epoch=end_epoch
            ) 
        elif self.config.mode == RunMode.INFERENCE:
            runner.inference()
        else:
            raise ValueError(f"Invalid mode: {self.config.mode}, must be one of {RunMode.TRAIN} or {RunMode.INFERENCE}")

