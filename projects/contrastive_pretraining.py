import os
import wandb
import pickle

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from typing import Any
from models.text_encoder import TextEncoder
from models.video_encoder import VideoEncoder

from runners.video_constrative_learning import VideoContrastiveLearningRunner

from utils.ddp import DDP
from utils.losses import get_loss_fn
from utils.config import HeartWiseConfig
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry, 
    RunnerRegistry, 
    ProjectRegistry, 
)
from utils.logging import create_logger
from dataloaders.stats_dataset import get_stats_dataloader
from dataloaders.video_dataset import get_distributed_video_dataloader


def save_checkpoint(model_dict, metrics_dict, output_path, is_best=False):
    """
    Save model checkpoint with metrics.

    Args:
        model_dict (dict): Dictionary containing model states
        metrics_dict (dict): Dictionary containing training metrics
        output_path (str/Path): Path to save the checkpoint
        is_best (bool): Whether this is the best model so far
    """
    # Ensure parent directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine model and metrics into one checkpoint
    checkpoint = {**model_dict, **metrics_dict}

    # Save checkpoint
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")


def load_train_objs(
    config: HeartWiseConfig,
)->dict:
    wandb_run = None
    if config.is_ref_device:
        wandb_run = create_logger(config=config)

        # After wandb.init(), wandb.config is available.
        # Override args with wandb.config parameters if present.
        if wandb_run is not None and len(wandb_run.config.keys()) > 0:
            for key, value in wandb_run.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"Warning: {key} in wandb.config not recognized as an arg.")    
                    
    # Calculate dataset statistics (only on rank 0)
    mean, std = None, None
    if config.is_ref_device:
        print("\n=== Calculating Dataset Statistics ===")

        stats_loader: DataLoader = get_stats_dataloader(config)

        num_stats_samples = min(100, 1000)
        print(f"Stats dataset length: {len(stats_loader)}")
        if len(stats_loader) > num_stats_samples:
            indices = torch.linspace(0, len(stats_loader) - 1, num_stats_samples).long().tolist()
            stats_loader = torch.utils.data.Subset(stats_loader, indices)

        print(f"\nUsing {num_stats_samples} samples for statistics calculation")
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
        print(f"Calculated from {num_stats_samples} samples ({pixel_count:,} pixels)")
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


    train_loader: DataLoader = get_distributed_video_dataloader(
        config, 
        split="train", 
        mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
        std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
        shuffle=True,
        num_replicas=config.world_size,
        rank=config.gpu,
    )
    val_loader: DataLoader = get_distributed_video_dataloader(
        config, 
        split="val", 
        mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
        std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
        shuffle=False,
        num_replicas=config.world_size,
        rank=config.gpu,
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
    )
    video_encoder = video_encoder.to(config.gpu).float()

    text_encoder: TextEncoder = ModelRegistry.get(
        name="text_encoder"
    )()
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

    # Include the temperature parameter in the optimizer
    optimizer_class: torch.optim.Optimizer = getattr(torch.optim, config.optimizer)
    optimizer: torch.optim.Optimizer = optimizer_class(
        [
            {"params": video_encoder.parameters()},
            {"params": text_encoder.parameters()},
            {"params": [log_temperature]},
        ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler: LRScheduler = get_scheduler(
        scheduler_name=config.scheduler_type,
        optimizer=optimizer,
        num_epochs=config.epochs,
        factor=config.factor,
        step_size=config.lr_step_period,
    )

    scaler: GradScaler = GradScaler() if config.use_amp else None

    if config.is_ref_device:
        print("\n=== Dataset Information ===")
        print(f"Training:   {len(train_loader):,} videos")
        print(f"Validation: {len(val_loader):,} videos")
        print(f"Total:      {len(train_loader) + len(val_loader):,} videos")
        print(f"\nBatch Size: {config.batch_size}")
        print(f"Training Batches: {len(train_loader) // config.batch_size:,}")
        print(
            f"Validation Batches: {len(val_loader) // config.batch_size + (1 if len(val_loader) % config.batch_size else 0):,}"
        )
        print("===========================\n")

    return {
        "video_encoder": video_encoder,
        "text_encoder": text_encoder,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": config.gpu,
        "wandb_run": wandb_run,
        "scaler": scaler,
        "log_temp": log_temperature,
    }


@ProjectRegistry.register('contrastive_pretraining')
class ContrastivePretraining:
    def __init__(
        self, 
        config: HeartWiseConfig,
    ):
        self.config: HeartWiseConfig = config
    
    def run(self):
        training_setup: dict[str, Any] = load_train_objs(
            config=self.config
        )
                
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
            wandb_wrapper=training_setup["wandb_run"],
            optimizer=training_setup["optimizer"],
            scaler=training_setup["scaler"],
            log_temp=training_setup["log_temp"],
            lr_scheduler=training_setup["scheduler"],
            loss_fn=get_loss_fn(self.config.loss_name),
        )
        
        runner.train()        
        # is_distributed = self.world_size > 1
        # text_encoder = training_setup["text_encoder"]
        # train_loader = training_setup["train_loader"]
        # val_loader = training_setup["val_loader"]
        # device = self.device
        # log_temp = training_setup["log_temp"] if "log_temp" in training_setup else None

        # best_val_loss = float("inf")
        # best_epoch = -1

        # # Main training loop
        # for epoch in range(self.args.epochs):
        #     if self.device == 0:
        #         print(f"\nEpoch {epoch + 1}/{self.args.epochs}")

        #     train_loss, train_metrics = train_epoch(
        #         args=self.args,
        #         video_encoder=training_setup["video_encoder"],
        #         text_encoder=text_encoder,
        #         dataloader=train_loader,
        #         optimizer=training_setup["optimizer"],
        #         device=self.device,
        #         wandb_run=training_setup["wandb_run"],
        #         rank=self.args.rank,
        #         world_size=self.world_size,
        #         epoch=epoch,
        #     )

        #     # Validate on validation-only embeddings (use_val_only_pool=True)
        #     val_loss_valpool, val_metrics_valpool, _ = validate_epoch(
        #         args=self.args,
        #         video_encoder=training_setup["video_encoder"],
        #         text_encoder=text_encoder,
        #         dataloader=val_loader,
        #         device=self.device,
        #         wandb_run=training_setup["wandb_run"],
        #         rank=self.args.rank,
        #         world_size=self.world_size,
        #         epoch=epoch,
        #         all_text_embeddings=val_text_embeddings,
        #         all_reports=val_reports,
        #         text_embedding_pickle_path=os.path.join(
        #             full_output_path, "val_text_embeddings.pkl"
        #         ),
        #         output_dir=full_output_path,
        #         report_to_global_index=val_report_to_index,
        #         use_val_only_pool=True,  # <-- Val-only retrievals
        #     )

        #     # Validate on global embeddings (train+val) (use_val_only_pool=False)
        #     val_loss_global, val_metrics_global, _ = validate_epoch(
        #         args=self.args,
        #         video_encoder=training_setup["video_encoder"],
        #         text_encoder=text_encoder,
        #         dataloader=val_loader,
        #         device=self.device,
        #         wandb_run=training_setup["wandb_run"],
        #         rank=self.args.rank,
        #         world_size=self.world_size,
        #         epoch=epoch,
        #         all_text_embeddings=all_global_text_embeddings,
        #         all_reports=all_global_reports,
        #         text_embedding_pickle_path=os.path.join(
        #             full_output_path, "global_text_embeddings.pkl"
        #         ),
        #         output_dir=full_output_path,
        #         report_to_global_index=report_to_global_index,
        #         use_val_only_pool=False,  # <-- Global retrievals without top/bottom examples
        #     )

        #     # Choose one for best model comparison (typically val-only)
        #     current_val_loss = val_loss_valpool

        #     if self.device == 0 and training_setup["wandb_run"] is not None:
        #         log_data = {
        #             "epoch": epoch,
        #             "train/loss": train_loss,
        #             "train/learning_rate": training_setup["optimizer"].param_groups[0]["lr"],
        #             "val_only/loss": val_loss_valpool,
        #             **{f"val_only/{k}": v for k, v in val_metrics_valpool.items()},
        #             "val_global/loss": val_loss_global,
        #             **{f"val_global/{k}": v for k, v in val_metrics_global.items()},
        #             "best_val_loss": best_val_loss,  # Log the current best_val_loss each epoch
        #         }
        #         # Log temperature if available
        #         if log_temp is not None:
        #             current_temp = torch.exp(log_temp).item()
        #             log_data["temperature"] = current_temp
        #             log_data["log_temp"] = log_temp.item()

        #         training_setup["wandb_run"].log(log_data)

        #     if training_setup["scheduler"] is not None:
        #         training_setup["scheduler"].step()

        #     if self.device == 0:
        #         model_dict = {
        #             "video_encoder": (
        #                 training_setup["video_encoder"].module.state_dict()
        #                 if is_distributed
        #                 else training_setup["video_encoder"].state_dict()
        #             ),
        #             "text_encoder": (
        #                 text_encoder.module.state_dict()
        #                 if is_distributed
        #                 else text_encoder.state_dict()
        #             ),
        #             "optimizer": training_setup["optimizer"].state_dict(),
        #             "scheduler": (
        #                 training_setup["scheduler"].state_dict()
        #                 if training_setup["scheduler"] is not None
        #                 else None
        #             ),
        #             "epoch": epoch,
        #         }

        #         metrics_dict = {
        #             "train_loss": train_loss,
        #             "val_loss_valpool": val_loss_valpool,
        #             "val_loss_global": val_loss_global,
        #             "best_val_loss": best_val_loss,
        #             "best_epoch": best_epoch,
        #             **train_metrics,
        #             **val_metrics_valpool,  # store the val-only metrics
        #             **{f"global_{k}": v for k, v in val_metrics_global.items()},
        #         }

        #         checkpoint_dir = Path(full_output_path) / "checkpoints"
        #         checkpoint_dir.mkdir(parents=True, exist_ok=True)

        #         latest_path = checkpoint_dir / "latest.pt"
        #         save_checkpoint(model_dict, metrics_dict, latest_path)
        #         print(f"\nSaved latest checkpoint at epoch {epoch + 1}")

        #         # Update best model based on val-only performance
        #         if current_val_loss < best_val_loss:
        #             previous_best = best_val_loss
        #             best_val_loss = current_val_loss
        #             best_epoch = epoch
        #             best_path = checkpoint_dir / "best.pt"
        #             save_checkpoint(model_dict, metrics_dict, best_path, is_best=True)
        #             print(
        #                 f"\nNew best model saved! Val Loss (val-only): {current_val_loss:.4f} (previous: {previous_best:.4f})"
        #             )

        #             if training_setup["wandb_run"] is not None:
        #                 # Also log the new best_val_loss immediately when found
        #                 training_setup["wandb_run"].log(
        #                     {
        #                         "best_val_loss": best_val_loss,
        #                         "best_epoch": best_epoch,
        #                         "epoch": epoch,
        #                     }
        #                 )
        #                 training_setup["wandb_run"].save(str(best_path))      

        wandb.finish()


