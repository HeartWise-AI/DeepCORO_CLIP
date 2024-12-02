"""Training script for DeepCORO_CLIP model."""

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.distributed import _find_tensors
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.model import TextEncoder, VideoEncoder, contrastive_loss
from utils.data_processing.video import (
    StatsDataset,
    Video,
    VideoDataset,
    custom_collate_fn,
    load_video,
)
from utils.logging import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepCORO_CLIP model")
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU index to use (forces single GPU training)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with model weight biases"
    )
    parser.add_argument(
        "--temp", type=float, default=0.1, help="Temperature for contrastive loss"
    )
    args = parser.parse_args()

    # Also check environment variable as recommended by PyTorch
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    return args


def custom_collate_fn(batch):
    """Custom collate function to handle video and text data.

    Args:
        batch: List of tuples (video, encoded_text, path)
        Each video has shape [F, H, W, C]
    Returns:
        videos: Tensor of shape (batch_size, C, F, H, W) for MViT compatibility
        encoded_texts: Dictionary with input_ids and attention_mask tensors
        paths: List of file paths
    """
    videos, encoded_texts, paths = zip(*batch)

    # Stack videos - handle both tensor and numpy inputs
    videos = torch.stack([torch.from_numpy(v) for v in videos])  # Shape: [B, F, H, W, C]

    # Permute dimensions from [B, F, H, W, C] to [B, C, F, H, W] for MViT
    videos = videos.permute(0, 4, 1, 2, 3)

    # Combine encoded texts
    if encoded_texts[0] is not None:
        combined_texts = {
            "input_ids": torch.stack([text["input_ids"] for text in encoded_texts]),
            "attention_mask": torch.stack([text["attention_mask"] for text in encoded_texts]),
        }
    else:
        combined_texts = None

    return videos, combined_texts, paths


def setup_ddp(rank, world_size):
    """Initialize DDP process group with proper error handling"""
    try:
        # Set the device
        if torch.cuda.is_available():
            # Remove any CUDA_VISIBLE_DEVICES restriction
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            torch.cuda.set_device(rank)  # Set to use the correct GPU
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        # Initialize process group
        init_method = "env://"  # Using environment variables for initialization
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),
        )

        # Verify initialization
        if not dist.is_initialized():
            raise RuntimeError("Failed to initialize process group")

        if rank == 0:
            print(f"Initialized process group: rank {rank} on device {device}")
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        return device

    except Exception as e:
        print(f"Error in DDP setup on rank {rank}: {str(e)}")
        raise e


def cleanup_ddp():
    """Clean up DDP process group with proper error handling"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Error in DDP cleanup: {str(e)}")


def train_epoch(
    video_encoder,
    text_encoder,
    dataloader,
    optimizer,
    device,
    logger,
    rank=0,
    world_size=1,
    epoch=0,
):
    """Training epoch with proper DDP synchronization"""
    video_encoder.train()
    text_encoder.train()

    # Print GPU information at the start of each epoch
    if rank == 0:
        print(f"\nGPU Usage for Epoch {epoch}:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"Memory allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB"
            )
            print(f"Memory cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

    total_loss = 0.0
    num_batches = 0

    # Use different progress bars for main process vs others
    if rank == 0:
        progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        progress = dataloader

    for batch_idx, batch in enumerate(progress):
        try:
            # Unpack batch
            videos, encoded_texts, _ = batch

            # Skip batch if no text data
            if encoded_texts is None:
                if rank == 0:
                    print(f"Skipping batch {batch_idx} due to missing text data")
                continue

            # Print device information for first batch
            if batch_idx == 0:
                print(f"Rank {rank} - Processing on device: {device}")
                print(f"Rank {rank} - Video tensor device: {videos.device}")
                print(f"Rank {rank} - Text tensor device: {encoded_texts['input_ids'].device}")
                print(f"Rank {rank} - Text tensor shape: {encoded_texts['input_ids'].shape}")
                print(f"Rank {rank} - Text tensor max value: {encoded_texts['input_ids'].max()}")

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Move data to device
            videos = videos.to(device, non_blocking=True)
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            # Forward passes
            video_features = video_encoder(videos)
            text_features = text_encoder(input_ids, attention_mask)

            # Print shapes for debugging
            if batch_idx == 0:
                print(f"Rank {rank} - Video features shape: {video_features.shape}")
                print(f"Rank {rank} - Text features shape: {text_features.shape}")

            # Compute loss
            loss = contrastive_loss(video_features, text_features)

            # Check for invalid loss
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"Warning: Batch {batch_idx} - Invalid loss value: {loss.item()}")
                    print(f"  video_features norm: {torch.norm(video_features)}")
                    print(f"  text_features norm: {torch.norm(text_features)}")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=5.0)

            # Step optimizer
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Log progress and metrics on rank 0
            if rank == 0:
                progress.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{(total_loss/num_batches):.4f}",
                        "device": str(device),
                    }
                )

                # Log to wandb
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "batch_avg_loss": total_loss / num_batches,
                        "batch": batch_idx + len(dataloader) * epoch,
                        "video_features_norm": torch.norm(video_features).item(),
                        "text_features_norm": torch.norm(text_features).item(),
                        "epoch": epoch,
                        "gpu_memory": {
                            f"gpu_{i}": torch.cuda.memory_allocated(i) / 1024**2
                            for i in range(torch.cuda.device_count())
                        },
                    }
                )

        except RuntimeError as e:
            print(f"Error in batch {batch_idx} on rank {rank} (device {device}): {str(e)}")
            print(f"Debug info:")
            print(f"  Video shape: {videos.shape}")
            print(f"  Input IDs shape: {input_ids.shape}")
            print(f"  Input IDs max value: {input_ids.max().item()}")
            print(f"  Attention mask shape: {attention_mask.shape}")
            continue

    # Average loss across all processes
    if world_size > 1:
        # All-reduce for loss synchronization
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Log epoch metrics on rank 0
    if rank == 0:
        print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")
        print(f"Processed {num_batches} batches")
        wandb.log({"epoch": epoch, "epoch_avg_loss": avg_loss, "num_batches": num_batches})

    return avg_loss


def remove_model_biases(model):
    """Initialize model weights properly for training."""
    for name, param in model.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param.data)
        elif "weight" in name:
            if len(param.shape) > 1:
                # For weight matrices, use Xavier initialization
                nn.init.xavier_uniform_(param.data)
            else:
                # For 1D weights (like in LayerNorm), use normal initialization
                nn.init.normal_(param.data, mean=0.0, std=0.02)
    return model


def main(rank=0, world_size=1, args=None):
    """Main training function with proper DDP setup"""
    if args is None:
        args = parse_args()

    logger = None
    try:
        # Determine if we're doing distributed training
        is_distributed = world_size > 1 and args.gpu is None

        # Set up device and initialize process group first
        if is_distributed:
            # Only initialize process group for distributed training
            device = setup_ddp(rank, world_size)
            # Disable wandb for non-zero ranks
            if rank != 0:
                os.environ["WANDB_MODE"] = "disabled"
        else:
            # Single GPU setup - no DDP needed
            if args.gpu is not None:
                device = torch.device(f"cuda:{args.gpu}")
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize wandb only on rank 0 (both for distributed and single GPU)
        should_log = rank == 0  # Only log on rank 0, regardless of distributed mode

        if should_log:
            # Ensure wandb is not already initialized
            try:
                wandb.finish()
            except:
                pass

            # Set unique run ID to prevent multiple runs
            os.environ["WANDB_RUN_ID"] = f"run_{int(time.time())}"

            wandb.init(
                project="DeepCORO_CLIP",
                config={
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "num_workers": args.num_workers,
                    "debug_mode": args.debug,
                    "world_size": world_size,
                    "distributed": is_distributed,
                    "device": str(device),
                    "rank": rank,
                },
            )
            logger = create_logger(args)

        # Create models
        video_encoder = VideoEncoder().to(device)
        text_encoder = TextEncoder().to(device)

        # Remove model biases if not in debug mode
        if not args.debug:
            if should_log:
                print("Debug mode disabled - Removing model biases and setting uniform weights")
            video_encoder = remove_model_biases(video_encoder)
            text_encoder = remove_model_biases(text_encoder)
        elif should_log:
            print("Debug mode enabled - Keeping original model weights and biases")

        # Set up distributed training if needed
        if is_distributed:
            # Convert BatchNorm to SyncBatchNorm for multi-GPU training
            video_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(video_encoder)
            text_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(text_encoder)

            # Wrap models with DDP
            video_encoder = DDP(
                video_encoder,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
                broadcast_buffers=True,
            )
            text_encoder = DDP(
                text_encoder,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
                broadcast_buffers=True,
            )

        # Create dataset
        train_dataset = VideoDataset(
            root="data/",
            data_filename="processed/reports/reports_sampled_1000.csv",
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            num_frames=16,
            backbone="mvit",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            normalize=True,
            debug=should_log,
            batch_size=args.batch_size,
        )

        if should_log:
            print(f"Dataset size: {len(train_dataset)}")

        # Create sampler and dataloader
        train_sampler = (
            DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
            )
            if is_distributed
            else None
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=custom_collate_fn,
        )

        # Optimizer with proper DDP parameter handling
        params = [{"params": video_encoder.parameters()}, {"params": text_encoder.parameters()}]
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        # Training loop
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss = train_epoch(
                video_encoder=video_encoder,
                text_encoder=text_encoder,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                logger=logger,
                rank=rank if is_distributed else 0,
                world_size=world_size if is_distributed else 1,
                epoch=epoch,
            )

            # Save checkpoints only on main process
            if should_log:
                checkpoint = {
                    "video_encoder": (
                        video_encoder.module.state_dict()
                        if is_distributed
                        else video_encoder.state_dict()
                    ),
                    "text_encoder": (
                        text_encoder.module.state_dict()
                        if is_distributed
                        else text_encoder.state_dict()
                    ),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": train_loss,
                }

                if (epoch + 1) % 5 == 0:
                    checkpoint_dir = Path("models/checkpoints")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint for epoch {epoch+1}")
                    wandb.save(str(checkpoint_path))

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e

    finally:
        if logger:
            logger.finish()
        if is_distributed:
            cleanup_ddp()


if __name__ == "__main__":
    args = parse_args()

    if args.gpu is not None:
        # Use specified GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        main(rank=0, world_size=1, args=args)
    else:
        # Default to GPU 0 for single GPU mode
        if torch.cuda.device_count() == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            main(rank=0, world_size=1, args=args)
        else:
            # Multi-GPU mode - remove any GPU restrictions
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            main(rank=0, world_size=1, args=args)

    # Multi-GPU mode must be explicitly triggered using torchrun or torch.distributed.launch
