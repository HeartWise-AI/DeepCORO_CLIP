"""Training script for DeepCORO_CLIP model."""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.distributed import _find_tensors
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_processing.video import StatsDataset, Video, VideoDataset, load_video
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    return parser.parse_args()


def custom_collate_fn(
    batch: List[Tuple[np.ndarray, Any, str]]
) -> Tuple[torch.Tensor, List[Any], List[str]]:
    """Custom collate function to handle video and text data.

    Args:
        batch: List of tuples (video, encoded_text, path)
    Returns:
        videos: Tensor of shape (batch_size, channels, frames, height, width)
        encoded_texts: Dictionary with input_ids and attention_mask tensors
        paths: List of file paths
    """
    videos, encoded_texts, paths = zip(*batch, strict=False)

    # Stack videos - they should all have the same shape after preprocessing
    videos = torch.stack([torch.from_numpy(v) for v in videos])

    # Combine encoded texts
    if encoded_texts[0] is not None:
        combined_texts = {
            "input_ids": torch.cat([text["input_ids"] for text in encoded_texts], dim=0),
            "attention_mask": torch.cat(
                [text["attention_mask"] for text in encoded_texts], dim=0
            ),
        }
    else:
        combined_texts = None

    return videos, combined_texts, paths


class VideoEncoder(nn.Module):
    def __init__(self, input_channels=1):  # Default to 1 for grayscale
        super().__init__()
        self.input_channels = input_channels

        # Convert grayscale to 3 channels if needed
        if input_channels == 1:
            self.to_rgb = nn.Sequential(
                # First expand to 3 channels
                nn.Conv3d(1, 3, kernel_size=1, bias=False),
                nn.BatchNorm3d(3),
                nn.ReLU(inplace=True),
            )
        else:
            self.to_rgb = None

        # MViT outputs 512 dimensions directly to match text encoder
        self.mvit = timm.create_model("mvitv2_base", pretrained=True, num_classes=512)

        # Modify first conv layer to accept video input
        first_conv = self.mvit.patch_embed.proj
        self.mvit.patch_embed.proj = nn.Conv2d(
            3 * 32,  # 3 channels * num_frames
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
        )

    def forward(self, x):
        # Input shape: [batch_size, channels, frames, height, width]
        # x is already in the correct format [B, C, T, H, W]

        # Convert grayscale to RGB if needed
        if self.to_rgb is not None:
            # Take only the first channel if input has 3 channels
            if x.shape[1] == 3:
                x = x[:, 0:1]
            x = self.to_rgb(x)  # [B, 3, T, H, W]

        B, C, T, H, W = x.shape

        # Reshape for MViT: [batch_size, T*C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B, T * C, H, W)  # [B, T*C, H, W]

        # Get features from MViT (outputs [B, 512] directly)
        x = self.mvit(x)

        return x


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        self.proj = nn.Linear(768, 512)

    def forward(self, input_ids, attention_mask):
        features = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.proj(features)


def contrastive_loss(video_emb, text_emb, temperature=0.07):
    """Compute contrastive loss between video and text embeddings."""
    video_emb = nn.functional.normalize(video_emb, dim=1)
    text_emb = nn.functional.normalize(text_emb, dim=1)

    logits = torch.matmul(video_emb, text_emb.t()) / temperature
    labels = torch.arange(len(video_emb)).to(video_emb.device)

    loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2
    return loss


def setup_ddp(rank, world_size):
    """Initialize DDP process group with proper error handling"""
    try:
        # Set the device
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
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
    video_encoder, text_encoder, dataloader, optimizer, device, logger, rank=0, world_size=1
):
    """Training epoch with proper DDP synchronization"""
    video_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    num_batches = 0

    # Use different progress bars for main process vs others
    if rank == 0:
        progress = tqdm(dataloader, desc=f"Epoch {logger.epoch if logger else 0}")
    else:
        progress = dataloader

    for batch_idx, batch in enumerate(progress):
        try:
            # Unpack batch
            videos, encoded_texts, _ = batch

            # Skip batch if no text data
            if encoded_texts is None:
                continue

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Move data to device
            videos = videos.to(device, non_blocking=True)
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            # Forward passes
            video_features = video_encoder(videos)
            text_features = text_encoder(input_ids, attention_mask)

            # Compute loss
            loss = contrastive_loss(video_features, text_features)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)

            # Step optimizer
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Log progress (now updated every step)
            if rank == 0:
                progress.set_postfix({"loss": loss.item()})

        except RuntimeError as e:
            print(f"Error in batch {batch_idx} on rank {rank}: {str(e)}")
            continue

    # Average loss across all processes
    if world_size > 1:
        # All-reduce for loss synchronization
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Only print on main process
    if rank == 0:
        print(f"\nEpoch {logger.epoch if logger else 0} - Average Loss: {avg_loss:.4f}")

    return avg_loss


def main(rank=0, world_size=1, args=None):
    """Main training function with proper DDP setup"""
    if args is None:
        args = parse_args()

    try:
        # Initialize DDP if using multiple GPUs
        if world_size > 1:
            device = setup_ddp(rank, world_size)
        else:
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Initialize logger only on rank 0
        logger = create_logger(args) if rank == 0 else None

        # Create models
        video_encoder = VideoEncoder().to(device)
        text_encoder = TextEncoder().to(device)

        # Convert BatchNorm to SyncBatchNorm for multi-GPU training
        if world_size > 1:
            video_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(video_encoder)
            text_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(text_encoder)

        # Wrap models with DDP if using multiple GPUs
        if world_size > 1:
            video_encoder = DDP(
                video_encoder,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
                broadcast_buffers=True,
            )
            text_encoder = DDP(
                text_encoder,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
                broadcast_buffers=True,
            )

        # Create dataset
        train_dataset = VideoDataset(
            root="data/",
            data_filename="processed/reports/reports_sampled_1000.csv",
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            normalize=True,
            debug=rank == 0,
        )

        if rank == 0:
            print(f"Dataset size: {len(train_dataset)}")

        # Create sampler and dataloader
        train_sampler = (
            DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
            )
            if world_size > 1
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
                rank=rank,
                world_size=world_size,
            )

            # Save checkpoints only on rank 0
            if rank == 0:
                checkpoint = {
                    "video_encoder": (
                        video_encoder.module.state_dict()
                        if world_size > 1
                        else video_encoder.state_dict()
                    ),
                    "text_encoder": (
                        text_encoder.module.state_dict()
                        if world_size > 1
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

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e

    finally:
        # Cleanup
        if rank == 0 and logger:
            logger.finish()

        if world_size > 1:
            cleanup_ddp()


if __name__ == "__main__":
    args = parse_args()

    # Force single GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        main(rank=0, world_size=1, args=args)
    else:
        # Multi-GPU setup
        world_size = torch.cuda.device_count()
        if world_size > 1:
            # Set environment variables
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # Launch processes
            import torch.multiprocessing as mp

            mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
        else:
            # Single GPU setup
            main(rank=0, world_size=1, args=args)
