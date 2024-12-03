"""Training script for DeepCORO_CLIP model."""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import imageio
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
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
    stats_collate_fn,
)
from utils.logging import (
    cleanup_temp_video,
    convert_video_for_wandb,
    create_logger,
    get_best_and_worst_retrievals,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments and optionally load config file."""
    parser = argparse.ArgumentParser(description="Train DeepCORO_CLIP model")

    # Config file argument
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Training parameters
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
        "--debug",
        action="store_true",
        help="Enable debug mode with model weight biases",
    )
    parser.add_argument(
        "--temp", type=float, default=0.1, help="Temperature for contrastive loss"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use Automatic Mixed Precision training",
    )

    # Data parameters
    parser.add_argument(
        "--data-filename",
        type=str,
        default="processed/reports/reports_sampled_1000.csv",
        help="Path to data CSV file",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/",
        help="Root directory for data",
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default="Report",
        help="Column name for target text",
    )
    parser.add_argument(
        "--datapoint-loc-label",
        type=str,
        default="FileName",
        help="Column name for file paths",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of frames to sample from each video",
    )

    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="mvit_v2_s",
        help="Name of video backbone model",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained backbone",
    )

    # Optimization parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer type",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="step",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr-step-period",
        type=int,
        default=15,
        help="Period for learning rate steps",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.3,
        help="Factor for learning rate scheduler",
    )

    # Logging parameters
    parser.add_argument(
        "--project",
        type=str,
        default="deepcoro_clip",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity name",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Additional tag for run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save model checkpoints and logs",
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values, but CLI args take precedence
        args_dict = vars(args)
        for key, value in config.items():
            if key in args_dict and args_dict[key] == parser.get_default(key):
                args_dict[key] = value

    # Also check environment variable as recommended by PyTorch
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    return args


def setup_training(args, rank=0):
    """Set up training environment and parameters.

    Args:
        args: Parsed command line arguments
        rank: Process rank for distributed training (default: 0)

    Returns:
        Dictionary containing training setup
    """
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate dataset statistics (only on rank 0)
    if rank == 0:
        print("\n=== Calculating Dataset Statistics ===")
        stats_dataset = StatsDataset(
            root=args.root,
            data_filename=args.data_filename,
            split="train",
            target_label=args.target_label,
            datapoint_loc_label=args.datapoint_loc_label,
            num_frames=args.frames,
            backbone=args.model_name,
        )

        # Create subset indices for statistics calculation
        num_stats_samples = min(100, len(stats_dataset))
        if len(stats_dataset) > num_stats_samples:
            # Use evenly spaced indices
            indices = torch.linspace(0, len(stats_dataset) - 1, num_stats_samples).long().tolist()
            stats_dataset = torch.utils.data.Subset(stats_dataset, indices)

        print(f"\nUsing {num_stats_samples} samples for statistics calculation")
        print(f"Frame count per video: {args.frames}")

        stats_loader = DataLoader(
            stats_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=stats_collate_fn,
        )

        # Calculate mean and std in a single pass
        mean_sum = 0.0
        squared_sum = 0.0
        pixel_count = 0

        for batch in tqdm(stats_loader, desc="Calculating statistics"):
            # batch shape: [B, F, H, W, C]
            batch = batch.float()  # Ensure float type
            b, f, h, w, c = batch.shape

            # Reshape to [B*F*H*W, C] for statistics
            batch = batch.reshape(-1, c)

            # Update sums
            mean_sum += batch.sum(dim=0)
            squared_sum += (batch**2).sum(dim=0)
            pixel_count += batch.shape[0]

        # Calculate final statistics
        mean = mean_sum / pixel_count
        std = torch.sqrt((squared_sum / pixel_count) - (mean**2))

        print("\nDataset Statistics:")
        print(f"Mean: {mean.tolist()}")
        print(f"Std:  {std.tolist()}")
        print(f"Calculated from {num_stats_samples} samples ({pixel_count:,} pixels)")
        print("===========================\n")
    else:
        mean = None
        std = None

    # Broadcast statistics to all processes if using distributed training
    if torch.distributed.is_initialized():
        if mean is not None:
            mean = mean.cuda()
            std = std.cuda()
        mean_tensor = torch.zeros(3, device="cuda")
        std_tensor = torch.zeros(3, device="cuda")
        if rank == 0:
            mean_tensor.copy_(mean)
            std_tensor.copy_(std)
        torch.distributed.broadcast(mean_tensor, 0)
        torch.distributed.broadcast(std_tensor, 0)
        mean = mean_tensor.cpu()
        std = std_tensor.cpu()

    # Create datasets with computed statistics
    train_dataset = VideoDataset(
        root=args.root,
        data_filename=args.data_filename,
        split="train",
        target_label=args.target_label,
        datapoint_loc_label=args.datapoint_loc_label,
        num_frames=args.frames,
        backbone=args.model_name,
        mean=(
            mean.tolist() if mean is not None else [0.485, 0.456, 0.406]
        ),  # ImageNet defaults if no stats
        std=std.tolist() if std is not None else [0.229, 0.224, 0.225],
    )

    val_dataset = VideoDataset(
        root=args.root,
        data_filename=args.data_filename,
        split="val",
        target_label=args.target_label,
        datapoint_loc_label=args.datapoint_loc_label,
        num_frames=args.frames,
        backbone=args.model_name,
        mean=mean.tolist() if mean is not None else [0.485, 0.456, 0.406],
        std=std.tolist() if std is not None else [0.229, 0.224, 0.225],
    )

    # Ensure validation set exists
    if len(val_dataset) == 0:
        raise ValueError(
            "No validation samples found! Please ensure your dataset has a validation split."
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    # Create models and move to device
    video_encoder = VideoEncoder(
        backbone=args.model_name,
        input_channels=3,
        num_frames=args.frames,
        pretrained=args.pretrained,
        output_dim=512,
    )

    text_encoder = TextEncoder()

    # Move models to device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_encoder = video_encoder.to(device)
    text_encoder = text_encoder.to(device)

    # Convert model parameters to float32 and ensure all parameters have the same dtype
    video_encoder = video_encoder.float()
    text_encoder = text_encoder.float()

    # Double check all parameters are float32
    for param in video_encoder.parameters():
        param.data = param.data.float()
    for param in text_encoder.parameters():
        param.data = param.data.float()

    # Create optimizer after model parameter conversion
    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_class(
        [
            {"params": video_encoder.parameters()},
            {"params": text_encoder.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler if specified
    scheduler = None
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_period,
            gamma=args.factor,
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    # Print dataset information
    if rank == 0:
        print("\n=== Dataset Information ===")
        print(f"Training:   {len(train_dataset):,} videos")
        print(f"Validation: {len(val_dataset):,} videos")
        print(f"Total:      {len(train_dataset) + len(val_dataset):,} videos")
        print(f"\nBatch Information:")
        print(f"Batch Size: {args.batch_size}")
        print(f"Training Batches: {len(train_dataset) // args.batch_size:,}")
        print(
            f"Validation Batches: {len(val_dataset) // args.batch_size + (1 if len(val_dataset) % args.batch_size else 0):,}"
        )
        print("===========================\n")

    return {
        "video_encoder": video_encoder,
        "text_encoder": text_encoder,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }


def setup_data(args):
    """Set up data loaders.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary containing data loaders
    """
    # Create dataset
    train_dataset = VideoDataset(
        root=args.root,
        data_filename=args.data_filename,
        split="train",
        target_label=args.target_label,
        datapoint_loc_label=args.datapoint_loc_label,
        num_frames=args.frames,
        backbone=args.model_name,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        normalize=True,
        debug=args.debug,
        batch_size=args.batch_size,
    )

    # Create sampler for distributed training
    train_sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None

    # Create data loader
    train_loader = DataLoader(
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

    return {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "train_sampler": train_sampler,
    }


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


def compute_recall_at_k(similarity_matrix, k_values=[1, 5]):
    """Compute Recall@K for both V2T and T2V."""
    batch_size = similarity_matrix.shape[0]
    metrics = {}

    # For each k
    for k in k_values:
        # Video to Text (V2T)
        v2t_topk = torch.topk(similarity_matrix, k, dim=1)[1]
        v2t_correct = v2t_topk == torch.arange(
            batch_size, device=similarity_matrix.device
        ).unsqueeze(1)
        v2t_recall = (v2t_correct.sum(dim=1) > 0).float().mean().item()
        metrics[f"Recall@{k}_V2T"] = v2t_recall

        # Text to Video (T2V)
        t2v_topk = torch.topk(similarity_matrix.t(), k, dim=1)[1]
        t2v_correct = t2v_topk == torch.arange(
            batch_size, device=similarity_matrix.device
        ).unsqueeze(1)
        t2v_recall = (t2v_correct.sum(dim=1) > 0).float().mean().item()
        metrics[f"Recall@{k}_T2V"] = t2v_recall

    return metrics


def compute_mrr(similarity_matrix):
    """Compute Mean Reciprocal Rank for both V2T and T2V."""
    batch_size = similarity_matrix.shape[0]
    device = similarity_matrix.device

    # Video to Text (V2T)
    v2t_ranks = (
        (
            similarity_matrix
            >= similarity_matrix.gather(1, torch.arange(batch_size, device=device).unsqueeze(1))
        )
        .sum(1)
        .float()
    )
    v2t_mrr = (1 / v2t_ranks).mean().item()

    # Text to Video (T2V)
    t2v_ranks = (
        (
            similarity_matrix.t()
            >= similarity_matrix.t().gather(
                1, torch.arange(batch_size, device=device).unsqueeze(1)
            )
        )
        .sum(1)
        .float()
    )
    t2v_mrr = (1 / t2v_ranks).mean().item()

    return {"MRR_V2T": v2t_mrr, "MRR_T2V": t2v_mrr}


def compute_embedding_norms(video_features, text_features):
    """Compute L2 norms of video and text embeddings."""
    video_norms = torch.norm(video_features, dim=1).mean().item()
    text_norms = torch.norm(text_features, dim=1).mean().item()
    return {"video_norm": video_norms, "text_norm": text_norms}


def compute_alignment_score(video_features, text_features):
    """Compute average cosine similarity of positive pairs."""
    normalized_video = nn.functional.normalize(video_features, dim=1)
    normalized_text = nn.functional.normalize(text_features, dim=1)
    alignment_scores = torch.sum(normalized_video * normalized_text, dim=1)
    return alignment_scores.mean().item()


def train_epoch(
    video_encoder,
    text_encoder,
    dataloader,
    optimizer,
    device,
    wandb_run,
    rank=0,
    world_size=1,
    epoch=0,
    scaler=None,
):
    """Training epoch with proper DDP synchronization"""
    video_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    num_batches = 0

    # Initialize metric accumulators
    epoch_metrics = {
        "Recall@1_V2T": 0.0,
        "Recall@5_V2T": 0.0,
        "Recall@1_T2V": 0.0,
        "Recall@5_T2V": 0.0,
        "MRR_V2T": 0.0,
        "MRR_T2V": 0.0,
        "video_norm": 0.0,
        "text_norm": 0.0,
        "alignment_score": 0.0,
    }

    # Use different progress bars for main process vs others
    if rank == 0:
        progress = tqdm(dataloader, desc="Training")
    else:
        progress = dataloader

    for batch_idx, batch in enumerate(progress):
        try:
            # Unpack batch
            videos, encoded_texts, paths = batch
            if videos is None or encoded_texts is None:
                print(f"Skipping invalid batch {batch_idx}")
                continue

            # Move data to device and ensure float32
            videos = videos.to(device, non_blocking=True).float()
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            # Forward passes with optional AMP
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    video_features = video_encoder(videos)
                    text_features = text_encoder(input_ids, attention_mask)
                    # Compute similarity matrix for metrics
                    normalized_video = nn.functional.normalize(video_features, dim=1)
                    normalized_text = nn.functional.normalize(text_features, dim=1)
                    similarity_matrix = torch.matmul(normalized_video, normalized_text.t())
                    # Compute loss
                    loss = contrastive_loss(video_features, text_features)

                # Backward pass with AMP
                scaler.scale(loss).backward()
                # Gradient clipping with AMP
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=5.0)
                # Step optimizer with AMP
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular forward pass without AMP
                video_features = video_encoder(videos)
                text_features = text_encoder(input_ids, attention_mask)
                # Compute similarity matrix for metrics
                normalized_video = nn.functional.normalize(video_features, dim=1)
                normalized_text = nn.functional.normalize(text_features, dim=1)
                similarity_matrix = torch.matmul(normalized_video, normalized_text.t())
                # Compute loss
                loss = contrastive_loss(video_features, text_features)
                # Regular backward pass
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=5.0)
                # Step optimizer
                optimizer.step()

            optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if rank == 0:
                progress.set_postfix(
                    {
                        "train_loss": f"{loss.item():.4f}",
                        "avg_train_loss": f"{(total_loss/num_batches):.4f}",
                    }
                )

            # Compute metrics only if we have enough items for k=5
            if videos.size(0) >= 5:
                recall_metrics = compute_recall_at_k(
                    similarity_matrix, k_values=[1, min(5, videos.size(0))]
                )
                mrr_metrics = compute_mrr(similarity_matrix)
                norm_metrics = compute_embedding_norms(video_features, text_features)
                alignment_score = compute_alignment_score(video_features, text_features)

                # Update epoch metric accumulators
                for metric_name, value in recall_metrics.items():
                    epoch_metrics[metric_name] += value
                for metric_name, value in mrr_metrics.items():
                    epoch_metrics[metric_name] += value
                epoch_metrics["video_norm"] += norm_metrics["video_norm"]
                epoch_metrics["text_norm"] += norm_metrics["text_norm"]
                epoch_metrics["alignment_score"] += alignment_score
            else:
                print(
                    f"Skipping metrics for batch {batch_idx} due to small batch size ({videos.size(0)})"
                )

        except Exception as e:
            print(f"Error in training batch {batch_idx} on rank {rank}: {str(e)}")
            print(f"Batch size: {videos.size(0) if 'videos' in locals() else 'unknown'}")
            continue

    # Average loss and metrics
    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Average epoch metrics
    metric_batches = num_batches  # Number of batches that contributed to metrics
    for metric_name in epoch_metrics:
        epoch_metrics[metric_name] /= metric_batches if metric_batches > 0 else 1

    if rank == 0:
        print(f"\nTraining Loss: {avg_loss:.4f}")

    return avg_loss, epoch_metrics


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


def create_logger(args):
    """Create logger with proper WandB configuration.

    Args:
        args: Parsed command line arguments with config values

    Returns:
        WandbLogger instance
    """
    # Create config dictionary from args
    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "gpu": args.gpu,
        "model_name": args.model_name,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler_type": args.scheduler_type,
        "lr_step_period": args.lr_step_period,
        "factor": args.factor,
        "frames": args.frames,
        "pretrained": args.pretrained,
    }

    # Add any additional args to config
    for key, value in vars(args).items():
        if key not in config:
            config[key] = value

    # Initialize wandb with proper project and entity
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.tag,
        config=config,
    )

    return wandb.run


def generate_output_dir_name(args, run_id):
    """
    Generates a directory name for output based on the provided configuration.

    Args:
        args: The arguments containing training parameters
        run_id (str): The ID of the current run

    Returns:
        str: The generated directory name for saving output
    """
    import time

    # Get current time to create a unique directory name
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Extract relevant information from args
    model_name = args.model_name.split("/")[-1]  # Get last part of model name
    batch_size = args.batch_size
    frames = args.frames
    optimizer = args.optimizer
    lr = args.lr
    tag = args.tag if args.tag else "default"
    project = args.project if args.project else "default_project"

    # Create model directory name starting with tag
    model_dir = (
        f"{tag}_{model_name}_b{batch_size}_f{frames}_{optimizer}_lr{lr}_{current_time}_{run_id}"
    )

    # Include project name in the path
    dir_name = os.path.join(project, model_dir)

    return dir_name


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


def main(rank=0, world_size=1, args=None):
    """Main training function with proper DDP setup"""
    if args is None:
        args = parse_args()

    wandb_run = None
    try:
        # Determine if we're doing distributed training
        is_distributed = world_size > 1 and args.gpu is None

        # Set up device and initialize process group first
        if is_distributed:
            device = setup_ddp(rank, world_size)
            if rank != 0:
                os.environ["WANDB_MODE"] = "disabled"
        else:
            if args.gpu is not None:
                device = torch.device(f"cuda:{args.gpu}")
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize wandb only on rank 0
        if rank == 0:
            wandb_run = wandb.init(
                project=args.project,
                entity=args.entity,
                name=args.tag,
                config=args,
            )
            # Generate output directory name using run ID
            output_dir = generate_output_dir_name(args, wandb_run.id)
            args.output_dir = os.path.join(args.output_dir, output_dir)
            print(f"\nOutput directory: {args.output_dir}")

        # Set up training components
        training_setup = setup_training(args, rank=rank)
        training_setup["device"] = device

        # Initialize gradient scaler for AMP if enabled
        scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
        if rank == 0 and args.use_amp:
            print("\nUsing Automatic Mixed Precision training")

        # Initialize best loss tracking
        best_loss = float("inf")
        best_val_loss = float("inf")

        # Training loop
        for epoch in range(args.epochs):
            # Training phase
            train_loss, train_metrics = train_epoch(
                video_encoder=training_setup["video_encoder"],
                text_encoder=training_setup["text_encoder"],
                dataloader=training_setup["train_loader"],
                optimizer=training_setup["optimizer"],
                device=device,
                wandb_run=wandb_run,
                rank=rank if is_distributed else 0,
                world_size=world_size if is_distributed else 1,
                epoch=epoch,
                scaler=scaler,
            )

            # Validation phase
            with torch.no_grad():
                val_loss, val_metrics, val_examples = validate_epoch(
                    video_encoder=training_setup["video_encoder"],
                    text_encoder=training_setup["text_encoder"],
                    dataloader=training_setup["val_loader"],
                    device=device,
                    wandb_run=wandb_run,
                    rank=rank,
                    world_size=world_size,
                    epoch=epoch,
                )

            # Update best losses
            best_loss = min(best_loss, train_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            # Log metrics and examples
            if rank == 0 and wandb_run is not None:
                # Log metrics
                wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/best_loss": best_loss,
                        "train/learning_rate": training_setup["optimizer"].param_groups[0]["lr"],
                        **{f"train/{k}": v for k, v in train_metrics.items()},
                        "val/loss": val_loss,
                        "val/best_loss": best_val_loss,
                        **{f"val/{k}": v for k, v in val_metrics.items()},
                    }
                )

                # Log qualitative examples from validation if loss improved
                if val_loss < best_val_loss:
                    wandb_run.log_validation_examples(
                        val_best_videos=val_examples["best_videos"],
                        val_best_reports=val_examples["best_reports"],
                        val_worst_videos=val_examples["worst_videos"],
                        val_worst_reports=val_examples["worst_reports"],
                    )

            # Step scheduler if it exists
            if training_setup["scheduler"] is not None:
                training_setup["scheduler"].step()

            # Save checkpoints only on main process
            if rank == 0:
                # Prepare model states
                model_dict = {
                    "video_encoder": (
                        training_setup["video_encoder"].module.state_dict()
                        if is_distributed
                        else training_setup["video_encoder"].state_dict()
                    ),
                    "text_encoder": (
                        training_setup["text_encoder"].module.state_dict()
                        if is_distributed
                        else training_setup["text_encoder"].state_dict()
                    ),
                    "optimizer": training_setup["optimizer"].state_dict(),
                }

                # Prepare metrics
                metrics_dict = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_loss": best_loss,
                    "best_val_loss": best_val_loss,
                }

                # Create checkpoints directory
                checkpoint_dir = Path(args.output_dir) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save latest checkpoint
                latest_path = checkpoint_dir / "latest.pt"
                save_checkpoint(model_dict, metrics_dict, latest_path)

                # Save best checkpoint if current validation loss is better
                if val_loss < best_val_loss:
                    best_path = checkpoint_dir / "best.pt"
                    save_checkpoint(model_dict, metrics_dict, best_path, is_best=True)
                    print(
                        f"\nNew best validation loss: {val_loss:.4f} (previous: {best_val_loss:.4f})"
                    )

                    if wandb_run is not None:
                        wandb.save(str(best_path))

                    best_val_loss = val_loss  # Update best_val_loss after saving

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e

    finally:
        if wandb_run is not None:
            wandb.finish()
        if is_distributed:
            cleanup_ddp()


def validate_epoch(
    video_encoder,
    text_encoder,
    dataloader,
    device,
    wandb_run,
    rank=0,
    world_size=1,
    epoch=0,
):
    """Validation epoch with metrics computation."""
    video_encoder.eval()
    text_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    # Initialize metric accumulators
    epoch_metrics = {
        "Recall@1_V2T": 0.0,
        "Recall@5_V2T": 0.0,
        "Recall@1_T2V": 0.0,
        "Recall@5_T2V": 0.0,
        "MRR_V2T": 0.0,
        "MRR_T2V": 0.0,
        "video_norm": 0.0,
        "text_norm": 0.0,
        "alignment_score": 0.0,
    }

    # Initialize best/worst tracking lists for validation
    val_best_videos = []
    val_best_reports = []
    val_best_scores = []
    val_worst_videos = []
    val_worst_reports = []
    val_worst_scores = []

    # Use different progress bars for main process vs others
    if rank == 0:
        progress = tqdm(dataloader, desc="Validation")
    else:
        progress = dataloader

    for batch_idx, batch in enumerate(progress):
        try:
            # Unpack batch
            videos, encoded_texts, paths = batch
            reports = dataloader.dataset.get_reports(paths)
            if videos is None or encoded_texts is None:
                print(f"Skipping invalid batch {batch_idx}")
                continue

            # Get batch size for this batch
            batch_size = videos.size(0)
            if batch_size < 2:  # Need at least 2 samples for contrastive loss
                print(f"Skipping batch {batch_idx} - too few samples ({batch_size})")
                continue

            # Move data to device and ensure float32
            videos = videos.to(device, non_blocking=True).float()
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            # Forward pass
            video_features = video_encoder(videos)
            text_features = text_encoder(input_ids, attention_mask)

            # Compute similarity matrix for metrics
            normalized_video = nn.functional.normalize(video_features, dim=1)
            normalized_text = nn.functional.normalize(text_features, dim=1)
            similarity_matrix = torch.matmul(normalized_video, normalized_text.t())

            # Compute loss
            loss = contrastive_loss(video_features, text_features)

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if rank == 0:
                progress.set_postfix(
                    {
                        "val_loss": f"{loss.item():.4f}",
                        "avg_val_loss": f"{(total_loss/num_batches):.4f}",
                    }
                )

            # Store best and worst examples if this batch has better examples
            (
                best_indices,
                worst_indices,
                best_scores,
                worst_scores,
                best_text_indices,
                worst_text_indices,
            ) = get_best_and_worst_retrievals(similarity_matrix, paths, reports, k=2)

            # Update best/worst lists if we find better examples
            for i, (idx, score) in enumerate(zip(best_indices, best_scores)):
                if len(val_best_scores) < 2 or score > min(val_best_scores):
                    video_similarities = similarity_matrix[idx]
                    top_5_text_indices = torch.argsort(video_similarities, descending=True)[:5]
                    predicted_reports = [
                        reports[text_idx.item()] for text_idx in top_5_text_indices
                    ]

                    val_best_videos.append(str(paths[idx]))
                    val_best_reports.append(
                        {
                            "ground_truth": reports[idx],
                            "predicted": predicted_reports,
                            "similarity_score": score.item(),
                        }
                    )
                    val_best_scores.append(score.item())

                    # Keep only top 2
                    if len(val_best_scores) > 2:
                        min_idx = val_best_scores.index(min(val_best_scores))
                        val_best_videos.pop(min_idx)
                        val_best_reports.pop(min_idx)
                        val_best_scores.pop(min_idx)

            for i, (idx, score) in enumerate(zip(worst_indices, worst_scores)):
                if len(val_worst_scores) < 2 or score < max(val_worst_scores):
                    video_similarities = similarity_matrix[idx]
                    top_5_text_indices = torch.argsort(video_similarities, descending=True)[:5]
                    predicted_reports = [
                        reports[text_idx.item()] for text_idx in top_5_text_indices
                    ]

                    val_worst_videos.append(str(paths[idx]))
                    val_worst_reports.append(
                        {
                            "ground_truth": reports[idx],
                            "predicted": predicted_reports,
                            "similarity_score": score.item(),
                        }
                    )
                    val_worst_scores.append(score.item())

                    # Keep only bottom 2
                    if len(val_worst_scores) > 2:
                        max_idx = val_worst_scores.index(max(val_worst_scores))
                        val_worst_videos.pop(max_idx)
                        val_worst_reports.pop(max_idx)
                        val_worst_scores.pop(max_idx)

            # Compute metrics only if we have enough items for k=5
            if batch_size >= 5:
                recall_metrics = compute_recall_at_k(
                    similarity_matrix, k_values=[1, min(5, batch_size)]
                )
                mrr_metrics = compute_mrr(similarity_matrix)
                norm_metrics = compute_embedding_norms(video_features, text_features)
                alignment_score = compute_alignment_score(video_features, text_features)

                # Update epoch metric accumulators
                for metric_name, value in recall_metrics.items():
                    epoch_metrics[metric_name] += value
                for metric_name, value in mrr_metrics.items():
                    epoch_metrics[metric_name] += value
                epoch_metrics["video_norm"] += norm_metrics["video_norm"]
                epoch_metrics["text_norm"] += norm_metrics["text_norm"]
                epoch_metrics["alignment_score"] += alignment_score
            else:
                print(
                    f"Skipping metrics for batch {batch_idx} due to small batch size ({batch_size})"
                )

        except Exception as e:
            print(f"Error in validation batch {batch_idx} on rank {rank}: {str(e)}")
            print(f"Batch size: {batch_size if 'batch_size' in locals() else 'unknown'}")
            continue

    # Average loss and metrics
    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Average epoch metrics
    metric_batches = num_batches  # Number of batches that contributed to metrics
    for metric_name in epoch_metrics:
        epoch_metrics[metric_name] /= metric_batches if metric_batches > 0 else 1

    if rank == 0:
        print(f"\nValidation Loss: {avg_loss:.4f}")

        # Log best and worst examples
        if wandb_run is not None:
            wandb_run.log_validation_examples(
                val_best_videos, val_best_reports, val_worst_videos, val_worst_reports
            )

    # Return validation loss, metrics, and best/worst examples
    return (
        avg_loss,
        epoch_metrics,
        {
            "best_videos": val_best_videos,
            "best_reports": val_best_reports,
            "best_scores": val_best_scores,
            "worst_videos": val_worst_videos,
            "worst_reports": val_worst_reports,
            "worst_scores": val_worst_scores,
        },
    )


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
