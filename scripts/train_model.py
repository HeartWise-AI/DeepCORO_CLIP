"""Training script for DeepCORO_CLIP model."""

import argparse
import os
import pickle
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.model import TextEncoder, VideoEncoder, clip_style_loss
from utils.data_processing.video import (
    StatsDataset,
    VideoDataset,
    custom_collate_fn,
    stats_collate_fn,
)
from utils.logging import (
    cleanup_temp_video,
    compute_map,
    compute_median_rank,
    compute_ndcg,
    convert_video_for_wandb,
    create_logger,
    get_best_and_worst_retrievals,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments and optionally load config file."""
    parser = argparse.ArgumentParser(description="Train DeepCORO_CLIP model")

    # Config file argument
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Training parameters
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="Local rank")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for contrastive loss"
    )

    # Data parameters
    parser.add_argument(
        "--data_filename",
        type=str,
        default="processed/reports/reports_sampled_1000.csv",
        help="Data CSV",
    )
    parser.add_argument("--root", type=str, default="data/", help="Root directory")
    parser.add_argument("--target_label", type=str, default="Report", help="Target text column")
    parser.add_argument("--datapoint_loc_label", type=str, default="FileName", help="Path column")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--stride", type=int, default=2, help="Frame sampling stride")
    parser.add_argument(
        "--random_augment", type=bool, default=False, help="Use random augmentation"
    )
    # Model parameters
    parser.add_argument(
        "--model_name", type=str, default="mvit_v2_s", help="Video backbone model name"
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone")

    # Optimization parameters
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--scheduler_type", type=str, default="step", help="LR scheduler type")
    parser.add_argument("--lr_step_period", type=int, default=15, help="LR step period")
    parser.add_argument("--factor", type=float, default=0.3, help="Factor for scheduler")
    parser.add_argument("--use_amp", action="store_true", help="Use AMP training")

    # Logging parameters
    parser.add_argument("--project", type=str, default="deepcoro_clip", help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--tag", type=str, default=None, help="Additional tag")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args_dict = vars(args)
        for key, value in config.items():
            if key in args_dict and args_dict[key] == parser.get_default(key):
                args_dict[key] = value
        # Explicitly cast known numeric parameters to float
        args.learning_rate = float(args.learning_rate)
        args.weight_decay = float(args.weight_decay)
        args.factor = float(args.factor)

    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    return args


def generate_output_dir_name(args, run_id):
    """
    Generates a directory name for output based on the provided configuration.
    """
    import time

    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.model_name.split("/")[-1]
    batch_size = args.batch_size
    frames = args.frames
    optimizer = args.optimizer
    lr = args.learning_rate
    tag = args.tag if args.tag else "default"
    project = args.project if args.project else "default_project"

    model_dir = (
        f"{tag}_{model_name}_b{batch_size}_f{frames}_{optimizer}_lr{lr}_{current_time}_{run_id}"
    )

    dir_name = os.path.join(project, model_dir)
    return dir_name


def setup_training(args, rank=0):
    """Set up training environment and parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb if on rank 0

    wandb_run = None
    if rank == 0:
        wandb_run = create_logger(args)

        # After wandb.init(), wandb.config is available.
        # Override args with wandb.config parameters if present.
        if wandb_run is not None and len(wandb.config.keys()) > 0:
            for key, value in wandb.config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: {key} in wandb.config not recognized as an arg.")

    print("Args: ", args)

    # Calculate dataset statistics (only on rank 0)
    mean, std = None, None
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
            stride=args.stride,
        )

        num_stats_samples = min(100, 1000)
        print(f"Stats dataset length: {len(stats_dataset)}")
        if len(stats_dataset) > num_stats_samples:
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

        mean_sum, squared_sum, pixel_count = 0.0, 0.0, 0
        for batch in tqdm(stats_loader, desc="Calculating statistics"):
            batch = batch.float()
            b, f, h, w, c = batch.shape
            batch = batch.reshape(-1, c)
            mean_sum += batch.sum(dim=0)
            squared_sum += (batch**2).sum(dim=0)
            pixel_count += batch.shape[0]

        mean = mean_sum / pixel_count
        std = torch.sqrt((squared_sum / pixel_count) - (mean**2))

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
        if rank == 0:
            mean_tensor.copy_(mean)
            std_tensor.copy_(std)
        torch.distributed.broadcast(mean_tensor, 0)
        torch.distributed.broadcast(std_tensor, 0)
        mean = mean_tensor.cpu()
        std = std_tensor.cpu()

    train_dataset = VideoDataset(
        root=args.root,
        data_filename=args.data_filename,
        split="train",
        target_label=args.target_label,
        datapoint_loc_label=args.datapoint_loc_label,
        num_frames=args.frames,
        backbone=args.model_name,
        mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
        std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
        rand_augment=args.random_augment,
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

    if len(val_dataset) == 0:
        raise ValueError("No validation samples found! Check your dataset split.")

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
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    video_encoder = (
        VideoEncoder(
            backbone=args.model_name,
            input_channels=3,
            num_frames=args.frames,
            pretrained=args.pretrained,
            output_dim=512,
        )
        .to(device)
        .float()
    )

    text_encoder = TextEncoder().to(device).float()

    for param in video_encoder.parameters():
        param.data = param.data.float()
    for param in text_encoder.parameters():
        param.data = param.data.float()

    # Make temperature a trainable parameter directly on the device
    log_temp = nn.Parameter(
        torch.log(torch.tensor([args.temperature], dtype=torch.float32, device=device))
    )

    # Include the temperature parameter in the optimizer
    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_class(
        [
            {"params": video_encoder.parameters(), "lr": 1e-3},  # Lower LR for video model
            {"params": text_encoder.parameters(), "lr": 1e-3},  # Lower LR for PubMedBERT
            {"params": [log_temp]},  # Add the temperature parameter
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

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

    # Log dataset sizes to W&B configuration if rank=0 and run is initialized
    if rank == 0 and wandb.run is not None:
        wandb.config.update(
            {
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
            },
            allow_val_change=True,
        )

        print("\n=== Dataset Information ===")
        print(f"Training:   {len(train_dataset):,} videos")
        print(f"Validation: {len(val_dataset):,} videos")
        print(f"Total:      {len(train_dataset) + len(val_dataset):,} videos")
        print(f"\nBatch Size: {args.batch_size}")
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
        "device": device,
        "wandb_run": wandb_run,
        "log_temp": log_temp,
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


def compute_recall_at_k(similarity_matrix, global_gt_indices, k_values=[1, 5]):
    batch_size = similarity_matrix.shape[0]
    metrics = {}

    for k in k_values:
        v2t_topk = torch.topk(similarity_matrix, k, dim=1)[1]
        v2t_correct = v2t_topk == global_gt_indices.unsqueeze(1)
        v2t_recall = (v2t_correct.sum(dim=1) > 0).float().mean().item()
        metrics[f"Recall@{k}_V2T"] = v2t_recall
        ## A report text can haev multiple video ground truths so cant calculate t2v
        # t2v_topk = torch.topk(similarity_matrix.t(), k, dim=1)[1]
        # t2v_correct = t2v_topk == global_gt_indices.unsqueeze(1)
        # t2v_recall = (t2v_correct.sum(dim=1) > 0).float().mean().item()
        # metrics[f"Recall@{k}_T2V"] = t2v_recall

    return metrics


def compute_mrr(similarity_matrix, global_gt_indices):
    batch_size = similarity_matrix.shape[0]
    device = similarity_matrix.device

    # Video to Text
    target_scores = similarity_matrix.gather(1, global_gt_indices.unsqueeze(1))
    v2t_ranks = (similarity_matrix >= target_scores).sum(1).float()
    v2t_mrr = (1 / v2t_ranks).mean().item()

    return {"MRR_V2T": v2t_mrr}


def compute_embedding_norms(video_features, text_features):
    """Compute L2 norms of video and text embeddings."""
    video_norms = torch.norm(video_features, dim=1).mean().item()
    text_norms = torch.norm(text_features, dim=1).mean().item()
    return {"video_norm": video_norms, "text_norm": text_norms}


def compute_alignment_score(
    video_features,
    text_features,
    all_video_embeddings=None,
    all_text_embeddings=None,
    global_ground_truth_indices_tensor=None,
):
    """
    Compute average cosine similarity of positive pairs.

    Parameters:
    - video_features: torch.Tensor (batch local video embeddings)
    - text_features: torch.Tensor (batch local text embeddings)
    - all_video_embeddings: torch.Tensor of all validation video embeddings [N_videos, dim] (optional)
    - all_text_embeddings: torch.Tensor of all global text embeddings [N_texts, dim] (optional)
    - global_ground_truth_indices_tensor: torch.Tensor of global GT indices for each video (optional)

    If all_video_embeddings, all_text_embeddings, and global_ground_truth_indices_tensor
    are provided, compute global alignment using global embeddings.

    Otherwise, compute local alignment score assuming a one-to-one mapping between
    video_features[i] and text_features[i].
    """
    if (
        all_video_embeddings is not None
        and all_text_embeddings is not None
        and global_ground_truth_indices_tensor is not None
    ):
        # Global alignment scenario (for validation)
        correct_text_embeddings = all_text_embeddings[global_ground_truth_indices_tensor]
        normalized_video = nn.functional.normalize(all_video_embeddings, dim=1)
        normalized_text = nn.functional.normalize(correct_text_embeddings, dim=1)
        alignment_scores = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()
    else:
        # Local alignment scenario (for training)
        normalized_video = nn.functional.normalize(video_features, dim=1)
        normalized_text = nn.functional.normalize(text_features, dim=1)
        alignment_scores = (normalized_video * normalized_text).sum(dim=1)
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
    log_temp=None,
):
    """Training epoch with local metrics only."""
    video_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    num_batches = 0

    # Initialize metric accumulators
    epoch_metrics = {
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
                # Skip invalid batch
                continue

            # Move data to device and ensure float32
            videos = videos.to(device, non_blocking=True).float()
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward passes with optional AMP
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    # Inside the training loop where loss is computed:
                    video_features = video_encoder(videos)
                    text_features = text_encoder(input_ids, attention_mask)

                    loss = clip_style_loss(video_features, text_features, log_temp)

                # Backward pass with AMP
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                video_features = video_encoder(videos)
                text_features = text_encoder(input_ids, attention_mask)
                loss = clip_style_loss(video_features, text_features, log_temp)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=5.0)
                # torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=5.0)
                optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            if rank == 0:
                progress.set_postfix(
                    {
                        "train_loss": f"{loss.item():.4f}",
                        "avg_train_loss": f"{(total_loss / num_batches):.4f}",
                    }
                )

            # Compute local metrics (no global embeddings or global indices)
            if videos.size(0) >= 5:
                norm_metrics = compute_embedding_norms(video_features, text_features)
                alignment_score = compute_alignment_score(video_features, text_features)

                epoch_metrics["video_norm"] += norm_metrics["video_norm"]
                epoch_metrics["text_norm"] += norm_metrics["text_norm"]
                epoch_metrics["alignment_score"] += alignment_score

            # Clear tensors
            del (
                videos,
                input_ids,
                attention_mask,
                video_features,
                text_features,
                loss,
            )
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in training batch {batch_idx} on rank {rank}: {str(e)}")
            continue

    # Average loss and metrics
    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Average epoch metrics
    metric_batches = num_batches if num_batches > 0 else 1
    for metric_name in epoch_metrics:
        epoch_metrics[metric_name] /= metric_batches

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
        "learning_rate": args.learning_rate,
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
    print(config)
    # Initialize wandb with proper project and entity
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.tag,
        config=config,
    )

    return wandb.run


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


def create_global_text_pool(train_dataset, val_dataset, test_dataset=None):
    """
    Create a global pool of text reports from train, val, and optionally test sets.
    Ensures each report is included and assign them global indices.
    If duplicates need to be removed, we can use a set or dict.

    Args:
        train_dataset: Dataset instance for training set
        val_dataset: Dataset instance for validation set
        test_dataset: Dataset instance for test set (optional)

    Returns:
        all_global_reports: A list of all reports from train, val (and test if given)
    """
    train_reports = train_dataset.get_all_reports()
    val_reports = val_dataset.get_all_reports()
    test_reports = test_dataset.get_all_reports() if test_dataset is not None else []

    # If you want uniqueness:
    # unique_reports = list(set(train_reports + val_reports + test_reports))
    # But for consistent indexing, might need order:
    # Use a dict to preserve order:
    seen = {}
    for r in train_reports:
        if r not in seen:
            seen[r] = len(seen)
    for r in val_reports:
        if r not in seen:
            seen[r] = len(seen)
    for r in test_reports:
        if r not in seen:
            seen[r] = len(seen)

    # Convert keys to a list ordered by insertion
    all_global_reports = [None] * len(seen)
    for report, idx in seen.items():
        all_global_reports[idx] = report

    return all_global_reports


def precompute_global_text_embeddings(
    text_encoder, all_global_reports, tokenizer, device, batch_size=64, num_workers=4
):
    """
    Precompute embeddings for a global set of reports.

    Args:
        text_encoder: The text encoder model
        all_global_reports: A list of all global reports
        tokenizer: The tokenizer associated with the text encoder
        device: Torch device
        batch_size: Batch size for encoding
        num_workers: Number of workers for DataLoader

    Returns:
        all_global_reports: same list as input
        all_global_text_embeddings: normalized embeddings for all reports
    """
    from torch.utils.data import DataLoader, Dataset

    class GlobalTextDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.squeeze(0) for k, v in encoded.items()}
            return encoded

    text_dataset = GlobalTextDataset(all_global_reports, tokenizer)
    text_loader = DataLoader(
        text_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    all_text_embeddings = []
    text_encoder.eval()

    with torch.no_grad():
        for batch_texts in text_loader:
            input_ids = batch_texts["input_ids"].to(device)
            attention_mask = batch_texts["attention_mask"].to(device)
            text_features = text_encoder(input_ids, attention_mask)
            text_features = nn.functional.normalize(text_features, dim=1)
            all_text_embeddings.append(text_features.cpu())

    all_global_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    return all_global_reports, all_global_text_embeddings


def get_best_and_worst_retrievals(similarity_matrix, paths, reports, k=2):
    """Get the best and worst retrievals based on similarity scores, along with their top text matches.

    Args:
        similarity_matrix: Tensor of shape (num_videos, num_queries)
        paths: List of video paths
        reports: List of report texts
        k: Number of best/worst examples to return

    Returns:
        tuple: (best_indices, worst_indices, best_scores, worst_scores, best_text_indices, worst_text_indices)
    """
    # Get mean similarity score for each video-query pair
    mean_similarities = similarity_matrix.mean(dim=1)

    # Adjust k to not exceed batch size
    k = min(k, len(mean_similarities))

    # Get indices of best and worst k videos
    best_values, best_indices = torch.topk(mean_similarities, k=k)
    worst_values, worst_indices = torch.topk(mean_similarities, k=k, largest=False)

    # Get top-5 text matches for each video
    best_text_indices = []
    worst_text_indices = []

    for idx in best_indices:
        # Get top N text matches for this video, where N is min(5, batch_size)
        n_texts = min(5, similarity_matrix.size(1))
        _, top_n_texts = torch.topk(similarity_matrix[idx], k=n_texts)
        best_text_indices.append(top_n_texts)

    for idx in worst_indices:
        # Get top N text matches for this video, where N is min(5, batch_size)
        n_texts = min(5, similarity_matrix.size(1))
        _, top_n_texts = torch.topk(similarity_matrix[idx], k=n_texts)
        worst_text_indices.append(top_n_texts)

    return (
        best_indices,
        worst_indices,
        best_values,
        worst_values,
        best_text_indices,
        worst_text_indices,
    )


def validate_epoch(
    video_encoder,
    text_encoder,
    dataloader,
    device,
    wandb_run,
    rank=0,
    world_size=1,
    epoch=0,
    all_text_embeddings=None,
    all_reports=None,
    text_embedding_pickle_path="text_embeddings.pkl",
    output_dir="outputs",
    report_to_global_index=None,
    use_val_only_pool=True,
    log_temp=None,
):
    """
    Validation epoch with retrieval computation and logging.

    Args:
        video_encoder, text_encoder: Models
        dataloader: Validation DataLoader
        device: torch device
        wandb_run: W&B logger
        rank: DDP rank
        world_size: DDP world size
        epoch: current epoch
        all_text_embeddings: precomputed text embeddings for the pool (val-only or global)
        all_reports: corresponding reports (val-only or global)
        report_to_global_index: mapping from report to index in `all_reports`
        use_val_only_pool (bool): If True, evaluate retrieval using only the val set embeddings,
                                  and log best/worst examples. If False, only compute metrics.

    Returns:
        avg_loss, epoch_metrics, examples_dict
    """
    video_encoder.eval()
    text_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    epoch_metrics = {
        "Recall@1_V2T": 0.0,
        "Recall@5_V2T": 0.0,
        "Recall@10_V2T": 0.0,
        "Recall@50_V2T": 0.0,
        "MRR_V2T": 0.0,
        "NDCG@5_V2T": 0.0,
        "MedianRank_V2T": 0.0,
        "video_norm": 0.0,
        "text_norm": 0.0,
        "alignment_score": 0.0,
        "MAP": 0.0,
    }

    all_video_embeddings = []
    all_paths = []
    all_ground_truth_reports = []

    progress = tqdm(dataloader, desc="Validation") if rank == 0 else dataloader

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            videos, encoded_texts, paths = batch
            batch_reports = dataloader.dataset.get_reports(paths)

            if videos is None or encoded_texts is None:
                continue

            batch_size = videos.size(0)
            if batch_size < 2:
                continue

            videos = videos.to(device, non_blocking=True).float()
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            video_features = video_encoder(videos)
            text_features = text_encoder(input_ids, attention_mask)
            # Compute validation loss on this batch
            loss = clip_style_loss(video_features, text_features, log_temp)
            total_loss += loss.item()
            num_batches += 1

            # Store raw video embeddings (not normalized yet)
            all_video_embeddings.append(video_features.cpu())

            all_paths.extend(paths)
            all_ground_truth_reports.extend(batch_reports)

            # Clean up
            del videos, input_ids, attention_mask, video_features, text_features, loss
            torch.cuda.empty_cache()

    # Reduce loss across processes if distributed
    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # If no batches or no embeddings, return early
    if num_batches == 0 or len(all_video_embeddings) == 0:
        if rank == 0:
            print("\nNo validation batches processed or no valid data.")
        return (
            avg_loss,
            epoch_metrics,
            {
                "best_videos": [],
                "best_reports": [],
                "best_scores": [],
                "worst_videos": [],
                "worst_reports": [],
                "worst_scores": [],
            },
        )

    # Concatenate all video embeddings
    all_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(device)
    all_text_embeddings = all_text_embeddings.to(device)

    # Compute norms before normalization
    max_len = min(all_video_embeddings.size(0), all_text_embeddings.size(0))
    truncated_video = all_video_embeddings[:max_len]
    truncated_text = all_text_embeddings[:max_len]

    # Compute norms from raw embeddings
    norm_metrics = compute_embedding_norms(truncated_video, truncated_text)
    epoch_metrics["video_norm"] = norm_metrics["video_norm"]
    epoch_metrics["text_norm"] = norm_metrics["text_norm"]

    # Now normalize embeddings for similarity calculation
    all_video_embeddings = nn.functional.normalize(all_video_embeddings, dim=1)
    all_text_embeddings = nn.functional.normalize(all_text_embeddings, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_video_embeddings, all_text_embeddings.T)

    effective_k = min(5, similarity_matrix.size(0), similarity_matrix.size(1))
    if similarity_matrix.size(0) >= 5 and similarity_matrix.size(1) >= 5:
        global_ground_truth_indices = [
            report_to_global_index[gt_report] for gt_report in all_ground_truth_reports
        ]
        global_ground_truth_indices_tensor = torch.tensor(
            global_ground_truth_indices, device=device
        )

        recall_metrics = compute_recall_at_k(
            similarity_matrix, global_ground_truth_indices_tensor, k_values=[1, 5, 10, 50]
        )
        mrr_metrics = compute_mrr(similarity_matrix, global_ground_truth_indices_tensor)
        epoch_metrics["NDCG@5_V2T"] = compute_ndcg(
            similarity_matrix, global_ground_truth_indices_tensor, k=5
        )
        epoch_metrics["MedianRank_V2T"] = compute_median_rank(
            similarity_matrix, global_ground_truth_indices_tensor
        )
        epoch_metrics["MAP"] = compute_map(similarity_matrix, global_ground_truth_indices_tensor)

        alignment_score = compute_alignment_score(
            all_video_embeddings[:max_len],
            all_text_embeddings[:max_len],
        )

        for metric_name, value in recall_metrics.items():
            epoch_metrics[metric_name] = value
        for metric_name, value in mrr_metrics.items():
            epoch_metrics[metric_name] = value
        epoch_metrics["alignment_score"] = alignment_score

    # Only log best/worst retrieval if we are using val-only pool
    val_best_videos = []
    val_best_reports = []
    val_worst_videos = []
    val_worst_reports = []

    if use_val_only_pool:
        # For logging best and worst retrieval examples
        max_scores, _ = similarity_matrix.max(dim=1)
        k = 1  # Only pick top 1 best and worst
        best_scores, best_indices = torch.topk(max_scores, k=k)
        worst_scores, worst_indices = torch.topk(max_scores, k=k, largest=False)

        # Prepare CSV with ground truth and top 5 predictions for each sample
        val_csv_path = os.path.join(output_dir, f"val_epoch{epoch}.csv")
        import csv

        with open(val_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "FileName",
                "ground_truth_idx",
                "predicted_idx_1",
                "sim_1",
                "predicted_idx_2",
                "sim_2",
                "predicted_idx_3",
                "sim_3",
                "predicted_idx_4",
                "sim_4",
                "predicted_idx_5",
                "sim_5",
            ]
            writer.writerow(header)

            # For each video, get top-5 predictions and their similarity scores
            for i, path in enumerate(all_paths):
                top_5_text_indices = torch.argsort(similarity_matrix[i], descending=True)[:5]
                predicted_indices = [idx.item() for idx in top_5_text_indices]
                predicted_sims = [similarity_matrix[i, idx].item() for idx in top_5_text_indices]

                row_data = [path, i]
                for p_idx, p_sim in zip(predicted_indices, predicted_sims):
                    row_data.append(p_idx)
                    row_data.append(f"{p_sim:.4f}")

                writer.writerow(row_data)

        if best_indices.numel() > 0:
            idx = best_indices[0].item()
            score = best_scores[0].item()
            top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
            predicted_reports = [all_reports[j.item()] for j in top_5_text_indices]
            val_best_videos.append(str(all_paths[idx]))
            val_best_reports.append(
                {
                    "ground_truth": all_ground_truth_reports[idx],
                    "predicted": predicted_reports,
                    "similarity_score": score,
                }
            )

        if worst_indices.numel() > 0:
            idx = worst_indices[0].item()
            score = worst_scores[0].item()
            top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
            predicted_reports = [all_reports[j.item()] for j in top_5_text_indices]
            val_worst_videos.append(str(all_paths[idx]))
            val_worst_reports.append(
                {
                    "ground_truth": all_ground_truth_reports[idx],
                    "predicted": predicted_reports,
                    "similarity_score": score,
                }
            )

        # Log the best example video and top-5 texts
        if val_best_videos and val_best_reports and wandb_run is not None:
            # Convert video and get path + cleanup flag
            mp4_path, is_temp = convert_video_for_wandb(val_best_videos[0])

            wandb_run.log(
                {
                    "qualitative/good_retrieval": wandb.Video(
                        mp4_path,
                        caption=f"Sim: {val_best_reports[0]['similarity_score']:.3f}",
                        format="mp4",
                    ),
                    "epoch": epoch,
                }
            )

            predicted_html = "<br>".join(
                [f"{i+1}. {text}" for i, text in enumerate(val_best_reports[0]["predicted"])]
            )
            ground_truth_html = (
                f"<b>Ground Truth:</b> {val_best_reports[0]['ground_truth']}<br>"
                f"<b>Top 5 Predicted:</b><br>{predicted_html}"
            )
            wandb_run.log(
                {
                    f"qualitative/good_retrieval_text": wandb.Html(ground_truth_html),
                    "epoch": epoch,
                }
            )

            # Clean up temp file if needed
            if is_temp:
                cleanup_temp_video(mp4_path)

        # Similarly for worst retrieval
        if val_worst_videos and val_worst_reports and wandb_run is not None:
            # Convert video and get path + cleanup flag
            mp4_path, is_temp = convert_video_for_wandb(val_worst_videos[0])

            wandb_run.log(
                {
                    "qualitative/bad_retrieval": wandb.Video(
                        mp4_path,
                        caption=f"Sim: {val_worst_reports[0]['similarity_score']:.3f}",
                        format="mp4",
                    ),
                    "epoch": epoch,
                }
            )

            predicted_html = "<br>".join(
                [f"{i+1}. {text}" for i, text in enumerate(val_worst_reports[0]["predicted"])]
            )
            ground_truth_html = (
                f"<b>Ground Truth:</b> {val_worst_reports[0]['ground_truth']}<br>"
                f"<b>Top 5 Predicted:</b><br>{predicted_html}"
            )
            wandb_run.log(
                {f"qualitative/bad_retrieval_text": wandb.Html(ground_truth_html), "epoch": epoch}
            )

            # Clean up temp file if needed
            if is_temp:
                cleanup_temp_video(mp4_path)

    avg_text_embedding = all_text_embeddings.mean(dim=0)
    prefix = "val_only" if use_val_only_pool else "global_val"

    if rank == 0:
        print(f"\nAverage text embedding (first 5 dims): {avg_text_embedding[:5]}")
        print(f"Validation Loss: {avg_loss:.4f}")

        if wandb_run is not None:
            wandb_run.log({f"{prefix}/avg_loss": avg_loss, "epoch": epoch})
            for metric_name, val in epoch_metrics.items():
                wandb_run.log({f"{prefix}/{metric_name}": val, "epoch": epoch})
    return (
        avg_loss,
        epoch_metrics,
        {
            "best_videos": val_best_videos,
            "best_reports": val_best_reports,
            "best_scores": [br["similarity_score"] for br in val_best_reports],
            "worst_videos": val_worst_videos,
            "worst_reports": val_worst_reports,
            "worst_scores": [wr["similarity_score"] for wr in val_worst_reports],
        },
    )


def main(rank=0, world_size=1, args=None):
    training_setup = None
    try:
        wandb_run = None
        if rank == 0:
            wandb_run = create_logger(args)

        run_id = wandb_run.id if wandb_run is not None else "run_id_001"
        output_subdir = generate_output_dir_name(args, run_id)
        full_output_path = os.path.join(args.output_dir, output_subdir)
        os.makedirs(full_output_path, exist_ok=True)
        print("Args: ", args)
        training_setup = setup_training(args, rank=rank)
        if wandb_run is None:
            wandb_run = training_setup["wandb_run"]

        is_distributed = world_size > 1
        text_encoder = training_setup["text_encoder"]
        train_dataset = training_setup["train_dataset"]
        val_dataset = training_setup["val_dataset"]
        device = training_setup["device"]
        log_temp = training_setup["log_temp"] if "log_temp" in training_setup else None

        best_val_loss = float("inf")
        best_epoch = -1

        # === Create Global Pool (train + val) ===
        all_global_reports = create_global_text_pool(train_dataset, val_dataset, None)
        all_global_reports, all_global_text_embeddings = precompute_global_text_embeddings(
            text_encoder, all_global_reports, train_dataset.tokenizer, device
        )
        report_to_global_index = {r: i for i, r in enumerate(all_global_reports)}

        with open(os.path.join(full_output_path, "global_text_embeddings.pkl"), "wb") as f:
            pickle.dump(
                (all_global_reports, all_global_text_embeddings, report_to_global_index), f
            )

        # === Create Validation-Only Pool ===
        # Extract reports only from val_dataset.
        val_reports = val_dataset.get_all_reports()
        val_unique_reports = list(dict.fromkeys(val_reports))  # Preserve order, remove duplicates
        val_report_to_index = {r: i for i, r in enumerate(val_unique_reports)}

        # Precompute validation-only text embeddings
        val_reports, val_text_embeddings = precompute_global_text_embeddings(
            text_encoder, val_unique_reports, train_dataset.tokenizer, device
        )

        # Create the GradScaler if AMP is enabled
        scaler = torch.amp.GradScaler(enabled=args.use_amp, device=device)

        # Main training loop
        for epoch in range(args.epochs):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_loss, train_metrics = train_epoch(
                video_encoder=training_setup["video_encoder"],
                text_encoder=text_encoder,
                dataloader=training_setup["train_loader"],
                optimizer=training_setup["optimizer"],
                device=device,
                wandb_run=wandb_run,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                scaler=scaler,
                log_temp=log_temp,
            )

            # Recompute text embeddings here, using the current (just-trained) state of `text_encoder` otherwise validation metirvcs stay static
            val_reports, val_text_embeddings = precompute_global_text_embeddings(
                text_encoder, val_unique_reports, train_dataset.tokenizer, device
            )

            # Now perform validation using the freshly computed text embeddings:
            val_loss_valpool, val_metrics_valpool, _ = validate_epoch(
                video_encoder=training_setup["video_encoder"],
                text_encoder=text_encoder,
                dataloader=training_setup["val_loader"],
                device=device,
                wandb_run=wandb_run,
                rank=0,
                world_size=1,
                epoch=epoch,
                all_text_embeddings=val_text_embeddings,
                all_reports=val_reports,
                text_embedding_pickle_path=os.path.join(
                    full_output_path, "val_text_embeddings.pkl"
                ),
                output_dir=full_output_path,
                report_to_global_index=val_report_to_index,
                use_val_only_pool=True,  # Val-only retrievals
                log_temp=log_temp,
            )

            # Choose one for best model comparison (typically val-only)
            current_val_loss = val_loss_valpool
            del val_text_embeddings
            torch.cuda.empty_cache()

            if rank == 0 and wandb_run is not None:
                log_data = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/learning_rate": training_setup["optimizer"].param_groups[0]["lr"],
                    "val_only/loss": val_loss_valpool,
                    **{f"val_only/{k}": v for k, v in val_metrics_valpool.items()},
                    "best_val_loss": best_val_loss,  # Log the current best_val_loss each epoch
                }
                # Log temperature if available
                if log_temp is not None:
                    current_temp = torch.exp(log_temp).item()
                    log_data["exp_of_log_temp"] = current_temp
                    log_data["log_temp"] = log_temp.item()

                wandb_run.log(log_data)

            if training_setup["scheduler"] is not None:
                training_setup["scheduler"].step()

            if rank == 0:
                model_dict = {
                    "video_encoder": (
                        training_setup["video_encoder"].module.state_dict()
                        if is_distributed
                        else training_setup["video_encoder"].state_dict()
                    ),
                    "text_encoder": (
                        text_encoder.module.state_dict()
                        if is_distributed
                        else text_encoder.state_dict()
                    ),
                    "optimizer": training_setup["optimizer"].state_dict(),
                    "scheduler": (
                        training_setup["scheduler"].state_dict()
                        if training_setup["scheduler"] is not None
                        else None
                    ),
                    "epoch": epoch,
                }

                metrics_dict = {
                    "train_loss": train_loss,
                    "val_loss_valpool": val_loss_valpool,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    **train_metrics,
                    **val_metrics_valpool,  # store the val-only metrics
                }

                checkpoint_dir = Path(full_output_path) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                latest_path = checkpoint_dir / "latest.pt"
                save_checkpoint(model_dict, metrics_dict, latest_path)
                print(f"\nSaved latest checkpoint at epoch {epoch + 1}")

                # Update best model based on val-only performance
                if current_val_loss < best_val_loss:
                    previous_best = best_val_loss
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    best_path = checkpoint_dir / "best.pt"
                    save_checkpoint(model_dict, metrics_dict, best_path, is_best=True)
                    print(
                        f"\nNew best model saved! Val Loss (val-only): {current_val_loss:.4f} (previous: {previous_best:.4f})"
                    )

                    if wandb_run is not None:
                        # Also log the new best_val_loss immediately when found
                        wandb_run.log(
                            {
                                "best_val_loss": best_val_loss,
                                "best_epoch": best_epoch,
                                "epoch": epoch,
                            }
                        )
                        wandb.save(str(best_path))



    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e
    finally:
        if training_setup is not None and training_setup["wandb_run"] is not None:
            wandb.finish()
        if "is_distributed" in locals() and is_distributed:
            cleanup_ddp()


if __name__ == "__main__":
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    if args.gpu is not None and args.gpu >= num_gpus:
        print(
            f"WARNING: Requested GPU index {args.gpu} is not available. "
            f"Only {num_gpus} GPU(s) detected. Falling back to CPU."
        )
        args.gpu = None

    # Initialize W&B run before adjusting args with wandb.config in main()
    # main() will call create_logger(args) which calls wandb.init().
    # We will adjust args AFTER wandb.init() so wandb.config is available.

    # Temporarily start a run with minimal init to ensure wandb.config is populated:
    # (W&B agent populates wandb.config as soon as wandb.init() is called.)
    # This is handled inside create_logger(). So we just call main and inside main, after create_logger, we override args.

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        main(rank=0, world_size=1, args=args)
    else:
        if torch.cuda.device_count() == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            main(rank=0, world_size=1, args=args)
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            main(rank=0, world_size=1, args=args)
