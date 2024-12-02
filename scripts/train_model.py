"""Training script for DeepCORO_CLIP model."""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

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
)
from utils.logging import create_logger


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


def setup_training(args):
    """Set up training environment and parameters.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary containing training setup
    """
    # Create output directory
    output_dir = Path(args.output_dir if hasattr(args, "output_dir") else "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    if hasattr(args, "seed"):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Setup device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    video_encoder = VideoEncoder(
        backbone=args.model_name,
        input_channels=3,
        num_frames=args.frames,
        pretrained=args.pretrained,
        output_dim=512,
    ).to(device)

    text_encoder = TextEncoder().to(device)

    # Create optimizer
    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_class(
        [
            {"params": video_encoder.parameters()},
            {"params": text_encoder.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_period,
            gamma=args.factor,
        )
    else:
        scheduler = None

    return {
        "device": device,
        "video_encoder": video_encoder,
        "text_encoder": text_encoder,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "output_dir": output_dir,
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

    # Metric accumulators for epoch-level metrics
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

            # Move data to device
            videos = videos.to(device, non_blocking=True)
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            # Forward passes
            video_features = video_encoder(videos)
            text_features = text_encoder(input_ids, attention_mask)

            # Compute similarity matrix for metrics
            normalized_video = nn.functional.normalize(video_features, dim=1)
            normalized_text = nn.functional.normalize(text_features, dim=1)
            similarity_matrix = torch.matmul(normalized_video, normalized_text.t())

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
            optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Log loss immediately for continuous monitoring
            if rank == 0 and wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/batch": batch_idx,
                        "train/progress": (batch_idx + 1) / len(dataloader),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/batch_size": len(videos),
                        "train/global_step": epoch * len(dataloader) + batch_idx,
                    }
                )

                progress.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{(total_loss/num_batches):.4f}",
                        "device": str(device),
                    }
                )

            # Compute and accumulate metrics for epoch-level logging
            recall_metrics = compute_recall_at_k(similarity_matrix, k_values=[1, 5])
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

        except RuntimeError as e:
            print(f"Error in batch {batch_idx} on rank {rank} (device {device}): {str(e)}")
            print(f"Debug info:")
            print(f"  Video shape: {videos.shape}")
            print(f"  Input IDs shape: {input_ids.shape}")
            print(f"  Input IDs max value: {input_ids.max().item()}")
            print(f"  Attention mask shape: {attention_mask.shape}")
            continue

    # Average loss and metrics across all processes
    if world_size > 1:
        # All-reduce for loss synchronization
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Average epoch metrics
    for metric_name in epoch_metrics:
        epoch_metrics[metric_name] /= num_batches

    # Log epoch-level metrics on rank 0
    if rank == 0 and wandb_run is not None:
        print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")
        print(f"Processed {num_batches} batches")

        # Log all epoch metrics
        wandb_run.log(
            {
                "epoch": epoch,
                "epoch/avg_loss": avg_loss,
                "epoch/num_batches": num_batches,
                **{f"epoch/retrieval/{k}": v for k, v in recall_metrics.items()},
                **{f"epoch/retrieval/{k}": v for k, v in mrr_metrics.items()},
                "epoch/embeddings/video_norm": epoch_metrics["video_norm"],
                "epoch/embeddings/text_norm": epoch_metrics["text_norm"],
                "epoch/embeddings/alignment_score": epoch_metrics["alignment_score"],
            }
        )

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
            # Create logger with proper WandB configuration
            wandb_run = create_logger(args)

        # Set up training components
        training_setup = setup_training(args)
        data_setup = setup_data(args)

        # Training loop
        for epoch in range(args.epochs):
            if data_setup["train_sampler"] is not None:
                data_setup["train_sampler"].set_epoch(epoch)

            train_loss = train_epoch(
                video_encoder=training_setup["video_encoder"],
                text_encoder=training_setup["text_encoder"],
                dataloader=data_setup["train_loader"],
                optimizer=training_setup["optimizer"],
                device=training_setup["device"],
                wandb_run=wandb_run,  # Pass wandb run instead of logger
                rank=rank if is_distributed else 0,
                world_size=world_size if is_distributed else 1,
                epoch=epoch,
            )

            # Step scheduler if it exists
            if training_setup["scheduler"] is not None:
                training_setup["scheduler"].step()

            # Save checkpoints only on main process
            if should_log:
                checkpoint = {
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
                    "epoch": epoch,
                    "loss": train_loss,
                }

                if (epoch + 1) % 5 == 0:
                    checkpoint_dir = Path(training_setup["output_dir"]) / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint for epoch {epoch+1}")
                    if wandb_run is not None:
                        wandb.save(str(checkpoint_path))

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e

    finally:
        if wandb_run is not None:
            wandb.finish()
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
