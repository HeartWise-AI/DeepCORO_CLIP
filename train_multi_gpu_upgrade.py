import os
import sys
import yaml
import time
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.model import TextEncoder, VideoEncoder, contrastive_loss
from utils.data_processing.video import VideoDataset, custom_collate_fn
from utils.logging import create_logger

# Set up logging
def setup_logging(args, rank):
    """Set up logging configuration"""
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_rank{rank}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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

    # Add selected_gpus argument
    parser.add_argument(
        '--selected_gpus', 
        type=int, 
        nargs='+', 
        default=[0, 1, 2, 3], 
        help='List of GPUs to use'
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
    """Set up training environment and parameters."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb if on rank 0
    wandb_run = None
    if rank == 0:
        wandb_run = create_logger(args)

    # Use ImageNet defaults for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Create datasets with ImageNet statistics
    train_dataset = VideoDataset(
        root=args.root,
        data_filename=args.data_filename,
        split="train",
        target_label=args.target_label,
        datapoint_loc_label=args.datapoint_loc_label,
        num_frames=args.frames,
        backbone=args.model_name,
        mean=mean,
        std=std
    )

    val_dataset = VideoDataset(
        root=args.root,
        data_filename=args.data_filename,
        split="val",
        target_label=args.target_label,
        datapoint_loc_label=args.datapoint_loc_label,
        num_frames=args.frames,
        backbone=args.model_name,
        mean=mean,
        std=std
    )

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

    # Initialize distributed training if using multiple GPUs
    if len(args.selected_gpus) > 1:
        # Set device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # Wrap models in DistributedDataParallel
        video_encoder = DDP(
            video_encoder.to(device), 
            device_ids=[rank], 
            find_unused_parameters=True
        )
        text_encoder = DDP(
            text_encoder.to(device), 
            device_ids=[rank], 
            find_unused_parameters=True
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        video_encoder = video_encoder.to(device)
        text_encoder = text_encoder.to(device)

    # Create data samplers for distributed training
    train_sampler = None
    val_sampler = None
    if len(args.selected_gpus) > 1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Update dataloaders with samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

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
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
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
            >= similarity_matrix.gather(
                1, torch.arange(batch_size, device=device).unsqueeze(1)
            )
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
    logger=None,
):
    """Training epoch with proper DDP synchronization and detailed logging"""
    video_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    num_batches = 0
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    # Initialize metric accumulators
    epoch_metrics = {
        "Recall@1_V2T": AverageMeter('R@1_V2T', ':6.2f'),
        "Recall@5_V2T": AverageMeter('R@5_V2T', ':6.2f'),
        "Recall@1_T2V": AverageMeter('R@1_T2V', ':6.2f'),
        "Recall@5_T2V": AverageMeter('R@5_T2V', ':6.2f'),
        "MRR_V2T": AverageMeter('MRR_V2T', ':6.2f'),
        "MRR_T2V": AverageMeter('MRR_T2V', ':6.2f'),
        "video_norm": AverageMeter('VNorm', ':6.2f'),
        "text_norm": AverageMeter('TNorm', ':6.2f'),
        "alignment_score": AverageMeter('Align', ':6.2f'),
    }

    end = time.time()
    
    # Use different progress bars for main process vs others
    if rank == 0:
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            leave=True,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
    else:
        progress = dataloader

    for batch_idx, batch in enumerate(progress):
        try:
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Unpack batch
            videos, encoded_texts, paths = batch
            if videos is None or encoded_texts is None:
                logger.warning(f"Skipping invalid batch {batch_idx}")
                continue

            # Move data to device and ensure float32
            videos = videos.to(device, non_blocking=True).float()
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            batch_size = videos.size(0)
            if batch_size < 2:
                logger.warning(f"Skipping batch {batch_idx} - too few samples ({batch_size})")
                continue

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward passes with optional AMP
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    video_features = video_encoder(videos)
                    text_features = text_encoder(input_ids, attention_mask)
                    normalized_video = nn.functional.normalize(video_features, dim=1)
                    normalized_text = nn.functional.normalize(text_features, dim=1)
                    similarity_matrix = torch.matmul(normalized_video, normalized_text.t())
                    loss = contrastive_loss(video_features, text_features)

                # Backward pass with AMP
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                video_features = video_encoder(videos)
                text_features = text_encoder(input_ids, attention_mask)
                normalized_video = nn.functional.normalize(video_features, dim=1)
                normalized_text = nn.functional.normalize(text_features, dim=1)
                similarity_matrix = torch.matmul(normalized_video, normalized_text.t())
                loss = contrastive_loss(video_features, text_features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(video_encoder.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=5.0)
                optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Update metrics
            losses.update(loss.item(), batch_size)
            total_loss += loss.item()
            num_batches += 1

            # Compute metrics only if we have enough items for k=5
            if batch_size >= 5:
                recall_metrics = compute_recall_at_k(similarity_matrix, k_values=[1, min(5, batch_size)])
                mrr_metrics = compute_mrr(similarity_matrix)
                norm_metrics = compute_embedding_norms(video_features, text_features)
                alignment_score = compute_alignment_score(video_features, text_features)

                # Update metric accumulators
                for metric_name, value in recall_metrics.items():
                    epoch_metrics[metric_name].update(value, batch_size)
                for metric_name, value in mrr_metrics.items():
                    epoch_metrics[metric_name].update(value, batch_size)
                epoch_metrics["video_norm"].update(norm_metrics["video_norm"], batch_size)
                epoch_metrics["text_norm"].update(norm_metrics["text_norm"], batch_size)
                epoch_metrics["alignment_score"].update(alignment_score, batch_size)

            # Update progress bar
            if rank == 0:
                progress.set_postfix({
                    'loss': f'{losses.val:.4f} ({losses.avg:.4f})',
                    'time': f'{batch_time.val:.3f} ({batch_time.avg:.3f})',
                    'data': f'{data_time.val:.3f} ({data_time.avg:.3f})',
                    'R@1': f'{epoch_metrics["Recall@1_V2T"].avg:.2f}',
                    'MRR': f'{epoch_metrics["MRR_V2T"].avg:.2f}'
                })

                # Log to wandb periodically
                if batch_idx % 10 == 0 and wandb_run is not None:
                    wandb_run.log({
                        "train/batch_loss": losses.val,
                        "train/avg_loss": losses.avg,
                        "train/batch_time": batch_time.val,
                        "train/data_time": data_time.val,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        **{f"train/{k}": v.avg for k, v in epoch_metrics.items()},
                        "train/epoch": epoch,
                        "train/step": batch_idx + epoch * len(dataloader),
                    })

            # Log detailed metrics periodically
            if batch_idx % 50 == 0 and logger is not None:
                logger.info(
                    f"Epoch: [{epoch}][{batch_idx}/{len(dataloader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"R@1_V2T {epoch_metrics['Recall@1_V2T'].avg:.2f} "
                    f"MRR_V2T {epoch_metrics['MRR_V2T'].avg:.2f}"
                )

            # Explicitly clear tensors from GPU memory
            del videos, input_ids, attention_mask, video_features, text_features
            del normalized_video, normalized_text, similarity_matrix, loss
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in training batch {batch_idx} on rank {rank}: {str(e)}")
            logger.error(f"Batch size: {videos.size(0) if 'videos' in locals() else 'unknown'}")
            continue

    # Average loss and metrics
    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Log epoch summary
    if rank == 0 and logger is not None:
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Average Batch Time: {batch_time.avg:.3f}")
        logger.info(f"Average Data Time: {data_time.avg:.3f}")
        logger.info("Metrics:")
        for metric_name, meter in epoch_metrics.items():
            logger.info(f"{metric_name}: {meter.avg:.4f}")

    return avg_loss, {k: v.avg for k, v in epoch_metrics.items()}


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


def convert_to_mp4(input_path):
    """Convert video to MP4 format for wandb logging with reduced size.

    Args:
        input_path: Path to input video file

    Returns:
        str: Path to converted MP4 file or None if conversion fails
    """
    import os
    import subprocess
    import tempfile

    # Create temporary file with .mp4 extension
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)

    try:
        # Convert to MP4 using ffmpeg with reduced quality and size
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-crf",
                "35",  # Higher CRF = lower quality/size
                "-preset",
                "veryfast",  # Faster encoding
                "-vf",
                "scale=320:-1",  # Reduce resolution to 320p width
                "-r",
                "15",  # Reduce framerate to 15fps
                "-y",
                temp_path,
            ],
            check=True,
            capture_output=True,
        )
        return temp_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to convert video {input_path} to MP4: {e.stderr.decode()}")
        os.unlink(temp_path)
        return None


def validate_epoch(video_encoder, text_encoder, dataloader, device, wandb_run, rank=0, world_size=1, epoch=0):
    """Simplified validation epoch."""
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
    }

    if rank == 0:
        progress = tqdm(dataloader, desc="Validation")
    else:
        progress = dataloader

    with torch.no_grad():
        for batch in progress:
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
                ) = get_best_and_worst_retrievals(
                    similarity_matrix,
                    paths,
                    reports,
                    k=min(2, batch_size),  # Adjust k based on batch size
                )

                # Update best/worst lists if we find better examples
                for i, (idx, score) in enumerate(zip(best_indices, best_scores)):
                    if len(val_best_scores) < 2 or score > min(val_best_scores):
                        video_similarities = similarity_matrix[idx]
                        n_texts = min(5, similarity_matrix.size(1))
                        top_5_text_indices = torch.argsort(video_similarities, descending=True)[:n_texts]
                        predicted_reports = [reports[text_idx.item()] for text_idx in top_5_text_indices]

                        val_best_videos.append(str(paths[idx]))
                        val_best_reports.append({
                            "ground_truth": reports[idx],
                            "predicted": predicted_reports,
                            "similarity_score": score.item(),
                        })
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
                        predicted_reports = [reports[text_idx.item()] for text_idx in top_5_text_indices]

                        val_worst_videos.append(str(paths[idx]))
                        val_worst_reports.append({
                            "ground_truth": reports[idx],
                            "predicted": predicted_reports,
                            "similarity_score": score.item(),
                        })
                        val_worst_scores.append(score.item())

                        # Keep only bottom 2
                        if len(val_worst_scores) > 2:
                            max_idx = val_worst_scores.index(max(val_worst_scores))
                            val_worst_videos.pop(max_idx)
                            val_worst_reports.pop(max_idx)
                            val_worst_scores.pop(max_idx)

                # Compute metrics only if we have enough items for k=5
                if batch_size >= 5:
                    recall_metrics = compute_recall_at_k(similarity_matrix, k_values=[1, min(5, batch_size)])
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

                # Explicitly clear tensors from GPU memory
                del videos, input_ids, attention_mask, video_features, text_features
                del normalized_video, normalized_text, similarity_matrix, loss
                torch.cuda.empty_cache()  # Clear GPU cache

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
    for metric_name in epoch_metrics:
        epoch_metrics[metric_name] /= num_batches if num_batches > 0 else 1

    if rank == 0:
        print(f"\nValidation Loss: {avg_loss:.4f}")

    return avg_loss, epoch_metrics


def ddp_setup():
    """Initialize DDP process group"""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


class AverageMeter:
    """Computes and stores the average and current value.
    
    Attributes:
        name (str): Name of the meter for printing
        fmt (str): Format string for printing
        val (float): Most recent value
        avg (float): Running average of all values
        sum (float): Sum of all values
        count (int): Count of values
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics with new value.
        
        Args:
            val (float): Value to update with
            n (int): Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation for printing."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Trainer:
    def __init__(
        self,
        video_encoder: torch.nn.Module,
        text_encoder: torch.nn.Module, 
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        save_every: int,
        snapshot_path: str,
        args: argparse.Namespace,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.args = args
        self.video_encoder = video_encoder.to(self.gpu_id)
        self.text_encoder = text_encoder.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        
        # Set up logging
        self.logger = setup_logging(args, self.gpu_id)
        
        if os.path.exists(snapshot_path):
            self.logger.info("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.video_encoder = DDP(
            self.video_encoder,
            device_ids=[self.gpu_id],
            output_device=self.gpu_id,
            find_unused_parameters=True,
        )
        self.text_encoder = DDP(
            self.text_encoder,
            device_ids=[self.gpu_id],
            output_device=self.gpu_id,
            find_unused_parameters=True,
        )
        
        # Initialize wandb only on main process
        self.wandb_run = None
        if self.gpu_id == 0:
            self.wandb_run = create_logger(args)

        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        
        # Log model information
        if self.gpu_id == 0:
            self.logger.info("\nModel Information:")
            self.logger.info(f"Video Encoder: {args.model_name}")
            self.logger.info(f"Pretrained: {args.pretrained}")
            self.logger.info(f"Number of frames: {args.frames}")
            self.logger.info(f"\nTraining Configuration:")
            self.logger.info(f"Batch size per GPU: {args.batch_size}")
            self.logger.info(f"Number of GPUs: {dist.get_world_size()}")
            self.logger.info(f"Total batch size: {args.batch_size * dist.get_world_size()}")
            self.logger.info(f"Learning rate: {args.lr}")
            self.logger.info(f"Weight decay: {args.weight_decay}")
            self.logger.info(f"Mixed Precision Training: {args.use_amp}")
            self.logger.info("-" * 50)

    def _load_snapshot(self, snapshot_path):
        try:
            loc = f"cuda:{self.gpu_id}"
            snapshot = torch.load(snapshot_path, map_location=loc)
            self.video_encoder.load_state_dict(snapshot["VIDEO_ENCODER_STATE"])
            self.text_encoder.load_state_dict(snapshot["TEXT_ENCODER_STATE"])
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
            self.epochs_run = snapshot["EPOCHS_RUN"]
            self.logger.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        except Exception as e:
            self.logger.error(f"Error loading snapshot: {str(e)}")
            raise e

    def _save_snapshot(self, epoch, is_best=False):
        snapshot = {
            "VIDEO_ENCODER_STATE": self.video_encoder.module.state_dict(),
            "TEXT_ENCODER_STATE": self.text_encoder.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        
        # Save regular checkpoint
        torch.save(snapshot, self.snapshot_path)
        self.logger.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        
        # Save best model if specified
        if is_best:
            best_path = os.path.join(os.path.dirname(self.snapshot_path), "best_model.pt")
            torch.save(snapshot, best_path)
            self.logger.info(f"New best model saved at {best_path}")

    def train(self, max_epochs: int):
        self.logger.info(f"\nStarting training for {max_epochs} epochs...")
        best_val_loss = float('inf')
        train_start_time = time.time()
        
        for epoch in range(self.epochs_run, max_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_metrics = train_epoch(
                self.video_encoder,
                self.text_encoder,
                self.train_data,
                self.optimizer,
                self.gpu_id,
                self.wandb_run,
                rank=self.gpu_id,
                world_size=dist.get_world_size(),
                epoch=epoch,
                scaler=self.scaler,
                logger=self.logger
            )
            
            # # Validation
            # val_loss, val_metrics = validate_epoch(
            #     self.video_encoder,
            #     self.text_encoder,
            #     self.val_data,
            #     self.gpu_id,
            #     self.wandb_run,
            #     rank=self.gpu_id,
            #     world_size=dist.get_world_size(),
            #     epoch=epoch,
            #     logger=self.logger
            # )
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics on main process
            if self.gpu_id == 0:
                # Log to wandb
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss,
                        **{f"train/{k}": v for k, v in train_metrics.items()},
                        **{f"val/{k}": v for k, v in val_metrics.items()},
                        "epoch": epoch,
                        "epoch_time": epoch_time,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                    })
                
                # Save best model
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    self.logger.info(f"\nNew best model! Validation Loss: {val_loss:.4f}")
                
                # Regular checkpoint saving
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch, is_best=is_best)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                if self.gpu_id == 0:
                    self.logger.info(f"Learning rate adjusted to: {self.optimizer.param_groups[0]['lr']:.1e}")
            
            # Log epoch summary
            if self.gpu_id == 0:
                self.logger.info(f"\nEpoch {epoch} completed in {epoch_time:.2f} seconds")
                self.logger.info(f"Training Loss: {train_loss:.4f}")
                self.logger.info(f"Validation Loss: {val_loss:.4f}")
                self.logger.info("-" * 50)
        
        # Log final training summary
        if self.gpu_id == 0:
            total_time = time.time() - train_start_time
            self.logger.info("\nTraining completed!")
            self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
            self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
            self.logger.info("=" * 50)


def load_train_objs(args):
    """Load training objects - models, datasets, optimizer etc."""
    # Create datasets
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
    )

    val_dataset = VideoDataset(
        root=args.root,
        data_filename=args.data_filename,
        split="val",
        target_label=args.target_label,
        datapoint_loc_label=args.datapoint_loc_label,
        num_frames=args.frames,
        backbone=args.model_name,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],        
    )

    # Create models
    video_encoder = VideoEncoder(
        backbone=args.model_name,
        input_channels=3,
        num_frames=args.frames,
        pretrained=args.pretrained,
        output_dim=512,
    )

    text_encoder = TextEncoder()

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

    return train_dataset, val_dataset, video_encoder, text_encoder, optimizer, scheduler


def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        sampler=DistributedSampler(dataset),
        collate_fn=custom_collate_fn,
    )


def main(args):
    ddp_setup()
    
    # Load training objects
    train_dataset, val_dataset, video_encoder, text_encoder, optimizer, scheduler = load_train_objs(args)
    
    # Prepare dataloaders
    train_loader = prepare_dataloader(train_dataset, args.batch_size, args.num_workers)
    val_loader = prepare_dataloader(val_dataset, args.batch_size, args.num_workers)
    
    # Create trainer
    trainer = Trainer(
        video_encoder=video_encoder,
        text_encoder=text_encoder,
        train_data=train_loader,
        val_data=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        save_every=args.save_every if hasattr(args, 'save_every') else 1,
        snapshot_path=os.path.join(args.output_dir, "snapshot.pt"),
        args=args,
    )
    
    # Train model
    trainer.train(args.epochs)
    
    # Cleanup
    destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)