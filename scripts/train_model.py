"""
Training script for DeepCORO_CLIP model with a built-in VideoAggregator inside VideoEncoder.
(No double-logging inside validate_epoch)
"""

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
    MultiVideoDataset,
    multi_video_collate_fn,
)
from utils.logging import (
    cleanup_temp_video,
    compute_map,
    compute_median_rank,
    compute_ndcg,
    convert_video_for_wandb,
    create_logger,
    get_best_and_worst_retrievals,
    log_val_only_retrievals,
    compute_alignment_score,
    compute_embedding_norms,
    compute_recall_at_k,
    compute_mrr,
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
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
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
    parser.add_argument("--random_augment", type=bool, default=False, help="Use random augmentation")

    # Model parameters
    parser.add_argument(
        "--model_name", type=str, default="mvit_v2_s", help="Video backbone model name"
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone")

    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability in heads")
    parser.add_argument(
        "--video_freeze_ratio",
        type=float,
        default=0.8,
        help="Fraction of video backbone layers to freeze (0.0–1.0)",
    )
    parser.add_argument(
        "--text_freeze_ratio",
        type=float,
        default=0.5,
        help="Fraction of BERT encoder layers to freeze (0.0–1.0).",
    )

    # Optimization parameters
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--scheduler_type", type=str, default="step", help="LR scheduler type")
    parser.add_argument("--lr_step_period", type=int, default=15, help="LR step period")
    parser.add_argument("--factor", type=float, default=0.3, help="Factor for scheduler")
    parser.add_argument("--use_amp", action="store_true", help="Use AMP training")
    parser.add_argument(
        "--patience", type=int, default=5, help="Num. epochs to wait for val improvement"
    )

    # Logging parameters
    parser.add_argument("--project", type=str, default="deepcoro_clip", help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--tag", type=str, default=None, help="Additional tag")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")

    # Multi-video parameters
    parser.add_argument("--multi_video", action="store_true", help="Enable multi-video per study")
    parser.add_argument("--n_video", type=int, default=4, help="Number of videos per study")
    parser.add_argument(
        "--scorn",
        type=str,
        default="StudyInstanceUID",
        help="Column name for study ID or similar grouping key",
    )
    parser.add_argument(
        "--aggregate_function",
        type=str,
        default="mean",
        choices=["mean", "max", "median"],
        help="Aggregation function across multi-video embeddings",
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
        if wandb_run is not None and len(wandb.config.keys()) > 0:
            for key, value in wandb.config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: {key} in wandb.config not recognized as an arg.")

    print("Args: ", args)

    # === Calculate dataset statistics (rank=0 only) ===
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

    # === Datasets ===
    if args.multi_video:
        print("Using MultiVideoDataset")
        train_dataset = MultiVideoDataset(
            root=args.root,
            data_filename=args.data_filename,
            split="train",
            target_label=args.target_label,
            datapoint_loc_label=args.datapoint_loc_label,
            scorn=args.scorn,
            num_videos=args.n_video,
            mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
            std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
            random_augment=args.random_augment,
            backbone=args.model_name,
        )
    else:
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

    if args.multi_video:
        val_dataset = MultiVideoDataset(
            root=args.root,
            data_filename=args.data_filename,
            split="val",
            target_label=args.target_label,
            datapoint_loc_label=args.datapoint_loc_label,
            scorn=args.scorn,
            num_videos=args.n_video,
            mean=mean.tolist() if mean is not None else [0.485, 0.456, 0.406],
            std=std.tolist() if std is not None else [0.229, 0.224, 0.225],
            backbone=args.model_name,
        )
    else:
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

    # === Dataloaders ===
    if args.multi_video:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=multi_video_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=multi_video_collate_fn,
        )
    else:
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

    # === Model Creation ===
    video_encoder = VideoEncoder(
        backbone=args.model_name,
        input_channels=3,
        num_frames=args.frames,
        pretrained=args.pretrained,
        output_dim=512,
        dropout=args.dropout,
        freeze_ratio=args.video_freeze_ratio,
    ).to(device).float()

    text_encoder = TextEncoder(
        dropout=args.dropout,
        freeze_ratio=args.text_freeze_ratio,
    ).to(device).float()

    for param in video_encoder.parameters():
        param.data = param.data.float()
    for param in text_encoder.parameters():
        param.data = param.data.float()

    # Trainable temperature
    temp_tensor = torch.tensor([args.temperature], dtype=torch.float32, device=device)
    if torch.isnan(temp_tensor).any():
        raise ValueError("Temperature value is NaN")
    log_temperature = nn.Parameter(torch.log(temp_tensor))

    # === Optimizer & Scheduler ===
    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_class(
        [
            {"params": video_encoder.parameters()},
            {"params": text_encoder.parameters()},
            {"params": [log_temperature]},
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

    # === Logging Info ===
    if rank == 0 and wandb.run is not None:
        wandb.config.update(
            {
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
            },
            allow_val_change=True,
        )
        print("\n=== Dataset Information ===")
        print(f"Training:   {len(train_dataset):,} videos (or studies if multi-video)")
        print(f"Validation: {len(val_dataset):,} videos (or studies if multi-video)")
        print(f"Total:      {len(train_dataset) + len(val_dataset):,}")
        print(f"\nBatch Size: {args.batch_size}")
        print(f"Training Batches: {len(train_loader):,}")
        print(f"Validation Batches: {len(val_loader):,}")
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
        "wandb_run": wandb.run,
        "log_temperature": log_temperature,
    }


def setup_ddp(rank, world_size):
    """Initialize DDP process group with proper error handling"""
    try:
        if torch.cuda.is_available():
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        init_method = "env://"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),
        )

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
    wandb_run,
    rank=0,
    world_size=1,
    epoch=0,
    scaler=None,
    log_temperature=None,
):
    """
    A single training epoch that can handle single-video or multi-video inputs.
    """
    video_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    num_batches = 0
    epoch_metrics = {
        "video_norm": 0.0,
        "text_norm": 0.0,
        "alignment_score": 0.0,
    }

    if rank == 0:
        progress = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    else:
        progress = dataloader

    for batch_idx, batch in enumerate(progress):
        try:
            videos, encoded_texts, _ = batch
            videos = videos.to(device, non_blocking=True).float()
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # unify shape for aggregator (N dimension if single)
            if videos.dim() == 5:
                videos = videos.unsqueeze(1)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    video_embeds = video_encoder(videos)
                    text_embeds = text_encoder(input_ids, attention_mask)
                    loss = clip_style_loss(video_embeds, text_embeds, log_temperature)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                video_embeds = video_encoder(videos)
                text_embeds = text_encoder(input_ids, attention_mask)
                loss = clip_style_loss(video_embeds, text_embeds, log_temperature)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if rank == 0:
                progress.set_postfix(
                    {
                        "train_loss": f"{loss.item():.4f}",
                        "avg_train_loss": f"{(total_loss / num_batches):.4f}",
                    }
                )

            # local norms
            with torch.no_grad():
                if video_embeds.size(0) >= 5:
                    norm_metrics = compute_embedding_norms(video_embeds, text_embeds)
                    alignment_score = compute_alignment_score(video_embeds, text_embeds)
                    epoch_metrics["video_norm"] += norm_metrics["video_norm"]
                    epoch_metrics["text_norm"] += norm_metrics["text_norm"]
                    epoch_metrics["alignment_score"] += alignment_score

            del loss, video_embeds, text_embeds
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in training batch {batch_idx} on rank {rank}: {str(e)}")
            continue

    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    metric_batches = num_batches if num_batches > 0 else 1
    for metric_name in epoch_metrics:
        epoch_metrics[metric_name] /= metric_batches

    if rank == 0:
        print(f"[Train] Epoch={epoch}, Loss={avg_loss:.4f}")

    return avg_loss, epoch_metrics


def create_logger(args):
    """Create logger with W&B configuration."""
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
    for key, value in vars(args).items():
        if key not in config:
            config[key] = value
    print(config)
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
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {**model_dict, **metrics_dict}
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")


def create_global_text_pool(train_dataset, val_dataset, test_dataset=None):
    train_reports = train_dataset.get_all_reports()
    val_reports = val_dataset.get_all_reports()
    test_reports = test_dataset.get_all_reports() if test_dataset is not None else []

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

    all_global_reports = [None] * len(seen)
    for report, idx in seen.items():
        all_global_reports[idx] = report

    return all_global_reports


def precompute_global_text_embeddings(
    text_encoder, all_global_reports, tokenizer, device, batch_size=64, num_workers=4
):
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
            all_text_embeddings.append(text_features.cpu())

    all_global_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    return all_global_reports, all_global_text_embeddings


def validate_epoch(
    video_encoder,
    text_encoder,
    dataloader,
    device,
    # wandb_run,  # <--- no logging inside validate_epoch, remove wandb here
    rank=0,
    world_size=1,
    epoch=0,
    all_text_embeddings=None,
    all_reports=None,
    output_dir="outputs",
    report_to_global_index=None,
    use_val_only_pool=True,
    log_temperature=None,
):
    """
    Validation epoch that handles single or multi-video, but NO direct wandb.log calls.
    We'll return all metrics, and the main() loop will do the logging afterwards.
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

    # no wandb logging here
    progress = tqdm(dataloader, desc=f"Validation Epoch {epoch}") if rank == 0 else dataloader

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            videos, encoded_texts, paths_or_ids = batch
            videos = videos.to(device).float()
            input_ids = encoded_texts["input_ids"].to(device)
            attention_mask = encoded_texts["attention_mask"].to(device)

            if videos.dim() == 5:
                videos = videos.unsqueeze(1)

            video_embeds = video_encoder(videos)
            text_embeds = text_encoder(input_ids, attention_mask)
            loss = clip_style_loss(video_embeds, text_embeds, log_temperature)

            total_loss += loss.item()
            num_batches += 1

            all_video_embeddings.append(video_embeds.cpu())

            # gather file paths or IDs
            if videos.dim() == 6:
                # multi-video => 'paths_or_ids' is list of study IDs
                this_paths = []
                if hasattr(dataloader.dataset, "study_to_videos"):
                    for sid in paths_or_ids:
                        vid_list = dataloader.dataset.study_to_videos.get(sid, [])
                        if vid_list:
                            this_paths.append(vid_list[0])
                        else:
                            this_paths.append(sid)
                else:
                    this_paths = [str(s) for s in paths_or_ids]
            else:
                if isinstance(paths_or_ids, (list, tuple)):
                    this_paths = paths_or_ids
                else:
                    this_paths = [paths_or_ids]

            all_paths.extend(this_paths)
            batch_reports = dataloader.dataset.get_reports(paths_or_ids)
            all_ground_truth_reports.extend(batch_reports)

            del loss, video_embeds, text_embeds
            torch.cuda.empty_cache()

    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

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

    # unify => shape [N_total, 512]
    all_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(device)
    all_text_embeddings = all_text_embeddings.to(device)

    max_len = min(all_video_embeddings.size(0), all_text_embeddings.size(0))
    truncated_video = all_video_embeddings[:max_len]
    truncated_text = all_text_embeddings[:max_len]

    norm_metrics = compute_embedding_norms(truncated_video, truncated_text)
    epoch_metrics["video_norm"] = norm_metrics["video_norm"]
    epoch_metrics["text_norm"] = norm_metrics["text_norm"]

    all_video_embeddings = nn.functional.normalize(all_video_embeddings, dim=1)
    all_text_embeddings = nn.functional.normalize(all_text_embeddings, dim=1)
    similarity_matrix = torch.matmul(all_video_embeddings, all_text_embeddings.T)

    max_k = min(similarity_matrix.size(0), similarity_matrix.size(1))
    k_values = [k for k in [1, 5, 10, 50] if k <= max_k]

    if max_k >= 1:
        global_ground_truth_indices = [
            report_to_global_index[gt] for gt in all_ground_truth_reports
        ]
        gt_tensor = torch.tensor(global_ground_truth_indices, device=device)

        recall_metrics = compute_recall_at_k(similarity_matrix, gt_tensor, k_values=k_values)
        mrr_metrics = compute_mrr(similarity_matrix, gt_tensor)
        epoch_metrics["NDCG@5_V2T"] = compute_ndcg(similarity_matrix, gt_tensor, k=5)
        epoch_metrics["MedianRank_V2T"] = compute_median_rank(similarity_matrix, gt_tensor)
        epoch_metrics["MAP"] = compute_map(similarity_matrix, gt_tensor)
        alignment_score = compute_alignment_score(truncated_video, truncated_text)

        for metric_name, value in recall_metrics.items():
            epoch_metrics[metric_name] = value
        for metric_name, value in mrr_metrics.items():
            epoch_metrics[metric_name] = value
        epoch_metrics["alignment_score"] = alignment_score

    # We might optionally do log_val_only_retrievals for top examples
    if use_val_only_pool:
        log_val_only_retrievals(
            similarity_matrix=similarity_matrix,
            all_paths=all_paths,
            all_ground_truth_reports=all_ground_truth_reports,
            all_reports=all_reports,
            epoch=epoch,
            wandb_run=None,  # or pass none so it doesn't log
            output_dir=output_dir,
            k=1,
        )

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
        device = training_setup["device"]
        text_encoder = training_setup["text_encoder"]
        train_loader = training_setup["train_loader"]
        val_loader = training_setup["val_loader"]
        train_dataset = training_setup["train_dataset"]
        val_dataset = training_setup["val_dataset"]

        if "log_temperature" not in training_setup:
            raise ValueError("log_temperature not found in training setup")
        log_temperature = training_setup["log_temperature"]

        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0

        # Build validation-only pool
        val_reports = val_dataset.get_all_reports()
        val_unique_reports = list(dict.fromkeys(val_reports))
        val_report_to_index = {r: i for i, r in enumerate(val_unique_reports)}

        scaler = torch.amp.GradScaler(enabled=args.use_amp, device=device)

        for epoch in range(args.epochs):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{args.epochs}")

            # ==== Train Phase ====
            train_loss, train_metrics = train_epoch(
                video_encoder=training_setup["video_encoder"],
                text_encoder=text_encoder,
                dataloader=train_loader,
                optimizer=training_setup["optimizer"],
                device=device,
                wandb_run=wandb_run,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                scaler=scaler,
                log_temperature=log_temperature,
            )

            # ==== Validation Phase (we do no wandb logging here) ====
            val_reports, val_text_embeddings = precompute_global_text_embeddings(
                text_encoder, val_unique_reports, train_dataset.tokenizer, device
            )

            val_loss_valpool, val_metrics_valpool, _ = validate_epoch(
                video_encoder=training_setup["video_encoder"],
                text_encoder=text_encoder,
                dataloader=val_loader,
                device=device,
                # wandb_run=wandb_run, # no direct logging here
                rank=0,
                world_size=1,
                epoch=epoch,
                all_text_embeddings=val_text_embeddings,
                all_reports=val_reports,
                output_dir=full_output_path,
                report_to_global_index=val_report_to_index,
                use_val_only_pool=True,
                log_temperature=log_temperature,
            )
            current_val_loss = val_loss_valpool
            del val_text_embeddings
            torch.cuda.empty_cache()

            # === Single Logging after both train & val ===
            if rank == 0 and wandb_run is not None:
                # Combine train & val metrics in one call
                log_data = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/learning_rate": training_setup["optimizer"].param_groups[0]["lr"],
                    "val_only/loss": val_loss_valpool,
                    **{f"val_only/{k}": v for k, v in val_metrics_valpool.items()},
                    "best_val_loss": best_val_loss,
                }
                if log_temperature is not None:
                    current_temp = torch.exp(log_temperature).item()
                    log_data["temp"] = current_temp
                    log_data["log_temperature"] = log_temperature.item()

                # Now do a single wandb.log
                wandb_run.log(log_data)

            # Step LR
            if training_setup["scheduler"] is not None:
                training_setup["scheduler"].step()

            # Early stopping
            if current_val_loss < best_val_loss:
                previous_best = best_val_loss
                best_val_loss = current_val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    if rank == 0:
                        print(f"Early stopping triggered at epoch {epoch+1} with no improvement.")
                    break

            # === Save Checkpoints ===
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
                    **val_metrics_valpool,
                }

                checkpoint_dir = Path(full_output_path) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                latest_path = checkpoint_dir / "latest.pt"
                save_checkpoint(model_dict, metrics_dict, latest_path)
                print(f"\nSaved latest checkpoint at epoch {epoch + 1}")

                if best_epoch == epoch:
                    best_path = checkpoint_dir / "best.pt"
                    save_checkpoint(model_dict, metrics_dict, best_path, is_best=True)
                    print(
                        f"\nNew best model saved! Val Loss (val-only): {current_val_loss:.4f} "
                        f"(previous: {previous_best:.4f})"
                    )
                    if wandb_run is not None:
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