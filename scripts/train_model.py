"""
Training script for DeepCORO_CLIP model with a built-in VideoAggregator inside VideoEncoder.
(No double-logging inside validate_epoch)
"""

import argparse
import os
import pickle
import sys
import random
import numpy as np
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

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


def set_seed(seed):
    """
    Seeds Python, NumPy, and PyTorch for reproducibility.
    Also makes CuDNN deterministic if you want fully consistent runs
    (at the cost of performance).
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    DataLoader worker_init_fn to ensure each worker uses a reproducible seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

    # -- NEW: Resume arguments
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to the latest.pt or best.pt checkpoint to resume from",
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
    parser.add_argument("--max_num_videos", type=int, default=4, help="Number of videos per study")
    parser.add_argument(
        "--groupby_column",
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
    parser.add_argument("--shuffle_videos", action="store_true", help="Shuffle videos for each study")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for video sampling")

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


def preview_checkpoint_for_resuming(checkpoint_path: str):
    """
    Quick "preview" function that loads minimal fields from the checkpoint 
    on CPU, to retrieve wandb_run, epoch, best_val_loss, best_epoch, etc.
    Returns:
      wandb_run (str or None)
      start_epoch (int)
      best_val_loss (float)
      best_epoch (int)
    """
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: checkpoint not found at {checkpoint_path}.")
        return None, 0, float('inf'), -1

    print(f"[Preview] Loading minimal info from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    wandb_run = checkpoint.get("wandb_run", None)
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    best_epoch = checkpoint.get("best_epoch", -1)

    print(
        f"[Preview] Found run_id={wandb_run}, "
        f"start_epoch={start_epoch}, best_val_loss={best_val_loss}, best_epoch={best_epoch}"
    )
    return wandb_run, start_epoch, best_val_loss, best_epoch


def load_full_checkpoint(checkpoint_path: str, device, training_setup):
    """
    After the model/optimizer/etc. is initialized, call this 
    to actually load states from the checkpoint into them.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: checkpoint not found at {checkpoint_path}. No loading done.")
        return

    print(f"[Full Load] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    video_encoder = training_setup["video_encoder"]
    text_encoder = training_setup["text_encoder"]
    optimizer = training_setup["optimizer"]
    scheduler = training_setup["scheduler"]

    if "video_encoder" in checkpoint:
        video_encoder.load_state_dict(checkpoint["video_encoder"], strict=False)
    if "text_encoder" in checkpoint:
        text_encoder.load_state_dict(checkpoint["text_encoder"], strict=False)

    if "optimizer" in checkpoint and checkpoint["optimizer"]:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint and checkpoint["scheduler"] and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])


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


def setup_training(args, wandb_run, rank=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === SEEDING ===
    set_seed(args.seed)

    # === Stats
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
            worker_init_fn=seed_worker,  # Ensures reproducible data loading
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

    # === Datasets
    if args.multi_video:
        print("Using MultiVideoDataset")
        train_dataset = MultiVideoDataset(
            root=args.root,
            data_filename=args.data_filename,
            split="train",
            target_label=args.target_label,
            datapoint_loc_label=args.datapoint_loc_label,
            groupby_column=args.groupby_column,
            max_num_videos=args.max_num_videos,
            mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
            std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
            random_augment=args.random_augment,
            backbone=args.model_name,
            shuffle_videos=args.shuffle_videos,
            seed=args.seed,  # Pass along seed
        )
        val_dataset = MultiVideoDataset(
            root=args.root,
            data_filename=args.data_filename,
            split="val",
            target_label=args.target_label,
            datapoint_loc_label=args.datapoint_loc_label,
            groupby_column=args.groupby_column,
            max_num_videos=args.max_num_videos,
            mean=mean.tolist() if mean is not None else [0.485, 0.456, 0.406],
            std=std.tolist() if std is not None else [0.229, 0.224, 0.225],
            backbone=args.model_name,
            shuffle_videos=False,
            seed=args.seed,  # Pass along seed
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
        raise ValueError("No validation samples found!")

    if args.multi_video:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=multi_video_collate_fn,
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=multi_video_collate_fn,
            worker_init_fn=seed_worker,
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
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn,
            worker_init_fn=seed_worker,
        )

    # === Model
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

    temp_tensor = torch.tensor([args.temperature], dtype=torch.float32, device=device)
    if torch.isnan(temp_tensor).any():
        raise ValueError("Temperature value is NaN")
    log_temperature = nn.Parameter(torch.log(temp_tensor))

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

    if rank == 0 and wandb_run is not None:
        wandb.config.update(
            {
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
            },
            allow_val_change=True,
        )
        print("\n=== Dataset Information ===")
        print(f"Training:   {len(train_dataset):,} (studies if multi-video)")
        print(f"Validation: {len(val_dataset):,} (studies if multi-video)")
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
        "wandb_run": wandb_run.id if wandb_run else None,
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


def save_checkpoint(model_dict, metrics_dict, output_path, is_best=False, wandb_run=None):
    """
    Save model checkpoint with metrics, plus wandb_run if provided.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {**model_dict, **metrics_dict}

    if wandb_run is not None:
        checkpoint["wandb_run"] = wandb_run

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
    """
    Return:
       all_global_reports => the same unique list
       all_global_text_embeddings => shape [num_unique, emb_dim]
    """
    from torch.utils.data import Dataset, DataLoader

    class GlobalTextDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            return enc

    text_dataset = GlobalTextDataset(all_global_reports, tokenizer)
    text_loader = DataLoader(text_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    all_text_embeddings = []
    text_encoder.eval()
    with torch.no_grad():
        for batch_texts in text_loader:
            input_ids = batch_texts["input_ids"].to(device)
            attention_mask = batch_texts["attention_mask"].to(device)
            txt_feats = text_encoder(input_ids, attention_mask)
            all_text_embeddings.append(txt_feats.cpu())
    all_global_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    return all_global_reports, all_global_text_embeddings


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
    output_dir="outputs",
    report_to_global_index=None,
    use_val_only_pool=True,
    log_temperature=None,
):
    """
    Validation epoch with retrieval computation and logging.
    This version assumes your VideoEncoder already does aggregation internally.

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
        use_val_only_pool (bool): If True, evaluate retrieval using only the val set embeddings
        log_temperature: learned temperature for contrastive loss

    Returns:
        avg_loss, epoch_metrics, examples_dict
          where examples_dict includes info about best/worst retrieval examples
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
    all_text_embeddings = []
    all_ground_truth_reports = []

    if rank == 0:
        progress = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
    else:
        progress = dataloader

    with torch.no_grad():
        for batch_idx, (videos, encoded_texts, study_ids) in enumerate(progress):
            videos = videos.to(device).float()

            input_ids = encoded_texts["input_ids"].to(device)
            attention_mask = encoded_texts["attention_mask"].to(device)

            video_embeds = video_encoder(videos)
            text_embeds = text_encoder(input_ids, attention_mask)

            loss = clip_style_loss(video_embeds, text_embeds, log_temperature)
            total_loss += loss.item()
            num_batches += 1

            all_video_embeddings.append(video_embeds.cpu())
            all_text_embeddings.append(text_embeds.cpu())

            if hasattr(dataloader.dataset, "get_video_paths"):
                for sid in study_ids:
                    vid_list = dataloader.dataset.get_video_paths(sid)
                    if len(vid_list) > 0:
                        all_paths.append(vid_list[0])
                    else:
                        all_paths.append(str(sid))
            else:
                for p in study_ids:
                    all_paths.append(str(p))

            batch_reports = dataloader.dataset.get_reports(study_ids)
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
            print("No validation data processed.")
        return avg_loss, epoch_metrics, {
            "best_videos": [],
            "best_reports": [],
            "best_scores": [],
            "worst_videos": [],
            "worst_reports": [],
            "worst_scores": [],
        }

    all_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(device)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(device)

    all_video_embeddings = nn.functional.normalize(all_video_embeddings, dim=1)
    all_text_embeddings = nn.functional.normalize(all_text_embeddings, dim=1)
    similarity_matrix = torch.matmul(all_video_embeddings, all_text_embeddings.T)

    max_len = min(all_video_embeddings.size(0), all_text_embeddings.size(0))
    nm = compute_embedding_norms(
        all_video_embeddings[:max_len],
        all_text_embeddings[:max_len],
    )
    epoch_metrics["video_norm"] = nm["video_norm"]
    epoch_metrics["text_norm"] = nm["text_norm"]

    num_items, num_texts = similarity_matrix.shape
    if num_items > 0 and num_texts > 0:
        global_gt_indices = []
        for gt_text in all_ground_truth_reports:
            global_gt_indices.append(report_to_global_index[gt_text])
        gt_tensor = torch.tensor(global_gt_indices, device=device)

        max_k = min(num_items, num_texts)
        k_values = [k for k in [1, 5, 10, 50] if k <= max_k]
        if len(k_values) > 0:
            recall_metrics = compute_recall_at_k(similarity_matrix, gt_tensor, k_values)
            epoch_metrics.update(recall_metrics)

            epoch_metrics["MRR_V2T"] = compute_mrr(similarity_matrix, gt_tensor)["MRR_V2T"]
            epoch_metrics["NDCG@5_V2T"] = compute_ndcg(similarity_matrix, gt_tensor, k=5)
            epoch_metrics["MedianRank_V2T"] = compute_median_rank(similarity_matrix, gt_tensor)
            epoch_metrics["MAP"] = compute_map(similarity_matrix, gt_tensor)
        else:
            print("No valid k values found for retrieval metrics.")

        truncated_video = all_video_embeddings[:max_len]
        truncated_text = all_text_embeddings[:max_len]
        epoch_metrics["alignment_score"] = compute_alignment_score(truncated_video, truncated_text)

    best_videos = []
    best_reports = []
    worst_videos = []
    worst_reports = []

    if use_val_only_pool:
        log_val_only_retrievals(
            similarity_matrix=similarity_matrix,
            all_paths=all_paths,
            all_ground_truth_reports=all_ground_truth_reports,
            all_reports=all_reports,
            epoch=epoch,
            wandb_run=wandb_run,
            output_dir=output_dir,
            k=1,
            report_to_global_index=report_to_global_index,
        )

    if rank == 0:
        print(f"[Val] Epoch={epoch}, Loss={avg_loss:.4f}")

    return (
        avg_loss,
        epoch_metrics,
        {
            "best_videos": best_videos,
            "best_reports": best_reports,
            "best_scores": [x["similarity_score"] for x in best_reports],
            "worst_videos": worst_videos,
            "worst_reports": worst_reports,
            "worst_scores": [x["similarity_score"] for x in worst_reports],
        },
    )


def save_texts_csv(output_dir, texts):
    import csv
    csv_path = os.path.join(output_dir, "val_texts.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Text"])
        for idx, txt in enumerate(texts):
            writer.writerow([idx, txt])
    print(f"Saved {len(texts)} val texts to {csv_path}")


def main(rank=0, world_size=1, args=None):
    training_setup = None
    try:
        wandb_run = None
        start_epoch = 0
        best_val_loss = float("inf")
        best_epoch = -1
        if args.resume and args.resume_checkpoint:
            (wandb_run, start_epoch, best_val_loss, best_epoch) = preview_checkpoint_for_resuming(
                args.resume_checkpoint
            )

        if rank == 0:
            wandb_kwargs = {
                "project": args.project,
                "entity": args.entity,
                "name": args.tag,
            }
            if wandb_run:
                wandb_kwargs["id"] = wandb_run.id
                wandb_kwargs["resume"] = "allow"
                print(f"[W&B] Resuming run_id={wandb_run.id}")
                wandb_run = wandb.init(**wandb_kwargs)
            else:
                wandb_run = create_logger(args)
                if wandb_run is not None and len(wandb.config.keys()) > 0:
                    for key, value in wandb.config.items():
                        if hasattr(args, key):
                            setattr(args, key, value)
                print("[W&B] Starting new run (no run_id found).")

        training_setup = setup_training(args, wandb_run, rank=rank)

        device = training_setup["device"]
        video_encoder = training_setup["video_encoder"]
        text_encoder = training_setup["text_encoder"]
        optimizer = training_setup["optimizer"]
        scheduler = training_setup["scheduler"]
        train_loader = training_setup["train_loader"]
        val_loader = training_setup["val_loader"]
        train_dataset = training_setup["train_dataset"]
        val_dataset = training_setup["val_dataset"]
        log_temperature = training_setup["log_temperature"]
        is_distributed = world_size > 1

        if args.resume and args.resume_checkpoint:
            load_full_checkpoint(args.resume_checkpoint, device, training_setup)

        print(f"[Resume] start_epoch={start_epoch}, best_val_loss={best_val_loss}, best_epoch={best_epoch}")

        val_reports = val_dataset.get_all_reports()
        val_unique_reports = list(dict.fromkeys(val_reports))
        val_unique_report_to_index = {r: i for i, r in enumerate(val_unique_reports)}
        val_report_to_index = {r: i for i, r in enumerate(val_reports)}

        output_subdir = generate_output_dir_name(args, wandb_run.id if wandb_run else "offline")
        full_output_path = os.path.join(args.output_dir, output_subdir)

        if rank == 0:
            if not os.path.exists(full_output_path):
                os.makedirs(full_output_path)
            save_texts_csv(full_output_path, val_reports)

        scaler = torch.amp.GradScaler(enabled=args.use_amp, device=device)

        best_epoch = -1 if best_epoch < 0 else best_epoch
        patience_counter = 0

        for epoch in range(start_epoch, args.epochs):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_loss, train_metrics = train_epoch(
                video_encoder=video_encoder,
                text_encoder=text_encoder,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                wandb_run=wandb_run,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                scaler=scaler,
                log_temperature=log_temperature,
            )

            val_loss_valpool, val_metrics_valpool, _ = validate_epoch(
                video_encoder=video_encoder,
                text_encoder=text_encoder,
                dataloader=val_loader,
                device=device,
                wandb_run=wandb_run,
                rank=0,
                world_size=1,
                epoch=epoch,
                all_text_embeddings=None,
                all_reports=val_reports,
                output_dir=full_output_path,
                report_to_global_index=val_report_to_index,
                use_val_only_pool=True,
                log_temperature=log_temperature,
            )
            current_val_loss = val_loss_valpool

            torch.cuda.empty_cache()

            if rank == 0 and wandb_run is not None:
                log_data = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "val_only/loss": val_loss_valpool,
                    **{f"val_only/{k}": v for k, v in val_metrics_valpool.items()},
                    "best_val_loss": best_val_loss,
                }
                if log_temperature is not None:
                    current_temp = torch.exp(log_temperature).item()
                    log_data["temp"] = current_temp
                    log_data["log_temperature"] = log_temperature.item()

                wandb_run.log(log_data)

            if scheduler is not None:
                scheduler.step()

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

            if rank == 0:
                model_dict = {
                    "video_encoder": (
                        video_encoder.module.state_dict() if is_distributed else video_encoder.state_dict()
                    ),
                    "text_encoder": (
                        text_encoder.module.state_dict() if is_distributed else text_encoder.state_dict()
                    ),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
        if dist.is_initialized():
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