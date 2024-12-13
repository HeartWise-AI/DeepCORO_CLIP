import os

# Set tokenizer parallelism before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from train_model import TextEncoder, VideoDataset, VideoEncoder, contrastive_loss, train_epoch

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument("--resume", type=str, help="path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=24, help="base batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--debug", action="store_true", help="enable CUDA debug mode")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    return parser.parse_args()


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


def main(rank, world_size, args):
    # Enable CUDA debugging if requested
    if args.debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.backends.cudnn.enabled = False

    # Setup DDP
    device = setup_ddp(rank, world_size)

    # Setup wandb (only on rank 0)
    if rank == 0:
        wandb.init(project="deepcoro_clip")
        print(f"Found {world_size} GPUs")
        # Print CUDA memory info
        for i in range(world_size):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB")
            print(f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f}GB")
        print(f"Using device: {device}")

    try:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Initialize models
        video_encoder = VideoEncoder().to(device)
        text_encoder = TextEncoder().to(device)

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
            root="data/processed/reports",
            data_filename="reports_sampled_1000.csv",
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            mean=[107.56801, 107.56801, 107.56801],
            std=[40.988625, 40.988625, 40.988625],
        )

        if rank == 0:
            print(f"Dataset size: {len(train_dataset)}")

        # Create distributed sampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )

        # Create dataloader with distributed sampler
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,  # This is now per-GPU batch size
            shuffle=False,  # Shuffle is handled by sampler
            num_workers=min(6, os.cpu_count() or 1),  # Reduce workers per GPU
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        # Optimizer with proper DDP parameter handling
        params = [
            {"params": [p for p in video_encoder.module.parameters() if p.requires_grad]},
            {"params": [p for p in text_encoder.module.parameters() if p.requires_grad]},
        ]
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        # Load checkpoint if resuming
        start_epoch = 0
        best_loss = float("inf")
        if args.resume:
            if os.path.isfile(args.resume):
                if rank == 0:
                    print(f"Loading checkpoint '{args.resume}'")
                # Map model to be loaded to specified single gpu
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint["epoch"] + 1
                best_loss = checkpoint.get("best_loss", float("inf"))
                video_encoder.module.load_state_dict(checkpoint["video_encoder"])
                text_encoder.module.load_state_dict(checkpoint["text_encoder"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                if rank == 0:
                    print(f"Resumed from epoch {start_epoch}")
            elif rank == 0:
                print(f"No checkpoint found at '{args.resume}'")

        # Create checkpoint directory (only on rank 0)
        if rank == 0:
            checkpoint_dir = Path("models/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        for epoch in range(start_epoch, args.epochs):
            # Set epoch for sampler
            train_sampler.set_epoch(epoch)

            if rank == 0:
                print(f"\nEpoch {epoch+1}/{args.epochs}")

            try:
                train_loss = train_epoch(
                    video_encoder=video_encoder,
                    text_encoder=text_encoder,
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                )

                # Log metrics (only on rank 0)
                if rank == 0:
                    print(f"Training loss: {train_loss:.4f}")
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "gpu_memory": torch.cuda.memory_allocated(rank) / 1e9,
                        }
                    )

                    # Save checkpoint
                    is_best = train_loss < best_loss
                    best_loss = min(train_loss, best_loss)

                    checkpoint = {
                        "video_encoder": video_encoder.module.state_dict(),
                        "text_encoder": text_encoder.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": train_loss,
                        "best_loss": best_loss,
                        "batch_size": args.batch_size * world_size,
                    }

                    # Save regular checkpoint
                    if (epoch + 1) % 5 == 0:
                        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        print(f"Saved checkpoint for epoch {epoch+1}")

                    # Save best model
                    if is_best:
                        best_model_path = checkpoint_dir / "best_model.pt"
                        torch.save(checkpoint, best_model_path)
                        print(f"Saved new best model with loss: {train_loss:.4f}")

                # Synchronize processes after each epoch
                dist.barrier()

            except RuntimeError as e:
                print(f"Error during training on rank {rank}: {str(e)}")
                print("Attempting to recover...")
                torch.cuda.empty_cache()
                continue

    except Exception as e:
        print(f"Training failed on rank {rank}: {str(e)}")
        raise e
    finally:
        # Cleanup
        if rank == 0:
            wandb.finish()
        cleanup_ddp()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()

    # Get world size from environment variable
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    if world_size > 1:
        # Multi-GPU setup
        import torch.multiprocessing as mp

        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU setup
        main(rank=0, world_size=1, args=args)