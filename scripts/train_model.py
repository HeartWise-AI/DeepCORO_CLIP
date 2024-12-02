"""Training script for DeepCORO_CLIP model."""

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_processing import get_mean_and_std
from utils.data_processing.video import Video, load_video


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


class StatsDataset(Video):
    """Dataset class for calculating mean and std statistics."""

    def __init__(
        self,
        root,
        data_filename,
        split,
        target_label,
        datapoint_loc_label="target_video_path",
        num_frames=32,
    ):
        super().__init__(
            root=root,
            data_filename=data_filename,
            split=split,
            target_label=target_label,
            datapoint_loc_label=datapoint_loc_label,
            resize=224,
            length=num_frames,
            period=1,
            normalize=False,  # Don't normalize when calculating stats
        )
        self.valid_indices = range(len(self.fnames))


class VideoDataset(Video):
    """Dataset class for video-text pairs."""

    def __init__(
        self,
        root,
        data_filename,
        split,
        target_label,
        datapoint_loc_label="target_video_path",
        num_frames=32,
        **kwargs,
    ):
        # Add debug print to see data file structure
        data_path = os.path.join(root, data_filename)
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, sep="α")
            print(f"Available columns in {data_filename}:")
            print(df.columns.tolist())
        else:
            print(f"Data file not found: {data_path}")

        stats_dataset = None
        # First create a temporary instance without normalization to calculate stats
        if "mean" not in kwargs or "std" not in kwargs:
            print("Creating temporary dataset to calculate mean and std...")
            stats_dataset = StatsDataset(
                root=root,
                data_filename=data_filename,
                split=split,
                target_label=target_label,
                datapoint_loc_label=datapoint_loc_label,
                num_frames=num_frames,
            )

            if len(stats_dataset) == 0:
                raise ValueError("No valid videos found in the dataset!")

            # Sample only 1000 videos max
            if len(stats_dataset) > 1000:
                indices = np.random.choice(len(stats_dataset), 1000, replace=False)
                stats_dataset = torch.utils.data.Subset(stats_dataset, indices)
                print(f"Sampled {len(stats_dataset)} videos for mean and std calculation")
            # Calculate mean and std from the dataset
            print("Calculating dataset mean and std...")
            mean, std = get_mean_and_std(
                dataset=stats_dataset,
                samples=None,  # Use all samples
                batch_size=8,
                num_workers=4,
            )
        else:
            mean = kwargs.pop("mean")
            std = kwargs.pop("std")
        print(f"Dataset mean: {mean}")
        print(f"Dataset std: {std}")

        # Remove normalize from kwargs if it exists to avoid duplicate argument
        kwargs.pop("normalize", None)

        # Now initialize the actual dataset with the calculated statistics
        super().__init__(
            root=root,
            data_filename=data_filename,
            split=split,
            target_label=target_label,
            datapoint_loc_label=datapoint_loc_label,
            resize=224,
            length=num_frames,  # Use the same number of frames
            period=1,
            normalize=True,
            mean=mean,
            std=std,
            **kwargs,
        )

        # Store the calculated statistics
        self.calculated_mean = mean
        self.calculated_std = std
        self.num_frames = num_frames
        # Store valid indices from stats dataset if present else default to full dataset
        if stats_dataset is not None:
            self.valid_indices = getattr(
                stats_dataset, "valid_indices", range(len(stats_dataset))
            )
        else:
            self.valid_indices = range(len(self.fnames))
        print(f"Using {len(self.valid_indices)} valid videos for training")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            use_fast=True,
            model_max_length=512,
            padding_side="right",
            truncation_side="right",
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[dict], str]:
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[index]

        # Find filename of video
        if self.split == "external_test":
            video_fname = os.path.join(self.external_test_location, self.fnames[actual_idx])
        elif self.split == "clinical_test":
            video_fname = os.path.join(
                self.folder, "ProcessedStrainStudyA4c", self.fnames[actual_idx]
            )
        else:
            video_fname = self.fnames[actual_idx]

        try:
            # Load and preprocess video
            video = load_video(video_fname).astype(np.float32)

            # Apply mask if needed
            if self.apply_mask:
                path = video_fname.rsplit("/", 2)
                mask_filename = f"{path[0]}/mask/{path[2]}"
                mask_filename = mask_filename.split(".avi")[0] + ".npy"
                mask = np.load(mask_filename).transpose(2, 0, 1)
                # Apply mask processing...

            # Process video
            video = torch.from_numpy(video)
            if self.transforms is not None:
                video = self.transforms(video)

            # Convert back to numpy for collate function
            video = video.numpy()

            if self.target_label is not None:
                text = self.outcome[actual_idx]
                # Tokenize text
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                return video, encoded, video_fname
            return video, None, video_fname

        except Exception as e:
            print(f"Error loading video {video_fname}: {str(e)}")
            # Try the next valid index
            if index + 1 < len(self.valid_indices):
                return self.__getitem__(index + 1)
            else:
                raise e from None


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
        # Input shape: [batch_size, frames, channels, height, width]
        # Need: [batch_size, channels, frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)

        # Convert grayscale to RGB if needed
        if self.to_rgb is not None:
            # Take only the first channel if input has 3 channels
            if x.shape[1] == 3:
                x = x[:, 0:1]
            x = self.to_rgb(x)

        B, C, F, H, W = x.shape

        # Reshape for MViT: [batch_size, C*F, H, W]
        x = x.reshape(B, C * F, H, W)

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
    """Initialize DDP process group"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group"""
    dist.destroy_process_group()


def train_epoch(video_encoder, text_encoder, dataloader, optimizer, device, rank):
    video_encoder.train()
    text_encoder.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=rank != 0)):
        videos, encoded_texts, _ = batch

        # Skip batch if no text data
        if encoded_texts is None:
            continue

        try:
            # Move data to device
            videos = videos.to(device)
            input_ids = encoded_texts["input_ids"].to(device)
            attention_mask = encoded_texts["attention_mask"].to(device)

            optimizer.zero_grad()

            # Forward pass
            video_emb = video_encoder(videos)
            text_emb = text_encoder(input_ids, attention_mask)

            # Compute loss
            loss = contrastive_loss(video_emb, text_emb)
            loss.backward()
            optimizer.step()

            # Print batch loss every 10 batches (only from rank 0)
            if rank == 0 and (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            total_loss += loss.item()
            num_batches += 1

        except RuntimeError as e:
            if rank == 0:
                print(f"Error in batch {batch_idx + 1}: {str(e)}")
                print(f"Video shape: {videos.shape}")
                if encoded_texts is not None:
                    print(f"Text input_ids shape: {input_ids.shape}")
                    print(f"Text attention_mask shape: {attention_mask.shape}")
            continue

    # Average loss across all processes
    if world_size > 1:
        dist.all_reduce(torch.tensor([total_loss]).to(device))
        dist.all_reduce(torch.tensor([num_batches]).to(device))

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss


def main(rank, world_size):
    """Main training function."""
    # Initialize DDP
    if world_size > 1:
        setup_ddp(rank, world_size)

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(project="deepcoro_clip")

    # Setup
    device = torch.device(f"cuda:{rank}")
    video_encoder = VideoEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    # Wrap models with DDP if using multiple GPUs
    if world_size > 1:
        video_encoder = DDP(video_encoder, device_ids=[rank])
        text_encoder = DDP(text_encoder, device_ids=[rank])

    # Create dataset (mean and std will be calculated automatically)
    train_dataset = VideoDataset(
        root="data/",
        data_filename="processed/reports/reports_sampled_1000.csv",
        split="train",
        target_label="Report",  # Using the report column from your data
        datapoint_loc_label="FileName",  # Using the FileName column from your data
        mean=[0.485, 0.456, 0.406],  # ImageNet mean values
        std=[0.229, 0.224, 0.225],  # ImageNet std values
        normalize=True,
        debug=True,  # Enable debug mode to see more information
    )

    if rank == 0:
        print(f"Dataset size: {len(train_dataset)}")
        print("First few file paths:")
        for i in range(min(5, len(train_dataset.fnames))):
            print(f"  {train_dataset.fnames[i]}")

    # Create distributed sampler
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None

    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        sampler=train_sampler,
    )

    # Training parameters
    num_epochs = 50  # Changed from 60 to 50 epochs
    learning_rate = 1e-4

    # Optimizer
    params = list(video_encoder.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    # Create checkpoint directory
    if rank == 0:
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            video_encoder, text_encoder, train_dataloader, optimizer, device, rank
        )

        if rank == 0:
            wandb.log({"epoch": epoch, "train_loss": train_loss})

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
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
                    "mean": train_dataset.calculated_mean,
                    "std": train_dataset.calculated_std,
                    "datapoint_loc_label": "target_video_path",
                }
                torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    # Set default world size to 2 for 2 GPUs
    world_size = 2

    # Set environment variables for distributed training
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Launch with torch.distributed.launch
    import torch.multiprocessing as mp

    mp.spawn(main, args=(world_size,), nprocs=world_size)
