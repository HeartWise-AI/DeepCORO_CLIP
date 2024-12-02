import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import timm
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import sys
import os
import numpy as np
from torchvision.transforms import v2
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_processing.video import Video
from utils.data_processing import loadvideo, get_mean_and_std 

def custom_collate_fn(batch):
    """Custom collate function to handle video and text data.
    
    Args:
        batch: List of tuples (video, text, path)
    Returns:
        videos: Tensor of shape (batch_size, channels, frames, height, width)
        texts: List of text strings
        paths: List of file paths
    """
    videos, texts, paths = zip(*batch)
    
    # Stack videos - they should all have the same shape after preprocessing
    videos = torch.stack([torch.from_numpy(v) for v in videos])
    
    return videos, texts, paths

class StatsDataset(Video):
    """Special dataset class just for calculating mean and std, ignoring text data"""
    def __init__(self, root, data_filename, split, target_label, datapoint_loc_label="target_video_path", num_frames=32):
        super().__init__(
            root=root,
            data_filename=data_filename,
            split=split,
            target_label=target_label,
            datapoint_loc_label=datapoint_loc_label,
            resize=224,
            length=32,
            period=1,
            normalize=False
        )
        self.num_frames = num_frames
        # Keep track of valid videos
        self.valid_indices = []
        self._validate_videos()
    
    def _validate_videos(self):
        """Pre-validate all videos and store valid indices"""
        print("Pre-validating videos...")
        for idx in tqdm(range(len(self.fnames))):
            try:
                video_fname = self.fnames[idx]
                video = loadvideo(video_fname)
                # Accept both grayscale and RGB videos
                if video is not None and video.size > 0 and len(video.shape) >= 3:
                    if len(video.shape) == 3:  # [frames, height, width]
                        video = video[..., np.newaxis]  # Add channel dimension
                    elif len(video.shape) == 4 and video.shape[1] == 3:  # [frames, channels, height, width]
                        video = video[:, 0:1]  # Take first channel for grayscale
                    self.valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping invalid video {video_fname}: {str(e)}")
        print(f"Found {len(self.valid_indices)} valid videos out of {len(self.fnames)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[index]
        video_fname = self.fnames[actual_idx]
        
        try:
            # Load video
            video = loadvideo(video_fname).astype(np.float32)
            
            # Handle grayscale videos
            if len(video.shape) == 3:  # [frames, height, width]
                video = video[..., np.newaxis]  # Add channel dimension: [frames, height, width, 1]
            elif len(video.shape) == 4 and video.shape[1] == 3:  # [frames, channels, height, width]
                video = video[:, 0:1]  # Take first channel for grayscale
            
            # Ensure consistent number of frames through temporal sampling
            total_frames = video.shape[0]
            if total_frames == 0:
                raise ValueError("Video has 0 frames")
                
            if total_frames < self.num_frames:
                # If video is too short, loop it
                indices = np.arange(self.num_frames) % total_frames
            else:
                # If video is too long, sample evenly spaced frames
                indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            video = video[indices]
            
            # Convert to tensor and resize
            video = torch.from_numpy(video)
            if self.resize is not None:
                video = v2.Resize((self.resize, self.resize), antialias=True)(video)
            
            # Ensure channel dimension is 1
            if video.shape[1] == 3:
                video = video[:, 0:1]
            
            return video
            
        except Exception as e:
            print(f"Error loading video {video_fname}: {str(e)}")
            # Try the next valid index
            if index + 1 < len(self.valid_indices):
                return self.__getitem__(index + 1)
            else:
                raise e

class VideoDataset(Video):
    def __init__(self, root, data_filename, split, target_label, datapoint_loc_label="target_video_path", num_frames=32, **kwargs):
        """
        Args:
            root (str): Root directory for data
            data_filename (str): Name of data file
            split (str): 'train', 'val', or 'test'
            target_label (str): Column name for target labels
            datapoint_loc_label (str): Column name for video file paths. Defaults to "target_video_path"
            num_frames (int): Number of frames to sample from each video
            **kwargs: Additional arguments passed to Video class
        """
        stats_dataset = None
        # First create a temporary instance without normalization to calculate stats
        if 'mean' not in kwargs or 'std' not in kwargs:
            print("Creating temporary dataset to calculate mean and std...")
            stats_dataset = StatsDataset(
                root=root,
                data_filename=data_filename,
                split=split,
                target_label=target_label,
                datapoint_loc_label=datapoint_loc_label,
                num_frames=num_frames
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
                num_workers=4
            )
        else:
            mean = kwargs.pop('mean')
            std = kwargs.pop('std')
        print(f"Dataset mean: {mean}")
        print(f"Dataset std: {std}")
        
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
            **kwargs
        )
        
        # Store the calculated statistics
        self.calculated_mean = mean
        self.calculated_std = std
        self.num_frames = num_frames
        # Store valid indices from stats dataset if present else default to full dataset
        if stats_dataset is not None:
            self.valid_indices = getattr(stats_dataset, 'valid_indices', range(len(stats_dataset)))
        else:
            self.valid_indices = range(len(self.fnames))
        print(f"Using {len(self.valid_indices)} valid videos for training")
        # Initialize tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            use_fast=True,  # Use fast tokenizer
            model_max_length=512,  # Set maximum length
            padding_side='right',  # Consistent padding
            truncation_side='right'  # Consistent truncation
        )
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[index]
        
        # Find filename of video
        if self.split == "external_test":
            video_fname = os.path.join(self.external_test_location, self.fnames[actual_idx])
        elif self.split == "clinical_test":
            video_fname = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[actual_idx])
        else:
            video_fname = self.fnames[actual_idx]
        
        #print(f"Loading video from {video_fname}")
        try:
            # Load and preprocess video
            video = loadvideo(video_fname).astype(np.float32)
            
            # Ensure video is not empty and has correct shape
            if video.size == 0 or len(video.shape) != 4:
                raise ValueError(f"Invalid video shape: {video.shape}")
            
            # Ensure consistent number of frames through temporal sampling
            total_frames = video.shape[0]
            if total_frames == 0:
                raise ValueError("Video has 0 frames")
                
            if total_frames < self.num_frames:
                # If video is too short, loop it
                indices = np.arange(self.num_frames) % total_frames
            else:
                # If video is too long, sample evenly spaced frames
                indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            video = video[indices]
            
            # Process video through transformations
            video = torch.from_numpy(video)
            if self.resize is not None:
                video = v2.Resize((self.resize, self.resize), antialias=True)(video)
            if self.normalize and hasattr(self, "mean") and hasattr(self, "std"):
                video = v2.Normalize(self.mean, self.std)(video)
            
            # Convert back to numpy for collate function
            video = video.numpy()
            
            if self.target_label is not None:
                text = self.outcome[actual_idx]
                # Tokenize text
                encoded = self.tokenizer(
                    text,
                    padding='max_length',
                    max_length=512,  # Increased for longer medical texts
                    truncation=True,
                    return_tensors='pt'
                )
                return video, encoded, video_fname
            return video, None, video_fname
            
        except Exception as e:
            print(f"Error loading video {video_fname}: {str(e)}")
            # Try the next valid index
            if index + 1 < len(self.valid_indices):
                return self.__getitem__(index + 1)
            else:
                raise e


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
                nn.ReLU(inplace=True)
            )
        else:
            self.to_rgb = None
        
        # MViT outputs 512 dimensions directly to match text encoder
        self.mvit = timm.create_model('mvitv2_base', pretrained=True, num_classes=512)
        
        # Modify first conv layer to accept video input
        first_conv = self.mvit.patch_embed.proj
        self.mvit.patch_embed.proj = nn.Conv2d(
            3 * 32,  # 3 channels * num_frames
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding
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
        self.bert = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.proj = nn.Linear(768, 512)
        
    def forward(self, input_ids, attention_mask):
        features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.proj(features)

def contrastive_loss(video_emb, text_emb, temperature=0.07):
    """Compute contrastive loss between video and text embeddings."""
    video_emb = nn.functional.normalize(video_emb, dim=1)
    text_emb = nn.functional.normalize(text_emb, dim=1)
    
    logits = torch.matmul(video_emb, text_emb.t()) / temperature
    labels = torch.arange(len(video_emb)).to(video_emb.device)
    
    loss = (nn.CrossEntropyLoss()(logits, labels) + 
            nn.CrossEntropyLoss()(logits.t(), labels)) / 2
    return loss

def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
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
        
        # Move data to device
        videos = videos.to(device)
        input_ids = encoded_texts['input_ids'].squeeze(1).to(device)
        attention_mask = encoded_texts['attention_mask'].squeeze(1).to(device)
        
        try:
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
            continue
    
    # Average loss across all processes
    if world_size > 1:
        dist.all_reduce(torch.tensor([total_loss]).to(device))
        dist.all_reduce(torch.tensor([num_batches]).to(device))
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

def main(rank, world_size):
    # Initialize DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(project="deepcoro_clip")
    
    # Setup
    device = torch.device(f'cuda:{rank}')
    video_encoder = VideoEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    
    # Wrap models with DDP if using multiple GPUs
    if world_size > 1:
        video_encoder = DDP(video_encoder, device_ids=[rank])
        text_encoder = DDP(text_encoder, device_ids=[rank])
    
    # Create dataset (mean and std will be calculated automatically)
    train_dataset = VideoDataset(
        root="../../data/",
        data_filename="train_data.txt",
        split="train",
        target_label="text",
        datapoint_loc_label="target_video_path"
    )
    
    if rank == 0:
        print(f"Using calculated mean: {train_dataset.calculated_mean}")
        print(f"Using calculated std: {train_dataset.calculated_std}")
    
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
        sampler=train_sampler
    )
    
    # Training parameters
    num_epochs = 60
    learning_rate = 1e-4
    
    # Optimizer
    params = list(video_encoder.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    
    # Create checkpoint directory
    if rank == 0:
        checkpoint_dir = Path('models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        train_loss = train_epoch(video_encoder, text_encoder, train_dataloader, 
                               optimizer, device, rank)
        
        if rank == 0:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss
            })
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'video_encoder': video_encoder.module.state_dict() if world_size > 1 else video_encoder.state_dict(),
                    'text_encoder': text_encoder.module.state_dict() if world_size > 1 else text_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'mean': train_dataset.calculated_mean,
                    'std': train_dataset.calculated_std,
                    'datapoint_loc_label': 'target_video_path'
                }
                torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()

if __name__ == '__main__':
    # Get world size from environment variable
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        # Launch with torch.distributed.launch
        import torch.multiprocessing as mp
        mp.spawn(main, args=(world_size,), nprocs=world_size)
    else:
        # Single GPU or CPU training
        main(0, 1) 