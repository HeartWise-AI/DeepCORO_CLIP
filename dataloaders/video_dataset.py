import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import (
    Any, 
    Dict,
    List, 
    Optional, 
    Tuple
)

from utils.seed import seed_worker
from utils.ddp import DistributedUtils
from utils.video import load_video, format_mean_std
from utils.config.heartwise_config import HeartWiseConfig

# DEPRECATED: Use VideoClipDataset instead. This class will be removed in a future release.
class VideoDataset(torch.utils.data.Dataset):
    """
    DEPRECATED: Use VideoClipDataset instead. This class will be removed in a future release.
    Single-video dataset class for video-text pairs.
    """ 

    def __init__(
        self,
        data_filename: str,
        split: str,
        target_label: Optional[List[str]] = None,
        datapoint_loc_label: str = "target_video_path",
        num_frames: int = 32,
        backbone: str = "default",
        debug_mode: bool = False,
        normalize: bool = True,
        mean: Optional[Any] = None,
        std: Optional[Any] = None,
        rand_augment: bool = False,
        stride: int = 1,
        **kwargs,
    ):
        import warnings
        warnings.warn(
            "VideoDataset is deprecated and will be removed in a future release. Use VideoClipDataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.filename: str = data_filename
        self.split: str = split
        self.datapoint_loc_label: str = datapoint_loc_label
        self.debug_mode: bool = debug_mode
        self.backbone: str = backbone
        self.num_frames: int = num_frames
        self.mean: List[float] = format_mean_std(mean)
        self.std: List[float] = format_mean_std(std)
        self.normalize: bool = normalize
        self.rand_augment: bool = rand_augment
        
        self.stride: int = stride
        
        # Define X3D model parameters
        self.x3d_params = {
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            }
        }
        
        # Set specific parameters based on backbone
        if self.backbone.lower() == "mvit":
            self.num_frames = 16
            print(f"Using MViT backbone - forcing exactly {self.num_frames} frames with stride {self.stride}")
            if "length" in kwargs:
                kwargs["length"] = 16

        self.video_transforms: Optional[Any] = kwargs.pop("video_transforms", None)
        self.resize: int = kwargs.pop("resize", 224)
        
        # For X3D models, override resize with their specific side_size
        if self.backbone.lower() in ["x3d_s", "x3d_m"]:
            self.resize = self.x3d_params[self.backbone.lower()]["side_size"]

        self.target_label: Optional[List[str]] = target_label
        self.external_test_location: Optional[str] = kwargs.pop("external_test_location", None)

        self.fnames, self.outcomes, self.target_indexes = self.load_data(
            self.split, self.target_label
        )

        if self.debug_mode:
            self.valid_indices = self._validate_all_videos()
        else:
            self.valid_indices = list(range(len(self.fnames)))

    def __len__(self):
        return len(self.valid_indices)

    def load_data(self, split, target_labels):
        """
        Load data from the CSV file and extract filenames and outcomes.

        Args:
            split (str): Dataset split ('train', 'val', 'test', 'all')
            target_labels (list): List of target label column names

        Returns:
            tuple: (filenames, outcomes, target_indices)
        """
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")

        # Handle target indices for multi-head case
        if target_labels is None:
            target_indices = None
        else:
            target_indices = {}
            for label in target_labels:
                try:
                    target_indices[label] = data.columns.get_loc(label)
                except KeyError:
                    raise ValueError(f"Target label '{label}' not found in data columns")

        fnames = []
        outcomes = []

        # Iterate through rows using iterrows
        for _, row in data.iterrows():
            file_name = row.iloc[filename_index]
            file_mode = row.iloc[split_index].lower()

            # Only process rows matching the split
            if split not in ["all", file_mode]:
                continue

            # Check if the video file exists
            if not os.path.exists(file_name):
                print(f"Skipping video {file_name} because file does not exist.")
                continue

            # If target labels are provided, ensure they are valid
            if target_indices is not None:
                skip_row = False
                row_outcomes = {}
                for label, idx in target_indices.items():
                    value = row.iloc[idx]
                    if pd.isna(value):
                        print(f"Skipping video {file_name} because target '{label}' is missing.")
                        skip_row = True
                        break
                    else:
                        row_outcomes[label] = value
                if skip_row:
                    continue
                outcomes.append(row_outcomes)
            else:
                outcomes.append(None)

            fnames.append(file_name)

        return fnames, outcomes, target_indices

    def _validate_all_videos(self):
        print("Validating all videos in dataset...")
        valid_indices = []
        self.failed_videos = []

        for idx, fname in enumerate(self.fnames):
            try:
                cap = cv2.VideoCapture(fname)
                if not cap.isOpened():
                    raise ValueError(f"Unable to open video {fname}")
                cap.release()
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Failed to load video {fname}: {str(e)}")
                self.failed_videos.append((fname, str(e)))

        print(f"Found {len(valid_indices)} valid videos out of {len(self.fnames)}")
        if self.failed_videos:
            print(f"Failed to load {len(self.failed_videos)} videos")

        return valid_indices

    def __getitem__(self, index: int) -> tuple:
        actual_idx: int = self.valid_indices[index]
        video_fname: str = self.fnames[actual_idx]

        try:
            video: np.ndarray = load_video(
                video_fname,
                n_frames=self.num_frames,
                resize=self.resize,
                normalize=self.normalize,
                mean=self.mean,
                std=self.std,
                video_transforms=self.video_transforms,
                rand_augment=self.rand_augment,
                backbone=self.backbone,
                stride=self.stride,
            )

            # Validate frame count for specific backbone models
            if self.backbone.lower() == "mvit" and video.shape[0] != 16:
                raise ValueError(f"Expected 16 frames for MViT, got {video.shape[0]}")
            elif self.backbone.lower() in ["x3d_s", "x3d_m"] and video.shape[0] != self.x3d_params[self.backbone.lower()]["num_frames"]:
                raise ValueError(f"Expected {self.x3d_params[self.backbone.lower()]['num_frames']} frames for {self.backbone}, got {video.shape[0]}")

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e   
        
        return video, self.outcomes[actual_idx], video_fname
    
def custom_collate_fn(batch: List[Tuple[np.ndarray, Dict[str, torch.Tensor], str]]) -> Dict[str, Any]:
    """Custom collate function to handle video, targets, and video_fname data.

    Args:
        batch: List of tuples (video, targets, video_fname)
        Each video has shape [F, H, W, C]
    Returns:
        videos: Tensor of shape (batch_size, C, F, H, W) for MViT compatibility
        targets: Dictionaries with labels as keys and values as tensors of targets
        video_fname: List of file paths
    """
    videos, targets, paths = zip(*batch)
    # Stack videos - handle both tensor and numpy inputs
    videos = torch.stack([torch.from_numpy(v) for v in videos])  # Shape: [B, F, H, W, C]
    
    # Convert targets to tensor - handle tuple of dictionaries
    targets_dict: dict[str, torch.Tensor] = defaultdict(list)
    for target in targets:
        for k, v in target.items():
            targets_dict[k].append(v)

    targets_dict = {k: torch.tensor(v, dtype=torch.bfloat16) for k, v in targets_dict.items()}
    
    return {
        "videos": videos,
        "targets": targets_dict,
        "video_fname": paths
    }

def get_distributed_video_dataloader(
    config: HeartWiseConfig,
    split: str,
    mean: List[float],
    std: List[float],
    shuffle: bool,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    drop_last: bool = True,
) -> DataLoader:
    # Create the video dataset
    video_dataset = VideoDataset(
        data_filename=config.data_filename,
        split=split,
        target_label=config.target_label,
        datapoint_loc_label=config.datapoint_loc_label,
        num_frames=config.frames,
        backbone=config.model_name,
        mean=mean,
        std=std,
        rand_augment=config.rand_augment,
        stride=config.stride,
    )

    # Create a sampler for distributed training
    sampler = DistributedUtils.DS.DistributedSampler(
        video_dataset, 
        shuffle=shuffle, 
        num_replicas=num_replicas, 
        rank=rank
    )

    # Create the dataloader
    return DataLoader(
        video_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
    )