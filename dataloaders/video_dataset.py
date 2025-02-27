import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
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

class VideoDataset(torch.utils.data.Dataset):
    """
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
        **kwargs,
    ):
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
        
        self.stride: int = kwargs.pop("stride", 1)
        if self.backbone.lower() == "mvit":
            self.num_frames = 16
            print(f"Using MViT backbone - forcing exactly {self.num_frames} frames with stride {self.stride}")
            if "length" in kwargs:
                kwargs["length"] = 16

        self.video_transforms: Optional[Any] = kwargs.pop("video_transforms", None)
        self.resize: int = kwargs.pop("resize", 224)

        self.target_label: Optional[List[str]] = target_label
        self.external_test_location: Optional[str] = kwargs.pop("external_test_location", None)

        self.fnames, self.outcome, self.target_index = self.load_data(
            self.split, self.target_label
        )

        if self.debug_mode:
            self.valid_indices = self._validate_all_videos()
        else:
            self.valid_indices = list(range(len(self.fnames)))

    def __len__(self):
        return len(self.valid_indices)

    def load_data(
        self, 
        split: str, 
        target_label: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], Optional[int]]:
        file_path: str = os.path.join(self.filename)
        data: pd.DataFrame = pd.read_csv(file_path, sep="α", engine="python")

        print(f"\nAvailable splits in dataset:")
        print(data["Split"].value_counts())

        filename_index: int = data.columns.get_loc(self.datapoint_loc_label)
        split_index: int = data.columns.get_loc("Split")

        target_index = None
        if target_label is not None:
            target_index = data.columns.get_loc(target_label[0])

        fnames: List[str] = []
        outcome: List[str] = []

        total_rows: int = 0
        valid_files: int = 0
        split_matches: int = 0

        for _, row in data.iterrows():
            total_rows += 1
            file_name: str = str(row.iloc[filename_index])
            file_mode: str = str(row.iloc[split_index]).lower().strip()

            if self.external_test_location and self.split == "external_test":
                full_path: str = os.path.join(self.external_test_location, file_name)
            else:
                full_path: str = file_name

            if os.path.exists(full_path):
                valid_files += 1
                if split in ["all", file_mode]:
                    split_matches += 1
                    fnames.append(full_path)
                    if target_index is not None:
                        outcome.append(row.iloc[target_index])

        print(f"\nDataset loading statistics for split '{split}':")
        print(f"Total rows in CSV: {total_rows}")
        print(f"Valid files found: {valid_files}")
        print(f"Matching split '{split}': {split_matches}")
        print(f"Final dataset size: {len(fnames)}")

        if len(fnames) == 0:
            raise ValueError(
                f"No samples found for split '{split}'. "
                f"Available splits: {data['Split'].unique()}. "
                "Check your data split assignments."
            )

        return fnames, outcome, target_index

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
                n_frames=16 if self.backbone.lower() == "mvit" else self.num_frames,
                resize=self.resize,
                normalize=self.normalize,
                mean=self.mean,
                std=self.std,
                video_transforms=self.video_transforms,
                rand_augment=self.rand_augment,
                backbone=self.backbone,
            )

            if self.backbone.lower() == "mvit" and video.shape[0] != 16:
                raise ValueError(f"Expected 16 frames for MViT, got {video.shape[0]}")

            targets: Dict[str, Any] = {}
            if self.target_label is not None:
                for label in self.target_label:
                    targets[label] = self.outcome[actual_idx]

            return video, targets, video_fname

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e   
    
# def custom_collate_fn(batch: List[Tuple[np.ndarray, Dict[str, Any], str]]) -> Dict[str, Any]:
#     """Custom collate function to handle video and text data.

#     Args:
#         batch: List of tuples (video, encoded_text, path)
#         Each video has shape [F, H, W, C]
#     Returns:
#         videos: Tensor of shape (batch_size, C, F, H, W) for MViT compatibility
#         encoded_texts: Dictionary with input_ids and attention_mask tensors
#         paths: List of file paths
#     """
#     videos, targets, paths = zip(*batch)

#     # Stack videos - handle both tensor and numpy inputs
#     videos = torch.stack([torch.from_numpy(v) for v in videos])  # Shape: [B, F, H, W, C]

#     return {
#         "videos": videos,
#         "targets": targets,
#         "paths": paths
#     }

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
        # collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
    )