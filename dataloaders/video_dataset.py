import os
import pathlib
from typing import Any, List, Optional

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils.ddp import DS
from utils.seed import seed_worker
from utils.config import HeartWiseConfig
from models.text_encoder import get_tokenizer
from utils.video import load_video, format_mean_std


class VideoDataset(torch.utils.data.Dataset):
    """
    Single-video dataset class for video-text pairs.
    """

    def __init__(
        self,
        root: str,
        data_filename: str,
        split: str,
        target_label: Optional[str],
        datapoint_loc_label: str = "target_video_path",
        num_frames: int = 32,
        backbone: str = "default",
        debug_mode: bool = False,
        normalize: bool = True,
        mean: Optional[Any] = None,
        std: Optional[Any] = None,
        **kwargs,
    ):
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.split = split
        self.datapoint_loc_label = datapoint_loc_label
        self.debug_mode = debug_mode
        self.backbone = backbone
        self.num_frames = num_frames
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.normalize = normalize

        self.stride = kwargs.pop("stride", 1)
        if self.backbone.lower() == "mvit":
            self.num_frames = 16
            self.stride = 1
            print(f"Using MViT backbone - forcing exactly {self.num_frames} frames with stride=1")
            if "length" in kwargs:
                kwargs["length"] = 16

        self.apply_mask = kwargs.pop("apply_mask", False)
        self.video_transforms = kwargs.pop("video_transforms", None)
        self.rand_augment = kwargs.pop("rand_augment", False)
        self.resize = kwargs.pop("resize", 224)
        self.pad = kwargs.pop("pad", None)
        self.noise = kwargs.pop("noise", None)
        self.weighted_sampling = kwargs.pop("weighted_sampling", False)
        self.max_length = kwargs.pop("max_length", 250)
        self.clips = kwargs.pop("clips", 1)

        target_label = (
            [target_label]
            if target_label and not isinstance(target_label, list)
            else target_label
        )
        self.target_label = target_label
        self.target_transform = kwargs.pop("target_transform", None)
        self.external_test_location = kwargs.pop("external_test_location", None)

        self.fnames, self.outcome, self.target_index = self.load_data(
            self.split, self.target_label
        )

        # Initialize tokenizer
        try:
            self.tokenizer = get_tokenizer()
            print("Tokenizer initialized successfully")
        except Exception as e:
            print(f"Error initializing tokenizer: {str(e)}")
            raise RuntimeError("Failed to initialize tokenizer") from e

        if self.debug_mode:
            self.valid_indices = self._validate_all_videos()
        else:
            self.valid_indices = list(range(len(self.fnames)))

    def __len__(self):
        return len(self.valid_indices)

    def load_data(self, split, target_label):
        file_path = os.path.join(self.folder, self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        print(f"\nAvailable splits in dataset:")
        print(data["Split"].value_counts())

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")

        target_index = None
        if target_label is not None:
            target_index = data.columns.get_loc(target_label[0])

        fnames = []
        outcome = []

        total_rows = 0
        valid_files = 0
        split_matches = 0

        for _, row in data.iterrows():
            total_rows += 1
            file_name = row.iloc[filename_index]
            file_mode = str(row.iloc[split_index]).lower().strip()

            if self.external_test_location and self.split == "external_test":
                full_path = os.path.join(self.external_test_location, file_name)
            else:
                full_path = file_name

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
        actual_idx = self.valid_indices[index]
        video_fname = self.fnames[actual_idx]

        try:
            video = load_video(
                video_fname,
                split=self.split,
                n_frames=16 if self.backbone.lower() == "mvit" else self.num_frames,
                stride=self.stride,
                resize=self.resize,
                apply_mask=self.apply_mask,
                normalize=self.normalize,
                mean=self.mean,
                std=self.std,
                noise=self.noise,
                pad=self.pad,
                video_transforms=self.video_transforms,
                rand_augment=self.rand_augment,
                backbone=self.backbone,
            )

            if self.backbone.lower() == "mvit" and video.shape[0] != 16:
                raise ValueError(f"Expected 16 frames for MViT, got {video.shape[0]}")

            encoded = None
            if self.target_label is not None and self.target_index is not None:
                text = self.outcome[actual_idx]
                if not isinstance(text, str):
                    text = str(text)
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.squeeze(0) for k, v in encoded.items()}
                # If you want them on GPU, do it in the training loop, not here
            else:
                print("No target label or target index")
                encoded = None

            return video, encoded, video_fname

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e

    def get_reports(self, video_paths: List[str]) -> List[str]:
        """
        Given a list of video file paths, return a list of text reports (strings).
        """
        reports = []
        for path in video_paths:
            try:
                idx = self.fnames.index(str(path))
                reports.append(str(self.outcome[idx]))
            except ValueError:
                print(f"Warning: No report found for video {path}")
                reports.append("")
        return reports

    def get_all_reports(self):
        """
        Return all text outcomes from the dataset.
        """
        return [str(o) for o in self.outcome]
    
    
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

    # Combine encoded texts
    if encoded_texts[0] is not None:
        combined_texts = {
            "input_ids": torch.stack([text["input_ids"] for text in encoded_texts]),
            "attention_mask": torch.stack([text["attention_mask"] for text in encoded_texts]),
        }
    else:
        combined_texts = None

    return {
        "videos": videos,
        "encoded_texts": combined_texts,
        "paths": paths
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
        root=config.root,
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
    sampler = DS.DistributedSampler(
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