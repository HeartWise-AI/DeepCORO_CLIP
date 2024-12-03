import collections
import os
import pathlib
import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torchvision.transforms import v2
from transformers import AutoTokenizer

# Global variable for directory paths
dir2 = os.path.abspath("/volume/DeepCORO_CLIP/orion")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

import orion

from models.model import get_tokenizer


def load_video(
    video_path,
    split="train",
    n_frames=32,
    period=1,
    resize=224,
    apply_mask=False,
    normalize=True,
    mean=0.0,
    std=1.0,
    noise=None,
    pad=None,
    video_transforms=None,
    rand_augment=False,
    backbone="default",
):
    """
    Load and process a video with center cropping and optional augmentations.
    Returns tensor in format [F, H, W, C].
    """
    # Force 16 frames for MViT backbone
    if backbone.lower() == "mvit":
        n_frames = 16
        period = 1

    try:
        # First try loading with Orion
        video = orion.utils.loadvideo(video_path)
        if video is None:
            raise ValueError("Orion loader returned None")
        if isinstance(video, (int, float)):
            raise ValueError("Orion loader returned a scalar instead of video array")
        if len(video.shape) < 3:
            raise ValueError(f"Invalid video shape: {video.shape}")
        video = video.astype(np.float32)
    except Exception as e:
        print(f"Orion loader failed for {video_path}: {str(e)}")
        print("Attempting to load with OpenCV...")

        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from video: {video_path}")

        video = np.stack(frames, axis=0).astype(np.float32)

    # Ensure video has correct number of dimensions
    if len(video.shape) == 3:  # [F, H, W] -> [F, H, W, C]
        video = np.expand_dims(video, axis=-1)
    elif len(video.shape) != 4:  # Must be [F, H, W, C]
        raise ValueError(f"Invalid video shape after loading: {video.shape}")

    # Convert to torch tensor
    video = torch.from_numpy(video)

    # Ensure video is in [F, C, H, W] format for transforms
    if video.shape[-1] in [1, 3]:  # If channel is last dimension
        video = video.permute(0, 3, 1, 2)

    # Resize spatial dimensions
    if resize is not None:
        video = v2.Resize((resize, resize), antialias=True)(video)

    # Handle frame sampling
    t, c, h, w = video.shape

    if backbone.lower() == "mvit":
        # For MViT, always sample exactly 16 frames
        if t < 16:
            # If too few frames, repeat the last frame
            last_frame = video[-1:].repeat(16 - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > 16:
            # If too many frames, sample evenly
            indices = torch.linspace(0, t - 1, 16).long()
            video = video[indices]
    else:
        # For other backbones, maintain original frame count
        target_frames = n_frames
        if t < target_frames:
            # Repeat last frame if needed
            last_frame = video[-1:].repeat(target_frames - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > target_frames:
            indices = torch.linspace(0, t - 1, target_frames).long()
            video = video[indices]

    # Apply normalization with default values if none provided
    if normalize:
        if mean is None or std is None:
            raise ValueError("Mean and standard deviation must be provided for normalization.")

        # Convert scalar values to lists
        if isinstance(mean, (int, float)):
            mean = [float(mean)] * c
        if isinstance(std, (int, float)):
            std = [float(std)] * c

        # Ensure we have the right number of channels
        mean = mean[:c]
        std = std[:c]

        video = v2.Normalize(mean=mean, std=std)(video)

    # Apply transforms
    if video_transforms is not None:
        transforms = v2.RandomApply(torch.nn.ModuleList(video_transforms), p=0.5)
        scripted_transforms = torch.jit.script(transforms)
        try:
            video = scripted_transforms(video)
        except RuntimeError as e:
            print(f"Warning: Error applying transforms to video {video_path}: {str(e)}")

    if rand_augment:
        raug = [v2.RandAugment(magnitude=9, num_layers=2, prob=0.5)]
        raug_composed = v2.Compose(raug)
        video = raug_composed(video)

    # Final verification
    t, c, h, w = video.shape
    expected_frames = 16 if backbone.lower() == "mvit" else n_frames

    if t != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {t}")
    if h != resize or w != resize:
        raise ValueError(f"Expected spatial dimensions {resize}x{resize}, got {h}x{w}")

    # Return video in [F, H, W, C] format
    video = video.permute(0, 2, 3, 1).contiguous()
    return video.numpy()


class Video(torch.utils.data.Dataset):
    """Base dataset class for handling video data."""

    def __init__(
        self,
        root="../../data/",
        data_filename=None,
        split="train",
        target_label=None,
        datapoint_loc_label="FileName",
        resize=224,
        mean=0.0,
        std=1.0,
        length=32,
        period=1,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        video_transforms=None,
        rand_augment=False,
        apply_mask=False,
        target_transform=None,
        external_test_location=None,
        weighted_sampling=False,
        normalize=True,
        debug=False,
    ) -> None:
        # Initialize instance variables
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        if not isinstance(target_label, list):
            target_label = [target_label]
        self.target_label = target_label
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.rand_augment = rand_augment
        self.apply_mask = apply_mask
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.resize = resize
        self.weighted_sampling = weighted_sampling
        self.debug = debug
        self.normalize = normalize

        self.fnames, self.outcome = [], []
        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(os.path.join(self.folder, self.filename)) as f:
                self.header = f.readline().strip().split("α")
                if len(self.header) == 1:
                    raise ValueError(
                        "Header was not split properly. Please ensure the file uses 'α' (alpha) as the delimiter."
                    )

            self.fnames, self.outcomes, target_index = self.load_data(split, target_label)
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

        if self.weighted_sampling is True:
            # define weights for weighted sampling
            labels = np.array(
                [self.outcome[ind][target_index] for ind in range(len(self.outcome))], dtype=int
            )

            # binary weights length == 2
            weights = 1 - (np.bincount(labels) / len(labels))
            self.weight_list = np.zeros(len(labels))

            for label in range(len(weights)):
                weight = weights[label]
                self.weight_list[np.where(labels == label)] = weight

    def load_data(self, split, target_label):
        """Load data from CSV file and filter by split.

        Args:
            split: Which split to load ('train', 'val', or 'all')
            target_label: Column name(s) for target values

        Returns:
            tuple: (fnames, outcomes, target_index)
        """
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.folder, self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        # Print available splits for debugging
        print(f"\nAvailable splits in dataset:")
        print(data["Split"].value_counts())

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        if target_label is None:
            target_index = None
        else:
            target_index = data.columns.get_loc(target_label[0])

        self.fnames = []
        self.outcome = []

        # Track counts for debugging
        total_rows = 0
        valid_files = 0
        split_matches = 0

        # Iterate through rows using iterrows
        for index, row in data.iterrows():
            total_rows += 1
            file_name = row.iloc[filename_index]
            file_mode = str(row.iloc[split_index]).lower().strip()

            if os.path.exists(file_name):
                valid_files += 1
                if split in ["all", file_mode]:
                    split_matches += 1
                    self.fnames.append(file_name)
                    self.outcome.append(row.iloc[target_index])

        # Print debugging information
        print(f"\nDataset loading statistics for split '{split}':")
        print(f"Total rows in CSV: {total_rows}")
        print(f"Valid files found: {valid_files}")
        print(f"Matching split '{split}': {split_matches}")
        print(f"Final dataset size: {len(self.fnames)}")

        if len(self.fnames) == 0:
            raise ValueError(
                f"No samples found for split '{split}'. "
                f"Available splits are: {data['Split'].unique()}. "
                "Please check your data split assignments."
            )

        return self.fnames, self.outcome, target_index

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "external_test":
            video_fname = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video_fname = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_fname = self.fnames[index]

        try:
            # Load and preprocess video
            video = load_video(
                video_fname,
                split=self.split,
                n_frames=self.length,
                period=self.period,
                resize=self.resize,
                apply_mask=self.apply_mask,
                normalize=self.normalize,
                mean=self.mean,
                std=self.std,
                noise=self.noise,
                pad=self.pad,
                video_transforms=self.video_transforms,
                rand_augment=self.rand_augment,
            ).astype(np.float32)

            if self.target_label is not None:
                text = self.outcome[index]
                # Ensure text is a string
                if not isinstance(text, str):
                    text = str(text)

                # Tokenize text with shared configuration
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )

                # Remove batch dimension from tokenizer output and reshape to [sequence_length]
                encoded = {k: v.squeeze(0) for k, v in encoded.items()}

                # Move tensors to device if specified
                if hasattr(self, "device"):
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}

                return video, encoded, video_fname
            return video, None, video_fname

        except Exception as e:
            print(f"Error loading video {video_fname}: {str(e)}")
            # Try the next valid index
            if index + 1 < len(self.fnames):
                return self.__getitem__(index + 1)
            else:
                raise e from None

    def __len__(self):
        return len(self.fnames)


def stats_collate_fn(batch):
    """Custom collate function for StatsDataset that only handles video tensors.

    Args:
        batch: List of video tensors

    Returns:
        Stacked video tensors
    """
    # Filter out None values
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        raise RuntimeError("No valid samples in batch")

    # Extract videos and stack them
    videos = torch.stack([torch.from_numpy(sample[0]) for sample in valid_samples])
    return videos


class StatsDataset(torch.utils.data.Dataset):
    """Dataset class for calculating mean and std statistics."""

    def __init__(
        self,
        root,
        data_filename,
        split,
        target_label,
        datapoint_loc_label="target_video_path",
        num_frames=32,
        backbone="default",
    ):
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        if not isinstance(target_label, list):
            target_label = [target_label]
        self.target_label = target_label
        self.num_frames = 16 if backbone.lower() == "mvit" else num_frames
        self.backbone = backbone

        # Load data
        self.fnames, self.outcome, _ = self.load_data(split, target_label)

    def load_data(self, split, target_label):
        """Load data from CSV file and filter by split."""
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.folder, self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        if target_label is None:
            target_index = None
        else:
            target_index = data.columns.get_loc(target_label[0])

        fnames = []
        outcome = []
        # Iterate through rows using iterrows
        for index, row in data.iterrows():
            file_name = row.iloc[filename_index]
            file_mode = str(row.iloc[split_index]).lower().strip()

            if split in ["all", file_mode] and os.path.exists(file_name):
                fnames.append(file_name)
                if target_index is not None:
                    outcome.append(row.iloc[target_index])

        return fnames, outcome, target_index

    def __getitem__(self, index):
        """Get a single item from the dataset.

        Returns:
            tuple: (video tensor, None, filename) - None is included for compatibility
        """
        video_fname = self.fnames[index]
        try:
            # Load video with proper frame count and backbone settings
            video = load_video(
                video_fname,
                split=self.split,
                n_frames=self.num_frames,
                normalize=False,  # Don't normalize when calculating stats
                backbone=self.backbone,  # Pass backbone parameter
            )
            return video, None, video_fname
        except Exception as e:
            print(f"Error loading video {video_fname}: {str(e)}")
            return None, None, video_fname

    def __len__(self):
        return len(self.fnames)


class VideoDataset(Video):
    """Dataset class for video-text pairs with configurable backbone support."""

    def __init__(
        self,
        root,
        data_filename,
        split,
        target_label,
        datapoint_loc_label="target_video_path",
        num_frames=32,
        backbone="default",
        debug_mode=False,
        normalize=True,
        mean=None,
        std=None,
        **kwargs,
    ):
        # Set default period
        period = kwargs.pop("period", 1)

        # Force 16 frames for MViT backbone
        if backbone.lower() == "mvit":
            num_frames = 16
            period = 1
            print(f"Using MViT backbone - forcing exactly {num_frames} frames with period=1")
            if "length" in kwargs:
                kwargs["length"] = 16

        self.debug_mode = debug_mode
        self.backbone = backbone
        self.num_frames = num_frames
        self.batch_size = kwargs.pop("batch_size", 16)
        self.failed_videos = []

        # Remove normalize from kwargs if it exists to avoid duplicate
        kwargs.pop("normalize", None)

        # Initialize parent class
        super().__init__(
            root=root,
            data_filename=data_filename,
            split=split,
            target_label=target_label,
            datapoint_loc_label=datapoint_loc_label,
            resize=224,
            length=num_frames,
            period=period,
            normalize=normalize,
            mean=mean,
            std=std,
            **kwargs,
        )

        # Initialize tokenizer
        try:
            self.tokenizer = get_tokenizer()
            print("Tokenizer initialized successfully")
        except Exception as e:
            print(f"Error initializing tokenizer: {str(e)}")
            raise RuntimeError("Failed to initialize tokenizer") from e

        # Double-check frame count for MViT
        if self.backbone.lower() == "mvit":
            if self.num_frames != 16 or self.period != 1:
                raise ValueError(
                    f"MViT requires exactly 16 frames with period=1, but got frames={self.num_frames}, period={self.period}"
                )

        # Initialize valid_indices
        if self.debug_mode:
            self._validate_all_videos()
        else:
            self.valid_indices = list(range(len(self.fnames)))

    def _validate_all_videos(self):
        """Pre-validate all videos to catch any issues early."""
        print("Validating all videos in dataset...")
        valid_indices = []

        for idx in range(len(self.fnames)):
            try:
                video_fname = self.fnames[idx]
                # Try loading video without full processing
                video = orion.utils.loadvideo(video_fname)
                if isinstance(video, (int, float)):
                    raise ValueError(f"Invalid video data for {video_fname}")
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Failed to load video {video_fname}: {str(e)}")
                self.failed_videos.append((video_fname, str(e)))

        self.valid_indices = valid_indices
        print(f"Found {len(valid_indices)} valid videos out of {len(self.fnames)}")
        if self.failed_videos:
            print(f"Failed to load {len(self.failed_videos)} videos")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Optional[dict], str]:
        """Get a single item from the dataset.

        Args:
            index: Index of the item to get

        Returns:
            tuple: (video tensor, encoded text dict, file path)

        Raises:
            RuntimeError: If video cannot be loaded
        """
        actual_idx = self.valid_indices[index]
        video_fname = self.fnames[actual_idx]

        try:
            # Load video (returns tensor directly)
            video = load_video(
                video_fname,
                split=self.split,
                n_frames=16 if self.backbone.lower() == "mvit" else self.num_frames,
                period=1 if self.backbone.lower() == "mvit" else self.period,
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

            # Ensure video is in correct format [T, C, H, W]
            if self.backbone.lower() == "mvit":
                if video.shape[0] != 16:
                    raise ValueError(f"Expected 16 frames, got {video.shape[0]}")

            if self.target_label is not None:
                text = self.outcome[actual_idx]
                if not isinstance(text, str):
                    text = str(text)

                try:
                    encoded = self.tokenizer(
                        text,
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoded = {k: v.squeeze(0) for k, v in encoded.items()}

                    if hasattr(self, "device"):
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}

                    return video, encoded, video_fname
                except Exception as e:
                    raise RuntimeError(f"Failed to tokenize text for {video_fname}: {str(e)}")

            return video, None, video_fname

        except Exception as e:
            # Remove the invalid video from the dataset
            self.remove_invalid_video(index)
            raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}")

    def remove_invalid_video(self, index: int):
        """Remove a video from valid_indices if it's found to be invalid during training.

        Args:
            index: Index to remove from valid_indices
        """
        if index in self.valid_indices:
            invalid_path = self.fnames[self.valid_indices[index]]
            self.valid_indices.remove(index)
            print(
                f"Removed invalid video at index {index} ({invalid_path}). "
                f"Remaining valid videos: {len(self.valid_indices)}"
            )

    def __len__(self):
        return len(self.valid_indices)

    def get_reports(self, video_paths):
        """Get report texts for given video paths.

        Args:
            video_paths: List of video file paths

        Returns:
            List of corresponding report texts
        """
        reports = []
        for path in video_paths:
            # Find the index in self.fnames that matches this path
            try:
                idx = self.fnames.index(str(path))
                reports.append(str(self.outcome[idx]))
            except ValueError:
                # If path not found, use empty string
                print(f"Warning: No report found for video {path}")
                reports.append("")
        return reports


def _defaultdict_of_lists():
    """Returns a defaultdict of lists."""
    return collections.defaultdict(list)


def format_mean_std(input_value):
    """Format mean/std values to proper list format."""
    if isinstance(input_value, (int, float)):
        return [float(input_value)]
    return input_value


def custom_collate_fn(
    batch: List[Tuple[np.ndarray, Any, str]]
) -> Tuple[torch.Tensor, dict, List[str]]:
    """Custom collate function to handle video and text data.

    Args:
        batch: List of tuples (video, encoded_text, path)
    Returns:
        videos: Tensor of shape (batch_size, channels, frames, height, width)
        encoded_texts: Dictionary with input_ids and attention_mask tensors
        paths: List of file paths
    """
    videos, encoded_texts, paths = zip(*batch)

    # Stack videos
    videos = torch.stack([torch.from_numpy(v) for v in videos])

    # Combine encoded texts
    if encoded_texts[0] is not None:
        # Stack text tensors along batch dimension
        combined_texts = {
            "input_ids": torch.stack([text["input_ids"] for text in encoded_texts]),
            "attention_mask": torch.stack([text["attention_mask"] for text in encoded_texts]),
        }
    else:
        combined_texts = None

    return videos, combined_texts, paths
