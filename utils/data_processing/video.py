import os
import pathlib
import sys

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Global variable for directory paths
dir2 = os.path.abspath("/volume/DeepCORO_CLIP/orion")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

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
    Load and process a video with optional resizing, normalization, and augmentations.
    Returns tensor in format [F, H, W, C].
    """
    # Force 16 frames for MViT backbone
    if backbone.lower() == "mvit":
        n_frames = 16
        period = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {video_path}")

    video = np.stack(frames, axis=0).astype(np.float32)

    # Ensure video shape is [F, H, W, C]
    if video.ndim == 3:
        video = np.expand_dims(video, axis=-1)  # [F,H,W] -> [F,H,W,C]
    elif video.ndim != 4:
        raise ValueError(f"Invalid video shape after loading: {video.shape}")

    # Convert to torch tensor [F,H,W,C]
    video = torch.from_numpy(video)

    # Permute to [F,C,H,W]
    if video.shape[-1] in [1, 3]:
        video = video.permute(0, 3, 1, 2)

    # Resize spatial dimensions
    if resize is not None:
        video = v2.Resize((resize, resize), antialias=True)(video)

    t, c, h, w = video.shape

    # Frame sampling
    if backbone.lower() == "mvit":
        # Exactly 16 frames
        if t < 16:
            last_frame = video[-1:].repeat(16 - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > 16:
            indices = torch.linspace(0, t - 1, 16).long()
            video = video[indices]
    else:
        # Keep original frame count to n_frames
        if t < n_frames:
            last_frame = video[-1:].repeat(n_frames - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > n_frames:
            indices = torch.linspace(0, t - 1, n_frames).long()
            video = video[indices]

    if normalize:
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided for normalization.")
        if isinstance(mean, (int, float)):
            mean = [float(mean)] * c
        if isinstance(std, (int, float)):
            std = [float(std)] * c
        mean = mean[:c]
        std = std[:c]
        video = v2.Normalize(mean=mean, std=std)(video)

    # Optional transforms
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

    # Final checks
    t, c, h, w = video.shape
    expected_frames = 16 if backbone.lower() == "mvit" else n_frames
    if t != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {t}")
    if h != resize or w != resize:
        raise ValueError(f"Expected spatial dimensions {resize}x{resize}, got {h}x{w}")

    # Return [F,H,W,C]
    video = video.permute(0, 2, 3, 1).contiguous()
    return video.numpy()


class VideoDataset(torch.utils.data.Dataset):
    """
    Dataset class for video-text pairs. Incorporates logic from the old Video class directly.
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

        # Extract additional parameters
        self.period = kwargs.pop("period", 1)
        if self.backbone.lower() == "mvit":
            self.num_frames = 16
            self.period = 1
            print(f"Using MViT backbone - forcing exactly {self.num_frames} frames with period=1")
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
        target_label = [target_label] if target_label and not isinstance(target_label, list) else target_label
        self.target_label = target_label
        self.target_transform = kwargs.pop("target_transform", None)
        self.external_test_location = kwargs.pop("external_test_location", None)

        # Load data
        self.fnames, self.outcome, self.target_index = self.load_data(self.split, self.target_label)

        # Weighted sampling logic
        if self.weighted_sampling and self.target_label and self.target_index is not None:
            labels = np.array([self.outcome[ind][self.target_index] for ind in range(len(self.outcome))], dtype=int)
            weights = 1 - (np.bincount(labels) / len(labels))
            self.weight_list = np.zeros(len(labels))
            for label_val in range(len(weights)):
                weight = weights[label_val]
                self.weight_list[np.where(labels == label_val)] = weight
        else:
            self.weight_list = None

        # Initialize tokenizer
        try:
            self.tokenizer = get_tokenizer()
            print("Tokenizer initialized successfully")
        except Exception as e:
            print(f"Error initializing tokenizer: {str(e)}")
            raise RuntimeError("Failed to initialize tokenizer") from e

        # Validate videos if debug_mode
        if self.debug_mode:
            self.valid_indices = self._validate_all_videos()
        else:
            self.valid_indices = list(range(len(self.fnames)))

    def load_data(self, split, target_label):
        """Load data from CSV file and filter by split."""
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
        """Pre-validate all videos to catch any issues early."""
        print("Validating all videos in dataset...")
        valid_indices = []
        self.failed_videos = []

        # Note: Remove references to `orion.utils.loadvideo` since it's not defined here.
        # We'll just do a minimal validation by trying to open the video.
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

            # Ensure correct frame count for MViT
            if self.backbone.lower() == "mvit" and video.shape[0] != 16:
                raise ValueError(f"Expected 16 frames for MViT, got {video.shape[0]}")

            encoded = None
            if self.target_label is not None and self.target_index is not None:
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
                except Exception as e:
                    raise RuntimeError(f"Failed to tokenize text for {video_fname}: {str(e)}")

            return video, encoded, video_fname

        except Exception as e:
            self.remove_invalid_video(index)
            raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e

    def remove_invalid_video(self, index: int):
        """Remove a video from valid_indices if it's found invalid."""
        if index in self.valid_indices:
            invalid_path = self.fnames[self.valid_indices[index]]
            self.valid_indices.remove(index)
            print(
                f"Removed invalid video at index {index} ({invalid_path}). "
                f"Remaining valid videos: {len(self.valid_indices)}"
            )

    def __len__(self):
        return len(self.valid_indices)

    def get_reports(self, video_paths: List[str]) -> List[str]:
        """Get report texts for given video paths."""
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
        Return all reports (text outcomes) from the dataset as a list of strings.
        """
        return [str(o) for o in self.outcome]



class SimpleTextDataset(Dataset):
    """Allow me to encode all reportsi n the valdiation dataset at once for validation metrics"""
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
        # Squeeze to remove batch dimension
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

class StatsDataset(torch.utils.data.Dataset):
    """Dataset class for calculating mean and std statistics without the Video base class."""

    def __init__(
        self,
        root,
        data_filename,
        split,
        target_label,
        datapoint_loc_label="target_video_path",
        num_frames=32,
        backbone="default",
        max_samples=128,
    ):
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        self.num_frames = 16 if backbone.lower() == "mvit" else num_frames
        self.backbone = backbone
        self.max_samples = max_samples

        if target_label and not isinstance(target_label, list):
            target_label = [target_label]
        self.target_label = target_label

        self.fnames, self.outcome, _ = self.load_data(split, target_label)
        if self.max_samples and len(self.fnames) > self.max_samples:
            self.fnames = self.fnames[:self.max_samples]
            if self.outcome:
                self.outcome = self.outcome[:self.max_samples]
            print(f"Limited dataset to {self.max_samples} samples")

    def load_data(self, split, target_label):
        file_path = os.path.join(self.folder, self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        target_index = None
        if target_label is not None:
            target_index = data.columns.get_loc(target_label[0])

        fnames = []
        outcome = []
        for _, row in data.iterrows():
            file_name = row.iloc[filename_index]
            file_mode = str(row.iloc[split_index]).lower().strip()

            if split in ["all", file_mode] and os.path.exists(file_name):
                fnames.append(file_name)
                if target_index is not None:
                    outcome.append(row.iloc[target_index])

                if self.max_samples and len(fnames) >= self.max_samples:
                    return fnames, outcome, target_index

        return fnames, outcome, target_index

    def __getitem__(self, index):
        video_fname = self.fnames[index]
        try:
            video = load_video(
                video_fname,
                split=self.split,
                n_frames=self.num_frames,
                normalize=False,
                backbone=self.backbone,
            )
            return video, None, video_fname
        except Exception as e:
            print(f"Error loading video {video_fname}: {str(e)}")
            return None, None, video_fname

    def __len__(self):
        return len(self.fnames)


def stats_collate_fn(batch):
    """Collate function for StatsDataset that stacks video tensors."""
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        raise RuntimeError("No valid samples in batch")
    videos = torch.stack([torch.from_numpy(sample[0]) for sample in valid_samples])
    return videos


def format_mean_std(input_value):
    """Format mean/std values to proper list format."""
    if isinstance(input_value, (int, float)):
        return [float(input_value)]
    return input_value


def custom_collate_fn(
    batch: List[Tuple[np.ndarray, Any, str]]
) -> Tuple[torch.Tensor, dict, List[str]]:
    """Custom collate function to handle video and text data."""
    videos, encoded_texts, paths = zip(*batch)

    videos = torch.stack([torch.from_numpy(v) for v in videos])

    if encoded_texts[0] is not None:
        combined_texts = {
            "input_ids": torch.stack([text["input_ids"] for text in encoded_texts]),
            "attention_mask": torch.stack([text["attention_mask"] for text in encoded_texts]),
        }
    else:
        combined_texts = None

    return videos, combined_texts, paths


def convert_to_mp4(input_path):
    """Convert video to MP4 format for wandb logging with reduced size."""
    import subprocess
    import tempfile

    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "mpeg4",
                "-vf", "scale=320:-1",
                "-r", "15",
                "-y", temp_path,
            ],
            check=True,
            capture_output=True,
        )
        return temp_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to convert video {input_path} to MP4: {e.stderr.decode()}")
        os.unlink(temp_path)
        return None
