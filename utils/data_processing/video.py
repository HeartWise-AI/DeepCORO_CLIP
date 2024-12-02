"""Video dataset and related utilities."""

import collections
import os
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torchvision.transforms import v2
from transformers import AutoTokenizer

# Global variable for directory paths
dir2 = os.path.abspath("/volume/DeepCORO_CLIP/orion")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)


def _defaultdict_of_lists() -> collections.defaultdict:
    """Helper function to create defaultdict of lists."""
    return collections.defaultdict(list)


def load_video(filename: str) -> np.ndarray:
    """Load video from file.

    Args:
        filename: Path to video file

    Returns:
        Video as numpy array with shape (frames, channels, height, width)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    file_extension = os.path.splitext(filename)[1]

    if file_extension in [".mp4", ".avi"]:
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_count, frame_width, frame_height, 3), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError(f"Failed to load frame #{count} of {filename}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            v[count] = frame

        vid = v.transpose((0, 3, 1, 2))
        return vid
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def sample_frames(video: np.ndarray, num_frames: int) -> np.ndarray:
    """Sample a fixed number of frames from a video.

    Args:
        video: Video array of shape [frames, channels, height, width]
        num_frames: Number of frames to sample

    Returns:
        Sampled video array of shape [num_frames, channels, height, width]
    """
    total_frames = len(video)

    if total_frames == 0:
        raise ValueError("Video has 0 frames")

    if total_frames < num_frames:
        # If video is too short, loop it
        indices = np.arange(num_frames) % total_frames
    else:
        # If video is too long, sample evenly spaced frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    return video[indices]


class Video(torch.utils.data.Dataset):
    """A dataset class for handling orion video data."""

    def __init__(
        self,
        root: str = "../../data/",
        data_filename: Optional[str] = None,
        split: str = "train",
        target_label: Optional[Union[str, List[str]]] = None,
        datapoint_loc_label: str = "FileName",
        resize: Optional[int] = 224,
        mean: Union[float, List[float], np.ndarray] = 0.0,
        std: Union[float, List[float], np.ndarray] = 1.0,
        length: int = 32,
        period: int = 1,
        max_length: int = 250,
        clips: int = 1,
        pad: Optional[int] = None,
        noise: Optional[float] = None,
        video_transforms: Optional[List] = None,
        rand_augment: bool = False,
        apply_mask: bool = False,
        target_transform: Optional[Callable] = None,
        external_test_location: Optional[str] = None,
        weighted_sampling: bool = False,
        normalize: bool = True,
        debug: bool = False,
    ) -> None:
        """Initialize Video dataset."""
        self.folder: pathlib.Path = pathlib.Path(root)
        self.filename: Optional[str] = data_filename
        self.datapoint_loc_label: str = datapoint_loc_label
        self.split: str = split
        if isinstance(target_label, list):
            self.target_label: List[Optional[str]] = target_label
        else:
            self.target_label: List[Optional[str]] = [target_label]
        self.mean = mean
        self.std = std
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

        # Initialize transforms
        self.transforms = self._init_transforms()

        self.fnames: List[str] = []
        self.outcome: List[Any] = []
        self.frames: Dict[str, List[Any]] = collections.defaultdict(list)
        self.trace: Dict[str, Dict[str, List[Any]]] = collections.defaultdict(
            _defaultdict_of_lists
        )

        self._load_dataset()

    def _init_transforms(self) -> Optional[v2.Compose]:
        """Initialize video transforms."""
        transforms = []
        if self.resize is not None:
            transforms.append(v2.Resize((self.resize, self.resize), antialias=True))
        if self.normalize:
            transforms.append(v2.Normalize(mean=self.mean, std=self.std))
        if self.video_transforms is not None:
            transforms.extend(self.video_transforms)
        if self.rand_augment:
            transforms.append(v2.RandAugment(magnitude=9, num_ops=2))
        return v2.Compose(transforms) if transforms else None

    def _load_dataset(self) -> None:
        """Load dataset based on split type."""
        if self.split == "external_test":
            if self.external_test_location:
                self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            if self.filename and os.path.exists(os.path.join(self.folder, self.filename)):
                self.fnames, self.outcome = self.load_data()

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.fnames)
