import collections
import os
import pathlib
import sys
from typing import Optional

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
):
    """
    Load and process a video with center cropping and optional augmentations.

    Args:
        video_path (str): Path to the video file
        split (str): Dataset split ('train', 'val', 'test', 'external_test', 'clinical_test')
        n_frames (int): Number of frames to extract
        period (int): Sampling period for frames
        resize (int): Size to resize frames to
        apply_mask (bool): Whether to apply masking
        normalize (bool): Whether to normalize the video
        mean (float or list): Mean for normalization (per channel)
        std (float or list): Standard deviation for normalization (per channel)
        noise (float): Fraction of pixels to black out
        pad (int): Padding size
        video_transforms (list): List of video transforms
        rand_augment (bool): Whether to apply random augmentation

    Returns:
        np.ndarray: Processed video array
    """
    try:
        # First try loading with Orion
        video = orion.utils.loadvideo(video_path)
        if isinstance(video, (int, float)):
            raise ValueError("Orion loader returned a scalar instead of video array")
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
            # Convert BGR to grayscale if it's color
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from video: {video_path}")

        # Stack frames into array [num_frames, height, width]
        video = np.stack(frames, axis=0).astype(np.float32)
        # Add channel dimension [num_frames, channels, height, width]
        video = np.expand_dims(video, axis=1)

    # Convert mean and std to lists if they're not already
    if isinstance(mean, (int, float)):
        mean = [float(mean)]
    if isinstance(std, (int, float)):
        std = [float(std)]

    # Handle masking if required
    if apply_mask:
        path = video_path.rsplit("/", 2)
        mask_filename = f"{path[0]}/mask/{path[2]}"
        mask_filename = mask_filename.split(".avi")[0] + ".npy"

        if os.path.exists(mask_filename):
            mask = np.load(mask_filename).transpose(2, 0, 1)
            length = video.shape[2]

            # Fix mask shapes
            if mask.shape[1] < length:
                mask = np.pad(mask, [(0, 0), (length - mask.shape[1], 0), (0, 0)])
            if mask.shape[2] < length:
                mask = np.pad(mask, [(0, 0), (0, 0), (length - mask.shape[2], 0)])
            if mask.shape[1] > length:
                mask = mask[:, :length, :]
            if mask.shape[2] > length:
                mask = mask[:, :, :length]

            # Apply mask to each frame
            for ind in range(video.shape[0]):
                video[ind, :, :, :] = video[ind, :, :, :] * mask

    # Add noise if specified
    if noise is not None:
        n = video.shape[1] * video.shape[2] * video.shape[3]
        ind = np.random.choice(n, round(noise * n), replace=False)
        f = ind % video.shape[1]
        ind //= video.shape[1]
        i = ind % video.shape[2]
        ind //= video.shape[2]
        j = ind
        video[:, f, i, j] = 0

    # Convert to torch tensor for transforms
    video = torch.from_numpy(video)

    # Resize if specified
    if resize is not None:
        video = v2.Resize((resize, resize), antialias=True)(video)

    # Apply normalization if specified
    if normalize:
        video = v2.Normalize(mean=mean, std=std)(video)

    # Apply video transforms if specified
    if video_transforms is not None:
        transforms = v2.RandomApply(torch.nn.ModuleList(video_transforms), p=0.5)
        scripted_transforms = torch.jit.script(transforms)
        try:
            video = scripted_transforms(video)
        except RuntimeError as e:
            print(f"Error applying transforms to video {video_path}: {str(e)}")

    # Apply random augmentation if specified
    if rand_augment:
        raug = [v2.RandAugment(magnitude=9, num_layers=2, prob=0.5)]
        raug_composed = v2.Compose(raug)
        video = raug_composed(video)

    # Convert back to numpy and handle frame ordering
    video = video.permute(1, 0, 2, 3).numpy()

    # Center crop/pad frames
    c, f, h, w = video.shape
    target_frames = n_frames * period

    if f < target_frames:
        # Pad if video is too short
        padding = target_frames - f
        pad_left = padding // 2
        pad_right = padding - pad_left
        video = np.pad(video, ((0, 0), (pad_left, pad_right), (0, 0), (0, 0)), mode="constant")
    else:
        # Center crop
        start_frame = (f - target_frames) // 2
        video = video[:, start_frame : start_frame + target_frames : period, :, :]

    # Add padding if specified
    if pad is not None:
        c, l, h, w = video.shape
        temp = np.zeros((c, l, h + 2 * pad, w + 2 * pad), dtype=video.dtype)
        temp[:, :, pad:-pad, pad:-pad] = video
        i, j = np.random.randint(0, 2 * pad, 2)
        video = temp[:, :, i : (i + h), j : (j + w)]

    return video


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
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.folder, self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        if target_label is None:
            target_index = None
        else:
            target_index = data.columns.get_loc(target_label[0])

        self.fnames = []
        self.outcome = []
        # Iterate through rows using iterrows
        for index, row in data.iterrows():
            file_name = row.iloc[filename_index]
            file_mode = row.iloc[split_index].lower()

            if split in ["all", file_mode] and os.path.exists(file_name):
                self.fnames.append(file_name)
                self.outcome.append(row.iloc[target_index])

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
            if index + 1 < len(self.fnames):
                return self.__getitem__(index + 1)
            else:
                raise e from None

    def __len__(self):
        return len(self.fnames)


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

        # Convert mean and std to lists if they're not already
        if isinstance(mean, (int, float)):
            mean = [float(mean)]
        if isinstance(std, (int, float)):
            std = [float(std)]

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

    def __getitem__(self, index: int) -> tuple[np.ndarray, Optional[dict], str]:
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[index]

        # Find filename of video
        video_fname = self.fnames[actual_idx]

        try:
            # Load and preprocess video
            video = load_video(
                video_fname,
                split=self.split,
                n_frames=self.num_frames,
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


def _defaultdict_of_lists():
    """Returns a defaultdict of lists."""
    return collections.defaultdict(list)


def format_mean_std(input_value):
    """Format mean/std values to proper list format."""
    if isinstance(input_value, (int, float)):
        return [float(input_value)]
    return input_value
