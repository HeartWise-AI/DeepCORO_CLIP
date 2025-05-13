import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import (
    Any, 
    Dict,
    List, 
    Optional, 
    Tuple,
    Union
)

from utils.seed import seed_worker
from utils.ddp import DistributedUtils
from utils.video import load_video, format_mean_std
from utils.config.heartwise_config import HeartWiseConfig

class VideoDataset(torch.utils.data.Dataset):
    """Single or multi-video dataset class for video-text pairs.""" 

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
        resize: int = 224,
        multi_video: bool = False,
        groupby_column: str = "StudyInstanceUID",
        num_videos: int = 1,
        shuffle_videos: bool = True,
        **kwargs,
    ):
        self.filename: str = data_filename
        self.split: str = split
        self.datapoint_loc_label: str = datapoint_loc_label
        self.debug_mode: bool = debug_mode
        self.backbone: str = backbone
        
        # Multi-video parameters
        self.multi_video: bool = multi_video
        self.groupby_column: str = groupby_column
        self.num_videos: int = num_videos
        self.shuffle_videos: bool = shuffle_videos
        
        # Initialize general versions first, potentially overridden by backbone specifics
        self.num_frames: int = num_frames
        self.resize: int = resize 

        self.mean: List[float] = format_mean_std(mean) if mean is not None else [0.485, 0.456, 0.406]
        self.std: List[float] = format_mean_std(std) if std is not None else [0.229, 0.224, 0.225]
        self.normalize: bool = normalize
        self.rand_augment: bool = rand_augment
        self.stride: int = stride
        
        # Define X3D model specific parameters (requirements)
        self.x3d_default_params = {
            "x3d_s": {"side_size": 182, "crop_size": 182, "num_frames": self.num_frames, "sampling_rate": 6},
            "x3d_m": {"side_size": self.resize, "crop_size": self.resize, "num_frames": self.num_frames, "sampling_rate": 5},
        }
        
        # Adjust num_frames and resize based on backbone requirements
        if self.backbone.lower() == "mvit":
            self.num_frames = 16 # MViT specific requirement
            print(f"Using MViT backbone - forcing {self.num_frames} frames.")
        elif self.backbone.lower() in self.x3d_default_params:
            params = self.x3d_default_params[self.backbone.lower()]
            self.resize = params["side_size"]    # X3D specific resize
            self.num_frames = params["num_frames"] # X3D specific num_frames
            print(f"Using {self.backbone} backbone - forcing resize to {self.resize} and {self.num_frames} frames.")

        # video_transforms are applied after backbone-specific adjustments
        self.video_transforms: Optional[Any] = kwargs.pop("video_transforms", None)
        
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
        For multi-video mode, groups videos by groupby_column.

        Args:
            split (str): Dataset split ('train', 'val', 'test', 'all')
            target_labels (list): List of target label column names

        Returns:
            tuple: (filenames, outcomes, target_indices)
            For multi-video: filenames is a list of lists of filenames per group
            For single-video: filenames is a list of filenames
        """
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        # Get column indices
        filename_col = self.datapoint_loc_label
        split_col = "Split"

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

        if self.multi_video:
            # Group by specified column
            print("Loading multi-video data... grouping by ", self.groupby_column)
            grouped_data = data.groupby(self.groupby_column)
            group_fnames = []
            group_outcomes = []

            for _, group in grouped_data:
                group_videos = []
                group_outcome = None

                for _, row in group.iterrows():
                    file_name = row[filename_col]
                    file_mode = row[split_col].lower()

                    # Only process rows matching the split
                    if split not in ["all", file_mode]:
                        continue

                    # Check if the video file exists
                    if not os.path.exists(file_name):
                        print(f"Skipping video {file_name} because file does not exist.")
                        continue

                    # If target labels are provided, ensure they are valid
                    if target_indices is not None:
                        if group_outcome is None:  # Only get outcome once per group
                            row_outcomes = {}
                            skip_row = False
                            for label in target_labels:
                                value = row[label]
                                if pd.isna(value):
                                    print(f"Skipping group with video {file_name} because target '{label}' is missing.")
                                    skip_row = True
                                    break
                                else:
                                    row_outcomes[label] = value
                            if skip_row:
                                continue
                            group_outcome = row_outcomes

                    group_videos.append(file_name)

                if group_videos and group_outcome is not None:
                    group_fnames.append(group_videos)
                    group_outcomes.append(group_outcome)

            return group_fnames, group_outcomes, target_indices
        else:
            # Original single-video logic
            fnames = []
            outcomes = []

            for _, row in data.iterrows():
                file_name = row[filename_col]
                file_mode = row[split_col].lower()

                if split not in ["all", file_mode]:
                    continue

                if not os.path.exists(file_name):
                    print(f"Skipping video {file_name} because file does not exist.")
                    continue

                if target_indices is not None:
                    skip_row = False
                    row_outcomes = {}
                    for label in target_labels:
                        value = row[label]
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

        if self.multi_video:
            for idx, group_fnames in enumerate(self.fnames):
                valid_group = True
                for fname in group_fnames:
                    try:
                        cap = cv2.VideoCapture(fname)
                        if not cap.isOpened():
                            raise ValueError(f"Unable to open video {fname}")
                        cap.release()
                    except Exception as e:
                        print(f"Warning: Failed to load video {fname}: {str(e)}")
                        self.failed_videos.append((fname, str(e)))
                        valid_group = False
                        break
                if valid_group:
                    valid_indices.append(idx)
        else:
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

        print(f"Found {len(valid_indices)} valid videos/groups out of {len(self.fnames)}")
        if self.failed_videos:
            print(f"Failed to load {len(self.failed_videos)} videos")

        return valid_indices

    def __getitem__(self, index: int) -> tuple:
        actual_idx: int = self.valid_indices[index]
        
        if self.multi_video:
            group_fnames_in_item = self.fnames[actual_idx] # List of all video paths for this item's group
            outcome_for_item = self.outcomes[actual_idx]
            
            current_selected_fnames: List[str]
            # Select video fnames to load, based on self.num_videos and shuffle_videos
            if self.shuffle_videos:
                if len(group_fnames_in_item) > self.num_videos:
                    current_selected_fnames = random.sample(group_fnames_in_item, self.num_videos)
                else:
                    # Take all available videos from the group; they are not further shuffled among themselves here.
                    # If self.num_videos is larger, padding will occur later.
                    current_selected_fnames = list(group_fnames_in_item) # Make a copy if shuffle_videos implies modification later (not the case here but good practice)
                    random.shuffle(current_selected_fnames) # Shuffle the selected videos if shuffle_videos is true
            else: # Not self.shuffle_videos
                current_selected_fnames = group_fnames_in_item[:self.num_videos]
            
            loaded_video_numpy_arrays: List[np.ndarray] = []
            processed_fnames: List[str] = [] # Stores actual fnames or "PAD"

            first_video_shape_info: Optional[Tuple[int, ...]] = None
            first_video_dtype_info: Optional[Any] = None # Using Any for np.dtype

            # Load selected videos
            for video_fname in current_selected_fnames:
                try:
                    video_np = load_video(
                        video_fname,
                        n_frames=self.num_frames,
                        resize=self.resize,
                        normalize=self.normalize,
                        mean=self.mean[0], # Note: uses only the first element of the mean list
                        std=self.std[0],   # Note: uses only the first element of the std list
                        video_transforms=self.video_transforms,
                        rand_augment=self.rand_augment,
                        backbone=self.backbone,
                        stride=self.stride,
                    )
                    loaded_video_numpy_arrays.append(video_np)
                    processed_fnames.append(video_fname)
                    if first_video_shape_info is None: # Capture shape and dtype from first successfully loaded video
                        first_video_shape_info = video_np.shape
                        first_video_dtype_info = video_np.dtype
                except Exception as e:
                    # If a specific video fails to load, propagate the error.
                    # The original __getitem__ raised RuntimeError here.
                    raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e
            
            # Pad with zero-videos if fewer than self.num_videos were loaded/selected
            num_actually_loaded = len(loaded_video_numpy_arrays)
            num_to_pad = self.num_videos - num_actually_loaded

            if num_to_pad > 0:
                pad_shape: Tuple[int, ...]
                pad_dtype: Any # Using Any for np.dtype

                if first_video_shape_info is not None and first_video_dtype_info is not None:
                    pad_shape = first_video_shape_info
                    pad_dtype = first_video_dtype_info
                else:
                    # Fallback if no videos were loaded (e.g., current_selected_fnames was empty initially)
                    # Default to 3 channels (RGB)
                    channels = 3
                    pad_shape = (self.num_frames, self.resize, self.resize, channels)
                    pad_dtype = np.dtype(np.float32) if self.normalize else np.dtype(np.uint8)
                
                for _ in range(num_to_pad):
                    padding_video_np = np.zeros(pad_shape, dtype=pad_dtype)
                    loaded_video_numpy_arrays.append(padding_video_np)
                    processed_fnames.append("PAD") # Mark filename for padded slots

            # Stack all video arrays (actual + padded) into a single numpy array
            final_stacked_videos_np: np.ndarray
            if not loaded_video_numpy_arrays:
                # This case occurs if self.num_videos is 0.
                # For self.num_videos > 0, padding ensures the list is not empty.
                channels = 3 # Default assumption
                dtype_for_empty = np.dtype(np.float32) if self.normalize else np.dtype(np.uint8)
                final_stacked_videos_np = np.empty(
                    (0, self.num_frames, self.resize, self.resize, channels), 
                    dtype=dtype_for_empty
                )
            else:
                # All arrays in loaded_video_numpy_arrays should now have the same shape,
                # and there should be self.num_videos of them.
                final_stacked_videos_np = np.stack(loaded_video_numpy_arrays, axis=0)
            
            # Ensure the number of processed fnames and the first dimension of the stacked videos match self.num_videos
            # This helps catch errors if self.num_videos is positive.
            if self.num_videos > 0:
                if len(processed_fnames) != self.num_videos:
                     raise AssertionError(f"Internal logic error: Mismatch in length of processed_fnames ({len(processed_fnames)}) and self.num_videos ({self.num_videos})")
                if final_stacked_videos_np.shape[0] != self.num_videos:
                     raise AssertionError(f"Internal logic error: Mismatch in final_stacked_videos_np.shape[0] ({final_stacked_videos_np.shape[0]}) and self.num_videos ({self.num_videos})")
            elif self.num_videos == 0: # If num_videos is 0, expect empty lists/arrays
                if len(processed_fnames) != 0:
                    raise AssertionError(f"Internal logic error: processed_fnames should be empty when self.num_videos is 0, but got {len(processed_fnames)}")
                if final_stacked_videos_np.shape[0] != 0:
                    raise AssertionError(f"Internal logic error: final_stacked_videos_np.shape[0] should be 0 when self.num_videos is 0, but got {final_stacked_videos_np.shape[0]}")
                
            return final_stacked_videos_np, outcome_for_item, processed_fnames
        
        else: # Single video logic (remains unchanged from original)
            video_fname: str = self.fnames[actual_idx]
            try:
                video = load_video(
                    video_fname,
                    n_frames=self.num_frames,
                    resize=self.resize,
                    normalize=self.normalize,
                    mean=self.mean[0],
                    std=self.std[0],
                    video_transforms=self.video_transforms,
                    rand_augment=self.rand_augment,
                    backbone=self.backbone,
                    stride=self.stride,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e   
            
            return video, self.outcomes[actual_idx], video_fname
    
def custom_collate_fn(batch: list[tuple[np.ndarray, dict, Union[str, List[str]]]]) -> dict: # Adjusted type hint for paths
    videos, targets, paths = zip(*batch)
    # Multi-video: videos[0] is np.ndarray [num_videos, F, H, W, C]
    if isinstance(videos[0], np.ndarray) and videos[0].ndim == 5:
        # All videos in the batch now have the same shape[0] due to padding in __getitem__
        videos_tensor = torch.stack([torch.from_numpy(v) for v in videos])  # [B, num_videos, F, H, W, C]
        B = videos_tensor.shape[0]
        N = videos_tensor.shape[1] # num_videos
    # Single video: videos[0] is np.ndarray [F, H, W, C]
    elif isinstance(videos[0], np.ndarray) and videos[0].ndim == 4:
        B = len(videos)
        N = 1
        videos_tensor = torch.stack([torch.from_numpy(v[np.newaxis, ...]) for v in videos])  # [B, 1, F, H, W, C]
    else:
        raise ValueError(f"Unexpected video format or shape: type {type(videos[0])}, ndim {videos[0].ndim if isinstance(videos[0], np.ndarray) else 'N/A'}")
    # Permute to [B, N, C, F, H, W]

    video_indices = torch.arange(B).repeat_interleave(N) if N > 1 else None
    # Convert targets to tensor
    temp_targets_dict: dict = defaultdict(list)
    for target in targets:
        if target is not None:
            for k, v in target.items():
                temp_targets_dict[k].append(v)
    final_targets_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in temp_targets_dict.items()}
    return {
        "videos": videos_tensor,
        "targets": final_targets_dict,
        "video_indices": video_indices,
        "video_fname": paths
    }

def get_distributed_video_dataloader(
    config: Any,
    split: str,
    mean: List[float],
    std: List[float],
    shuffle: bool,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    drop_last: bool = True,
    multi_video: Optional[bool] = None,
    groupby_column: Optional[str] = None,
    num_videos: Optional[int] = None,
    shuffle_videos: Optional[bool] = None,
) -> DataLoader:
    # Create the video dataset with multi-video parameters
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
        resize=config.resize,
        multi_video=multi_video if multi_video is not None else getattr(config, 'multi_video', False),
        groupby_column=groupby_column if groupby_column is not None else getattr(config, 'groupby_column', 'StudyInstanceUID'),
        num_videos=num_videos if num_videos is not None else getattr(config, 'num_videos', 1),
        shuffle_videos=shuffle_videos if shuffle_videos is not None else getattr(config, 'shuffle_videos', True),
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