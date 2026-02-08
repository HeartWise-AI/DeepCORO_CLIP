import os
import torch
import random
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import (
    Any, 
    List, 
    Optional, 
    Tuple,
    Union
)

from utils.enums import RunMode
from utils.seed import seed_worker
from utils.ddp import DistributedUtils
from utils.video import load_video, format_mean_std, create_video_capture
from utils.config.heartwise_config import HeartWiseConfig

class VideoDataset(torch.utils.data.Dataset):
    """Single or multi-video dataset class for video-text pairs.""" 

    def __init__(
        self,
        data_filename: str,
        split: Union[str, RunMode],
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
        view_column: Optional[str] = None,
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

        # View embedding parameter (per-video view/angle class column)
        self.view_column: Optional[str] = view_column
        
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

        self.fnames, self.outcomes, self.target_indexes, self.view_classes = self.load_data(
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
        split: Union[str, RunMode],
        target_labels: Optional[List[str]]
    ) -> Tuple[List[str], Optional[List[Optional[dict]]], Optional[dict], Optional[List]]:
        """
        Load data from the CSV file and extract filenames and outcomes.
        For multi-video mode, groups videos by groupby_column.

        Args:
            split (Union[str, RunMode]): Dataset split (TRAIN, VALIDATE, TEST, INFERENCE)
            target_labels (list): List of target label column names

        Returns:
            tuple: (filenames, outcomes, target_indices, view_classes)
            For multi-video: filenames is a list of lists of filenames per group
            For single-video: filenames is a list of filenames
            For inference mode: outcomes is a list of None values (consistent between modes)
            view_classes: parallel list of per-video view class strings (multi-video only), or None
        """
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        # Get column indices
        filename_col = self.datapoint_loc_label
        split_col = "Split"

        # Convert split to string for CSV filtering, but keep original for enum comparison
        split_str = str(split)
        split_dataset = data[data[split_col] == split_str]
        if split_dataset.empty:
            raise ValueError(f"No data found in {self.filename} for split {split_str}")

        # Handle target indices for multi-head case
        if target_labels is None or split == RunMode.INFERENCE:
            target_indices = None
        else:
            target_indices = {}
            for label in target_labels:
                try:
                    target_indices[label] = data.columns.get_loc(label)
                except KeyError:
                    raise ValueError(f"Target label '{label}' not found in data columns")

        # Check if view_column exists in data
        has_view_column = (
            self.view_column is not None
            and self.view_column in data.columns
        )

        if self.multi_video:
            # Group by specified column
            print("Loading multi-video data... grouping by ", self.groupby_column)
            grouped_data = split_dataset.groupby(self.groupby_column)
            group_fnames = []
            group_outcomes = []
            group_view_classes = [] if has_view_column else None

            for group_id, group in grouped_data:
                group_videos = []
                group_views = [] if has_view_column else None
                group_outcome = None
                skip_group = True

                for row_id, row in group.iterrows():
                    file_path = row[filename_col]

                    # Check if the video file exists
                    if not os.path.exists(file_path):
                        print(f"Skipping group {row_id} in {group_id} because video {file_path} does not exist.")
                        continue

                    # If target labels are provided, ensure they are valid
                    if target_indices is not None:
                        if group_outcome is None:  # Only get outcome once per group
                            row_outcomes = {}
                            skip_row = False
                            for label in target_labels:
                                value = row[label]
                                if pd.isna(value):
                                    print(f"Skipping group {row_id} in {group_id} because target '{label}' is missing.")
                                    skip_row = True
                                    break
                                else:
                                    row_outcomes[label] = value
                            if skip_row:
                                continue
                            group_outcome = row_outcomes
                    skip_group = False
                    group_videos.append(file_path)
                    if has_view_column:
                        view_val = row[self.view_column]
                        group_views.append(str(view_val) if not pd.isna(view_val) else "Other")

                if skip_group:
                    continue

                group_fnames.append(group_videos)
                group_outcomes.append(group_outcome)
                if has_view_column:
                    group_view_classes.append(group_views)

            return group_fnames, group_outcomes, target_indices, group_view_classes
        else:
            # Original single-video logic
            fnames = []
            outcomes = []

            for _, row in split_dataset.iterrows():
                file_name = row[filename_col]

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
                    fnames.append(file_name)

                else:
                    # Inference mode or no taget labels
                    fnames.append(file_name)

            if not target_indices:
                # For inference mode, return a list of None values to maintain consistency
                # with multi-video mode and avoid TypeError in __getitem__
                return fnames, [None] * len(fnames), None, None

            return fnames, outcomes, target_indices, None

    def _validate_all_videos(self):
        print("Validating all videos in dataset...")
        valid_indices = []
        self.failed_videos = []

        if self.multi_video:
            for idx, group_fnames in enumerate(self.fnames):
                valid_group = True
                for fname in group_fnames:
                    try:
                        cap = create_video_capture(fname)
                        if cap is None:
                            raise ValueError(f"Unable to open video {fname}")
                        try:
                            if not cap.isOpened():
                                raise ValueError(f"Unable to open video {fname}")
                        finally:
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
                    cap = create_video_capture(fname)
                    if cap is None:
                        raise ValueError(f"Unable to open video {fname}")
                    try:
                        if not cap.isOpened():
                            raise ValueError(f"Unable to open video {fname}")
                    finally:
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
            group_views_in_item = (
                self.view_classes[actual_idx] if self.view_classes is not None else None
            )

            # Build paired list of (fname, view_class) for coordinated selection
            if group_views_in_item is not None:
                paired = list(zip(group_fnames_in_item, group_views_in_item))
            else:
                paired = [(f, None) for f in group_fnames_in_item]

            # Select videos based on self.num_videos and shuffle_videos
            if self.shuffle_videos:
                if len(paired) > self.num_videos:
                    paired = random.sample(paired, self.num_videos)
                else:
                    paired = list(paired)
                    random.shuffle(paired)
            else:
                paired = paired[:self.num_videos]

            loaded_video_numpy_arrays: List[np.ndarray] = []
            processed_fnames: List[str] = []
            processed_views: List[Optional[str]] = []

            first_video_shape_info: Optional[Tuple[int, ...]] = None
            first_video_dtype_info: Optional[Any] = None

            # Load selected videos
            for video_fname, view_cls in paired:
                try:
                    video_np = load_video(
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
                    loaded_video_numpy_arrays.append(video_np)
                    processed_fnames.append(video_fname)
                    processed_views.append(view_cls)
                    if first_video_shape_info is None:
                        first_video_shape_info = video_np.shape
                        first_video_dtype_info = video_np.dtype
                except Exception as e:
                    raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e

            # Pad with zero-videos if fewer than self.num_videos were loaded/selected
            num_actually_loaded = len(loaded_video_numpy_arrays)
            num_to_pad = self.num_videos - num_actually_loaded

            if num_to_pad > 0:
                pad_shape: Tuple[int, ...]
                pad_dtype: Any

                if first_video_shape_info is not None and first_video_dtype_info is not None:
                    pad_shape = first_video_shape_info
                    pad_dtype = first_video_dtype_info
                else:
                    channels = 3
                    pad_shape = (self.num_frames, self.resize, self.resize, channels)
                    pad_dtype = np.dtype(np.float32) if self.normalize else np.dtype(np.uint8)

                for _ in range(num_to_pad):
                    padding_video_np = np.zeros(pad_shape, dtype=pad_dtype)
                    loaded_video_numpy_arrays.append(padding_video_np)
                    processed_fnames.append("PAD")
                    processed_views.append("PAD")

            # Stack all video arrays (actual + padded) into a single numpy array
            final_stacked_videos_np: np.ndarray
            if not loaded_video_numpy_arrays:
                channels = 3
                dtype_for_empty = np.dtype(np.float32) if self.normalize else np.dtype(np.uint8)
                final_stacked_videos_np = np.empty(
                    (0, self.num_frames, self.resize, self.resize, channels),
                    dtype=dtype_for_empty
                )
            else:
                final_stacked_videos_np = np.stack(loaded_video_numpy_arrays, axis=0)

            # Validation assertions
            if self.num_videos > 0:
                if len(processed_fnames) != self.num_videos:
                     raise AssertionError(f"Internal logic error: Mismatch in length of processed_fnames ({len(processed_fnames)}) and self.num_videos ({self.num_videos})")
                if final_stacked_videos_np.shape[0] != self.num_videos:
                     raise AssertionError(f"Internal logic error: Mismatch in final_stacked_videos_np.shape[0] ({final_stacked_videos_np.shape[0]}) and self.num_videos ({self.num_videos})")
            elif self.num_videos == 0:
                if len(processed_fnames) != 0:
                    raise AssertionError(f"Internal logic error: processed_fnames should be empty when self.num_videos is 0, but got {len(processed_fnames)}")
                if final_stacked_videos_np.shape[0] != 0:
                    raise AssertionError(f"Internal logic error: final_stacked_videos_np.shape[0] should be 0 when self.num_videos is 0, but got {final_stacked_videos_np.shape[0]}")

            # Return view classes if view_column is configured
            if self.view_classes is not None:
                return final_stacked_videos_np, outcome_for_item, processed_fnames, processed_views
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
    
def custom_collate_fn(
    batch: list[tuple],
    labels_map: dict = None,
    view_labels_map: dict = None,
    num_view_classes: int = 12,
) -> dict:
    # Detect whether batch items include view_classes (4-tuple vs 3-tuple)
    has_view_classes = len(batch[0]) == 4

    if has_view_classes:
        videos, targets, paths, view_classes_list = zip(*batch)
    else:
        videos, targets, paths = zip(*batch)
        view_classes_list = None

    # Multi-video: videos[0] is np.ndarray [num_videos, F, H, W, C]
    if isinstance(videos[0], np.ndarray) and videos[0].ndim == 5:
        videos_tensor = torch.stack([torch.from_numpy(v) for v in videos])  # [B, num_videos, F, H, W, C]
        B = videos_tensor.shape[0]
        N = videos_tensor.shape[1] # num_videos
    # Single video: videos[0] is np.ndarray [F, H, W, C]
    elif isinstance(videos[0], np.ndarray) and videos[0].ndim == 4:
        B = len(videos)
        videos_tensor = torch.stack([torch.from_numpy(v) for v in videos])  # [B, F, H, W, C]
        N = 1
    else:
        raise ValueError(f"Unexpected video format or shape: type {type(videos[0])}, ndim {videos[0].ndim if isinstance(videos[0], np.ndarray) else 'N/A'}")

    video_indices = torch.arange(B).repeat_interleave(N) if N > 1 else None

    # Convert targets to tensor
    temp_targets_dict: dict = defaultdict(list)
    for target in targets:
        if target is not None:
            for k, v in target.items():
                temp_targets_dict[k].append(v)

    # Apply labels_map to convert categorical strings to integers
    final_targets_dict = {}
    for k, v in temp_targets_dict.items():
        if labels_map and k in labels_map:
            mapped_values = []
            for value in v:
                if isinstance(value, str):
                    if value in labels_map[k]:
                        mapped_values.append(labels_map[k][value])
                    else:
                        raise ValueError(f"Label '{value}' not found in labels_map for column '{k}'. Available labels: {list(labels_map[k].keys())}")
                else:
                    mapped_values.append(value)
            final_targets_dict[k] = torch.tensor(mapped_values, dtype=torch.long)
        else:
            final_targets_dict[k] = torch.tensor(v, dtype=torch.float32)

    # Build view_ids tensor if view classes are present
    view_ids_tensor = None
    if view_classes_list is not None and view_labels_map is not None:
        pad_id = num_view_classes  # PAD ID is the last class
        batch_view_ids = []
        for sample_views in view_classes_list:
            sample_ids = []
            for vc in sample_views:
                if vc == "PAD" or vc is None:
                    sample_ids.append(pad_id)
                elif vc in view_labels_map:
                    sample_ids.append(view_labels_map[vc])
                else:
                    sample_ids.append(pad_id)  # Unknown view → PAD
            batch_view_ids.append(sample_ids)
        view_ids_tensor = torch.tensor(batch_view_ids, dtype=torch.long)  # [B, N]

    result = {
        "videos": videos_tensor,
        "targets": final_targets_dict,
        "video_indices": video_indices,
        "video_fname": paths,
    }
    if view_ids_tensor is not None:
        result["view_ids"] = view_ids_tensor
    return result

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
    labels_map: Optional[dict] = None,
    view_column: Optional[str] = None,
    view_labels_map: Optional[dict] = None,
    num_view_classes: int = 12,
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
        view_column=view_column,
    )

    # Create a sampler for distributed training
    sampler = DistributedUtils.DS.DistributedSampler(
        video_dataset,
        shuffle=shuffle,
        num_replicas=num_replicas,
        rank=rank
    )

    # Create a partial function to bind labels_map and view_labels_map to the collate function
    collate_fn_bound = partial(
        custom_collate_fn,
        labels_map=labels_map,
        view_labels_map=view_labels_map,
        num_view_classes=num_view_classes,
    )

    # Create the dataloader
    return DataLoader(
        video_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn_bound,
        worker_init_fn=seed_worker,
    )
