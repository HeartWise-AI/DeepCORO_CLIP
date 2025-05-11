import os
import pathlib
from typing import Any, List, Optional

import cv2
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.seed import seed_worker
from utils.ddp import DistributedUtils
from models.text_encoder import get_tokenizer
from utils.video import load_video, format_mean_std
from utils.config.heartwise_config import HeartWiseConfig


class VideoClipDataset(torch.utils.data.Dataset):
    """
    Unified dataset class for single- and multi-video (grouped) video-text pairs.
    """

    def __init__(
        self,
        root: str,
        data_filename: str,
        split: str,
        target_label: Optional[str],
        datapoint_loc_label: str = "target_video_path",
        num_frames: int = 16,
        backbone: str = "default",
        debug_mode: bool = False,
        normalize: bool = True,
        mean: Optional[Any] = None,
        std: Optional[Any] = None,
        stride: int = 1,
        groupby_column: Optional[str] = None,
        num_videos: int = 4,
        shuffle_videos: bool = False,
        seed: Optional[int] = None,
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
        self.stride = stride
        self.groupby_column = groupby_column
        self.num_videos = num_videos
        self.shuffle_videos = shuffle_videos
        self.seed = seed
        self.multi_video_mode = kwargs.pop("multi_video", False)


        self.video_transforms = kwargs.pop("video_transforms", None)
        self.rand_augment = kwargs.pop("rand_augment", False)
        self.resize = kwargs.pop("resize", 224)
        self.max_length = kwargs.pop("max_length", 250)

        if self.seed is not None:
            import random
            random.seed(self.seed)
            print(f"[VideoClipDataset] seed={self.seed} for random video sampling")
        else:
            print(f"[VideoClipDataset] no seed for random video sampling")

        if self.split != "inference":
            target_label = (
                [target_label]
                if target_label and not isinstance(target_label, list)
                else target_label
            )
            self.target_label = target_label
            self.external_test_location = kwargs.pop("external_test_location", None)

            if self.multi_video_mode:
                print("Initializing multi-video mode")
                self._init_multi_video()
            else:
                print("Initializing single-video mode")
                self.fnames, self.outcome, self.target_index = self.load_data(
                    self.split, self.target_label
                )

            # Initialize tokenizer only once
            if not hasattr(self, 'tokenizer'):
                try:
                    self.tokenizer = get_tokenizer()
                    print("Tokenizer initialized successfully")
                except Exception as e:
                    print(f"Error initializing tokenizer: {str(e)}")
                    raise RuntimeError("Failed to initialize tokenizer") from e

        if self.debug_mode and not self.multi_video_mode:
            print("Validating all videos in single-video mode with groupby_column", self.groupby_column)
            self.valid_indices = self._validate_all_videos()
        elif not self.multi_video_mode:
            print("Initializing single-video mode")
            self.valid_indices = list(range(len(self.fnames)))

    def _init_multi_video(self):
        import collections
        self.study_to_videos = collections.defaultdict(list)
        self.study_to_text = {}
        csv_path = self.folder / self.filename
        df = pd.read_csv(csv_path, sep="α", engine="python")
        df_split = df[df["Split"].str.lower() == self.split.lower()].copy()
        missing_studies = 0
        # Determine the text column name
        text_col = None
        if self.target_label is not None:
            if isinstance(self.target_label, list):
                text_col = self.target_label[0]
            else:
                text_col = self.target_label
        for _, row in df_split.iterrows():
            group_val = row.get(self.groupby_column, None)
            if group_val is None or not pd.notna(group_val):
                continue
            sid = str(group_val)
            fpath = row[self.datapoint_loc_label]
            if not os.path.exists(fpath):
                missing_studies += 1
                continue
            self.study_to_videos[sid].append(fpath)
            text_val = row.get(text_col, None) if text_col is not None else None
            if text_val is not None and pd.notna(text_val):
                val = text_val
            else:
                val = "No Report"
            self.study_to_text[sid] = str(val)
        # Filter out None keys
        self.study_ids = sorted([k for k in self.study_to_videos.keys() if k is not None])
        print(f"[VideoClipDataset] Found {len(self.study_ids)} studies in split='{self.split}'")
        print(f"[VideoClipDataset] Missing {missing_studies} studies in split='{self.split}'")

    def __len__(self):
        if self.multi_video_mode:
            return len(self.study_ids)
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
        if self.multi_video_mode:
            sid = self.study_ids[index]
            assert isinstance(sid, str), f"sid must be a string, got {type(sid)}"
            vid_paths = self.study_to_videos[sid]
            text_report = self.study_to_text[sid]
            import random
            if self.shuffle_videos:
                vid_paths = random.sample(vid_paths, len(vid_paths))
            if len(vid_paths) > self.num_videos:
                chose_paths = vid_paths[:self.num_videos]
            else:
                chose_paths = vid_paths
            loaded = []
            for vp in chose_paths:
                try:
                    arr = load_video(
                        vp,
                        n_frames=16 if self.backbone.lower() == "mvit" else self.num_frames,
                        resize=self.resize,
                        normalize=self.normalize,
                        mean=self.mean[0] if isinstance(self.mean, list) else self.mean,
                        std=self.std[0] if isinstance(self.std, list) else self.std,
                        video_transforms=self.video_transforms,
                        rand_augment=self.rand_augment,
                        backbone=self.backbone,
                        stride=self.stride,
                    )
                except Exception as e:
                    print(f"Warning: {vp} load error: {e}")
                    arr = np.zeros((16 if self.backbone.lower() == "mvit" else self.num_frames, self.resize, self.resize, 3), dtype=np.float32)
                loaded.append(arr)
            while len(loaded) < self.num_videos:
                arr = np.zeros((16 if self.backbone.lower() == "mvit" else self.num_frames, self.resize, self.resize, 3), dtype=np.float32)
                loaded.append(arr)
            multi_stack = np.stack(loaded, axis=0)

            encoded = self.tokenizer(
                text_report,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.squeeze(0) for k, v in encoded.items()}
            return multi_stack, encoded, sid
        else:
            actual_idx = self.valid_indices[index]
            video_fname = self.fnames[actual_idx]

            try:
                video = load_video(
                    video_fname,
                    n_frames=16 if self.backbone.lower() == "mvit" else self.num_frames,
                    resize=self.resize,
                    normalize=self.normalize,
                    mean=self.mean[0] if isinstance(self.mean, list) else self.mean,
                    std=self.std[0] if isinstance(self.std, list) else self.std,
                    video_transforms=self.video_transforms,
                    rand_augment=self.rand_augment,
                    backbone=self.backbone,
                    stride=self.stride,
                )

                if self.backbone.lower() == "mvit" and video.shape[0] != 16:
                    raise ValueError(f"Expected 16 frames for MViT, got {video.shape[0]}")
 
                encoded = None
                if self.split != "inference" and self.target_label is not None and self.target_index is not None:
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

    def get_reports(self, ids: List[str]) -> List[str]:
        filtered_ids = [x for x in ids if x is not None]
        if self.multi_video_mode:
            return [self.study_to_text.get(sid, "") for sid in filtered_ids]
        else:
            reports = []
            for path in filtered_ids:
                try:
                    idx = self.fnames.index(str(path))
                    reports.append(str(self.outcome[idx]))
                except ValueError:
                    print(f"Warning: No report found for video {path}")
                    reports.append("")
            return reports

    def get_all_reports(self):
        if self.multi_video_mode:
            return [self.study_to_text[sid] for sid in self.study_ids]
        else:
            return [str(o) for o in self.outcome]

    def get_video_paths(self, sid: Optional[str]) -> List[str]:
        if sid is None:
            return []
        if self.multi_video_mode:
            return self.study_to_videos.get(sid, [])
        else:
            return [sid] if sid in self.fnames else []

def custom_collate_fn(batch):
    """Custom collate function to handle video and text data.
    Args:
        batch: List of tuples (video, encoded_text, path_or_sid)
    Returns:
        For multi-video: videos: Tensor (B, N, F, H, W, C), encoded_texts: dict, paths: List[sid]
        For single-video: videos: Tensor (B, F, H, W, C), encoded_texts: dict, paths: List[path]
    """
    videos, encoded_texts, paths = zip(*batch)
    import numpy as np
    import torch
    if isinstance(videos[0], np.ndarray) and videos[0].ndim == 5:
        # Multi-video mode: (N, F, H, W, C)
        videos = torch.from_numpy(np.stack(videos, axis=0))  # (B, N, F, H, W, C)
    else:
        # Single-video mode: (F, H, W, C)
        # Stack to (B, F, H, W, C) and then unsqueeze to (B, 1, F, H, W, C) for consistent N dimension
        videos = torch.stack([torch.from_numpy(v) for v in videos]).unsqueeze(1)
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
        "paths": list(paths)
    }

def get_distributed_video_clip_dataloader(
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
    video_dataset = VideoClipDataset(
        root=getattr(config, 'root', '') or '',
        data_filename=getattr(config, 'data_filename', '') or '',
        split=split,
        target_label=getattr(config, 'target_label', '') or '',
        datapoint_loc_label=getattr(config, 'datapoint_loc_label', '') or '',
        num_frames=getattr(config, 'frames', 32),
        backbone=getattr(config, 'model_name', 'default'),
        mean=mean,
        std=std,
        rand_augment=getattr(config, 'rand_augment', False),
        stride=getattr(config, 'stride', 1),
        groupby_column=getattr(config, 'groupby_column', None),
        num_videos=getattr(config, 'num_videos', 4),
        shuffle_videos=getattr(config, 'shuffle_videos', False),
        seed=getattr(config, 'seed', None),
        multi_video=getattr(config, 'multi_video', False),
        video_transforms=getattr(config, 'video_transforms', None),
        resize=getattr(config, 'resize', 224),
        max_length=getattr(config, 'max_length', 250),
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
        batch_size=getattr(config, 'batch_size', 1),
        sampler=sampler,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
    )