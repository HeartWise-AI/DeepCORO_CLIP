import os
import pathlib
import sys

from typing import Any, List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import AutoTokenizer

dir2 = os.path.abspath("/volume/DeepCORO_CLIP")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

from models.model import get_tokenizer, VideoEncoder  


def load_video(
    video_path,
    split="train",
    n_frames=32,
    stride=1,
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
    if backbone.lower() == "mvit":
        n_frames = 16
        stride = 1

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

    if video.ndim == 3:
        video = np.expand_dims(video, axis=-1)
    elif video.ndim != 4:
        raise ValueError(f"Invalid video shape after loading: {video.shape}")

    video = torch.from_numpy(video)

    # Permute to [F, C, H, W] if last dim is color channels
    if video.shape[-1] in [1, 3]:
        video = video.permute(0, 3, 1, 2)

    # Resize
    if resize is not None:
        video = v2.Resize((resize, resize), antialias=True)(video)

    t, c, h, w = video.shape

    # Force exactly n_frames or 16 frames for MViT
    if backbone.lower() == "mvit":
        if t < 16:
            last_frame = video[-1:].repeat(16 - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > 16:
            indices = torch.linspace(0, t - 1, 16).long()
            video = video[indices]
        expected_frames = 16
    else:
        if t < n_frames:
            last_frame = video[-1:].repeat(n_frames - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > n_frames:
            indices = torch.linspace(0, t - 1, n_frames).long()
            video = video[indices]
        expected_frames = n_frames

    # Optional normalization
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

    # Optional RandAugment
    if rand_augment:
        raug = [v2.RandAugment(magnitude=9, num_ops=2)]
        raug_composed = v2.Compose(raug)
        video = raug_composed(video)

    # Final shape check
    t, c, h, w = video.shape
    if t != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {t}")
    if h != resize or w != resize:
        raise ValueError(f"Expected spatial dimensions {resize}x{resize}, got {h}x{w}")

    # Return shape [F, H, W, C]
    video = video.permute(0, 2, 3, 1).contiguous()
    return video.numpy()


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


class SimpleTextDataset(Dataset):
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
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded


class StatsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        data_filename,
        split,
        target_label,
        datapoint_loc_label="target_video_path",
        num_frames=32,
        stride=1,
        backbone="default",
        max_samples=128,
    ):
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        self.num_frames = 16 if backbone.lower() == "mvit" else num_frames
        self.stride = stride
        self.backbone = backbone
        self.max_samples = max_samples

        if target_label and not isinstance(target_label, list):
            target_label = [target_label]
        self.target_label = target_label

        self.fnames, self.outcome, _ = self.load_data(split, target_label)
        if self.max_samples and len(self.fnames) > self.max_samples:
            self.fnames = self.fnames[: self.max_samples]
            if self.outcome:
                self.outcome = self.outcome[: self.max_samples]
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
                stride=self.stride,
            )
            return video, None, video_fname
        except Exception as e:
            print(f"Error loading video {video_fname}: {str(e)}")
            return None, None, video_fname

    def __len__(self):
        return len(self.fnames)


def stats_collate_fn(batch):
    """
    Used for StatsDataset – collects videos into a single tensor for mean/std calc.
    """
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        raise RuntimeError("No valid samples in batch")
    videos = torch.stack([torch.from_numpy(sample[0]) for sample in valid_samples])
    return videos


def format_mean_std(input_value):
    """
    Helper to ensure mean/std are lists (e.g. [0.485, 0.456, 0.406])
    """
    if isinstance(input_value, (int, float)):
        return [float(input_value)]
    return input_value


def custom_collate_fn(batch):
    """
    Original single-video collate: 
      returns raw video tensor => shape [B, C, F, H, W].
      also returns text dict => input_ids, attention_mask, shape [B, seq_len]
      plus paths => list of file paths
    """
    videos, encoded_texts, paths = zip(*batch)
    videos = torch.stack([torch.from_numpy(v) for v in videos])
    # videos shape => [B, F, H, W, C]
    # reorder to => [B, C, F, H, W]
    videos = videos.permute(0, 4, 1, 2, 3)

    if encoded_texts[0] is not None:
        combined_texts = {
            "input_ids": torch.stack([text["input_ids"] for text in encoded_texts]),
            "attention_mask": torch.stack([text["attention_mask"] for text in encoded_texts]),
        }
    else:
        combined_texts = None

    return videos, combined_texts, paths




import random

class MultiVideoDataset(Dataset):
    """
    Groups video paths by some 'study' key (like StudyInstanceUID).
    Each item => up to N videos, each with exactly 16 frames (for MViT).
    Returns shape => (N,16,H,W,3) as raw floats, plus text, plus study_id.

    - If `shuffle_=True`, then if a study has more than N videos,
      we randomly sample N of them (shuffle). Otherwise we take the
      first N in the original order.
    """

    def __init__(
        self,
        root: str,
        data_filename: str,
        split: str,
        target_label: Optional[str] = "Report",
        datapoint_loc_label: str = "FileName",
        groupby_column: str = "StudyInstanceUID",
        max_num_videos: int = 4,
        backbone: str = "mvit",  # forcibly for 16 frames
        resize: int = 224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        random_augment: bool = False,
        shuffle_videos: bool = False,         
        seed: Optional[int] = None,          
    ):
        super().__init__()
        self.root = pathlib.Path(root)
        self.filename = data_filename
        self.split = split
        self.datapoint_loc_label = datapoint_loc_label
        self.groupby_column = groupby_column
        self.max_num_videos = max_num_videos
        self.backbone = backbone.lower()
        self.resize = resize
        self.mean = mean
        self.std = std
        self.random_augment = random_augment
        self.shuffle_videos = shuffle_videos
        if seed is not None:
            random.seed(seed)  # so we can have reproducible sampling if desired

        # We'll store the text in a dictionary: study_to_text[sid] => str
        # We'll store the list of video paths: study_to_videos[sid] => [paths...]
        self.study_to_videos: Dict[str, List[str]] = {}
        self.study_to_text: Dict[str, str] = {}
        
        csv_path = self.root / self.filename
        df = pd.read_csv(csv_path, sep="α", engine="python")
        df_split = df[df["Split"].str.lower() == split.lower()].copy()

        for i, row in df_split.iterrows():
            sid = str(row[self.groupby_column])
            fpath = row[self.datapoint_loc_label]
            if not os.path.exists(fpath):
                continue
            if sid not in self.study_to_videos:
                self.study_to_videos[sid] = []
            self.study_to_videos[sid].append(fpath)

            # store text in dictionary
            if target_label in row:
                self.study_to_text[sid] = str(row[target_label])
            else:
                self.study_to_text[sid] = "No Report"

        self.study_ids = sorted(list(self.study_to_videos.keys()))
        print(f"[MultiVideoDataset] Found {len(self.study_ids)} studies in split='{split}'")

        # 3) Initialize tokenizer
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        vid_paths = self.study_to_videos[sid]
        text_report = self.study_to_text[sid]

        # If there are more than max_num_videos paths, either slice them in order OR randomly sample
        if len(vid_paths) > self.max_num_videos:
            if self.shuffle_videos:
                # random sample exactly max_num_videos
                chosen_paths = random.sample(vid_paths, self.max_num_videos)
            else:
                # keep original order
                chosen_paths = vid_paths[: self.max_num_videos]
        else:
            chosen_paths = vid_paths  # less or equal => use all

        loaded = []
        for vp in chosen_paths:
            try:
                arr = load_video(
                    vp,
                    n_frames=16,
                    resize=self.resize,
                    mean=self.mean,
                    std=self.std,
                    backbone=self.backbone,
                )  # shape => (16,224,224,3)
            except Exception as e:
                print(f"Warning: {vp} load error: {e}")
                arr = np.zeros((16, self.resize, self.resize, 3), dtype=np.float32)
            loaded.append(arr)

        # If fewer than max_num_videos => pad with zeros
        while len(loaded) < self.max_num_videos:
            arr = np.zeros((16, self.resize, self.resize, 3), dtype=np.float32)
            loaded.append(arr)

        # stack => (N,16,H,W,3)
        multi_stack = np.stack(loaded, axis=0)

        # tokenize text
        encoded = self.tokenizer(
            text_report,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return multi_stack, encoded, sid

    def get_reports(self, study_ids: List[str]) -> List[str]:
        out = []
        for sid in study_ids:
            out.append(self.study_to_text.get(sid, ""))
        return out

    def get_all_reports(self):
        return [self.study_to_text[sid] for sid in self.study_ids]
    
    def get_video_paths(self, sid: str) -> List[str]:
        return self.study_to_videos.get(sid, [])

def multi_video_collate_fn(batch):
    """
    Collate multi-video items:
     - (multi_stack, text_dict, sid)
    multi_stack shape => (N,16,H,W,3)
    We'll stack => shape (B,N,16,H,W,3)
    Also stack text => (B, seq_len)

    Return => (video_tensor, text_dict, [sid,...])
    """
    multi_stacks, text_list, sid_list = zip(*batch)
    # shape => (B,N,16,H,W,3)
    video_tensor = torch.from_numpy(np.stack(multi_stacks, axis=0))  # => float32

    input_ids = torch.stack([x["input_ids"] for x in text_list], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in text_list], dim=0)
    text_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

    return video_tensor, text_dict, list(sid_list)