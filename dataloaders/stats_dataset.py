import os
import torch

import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from utils.seed import seed_worker
from utils.video import load_video
from utils.config.heartwise_config import HeartWiseConfig


class StatsDataset(torch.utils.data.Dataset):
    """Dataset class for calculating mean and std statistics without the Video base class."""

    def __init__(
        self,
        data_filename,
        split,
        target_label,
        datapoint_loc_label="target_video_path",
        num_frames=32,
        stride=1,
        backbone="default",
        max_samples=128,
    ):
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

    def _read_metadata_csv(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Stats dataset CSV not found at {csv_path}")

        try:
            df_alpha = pd.read_csv(csv_path, sep="α", engine="python")
            if df_alpha.shape[1] > 1:
                return df_alpha
        except Exception:
            pass

        return pd.read_csv(csv_path)

    def load_data(self, split, target_label):
        data = self._read_metadata_csv(self.filename)

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        target_index = None
        if target_label is not None:
            first_label = target_label[0]
            if first_label in data.columns:
                target_index = data.columns.get_loc(first_label)
            else:
                target_index = None

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
    """Collate function for StatsDataset that stacks video tensors."""
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        raise RuntimeError("No valid samples in batch")
    videos = torch.stack([torch.from_numpy(sample[0]) for sample in valid_samples])
    return videos


def get_stats_dataloader(config: HeartWiseConfig):
    """Create a lightweight dataloader for the stats pass.

    The main training loader can be fairly heavy (large batch size, many
    workers, prefetching). Reuse only what we need here to keep the startup
    latency low, especially when the reference rank is responsible for
    computing the statistics for every distributed process.
    """

    stats_batch_size = getattr(config, 'stats_batch_size', None)
    if stats_batch_size is None or stats_batch_size <= 0:
        stats_batch_size = max(1, min(getattr(config, 'batch_size', 1), 4))

    stats_num_workers = getattr(config, 'stats_num_workers', None)
    if stats_num_workers is None or stats_num_workers < 0:
        stats_num_workers = min(getattr(config, 'num_workers', 0), 2)

    stats_max_samples = getattr(config, 'stats_max_samples', None)
    if stats_max_samples is None or stats_max_samples <= 0:
        stats_max_samples = max(stats_batch_size, 32)

    stats_dataset = StatsDataset(
        data_filename=config.data_filename,
        split=config.run_mode,
        target_label=config.target_label,
        datapoint_loc_label=config.datapoint_loc_label,
        num_frames=config.frames,
        backbone=config.model_name,
        stride=config.stride,
        max_samples=stats_max_samples,
    )

    prefetch_factor = getattr(config, 'stats_prefetch_factor', None)
    if prefetch_factor is None or prefetch_factor <= 0:
        prefetch_factor = 1 if stats_num_workers > 0 else None

    return DataLoader(
        stats_dataset,
        batch_size=stats_batch_size,
        num_workers=stats_num_workers,
        shuffle=False,
        collate_fn=stats_collate_fn,
        worker_init_fn=seed_worker if stats_num_workers > 0 else None,
        prefetch_factor=prefetch_factor if stats_num_workers > 0 else None,
        persistent_workers=False,
    )    
    
