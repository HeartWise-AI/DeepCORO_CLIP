import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.enums import RunMode
from utils.seed import seed_worker
from utils.video import load_video
from utils.config.heartwise_config import HeartWiseConfig
from dataloaders.csv_utils import read_csv_with_fallback


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
        self.stride = np.random.randint(1, stride + 1) if stride > 1 else 1 if split == RunMode.TRAIN else stride
        self.backbone = backbone
        self.max_samples = max_samples

        if split != RunMode.INFERENCE:
            if target_label and not isinstance(target_label, list):
                target_label = [target_label]
        else:
            target_label = None
        self.target_label = target_label
        
        self.fnames, self.outcome, _ = self.load_data(split, target_label)
        
        if self.max_samples and len(self.fnames) > self.max_samples:
            self.fnames = self.fnames[: self.max_samples]
            if self.outcome:
                self.outcome = self.outcome[: self.max_samples]
            print(f"Limited dataset to {self.max_samples} samples")

    def load_data(self, split, target_label):
        expected_cols = [self.datapoint_loc_label, "Split"]
        if target_label and len(target_label) > 0 and target_label[0]:
            expected_cols.append(target_label[0])
        data = read_csv_with_fallback(self.filename, expected_columns=expected_cols)

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        
        target_index = None
        if target_label and len(target_label) > 0 and target_label[0]:
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
    stats_dataset = StatsDataset(
        data_filename=config.data_filename,
        split=config.run_mode,
        target_label=config.target_label,
        datapoint_loc_label=config.datapoint_loc_label,
        num_frames=config.frames,
        backbone=config.model_name,
        stride=config.stride,
    )

    return DataLoader(
        stats_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=stats_collate_fn,
        worker_init_fn=seed_worker,
    )    
    
