import os
import torch
import pathlib
import random
import collections
import numpy as np
import pandas as pd

from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader

from utils.ddp import DS
from utils.seed import seed_worker
from utils.video import load_video
from utils.config.heartwise_config import HeartWiseConfig
from models.text_encoder import get_tokenizer


class MultiVideoDataset(Dataset):
    """
    Groups video paths by some 'study' key (like StudyInstanceUID).
    Each item => up to N videos, each with exactly 16 frames.
    Returns shape => (N,16,H,W,3) as raw floats, plus text, plus study_id.
    We do NOT accept an 'aggregator' param here. 
    If you want aggregator logic, handle it in the training loop (or collate).
    """

    def __init__(
        self,
        root: str,
        data_filename: str,
        split: str,
        target_label: str = "Report",
        datapoint_loc_label: str = "FileName",
        groupby_column: str = "StudyInstanceUID",
        num_videos: int = 4,
        backbone: str = "mvit",  # forcibly for 16 frames
        resize: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        random_augment: bool = False,
        shuffle_videos: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.root = pathlib.Path(root)
        self.filename = data_filename
        self.split = split
        self.target_label = target_label
        self.datapoint_loc_label = datapoint_loc_label
        self.groupby_column = groupby_column
        self.num_videos = num_videos
        self.backbone = backbone.lower()
        self.resize = resize
        self.mean = mean
        self.std = std
        self.random_augment = random_augment
        self.shuffle_videos = shuffle_videos
        self.seed = seed
        print(f"[MultiVideoDataset] resize={self.resize}")
        print(f"[MultiVideoDataset] mean={self.mean}")
        print(f"[MultiVideoDataset] std={self.std}")
        print(f"[MultiVideoDataset] shuffle_videos={self.shuffle_videos}")
        print(f"[MultiVideoDataset] seed={self.seed}")

        if self.seed is not None:
            random.seed(self.seed)
            print(f"[MultiVideoDataset] seed={self.seed} for random video sampling")
        else:
            print(f"[MultiVideoDataset] no seed for random video sampling")

        # We'll store the text in a dictionary: study_to_text[sid] => str
        # We'll store the list of video paths: study_to_videos[sid] => [paths...]
        self.study_to_videos: Dict[str, List[str]] = collections.defaultdict(list)
        self.study_to_text: Dict[str, str] = {}

        csv_path = self.root / self.filename
        df = pd.read_csv(csv_path, sep="α", engine="python")
        df_split = df[df["Split"].str.lower() == split.lower()].copy()
        missing_studies = 0
        for _, row in df_split.iterrows():
            sid = str(row[self.groupby_column])
            fpath = row[self.datapoint_loc_label]
            if not os.path.exists(fpath):
                missing_studies += 1
                continue
            self.study_to_videos[sid].append(fpath)

            # store text in dictionary
            self.study_to_text[sid] = str(row.get(self.target_label, "No Report"))

        self.study_ids = sorted(list(self.study_to_videos.keys()))
        print(f"[MultiVideoDataset] Found {len(self.study_ids)} studies in split='{split}'")
        print(f"[MultiVideoDataset] Missing {missing_studies} studies in split='{split}'")
        
        # 3) Initialize tokenizer
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        vid_paths = self.study_to_videos[sid]
        text_report = self.study_to_text[sid]

        # If there are more than max_num_videos paths, either slice them in order OR randomly sample
        if len(vid_paths) > self.num_videos:
            if self.shuffle_videos:
                # random sample exactly max_num_videos
                chose_paths = random.sample(vid_paths, self.num_videos)
            else:
                # keep original order
                chose_paths = vid_paths[: self.num_videos]
        else:
            chose_paths = vid_paths  # less or equal => use all

        loaded = []
        for vp in chose_paths:
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

        # if fewer than num_videos => pad with zeros
        while len(loaded) < self.num_videos:
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

    return {
        "videos": video_tensor,
        "encoded_texts": text_dict,
        "paths": list(sid_list)
    }


def get_distributed_multi_video_dataloader(
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
    video_dataset = MultiVideoDataset(
        root=config.root,
        data_filename=config.data_filename,
        split=split,
        target_label=config.target_label,
        datapoint_loc_label=config.datapoint_loc_label,
        groupby_column=config.groupby_column,
        num_videos=config.num_videos,
        backbone=config.model_name,
        mean=mean,
        std=std,
        random_augment=config.rand_augment,
        shuffle_videos=config.shuffle_videos,
        seed=config.seed
    )

    # Create a sampler for distributed training
    sampler = DS.DistributedSampler(
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
        collate_fn=multi_video_collate_fn,
        worker_init_fn=seed_worker,
    )
