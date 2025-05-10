# project/multiview.dataset.py

import os  # gestion des chemins de fichiers
import pandas as pd  # manipulation du CSV
import numpy as np  # opérations sur tableaux
import torch  # gestion des tenseurs
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from utils.video import load_video  # utilitaire de chargement vidéo
from utils.config.heartwise_config import HeartWiseConfig  # config générale du projet
import pathlib
from utils.seed import seed_worker




def multiview_collate_fn(batch):
    """
    Collate function for MultiViewDataset:
    Batch structure:
      - videos: list of tensors of shape (num_videos, frames, C, H, W)
      - examen_id: list of IDs
    Returns:
      dict with:
        - 'videos': Tensor of shape (B, num_videos, frames, C, H, W)
        - 'examen_id': List[str]
    """
    videos = torch.stack([item['videos'] for item in batch], dim=0)
    examen_ids = [item['examen_id'] for item in batch]
    return {'examen_id': examen_ids, 'videos': videos}


class MultiViewDataset(Dataset):
    """
    DataLoader PyTorch pour charger plusieurs vidéos d'un même examen.

    - root: dossier racine contenant les données et le CSV
    - data_filename: chemin relatif du CSV par rapport à root
    - split: 'train', 'val' ou 'test'
    - object_value_filter: filtre sur la colonne 'object_value'
    - num_videos: nombre fixe de vidéos par examen
    - frames, resize, mean, std, rand_augment, model_name, stride: paramètres pour load_video
    """

    def __init__(
        self,
        root: str,
        data_filename: str,
        split: str = 'train',
        datapoint_loc_label: str = "FileName",
        object_value_filter: Optional[int] = None,
        model_name: str = "mvit",  # forcibly for 16 frames
        num_videos: int = 3,
        frames: int = 16,
        resize: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        rand_augment: bool = False,
        stride: int = 1,
    ):
        super().__init__()
        # 1) Paramètres
        self.root = pathlib.Path(root)
        self.split = split.lower()
        self.filename = data_filename
        self.data_filename = data_filename  # Correction: Initialize the missing attribute
        self.object_value_filter = object_value_filter
        self.model_name = model_name.lower()
        self.num_videos = num_videos
        self.frames = frames
        self.resize = resize
        self.mean = mean
        self.std = std
        self.rand_augment = rand_augment
        self.model_name = model_name
        self.stride = stride
        print(f"[multiviewdataset] resize={self.resize}")
        print(f"[multiviewdataset] mean={self.mean}")
        print(f"[multiviewdataset] std={self.std}")

        # 2) Lecture du CSV
        csv_path = self.root / self.data_filename
        df = pd.read_csv(csv_path, sep='α', engine='python')

        # 3) Filtrage par split
        if 'Split' in df.columns:
            df = df[df['Split'].str.lower() == self.split]

        # 4) Filtrage optionnel par object_value
        if self.object_value_filter is not None and 'object_value' in df.columns:
            df = df[df['object_value'] == self.object_value_filter]

        # 5) Groupement par examen et collecte des vidéos
        self.samples = []
        grouped = df.groupby('EXAMEN_ID')['FileName'].apply(list).reset_index()
        for _, row in grouped.iterrows():
            examen_id = row['EXAMEN_ID']
            paths = row['FileName'][:self.num_videos]
            # Garder uniquement les fichiers existants
            valid_paths = [self.root / p for p in paths if (self.root / p).is_file()]
            if not valid_paths:
                continue
            self.samples.append((examen_id, valid_paths))

        print(f"[MultiViewDataset] Chargés {len(self.samples)} examens "
              f"(filter={self.object_value_filter}, max {self.num_videos} vidéos)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        examen_id, paths = self.samples[idx]
        tensors: List[torch.Tensor] = []

        for video_path in paths:
            try:
                arr = load_video(
                    str(video_path),
                    n_frames=self.frames,
                    resize=self.resize,
                    mean=self.mean,
                    std=self.std,
                    rand_augment=self.rand_augment,
                    backbone=self.model_name,
                    stride=self.stride,
                )  # shape => (frames, H, W, C)
                if isinstance(arr, np.ndarray):
                    vid = torch.from_numpy(arr).permute(0, 3, 1, 2)
                else:
                    vid = arr
            except Exception as e:
                print(f"[MultiViewDataset] Erreur chargement {video_path}: {e}")
                dummy = np.zeros((self.frames, self.resize, self.resize, 3), dtype=np.float32)
                vid = torch.from_numpy(dummy).permute(0, 3, 1, 2)
            tensors.append(vid)

        # 6) Padding si moins de vidéos
        while len(tensors) < self.num_videos:
            tensors.append(torch.zeros_like(tensors[0]))

        # 7) Empilement final
        videos = torch.stack(tensors, dim=0)  # shape => (num_videos, frames, C, H, W)

        return {'examen_id': examen_id, 'videos': videos}


def get_multiview_loader(
    root: str,
    data_filename: str,
    split: str = 'train',
    object_value_filter: Optional[int] = None,
    num_videos: int = 3,
    frames: int = 16,
    resize: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    rand_augment: bool = False,
    model_name: str = 'mvit',
    stride: int = 1,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
):
    """
    Wrapper DataLoader pour MultiViewDataset avec options.
    """
    dataset = MultiViewDataset(
        root=root,
        data_filename=data_filename,
        split=split,
        object_value_filter=object_value_filter,
        num_videos=num_videos,
        frames=frames,
        resize=resize,
        mean=mean,
        std=std,
        rand_augment=rand_augment,
        model_name=model_name,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=multiview_collate_fn,
        worker_init_fn=seed_worker,
    )
