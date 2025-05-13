import os  # système de fichiers
import pandas as pd  # lecture du CSV
import numpy as np  # tableaux numériques
import torch  # tenseurs PyTorch
from typing import List, Optional, Union
from torch.utils.data import Dataset, DataLoader  # abstractions PyTorch
from pathlib import Path  # manipulation de chemins
from utils.video import load_video  # utilitaire pour charger les vidéos
from utils.config.heartwise_config import HeartWiseConfig  # gestion de la config
from utils.seed import seed_worker  # pour la reproductibilité
from utils.ddp import DS  # DistributedSampler



def multiview_collate_fn(batch):
    """
    Empile un batch issu de MultiViewDataset:
    - videos        : Tensor (B, n, frames, C, H, W)
    - exam_embedding: Tensor (B, embedding_dim)
    - target_label  : Tensor (B, n, num_labels)
    - examen_id     : List[str]
    """
    
    # Empilement des embeddings moyens par examen
    embeddings = torch.stack([item['exam_embedding'] for item in batch], dim=0)
    # Empilement des labels par échantillon
    labels = torch.stack([item['target_label'] for item in batch], dim=0)
    # Récupération des identifiants d'examen
    ids = [item['examen_id'] for item in batch]
    return {
        'examen_id': ids,
        'exam_embedding': embeddings,
        'target_label': labels
    }


class MultiViewDataset(Dataset):
    """
    Charge :
    - plusieurs vidéos par examen (jusqu'à config.num_videos ou 'all')
    - labels associés (config.target_label)
    - embeddings pré-calculés (.pt) correspondant aux mêmes filenames
    Les embeddings sont groupés par examen_id puis moyennés.
    """
    def __init__(
        self,
        root: str,
        data_filename: str,
        split: str,
        config: HeartWiseConfig,
        frames: int = 16,
        resize: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        rand_augment: bool = False,
        model_name: str = 'mvit',
        stride: int = 1,
    ):
        super().__init__()
        # Initialisation des chemins et paramètres vidéo
        self.root = Path(root)
        self.csv_file = data_filename
        self.frames = frames; self.resize = resize
        self.mean = mean; self.std = std
        self.rand_augment = rand_augment
        self.model_name = model_name; self.stride = stride
        
        # Lecture de la configuration externe
        self.split = split.lower()
        self.target_label = config.target_label  # colonnes de labels à extraire
        self.datapoint_loc_label = config.datapoint_loc_label  # nom de la colonne FileName
        self.object_value_filter = config.object_value_filter  # filtre object_value (int, liste ou 'all')
        self.num_videos = config.num_videos  # int (max vidéos) ou 'all'
        self.embeddings_dir = Path(config.embeddings_dir)
        
        print(f"[multiviewdataset] resize={self.resize}")
        print(f"[multiviewdataset] mean={self.mean}")
        print(f"[multiviewdataset] std={self.std}")# dossier contenant fichiers .pt

        # Lecture du CSV et filtres
        df = pd.read_csv(self.root / self.csv_file, sep='α', engine='python')
        df.columns = df.columns.str.strip()  # Nettoie les espaces
        print("Colonnes du CSV :", df.columns.tolist())
        print("Nombre de lignes CSV avant filtrage :", len(df))
        # Filtrage selon la phase (train/val/test)
        if 'Split' in df.columns:
            df = df[df['Split'].str.lower() == self.split]
        print("Après filtrage split :", len(df))
        # Filtrage selon object_value s'il n'est pas 'all'
        if self.object_value_filter != 'all' and 'object_value' in df.columns:
            if isinstance(self.object_value_filter, list):
                df = df[df['object_value'].isin(self.object_value_filter)]
            else:
                df = df[df['object_value'] == self.object_value_filter]
        print("Après filtrage object_value :", len(df))

        
        # Construction de self.samples
        # self.samples est une liste de dictionnaires, chacun représentant un examen unique.
        # Chaque dict contient :
        #   - 'examen_id'  : identifiant de l'examen (clé de regroupement)
        #   - 'file_names' : liste des vidéos associées (du CSV) devant être chargées
        #   - 'labels'     : liste des vecteurs de labels correspondant à chaque vidéo
        # Cette structure permet à __len__ de connaître le nombre total d'examens,
        # et à __getitem__ d'accéder efficacement à toutes les informations
        # (vidéos, labels, embeddings) pour un examen donné.
        self.samples = []
        nb_without_emb = 0
        nb_with_emb = 0 
        for exam_id, group in df.groupby('EXAMEN_ID'):
            rows = list(group.itertuples(index=False))
            # Limitation du nombre de vidéos selon config.num_videos
            if isinstance(self.num_videos, int):
                rows = rows[:self.num_videos]
            # Si aucune vidéo restante, on skip cet examen
            if not rows:
                continue
            # Extraction des filenames
            fnames = [getattr(r, self.datapoint_loc_label) for r in rows]
            # Extraction de tous les labels, puis on prend le premier vecteur
            labs = [[getattr(r, col) for col in self.target_label] for r in rows]
            first_lbl = labs[0]
            # Vérification embeddings existants
            has_emb = any((self.embeddings_dir / f"{Path(fn).stem}.pt").is_file() for fn in fnames)
            if not has_emb:
                nb_without_emb += 1
                continue
            nb_with_emb += 1
                #print(f"PAS D'EMBEDDING pour {exam_id} (fichiers attendus : {[str(self.embeddings_dir / f'{Path(fn).stem}.pt') for fn in fnames]})")    
            self.samples.append({
                'examen_id': exam_id,
                'file_names': fnames,
                'target_label': first_lbl
            })

        assert len(self.samples) == len(df.EXAMEN_ID.unique()), \
            f"Erreur : {len(self.samples)} != {len(df.EXAMEN_ID.unique())} (split={self.split})"
        
        print(f"[MultiViewDataset] Chargés {len(self.samples)} examens "
            f"(split={self.split}, object_values={self.object_value_filter}, num_videos={self.num_videos})")
        print(f"[MultiViewDataset] Examens AVEC embeddings : {nb_with_emb}")
        print(f"[MultiViewDataset] Examens SANS embeddings : {nb_without_emb}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Récupération de la configuration de l'examen
        sample = self.samples[idx]
        exam_id = sample['examen_id']
        fnames = sample['file_names']
        labs = sample['target_label']

        # Chargement des embeddings correspondant aux fichiers
        embs = []
        for fname in fnames:
            stem = Path(fname).stem
            emb_file = self.embeddings_dir / f"{stem}.pt"
            # Vérification si le fichier d'embedding existe
            if emb_file.is_file():
                try:
                    emb = torch.load(str(emb_file), weights_only=True)  # Chargement de l'embedding avec sécurité
                    embs.append(emb.detach())  # Détacher le tenseur pour éviter les problèmes de sérialisation
                except Exception as e:
                    print(f"Erreur lors du chargement de l'embedding {emb_file}: {e}")

        # Moyenne des embeddings pour cet examen_id
        if embs:
            exam_embedding = torch.stack(embs, dim=0).mean(dim=0)
        else:
            # Si aucun embedding trouvé, vecteur nul
            exam_embedding = torch.zeros((1,), dtype=torch.float32)
            
        #print(f"[MultiViewDataset] exam_embedding size après mean : {exam_embedding.size()}")

        # Conversion des labels en tenseur Long
        target_label = torch.tensor(labs, dtype=torch.long)

        # Retourne uniquement les embeddings moyens, labels et id
        return {
            'examen_id': exam_id,
            'exam_embedding': exam_embedding,
            'target_label': target_label
        }


def get_multiview_loader(
    root: str,
    split: str,
    data_filename:str,
    config: HeartWiseConfig,
    frames: int = 16,
    resize: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    rand_augment: bool = False,
    model_name: str = 'mvit',
    stride: int = 1,
    batch_size: int = 1,
    shuffle: bool = False,
    rank: Optional[int] = None,
    num_replicas: Optional[int] = None,
    drop_last: bool = False,
    num_workers: int = 0  # Ajout de l'argument num_workers
    
):
    """
    Crée un DataLoader PyTorch pour MultiViewDataset.
    """
    dataset = MultiViewDataset(
        root=root,
        data_filename=data_filename,
        split=split,
        config=config,
        frames=frames,
        resize=resize,
        mean=mean,
        std=std,
        rand_augment=rand_augment,
        model_name=model_name,
        stride=stride
    )
    
    # Create a sampler for distributed training
    sampler = DS.DistributedSampler(
        dataset,
        shuffle=shuffle,
        num_replicas=num_replicas,
        rank=rank
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        #shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,        
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=multiview_collate_fn,
        worker_init_fn=seed_worker
    )
