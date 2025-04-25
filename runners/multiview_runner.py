import os
import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from typing import Any
from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.enums import RunMode
from utils.wandb_wrapper import WandbWrapper

# Import your model and loss here if needed
# from models.video_encoder import VideoEncoder
# from utils.loss.typing import Loss



@RunnerRegistry.register("DeepCORO_Multiview")
@RunnerRegistry.register("DeepCORO_Multiview_test")

class MultiviewRunner:
    """
    Runner simplifié pour tester le DataLoader Multiview :
    - itère sur train_loader et val_loader
    - affiche la shape des vidéos, les colonnes 'examen_id' (renvoyées par le dataset depuis 'EXAMEN_ID') et 'FileName'
    - stub inference() à implémenter ultérieurement

    Note:
    - La clé Python 'examen_id' correspond à la colonne CSV originale 'EXAMEN_ID'
    - Les noms de fichiers proviennent de la colonne CSV 'FileName'
    """
    """
    Runner simplifié pour tester le DataLoader Multiview :
    - itère sur train_loader et val_loader
    - affiche la shape des vidéos, les valeurs des colonnes 'EXAMEN_ID' et 'FileName'
    - stub inference() à implémenter ultérieurement

    Note:
    - 'examen_id' provient de la colonne CSV 'EXAMEN_ID'
    - 'file_names' (affichés) proviennent de la colonne CSV 'FileName'
    """
    """
    Runner simplifié pour tester le DataLoader Multiview :
    - itère sur train_loader et val_loader
    - affiche la shape des vidéos, les exam IDs, et premiers file names
    - stub inference() à implémenter ultérieurement
    """
    def __init__(
        self,
        config: Any,
        wandb_wrapper: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,  # Ajout de l'optimiseur
        lr_scheduler: LRScheduler,  # Ajout du scheduler
        scaler: GradScaler,  # Ajout du scaler
        output_dir: str

    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer  # Stocker l'optimiseur
        self.lr_scheduler = lr_scheduler  # Stocker le scheduler
        self.scaler = scaler  # Stocker le scaler
        self.output_dir = output_dir
        self.run_mode = getattr(config, 'run_mode', None)  # ex: 'train' ou 'inference'
        self.use_amp = getattr(config, 'use_amp', False)
        self.scheduler_name = getattr(config, 'scheduler_name', None)  # nom du scheduler

    def train(self) -> None:
        """
        Boucle de test pour l'entraînement :
        - affiche shape des vidéos et exam IDs
        - affiche les 5 premiers file names du premier examen
        """
                        # Nouveaux ajouts :

        print(f"Run mode: {self.run_mode}")
        print(f"Using AMP: {self.use_amp}, scaler: {self.scaler}")
        print(f"Scheduler: {self.scheduler_name}")
        print("=== Test Train Loop Multiview ===")
        print("=== Test Train Loop Multiview ===")
        for batch in self.train_loader:
            videos = batch['videos']  # Tensor de forme (B, num_videos, frames, C, H, W)
            exam_ids = batch['examen_id']  # Liste d'IDs d'examen
            print(f"Batch videos shape: {videos.shape}")
            print(f"Examens IDs: {exam_ids}")
            # Exemple d'utilisation de l'optimiseur
            self.optimizer.zero_grad()

            try:
                first_paths = self.train_loader.dataset.samples[0][1][:5]
                file_names = [os.path.basename(str(path)) for path in first_paths]
                print(f"First 5 video file names for first exam: {file_names}")
            except Exception as e:
                print(f"Unable to retrieve file names: {e}")

            break  # arrêt après le premier batch pour vérification

    def validation(self) -> None:
        """
        Boucle de test pour la validation : mêmes affichages que train()
        """
        print(f"Run mode: {self.run_mode}")
        print(f"Using AMP: {self.use_amp}, scaler: {self.scaler}")
        print(f"Scheduler: {self.scheduler_name}")
        print("=== Test Validation Loop Multiview ===")
        for batch in self.val_loader:
            videos = batch['videos']
            exam_ids = batch['examen_id']
            print(f"Val batch videos shape: {videos.shape}")
            print(f"Val Examens IDs: {exam_ids}")

            try:
                first_paths = self.val_loader.dataset.samples[0][1][:5]
                file_names = [os.path.basename(str(path)) for path in first_paths]
                print(f"First 5 video file names for first exam (val): {file_names}")
            except Exception as e:
                print(f"Unable to retrieve file names: {e}")

            break  # arrêt après le premier batch de validation

    def inference(self) -> None:
        """
        Inference non implémentée pour le moment
        """
        raise NotImplementedError("Inference not implemented for MultiviewRunner")

    def run(self) -> None:
        """
        Point d'entrée pour lancer le test complet : train + validation
        """
        #self.train()
        #self.validation() 
        # nouvelle ajout 
        if self.run_mode == "train":
            self.train()
        elif self.run_mode == "inference":
            self.inference()  # ou appel futur à self.inference()
        else:
            print(f"Unknown run_mode: {self.run_mode}, defaulting to train")
            self.train()
