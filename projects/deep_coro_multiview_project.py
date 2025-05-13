import os
import torch
from typing import Any
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import pathlib
import wandb


from runners.typing import Runner
from models.video_encoder import VideoEncoder
from models.linear_probing import LinearProbing
from projects.base_project import BaseProject
from utils.registry import (
    ProjectRegistry, 
    RunnerRegistry, 
    ModelRegistry,
    LossRegistry
)

from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.enums import RunMode, LossType
from utils.schedulers import get_scheduler
from utils.wandb_wrapper import WandbWrapper
from utils.files_handler import generate_output_dir_name
from utils.video_project import calculate_dataset_statistics_ddp
from utils.config.multiview_config import MultiviewConfig
from dataloaders.video_dataset import get_distributed_video_dataloader
from dataloaders.multiview_dataset import get_multiview_loader

@ProjectRegistry.register("DeepCORO_Multiview")
class MultiviewProject(BaseProject):
    def __init__(
        self, 
        config: MultiviewConfig,
        wandb_wrapper: WandbWrapper,
    ):
        self.config: MultiviewConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        self.multiview_loader = get_multiview_loader(
            root=self.config.root,
            data_filename=self.config.data_filename,
            config=self.config,  # <-- AJOUTE CETTE LIGNE
            split="train",
            #object_value_filter=self.config.object_value_filter,
            #num_videos=self.config.num_videos,
            frames=self.config.frames,
            resize=self.config.resize,
            mean=self.config.mean,
            std=self.config.std,
            rand_augment=self.config.rand_augment,
            model_name=self.config.model_name,
            stride=self.config.stride,
            batch_size=self.config.batch_size,
            #num_workers=self.config.num_workers,
            drop_last=False,
            #target_label=self.config.target_label,  # <-- AJOUT ICI

        )

    def _setup_training_objects(self) -> dict[str, Any]:
        # Création du modèle
        video_encoder: VideoEncoder = ModelRegistry.get(name="video_encoder")(
            backbone=self.config.model_name,
            input_channels=3,  # Valeur par défaut pour les canaux d'entrée
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            output_dim=512,  # Dimension de sortie par défaut
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth
        )
        # Création du module linear_probing
        from models.linear_probing import LinearProbing
        linear_probing = LinearProbing(
            backbone=video_encoder,
            linear_probing_head=self.config.linear_probing_head,
            head_structure=self.config.head_structure,
            dropout=self.config.dropout,
            freeze_backbone_ratio=self.config.video_freeze_ratio,
        )
        # Définition de l'optimiseur
        optimizer = torch.optim.Adam(video_encoder.parameters(), lr=self.config.lr)
    
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)
        
        # 2) Créer les DataLoaders multivues
        train_loader: DataLoader = get_multiview_loader(
            root=self.config.root,
            data_filename=self.config.data_filename,
            split="train",
            #num_videos=self.config.num_videos,
            frames=self.config.frames,
            resize=self.config.resize,
            config=self.config,
            mean=mean,
            std=std,
            rand_augment=self.config.rand_augment,
            model_name=self.config.model_name,
            stride=self.config.stride,
            batch_size=self.config.batch_size,
            shuffle=True,
            #num_workers=self.config.num_workers,
            drop_last=False,
            #target_label=self.config.target_label,  # <-- AJOUT ICI

        )
        val_loader: DataLoader = get_multiview_loader(
            root=self.config.root,
            data_filename=self.config.data_filename,
            split="val",
            #num_videos=self.config.num_videos,
            config=self.config,
            frames=self.config.frames,
            resize=self.config.resize,
            mean=self.config.mean,
            std=self.config.std,
            rand_augment=False,
            model_name=self.config.model_name,
            stride=self.config.stride,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
            #target_label=self.config.target_label,  # <-- AJOUT ICI

        )

        # Définition du scheduler avec les arguments requis
        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name=self.config.scheduler_name,
            num_epochs=self.config.epochs,  # Ajout de num_epochs depuis la configuration
            train_dataloader=train_loader  # Ajout du DataLoader d'entraînement
        )
    
        # Création du scaler pour AMP si nécessaire
        scaler = GradScaler() if self.config.use_amp else None

        # Create loss function
        loss_fn: Loss = Loss(
            loss_type=LossRegistry.get(
                name=LossType.MULTI_HEAD
            )(
                head_structure=self.config.head_structure,
                loss_structure=self.config.loss_structure,
                head_weights=self.config.head_weights,
            )
        )

        # Simplified setup for testing
        full_output_path = None
        if self.config.is_ref_device:
            # Generate output directory using wandb run ID that was already created
            run_id = self.wandb_wrapper.get_run_id() if self.wandb_wrapper.is_initialized() else ""
            output_subdir = generate_output_dir_name(self.config, run_id)
            full_output_path = os.path.join(self.config.output_dir, output_subdir)
            os.makedirs(full_output_path, exist_ok=True)

            if self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.config_update(
                    {
                        "train_dataset_size": len(train_loader),
                        "val_dataset_size": len(val_loader),
                    },
                )        
            print("\n=== Dataset Information ===")
            print(f"Training:   {len(train_loader):,} batches per GPU")
            print(f"Validation: {len(val_loader):,} batches per GPU")
            print(f"Total:      {(len(train_loader) + len(val_loader)):,} batches per GPU")
            print(f"\nBatch Size: {self.config.batch_size}")
            print(f"Training: {len(train_loader) * self.config.batch_size:,} videos per GPU")
            print(f"Validation: {len(val_loader) * self.config.batch_size:,} videos per GPU")
            print(f"Total: {(len(train_loader) + len(val_loader)) * self.config.batch_size:,} videos per GPU")
            print("===========================\n")

        print(f"Full output path: {full_output_path}")

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "output_dir": full_output_path,
            "video_encoder": video_encoder,
            "linear_probing": linear_probing,            
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "scaler": scaler,
            "loss_fn": loss_fn,  # Ajout de la fonction de perte
        }
        

    def _setup_inference_objects(
        self
    )->dict[str, Any]:
        raise NotImplementedError("Inference is not implemented for this project")

    def run(self):
        print("Running Multiview Project")
        print(self.config)        

        training_setup: dict[str, Any] = self._setup_training_objects()
        print("Training setup:", training_setup)

        # Initialisation du runner sans le modèle
        runner = RunnerRegistry.get("DeepCORO_Multiview")(
                config=self.config,
                wandb_wrapper=self.wandb_wrapper,
                train_loader=training_setup["train_loader"],
                val_loader=training_setup["val_loader"],
                optimizer=training_setup["optimizer"],  # Réintégration de l'optimiseur
                lr_scheduler=training_setup["lr_scheduler"],  # Réintégration du scheduler
                scaler=training_setup["scaler"],  # Réintégration du scaler
                output_dir=training_setup["output_dir"],
                loss_fn=training_setup["loss_fn"],
                video_encoder=training_setup["video_encoder"],
                linear_probing=training_setup["linear_probing"],
        )

        # Appel de la méthode run du runner
        try:
            runner.run()
        except Exception as e:
            print(f"[ERREUR] L'entraînement s'est arrêté à cause de : {e}")