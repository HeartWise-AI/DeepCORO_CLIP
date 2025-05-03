import os
import torch
from typing import Any
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

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

@ProjectRegistry.register("DeepCORO_Multiview")
class MultiviewProject(BaseProject):
    def __init__(
        self, 
        config: MultiviewConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: MultiviewConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper

    def _setup_training_objects(self) -> dict[str, Any]:
        # Simplified setup for testing
        full_output_path = None
        if self.config.is_ref_device:
            # Generate output directory using wandb run ID that was already created
            run_id = self.wandb_wrapper.get_run_id() if self.wandb_wrapper.is_initialized() else ""
            output_subdir = generate_output_dir_name(self.config, run_id)
            full_output_path = os.path.join(self.config.output_dir, output_subdir)
            os.makedirs(full_output_path, exist_ok=True)
        print(f"Full output path: {full_output_path}")

        return {
            "output_dir": full_output_path
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

