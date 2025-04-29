import os
import torch
from typing import Any
from abc import ABC, abstractmethod

from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig
from utils.files_handler import generate_output_dir_name, backup_config

class BaseProject(ABC):
    def __init__(
        self, 
        config: HeartWiseConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: HeartWiseConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def _setup_inference_objects(self)->dict[str, Any]:
        pass
    
    @abstractmethod
    def _setup_training_objects(self)->dict[str, Any]:
        pass
    
    def _setup_project(self):
        # Generate the output directory name
        self.config.output_dir = generate_output_dir_name(
            config=self.config, 
            run_id=self.wandb_wrapper.get_run_id() if self.wandb_wrapper.is_initialized() else None
        )
        
        print(f"Output directory: {self.config.output_dir}")        
        
        # Create the output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Backup the configuration file
        backup_config(
            config=self.config,
            output_dir=self.config.output_dir
        )
    
    def _load_checkpoint(
        self, 
        checkpoint_path: str
    )->dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        print(
            f"[BaseProject] Loading checkpoint: {checkpoint_path}"
        )
        
        return torch.load(checkpoint_path, map_location='cpu', weights_only=True)