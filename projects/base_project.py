import os
import torch
from typing import Any
from abc import ABC, abstractmethod

from utils.config.heartwise_config import HeartWiseConfig

class BaseProject(ABC):
    def __init__(self, config: HeartWiseConfig):
        self.config = config

    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def _setup_inference_objects(self)->dict[str, Any]:
        pass
    
    @abstractmethod
    def _setup_training_objects(self)->dict[str, Any]:
        pass
    
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