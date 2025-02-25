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
        if not self.config.resume_training:
            raise ValueError("Flag 'resume_training' is set, but no checkpoint provided")

        if not os.path.exists(self.config.checkpoint):
            raise ValueError(f"Checkpoint file does not exist: {self.config.checkpoint}")
        
        print(
            f"[VideoContrastiveLearningRunner] Resuming from checkpoint: {self.config.checkpoint}"
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    
    def _update_training_setup_with_checkpoint(
        self, 
        training_setup: dict[str, Any], 
        checkpoint: dict[str, Any]
    )->dict[str, Any]:
        print(f"Resuming from checkpoint: {checkpoint.keys()}")
        training_setup["video_encoder"].module.load_state_dict(checkpoint["video_encoder"])
        training_setup["text_encoder"].module.load_state_dict(checkpoint["text_encoder"])
        training_setup["optimizer"].load_state_dict(checkpoint["optimizer"])
        training_setup["scheduler"].load_state_dict(checkpoint["scheduler"])
        training_setup["scaler"].load_state_dict(checkpoint["scaler"])
        training_setup["log_temp"].data.copy_(checkpoint["train/temperature"])
        return training_setup    

