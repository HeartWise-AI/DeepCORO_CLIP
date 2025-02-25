from typing import Dict
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingConfig(HeartWiseConfig):    
    # Training parameters
    lr: float
    scheduler_type: str
    lr_step_period: int
    factor: float    
    optimizer: str
    weight_decay: float
    scheduler_type: str
    
    # Dataset parameters
    data_filename: str
    num_workers: int
    batch_size: int
    
    # Model parameters
    task: str
    head_structure: Dict[str, int]
    loss_structure: Dict[str, str]
    head_weights: Dict[str, float]
    model_checkpoint_path: str
