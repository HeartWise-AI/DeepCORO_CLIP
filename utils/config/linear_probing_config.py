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
    use_amp: bool
    # Dataset parameters
    data_filename: str
    num_workers: int
    batch_size: int
    
    # Video Encoder parameters
    backbone: str
    aggregator_depth: int
    num_heads: int
    num_frames: int
    video_freeze_ratio: float
    dropout: float
    pretrained: bool
    video_encoder_checkpoint_path: str
    
    # Linear Probing parameters
    task: str
    linear_probing_head: str
    head_structure: Dict[str, int]
    loss_structure: Dict[str, str]
    head_weights: Dict[str, float]
    head_dropout: Dict[str, float]