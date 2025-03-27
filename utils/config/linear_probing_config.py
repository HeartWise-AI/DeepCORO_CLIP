from typing import Dict, List
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register("DeepCORO_video_linear_probing")
@ConfigRegistry.register("DeepCORO_video_linear_probing_test")
class LinearProbingConfig(HeartWiseConfig):    
    # Training parameters
    lr: float
    epochs: int
    scheduler_name: str
    lr_step_period: int
    factor: float    
    optimizer: str
    weight_decay: float
    use_amp: bool
    gradient_accumulation_steps: int
    num_warmup_percent: float
    num_hard_restarts_cycles: float
    warm_restart_tmult: int
    
    # Dataset parameters
    data_filename: str
    num_workers: int
    batch_size: int
    datapoint_loc_label: str
    target_label: List[str]
    rand_augment: bool
    resize: int
    frames: int
    stride: int
    
    # Video Encoder parameters
    model_name: str
    aggregator_depth: int
    num_heads: int
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
    
    # Label mappings - Used for confusion matrix
    labels_map: Dict[str, Dict[str, int]]