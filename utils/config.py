import yaml
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

@dataclass
class HeartWiseConfig:    
    # Training parameters
    lr: float
    batch_size: int
    epochs: int
    num_workers: int
    debug: bool
    temperature: float

    # Data parameters
    data_filename: str
    root: str
    target_label: str
    datapoint_loc_label: str
    frames: int
    stride: int

    # Model parameters
    model_name: str
    pretrained: bool
    video_freeze_ratio: float
    text_freeze_ratio: float
    dropout: float
    
    # Optimization parameters
    optimizer: str
    scheduler_type: str
    lr_step_period: int
    factor: float
    weight_decay: float

    # System parameters
    output_dir: str
    seed: int
    use_amp: bool
    device: str
    period: int

    # Loss and metrics parameters
    loss_name: str
    metrics_control: Dict[str, Any]
    recall_k: List[int]
    ndcg_k: List[int]

    # Data augmentation parameters
    rand_augment: bool
    resize: int
    apply_mask: bool
    view_count: Optional[int]

    # Checkpointing parameters
    save_best: str
    resume: bool
    
    # Logging parameters
    tag: str
    name: str
    project: str
    entity: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'HeartWiseConfig':
        """Create a HeartWiseConfig instance from a YAML file."""
        return cls(**vars(args)) 

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return asdict(self) 