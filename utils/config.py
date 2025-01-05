import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

@dataclass
class HeartWiseConfig:
    # Training parameters
    epochs: int
    batch_size: int
    batch_size: int
    num_workers: int
    lr: float
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

    # Optimization parameters
    optimizer: str
    weight_decay: float
    scheduler_type: str
    lr_step_period: int
    factor: float

    # Additional parameters
    output_dir: str
    seed: int
    use_amp: bool
    device: str
    period: int

    # Loss parameters
    loss_name: str

    # Metrics control
    metrics_control: dict

    # Recall @k
    recall_k: List[int]

    # NDCG @k
    ndcg_k: List[int]

    # Data augmentation
    rand_augment: bool
    resize: int
    apply_mask: bool
    view_count: Optional[int]

    # Checkpointing
    save_best: str
    resume: bool
    
    # Logging
    project: str
    entity: str
    tag: str

    @classmethod
    def from_yaml(cls, config_path: str) -> 'HeartWiseConfig':
        """Create a HeartWiseConfig instance from a YAML file."""
        config_dict = load_config(config_path)
        return cls(**config_dict) 