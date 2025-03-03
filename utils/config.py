from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

from utils.files_handler import load_yaml

@dataclass
class HeartWiseConfig:    
    # Training parameters
    lr: float
    batch_size: int
    epochs: int
    num_workers: int
    debug: bool
    temperature: float
    mode: str

    # Data parameters
    data_filename: str
    root: str
    target_label: str
    datapoint_loc_label: str
    frames: int
    stride: int
    multi_video: bool
    num_videos: int
    groupby_column: str
    shuffle_videos: bool
    
    # Model parameters
    model_name: str
    pretrained: bool
    video_freeze_ratio: float
    text_freeze_ratio: float
    dropout: float
    num_heads: int
    aggregator_depth: int
    
    # Optimization parameters
    optimizer: str
    scheduler_name: str
    lr_step_period: int
    factor: float
    video_weight_decay: float
    text_weight_decay: float
    gradient_accumulation_steps: int
    num_warmup_percent: float
    num_hard_restarts_cycles: float
    warm_restart_tmult: int

    # System parameters
    output_dir: str
    seed: int
    use_amp: bool
    device: str
    period: int

    # Loss and metrics parameters
    loss_name: str
    recall_k: List[int]
    ndcg_k: List[int]

    # Data augmentation parameters
    rand_augment: bool
    resize: int
    apply_mask: bool
    view_count: Optional[int]

    # Checkpointing parameters
    save_best: str
    resume_training: bool
    checkpoint: Optional[str]
    
    # Logging parameters
    tag: str
    name: str
    project: str
    entity: str

    # Parameters with default values must come last used for testing
    world_size: int = 1  # Number of GPUs/processes for DDP
    is_ref_device: bool = True  # Whether this is the reference device for logging
       
    @classmethod
    def update_config_with_args(cls, base_config: 'HeartWiseConfig', args: Any) -> 'HeartWiseConfig':  
        """Update a HeartWiseConfig instance with command line arguments."""
        data_parameters: Dict[str, Any] = base_config.to_dict().copy()
        for key, value in vars(args).items():
            if value is not None and key in cls.__dataclass_fields__:
                data_parameters[key] = value
        return cls(**data_parameters)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HeartWiseConfig':
        """Create a HeartWiseConfig instance from a YAML file."""
        yaml_config: Dict[str, Any] = load_yaml(yaml_path)
        data_parameters: Dict[str, Any] = {}
        for key, value in yaml_config.items():
            if key in cls.__dataclass_fields__:
                data_parameters[key] = value
        return cls(**data_parameters)    

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return asdict(self) 
