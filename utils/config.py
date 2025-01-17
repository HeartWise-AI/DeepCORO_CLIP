from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List

from utils.files_handler import load_yaml

@dataclass
class BaseConfig:
    # Training parameters
    epochs: int
    num_workers: int
    debug: bool

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

    # System parameters
    output_dir: str
    seed: int
    use_amp: bool
    device: str
    period: int

    # Loss and metrics parameters
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
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """Create a BaseConfig instance from a YAML file."""
        yaml_config: Dict[str, Any] = load_yaml(yaml_path)
        data_parameters: Dict[str, Any] = {}
        for key, value in yaml_config.items():
            if key in cls.__dataclass_fields__:
                data_parameters[key] = value
        return cls(**data_parameters)

@dataclass
class SweepConfig:
    # DataLoader parameters
    batch_size: int
    
    # Optimizer parameters
    lr: float
    optimizer: str
    scheduler_type: str
    lr_step_period: int
    factor: float
    weight_decay: float
    loss_name: str
    temperature: float
    
    # Model parameters
    dropout: float
    video_freeze_ratio: float
    text_freeze_ratio: float
    
    # Logging parameters
    tag: str
    name: str
    project: str
    entity: str
    
    @classmethod
    def from_args(cls, args: Any) -> 'SweepConfig':        
        data_parameters: Dict[str, Any] = {}
        for key, value in vars(args).items():
            if key in cls.__dataclass_fields__:
                data_parameters[key] = value
        return cls(**data_parameters)

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
    
    @classmethod
    def from_config(cls, base_config: BaseConfig, sweep_config: SweepConfig) -> 'HeartWiseConfig':
        """Create a HeartWiseConfig instance from base and sweep configs."""
        data_parameters: Dict[str, Any] = {}
        for key, value in base_config.__dict__.items():
            data_parameters[key] = value
        for key, value in sweep_config.__dict__.items():
            data_parameters[key] = value
        return cls(**data_parameters)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return asdict(self) 