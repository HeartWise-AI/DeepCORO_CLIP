from typing import List, Optional
from dataclasses import dataclass
import os # Keep os import if used elsewhere, or remove if only for set_gpu_info_in_place

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register("DeepCORO_clip")
@ConfigRegistry.register("DeepCORO_clip_test")
class ClipConfig(HeartWiseConfig):
    # Training parameters
    lr: float
    batch_size: int
    num_workers: int
    debug: bool
    temperature: float
    max_grad_norm: float  # Maximum gradient norm for clipping

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
    aggregate_videos_tokens: bool
    per_video_pool: bool
    
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
    use_amp: bool
    period: int

    # Loss and metrics parameters
    loss_name: str
    recall_k: List[int]
    ndcg_k: List[int]

    # Data augmentation parameters
    rand_augment: bool
    resize: int
    apply_mask: bool

    # Checkpointing parameters
    save_best: str
    resume_training: bool
    checkpoint: Optional[str]
    
    # Inference parameters
    topk: int
    text_embeddings_path: str
    metadata_path: str
    inference_results_path: str

    # Optional parameters
    view_count: Optional[int] = None

    # Device and distributed info are now inherited from HeartWiseConfig
    # No local definition of device, world_size, is_ref_device, 
    # __post_init__ for device setup, or set_gpu_info_in_place needed here.