from typing import List, Optional
from dataclasses import dataclass
import os # Keep os import if used elsewhere, or remove if only for set_gpu_info_in_place

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register("DeepCORO_clip")
@ConfigRegistry.register("DeepCORO_clip_test")
class ClipConfig(HeartWiseConfig):
    # ===== FIELDS WITHOUT DEFAULTS (REQUIRED) =====
    
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
    
    # Video pooling configuration
    video_pooling_mode: str  # 'mean', 'attention', or 'cls_token'
    attention_pool_heads: int
    attention_pool_dropout: float
    
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

    # ===== FIELDS WITH DEFAULTS (OPTIONAL) =====
    
    # Training parameter defaults
    validation_batch_size: int = 16  # Separate validation batch size
    persistent_workers: bool = False  # Keep DataLoader workers alive
    prefetch_factor: int = 2  # Number of batches to prefetch
    
    # Optional parameters
    view_count: Optional[int] = None

    # Encoder checkpoint path (optional)
    encoder_path: Optional[str] = None  # Path to pretrained encoder checkpoint

    # CLS token configuration
    use_cls_token: bool = False

    # RoPE configuration (optional with defaults)
    use_rope: bool = False
    rope_base: float = 10000.0
    rope_temporal_scale: float = 1.0
    rope_normalize_mode: str = "separate"  # "separate", "max", or "min"
    
    # Temperature scheduling parameters (optional with defaults)
    temperature_schedule: str = "constant"  # "linear", "cosine", "exponential", or "constant"
    temperature_start: float = 1.0  # Starting temperature for scheduling
    temperature_end: float = 0.1  # Target temperature for scheduling
    temperature_warmup_epochs: int = 0  # Number of epochs for temperature warmup
    
    # Video freeze scheduling parameters (optional with defaults)
    video_freeze_schedule: str = "constant"  # "linear", "step", or "constant"  
    video_freeze_start: float = 0.95  # Starting freeze ratio for scheduling
    video_freeze_end: float = 0.0  # Target freeze ratio for scheduling
    video_freeze_warmup_epochs: int = 0  # Number of epochs for freeze warmup
    
    # Text freeze scheduling parameters (optional with defaults)
    text_freeze_schedule: str = "constant"  # "linear", "cosine", "step", or "constant"
    text_freeze_start: float = 0.95  # Starting freeze ratio for text encoder
    text_freeze_end: float = 0.7  # Target freeze ratio for text encoder
    text_freeze_warmup_epochs: int = 0  # Number of epochs for text freeze warmup

    # Device and distributed info are now inherited from HeartWiseConfig
    # No local definition of device, world_size, is_ref_device, 
    # __post_init__ for device setup, or set_gpu_info_in_place needed here.

    def __post_init__(self):
        # Set default values for list fields
        if self.recall_k is None:
            self.recall_k = [1, 5]
        if self.ndcg_k is None:
            self.ndcg_k = [5]
