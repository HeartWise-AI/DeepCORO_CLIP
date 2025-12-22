from typing import Dict, List, Optional
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
    loss_name: str  # Primary loss identifier
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

    # === Fields WITH defaults must come after fields WITHOUT defaults ===

    # Multi-loss configuration
    loss_array: Optional[List[Dict[str, float]]] = None  # e.g. [{"siglip": 1.0}, {"locca_caption": 0.5}]

    # LocCa Decoder parameters
    locca_enabled: bool = False
    locca_num_layers: int = 4
    locca_d_model: int = 512
    locca_num_heads: int = 8
    locca_dropout: float = 0.1
    locca_max_seq_len: int = 256

    # Optional data parameters
    data_mean: Optional[List[float]] = None
    data_std: Optional[List[float]] = None

    # Optional parameters
    view_count: Optional[int] = None
    video_max_grad_norm: Optional[float] = None
    text_max_grad_norm: Optional[float] = None

    # SigLIP parameters for multi-positive contrastive learning
    siglip_texts_path: Optional[str] = None
    siglip_max_positive_per_video: int = 8
    siglip_negatives_per_video: int = 0
    siglip_round_robin_sampling: bool = False
    siglip_max_segments_per_video: int = 15
    siglip_positive_severity_weights: Optional[Dict[str, float]] = None
    siglip_enable_severity_weighting: bool = False
    siglip_positive_loss_weight: float = 1.0
    siglip_negative_loss_weight: float = 1.0
    siglip_auto_positive_loss_weight: bool = False

    # SigLIP entropy regularization (prevents embedding collapse)
    siglip_entropy_regularization: bool = False
    siglip_entropy_weight: float = 0.1
    siglip_min_entropy_threshold: float = 2.0

    # Auxiliary losses
    main_structure_loss_weight: float = 0.0
    tree_loss_enabled: Optional[bool] = None
    tree_loss_weight: Optional[float] = None

    # Device and distributed info are now inherited from HeartWiseConfig
    # No local definition of device, world_size, is_ref_device,
    # __post_init__ for device setup, or set_gpu_info_in_place needed here.

    def __post_init__(self):
        # Set default values for list fields
        if self.recall_k is None:
            self.recall_k = [1, 5]
        if self.ndcg_k is None:
            self.ndcg_k = [5]
