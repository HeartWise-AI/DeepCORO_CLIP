from typing import List, Optional
from dataclasses import dataclass

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
    view_count: Optional[int]

    # Checkpointing parameters
    save_best: str
    resume_training: bool
    checkpoint: Optional[str]
    
    # Inference parameters
    topk: int
    text_embeddings_path: str
    metadata_path: str
    inference_results_path: str

    # Device and distributed info (must be last)
    device: object = None
    world_size: int = 1
    is_ref_device: bool = True

    @classmethod
    def set_gpu_info_in_place(cls, config: 'ClipConfig') -> None:
        """Set GPU information from environment variables."""
        import os
        if "LOCAL_RANK" in os.environ:
            config.device = int(os.environ["LOCAL_RANK"])
            config.world_size = int(os.environ["WORLD_SIZE"])
            config.is_ref_device = (int(os.environ["LOCAL_RANK"]) == 0)
        else: # This is mostly use for unit testing with github actions
            print("No GPU info found in environment variables")
            config.device = "cpu"
            config.world_size = 1
            config.is_ref_device = True

