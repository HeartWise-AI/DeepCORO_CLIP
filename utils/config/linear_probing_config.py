from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register("DeepCORO_video_linear_probing")
@ConfigRegistry.register("DeepCORO_video_linear_probing_test")
@ConfigRegistry.register("DeepCORO_video_linear_probing_cardio_syntax")
class LinearProbingConfig(HeartWiseConfig):    
    # Pipeline parameters
    pipeline_project: str
    base_checkpoint_path: str
    run_mode: str
    epochs: int
    seed: int
    output_dir: str
    
    # Training parameters
    head_lr: Dict[str, float] # learning rate for heads
    scheduler_name: str # scheduler name
    lr_step_period: int # learning rate step period
    factor: float # factor for learning rate scheduler
    optimizer: str # optimizer name
    head_weight_decay: Dict[str, float] # weight decay for heads
    video_encoder_weight_decay: float # weight decay for video encoder
    use_amp: bool # whether to use AMP
    gradient_accumulation_steps: int # number of gradient accumulation steps
    num_warmup_percent: float # number of warmup percent
    num_hard_restarts_cycles: float # number of hard restarts cycles
    warm_restart_tmult: int # warm restart tmult
    
    # Dataset parameters
    data_filename: str # path to dataset file
    num_workers: int # number of workers for dataloader
    batch_size: int # batch size
    datapoint_loc_label: str # label for datapoint location
    target_label: List[str] # target label
    rand_augment: bool # whether to use random augmentation
    resize: int # resize for video
    frames: int # number of frames
    stride: int # stride for video
    
    # Video Encoder parameters
    model_name: str # video encoder model name
    aggregator_depth: int # number of aggregator layers
    num_heads: int # number of heads
    video_freeze_ratio: float # freeze ratio for video encoder
    dropout: float # dropout for video encoder
    pretrained: bool # whether to use pretrained video encoder
    video_encoder_checkpoint_path: str # path to video encoder checkpoint
    video_encoder_lr: float # learning rate for video encoder
    
    # Linear Probing parameters
    head_structure: Dict[str, int] # output dimension of each head
    loss_structure: Dict[str, str] # loss function for each head
    head_weights: Dict[str, float] # weight for each head
    head_dropout: Dict[str, float] # dropout for each head
    head_task: Dict[str, str] # task for each head
    
    # Label mappings - Used for confusion matrix
    labels_map: Dict[str, Dict[str, str]]

    # Multi-Instance Learning parameters
    multi_video: bool = False # Whether to load multiple videos per sample
    groupby_column: str = "StudyInstanceUID" # Column to group videos by
    num_videos: int = 1 # Number of videos to load per sample/group
    shuffle_videos: bool = True # Whether to shuffle videos within a group
    pooling_mode: str = "mean" # Pooling mode ("mean", "max", "attention", "cls_token", "mean+cls_token", "attention+cls_token")
    attention_hidden: int = 128 # Hidden size for attention pooling
    dropout_attention: float = 0.0 # Dropout for attention pooling block
    attention_lr: float = 1e-4 # Learning rate for attention pooling parameters
    attention_weight_decay: float = 0.0 # Weight decay for attention pooling parameters

    # CLS Token parameters
    use_cls_token: bool = False # Whether to use learnable cls_token for aggregation
    num_attention_heads: int = 8 # Number of attention heads for cls_token processing
    separate_video_attention: bool = True # Whether to use separate attention layers for within/across-video attention
    normalization_strategy: str = "post_norm" # Normalization strategy: "pre_norm" or "post_norm"
    attention_within_lr: float = 1e-3 # Learning rate for within-video attention parameters
    attention_across_lr: float = 1e-3 # Learning rate for across-video attention parameters
    attention_within_weight_decay: float = 1e-5 # Weight decay for within-video attention parameters
    attention_across_weight_decay: float = 1e-5 # Weight decay for across-video attention parameters

    # Aggregation parameters
    aggregate_videos_tokens: bool = True # Whether to apply internal aggregator in VideoEncoder
    per_video_pool: bool = False # Whether to mean-pool patch tokens inside each video when aggregator disabled
    
    # Inference path
    inference_model_path: str = ""

    # Pre-configured dataset statistics (for inference reproducibility)
    # If specified, these will be used instead of recalculating from dataset
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None

    # Confidence interval settings
    ci_confidence_level: float = 0.95  # 95% confidence interval
    ci_n_bootstrap: int = 1000  # Number of bootstrap samples