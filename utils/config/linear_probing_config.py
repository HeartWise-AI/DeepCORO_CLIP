from typing import Dict, List
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register("DeepCORO_video_linear_probing")
@ConfigRegistry.register("DeepCORO_video_linear_probing_test")
class LinearProbingConfig(HeartWiseConfig):    
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
    head_linear_probing: Dict[str, str] # linear probing class for each head
    head_structure: Dict[str, int] # output dimension of each head
    loss_structure: Dict[str, str] # loss function for each head
    head_weights: Dict[str, float] # weight for each head
    head_dropout: Dict[str, float] # dropout for each head
    head_task: Dict[str, str] # task for each head
    
    # Label mappings - Used for confusion matrix
    labels_map: Dict[str, Dict[str, int]]