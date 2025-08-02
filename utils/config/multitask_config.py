from typing import List, Optional, Dict
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register("DeepCORO_multitask")
class MultitaskConfig(HeartWiseConfig):
    # Training parameters
    lr: float
    batch_size: int
    num_workers: int
    debug: bool
    temperature: float
    max_grad_norm: float
    
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
    
    # Captioning decoder parameters
    vocab_size: int
    decoder_layers: int
    decoder_heads: int
    decoder_intermediate_size: int
    max_position_embeddings: int
    use_biomed_tokenizer: bool
    max_text_length: int
    max_generation_length: int
    captioning_do_sample: bool
    captioning_temperature: float
    
    # Masked video modeling parameters
    mvm_decoder_hidden_size: int
    mvm_decoder_layers: int
    mvm_decoder_heads: int
    mask_ratio: float
    mask_token_learnable: bool
    norm_predict_loss: bool
    
    # Learning rates for different components
    text_lr: float
    captioning_lr: float
    captioning_weight_decay: float
    mvm_lr: float
    mvm_weight_decay: float
    
    # Loss configuration
    contrastive_loss_type: str
    captioning_loss_type: str
    masked_modeling_loss_type: str
    label_smoothing: float
    ignore_index: int
    
    # Loss weights
    loss_weights: Dict[str, float]
    
    # Loss weight scheduler (optional)
    use_loss_weight_scheduler: bool
    initial_loss_weights: Optional[Dict[str, float]] = None
    final_loss_weights: Optional[Dict[str, float]] = None
    loss_warmup_steps: Optional[int] = None
    loss_total_steps: Optional[int] = None
    loss_schedule_type: Optional[str] = None
    
    # Optional parameters
    view_count: Optional[int] = None
    
    def __post_init__(self):
        # Set default values for list fields
        if self.recall_k is None:
            self.recall_k = [1, 5]
        if self.ndcg_k is None:
            self.ndcg_k = [5]
        if self.loss_weights is None:
            self.loss_weights = {
                'contrastive': 1.0,
                'captioning': 1.0,
                'masked_modeling': 0.1,
                'distillation': 0.0
            }