from enum import Enum

class RunMode(str, Enum):
    """Enum for different run modes of the training script."""
    TEST = "test"
    TRAIN = "train"
    VALIDATE = "val"
    INFERENCE = "inference"
    
    def __str__(self):
        return self.value
    
class SubmoduleType(str, Enum):
    """Enum for different submodule types."""
    RUNNER = "runners"
    MODEL = "models"
    PROJECT = "projects"
    CONFIG = "utils.config"
    LOSS = "utils.loss"
    
    def __str__(self):
        return self.value
    
class LossType(str, Enum):
    """Enum for different loss types."""
    # =========================================================================
    # MAIN CONTRASTIVE LOSSES (use these)
    # =========================================================================
    # CLIP: Softmax cross-entropy (1 positive per video, auto-DDP)
    CLIP = "clip"
    CONTRASTIVE = "contrastive"  # Alias for CLIP
    CONTRASTIVE_DDP = "contrastive_ddp"  # Alias for CLIP (auto-detects DDP)

    # SigLIP: Sigmoid BCE (multi-positive per video, auto-DDP)
    SIGLIP = "siglip"  # TRUE SigLIP (BCE-based)
    SIGLIP_PAIRWISE = "siglip_pairwise"  # Alias for SIGLIP
    SIGLIP2_BCE = "siglip2_bce"  # Alias for SIGLIP
    SIGLIP2_BCE_DDP = "siglip2_bce_ddp"  # Alias for SIGLIP (auto-detects DDP)
    SIGLIP2_MULTI_POSITIVE = "siglip2_multi_positive"  # Alias for SIGLIP

    # =========================================================================
    # DEPRECATED (blocked - will raise error)
    # =========================================================================
    SIGLIP_DDP = "siglip_ddp"  # BLOCKED: Was softmax, use "siglip" instead
    CLIP_GATED = "clip_gated"  # BLOCKED: Old misnamed loss
    CLIP_GATED_DDP = "clip_gated_ddp"  # BLOCKED: Old misnamed loss

    # =========================================================================
    # OTHER LOSSES
    # =========================================================================
    INFO_NCE = "InfoNCE"
    MULTI_POSITIVE_INFONCE = "multi_positive_infonce"

    # LocCa losses (text generation for SigLIP 2)
    LOCCA_CAPTION = "locca_caption"
    LOCCA_REFERRING = "locca_referring"
    LOCCA_GROUNDED = "locca_grounded"
    LOCCA_COMBINED = "locca_combined"
    SIGLIP2_COMBINED = "siglip2_combined"

    # Torch losses
    MSE = "mse" # Mean Squared Error
    MAE = "mae" # Mean Absolute Error
    RMSE = "rmse" # Root Mean Squared Error
    BCE_LOGIT = "bce_logit" # Binary Cross Entropy Logits
    CE = "ce" # Cross Entropy
    HUBER = "huber" # Huber Loss

    # Multi-head loss
    MULTI_HEAD = "multi_head" # Multi-head loss

    # Custom losses
    MULTICLASS_FOCAL = "multiclass_focal" # Multi-class Focal Loss
    BINARY_FOCAL = "binary_focal" # Binary Focal Loss

    def __str__(self):
        return self.value
    
class MetricTask(str, Enum):
    """Enum for different metric tasks."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    
    def __str__(self):
        return self.value
