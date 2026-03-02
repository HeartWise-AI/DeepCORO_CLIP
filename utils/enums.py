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
    # Clip losses
    INFO_NCE = "InfoNCE"    
    CONTRASTIVE = "contrastive"
    CONTRASTIVE_DDP = "contrastive_ddp"
    SIGLIP = "siglip"
    SIGLIP_DDP = "siglip_ddp"
        
    # Torch losses
    MSE = "mse" # Mean Squared Error
    MAE = "mae" # Mean Absolute Error
    RMSE = "rmse" # Root Mean Squared Error
    BCE_LOGIT = "bce_logit" # Binary Cross Entropy Logits
    CE = "ce" # Cross Entropy
    HUBER = "huber" # Huber Loss
    
    # Multi-head loss
    MULTI_HEAD = "multi_head" # Multi-head loss
    
    # Multitask losses
    MULTITASK = "multitask" # Multitask loss combining contrastive, captioning, and masked modeling
    
    # Custom losses
    MULTICLASS_FOCAL = "multiclass_focal" # Multi-class Focal Loss
    BINARY_FOCAL = "binary_focal" # Binary Focal Loss

    # CLIP losses
    CLIP = "clip"

    # SigLIP variants
    SIGLIP_PAIRWISE = "siglip_pairwise"
    SIGLIP2_BCE = "siglip2_bce"
    SIGLIP2_BCE_DDP = "siglip2_bce_ddp"
    SIGLIP2_MULTI_POSITIVE = "siglip2_multi_positive"
    SIGLIP2_COMBINED = "siglip2_combined"

    # LOCCA losses
    LOCCA_CAPTION = "locca_caption"
    LOCCA_COMBINED = "locca_combined"
    LOCCA_GROUNDED = "locca_grounded"
    LOCCA_REFERRING = "locca_referring"

    # Multi-positive
    MULTI_POSITIVE_INFONCE = "multi_positive_infonce"

    def __str__(self):
        return self.value
    
class MetricTask(str, Enum):
    """Enum for different metric tasks."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    
    def __str__(self):
        return self.value