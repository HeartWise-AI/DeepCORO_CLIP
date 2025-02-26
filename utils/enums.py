from enum import Enum

class RunMode(str, Enum):
    """Enum for different run modes of the training script."""
    TRAIN = "train"
    VALIDATION = "val"
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
        
    # Torch losses
    MSE = "mse"
    L1 = "l1"
    RMSE = "rmse"
    BCE_LOGIT = "bce_logit"
    CE = "ce"
    HUBER = "huber"
    
    # Multi-head loss
    MULTI_HEAD = "multi_head"
    
    # Custom losses
    MULTICLASS_FOCAL = "multiclass_focal"
    BINARY_FOCAL = "binary_focal"
    
    def __str__(self):
        return self.value