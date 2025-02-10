from enum import Enum

class RunMode(str, Enum):
    """Enum for different run modes of the training script."""
    TRAIN = "train"
    VALIDATION = "val"
    INFERENCE = "inference"
    
    def __str__(self):
        return self.value