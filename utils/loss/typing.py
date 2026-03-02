from typing import Union
from dataclasses import dataclass
from utils.loss.losses import (
    ContrastiveLoss,
    ContrastiveLossDDP,
    InfoNCELoss,
    MultiHeadLoss,
    MSELoss,
    HuberLoss,
    MAELoss,
    RMSELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    MultiClassFocalLoss,
    BinaryFocalLoss
)
from utils.loss.multitask_loss import MultitaskLoss


@dataclass
class Loss:
    loss_type: Union[
        ContrastiveLoss, 
        ContrastiveLossDDP,
        InfoNCELoss,
        MultiHeadLoss,
        MSELoss,
        HuberLoss,
        MAELoss,
        RMSELoss,
        BCEWithLogitsLoss,
        CrossEntropyLoss,
        MultiClassFocalLoss,
        BinaryFocalLoss,
        MultitaskLoss
    ]
    
    def run(self, **kwargs):
        return self.loss_type(**kwargs)
