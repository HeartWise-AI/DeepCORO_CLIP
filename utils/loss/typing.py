
from typing import Union
from dataclasses import dataclass
from utils.loss.losses import (
    ContrastiveLoss,
    ContrastiveLossDDP,
    InfoNCELoss
)


@dataclass
class Loss:
    loss_type: Union[
        ContrastiveLoss, 
        ContrastiveLossDDP,
        InfoNCELoss
    ]
    
    def run(self, **kwargs):
        return self.loss_type(**kwargs)
