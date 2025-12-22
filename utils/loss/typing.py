from typing import Union
from dataclasses import dataclass

# Unified losses (RECOMMENDED - auto-detect DDP)
from utils.loss.contrastive import CLIPLoss, SigLIPLoss

# Legacy losses (kept for backwards compatibility)
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
from utils.loss.multi_positive_infonce import MultiPositiveInfoNCELoss
from utils.loss.siglip_pairwise import SiglipPairwiseLoss  # Legacy
from utils.loss.siglip2_bce import (
    SigLIP2BCELoss,
    SigLIP2BCELossDDP,
    SigLIP2MultiPositiveBCELoss,
)
from utils.loss.locca_loss import (
    LocCaCaptioningLoss,
    LocCaReferringExpressionLoss,
    LocCaGroundedCaptioningLoss,
    LocCaCombinedLoss,
    SigLIP2CombinedLoss,
)


@dataclass
class Loss:
    loss_type: Union[
        # Unified losses (RECOMMENDED)
        CLIPLoss,
        SigLIPLoss,
        # Legacy contrastive losses
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
        MultiPositiveInfoNCELoss,
        SiglipPairwiseLoss,
        # SigLIP 2 losses (legacy)
        SigLIP2BCELoss,
        SigLIP2BCELossDDP,
        SigLIP2MultiPositiveBCELoss,
        # LocCa losses
        LocCaCaptioningLoss,
        LocCaReferringExpressionLoss,
        LocCaGroundedCaptioningLoss,
        LocCaCombinedLoss,
        SigLIP2CombinedLoss,
    ]

    def run(self, **kwargs):
        return self.loss_type(**kwargs)
