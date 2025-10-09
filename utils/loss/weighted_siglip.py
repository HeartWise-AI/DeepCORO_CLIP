import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSigLIPLoss(nn.Module):
    """
    Bidirectional SigLIP reduction that accepts weighted positive pairs.

    Args:
        eps: Numerical stability constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted SigLIP loss.

        Args:
            logits: Similarity logits after temperature scaling, shape [B, M].
            positive_weights: Non-negative weights for positives, same shape as logits.

        Returns:
            Scalar loss value.
        """
        if positive_weights.shape != logits.shape:
            raise ValueError(
                f"positive_weights shape {positive_weights.shape} must match logits shape {logits.shape}."
            )

        pos = positive_weights.clamp(min=0.0)

        # Video-to-text direction.
        logprob_v2t = F.log_softmax(logits, dim=1)
        denom_v2t = pos.sum(dim=1).clamp_min(self.eps)
        loss_v2t = -(pos * logprob_v2t).sum(dim=1) / denom_v2t

        # Text-to-video direction.
        logprob_t2v = F.log_softmax(logits.t(), dim=1)
        pos_t = pos.t()
        denom_t2v = pos_t.sum(dim=1).clamp_min(self.eps)
        loss_t2v = -(pos_t * logprob_t2v).sum(dim=1) / denom_t2v

        return 0.5 * (loss_v2t.mean() + loss_t2v.mean())
