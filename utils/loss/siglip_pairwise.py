import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipPairwiseLoss(nn.Module):
    """Pairwise SigLIP-style BCE loss for multi-positive video-text alignment."""

    def __init__(
        self,
        *,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        reduction: str = "mean",
        use_positive_weights: bool = True,
        auto_positive_weight: bool = False,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction '{reduction}'. Expected 'mean' or 'sum'.")

        self.positive_weight = max(float(positive_weight), 1e-6)
        self.negative_weight = max(float(negative_weight), 0.0)
        self.reduction = reduction
        self.use_positive_weights = use_positive_weights
        self.auto_positive_weight = auto_positive_weight

    def forward(
        self,
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape {tuple(logits.shape)}")
        if pos_mask.shape != logits.shape:
            raise ValueError("pos_mask must match logits shape.")
        if pos_weights is not None and pos_weights.shape != logits.shape:
            raise ValueError("pos_weights must match logits shape.")

        targets = pos_mask.clamp_min(0.0).clamp_max(1.0)

        weight_matrix = torch.full_like(logits, self.negative_weight)
        if self.use_positive_weights and pos_weights is not None:
            positive_contrib = pos_weights * self.positive_weight
        else:
            positive_contrib = torch.full_like(logits, self.positive_weight)

        if self.auto_positive_weight:
            total_texts = logits.size(1)
            pos_counts = pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            neg_counts = total_texts - pos_counts
            ratio = (neg_counts / pos_counts).clamp_min(1.0)
            positive_contrib = ratio.expand_as(logits)

        weight_matrix = torch.where(pos_mask > 0, positive_contrib, weight_matrix)

        # Clamp logits to prevent numerical instability in BCE
        # sigmoid(-30) ≈ 0, sigmoid(30) ≈ 1, so gradients are well-behaved
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=weight_matrix,
            reduction="none",
        )

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
