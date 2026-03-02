import torch
import torch.nn as nn

from utils.enums import LossType
from utils.registry import LossRegistry


@LossRegistry.register(LossType.MULTI_POSITIVE_INFONCE)
class MultiPositiveInfoNCELoss(nn.Module):
    """Symmetric multi-positive InfoNCE for video-text alignment."""

    def __init__(self, reduction: str = "mean", use_importance_weighting: bool = False):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction '{reduction}'. Expected 'mean' or 'sum'.")
        self.reduction = reduction
        self.use_importance_weighting = use_importance_weighting

    @staticmethod
    def _weighted_ce(logits: torch.Tensor, pos_w: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Cross entropy where each row may have multiple positives with weights."""
        log_prob = torch.log_softmax(logits, dim=dim)  # [N, M]
        weights = pos_w.clamp_min(0)
        row_sum = weights.sum(dim=dim, keepdim=True).clamp_min(1.0)
        weights = weights / row_sum
        loss = -(weights * log_prob).sum(dim=dim)
        return loss

    def forward(
        self,
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [N, M] similarity scores.
            pos_mask: [N, M] binary mask for positives.
            pos_weights: optional [N, M] weights for positives.
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape {tuple(logits.shape)}")
        if pos_mask.shape != logits.shape:
            raise ValueError("pos_mask must match logits shape.")
        if pos_weights is not None and pos_weights.shape != logits.shape:
            raise ValueError("pos_weights must match logits shape.")

        weights = pos_mask if pos_weights is None else pos_mask * pos_weights

        v_has_pos = pos_mask.sum(dim=1) > 0  # [N]
        t_has_pos = pos_mask.sum(dim=0) > 0  # [M]

        losses: list[torch.Tensor] = []
        weighted_losses: list[torch.Tensor] = []
        weight_terms: list[torch.Tensor] = []

        importance_rows = importance_cols = None
        if self.use_importance_weighting:
            base = pos_weights if pos_weights is not None else pos_mask
            base = base.to(logits.dtype)
            importance_rows = base.sum(dim=1)
            importance_cols = base.sum(dim=0)

        if v_has_pos.any():
            v2t = self._weighted_ce(logits, weights, dim=1)
            selected = v2t[v_has_pos]
            if self.use_importance_weighting and importance_rows is not None:
                row_weights = importance_rows[v_has_pos].clamp_min(torch.finfo(logits.dtype).eps)
                weighted_losses.append(selected * row_weights)
                weight_terms.append(row_weights)
            else:
                losses.append(selected)
        if t_has_pos.any():
            t2v = self._weighted_ce(logits.transpose(0, 1), weights.transpose(0, 1), dim=1)
            selected = t2v[t_has_pos]
            if self.use_importance_weighting and importance_cols is not None:
                col_weights = importance_cols[t_has_pos].clamp_min(torch.finfo(logits.dtype).eps)
                weighted_losses.append(selected * col_weights)
                weight_terms.append(col_weights)
            else:
                losses.append(selected)

        if not losses and not weighted_losses:
            # No positives anywhere; return zero so gradients skip this batch.
            return logits.new_tensor(0.0)

        if self.use_importance_weighting and weighted_losses:
            total_loss = torch.cat(weighted_losses).sum()
            total_weight = torch.cat(weight_terms).sum().clamp_min(torch.finfo(logits.dtype).eps)
            if self.reduction == "mean":
                return total_loss / total_weight
            return total_loss

        stacked = torch.cat(losses) if losses else torch.cat(weighted_losses)
        if self.reduction == "mean":
            return stacked.mean()
        return stacked.sum()
