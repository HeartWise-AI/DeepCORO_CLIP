"""Multi-Instance Linear Probing model.

This module implements a generic multi-head, multi-instance learning (MIL)
architecture that *aggregates* per-instance embeddings with one of three
pooling modes (mean / max / gated-attention) and then feeds the aggregated
representation into one or more *linear heads* (one per task).

The core logic is adapted from a demo script discussed in a previous
conversation and refactored here so that it fits DeepCORO_CLIP’s module /
registry system.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import ModelRegistry

__all__ = [
    "MultiInstanceLinearProbing",
]


@ModelRegistry.register("multi_instance_linear_probing")
class MultiInstanceLinearProbing(nn.Module):
    """Multi-Instance Linear Probing (MIL-LP).

    Args
    -----
    embedding_dim:  Size of per-instance embeddings.
    head_structure: ``dict`` mapping *head name* to *#classes*.
    pooling_mode:   One of ``"mean" | "max" | "attention"``. Defaults to
                    ``"mean"``.
    attention_hidden:  Hidden size used inside the gated-attention module when
                       ``pooling_mode == 'attention'``.
    dropout:        Optional dropout *applied only inside the attention block*.
    """

    def __init__(
        self,
        embedding_dim: int,
        head_structure: Dict[str, int],
        pooling_mode: str = "mean",
        attention_hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if pooling_mode not in {"mean", "max", "attention"}:
            raise ValueError(
                "pooling_mode must be one of 'mean', 'max', or 'attention'"
            )
        if not head_structure:
            raise ValueError("head_structure cannot be empty")

        self.embedding_dim = embedding_dim
        self.pooling_mode = pooling_mode
        self.head_structure = head_structure

        # Attention parameters (if needed)
        if self.pooling_mode == "attention":
            self.attention_V = nn.Linear(embedding_dim, attention_hidden)
            self.attention_U = nn.Linear(embedding_dim, attention_hidden)
            self.attention_w = nn.Linear(attention_hidden, 1)
            self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Create one linear classification / regression head per task.
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(embedding_dim, n_classes)
                for name, n_classes in head_structure.items()
            }
        )

        self._reset_parameters()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args
        -----
        x:     ``[B, N, D]`` tensor where *N* is the #instances (views) per
               sample and *D == embedding_dim*.
        mask:  Optional ``[B, N]`` boolean / binary mask indicating valid
               positions. If ``None``, all instances are considered valid.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``head_name -> logits`` with shape ``[B, n_classes]`` for each
            registered head.
        """
        B, N, D = x.shape  # noqa: N806 (torch.Tensor)! pylint: disable=invalid-name
        if D != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim} but got {D}"
            )

        pooled = self._pool_instances(x, mask)  # [B, D]
        return {name: head(pooled) for name, head in self.heads.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pool_instances(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:  # noqa: D401
        """Aggregate over the *instance* dimension using the configured rule."""
        if self.pooling_mode == "mean":
            if mask is None:
                return x.mean(dim=1)
            # Masked mean
            mask_f = mask.unsqueeze(-1).float()  # [B, N, 1]
            sum_x = (x * mask_f).sum(dim=1)
            count = mask_f.sum(dim=1).clamp(min=1.0)
            return sum_x / count

        if self.pooling_mode == "max":
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            # torch.max returns (values, indices)
            pooled, _ = x.max(dim=1)
            return pooled

        if self.pooling_mode == "attention":
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, 0.0)

            # Gated attention mechanism (Ilse et al., 2018)
            A_V = torch.tanh(self.attention_V(x))  # [B, N, H]
            A_U = torch.sigmoid(self.attention_U(x))  # [B, N, H]
            A = self.attention_w(A_V * A_U)  # [B, N, 1]
            if mask is not None:
                A = A.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            A = F.softmax(A, dim=1)  # [B, N, 1]
            A = self.attn_dropout(A)
            return torch.sum(A * x, dim=1)  # [B, D]

        raise RuntimeError("Invalid pooling_mode – this should never happen")

    def _reset_parameters(self):
        """Initialize all parameters (Xavier for Linear layers)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
