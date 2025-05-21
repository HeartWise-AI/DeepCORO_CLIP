"""Multi-Instance Linear Probing model.

This module implements a generic multi-head, multi-instance learning (MIL)
architecture that *aggregates* per-instance embeddings with one of three
pooling modes (mean / max / gated-attention) and then feeds the aggregated
representation into one or more *linear heads* (one per task).

The core logic is adapted from a demo script discussed in a previous
conversation and refactored here so that it fits DeepCORO_CLIP's module /
registry system.
"""
from __future__ import annotations

from typing import Dict, List, Optional

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
        if x.ndim == 4:  # [B, N, L, D]
            B, N, L, D = x.shape  # noqa: N806
            if D != self.embedding_dim:
                raise ValueError(
                    f"Expected embedding_dim={self.embedding_dim} but got {D}"
                )
            if self.pooling_mode in {"mean", "max"}:
                import warnings
                warnings.warn(
                    f"[MultiInstanceLinearProbing] Received 4D input [B, N, L, D] with pooling_mode='{self.pooling_mode}'. "
                    "Automatically mean-pooling over patch tokens (L) to produce [B, N, D]. "
                    "If you want patch-level attention, use pooling_mode='attention'."
                )
                x = x.mean(dim=2)  # [B, N, D]
        elif x.ndim == 3:  # [B, N, D]
            B, N, D = x.shape  # noqa: N806
            if D != self.embedding_dim:
                raise ValueError(
                    f"Expected embedding_dim={self.embedding_dim} but got {D}"
                )
        else:
            raise ValueError(
                f"Unsupported input shape {x.shape}; expected 3-D or 4-D tensor."
            )

        pooled = self._pool_instances(x, mask)  # [B, D]
        return {name: head(pooled) for name, head in self.heads.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pool_instances(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Aggregate over the *instance* dimension using the configured rule.
        
        Args:
            x: Input tensor of shape [B, N, D] where N is the number of instances
            mask: Optional boolean mask of shape [B, N] indicating valid instances
            
        Returns:
            Tensor of shape [B, D] containing aggregated features
        """
        if mask is None:
            # If no mask provided, treat all instances as valid
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        
        # Ensure mask has correct shape and type
        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match input shape {x.shape[:2]}"
            )
        if not mask.dtype == torch.bool:
            mask = mask.bool()
            
        # Handle empty sequences (no valid instances)
        if not mask.any():
            return torch.zeros(
                x.shape[0], x.shape[2], 
                dtype=x.dtype, 
                device=x.device
            )
            
        if self.pooling_mode == "mean":
            # Masked mean pooling
            mask_f = mask.unsqueeze(-1).float()  # [B, N, 1]
            sum_x = (x * mask_f).sum(dim=1)  # [B, D]
            count = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
            return sum_x / count
            
        elif self.pooling_mode == "max":
            # Masked max pooling
            x_masked = x.clone()
            x_masked[~mask] = float('-inf')
            return x_masked.max(dim=1)[0]  # [B, D]
            
        elif self.pooling_mode == "attention":
            # --------------------------------------------------------------
            # If *hierarchical* inputs ([B, N, L, D]) are supplied we first
            # compute attention over the patch dimension (L) in each video and
            # produce per-video embeddings.  We then reuse the same (shared)
            # linear layers to perform a second round of attention across the
            # N video embeddings.  This keeps the implementation lightweight
            # while still capturing intra- and inter-video relationships.
            # --------------------------------------------------------------

            if x.ndim == 4:  # [B, N, L, D]
                print(f"x.shape: {x.shape}")
                ### Patch-level attention (within each video) ------------
                ## 	Level 1: Patch attention computes which parts of each video are important.
                B, N, L, D_ = x.shape  # noqa: N806 – for readability

                # ---- Patch-level attention (within each video) ------------
                x_patch = x.view(B * N, L, D_)
                A_V_p = torch.tanh(self.attention_V(x_patch))  # [B*N, L, H] (16*4, 1532, 512)
                A_U_p = torch.sigmoid(self.attention_U(x_patch))  # [B*N, L, H] (16*4, 1532, 512)
                A_p = self.attention_w(A_V_p * A_U_p)  # [B*N, L, 1] (16*4, 1532, 1)
                A_p = F.softmax(A_p, dim=1)  # [B*N, L, 1]
                A_p = self.attn_dropout(A_p)
                # Aggregate to per-video embedding
                video_emb = (A_p * x_patch).sum(dim=1)  # [B*N, D]
                video_emb = video_emb.view(B, N, D_)  # [B, N, D]

                # ---- Video-level attention (across videos) ----------------
                A_V = torch.tanh(self.attention_V(video_emb))  # [B, N, H] (16, 4, 512)
                A_U = torch.sigmoid(self.attention_U(video_emb))  # [B, N, H] (16, 4, 512)
                A = self.attention_w(A_V * A_U)  # [B, N, 1] (16, 4, 1)

                # Apply mask if provided (expects [B, N])
                if mask is not None:
                    A = A.masked_fill(~mask.unsqueeze(-1), float("-inf"))

                A = F.softmax(A, dim=1)  # [B, N, 1]
                A = self.attn_dropout(A)

                return (A * video_emb).sum(dim=1)  # [B, D]
            
            ## Level 2: Video attention computes which videos (segments) matter most for a prediction.
            A_V = torch.tanh(self.attention_V(x))  # [B, N, H]  w
            A_U = torch.sigmoid(self.attention_U(x))  # [B, N, H]
            A = self.attention_w(A_V * A_U)  # [B, N, 1] 
            
            # Apply mask to attention scores
            A = A.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            A = F.softmax(A, dim=1)  # [B, N, 1]
            A = self.attn_dropout(A)
            
            # Apply attention weights
            return (A * x).sum(dim=1)  # [B, D]
            
        raise RuntimeError("Invalid pooling_mode – this should never happen")

    def _reset_parameters(self):
        """Initialize all parameters (Xavier for Linear layers)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
