import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    A single Transformer-style block with:
      - LayerNorm -> Multihead Attn -> Dropout -> Residual
      - LayerNorm -> MLP -> Dropout -> Residual
    """
    _warned_non_finite: bool = False  # Class-level flag to warn once

    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        residual = x
        x = self.norm1(x)

        # Check for non-finite values before attention
        if not torch.isfinite(x).all():
            if not TransformerBlock._warned_non_finite:
                print(f"[TransformerBlock] WARNING: Non-finite values BEFORE attention: "
                      f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
                      f"has_nan={torch.isnan(x).any().item()}, has_inf={torch.isinf(x).any().item()}")
                TransformerBlock._warned_non_finite = True
            # Use variance-preserving replacement
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                finite_vals = x[finite_mask]
                replacement = torch.randn_like(x) * finite_vals.std().clamp(min=0.1) + finite_vals.mean()
            else:
                replacement = torch.randn_like(x) * 0.1
            x = torch.where(finite_mask, x, replacement)

        attn_out, _ = self.attn(x, x, x)  # shape: [B, N, D]

        # Check for non-finite values after attention
        if not torch.isfinite(attn_out).all():
            if not TransformerBlock._warned_non_finite:
                print(f"[TransformerBlock] WARNING: Non-finite values AFTER attention: "
                      f"min={attn_out.min().item():.4f}, max={attn_out.max().item():.4f}, "
                      f"has_nan={torch.isnan(attn_out).any().item()}, has_inf={torch.isinf(attn_out).any().item()}")
                TransformerBlock._warned_non_finite = True
            # Variance-preserving replacement to avoid LayerNorm NaN
            finite_mask = torch.isfinite(attn_out)
            if finite_mask.any():
                finite_vals = attn_out[finite_mask]
                replacement = torch.randn_like(attn_out) * finite_vals.std().clamp(min=0.1) + finite_vals.mean()
            else:
                replacement = torch.randn_like(attn_out) * 0.1
            attn_out = torch.where(finite_mask, attn_out, replacement)

        x = residual + self.dropout1(attn_out)

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)

        # Check for non-finite values after MLP
        if not torch.isfinite(x).all():
            if not TransformerBlock._warned_non_finite:
                print(f"[TransformerBlock] WARNING: Non-finite values AFTER MLP: "
                      f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
                      f"has_nan={torch.isnan(x).any().item()}, has_inf={torch.isinf(x).any().item()}")
                TransformerBlock._warned_non_finite = True
            # Variance-preserving replacement to avoid LayerNorm NaN
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                finite_vals = x[finite_mask]
                replacement = torch.randn_like(x) * finite_vals.std().clamp(min=0.1) + finite_vals.mean()
            else:
                replacement = torch.randn_like(x) * 0.1
            x = torch.where(finite_mask, x, replacement)

        x = residual + self.dropout2(x)
        return x


class EnhancedVideoAggregator(nn.Module):
    """
    A multi-layer Transformer-based aggregator that:
      1. Optionally applies a learnable positional encoding.
      2. Passes tokens through several Transformer blocks.
      3. Uses a learnable query vector to attend over all segments.

    Input:
      - x: [B, N, D], where B is batch size, N is # of segments, D is embedding dimension.
    Output:
      - [B, D]: aggregated embedding per sample in the batch.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        aggregator_depth: int = 2,
        max_segments: int = 1024
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_positional_encoding = use_positional_encoding
        self.aggregator_depth = aggregator_depth

        # Optional learnable positional encoding
        if self.use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, max_segments, embedding_dim)  # up to max_segments
            )
            nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        else:
            self.pos_encoding = None

        # Stacked Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout)
            for _ in range(aggregator_depth)
        ])

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(embedding_dim)

        # Learnable query for segment attention
        self.attn_query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        nn.init.normal_(self.attn_query, std=0.02)

    _warned_non_finite: bool = False  # Class-level flag to warn once
    _step_counter: int = 0  # Track steps for periodic logging

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        Return: [B, D] aggregated representation.
        """
        EnhancedVideoAggregator._step_counter += 1
        B, N, D = x.shape

        # Check input for non-finite values
        if not torch.isfinite(x).all():
            if not EnhancedVideoAggregator._warned_non_finite:
                print(f"[Aggregator] WARNING: Non-finite INPUT values: "
                      f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
                      f"has_nan={torch.isnan(x).any().item()}, has_inf={torch.isinf(x).any().item()}")
                EnhancedVideoAggregator._warned_non_finite = True
            # Use variance-preserving replacement to avoid LayerNorm NaN
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                finite_vals = x[finite_mask]
                replacement = torch.randn_like(x) * finite_vals.std().clamp(min=0.1) + finite_vals.mean()
            else:
                replacement = torch.randn_like(x) * 0.1
            x = torch.where(finite_mask, x, replacement)

        # Apply positional encoding if enabled
        if self.pos_encoding is not None:
            # Add the positional embeddings for the first N positions
            x = x + self.pos_encoding[:, :N, :]  # shape: [1, N, D] broadcast

        # Pass through each Transformer block
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Check after each block
            if not torch.isfinite(x).all():
                if not EnhancedVideoAggregator._warned_non_finite:
                    print(f"[Aggregator] WARNING: Non-finite values AFTER block {i}: "
                          f"min={x.min().item():.4f}, max={x.max().item():.4f}")
                    EnhancedVideoAggregator._warned_non_finite = True
                # Variance-preserving replacement to avoid LayerNorm NaN
                finite_mask = torch.isfinite(x)
                if finite_mask.any():
                    finite_vals = x[finite_mask]
                    replacement = torch.randn_like(x) * finite_vals.std().clamp(min=0.1) + finite_vals.mean()
                else:
                    replacement = torch.randn_like(x) * 0.1
                x = torch.where(finite_mask, x, replacement)

        # Final LayerNorm
        x = self.final_ln(x)

        # Learnable query-based attention
        # attn_query: [1, 1, D] => expand to [B, 1, D]
        query = self.attn_query.expand(B, -1, -1)         # shape: [B, 1, D]
        # Dot-product => [B, 1, N]
        attn_scores = torch.matmul(query, x.transpose(1, 2))
        # Scale by 1/sqrt(D) to prevent softmax saturation (standard attention scaling)
        attn_scores = attn_scores / math.sqrt(D)

        # Debug: check attention scores before clamping
        attn_min, attn_max = attn_scores.min().item(), attn_scores.max().item()
        if attn_max > 20.0 or attn_min < -20.0:
            if not EnhancedVideoAggregator._warned_non_finite:
                print(f"[Aggregator] WARNING: Attention scores out of range, CLAMPING: "
                      f"min={attn_min:.4f}, max={attn_max:.4f}")
                EnhancedVideoAggregator._warned_non_finite = True

        # Clamp attention scores to prevent numerical overflow in softmax
        attn_scores = torch.clamp(attn_scores, min=-20.0, max=20.0)
        # Normalize to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)     # shape: [B, 1, N]

        # Weighted sum of segments => [B, 1, D] => [B, D]
        out = torch.bmm(attn_weights, x).squeeze(1)       # shape: [B, D]

        # Final output check
        if not torch.isfinite(out).all():
            if not EnhancedVideoAggregator._warned_non_finite:
                print(f"[Aggregator] WARNING: Non-finite OUTPUT values: "
                      f"min={out.min().item():.4f}, max={out.max().item():.4f}")
                EnhancedVideoAggregator._warned_non_finite = True
            # Variance-preserving replacement for final output
            finite_mask = torch.isfinite(out)
            if finite_mask.any():
                finite_vals = out[finite_mask]
                replacement = torch.randn_like(out) * finite_vals.std().clamp(min=0.1) + finite_vals.mean()
            else:
                replacement = torch.randn_like(out) * 0.1
            out = torch.where(finite_mask, out, replacement)

        return out