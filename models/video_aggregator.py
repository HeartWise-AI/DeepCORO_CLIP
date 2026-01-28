import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TransformerBlock(nn.Module):
    """
    A single Transformer-style block with:
      - LayerNorm -> Multihead Attn -> Dropout -> Residual
      - LayerNorm -> MLP -> Dropout -> Residual
    """
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

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
        )  # shape: [B, N, D]
        x = residual + self.dropout1(attn_out)

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, N, D]
        Return: [B, D] aggregated representation.
        """
        B, N, D = x.shape

        if mask is not None:
            mask = mask.to(torch.bool)

        # Apply positional encoding if enabled
        if self.pos_encoding is not None:
            # Add the positional embeddings for the first N positions
            x = x + self.pos_encoding[:, :N, :]  # shape: [1, N, D] broadcast

        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x, key_padding_mask=mask)

        # Final LayerNorm
        x = self.final_ln(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        # Learnable query-based attention
        # attn_query: [1, 1, D] => expand to [B, 1, D]
        query = self.attn_query.expand(B, -1, -1)         # shape: [B, 1, D]
        # Dot-product => [B, 1, N]
        attn_scores = torch.matmul(query, x.transpose(1, 2))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_rows = attn_weights.sum(dim=-1, keepdim=True) <= 0
            if invalid_rows.any():
                valid = (~mask).float()
                fallback = valid / valid.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                attn_weights = torch.where(
                    invalid_rows,
                    fallback.unsqueeze(1),
                    attn_weights,
                )
        else:
            # Normalize to get attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)     # shape: [B, 1, N]

        # Weighted sum of segments => [B, 1, D] => [B, D]
        out = torch.bmm(attn_weights, x).squeeze(1)       # shape: [B, D]
        return out
