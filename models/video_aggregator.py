
import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoAggregator(nn.Module):
    """
    Learnable aggregator for combining multiple video embeddings.
    Uses a transformer-based approach with self-attention to learn
    relationships between different video segments.
    
    Input: [batch_size, num_segments, embedding_dim]
    Output: [batch_size, embedding_dim]
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_positional_encoding = use_positional_encoding

        # Positional encoding for sequence awareness
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, 512, embedding_dim)  # Max sequence length of 512
            )
            nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # Multi-head self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP for final transformation
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video embeddings [batch_size, num_segments, embedding_dim]
        Returns:
            Aggregated embedding [batch_size, embedding_dim]
        """
        B, N, D = x.shape

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = x + self.pos_encoding[:, :N, :]

        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        # Aggregate across segments using a learned weighting (softmax).
        attention_weights = F.softmax(x.mean(dim=-1), dim=-1)  # shape: [B, N]
        # Weighted sum => shape: [B, D]
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)

        return x
