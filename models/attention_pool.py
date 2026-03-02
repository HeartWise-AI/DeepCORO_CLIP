"""
CLIP-style attention pooling for aggregating patch tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Attention-based pooling mechanism inspired by CLIP.
    
    Uses a learnable query vector to attend over all input tokens,
    producing a single aggregated representation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        output_dim: int = None,
        dropout: float = 0.0,
    ):
        """
        Initialize the attention pooling module.
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            output_dim: Output dimension (if different from embed_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim or embed_dim
        
        # Learnable query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection if needed
        if self.output_dim != embed_dim:
            self.proj = nn.Linear(embed_dim, self.output_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply attention pooling to aggregate tokens.
        
        Args:
            x: Input tokens [batch_size, num_tokens, embed_dim]
            mask: Optional attention mask [batch_size, num_tokens]
                  True values are masked (ignored)
        
        Returns:
            Aggregated features [batch_size, output_dim]
        """
        B, N, D = x.shape
        assert D == self.embed_dim, f"Input dim {D} != expected {self.embed_dim}"
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # [B, 1, embed_dim]
        
        # Prepare attention mask if provided
        key_padding_mask = None
        if mask is not None:
            # For MultiheadAttention, key_padding_mask should be [B, N]
            # True values indicate padded positions to ignore
            key_padding_mask = mask
        
        # Apply multi-head attention
        # Query attends to all tokens in x
        attn_out, attn_weights = self.attn(
            query=query,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )  # [B, 1, embed_dim]
        
        # Apply layer norm
        attn_out = self.norm(attn_out)
        
        # Project and squeeze to remove sequence dimension
        out = self.proj(attn_out).squeeze(1)  # [B, output_dim]
        
        return out


class AttentionPoolWithCLS(nn.Module):
    """
    Attention pooling with an optional CLS token.
    
    This variant prepends a learnable CLS token to the sequence
    before applying self-attention, then uses the CLS token output
    as the aggregated representation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        output_dim: int = None,
        dropout: float = 0.0,
    ):
        """
        Initialize the attention pooling with CLS token.
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            output_dim: Output dimension (if different from embed_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim or embed_dim
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection if needed
        if self.output_dim != embed_dim:
            self.proj = nn.Linear(embed_dim, self.output_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply attention pooling with CLS token.
        
        Args:
            x: Input tokens [batch_size, num_tokens, embed_dim]
            mask: Optional attention mask [batch_size, num_tokens]
                  True values are masked (ignored)
        
        Returns:
            Aggregated features [batch_size, output_dim]
        """
        B, N, D = x.shape
        assert D == self.embed_dim, f"Input dim {D} != expected {self.embed_dim}"
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, embed_dim]
        
        # Update mask if provided
        if mask is not None:
            # Add False (not masked) for CLS token
            cls_mask = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # [B, N+1]
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, N+1, embed_dim]
        
        # Extract CLS token output
        cls_output = x[:, 0]  # [B, embed_dim]
        
        # Apply final norm and projection
        cls_output = self.norm(cls_output)
        out = self.proj(cls_output)  # [B, output_dim]
        
        return out