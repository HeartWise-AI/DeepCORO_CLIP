import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import (
    mvit_v2_s,
    r3d_18,
    mvit_v1_b,
)
from transformers import AutoModel, AutoTokenizer

def get_tokenizer(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
    """Get the tokenizer with proper configuration."""
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=512,
        padding_side="right",
        truncation_side="right",
    )


import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        # Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)  # shape: [B, N, D]
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
        max_segments: int = 512
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        Return: [B, D] aggregated representation.
        """
        B, N, D = x.shape

        # Apply positional encoding if enabled
        if self.pos_encoding is not None:
            # Add the positional embeddings for the first N positions
            x = x + self.pos_encoding[:, :N, :]  # shape: [1, N, D] broadcast

        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.final_ln(x)

        # Learnable query-based attention
        # attn_query: [1, 1, D] => expand to [B, 1, D]
        query = self.attn_query.expand(B, -1, -1)         # shape: [B, 1, D]
        # Dot-product => [B, 1, N]
        attn_scores = torch.matmul(query, x.transpose(1, 2))
        # Normalize to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)     # shape: [B, 1, N]

        # Weighted sum of segments => [B, 1, D] => [B, D]
        out = torch.bmm(attn_weights, x).squeeze(1)       # shape: [B, D]
        return out


class TransformerHead(nn.Module):
    """Transformer head for video classification."""

    def __init__(self, dim_in: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class VideoEncoder(nn.Module):
    """
    Video encoder with optional dropout & partial-freeze logic.
    - More standard: uses aggregator => EnhancedVideoAggregator (multi-layer).
    """

    def __init__(
        self,
        backbone="mvit",
        input_channels=3,
        num_frames=16,
        pretrained=True,
        output_dim=512,
        dropout=0.2,
        num_heads=4,
        freeze_ratio=0.8,
        aggregator_depth=2,        # NEW: how many layers in aggregator
    ):
        super().__init__()
        self.backbone = backbone
        self.input_channels = input_channels
        self.num_frames = num_frames
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.freeze_ratio = freeze_ratio

        # 1) Build backbone
        if self.backbone == "mvit":
            # Load the pretrained MViT v2 S
            self.model = mvit_v2_s(weights="KINETICS400_V1" if pretrained else None)
            in_features = None
            # Replace classification head with Identity
            for layer in reversed(self.model.head):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
            if in_features is None:
                in_features = 768
            self.model.head = nn.Identity()

        elif self.backbone == "r3d":
            self.model = r3d_18(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        # 2) Projection (use GELU instead of ReLU)
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(in_features, output_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

        # 3) Enhanced aggregator
        self.aggregator = EnhancedVideoAggregator(
            embedding_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=True,
            aggregator_depth=aggregator_depth,
        )

        # 4) Freeze partial layers
        self._freeze_partial_layers()

    def _freeze_partial_layers(self):
        all_named_params = list(self.model.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)

        # Freeze the bottom portion, keep top `train_count` trainable
        for i, (name, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape => [B, N, T, H, W, C]
        We'll reorder => [B, N, C, T, H, W], flatten => [B*N, C, T, H, W],
        pass backbone => aggregator => [B, output_dim].
        """
        # reorder => [B, N, C, T, H, W]
        x = x.permute(0, 1, 5, 2, 3, 4)  # [B,N,3,T,H,W]
        B, N, C, T, H, W = x.shape

        # flatten => [B*N, C, T, H, W]
        x = x.view(B*N, C, T, H, W)

        # pass backbone => [B*N, in_features]
        feats = self.model(x)
        # projection => [B*N, output_dim]
        feats = self.proj(feats)

        # reshape => [B, N, output_dim]
        feats = feats.view(B, N, self.output_dim)

        # aggregator => [B, output_dim]
        out = self.aggregator(feats)

        return out


class TextEncoder(nn.Module):
    """
    Text encoder with optional dropout & partial-freeze.
    - Uses BERT-based backbone and final projection with GELU.
    """

    def __init__(
        self,
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        output_dim=512,
        dropout=0.2,
        freeze_ratio=0.5,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.freeze_ratio = freeze_ratio

        # 1) BERT (or other) backbone
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # 2) Freeze partial layers
        self._freeze_partial_bert()

        # 3) Final projection (using GELU)
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_size, output_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

    def _freeze_partial_bert(self):
        all_named_params = list(self.bert.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)

        for i, (name, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token
        cls_token = outputs.last_hidden_state[:, 0]
        return self.proj(cls_token)


def clip_style_loss(
    video_features: torch.Tensor, text_features: torch.Tensor, log_temp: torch.Tensor
) -> torch.Tensor:
    """
    Compute the CLIP-style cross-entropy loss over video-text pairs.
    """
    # L2 normalize
    video_features = F.normalize(video_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    # similarity => [B, B]
    similarity_matrix = torch.matmul(video_features, text_features.t())

    # temperature
    temp = torch.exp(log_temp)
    logits = similarity_matrix / temp

    # standard targets
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return (loss_i2t + loss_t2i) * 0.5