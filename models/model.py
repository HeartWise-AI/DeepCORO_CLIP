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

    - `freeze_ratio` ∈ [0..1] => fraction of backbone layers to keep frozen.
      Example:
        freeze_ratio=1.0 => unfreeze everything,
        freeze_ratio=0.0 => freeze all,
        freeze_ratio=0.8 => freeze bottom 80% of parameters, unfreeze top 20%.
    """

    def __init__(
        self,
        backbone="mvit",
        input_channels=3,
        num_frames=16,
        pretrained=True,
        output_dim=512,
        dropout=0.2,
        freeze_ratio=0.8,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_channels = input_channels
        self.num_frames = num_frames
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.freeze_ratio = freeze_ratio

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

        # Projection from backbone dimension -> output_dim
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(in_features, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
        )

        # Aggregator: small transformer to combine multiple segments
        self.aggregator = VideoAggregator(
            embedding_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            use_positional_encoding=True
        )

        # Freeze partial layers
        self._freeze_partial_layers()

    def _freeze_partial_layers(self):
        """
        Freeze a fraction (1 - freeze_ratio) of the backbone's layers
        (from bottom to top), leaving top fraction = freeze_ratio un-frozen.
        """
        all_named_params = list(self.model.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)

        # Freeze the bottom portion, keep top `train_count` trainable
        for i, (name, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects shape: [B, N, T, H, W, C]
            B= batch size,
            N= # of segments or videos per study,
            T= #frames,
            H= height,
            W= width,
            C= channels=3
        We'll reorder => [B,N,C,T,H,W], flatten => [B*N, C, T, H, W],
        pass through the backbone, then aggregator => [B, output_dim].
        """
        # Reorder last dimension (C) to after N => [B, N, C, T, H, W]
        x = x.permute(0, 1, 5, 2, 3, 4)  # Now shape: [B, N, 3, T, H, W]
        B, N, C, T, H, W = x.shape
        
        # Flatten => [B*N, 3, T, H, W]
        x = x.view(B*N, C, T, H, W)
        
        # Pass through backbone => [B*N, in_features]
        x = self.model(x)
        # Then projection => [B*N, output_dim]
        x = self.proj(x)
        
        # Reshape => [B, N, self.output_dim]
        x = x.view(B, N, self.output_dim)

        # aggregator => [B, output_dim]
        x = self.aggregator(x)

        return x


class TextEncoder(nn.Module):
    """
    Text encoder with optional dropout & partial-freeze.

    - `freeze_ratio`: fraction of BERT layers to keep trainable (the rest are frozen).
    - final projection uses dropout to help with regularization.
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

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Freeze a portion of BERT's encoder layers
        self._freeze_partial_bert()

        # Final projection
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_size, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
        )

    def _freeze_partial_bert(self):
        """
        Freeze the bottom portion of the BERT parameters, leaving
        a fraction `freeze_ratio` trainable from the top.
        """
        all_named_params = list(self.bert.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)

        for i, (name, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token: outputs.last_hidden_state[:, 0]
        cls_token = outputs.last_hidden_state[:, 0]
        return self.proj(cls_token)


def clip_style_loss(
    video_features: torch.Tensor, text_features: torch.Tensor, log_temp: torch.Tensor
) -> torch.Tensor:
    """
    Compute the CLIP-style cross-entropy loss over video-text pairs.
    """
    video_features = F.normalize(video_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    similarity_matrix = torch.matmul(video_features, text_features.t())

    temp = torch.exp(log_temp)
    logits = similarity_matrix / temp

    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return (loss_i2t + loss_t2i) / 2.0