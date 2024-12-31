import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import (
    mvit_v1_b,
    r3d_18,
    mvit_v2_s,
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
            # self.model.head is a Sequential. We find the final nn.Linear layer:
            in_features = None
            for layer in reversed(self.model.head):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
            # If we didn't find a linear, default to a known dimension (e.g., 768)
            if in_features is None:
                in_features = 768

            # Replace classification head with Identity
            self.model.head = nn.Identity()

        elif self.backbone == "r3d":
            self.model = r3d_18(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        # Projection from backbone dimension -> output_dim
        # (includes dropout to help regularize)
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(in_features, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
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
        # Pass through backbone
        x = self.model(x)        # shape: [batch_size, in_features]
        x = self.proj(x)         # final projection layers
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