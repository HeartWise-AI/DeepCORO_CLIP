"""Model definitions for DeepCORO_CLIP."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mvit_v1_b, r3d_18
from transformers import AutoModel, AutoTokenizer


def get_tokenizer(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
    """Get the tokenizer with proper configuration.

    Args:
        model_name (str): Name of the pretrained model

    Returns:
        tokenizer: Configured tokenizer
    """
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
        """Initialize transformer head.

        Args:
            dim_in (int): Input dimension from backbone
            num_classes (int): Number of output classes/dimensions
            dropout (float): Dropout probability
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim_in]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class VideoEncoder(nn.Module):
    """Video encoder model based on specified backbone."""

    def __init__(
        self, backbone="mvit", input_channels=1, num_frames=16, pretrained=True, output_dim=512
    ):
        """Initialize the video encoder.

        Args:
            backbone (str): Name of the backbone model to use ("mvit" or "r3d")
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_frames (int): Number of frames in the input video
            pretrained (bool): Whether to use pretrained weights
            output_dim (int): Output dimension of the encoder
        """
        super().__init__()
        self.backbone = backbone
        self.input_channels = input_channels
        self.num_frames = num_frames
        self.output_dim = output_dim

        # Convert grayscale to 3 channels if needed
        if input_channels == 1:
            self.to_rgb = nn.Sequential(
                # First expand to 3 channels
                nn.Conv3d(1, 3, kernel_size=1, bias=False),
                nn.BatchNorm3d(3),
                nn.ReLU(inplace=True),
            )
        else:
            self.to_rgb = nn.Identity()

        # Load the specified backbone with pretrained weights if specified
        if backbone == "mvit":
            self.model = mvit_v1_b(weights="KINETICS400_V1" if pretrained else None)
            in_features = 768  # MViT v1's hidden dimension
            kinetics_classes = 400  # Number of Kinetics classes
            # Replace the classification head with a sequence of layers
            self.model.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, kinetics_classes),
            )
            # Add projection layer to match desired output dimension
            self.proj = nn.Sequential(
                nn.Linear(kinetics_classes, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(0.1),
            )
        elif backbone == "r3d":
            self.model = r3d_18(pretrained=pretrained)
            in_features = 512  # R3D-18's hidden dimension
            self.model.fc = TransformerHead(dim_in=in_features, num_classes=output_dim)
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze the weights of the pretrained model except the head
        frozen_layers = []
        unfrozen_layers = []
        for name, param in self.model.named_parameters():
            # Unfreeze the last 4 blocks
            if (
                "blocks.16" in name
                or "blocks.17" in name
                or "blocks.18" in name
                or "blocks.19" in name
                or name.startswith("head")
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"Frozen layers (VideoEncoder): {frozen_layers}")
        print(f"Unfrozen layers (VideoEncoder): {unfrozen_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the video encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, frames, height, width]

        Returns:
            torch.Tensor: Output features of shape [batch_size, output_dim]
        """

        # Input shape: [batch_size, channels, frames, height, width]
        # Backbone expects [batch_size, channels, time, height, width]

        # Convert grayscale to RGB if needed
        if self.input_channels == 1:
            # Take only the first channel if input has 3 channels
            if x.shape[1] == 3:
                x = x[:, 0:1]
            x = self.to_rgb(x)  # [B, 3, T, H, W]
        else:
            x = self.to_rgb(x)  # Pass through Identity if not grayscale

        # Forward through the backbone model
        x = self.model(
            x
        )  # Output shape: [batch_size, kinetics_classes] for mvit or [batch_size, output_dim] for r3d

        # Project to desired output dimension if needed
        x = self.proj(x)  # Output shape: [batch_size, output_dim]

        return x


class TextEncoder(nn.Module):
    """Text encoder model based on PubMedBERT."""

    def __init__(
        self,
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        output_dim=512,
    ):
        """Initialize the text encoder.

        Args:
            model_name (str): Name of the pretrained model to use
            output_dim (int): Output dimension to match video encoder
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim

        # Load model and get its config
        self.bert = AutoModel.from_pretrained(model_name)
        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last 6 encoder layers of BERT
        for layer in self.bert.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True

        config = self.bert.config

        # Project from BERT hidden size to match video encoder
        self.proj = nn.Linear(config.hidden_size, output_dim)

        # Print model configuration for debugging
        print(f"Initialized TextEncoder with:")
        print(f"  model_name: {model_name}")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  output_dim: {output_dim}")
        self.print_frozen_status()

    def print_frozen_status(self):
        """Print which layers are frozen/unfrozen in the text encoder."""
        frozen_layers = []
        unfrozen_layers = []
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                unfrozen_layers.append(name)
            else:
                frozen_layers.append(name)

        print(f"Frozen layers (TextEncoder): {frozen_layers}")
        print(f"Unfrozen layers (TextEncoder): {unfrozen_layers}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the text encoder.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, sequence_length]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Output features of shape [batch_size, output_dim]
        """

        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        # Get BERT features and take CLS token output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0]  # Take CLS token

        # Project to match video encoder dimension
        features = self.proj(features)

        return features


def clip_style_loss(
    video_features: torch.Tensor, text_features: torch.Tensor, log_temp: torch.Tensor
) -> torch.Tensor:
    """
    Compute the CLIP-style cross-entropy loss over video-text pairs.

    Args:
        video_features (torch.Tensor): Normalized video embeddings of shape [B, D]
        text_features (torch.Tensor): Normalized text embeddings of shape [B, D]
        log_temp (torch.Tensor): Logarithm of the temperature parameter, shape [1]

    Returns:
        torch.Tensor: A scalar loss value.
    """
    video_features = F.normalize(video_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    # Compute similarity matrix [B, B]
    similarity_matrix = torch.matmul(video_features, text_features.t())

    # Exponentiate log_temp to get the actual temperature (ensure positivity)
    temp = torch.exp(log_temp)
    logits = similarity_matrix / temp

    # Targets are simply [0, 1, 2, ..., B-1] since the i-th video corresponds to the i-th text.
    targets = torch.arange(logits.size(0), device=logits.device)

    # Compute cross-entropy loss for video->text (rows) and text->video (columns)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)

    # Final loss is the average
    loss = (loss_i2t + loss_t2i) / 2.0
    return loss
