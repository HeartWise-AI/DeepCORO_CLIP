"""Model definitions for DeepCORO_CLIP."""

import torch
import torch.nn as nn
from torchvision.models.video import mvit_v1_b, r3d_18
from transformers import AutoModel, AutoTokenizer

from utils.registry import register_model



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

@register_model("video_encoder")
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
        ### Freeze
        # Freeze the weights of the pretrained model except the head
        frozen_layers = []
        unfrozen_layers = []
        for name, param in self.model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
                frozen_layers.append(name)
            else:
                unfrozen_layers.append(name)

        print(f"Frozen layers: {frozen_layers}")
        print(f"Unfrozen layers: {unfrozen_layers}")

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




