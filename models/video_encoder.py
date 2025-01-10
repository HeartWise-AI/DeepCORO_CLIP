"""Model definitions for DeepCORO_CLIP."""

import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, r3d_18

from utils.registry import ModelRegistry



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

@ModelRegistry.register("video_encoder")
class VideoEncoder(nn.Module):
    """Video encoder model based on specified backbone."""

    def __init__(
        self, 
        backbone="mvit", 
        input_channels=3, 
        num_frames=16, 
        pretrained=True, 
        output_dim=512, 
        freeze_ratio=0.0
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
        self.freeze_ratio = freeze_ratio
        
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

        elif backbone == "r3d":
            self.model = r3d_18(pretrained=pretrained)
            in_features = 512  # R3D-18's hidden dimension
            self.model.fc = TransformerHead(dim_in=in_features, num_classes=output_dim)
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Projection from backbone dimension -> output_dim
        # (includes dropout to help regularize)
        self.proj = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.ReLU(inplace=True),
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
        for i, (_, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False


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




