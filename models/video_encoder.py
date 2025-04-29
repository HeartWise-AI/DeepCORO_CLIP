"""Model definitions for DeepCORO_CLIP."""

import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, r3d_18

from utils.registry import ModelRegistry
from models.video_aggregator import EnhancedVideoAggregator



@ModelRegistry.register("video_encoder")
class VideoEncoder(nn.Module):
    """Video encoder model based on specified backbone."""

    def __init__(
        self, 
        backbone: str = "mvit", 
        input_channels: int = 3, 
        num_frames: int = 16, 
        pretrained: bool = True, 
        output_dim: int = 512, 
        dropout: float = 0.2,
        num_heads: int = 4,
        freeze_ratio: float = 0.8,
        aggregator_depth: int = 2
    ):
        """Initialize the video encoder.

        Args:
            backbone (str): Name of the backbone model to use ("mvit", "r3d", "x3d_s", or "x3d_m")
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_frames (int): Number of frames in the input video
            pretrained (bool): Whether to use pretrained weights
            output_dim (int): Output dimension of the encoder
        """
        super().__init__()
        self.backbone: str = backbone
        self.input_channels: int = input_channels
        self.num_frames: int = num_frames
        self.output_dim: int = output_dim
        self.dropout: float = dropout
        self.freeze_ratio: float = freeze_ratio

        # 1) Build backbone
        if backbone == "mvit":
            # Load the pretrained MViT v2 S
            self.model: nn.Module = mvit_v2_s(weights="KINETICS400_V1" if pretrained else None)
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
            self.model: nn.Module = r3d_18(pretrained=pretrained)
            in_features: int = self.model.fc.in_features
            self.model.fc = nn.Identity()
            
        elif backbone in ["x3d_s", "x3d_m"]:
            # Load X3D model from torch.hub
            self.model = torch.hub.load('facebookresearch/pytorchvideo', backbone, pretrained=pretrained)
            # Get feature dimension from the head
            in_features = self.model.blocks[5].proj.in_features
            # Replace classification head with Identity
            self.model.blocks[5].proj = nn.Identity()
            self.model.blocks[5].activation = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
                
        # 2) Projection (use GELU instead of ReLU)
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(in_features, output_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )                
        # Check if output_dim is divisible by num_heads
        if output_dim % num_heads != 0:
            raise ValueError(f"Output dimension ({output_dim}) must be divisible by number of heads ({num_heads})")
        # 3) Enhanced aggregator
        self.aggregator = EnhancedVideoAggregator(
            embedding_dim=output_dim,
            num_heads=num_heads,
            dropout=self.dropout,
            use_positional_encoding=True,
            aggregator_depth=aggregator_depth
        )                
                
        # 4) Freeze partial layers
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
        feats = self.model(x)
        # Then projection => [B*N, output_dim]
        feats = self.proj(feats)
        
        # Reshape => [B, N, output_dim]
        feats = feats.view(B, N, self.output_dim)
        
        # aggregator => [B, output_dim]
        out = self.aggregator(feats)

        return out



