"""Model definitions for DeepCORO_CLIP."""

import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, r3d_18
from torch.amp.autocast_mode import autocast

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
        aggregator_depth: int = 2,
        aggregate_videos_tokens: bool = True,
        per_video_pool: bool = False,
    ):
        """Initialize the video encoder.

        Args:
            backbone (str): Name of the backbone model to use ("mvit", "r3d", "x3d_s", or "x3d_m")
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_frames (int): Number of frames in the input video
            pretrained (bool): Whether to use pretrained weights
            output_dim (int): Output dimension of the encoder
            aggregate_videos_tokens (bool): If False, the transformer-based aggregator is skipped and the
                per-segment embeddings ``[B, N, D]`` are returned. This is useful in multi-
                instance learning settings where another model is responsible for aggregating
                over the *N* dimension.
        """
        super().__init__()
        self.backbone: str = backbone
        self.input_channels: int = input_channels
        self.num_frames: int = num_frames
        self.output_dim: int = output_dim
        self.dropout: float = dropout
        self.freeze_ratio: float = freeze_ratio

        # Add embedding_dim property
        self._embedding_dim: int = output_dim

        # 1) Build backbone
        if backbone == "mvit":
            # Load the pretrained MViT v2 S
            self.model: nn.Module = mvit_v2_s(weights="KINETICS400_V1" if pretrained else None)
            # Start with a sane default. We will try to infer the real value
            # from the last ``nn.Linear`` layer in ``self.model.head``.
            in_features = 768
            for layer in reversed(self.model.head):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break

            # Replace classification head with an identity mapping so that the
            # original `forward()` no longer collapses tokens.  We will also
            # monkey-patch a new `forward_features()` method that exposes the
            # full token sequence produced **after** the transformer blocks
            # but **before** the classification logic.
            self.model.head = nn.Identity()  # type: ignore[assignment]
        
        elif backbone == "r3d":
            self.model: nn.Module = r3d_18(pretrained=pretrained)
            in_features: int = self.model.fc.in_features
            # Replace the classifier with identity. Annotate with *ignore* for
            # static type checkers that assume ``fc`` is always ``nn.Linear``.
            self.model.fc = nn.Identity()  # type: ignore[assignment]
            
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
        # 3) Enhanced aggregator (can be optionally disabled)
        self.aggregator = EnhancedVideoAggregator(
            embedding_dim=output_dim,
            num_heads=num_heads,
            dropout=self.dropout,
            use_positional_encoding=True,
            aggregator_depth=aggregator_depth
        )
                
        # 4) Freeze partial layers
        self._freeze_partial_layers()

        # Whether to apply the internal aggregator in the forward pass.
        self._apply_aggregator: bool = aggregate_videos_tokens
        self._per_video_pool: bool = per_video_pool

        # ------------------------------------------------------------------
        # Monkey-patch `forward_features` on the instance *once*.
        # TorchVision's MViT does not expose such a helper; we recreate
        # the internal steps from its `forward()` implementation.
        # ------------------------------------------------------------------
        from types import MethodType
        from torchvision.models.video.mvit import _unsqueeze  # type: ignore

        def _forward_features_mvit(self_mvit, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            """Return the token sequence [B, L, C] without pooling."""
            # (B, C, T, H, W) ➀ ensure temporal dim present
            x = _unsqueeze(x, 5, 2)[0]
            # ➁ Patchify then flatten spatial+temporal dims
            x = self_mvit.conv_proj(x)  # [B, C', T', H', W']
            x = x.flatten(2).transpose(1, 2)  # [B, L, C'] where L = T'·H'·W'

            # ➂ Add positional encoding
            x = self_mvit.pos_encoding(x)

            # ➃ Pass through transformer blocks
            thw = (self_mvit.pos_encoding.temporal_size,) + self_mvit.pos_encoding.spatial_size
            for blk in self_mvit.blocks:
                x, thw = blk(x, thw)

            # ➄ Final layer norm; *no* class-token selection / pooling
            x = self_mvit.norm(x)  # [B, L, C_final]
            return x

        # Attach to the specific instance so it does not affect other MViT objects.
        self.model.forward_features = MethodType(_forward_features_mvit, self.model)  # type: ignore[attr-defined]

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

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return token-level features from the underlying video backbone.

        The output shape is always ``[B*N, L, F]`` where
          • ``B`` – batch size in the original mini-batch,
          • ``N`` – number of video segments per study (flattened in the caller),
          • ``L`` – number of tokens / patch features. For models that do not
                     expose patch-tokens (e.g. ResNets, X3D, …) we treat the
                     single global feature vector as *one* token (so L = 1).
          • ``F`` – backbone feature dimension (*in_features*).
        """
        # ------------------------------------------------------------------
        # Mixed-precision safety: Certain backbone operations (especially large
        # Transformer attentions) are prone to numerical overflow when run in
        # fp16.  We therefore *always* execute the backbone forward in full
        # precision regardless of the surrounding autocast context.  This
        # incurs only a modest memory overhead yet eliminates NaN/Inf issues
        # observed when training with AMP.
        # ------------------------------------------------------------------
        with autocast("cuda", enabled=False):
            if self.backbone == "mvit" and hasattr(self.model, "forward_features"):
                # TorchVision's Multi-Scale ViT exposes forward_features that
                # returns the token sequence **before** classification pooling.
                feats = self.model.forward_features(x.float())  # type: ignore[attr-defined]
                # Typical return shape: [B*N, L, F] (no CLS token).
                if feats.ndim == 2:
                    # Edge-case: some versions may still pool -> make it a token.
                    feats = feats.unsqueeze(1)  # [B*N, 1, F]
            else:
                # Fallback – call the regular forward() and treat the global
                # pooled output as a single token so that downstream components
                # (projection / aggregator) see a consistent shape.
                feats = self.model(x.float())
                if feats.ndim == 1:
                    # In the unlikely event we get a 1-D tensor per sample.
                    feats = feats.unsqueeze(0)
                if feats.ndim == 2:
                    feats = feats.unsqueeze(1)  # [B*N, 1, F]

        return feats  # [B*N, L, F]

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

        if x.ndim == 5:
            # Received [B, T, H, W, C] => treat as single video per study.
            x = x.unsqueeze(1)  # -> [B, 1, T, H, W, C]

        if x.ndim == 7:
            # Workaround for unexpected 7D input.
            # The method expects 6D [B, D1, D2, T_actual, H_actual, W_actual, C_actual]
            # and D1, D2 should be combined into the 'N' dimension.
            s = x.shape
            x = x.view(s[0], s[1] * s[2], s[3], s[4], s[5], s[6])
            # Now x should be 6D: [B, N_combined, T, H, W, C]
        
        # Reorder last dimensio n (C) to after N => [B, N, C, T, H, W]
        x = x.permute(0, 1, 5, 2, 3, 4)  # Now shape: [B, N, 3, T, H, W]

        B, N, C, T, H, W = x.shape

        # Flatten => [B*N, 3, T, H, W]
        x = x.view(B*N, C, T, H, W)

        # ------------------------------------------------------------------
        # 1) Backbone ⇨ token sequences (before projection)
        # ------------------------------------------------------------------
        token_feats = self._extract_backbone_features(x)  # [B*N, L, in_features]

        # ------------------------------------------------------------------
        # 2) Linear projection (shared across all tokens)
        # ------------------------------------------------------------------
        # ``nn.Sequential`` modules in ``self.proj`` operate on the last dim, so
        # we can call them directly on a 3-D tensor.
        token_feats = self.proj(token_feats)  # [B*N, L, output_dim]

        # ------------------------------------------------------------------
        # 3) Reshape → [B, N*L, output_dim] so that the aggregator treats every
        #    patch token from every segment as an element in the sequence.
        # ------------------------------------------------------------------
        BNL, L, D_out = token_feats.shape  # BNL = B * N
        token_feats = token_feats.view(B, N, L, D_out)

#       print(f"token_feats.shape: {token_feats.shape}")
        if self._apply_aggregator:
            # Before passing to aggregator, convert to exactly N tokens. This
            # preserves backward-compatibility with existing training code &
            # tests that expect a study-level pooling over videos rather than
            # patches.
            feats = token_feats.mean(dim=2)  # [B, N, D_out]

            orig_dtype = feats.dtype
            with autocast('cuda', enabled=False):
                out = self.aggregator(feats.float())
            return out.to(orig_dtype)

        # Aggregator disabled → return either per-video or per-patch tokens
        if self._per_video_pool:
            print("Per-video pooling") 
            print(f"token_feats.shape: {token_feats.shape}")
            feats = token_feats.mean(dim=2)  # [B, N, D_out]
            print(f"feats.shape: {feats.shape}")
        else:
            print("Per-patch pooling")
            feats = token_feats.reshape(B, N * L, D_out)  # [B, N_tokens, D]
            print(f"feats.shape: {feats.shape}")
            368

        return feats



