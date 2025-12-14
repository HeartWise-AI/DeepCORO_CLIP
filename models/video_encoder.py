"""Model definitions for DeepCORO_CLIP."""

from typing import Dict, Tuple

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

        # LayerNorm to stabilize features before aggregator (prevents magnitude drift)
        self.pre_agg_norm = nn.LayerNorm(output_dim)

        # 4) Freeze partial layers
        self._freeze_partial_layers()

        # 5) Auxiliary classifier to predict the main coronary tree
        self.tree_classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, 1),
        )

        # Whether to apply the internal aggregator in the forward pass.
        self._apply_aggregator: bool = aggregate_videos_tokens
        self._per_video_pool: bool = per_video_pool
        self._warned_non_finite: bool = False  # Emit at most one warning per process
        self._last_forward_had_non_finite: bool = False
        self._nan_replacement_std: float = 1e-3
        self._nan_fallback_by_shape: Dict[
            Tuple[str, int | None, Tuple[int, ...]], torch.Tensor
        ] = {}

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
    def get_tokens(self, x: torch.Tensor, mode: str = "patch"):
        self._apply_aggregator = False
        self._per_video_pool = (mode == "video")
        return self.forward(x)["video_embeds"]
    
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        compute_tree_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Run the video branch in full precision even when global AMP is enabled.
        This prevents half-precision overflow inside the backbone/projection,
        which previously manifested as NaNs in the contrastive loss.
        """
        if x.is_cuda and torch.is_autocast_enabled():
            with autocast("cuda", enabled=False):
                video_embeds = self._forward_impl(x.float())
        else:
            video_embeds = self._forward_impl(x)

        outputs: dict[str, torch.Tensor] = {"video_embeds": video_embeds}
        if compute_tree_logits:
            outputs["tree_logits"] = self.classify_main_structure(video_embeds)
        return outputs

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
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

        self._last_forward_had_non_finite = False

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
        
        # Reorder last dimension (C) to after N => [B, N, C, T, H, W]
        x = x.permute(0, 1, 5, 2, 3, 4)  # Now shape: [B, N, 3, T, H, W]

        B, N, C, T, H, W = x.shape

        # Flatten => [B*N, 3, T, H, W]
        x = x.view(B * N, C, T, H, W)

        # ------------------------------------------------------------------
        # 1) Backbone ⇨ token sequences (before projection)
        # ------------------------------------------------------------------
        token_feats = self._extract_backbone_features(x)  # [B*N, L, in_features]

        # Guard against NaN from backbone (MViT attention can overflow)
        if not torch.isfinite(token_feats).all():
            print("[VideoEncoder] WARNING: Non-finite values from backbone, replacing")
            # Use small random noise instead of 0 to preserve variance for LayerNorm
            finite_mask = torch.isfinite(token_feats)
            if finite_mask.any():
                # Use mean/std of finite values for replacement
                finite_vals = token_feats[finite_mask]
                replacement = torch.randn_like(token_feats) * finite_vals.std() + finite_vals.mean()
            else:
                # Fallback to small random noise
                replacement = torch.randn_like(token_feats) * 0.1
            token_feats = torch.where(finite_mask, token_feats, replacement)
            # Clamp to prevent extreme values from corrupting projection
            token_feats = torch.clamp(token_feats, min=-100.0, max=100.0)

        # ------------------------------------------------------------------
        # 2) Linear projection (shared across all tokens)
        # ------------------------------------------------------------------
        # ``nn.Sequential`` modules in ``self.proj`` operate on the last dim, so
        # we can call them directly on a 3-D tensor.
        token_feats = self.proj(token_feats.float())  # [B*N, L, output_dim]

        # Clamp projection output to prevent GELU overflow (NaN when |x| > ~50)
        token_feats = torch.clamp(token_feats, min=-50.0, max=50.0)

        # Early NaN detection with recovery - catch issues before mean pooling propagates them
        if not torch.isfinite(token_feats).all():
            print("[VideoEncoder] WARNING: Non-finite values after projection, replacing")
            # Use variance-preserving replacement to avoid LayerNorm NaN
            finite_mask = torch.isfinite(token_feats)
            if finite_mask.any():
                finite_vals = token_feats[finite_mask]
                replacement = torch.randn_like(token_feats) * finite_vals.std().clamp(min=0.1) + finite_vals.mean()
                replacement = torch.clamp(replacement, min=-50.0, max=50.0)
            else:
                replacement = torch.randn_like(token_feats) * 0.1
            token_feats = torch.where(finite_mask, token_feats, replacement)

        # ------------------------------------------------------------------
        # 3) Reshape → [B, N*L, output_dim] so that the aggregator treats every
        #    patch token from every segment as an element in the sequence.
        # ------------------------------------------------------------------
        BNL, L, D_out = token_feats.shape  # BNL = B * N
        token_feats = token_feats.view(B, N, L, D_out)

        if self._apply_aggregator:
            # Before passing to aggregator, convert to exactly N tokens. This
            # preserves backward-compatibility with existing training code &
            # tests that expect a study-level pooling over videos rather than
            # patches.
            feats = token_feats.mean(dim=2)  # [B, N, D_out]

            # Normalize features before aggregator to stabilize magnitudes
            feats = self.pre_agg_norm(feats)

            with autocast("cuda", enabled=False):
                out = self.aggregator(feats.float())
            return self._sanitize_tensor(out, context="aggregated tokens")

        # Aggregator disabled → return either per-video or per-patch tokens
        if self._per_video_pool:
            feats = token_feats.mean(dim=2)  # [B, N, D_out]
            feats = self.pre_agg_norm(feats)  # Normalize for consistency
        else:
            feats = token_feats.reshape(B, N * L, D_out)  # [B, N_tokens, D]

        return self._sanitize_tensor(feats, context="token features")

    def _sanitize_tensor(self, tensor: torch.Tensor, *, context: str) -> torch.Tensor:
        """
        Replace NaN/inf values with cached fallbacks or small random noise so
        downstream normalization never encounters non-finite inputs.

        IMPORTANT: This method preserves gradient flow. The previous implementation
        used detach() which completely broke gradients through the video encoder,
        causing NaN gradients from batch 1.
        """
        if torch.isfinite(tensor).all():
            self._last_forward_had_non_finite = False
            self._update_nan_fallback_cache(tensor)
            return tensor

        self._last_forward_had_non_finite = True
        if not self._warned_non_finite:
            print(
                f"[VideoEncoder] Detected non-finite values in {context}; "
                "replacing them while preserving gradient flow."
            )
            self._warned_non_finite = True

        # Use torch.where to replace non-finite values while preserving gradients
        # for the finite values. This is critical for backward pass.
        finite_mask = torch.isfinite(tensor)

        if tensor.ndim < 2:
            # For 1D tensors, just replace non-finite with zeros
            return torch.where(finite_mask, tensor, torch.zeros_like(tensor))

        key = self._nan_fallback_key(tensor)

        # Get or create fallback values
        fallback = self._nan_fallback_by_shape.get(key) if key else None
        if fallback is None:
            fallback = torch.zeros(tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
            if fallback.numel() > 0:
                nn.init.normal_(fallback, std=self._nan_replacement_std)
            if key is not None:
                self._nan_fallback_by_shape[key] = fallback.clone()

        # Ensure fallback is on the same device and has correct dtype
        fallback = fallback.to(device=tensor.device, dtype=tensor.dtype)

        # Expand fallback to match tensor shape for broadcasting
        fallback_expanded = fallback.unsqueeze(0).expand_as(tensor)

        # Use torch.where to preserve gradients for finite values
        # Non-finite values get replaced with fallback (no gradient contribution)
        sanitized = torch.where(finite_mask, tensor, fallback_expanded)

        # Update cache with mean of finite rows for future fallbacks
        row_mask = finite_mask.reshape(tensor.shape[0], -1).all(dim=1)
        if row_mask.any() and key is not None:
            # Only update cache from fully-finite rows
            finite_rows = sanitized[row_mask]
            self._nan_fallback_by_shape[key] = finite_rows.detach().mean(dim=0).clone()

        return sanitized

    def _nan_fallback_key(
        self, tensor: torch.Tensor
    ) -> Tuple[str, int | None, Tuple[int, ...]] | None:
        """Generate a cache key for fallback embeddings of a given shape/device."""
        if tensor.ndim < 2:
            return None
        device = tensor.device
        shape = tuple(int(dim) for dim in tensor.shape[1:])
        return (device.type, device.index, shape)

    def _update_nan_fallback_cache(self, tensor: torch.Tensor) -> None:
        """Store the latest finite embedding template for future fallbacks."""
        key = self._nan_fallback_key(tensor)
        if key is None:
            return
        fallback = tensor.detach().mean(dim=0).clone()
        self._nan_fallback_by_shape[key] = fallback

    def had_non_finite_last_forward(self) -> bool:
        """Return True if the previous forward pass sanitized non-finite outputs."""
        return self._last_forward_had_non_finite

    def classify_main_structure(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict logits for the left/right coronary tree from embeddings."""
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.ndim > 2:
            embeddings = embeddings.view(-1, embeddings.shape[-1])
        with autocast("cuda", enabled=False):
            logits = self.tree_classifier(embeddings.float())
        return logits.squeeze(-1)
