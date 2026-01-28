"""Model definitions for DeepCORO_CLIP."""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict
from torchvision.models.video import mvit_v2_s, r3d_18
from torch.amp.autocast_mode import autocast

from utils.registry import ModelRegistry
from models.video_aggregator import EnhancedVideoAggregator
from models.attention_pool import AttentionPool, AttentionPoolWithCLS
from models.rope_3d import Rope3D



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
        token_pooling_mode: str = "mean",
        attention_pool_heads: int = 8,
        attention_pool_dropout: float = 0.1,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        rope_temporal_scale: float = 1.0,
        rope_normalize_mode: str = "separate",
        use_cls_token: bool = False,
        multi_video_cls_aggregation: str = "mean",  # "mean", "attention", or "first"
        encoder_path: Optional[str] = None,  # Path to pretrained encoder checkpoint
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
        
        # Store RoPE configuration
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.rope_temporal_scale = rope_temporal_scale
        self.rope_normalize_mode = rope_normalize_mode
        
        # Store CLS token configuration
        self.use_cls_token = use_cls_token or token_pooling_mode == "cls_token"
        self.multi_video_cls_aggregation = multi_video_cls_aggregation
        self.encoder_path = encoder_path

        # 1) Build backbone
        if backbone == "mvit" or backbone == "mvit_rope":
            # For mvit_rope, always load without pretrained weights initially
            # We'll load the checkpoint weights later
            if backbone == "mvit_rope":
                self.model: nn.Module = mvit_v2_s(weights=None)
                # Force RoPE to be enabled for mvit_rope
                self.use_rope = True
                print(f"[VideoEncoder] Initialized mvit_rope backbone (without pretrained weights)")
                if encoder_path:
                    print(f"[VideoEncoder] Will load encoder checkpoint from: {encoder_path}")
            else:
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
            
            # Initialize RoPE if enabled for MViT
            if self.use_rope:
                # MViT has blocks with different head dimensions
                # Most blocks use head_dim=96 which is divisible by 6
                # We'll create multiple RoPE modules for different configurations
                # Store as regular dict (not nn.ModuleDict) since RoPE has no parameters
                rope_modules = {}
                
                # Common configurations in MViT v2 S
                configs = [
                    (4, 96),   # Blocks 3-13: 4 heads, 96 dim/head
                    (8, 96),   # Blocks 14-15: 8 heads, 96 dim/head
                ]
                
                for num_heads, head_dim in configs:
                    if head_dim % 6 == 0:  # Check 3D RoPE compatibility
                        key = f"{num_heads}_{head_dim}"
                        rope_module = Rope3D(
                            embed_dim=num_heads * head_dim,
                            num_heads=num_heads,
                            temporal_base=self.rope_base,
                            spatial_base=self.rope_base,
                            temporal_scale=self.rope_temporal_scale,
                            normalize_mode=self.rope_normalize_mode
                        )
                        # Set to eval mode (RoPE has no trainable params)
                        rope_module.eval()
                        rope_modules[key] = rope_module
                        print(f"[VideoEncoder] Initialized 3D RoPE for {num_heads} heads, {head_dim} dim/head")
                
                # Store as a non-parameter attribute to avoid DDP tracking
                self._rope_modules = rope_modules
                print(f"[VideoEncoder] Created {len(rope_modules)} RoPE modules for MViT")
            else:
                self._rope_modules = None
                print(f"[VideoEncoder] RoPE DISABLED (use_rope={self.use_rope})")
            
            # Initialize CLS token if enabled for MViT
            if self.use_cls_token:
                # MViT internal dimension is 768 for v2 S
                cls_dim = 768
                self.cls_token = nn.Parameter(torch.randn(1, 1, cls_dim) * 0.02)
                print(f"[VideoEncoder] Added learnable CLS token (dim={cls_dim})")
            else:
                self.cls_token = None
        
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

        # 3.5) Load encoder checkpoint if provided
        if self.encoder_path:
            self._load_encoder_checkpoint()

        # 4) Freeze partial layers
        self._freeze_partial_layers()

        # Whether to apply the internal aggregator in the forward pass.
        self._apply_aggregator: bool = aggregate_videos_tokens
        self._per_video_pool: bool = per_video_pool
        self.token_pooling_mode: str = token_pooling_mode
        
        # Initialize attention pooling if needed
        self.attention_pool = None
        if self.token_pooling_mode == "attention":
            self.attention_pool = AttentionPool(
                embed_dim=output_dim,
                num_heads=attention_pool_heads,
                dropout=attention_pool_dropout
            )
            print(f"[VideoEncoder.__init__] Created AttentionPool with mode='{self.token_pooling_mode}'")
        elif self.token_pooling_mode == "cls_token":
            self.attention_pool = AttentionPoolWithCLS(
                embed_dim=output_dim,
                num_heads=attention_pool_heads,
                dropout=attention_pool_dropout
            )
            print(f"[VideoEncoder.__init__] Created AttentionPoolWithCLS for mode='{self.token_pooling_mode}'")
        else:
            print(f"[VideoEncoder.__init__] NOT creating AttentionPool, mode='{self.token_pooling_mode}'")

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
        
        # If RoPE is enabled for MViT, monkey-patch attention blocks
        if backbone == "mvit" and self.use_rope and self._rope_modules is not None:
            self._patch_mvit_attention_for_rope()

    def _patch_mvit_attention_for_rope(self):
        """Monkey-patch MViT's MultiScaleAttention to apply 3D RoPE."""
        from types import MethodType
        import torch.nn.functional as F
        from torchvision.models.video.mvit import _add_rel_pos
        
        # Store reference to rope modules
        rope_modules = self._rope_modules
        
        def patched_forward(self_attn, x, thw):
            """Patched forward for MultiScaleAttention with RoPE.
            
            Args:
                x: Input tensor [B, N, C]
                thw: Tuple of (T, H, W) dimensions
            """
            B, N, C = x.shape
            T, H, W = thw
            
            # Get actual dimensions from the attention module
            num_heads = self_attn.num_heads
            head_dim = self_attn.head_dim
            output_dim = self_attn.output_dim
            
            # Standard MViT attention computation
            # Project to Q, K, V  
            qkv = self_attn.qkv(x)
            # The output dimension is output_dim * 3 (for q, k, v)
            qkv = qkv.reshape(B, N, 3, num_heads, output_dim // num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Apply pooling if needed (MViT specific)
            if hasattr(self_attn, 'pool_q') and self_attn.pool_q is not None:
                q, thw_q = self_attn.pool_q(q, thw)
            else:
                thw_q = thw
                
            if hasattr(self_attn, 'pool_k') and self_attn.pool_k is not None:
                k, thw_k = self_attn.pool_k(k, thw)
            else:
                thw_k = thw
                
            if hasattr(self_attn, 'pool_v') and self_attn.pool_v is not None:
                v, thw_v = self_attn.pool_v(v, thw)
            else:
                thw_v = thw
            
            # Apply 3D RoPE to Q and K after pooling
            T_q, H_q, W_q = thw_q
            T_k, H_k, W_k = thw_k
            
            # Detect CLS tokens (N > T*H*W means we have special tokens)
            expected_q = T_q * H_q * W_q
            expected_k = T_k * H_k * W_k
            n_special_q = max(0, q.shape[2] - expected_q)  # Usually 0 or 1
            n_special_k = max(0, k.shape[2] - expected_k)
            
            # Apply RoPE if we have a matching module
            actual_head_dim = q.shape[-1]
            if rope_modules is not None and actual_head_dim % 6 == 0:
                rope_key = f"{num_heads}_{actual_head_dim}"
                if rope_key in rope_modules:
                    try:
                        rope_module = rope_modules[rope_key]
                        # Handle Q and K separately if they have different dimensions
                        if q.shape[2] == k.shape[2] and (T_q, H_q, W_q) == (T_k, H_k, W_k):
                            # Same dimensions, apply RoPE to both
                            q, k = rope_module(q, k, T_q, H_q, W_q, n_special=n_special_q)
                        else:
                            # Different dimensions after pooling, apply separately
                            # Apply to Q
                            q_rot, _ = rope_module(q, q, T_q, H_q, W_q, n_special=n_special_q)
                            q = q_rot
                            # Apply to K with its dimensions
                            if (T_k * H_k * W_k + n_special_k) == k.shape[2]:
                                _, k_rot = rope_module(k, k, T_k, H_k, W_k, n_special=n_special_k)
                                k = k_rot
                        # Log only once per training to avoid spam
                        if not hasattr(self_attn, '_rope_applied_logged'):
                            cls_info = f" (with {n_special_q} CLS)" if n_special_q > 0 else ""
                            print(f"[RoPE] Applied to block with {num_heads} heads, head_dim={actual_head_dim}, grid=({T_q}x{H_q}x{W_q}){cls_info}")
                            self_attn._rope_applied_logged = True
                    except Exception as e:
                        # Skip RoPE for this block if error
                        if not hasattr(self_attn, '_rope_error_logged'):
                            print(f"[RoPE] Error in block: {e}")
                            self_attn._rope_error_logged = True
            
            # Compute attention
            # MViT uses 'scaler' not 'scale'
            scale = 1.0 / math.sqrt(self_attn.head_dim) if hasattr(self_attn, 'head_dim') else 1.0 / math.sqrt(q.shape[-1])
            attn = (q @ k.transpose(-2, -1)) * scale
            
            # Apply relative position bias (fixes DDP unused parameters)
            if hasattr(self_attn, 'rel_pos_h') and hasattr(self_attn, 'rel_pos_w'):
                # Use MViT's relative position bias function
                attn = _add_rel_pos(
                    attn,
                    q,  # Need to pass q tensor
                    thw_q,  # q dimensions  
                    thw_k,  # k dimensions
                    self_attn.rel_pos_h,
                    self_attn.rel_pos_w,
                    self_attn.rel_pos_t if hasattr(self_attn, 'rel_pos_t') else None
                )
            
            attn = attn.softmax(dim=-1)
            attn = self_attn.attn_drop(attn) if hasattr(self_attn, 'attn_drop') else attn
            
            # Apply to values
            x = attn @ v  # [B, num_heads, N_q, head_dim]
            x = x.transpose(1, 2)  # [B, N_q, num_heads, head_dim]
            
            # Reshape to [B, N_q, output_dim]
            # Note: output_dim may be different from C (input dim)
            x = x.reshape(B, x.shape[1], -1)
            x = self_attn.project(x)  # MViT uses 'project' not 'proj'
            
            return x, thw_q
        
        # Patch all MultiScaleAttention blocks in the model
        for block in self.model.blocks:
            if hasattr(block, 'attn'):
                # Store original forward for potential restoration
                block.attn._original_forward = block.attn.forward
                # Apply patched forward
                block.attn.forward = MethodType(patched_forward, block.attn)
        
        print(f"[VideoEncoder] Patched {len(self.model.blocks)} MViT attention blocks for 3D RoPE")
    
    def _load_encoder_checkpoint(self):
        """Load encoder checkpoint weights from the specified path."""
        if not self.encoder_path:
            return

        import os
        print(f"[VideoEncoder] ========== ENCODER CHECKPOINT LOADING ==========")
        print(f"[VideoEncoder] Encoder checkpoint path: {self.encoder_path}")

        if not os.path.exists(self.encoder_path):
            print(f"[VideoEncoder] ⚠️  WARNING: Encoder checkpoint path does not exist!")
            print(f"[VideoEncoder] ========================================")
            return

        print(f"[VideoEncoder] ✅ Encoder checkpoint file found!")
        print(f"[VideoEncoder] Loading encoder checkpoint weights...")
        checkpoint = torch.load(self.encoder_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter out MVM-specific layers if loading from MVM pretraining
        encoder_state_dict = {}
        for k, v in state_dict.items():
            # Skip MVM decoder, head layers, and mask tokens
            if not any(skip in k for skip in ["decoder", "mvm", "mask_token", "head"]):
                # Remove 'module.' prefix if present (from DDP)
                key = k.replace("module.", "")
                # Also handle 'model.' prefix
                key = key.replace("model.", "")
                encoder_state_dict[key] = v

        # Load the weights
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(encoder_state_dict, strict=False)
            print(f"[VideoEncoder] ✅ ENCODER CHECKPOINT LOADED SUCCESSFULLY!")
            print(f"[VideoEncoder] Loaded {len(encoder_state_dict)} parameter tensors")
            if missing_keys:
                print(f"[VideoEncoder] Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[VideoEncoder] Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
            print(f"[VideoEncoder] ========================================")
        except Exception as e:
            print(f"[VideoEncoder] ❌ Error loading checkpoint: {e}")
            print(f"[VideoEncoder] Continuing with randomly initialized weights")
            print(f"[VideoEncoder] ========================================")

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
    
    def update_freeze_ratio(self, new_freeze_ratio: float):
        """
        Dynamically update the freeze ratio during training.
        
        Args:
            new_freeze_ratio: New freeze ratio (0.0 = all trainable, 1.0 = all frozen)
        """
        # Only update if the ratio actually changed
        if abs(self.freeze_ratio - new_freeze_ratio) < 1e-6:
            return  # No change needed
            
        old_freeze_ratio = self.freeze_ratio
        self.freeze_ratio = new_freeze_ratio
        
        # First, unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Then apply the new freeze ratio
        all_named_params = list(self.model.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)
        
        # Freeze the bottom portion, keep top `train_count` trainable
        for i, (name, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False
        
        # Count trainable parameters for logging
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        # Only print if the ratio actually changed
        print(f"[VideoEncoder] Updated freeze_ratio from {old_freeze_ratio:.2f} to {new_freeze_ratio:.2f}")
        print(f"[VideoEncoder] Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_percent:.1f}%)")

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    def get_tokens(self, x: torch.Tensor, mode: str = "patch", return_dict: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Return patch tokens while keeping study-level aggregation available."""
        prev_apply = self._apply_aggregator
        prev_per_video = self._per_video_pool
        self._apply_aggregator = False
        self._per_video_pool = (mode == "video")
        features = self._compute_feature_dict(x, compute_aggregated=True)
        self._apply_aggregator = prev_apply
        self._per_video_pool = prev_per_video

        if return_dict:
            return {
                "patch_tokens": features["patch_tokens"],
                "per_video_tokens": features["per_video_tokens"],
                "token_grid": features["token_grid"],
                "study_features": features["study_features"],
            }

        if mode == "video":
            per_video = features["per_video_tokens"]
            if per_video.dim() == 3 and per_video.shape[1] == 1:
                return per_video.squeeze(1)
            return per_video

        return features["patch_tokens"]

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

    def _compute_feature_dict(self, x: torch.Tensor, compute_aggregated: bool) -> Dict[str, torch.Tensor]:
        if x.ndim == 5:
            x = x.unsqueeze(1)
        if x.ndim == 7:
            s = x.shape
            x = x.view(s[0], s[1] * s[2], s[3], s[4], s[5], s[6])

        x = x.permute(0, 1, 5, 2, 3, 4)
        B, N, C, T, H, W = x.shape
        x = x.view(B * N, C, T, H, W)

        token_feats = self._extract_backbone_features(x)
        token_feats = self.proj(token_feats)
        _, L, D_out = token_feats.shape
        token_feats = token_feats.view(B, N, L, D_out)

        per_video = self._pool_video_tokens(token_feats)
        patch_tokens = token_feats.reshape(B, N * L, D_out)

        study_features = None
        if compute_aggregated and self.aggregator is not None:
            study_features = self._aggregate_video_features(per_video)

        return {
            "token_grid": token_feats,
            "per_video_tokens": per_video,
            "patch_tokens": patch_tokens,
            "study_features": study_features,
        }

    def _pool_video_tokens(self, token_feats: torch.Tensor) -> torch.Tensor:
        B, N, L, _ = token_feats.shape
        if self.attention_pool is not None:
            pooled = []
            for i in range(N):
                video_tokens = token_feats[:, i, :, :]
                pooled.append(self.attention_pool(video_tokens).unsqueeze(1))
            return torch.cat(pooled, dim=1)
        return token_feats.mean(dim=2)

    def _aggregate_video_features(
        self,
        per_video: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_dtype = per_video.dtype
        with autocast('cuda', enabled=False):
            aggregated = self.aggregator(per_video.float(), mask=mask)
        return aggregated.to(orig_dtype)

    def forward(self, x: torch.Tensor, return_tokens: bool = False, token_mode: str = "patch") -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self._compute_feature_dict(x, compute_aggregated=self._apply_aggregator)

        if self._apply_aggregator and features["study_features"] is not None:
            primary = features["study_features"]
        elif self._per_video_pool:
            primary = features["per_video_tokens"]
            if primary.dim() == 3 and primary.shape[1] == 1:
                primary = primary.squeeze(1)
        else:
            primary = features["patch_tokens"]

        if not return_tokens:
            return primary

        if token_mode == "video":
            tokens = features["per_video_tokens"]
        elif token_mode == "grid":
            tokens = features["token_grid"]
        else:
            tokens = features["patch_tokens"]

        return primary, tokens
