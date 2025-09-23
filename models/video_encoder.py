"""Model definitions for DeepCORO_CLIP."""

import torch
import torch.nn as nn
import math
from torchvision.models.video import mvit_v2_s, r3d_18
from torch.amp.autocast_mode import autocast

from utils.registry import ModelRegistry
from models.video_aggregator import EnhancedVideoAggregator
from models.attention_pool import AttentionPool
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
        self.use_cls_token = use_cls_token
        self.multi_video_cls_aggregation = multi_video_cls_aggregation

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
    def get_tokens(self, x: torch.Tensor, mode: str = "patch"):
        self._apply_aggregator = False
        self._per_video_pool = (mode == "video")
        return self.forward(x)
    
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
        
        # ------------------------------------------------------------------
        # 3.5) Handle CLS tokens for multi-video scenarios
        # ------------------------------------------------------------------
        if self.use_cls_token and hasattr(self, 'cls_token') and self.backbone == "mvit":
            # Extract CLS tokens if they exist (first token of each video)
            # MViT may have CLS tokens in some configurations
            cls_tokens = token_feats[:, :, 0:1, :]  # [B, N, 1, D_out]
            
            if N > 1 and self.multi_video_cls_aggregation != "none":
                # Multi-video: aggregate CLS tokens across videos
                if self.multi_video_cls_aggregation == "mean":
                    # Average CLS tokens across videos
                    aggregated_cls = cls_tokens.mean(dim=1)  # [B, 1, D_out]
                elif self.multi_video_cls_aggregation == "attention" and self.attention_pool is not None:
                    # Use attention pooling over CLS tokens
                    cls_for_pool = cls_tokens.squeeze(2)  # [B, N, D_out]
                    aggregated_cls = self.attention_pool(cls_for_pool)  # [B, D_out]
                    aggregated_cls = aggregated_cls.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D_out]
                elif self.multi_video_cls_aggregation == "first":
                    # Use only the first video's CLS token
                    aggregated_cls = cls_tokens[:, 0:1, :, :]  # [B, 1, 1, D_out]
                else:
                    aggregated_cls = cls_tokens  # Keep all CLS tokens
                
                # Option: Replace token_feats with aggregated CLS for study-level representation
                # This is useful when you want a single representation per study
                # Uncomment if needed:
                # token_feats = aggregated_cls  # Use aggregated CLS as the representation

#       print(f"token_feats.shape: {token_feats.shape}")
        if self._apply_aggregator:
            # Before passing to aggregator, convert to exactly N tokens. This
            # preserves backward-compatibility with existing training code &
            # tests that expect a study-level pooling over videos rather than
            # patches.
            if self.token_pooling_mode == "attention" and self.attention_pool is not None:
                # Apply attention pooling to each video's tokens separately
                B, N, L, D_out = token_feats.shape
                feats_list = []
                for i in range(N):
                    video_tokens = token_feats[:, i, :, :]  # [B, L, D_out]
                    pooled = self.attention_pool(video_tokens)  # [B, D_out]
                    feats_list.append(pooled.unsqueeze(1))  # [B, 1, D_out]
                feats = torch.cat(feats_list, dim=1)  # [B, N, D_out]
            else:
                feats = token_feats.mean(dim=2)  # [B, N, D_out]

            orig_dtype = feats.dtype
            with autocast('cuda', enabled=False):
                out = self.aggregator(feats.float())
            return out.to(orig_dtype)

        # Aggregator disabled → return either per-video or per-patch tokens
        if self._per_video_pool:
            #print("Per-video pooling") 
            #print(f"token_feats.shape: {token_feats.shape}")
            if self.token_pooling_mode == "attention" and self.attention_pool is not None:
                # Apply attention pooling to each video's tokens separately
                B, N, L, D_out = token_feats.shape
                feats_list = []
                for i in range(N):
                    video_tokens = token_feats[:, i, :, :]  # [B, L, D_out]
                    pooled = self.attention_pool(video_tokens)  # [B, D_out]
                    feats_list.append(pooled.unsqueeze(1))  # [B, 1, D_out]
                feats = torch.cat(feats_list, dim=1)  # [B, N, D_out]
            else:
                feats = token_feats.mean(dim=2)  # [B, N, D_out]
            # If N=1 (single video), squeeze to get [B, D_out] instead of [B, 1, D_out]
            if N == 1:
                feats = feats.squeeze(1)  # [B, D_out]
            #print(f"feats.shape: {feats.shape}")
        else:
            #print("Per-patch pooling")
            feats = token_feats.reshape(B, N * L, D_out)  # [B, N_tokens, D]
            #print(f"feats.shape: {feats.shape}")

        return feats



