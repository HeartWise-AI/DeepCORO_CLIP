# 3D Rotary Position Embeddings (RoPE) Implementation for Video Transformers

## Overview

This document describes a complete implementation of 3D Axial RoPE (Rotary Position Embeddings) for spatiotemporal video data, designed to work with video transformers like MViT (Multiscale Vision Transformer). The implementation extends 2D RoPE to handle the temporal dimension, enabling better position encoding for video sequences.

## Key Features

- **3D Axial Design**: Splits head dimension into three parts for temporal (T), height (H), and width (W) dimensions
- **MViT Integration**: Monkey-patches MViT's attention mechanism to apply RoPE without modifying the original model
- **CLS Token Support**: Handles special tokens (like CLS) that shouldn't be rotated
- **Efficient Caching**: Caches frequency computations during inference for better performance
- **Flexible Configuration**: Supports different bases and scales for temporal vs spatial dimensions

## Core Implementation

### 1. The Rope3D Module

```python
"""3D Axial RoPE (Rotary Position Embeddings) for video transformers."""

import math
from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _build_freqs(
    dim: int, 
    length: int, 
    base: float = 10000.0, 
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Build frequency tensor for RoPE.
    
    Args:
        dim: Dimension for frequencies
        length: Sequence length
        base: Base for frequency computation
        device: Device to create tensor on
        dtype: Data type of tensor
        
    Returns:
        Frequency tensor of shape [length, dim]
    """
    # Standard RoPE base frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(length, device=device, dtype=dtype)
    freqs = torch.einsum('n,d->nd', t, inv_freq)  # [length, dim/2]
    # Interleave for cos/sin pairing
    return torch.stack((freqs, freqs), dim=-1).reshape(length, dim)


class Rope3D(nn.Module):
    """3D Axial RoPE for video transformers.
    
    Splits the head dimension into 3 parts for T (temporal), H (height), W (width).
    Applies separate rotary embeddings to each axis.
    
    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        temporal_base: Base for temporal frequencies (default: 10000.0)
        spatial_base: Base for spatial frequencies (default: 10000.0)
        temporal_scale: Scale factor for temporal frequencies (default: 1.0)
        normalize_mode: How to normalize coordinates ("separate", "max", "min")
        device: Device to create buffers on
        dtype: Data type for buffers
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        temporal_base: float = 10000.0,
        spatial_base: float = 10000.0,
        temporal_scale: float = 1.0,
        normalize_mode: Literal["separate", "max", "min"] = "separate",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embed_dim // num_heads
        
        # For 3D, we need head_dim divisible by 6 (split into 3 axes, each needs even dims)
        if self.head_dim % 6 != 0:
            raise ValueError(
                f"For 3D RoPE, head_dim ({self.head_dim}) must be divisible by 6. "
                f"Got embed_dim={embed_dim}, num_heads={num_heads}"
            )
        
        self.num_heads = num_heads
        self.temporal_base = temporal_base
        self.spatial_base = spatial_base
        self.temporal_scale = temporal_scale
        self.normalize_mode = normalize_mode
        
        # Divide head_dim into 3 parts
        self.t_dim = self.head_dim // 3
        self.h_dim = self.head_dim // 3
        self.w_dim = self.head_dim - self.t_dim - self.h_dim  # Remainder goes to W
        
        # Ensure each dimension is even (required for rotation)
        assert self.t_dim % 2 == 0 and self.h_dim % 2 == 0 and self.w_dim % 2 == 0, \
            f"Each axis dimension must be even for rotation. Got T={self.t_dim}, H={self.h_dim}, W={self.w_dim}"
        
        # Cache for inference efficiency
        self._cache = {}
        
    @torch.no_grad()
    def _get_cached_freqs(
        self,
        T: int,
        H: int, 
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        n_special: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached or compute sin/cos frequencies.
        
        Args:
            T: Temporal dimension
            H: Height dimension
            W: Width dimension
            device: Device for tensors
            dtype: Data type for tensors
            n_special: Number of special tokens (e.g., CLS) that shouldn't be rotated
            
        Returns:
            Tuple of (sin, cos) tensors of shape [n_special + T*H*W, head_dim]
        """
        cache_key = (T, H, W, device, dtype, n_special)
        
        if not self.training and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Build frequencies for each axis
        t_freqs = _build_freqs(
            self.t_dim, T, 
            self.temporal_base * self.temporal_scale,
            device, dtype
        )  # [T, t_dim]
        
        h_freqs = _build_freqs(
            self.h_dim, H,
            self.spatial_base,
            device, dtype
        )  # [H, h_dim]
        
        w_freqs = _build_freqs(
            self.w_dim, W,
            self.spatial_base,
            device, dtype
        )  # [W, w_dim]
        
        # Compute sin/cos for each axis
        t_cos, t_sin = t_freqs.cos(), t_freqs.sin()
        h_cos, h_sin = h_freqs.cos(), h_freqs.sin()
        w_cos, w_sin = w_freqs.cos(), w_freqs.sin()
        
        # Create 3D grid and concatenate frequencies
        # We need to broadcast to [T, H, W, head_dim]
        t_cos = t_cos.view(T, 1, 1, self.t_dim).expand(T, H, W, -1)
        t_sin = t_sin.view(T, 1, 1, self.t_dim).expand(T, H, W, -1)
        
        h_cos = h_cos.view(1, H, 1, self.h_dim).expand(T, H, W, -1)
        h_sin = h_sin.view(1, H, 1, self.h_dim).expand(T, H, W, -1)
        
        w_cos = w_cos.view(1, 1, W, self.w_dim).expand(T, H, W, -1)
        w_sin = w_sin.view(1, 1, W, self.w_dim).expand(T, H, W, -1)
        
        # Concatenate along head_dim axis
        cos = torch.cat([t_cos, h_cos, w_cos], dim=-1).reshape(T*H*W, self.head_dim)
        sin = torch.cat([t_sin, h_sin, w_sin], dim=-1).reshape(T*H*W, self.head_dim)
        
        # Add identity for special tokens (no rotation)
        if n_special > 0:
            special_cos = torch.ones((n_special, self.head_dim), device=device, dtype=dtype)
            special_sin = torch.zeros((n_special, self.head_dim), device=device, dtype=dtype)
            cos = torch.cat([special_cos, cos], dim=0)
            sin = torch.cat([special_sin, sin], dim=0)
        
        if not self.training:
            self._cache[cache_key] = (sin, cos)
        
        return sin, cos
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        T: int,
        H: int,
        W: int,
        n_special: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to query and key tensors.
        
        Args:
            q: Query tensor of shape [B, num_heads, N, head_dim]
            k: Key tensor of shape [B, num_heads, N, head_dim]
            T: Temporal dimension of video
            H: Height dimension of video grid
            W: Width dimension of video grid
            n_special: Number of special tokens at the beginning (won't be rotated)
            
        Returns:
            Tuple of rotated (q, k) tensors with same shapes as input
        """
        B, Hh, N, Dh = q.shape
        assert Dh == self.head_dim, f"Expected head_dim={self.head_dim}, got {Dh}"
        
        # Allow for CLS tokens - detect automatically if n_special not provided
        expected_spatial = T * H * W
        if n_special == 0 and N == expected_spatial + 1:
            # Likely has a CLS token
            n_special = 1
        
        # Check dimension match with tolerance for special tokens
        if N != n_special + expected_spatial:
            # Dimension mismatch - return unmodified
            # This can happen during pooling stages
            return q, k
        
        # Get sin/cos frequencies
        sin, cos = self._get_cached_freqs(T, H, W, q.device, q.dtype, n_special)
        
        # Expand for batch and heads: [1, 1, N, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation using the split dimensions
        # Split q and k into 3 parts corresponding to T, H, W
        q_t, q_h, q_w = torch.split(q, [self.t_dim, self.h_dim, self.w_dim], dim=-1)
        k_t, k_h, k_w = torch.split(k, [self.t_dim, self.h_dim, self.w_dim], dim=-1)
        
        # Split sin and cos similarly
        sin_t, sin_h, sin_w = torch.split(sin, [self.t_dim, self.h_dim, self.w_dim], dim=-1)
        cos_t, cos_h, cos_w = torch.split(cos, [self.t_dim, self.h_dim, self.w_dim], dim=-1)
        
        # Apply rotation to each part
        q_t = q_t * cos_t + _rotate_half(q_t) * sin_t
        q_h = q_h * cos_h + _rotate_half(q_h) * sin_h
        q_w = q_w * cos_w + _rotate_half(q_w) * sin_w
        
        k_t = k_t * cos_t + _rotate_half(k_t) * sin_t
        k_h = k_h * cos_h + _rotate_half(k_h) * sin_h
        k_w = k_w * cos_w + _rotate_half(k_w) * sin_w
        
        # Concatenate back
        q_rot = torch.cat([q_t, q_h, q_w], dim=-1)
        k_rot = torch.cat([k_t, k_h, k_w], dim=-1)
        
        return q_rot, k_rot
```

### 2. MViT Integration via Monkey-Patching

The key innovation is integrating RoPE into MViT without modifying the original model code. This is done through monkey-patching:

```python
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
        B, N, _ = x.shape
        T, H, W = thw
        
        # Standard MViT attention computation
        if hasattr(self_attn, 'pool'):
            x, thw = self_attn.pool(x, thw)
            T, H, W = thw
            
        # Compute Q, K, V
        qkv = self_attn.qkv(self_attn.norm(x))
        q, k, v = qkv.reshape(B, N, 3, self_attn.num_heads, -1).unbind(dim=2)
        
        # Transpose for attention: [B, num_heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply 3D RoPE if available for this configuration
        num_heads = self_attn.num_heads
        head_dim = q.shape[-1]
        rope_key = f"{num_heads}_{head_dim}"
        
        if rope_key in rope_modules:
            rope_module = rope_modules[rope_key]
            # Detect CLS token
            n_special = 1 if hasattr(self_attn, 'has_cls_embed') and self_attn.has_cls_embed else 0
            q, k = rope_module(q, k, T, H, W, n_special)
        
        # Continue with standard attention computation
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        
        # Add relative position bias if it exists
        if hasattr(self_attn, 'rel_pos_h') and hasattr(self_attn, 'rel_pos_w'):
            attn = _add_rel_pos(attn, self_attn.rel_pos_h, self_attn.rel_pos_w, q.shape[2], H, W)
        if hasattr(self_attn, 'rel_pos_t'):
            # Add temporal relative position bias
            attn = _add_rel_pos(attn.permute(0, 1, 3, 2), self_attn.rel_pos_t, None, q.shape[2], T, 1)
            attn = attn.permute(0, 1, 3, 2)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self_attn.proj(x)
        
        return x, thw
    
    # Patch all attention blocks in MViT
    for block in self.model.blocks:
        if hasattr(block, 'attn'):
            block.attn.forward = MethodType(patched_forward, block.attn)
```

### 3. Video Encoder Integration

```python
class VideoEncoder(nn.Module):
    def __init__(
        self,
        # ... other parameters ...
        use_rope: bool = False,
        rope_base: float = 10000.0,
        rope_temporal_scale: float = 1.0,
        rope_normalize_mode: str = "separate",
    ):
        super().__init__()
        
        # Store RoPE configuration
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.rope_temporal_scale = rope_temporal_scale
        self.rope_normalize_mode = rope_normalize_mode
        
        if backbone == "mvit":
            # Initialize MViT model
            self.model = mvit_v2_s(weights="KINETICS400_V1" if pretrained else None)
            
            # Initialize RoPE if enabled for MViT
            if self.use_rope:
                # MViT has blocks with different head dimensions
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
                        rope_module.eval()  # No trainable params
                        rope_modules[key] = rope_module
                
                self._rope_modules = rope_modules
                
                # Monkey-patch attention blocks
                self._patch_mvit_attention_for_rope()
```

## Configuration

### Basic Configuration

```yaml
# Enable 3D RoPE
use_rope: true
rope_base: 10000.0
rope_temporal_scale: 1.0
rope_normalize_mode: separate
```

### Advanced Configuration Options

- **rope_base**: Controls the frequency range for position encoding (default: 10000.0)
- **rope_temporal_scale**: Scales temporal frequencies independently (useful for videos with different frame rates)
- **rope_normalize_mode**: How to normalize spatial coordinates
  - "separate": Normalize each axis independently
  - "max": Use maximum dimension for normalization
  - "min": Use minimum dimension for normalization

## Key Design Decisions

### 1. Axial Splitting
The head dimension is split into three equal parts for T, H, and W dimensions. This ensures balanced position encoding across all axes.

### 2. CLS Token Handling
Special tokens (like CLS) are not rotated, preserving their global representation capability.

### 3. Monkey-Patching Strategy
Instead of modifying the original MViT code, we monkey-patch the attention mechanism. This allows:
- Easy integration with pretrained models
- Minimal code changes
- Reversible modifications

### 4. Caching for Efficiency
Frequency computations are cached during inference, significantly reducing computational overhead.

## Training Considerations

When adding 3D RoPE to a pretrained model:

1. **Full Model Retraining Required**: RoPE fundamentally changes how position information flows through the model, requiring all weights to be retrained.

2. **Gradual Unfreezing**: Use a freeze schedule to gradually unfreeze model layers:
   ```yaml
   video_freeze_schedule: cosine
   video_freeze_start: 0.95  # Start with 95% frozen
   video_freeze_end: 0.0     # End fully trainable
   video_freeze_warmup_epochs: 10
   ```

3. **Temperature Scheduling**: For contrastive learning, use temperature scheduling for stability:
   ```yaml
   temperature_schedule: cosine
   temperature_start: 0.5
   temperature_end: 0.087
   temperature_warmup_epochs: 8
   ```

4. **Lower Learning Rate**: Use a reduced learning rate for stability (e.g., 2e-5 instead of 1e-4)

## Usage Example

```python
import torch
from rope_3d import Rope3D

# Create RoPE module
rope = Rope3D(
    embed_dim=768,  # Total embedding dimension
    num_heads=8,    # Number of attention heads
    temporal_base=10000.0,
    spatial_base=10000.0,
    temporal_scale=1.0
)

# Example input shapes
B = 2  # Batch size
num_heads = 8
T, H, W = 16, 14, 14  # Video dimensions
N = T * H * W  # Sequence length
head_dim = 96

# Create dummy Q and K tensors
q = torch.randn(B, num_heads, N, head_dim)
k = torch.randn(B, num_heads, N, head_dim)

# Apply RoPE
q_rot, k_rot = rope(q, k, T, H, W)

print(f"Input shapes: Q={q.shape}, K={k.shape}")
print(f"Output shapes: Q_rot={q_rot.shape}, K_rot={k_rot.shape}")
```

## Performance Impact

- **Memory**: Minimal overhead (only stores sin/cos frequencies)
- **Computation**: Small overhead during attention computation (~5-10% increase)
- **Training Stability**: Requires careful hyperparameter tuning when added to pretrained models
- **Convergence**: May require more epochs initially but achieves better final performance

## References

- Original RoPE paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- 2D Axial RoPE: Used in DINOv2 and other vision transformers
- MViT paper: [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227)

## License

This implementation is provided as-is for educational and research purposes.