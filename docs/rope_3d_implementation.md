# 3D Axial RoPE Implementation for Video-Text CLIP

## Overview

This implementation adds 3D Axial Rotary Position Embeddings (RoPE) to the MViT v2 S backbone for improved spatiotemporal modeling in video-text alignment tasks. Based on the 2D axial RoPE from DINOv3 and NAVER's rope-vit, extended to handle 3D video data (T×H×W).

## Key Features

- **3D Axial Split**: Divides head dimension into 3 parts for temporal (T), height (H), and width (W) axes
- **Flexible Frequency Control**: Separate frequency bases for temporal vs spatial dimensions
- **Zero Parameters**: No additional learnable parameters, purely position-based
- **Backward Compatible**: Can be enabled/disabled via configuration

## Architecture

### Dimension Splitting
For MViT v2 S with 768-dim embeddings and 16 heads:
- Head dimension: 768 / 16 = 48
- Split for 3D RoPE:
  - Temporal (T): 16 dimensions
  - Height (H): 16 dimensions  
  - Width (W): 16 dimensions

### Integration Points
1. **VideoEncoder**: Initializes RoPE module when `use_rope=True`
2. **Attention Blocks**: Monkey-patches MViT's MultiScaleAttention to apply RoPE after Q,K projection
3. **Multi-scale Support**: Handles different grid sizes after pooling stages

## Configuration

### In Python Code
```python
video_encoder = VideoEncoder(
    backbone="mvit",
    use_rope=True,                    # Enable RoPE
    rope_base=10000.0,                # Base frequency
    rope_temporal_scale=1.0,          # Scale for temporal frequencies
    rope_normalize_mode="separate",   # Normalization mode
    # ... other parameters
)
```

### In YAML Config
```yaml
# Enable 3D RoPE
use_rope: !!bool true
rope_base: !!float 10000.0
rope_temporal_scale: !!float 1.0
rope_normalize_mode: !!str separate  # Options: separate, max, min
```

## Usage Example

```python
from models.video_encoder import VideoEncoder

# Create encoder with RoPE
model = VideoEncoder(
    backbone="mvit",
    use_rope=True,
    rope_base=10000.0,
    rope_temporal_scale=1.0
)

# Process video
video = torch.randn(2, 1, 16, 224, 224, 3)  # [B, N, T, H, W, C]
features = model(video)  # [B, 512]
```

## Performance Considerations

- **Caching**: Sin/cos frequencies are cached during inference for efficiency
- **Selective Application**: RoPE only applied to blocks with compatible dimensions
- **Complementary to Relative Position Bias**: Can work alongside existing position encodings

## Testing

Run tests to verify the implementation:
```bash
# Unit tests for RoPE module
source .venv/bin/activate
python -m pytest tests/test_rope_3d.py -v

# Integration tests
python test/test_rope_integration.py
```

## Implementation Files

- `models/rope_3d.py`: Core 3D RoPE module
- `models/video_encoder.py`: Integration with MViT
- `utils/config/clip_config.py`: Configuration fields
- `tests/test_rope_3d.py`: Unit tests
- `examples/rope_3d_example.py`: Usage example

## Benefits for Video-Text CLIP

1. **Better Temporal Understanding**: Captures relationships between frames
2. **Preserved Spatial Structure**: Maintains 2D spatial relationships within frames
3. **Flexible Frequency Control**: Can emphasize temporal or spatial aspects
4. **No Training Overhead**: No additional parameters to learn

## Recommended Settings

- **Standard**: `rope_temporal_scale=1.0` (equal frequencies)
- **Temporal Focus**: `rope_temporal_scale=0.5` (higher temporal frequencies)
- **Spatial Focus**: `rope_temporal_scale=2.0` (lower temporal frequencies)
- **Normalize Mode**: `"separate"` (recommended for video)

## References

- DINOv3 (Meta FAIR, 2024): Uses axial RoPE in ViT backbones
- NAVER rope-vit (ECCV'24): Canonical 2D RoPE implementation
- Rotary Position Embedding (Su et al., 2021): Original RoPE paper