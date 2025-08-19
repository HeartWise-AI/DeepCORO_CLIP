"""Example usage of 3D Axial RoPE with CLIP training.

This example shows how to enable 3D RoPE for improved spatiotemporal
modeling in video-text alignment tasks.
"""

import torch
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder


def create_config_with_rope():
    """Create a configuration dictionary with RoPE enabled."""
    config = {
        # Basic settings
        'model_name': 'mvit',
        'pretrained': True,
        'batch_size': 4,
        'num_frames': 16,
        
        # RoPE settings (NEW!)
        'use_rope': True,
        'rope_base': 10000.0,
        'rope_temporal_scale': 1.0,  # Can be adjusted to weight temporal vs spatial
        'rope_normalize_mode': 'separate',  # Options: 'separate', 'max', 'min'
        
        # Other model settings
        'output_dim': 512,
        'dropout': 0.1,
        'num_heads': 8,
        'aggregator_depth': 2,
        'video_freeze_ratio': 0.8,
        'aggregate_videos_tokens': True,
        'per_video_pool': False,
        'token_pooling_mode': 'mean',
        'attention_pool_heads': 8,
        'attention_pool_dropout': 0.1,
    }
    return config


def main():
    """Demonstrate RoPE usage in video encoding."""
    print("=" * 60)
    print("3D Axial RoPE Example for Video-Text CLIP Training")
    print("=" * 60)
    
    # Load configuration
    config = create_config_with_rope()
    
    print("\n1. Creating VideoEncoder with 3D RoPE enabled...")
    print(f"   - RoPE enabled: {config['use_rope']}")
    print(f"   - RoPE base: {config['rope_base']}")
    print(f"   - Temporal scale: {config['rope_temporal_scale']}")
    print(f"   - Normalize mode: {config['rope_normalize_mode']}")
    
    # Create video encoder with RoPE
    video_encoder = VideoEncoder(
        backbone=config['model_name'],
        num_frames=config['num_frames'],
        pretrained=config['pretrained'],
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        num_heads=config['num_heads'],
        aggregator_depth=config['aggregator_depth'],
        freeze_ratio=config['video_freeze_ratio'],
        aggregate_videos_tokens=config['aggregate_videos_tokens'],
        per_video_pool=config['per_video_pool'],
        token_pooling_mode=config['token_pooling_mode'],
        attention_pool_heads=config['attention_pool_heads'],
        attention_pool_dropout=config['attention_pool_dropout'],
        # RoPE parameters
        use_rope=config['use_rope'],
        rope_base=config['rope_base'],
        rope_temporal_scale=config['rope_temporal_scale'],
        rope_normalize_mode=config['rope_normalize_mode'],
    )
    
    print("\n2. Processing sample video...")
    # Create sample video tensor
    # Shape: [batch, num_videos, frames, height, width, channels]
    sample_video = torch.randn(
        config['batch_size'], 
        1,  # Single video per sample
        config['num_frames'], 
        224, 224, 3
    )
    print(f"   Input shape: {sample_video.shape}")
    
    # Process video
    with torch.no_grad():
        video_features = video_encoder(sample_video)
    print(f"   Output shape: {video_features.shape}")
    print(f"   Output dtype: {video_features.dtype}")
    
    print("\n3. Benefits of 3D RoPE for video:")
    print("   ✓ Better temporal modeling - captures frame relationships")
    print("   ✓ Improved spatial awareness - maintains 2D structure")
    print("   ✓ Flexible frequency control - separate temporal/spatial scales")
    print("   ✓ No learnable parameters - purely position-based")
    
    print("\n4. Configuration tips:")
    print("   - rope_temporal_scale < 1.0: Higher frequency for temporal (more granular)")
    print("   - rope_temporal_scale > 1.0: Lower frequency for temporal (smoother)")
    print("   - rope_normalize_mode:")
    print("     * 'separate': Normalize T, H, W independently (recommended)")
    print("     * 'max': Normalize by max(T, H, W)")
    print("     * 'min': Normalize by min(T, H, W)")
    
    print("\n5. Training configuration example:")
    print("   To enable RoPE in training, add these lines to your config YAML:")
    print("""
   # In your config/clip/base_config.yaml or custom config:
   use_rope: !!bool true
   rope_base: !!float 10000.0
   rope_temporal_scale: !!float 1.0
   rope_normalize_mode: !!str separate
   """)
    
    print("\n6. Compatibility notes:")
    print("   - RoPE works with MViT backbone (recommended)")
    print("   - Requires head_dim divisible by 6 for 3D split")
    print("   - MViT v2 S: 768 dims, 16 heads = 48 dim/head ✓")
    print("   - Applied to blocks with matching dimensions only")
    
    print("\n" + "=" * 60)
    print("✅ 3D RoPE is successfully integrated and ready to use!")
    print("=" * 60)


if __name__ == "__main__":
    main()