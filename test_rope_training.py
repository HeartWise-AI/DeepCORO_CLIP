#!/usr/bin/env python
"""Quick test to verify RoPE is being applied during training."""

import torch
import yaml
from models.video_encoder import VideoEncoder
from utils.config.clip_config import ClipConfig

# Load config
with open('config/clip/base_config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

print("=" * 60)
print("Testing RoPE Integration")
print("=" * 60)
print(f"\nConfig RoPE settings:")
print(f"  use_rope: {config_dict.get('use_rope')}")
print(f"  rope_base: {config_dict.get('rope_base')}")
print(f"  rope_temporal_scale: {config_dict.get('rope_temporal_scale')}")
print(f"  rope_normalize_mode: {config_dict.get('rope_normalize_mode')}")

print("\n" + "=" * 60)
print("Creating VideoEncoder with RoPE...")
print("=" * 60)

# Create video encoder with RoPE parameters
video_encoder = VideoEncoder(
    backbone="mvit",
    num_frames=16,
    pretrained=False,  # Don't load pretrained weights for speed
    output_dim=512,
    dropout=0.1,
    num_heads=8,
    aggregator_depth=2,
    freeze_ratio=0.0,
    aggregate_videos_tokens=True,
    per_video_pool=False,
    token_pooling_mode='mean',
    attention_pool_heads=8,
    attention_pool_dropout=0.1,
    # RoPE parameters
    use_rope=config_dict.get('use_rope', False),
    rope_base=config_dict.get('rope_base', 10000.0),
    rope_temporal_scale=config_dict.get('rope_temporal_scale', 1.0),
    rope_normalize_mode=config_dict.get('rope_normalize_mode', 'separate'),
)

print("\n" + "=" * 60)
print("Running forward pass to trigger RoPE...")
print("=" * 60)

# Create sample input
batch_size = 2
x = torch.randn(batch_size, 1, 16, 224, 224, 3)

# Forward pass
with torch.no_grad():
    output = video_encoder(x)

print(f"\nOutput shape: {output.shape}")
print(f"Output dtype: {output.dtype}")

print("\n" + "=" * 60)
print("RoPE Test Complete!")
print("=" * 60)
print("\nIf you see '[RoPE] Applied to attention block' messages above,")
print("then RoPE is working correctly!")
print("\nIf you see '[VideoEncoder] Initialized 3D RoPE' but no 'Applied' messages,")
print("check that the model has 16 heads and head_dim divisible by 6.")