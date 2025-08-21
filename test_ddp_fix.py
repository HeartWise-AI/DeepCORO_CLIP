#!/usr/bin/env python
"""Test that DDP works with RoPE after fixing relative position bias."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

from models.video_encoder import VideoEncoder

# Load config
with open('config/clip/base_config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

print("Testing DDP with RoPE fix...")
print(f"use_rope: {config_dict.get('use_rope')}")

# Create video encoder with RoPE
video_encoder = VideoEncoder(
    backbone="mvit",
    num_frames=16,
    pretrained=False,
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

# Use CPU for testing to avoid OOM
device = torch.device("cpu")
video_encoder = video_encoder.to(device)

# Simulate DDP wrapping (without actual distributed setup)
print("\nSimulating DDP behavior...")

# Check for unused parameters
all_params = set(video_encoder.parameters())
used_params = set()

# Forward pass
x = torch.randn(2, 1, 16, 224, 224, 3).to(device)
output = video_encoder(x)

# Backward pass to see which params get gradients
loss = output.sum()
loss.backward()

# Check which parameters received gradients
for name, param in video_encoder.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        used_params.add(param)

unused_params = all_params - used_params
print(f"\nTotal parameters: {len(all_params)}")
print(f"Used parameters: {len(used_params)}")
print(f"Unused parameters: {len(unused_params)}")

if len(unused_params) > 0:
    print("\n⚠️  WARNING: Some parameters are unused!")
    print("This would cause DDP to fail without find_unused_parameters=True")
    # Find which parameters are unused
    unused_names = []
    for name, param in video_encoder.named_parameters():
        if param in unused_params:
            unused_names.append(name)
    print(f"Unused parameter names (first 10): {unused_names[:10]}")
else:
    print("\n✅ All parameters are used in the forward pass!")
    print("DDP should work without find_unused_parameters=True")

print("\nTest complete!")