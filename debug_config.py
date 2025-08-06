#!/usr/bin/env python
"""Debug why attention pool is None."""

import yaml

# Load config
with open('config/clip/multitask_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Config values:")
print(f"  video_pooling_mode: {config.get('video_pooling_mode', 'NOT_SET')}")
print(f"  attention_pool_heads: {config.get('attention_pool_heads', 'NOT_SET')}")
print(f"  attention_pool_dropout: {config.get('attention_pool_dropout', 'NOT_SET')}")

# Check if the token_pooling_mode is actually being set
import sys
sys.path.append('/workspace')

from models.video_encoder import VideoEncoder

# Test with explicit values
print("\nTest 1: Explicit 'attention' mode")
encoder1 = VideoEncoder(
    backbone='mvit',
    token_pooling_mode='attention',  # Explicitly set
    attention_pool_heads=8,
    attention_pool_dropout=0.1
)
print(f"  attention_pool is None: {encoder1.attention_pool is None}")
print(f"  token_pooling_mode: {encoder1.token_pooling_mode}")

# Test with value from config
print("\nTest 2: From config value")
encoder2 = VideoEncoder(
    backbone='mvit',
    token_pooling_mode=config.get('video_pooling_mode', 'mean'),
    attention_pool_heads=config.get('attention_pool_heads', 8),
    attention_pool_dropout=config.get('attention_pool_dropout', 0.1)
)
print(f"  attention_pool is None: {encoder2.attention_pool is None}")
print(f"  token_pooling_mode: {encoder2.token_pooling_mode}")

# Test with getattr pattern (like in project)
class DummyConfig:
    pass

dummy = DummyConfig()
for k, v in config.items():
    setattr(dummy, k, v)

print("\nTest 3: Using getattr pattern")
encoder3 = VideoEncoder(
    backbone='mvit',
    token_pooling_mode=getattr(dummy, 'video_pooling_mode', 'mean'),
    attention_pool_heads=getattr(dummy, 'attention_pool_heads', 8),
    attention_pool_dropout=getattr(dummy, 'attention_pool_dropout', 0.1)
)
print(f"  attention_pool is None: {encoder3.attention_pool is None}")
print(f"  token_pooling_mode: {encoder3.token_pooling_mode}")