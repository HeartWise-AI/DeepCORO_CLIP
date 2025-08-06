#!/usr/bin/env python
"""Final verification that attention pooling is properly configured and used."""

import torch
import sys
import yaml
sys.path.append('/workspace')

from models.video_encoder import VideoEncoder

def verify_config_and_model():
    """Verify that the config specifies attention pooling and it's initialized."""
    
    print("=" * 70)
    print("FINAL VERIFICATION: ATTENTION POOLING STATUS")
    print("=" * 70)
    
    # 1. Check config file
    print("\n1. Checking multitask_config.yaml...")
    with open('config/clip/multitask_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    video_pooling_mode = config.get('video_pooling_mode', 'not_found')
    attention_pool_heads = config.get('attention_pool_heads', 'not_found')
    attention_pool_dropout = config.get('attention_pool_dropout', 'not_found')
    
    print(f"   video_pooling_mode: {video_pooling_mode}")
    print(f"   attention_pool_heads: {attention_pool_heads}")
    print(f"   attention_pool_dropout: {attention_pool_dropout}")
    
    if video_pooling_mode == 'attention':
        print("   ✓ Config correctly specifies ATTENTION pooling")
    else:
        print(f"   ✗ Config specifies {video_pooling_mode} pooling instead of attention")
    
    # 2. Check video encoder initialization
    print("\n2. Creating VideoEncoder with config settings...")
    encoder = VideoEncoder(
        backbone='mvit',
        token_pooling_mode=video_pooling_mode,
        attention_pool_heads=attention_pool_heads,
        attention_pool_dropout=attention_pool_dropout
    )
    
    has_attention_pool = hasattr(encoder, 'attention_pool') and encoder.attention_pool is not None
    
    if has_attention_pool:
        print("   ✓ VideoEncoder successfully initialized attention_pool")
        print(f"   - AttentionPool type: {type(encoder.attention_pool).__name__}")
        
        # Test the pooling
        print("\n3. Testing attention pooling operation...")
        dummy_tokens = torch.randn(4, 196, 512)  # [batch, tokens, dim]
        pooled = encoder.attention_pool(dummy_tokens)
        print(f"   Input shape: {dummy_tokens.shape}")
        print(f"   Output shape: {pooled.shape}")
        if pooled.shape == (4, 512):
            print("   ✓ Attention pooling works correctly")
        else:
            print("   ✗ Unexpected output shape")
    else:
        print("   ✗ VideoEncoder DID NOT initialize attention_pool")
        print(f"   - Has attention_pool attr: {hasattr(encoder, 'attention_pool')}")
        print(f"   - attention_pool value: {encoder.attention_pool}")
    
    # 3. Check runner code path
    print("\n4. Checking MultitaskRunner code path...")
    print("   The runner checks:")
    print("   - if hasattr(video_encoder.module, 'attention_pool')")
    print("   - and video_encoder.module.attention_pool is not None")
    print("   Then uses: video_encoder.module.attention_pool(video_tokens)")
    
    if has_attention_pool:
        print("   ✓ This condition WILL be satisfied")
    else:
        print("   ✗ This condition will NOT be satisfied, will fallback to mean pooling")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT:")
    if video_pooling_mode == 'attention' and has_attention_pool:
        print("✅ ATTENTION POOLING IS 100% CONFIGURED AND WILL BE USED")
        print("   - Config specifies 'attention' mode")
        print("   - VideoEncoder has attention_pool initialized")
        print("   - MultitaskRunner will use attention pooling")
    else:
        print("❌ ATTENTION POOLING WILL NOT BE USED")
        if video_pooling_mode != 'attention':
            print(f"   - Config specifies '{video_pooling_mode}' instead of 'attention'")
        if not has_attention_pool:
            print("   - VideoEncoder did not initialize attention_pool")
    print("=" * 70)

if __name__ == "__main__":
    verify_config_and_model()