"""Integration test for RoPE with VideoEncoder."""

import torch
import sys
import os
sys.path.append('/volume/DeepCORO_CLIP')

from models.video_encoder import VideoEncoder


def test_video_encoder_with_rope():
    """Test VideoEncoder with RoPE enabled."""
    print("Testing VideoEncoder with 3D RoPE...")
    
    # Create model with RoPE enabled
    model = VideoEncoder(
        backbone="mvit",
        num_frames=16,
        pretrained=False,  # Don't load pretrained for testing
        output_dim=512,
        freeze_ratio=0.0,
        dropout=0.0,
        use_rope=True,  # Enable RoPE
        rope_base=10000.0,
        rope_temporal_scale=1.0,
        rope_normalize_mode="separate"
    )
    
    print(f"Model created with use_rope={model.use_rope}")
    
    # Create test input
    batch_size = 2
    num_segments = 1
    frames = 16
    height = 224
    width = 224
    channels = 3
    
    x = torch.randn(batch_size, num_segments, frames, height, width, channels)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Expected shape: [batch_size, output_dim]
    assert output.shape == (batch_size, 512), f"Expected shape {(batch_size, 512)}, got {output.shape}"
    
    # Check that output is not NaN
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    print("✓ VideoEncoder with RoPE works correctly!")
    return True


def test_video_encoder_without_rope():
    """Test VideoEncoder with RoPE disabled (baseline)."""
    print("\nTesting VideoEncoder without RoPE (baseline)...")
    
    # Create model without RoPE
    model = VideoEncoder(
        backbone="mvit",
        num_frames=16,
        pretrained=False,
        output_dim=512,
        freeze_ratio=0.0,
        dropout=0.0,
        use_rope=False  # Disable RoPE
    )
    
    print(f"Model created with use_rope={model.use_rope}")
    
    # Create test input
    batch_size = 2
    num_segments = 1
    x = torch.randn(batch_size, num_segments, 16, 224, 224, 3)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 512)
    assert not torch.isnan(output).any()
    
    print("✓ VideoEncoder without RoPE works correctly!")
    return True


def compare_rope_vs_baseline():
    """Compare outputs with and without RoPE."""
    print("\nComparing RoPE vs baseline...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create identical inputs
    x = torch.randn(2, 1, 16, 224, 224, 3)
    
    # Model with RoPE
    torch.manual_seed(42)
    model_rope = VideoEncoder(
        backbone="mvit",
        num_frames=16,
        pretrained=False,
        output_dim=512,
        freeze_ratio=0.0,
        dropout=0.0,
        use_rope=True
    )
    
    # Model without RoPE
    torch.manual_seed(42)
    model_baseline = VideoEncoder(
        backbone="mvit",
        num_frames=16,
        pretrained=False,
        output_dim=512,
        freeze_ratio=0.0,
        dropout=0.0,
        use_rope=False
    )
    
    # Forward passes
    with torch.no_grad():
        output_rope = model_rope(x)
        output_baseline = model_baseline(x)
    
    # Outputs should be different due to RoPE
    are_different = not torch.allclose(output_rope, output_baseline)
    print(f"Outputs are different: {are_different}")
    
    if are_different:
        # Compute similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            output_rope, output_baseline, dim=1
        ).mean()
        print(f"Average cosine similarity: {cosine_sim:.4f}")
        print("✓ RoPE changes the output as expected!")
    else:
        print("⚠ Warning: Outputs are identical, RoPE may not be applied correctly")
    
    return are_different


if __name__ == "__main__":
    print("=" * 60)
    print("3D Axial RoPE Integration Tests")
    print("=" * 60)
    
    # Run tests
    test1 = test_video_encoder_with_rope()
    test2 = test_video_encoder_without_rope()
    test3 = compare_rope_vs_baseline()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  VideoEncoder with RoPE: {'PASS' if test1 else 'FAIL'}")
    print(f"  VideoEncoder without RoPE: {'PASS' if test2 else 'FAIL'}")
    print(f"  RoPE vs Baseline comparison: {'PASS' if test3 else 'FAIL'}")
    print("=" * 60)
    
    if all([test1, test2, test3]):
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)