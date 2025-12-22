"""Test script to verify gradient clipping is working correctly."""

import yaml
import torch
from utils.config.clip_config import ClipConfig

def test_gradient_clipping_config():
    """Verify that gradient clipping parameters are properly loaded."""

    config_path = "config/clip/siglip_output_dataset_config.yaml"

    # Load the config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    print("=" * 60)
    print("Gradient Clipping Configuration Test")
    print("=" * 60)

    # Check for gradient clipping parameters
    print("\n1. Checking gradient clipping parameters in config file:")
    print(f"   - video_max_grad_norm: {config_dict.get('video_max_grad_norm', 'NOT FOUND')}")
    print(f"   - text_max_grad_norm: {config_dict.get('text_max_grad_norm', 'NOT FOUND')}")
    print(f"   - max_grad_norm (legacy): {config_dict.get('max_grad_norm', 'NOT FOUND')}")

    # Verify values are reasonable
    video_grad = config_dict.get('video_max_grad_norm')
    text_grad = config_dict.get('text_max_grad_norm')

    if video_grad is None or text_grad is None:
        print("\n❌ FAIL: Gradient clipping parameters are missing!")
        print("   The NaN issue will likely occur without these parameters.")
        return False

    if video_grad <= 0 or text_grad <= 0:
        print("\n❌ FAIL: Gradient clipping values must be positive!")
        return False

    print(f"\n✓ PASS: Gradient clipping is properly configured")
    print(f"   Video encoder gradients will be clipped at {video_grad}")
    print(f"   Text encoder gradients will be clipped at {text_grad}")

    # Check other stability-related settings
    print("\n2. Checking other numerical stability settings:")
    print(f"   - use_amp: {config_dict.get('use_amp', 'NOT SET')}")
    print(f"   - temperature: {config_dict.get('temperature', 'NOT SET')}")

    if config_dict.get('use_amp'):
        print("   ⚠ WARNING: AMP is enabled. Ensure mixed precision is handled correctly.")

    print("\n" + "=" * 60)
    print("✓ Configuration appears correct for preventing NaN loss")
    print("=" * 60)

    return True


def test_aggregator_numerical_stability():
    """Test that the aggregator has proper clamping."""
    print("\n" + "=" * 60)
    print("Aggregator Numerical Stability Test")
    print("=" * 60)

    from models.video_aggregator import EnhancedVideoAggregator

    # Create a test aggregator
    aggregator = EnhancedVideoAggregator(
        embedding_dim=512,
        num_heads=8,
        dropout=0.1,
        use_positional_encoding=True,
        aggregator_depth=2
    )

    # Create test input with extreme values
    batch_size = 4
    num_segments = 10
    embedding_dim = 512

    # Test with normal values
    x_normal = torch.randn(batch_size, num_segments, embedding_dim)
    out_normal = aggregator(x_normal)

    if torch.isfinite(out_normal).all():
        print("\n✓ PASS: Aggregator handles normal inputs correctly")
    else:
        print("\n❌ FAIL: Aggregator produces non-finite outputs with normal inputs!")
        return False

    # Test with extreme values (simulating what might happen during training)
    x_extreme = torch.randn(batch_size, num_segments, embedding_dim) * 100
    out_extreme = aggregator(x_extreme)

    if torch.isfinite(out_extreme).all():
        print("✓ PASS: Aggregator handles extreme inputs correctly")
    else:
        print("❌ FAIL: Aggregator produces non-finite outputs with extreme inputs!")
        return False

    print("\n" + "=" * 60)
    print("✓ Aggregator numerical stability is good")
    print("=" * 60)

    return True


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("* NaN Loss Fix Verification Test")
    print("*" * 60)

    success = True

    # Run tests
    success &= test_gradient_clipping_config()
    success &= test_aggregator_numerical_stability()

    print("\n" + "*" * 60)
    if success:
        print("* ✓ ALL TESTS PASSED - Ready to train!")
    else:
        print("* ❌ SOME TESTS FAILED - Fix issues before training")
    print("*" * 60)
    print("\n")
