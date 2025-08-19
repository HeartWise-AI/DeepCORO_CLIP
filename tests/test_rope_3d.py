"""Tests for 3D Axial RoPE implementation."""

import unittest
import torch
import torch.nn as nn
import math
from models.rope_3d import Rope3D, apply_rope_qk, _rotate_half


class TestRope3D(unittest.TestCase):
    """Test cases for 3D Axial RoPE."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test dimensions
        self.batch_size = 2
        self.num_heads = 8
        self.embed_dim = 384  # Divisible by 8, head_dim = 48
        self.head_dim = self.embed_dim // self.num_heads  # 48
        
        # Video dimensions
        self.T = 4  # Temporal
        self.H = 8  # Height  
        self.W = 8  # Width
        self.N = self.T * self.H * self.W  # Total tokens
        
    def test_rope_3d_initialization(self):
        """Test RoPE 3D initialization."""
        rope = Rope3D(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            temporal_base=10000.0,
            spatial_base=10000.0
        )
        
        # Check dimensions are correctly split
        self.assertEqual(rope.head_dim, 48)
        self.assertEqual(rope.t_dim, 16)  # 48 / 3
        self.assertEqual(rope.h_dim, 16)
        self.assertEqual(rope.w_dim, 16)
        
    def test_rope_3d_with_mvit_dimensions(self):
        """Test RoPE 3D with MViT v2 S dimensions."""
        # MViT v2 S uses 768-dim embeddings with 16 heads
        embed_dim = 768
        num_heads = 16
        head_dim = embed_dim // num_heads  # 48
        
        rope = Rope3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            temporal_base=10000.0,
            spatial_base=10000.0
        )
        
        # Check dimensions
        self.assertEqual(rope.head_dim, 48)
        self.assertEqual(rope.t_dim, 16)  # 48 / 3
        self.assertEqual(rope.h_dim, 16)
        self.assertEqual(rope.w_dim, 16)
        
        # Test with typical MViT token counts
        T, H, W = 8, 14, 14  # After patchification
        N = T * H * W
        
        # Create dummy Q, K tensors
        q = torch.randn(2, num_heads, N, head_dim)
        k = torch.randn(2, num_heads, N, head_dim)
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k, T, H, W, n_special=0)
        
        # Check shapes preserved
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        
    def test_norm_preservation(self):
        """Test that RoPE preserves vector norms."""
        rope = Rope3D(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        # Create test tensors
        q = torch.randn(self.batch_size, self.num_heads, self.N, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.N, self.head_dim)
        
        # Compute original norms
        q_norm_before = torch.norm(q, dim=-1)
        k_norm_before = torch.norm(k, dim=-1)
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k, self.T, self.H, self.W)
        
        # Compute norms after rotation
        q_norm_after = torch.norm(q_rot, dim=-1)
        k_norm_after = torch.norm(k_rot, dim=-1)
        
        # Check norms are preserved (within numerical tolerance)
        torch.testing.assert_close(q_norm_before, q_norm_after, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k_norm_before, k_norm_after, rtol=1e-5, atol=1e-5)
        
    def test_special_tokens_no_rotation(self):
        """Test that special tokens (e.g., CLS) are not rotated."""
        rope = Rope3D(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        n_special = 1  # One CLS token
        N_total = n_special + self.N
        
        # Create tensors with CLS token
        q = torch.randn(self.batch_size, self.num_heads, N_total, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, N_total, self.head_dim)
        
        # Store original CLS tokens
        q_cls_orig = q[:, :, :n_special, :].clone()
        k_cls_orig = k[:, :, :n_special, :].clone()
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k, self.T, self.H, self.W, n_special=n_special)
        
        # Check CLS tokens unchanged
        torch.testing.assert_close(q_rot[:, :, :n_special, :], q_cls_orig)
        torch.testing.assert_close(k_rot[:, :, :n_special, :], k_cls_orig)
        
        # Check other tokens are rotated (should be different)
        self.assertFalse(torch.allclose(q_rot[:, :, n_special:, :], q[:, :, n_special:, :]))
        self.assertFalse(torch.allclose(k_rot[:, :, n_special:, :], k[:, :, n_special:, :]))
        
    def test_frequency_generation(self):
        """Test that frequencies are generated correctly for 3D."""
        rope = Rope3D(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            temporal_scale=2.0  # Different scale for temporal
        )
        
        # Get frequencies
        sin, cos = rope._get_cached_freqs(
            self.T, self.H, self.W,
            device=self.device,
            dtype=torch.float32
        )
        
        # Check shapes
        expected_shape = (self.T * self.H * self.W, self.head_dim)
        self.assertEqual(sin.shape, expected_shape)
        self.assertEqual(cos.shape, expected_shape)
        
        # Check that sin^2 + cos^2 = 1 (within tolerance)
        unity = sin**2 + cos**2
        torch.testing.assert_close(unity, torch.ones_like(unity), rtol=1e-5, atol=1e-5)
        
    def test_gradient_flow(self):
        """Test that gradients flow through RoPE."""
        rope = Rope3D(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        # Create tensors requiring gradients
        q = torch.randn(self.batch_size, self.num_heads, self.N, self.head_dim, requires_grad=True)
        k = torch.randn(self.batch_size, self.num_heads, self.N, self.head_dim, requires_grad=True)
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k, self.T, self.H, self.W)
        
        # Compute a simple loss
        loss = (q_rot * k_rot).sum()
        loss.backward()
        
        # Check gradients exist and are non-zero
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertTrue(torch.any(q.grad != 0))
        self.assertTrue(torch.any(k.grad != 0))
        
    def test_rotate_half(self):
        """Test the _rotate_half helper function."""
        # Create test tensor
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
        
        # Apply rotation
        rotated = _rotate_half(x)
        
        # Expected: swap and negate pairs
        expected = torch.tensor([[-2, 1, -4, 3], [-6, 5, -8, 7]], dtype=torch.float32)
        torch.testing.assert_close(rotated, expected)
        
    def test_apply_rope_qk_utility(self):
        """Test the apply_rope_qk utility function."""
        # Create test tensors
        q = torch.randn(2, 8, 64, 48)
        k = torch.randn(2, 8, 64, 48)
        
        # Create sin/cos
        sin = torch.randn(64, 48)
        cos = torch.randn(64, 48)
        
        # Apply RoPE
        q_rot, k_rot = apply_rope_qk(q, k, sin, cos)
        
        # Check shapes preserved
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        
        # Check norms approximately preserved
        q_norm_before = torch.norm(q, dim=-1)
        q_norm_after = torch.norm(q_rot, dim=-1)
        # Note: This won't be exactly preserved with random sin/cos
        # but should be similar in magnitude
        self.assertTrue(torch.allclose(q_norm_before.mean(), q_norm_after.mean(), rtol=0.5))
        
    def test_different_scales(self):
        """Test RoPE with different temporal and spatial scales."""
        rope = Rope3D(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            temporal_base=10000.0,
            spatial_base=1000.0,  # Different from temporal
            temporal_scale=0.5  # Scale temporal frequencies
        )
        
        # Create test tensors
        q = torch.randn(self.batch_size, self.num_heads, self.N, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.N, self.head_dim)
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k, self.T, self.H, self.W)
        
        # Check output shapes
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        
        # Check that rotation was applied (tensors should be different)
        self.assertFalse(torch.allclose(q, q_rot))
        self.assertFalse(torch.allclose(k, k_rot))
        
    def test_caching(self):
        """Test that frequencies are cached correctly."""
        rope = Rope3D(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        # Set to eval mode for caching
        rope.eval()
        
        # First call - should compute and cache
        sin1, cos1 = rope._get_cached_freqs(
            self.T, self.H, self.W,
            device=self.device,
            dtype=torch.float32
        )
        
        # Check cache was populated
        self.assertEqual(len(rope._cache), 1)
        
        # Second call - should retrieve from cache
        sin2, cos2 = rope._get_cached_freqs(
            self.T, self.H, self.W,
            device=self.device,
            dtype=torch.float32
        )
        
        # Check same tensors returned (not just equal values)
        self.assertTrue(sin1 is sin2)
        self.assertTrue(cos1 is cos2)
        
        # Different dimensions - should create new cache entry
        sin3, cos3 = rope._get_cached_freqs(
            self.T * 2, self.H, self.W,
            device=self.device,
            dtype=torch.float32
        )
        
        # Check new cache entry created
        self.assertEqual(len(rope._cache), 2)
        self.assertFalse(sin1 is sin3)


if __name__ == "__main__":
    unittest.main()