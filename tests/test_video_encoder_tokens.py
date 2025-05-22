import unittest
import torch

from models.video_encoder import VideoEncoder


class TestVideoEncoderTokenLevel(unittest.TestCase):
    """Verify token-level output when aggregator is disabled."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.num_videos = 2
        self.frames = 16
        self.height = 224
        self.width = 224

        self.encoder = VideoEncoder(
            backbone="mvit",
            pretrained=False,         # speed up CI
            aggregate_videos_tokens=False,          # do NOT run internal aggregator
            per_video_pool=False,     # keep every patch token
            freeze_ratio=0.0,         # all layers trainable for the test
            dropout=0.0,              # disable dropout for deterministic output
        )

        # Shape: [B, N, T, H, W, C]
        self.inputs = torch.randn(
            self.batch_size,
            self.num_videos,
            self.frames,
            self.height,
            self.width,
            3,
        )

    def test_token_level_shape(self):
        """Output should be [B, N*L, D] where L > 1."""
        with torch.no_grad():
            out = self.encoder(self.inputs)

        # Basic dimensionality checks
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[-1], self.encoder.embedding_dim)

        # Token dimension should be divisible by the number of videos
        token_dim = out.shape[1]
        self.assertEqual(token_dim % self.num_videos, 0)

        tokens_per_video = token_dim // self.num_videos
        # Debug info
        print(f"Output shape: {out.shape} | tokens_per_video: {tokens_per_video}")
        # In the patched mvit we expect multiple patch tokens per video
        self.assertGreater(tokens_per_video, 1)


if __name__ == "__main__":
    unittest.main() 