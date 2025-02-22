import unittest
import torch

from models.video_encoder import VideoEncoder
from tests.templates import ModelTestsMixin


class TestVideoEncoder(unittest.TestCase, ModelTestsMixin):
    def setUp(self):
        """Set up test case with a video encoder and random inputs."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create model instance
        self.model = VideoEncoder()
        
        # Create random test inputs: [batch_size, num_frames, height, width, channels]
        self.test_inputs = torch.randn(4, 32, 224, 224, 3)
        
        # Expected output shape (batch_size, embedding_dim)
        self.expected_output_shape = torch.Size([4, 512])  # Assuming 512-dim embeddings
        
    def test_video_encoder_specific(self):
        """Test video encoder specific functionality."""
        # Test with different batch sizes
        batch_sizes = [1, 2, 8]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                inputs = torch.randn(batch_size, 32, 224, 224, 3)
                outputs = self.model(inputs)
                self.assertEqual(outputs.shape, (batch_size, 512))
        
        # Test with different frame counts
        frame_counts = [16, 32, 64]
        for frames in frame_counts:
            with self.subTest(frames=frames):
                inputs = torch.randn(4, frames, 224, 224, 3)
                outputs = self.model(inputs)
                self.assertEqual(outputs.shape, (4, 512))
                
    def test_normalization(self):
        """Test if output embeddings are normalized."""
        outputs = self.model(self.test_inputs)
        norms = torch.norm(outputs, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_gradient_flow(self):
        """Test if gradients flow through the entire model."""
        inputs = self.test_inputs.clone()
        inputs.requires_grad = True
        
        outputs = self.model(inputs)
        loss = outputs.mean()
        loss.backward()
        
        # Check if input gradients exist
        self.assertIsNotNone(inputs.grad)
        self.assertFalse(torch.all(inputs.grad == 0))
        
        # Check if all model parameters received gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with self.subTest(param=name):
                    self.assertIsNotNone(param.grad)
                    self.assertFalse(torch.all(param.grad == 0))


if __name__ == '__main__':
    unittest.main() 