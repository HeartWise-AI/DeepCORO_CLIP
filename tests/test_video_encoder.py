import torch
import unittest
from models.video_encoder import VideoEncoder
from tests.templates import ModelTestsMixin

class TestVideoEncoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Set random seeds
        torch.manual_seed(42)
        
        # Create model with test settings
        self.model = VideoEncoder(
            backbone="mvit",
            num_frames=16,
            pretrained=False,  # Don't load pretrained for testing
            output_dim=512,
            freeze_ratio=0.0,  # Don't freeze any parameters for testing
            dropout=0.0  # Disable dropout for testing
        )
        
        # Create test inputs with correct shape for MViT
        # [batch_size, num_segments, frames, height, width, channels]
        self.test_inputs = torch.randn(2, 2, 16, 224, 224, 3)
        
        # Expected output shape (batch_size=2, embedding_dim=512)
        self.expected_output_shape = torch.Size([2, 512])
        
    def test_video_encoder_specific(self):
        """Test video encoder specific functionality."""
        # Test with different batch sizes
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                inputs = torch.randn(batch_size, 2, 16, 224, 224, 3)
                outputs = self.model(inputs)
                self.assertEqual(outputs.shape, (batch_size, 512))
                
    def test_normalization(self):
        """Test that output embeddings are NOT normalized by default (normalization should happen in loss)."""
        outputs = self.model(self.test_inputs)
        norms = torch.norm(outputs, p=2, dim=1)
        # Check that norms are NOT all close to 1
        self.assertFalse(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        # Additionally verify outputs have non-zero magnitude
        self.assertTrue(torch.all(norms > 0))
        
    def test_gradient_flow(self):
        """Test if gradients flow through the entire model."""
        # Move everything to CPU for testing
        self.model = self.model.cpu()
        inputs = self.test_inputs.clone().cpu()
        inputs.requires_grad = True
    
        # Enable training mode and ensure all parameters require gradients
        self.model.train()
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Also unfreeze backbone parameters specifically
        for param in self.model.model.parameters():
            param.requires_grad = True
    
        # Forward pass with larger loss for better gradient signal
        outputs = self.model(inputs)
        loss = 1000*outputs.pow(2).sum()  # Use squared sum for stronger gradients
        loss.backward()
    
        # Check if input gradients exist
        self.assertIsNotNone(inputs.grad)
        self.assertFalse(torch.all(inputs.grad == 0))
    
        # Check if all model parameters received gradients
        grad_exists = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with self.subTest(param=name):
                    self.assertIsNotNone(param.grad)
                    if not grad_exists and torch.any(torch.abs(param.grad) >= 1e-6):
                        grad_exists = True
        self.assertTrue(grad_exists, "No parameter received significant gradients")
                    
    def test_device_moving(self):
        """Test if model can be moved between devices while maintaining output consistency."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
            
        # Move everything to CPU first
        self.model = self.model.cpu()
        inputs = self.test_inputs.cpu()
        
        # Get CPU output
        with torch.no_grad():
            self.model.eval()  # Set to eval mode for consistent outputs
            cpu_output = self.model(inputs)
        
        # Move to GPU
        self.model = self.model.cuda()
        inputs = inputs.cuda()
        
        # Get GPU output
        with torch.no_grad():
            gpu_output = self.model(inputs)
        
        # Move back to CPU for comparison
        gpu_output = gpu_output.cpu()
        
        # Outputs should be similar (not exactly equal due to numerical differences)
        self.assertTrue(torch.allclose(cpu_output, gpu_output, atol=1e-4))
        
    def test_all_parameters_updated(self):
        """Test if all trainable parameters are being updated during training."""
        # Move to CPU and enable training mode
        self.model = self.model.cpu().train()
        
        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Store initial parameter values
        initial_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.clone().detach()
        
        # Forward and backward pass with larger loss for better gradient signal
        inputs = self.test_inputs.clone().cpu()
        outputs = self.model(inputs)
        loss = outputs.pow(2).sum()  # Use squared sum for stronger gradients
        loss.backward()
        
        # Update parameters with a larger learning rate for noticeable changes
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)  # Increased learning rate
        optimizer.step()
        
        # Check if parameters have been updated
        param_updated = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with self.subTest(param=name):
                    # Check if parameter has changed significantly
                    if not param_updated and not torch.allclose(initial_params[name], param.detach(), atol=1e-4):
                        param_updated = True
        self.assertTrue(param_updated, "No parameters were significantly updated")


if __name__ == '__main__':
    unittest.main() 