import torch
import unittest
from models.text_encoder import TextEncoder
from tests.templates import ModelTestsMixin


class TestTextEncoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Set random seeds
        torch.manual_seed(42)
        
        # Create model with test settings
        self.model = TextEncoder(
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            output_dim=512,
            freeze_ratio=0.0,  # Don't freeze any parameters for testing
            dropout=0.0  # Disable dropout for testing
        )
        
        # Create test inputs (batch_size=2, seq_length=128)
        self.test_inputs = (
            torch.randint(0, 30522, (2, 128), dtype=torch.long),  # input_ids
            torch.ones(2, 128, dtype=torch.long)  # attention_mask
        )
        
        # Expected output shape (batch_size=2, embedding_dim=512)
        self.expected_output_shape = torch.Size([2, 512])
        
    def test_text_encoder_specific(self):
        """Test text encoder specific functionality."""
        # Test with different sequence lengths
        seq_lengths = [64, 128, 256]
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                input_ids = torch.randint(0, 30522, (2, seq_len))
                attention_mask = torch.ones_like(input_ids)
                outputs = self.model(input_ids, attention_mask)
                self.assertEqual(outputs.shape, (2, 512))
                
    def test_attention_mask(self):
        """Test if attention mask is properly applied."""
        input_ids = torch.randint(0, 30522, (2, 128))
        
        # Create attention mask with some tokens masked
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, 64:] = 0  # Mask second half of sequence
        
        # Get outputs with and without masking
        outputs_masked = self.model(input_ids, attention_mask)
        outputs_full = self.model(input_ids, torch.ones_like(input_ids))
        
        # Outputs should be different when masking is applied
        self.assertFalse(torch.allclose(outputs_masked, outputs_full))
        
    def test_gradient_flow(self):
        """Test if gradients flow through the entire model."""
        # Move model to CPU for gradient testing
        self.model = self.model.cpu()
    
        # Enable gradient tracking and training mode
        self.model.train()
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Also unfreeze BERT parameters specifically
        for param in self.model.bert.parameters():
            param.requires_grad = True
    
        # Convert input tensors to float for gradient computation
        input_ids = self.test_inputs[0].clone()
        attention_mask = self.test_inputs[1].clone().float()
    
        # Forward pass with larger loss for better gradient signal
        outputs = self.model(input_ids, attention_mask)
        loss = 10000*outputs.pow(2).sum()  # Use squared sum for stronger gradients
        loss.backward()
    
        # Check if all model parameters received gradients
        grad_exists = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with self.subTest(param=name):
                    self.assertIsNotNone(param.grad)
                    if not grad_exists and torch.any(torch.abs(param.grad) >= 1e-6):
                        grad_exists = True
        self.assertTrue(grad_exists, "No parameter received significant gradients")
                    
    def test_normalization(self):
        """Test that output embeddings are NOT normalized by default (normalization should happen in loss)."""
        outputs = self.model(self.test_inputs[0], self.test_inputs[1])
        norms = torch.norm(outputs, p=2, dim=1)
        # Check that norms are NOT all close to 1
        self.assertFalse(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        # Additionally verify outputs have non-zero magnitude
        self.assertTrue(torch.all(norms > 0))
        
    def test_device_moving(self):
        """Test if model can be moved between devices while maintaining output consistency."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
            
        # Move everything to CPU first
        self.model = self.model.cpu()
        input_ids = self.test_inputs[0].cpu()
        attention_mask = self.test_inputs[1].cpu()
        
        # Get CPU output
        with torch.no_grad():
            self.model.eval()  # Set to eval mode for consistent outputs
            cpu_output = self.model(input_ids, attention_mask)
        
        # Move to GPU
        self.model = self.model.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        
        # Get GPU output
        with torch.no_grad():
            gpu_output = self.model(input_ids, attention_mask)
        
        # Move back to CPU for comparison
        gpu_output = gpu_output.cpu()
        
        # Outputs should be similar (not exactly equal due to numerical differences)
        self.assertTrue(torch.allclose(cpu_output, gpu_output, atol=1e-4))
        
    def test_batch_independence(self):
        """Test if samples in a batch are processed independently."""
        # Create two identical batches
        input_ids = torch.randint(0, 30522, (2, 128))
        attention_mask = torch.ones_like(input_ids)
        
        # Process first batch
        self.model.eval()  # Set to eval mode for consistent outputs
        with torch.no_grad():
            outputs1 = self.model(input_ids, attention_mask)
        
        # Create a copy of the first sample
        input_ids_new = input_ids.clone()
        input_ids_new[1] = torch.randint(0, 30522, (128,))
        
        # Process modified batch
        with torch.no_grad():
            outputs2 = self.model(input_ids_new, attention_mask)
        
        # First sample's output should be identical
        self.assertTrue(torch.allclose(outputs1[0], outputs2[0], atol=1e-6))
        # Second sample's output should be different
        self.assertFalse(torch.allclose(outputs1[1], outputs2[1], atol=1e-6))


if __name__ == '__main__':
    unittest.main() 