import unittest
import torch

from models.text_encoder import TextEncoder
from tests.templates import ModelTestsMixin


class TestTextEncoder(unittest.TestCase, ModelTestsMixin):
    def setUp(self):
        """Set up test case with a text encoder and random inputs."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create model instance
        self.model = TextEncoder()
        
        # Create random test inputs
        batch_size = 4
        seq_length = 128
        self.test_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length)
        }
        
        # Expected output shape (batch_size, embedding_dim)
        self.expected_output_shape = torch.Size([4, 512])  # Assuming 512-dim embeddings
        
    def test_text_encoder_specific(self):
        """Test text encoder specific functionality."""
        # Test with different batch sizes
        batch_sizes = [1, 2, 8]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                inputs = {
                    "input_ids": torch.randint(0, 1000, (batch_size, 128)),
                    "attention_mask": torch.ones(batch_size, 128)
                }
                outputs = self.model(**inputs)
                self.assertEqual(outputs.shape, (batch_size, 512))
        
        # Test with different sequence lengths
        seq_lengths = [64, 128, 256]
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                inputs = {
                    "input_ids": torch.randint(0, 1000, (4, seq_len)),
                    "attention_mask": torch.ones(4, seq_len)
                }
                outputs = self.model(**inputs)
                self.assertEqual(outputs.shape, (4, 512))
                
    def test_attention_mask(self):
        """Test if attention mask properly masks tokens."""
        batch_size = 4
        seq_length = 128
        
        # Create inputs where half the sequence is masked
        inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length)
        }
        inputs["attention_mask"][:, seq_length//2:] = 0
        
        # Get outputs with masked sequence
        outputs_masked = self.model(**inputs)
        
        # Get outputs with full sequence
        inputs["attention_mask"] = torch.ones(batch_size, seq_length)
        outputs_full = self.model(**inputs)
        
        # Outputs should be different when using different masks
        self.assertFalse(torch.allclose(outputs_masked, outputs_full))
        
    def test_normalization(self):
        """Test if output embeddings are normalized."""
        outputs = self.model(**self.test_inputs)
        norms = torch.norm(outputs, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_gradient_flow(self):
        """Test if gradients flow through the entire model."""
        input_ids = self.test_inputs["input_ids"].clone()
        attention_mask = self.test_inputs["attention_mask"].clone()
        
        # Enable gradient tracking for input embeddings
        self.model.train()
        self.model.bert.embeddings.word_embeddings.weight.requires_grad = True
        
        outputs = self.model(input_ids, attention_mask)
        loss = outputs.mean()
        loss.backward()
        
        # Check if embedding gradients exist
        self.assertIsNotNone(self.model.bert.embeddings.word_embeddings.weight.grad)
        self.assertFalse(torch.all(self.model.bert.embeddings.word_embeddings.weight.grad == 0))
        
        # Check if all model parameters received gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with self.subTest(param=name):
                    self.assertIsNotNone(param.grad)
                    self.assertFalse(torch.all(param.grad == 0))


if __name__ == '__main__':
    unittest.main() 