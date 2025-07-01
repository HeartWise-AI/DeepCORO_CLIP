import unittest
import tempfile
import shutil
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict

from models.multi_instance_linear_probing import MultiInstanceLinearProbing
from utils.registry import ModelRegistry, register_submodules
from utils.enums import SubmoduleType


class TestMultiInstanceLinearProbing(unittest.TestCase):
    """Test suite for MultiInstanceLinearProbing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Register submodules
        register_submodules(SubmoduleType.MODEL)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Test parameters
        self.embedding_dim = 512
        self.head_structure = {
            "binary_head": 1,      # Binary classification
            "multiclass_head": 5,  # Multi-class classification  
            "regression_head": 1   # Regression
        }
        self.batch_size = 4
        self.num_instances = 8
        self.num_patches = 16
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_model_registry(self):
        """Test that MultiInstanceLinearProbing is properly registered."""
        model_class = ModelRegistry.get("multi_instance_linear_probing")
        self.assertIsNotNone(model_class)
        self.assertEqual(model_class, MultiInstanceLinearProbing)
        
    def test_initialization_mean_pooling(self):
        """Test model initialization with mean pooling."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="mean"
        )
        
        self.assertEqual(model.embedding_dim, self.embedding_dim)
        self.assertEqual(model.pooling_mode, "mean")
        self.assertEqual(len(model.heads), len(self.head_structure))
        
        # Check head dimensions
        for head_name, num_classes in self.head_structure.items():
            self.assertIn(head_name, model.heads)
            self.assertEqual(model.heads[head_name].out_features, num_classes)
            
    def test_initialization_cls_token_pooling(self):
        """Test model initialization with cls_token pooling."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="cls_token",
            num_attention_heads=8,
            separate_video_attention=False  # Use unified attention for simpler tests
        )
        
        self.assertEqual(model.pooling_mode, "cls_token")
        self.assertTrue(model.use_cls_token)
        self.assertEqual(model.num_attention_heads, 8)
        self.assertIsNotNone(model.cls_token)
        self.assertEqual(model.cls_token.shape, (1, 1, self.embedding_dim))
        
    def test_initialization_attention_pooling(self):
        """Test model initialization with attention pooling."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="attention",
            attention_hidden=128
        )
        
        self.assertEqual(model.pooling_mode, "attention")
        self.assertIsNotNone(model.attention_V)
        self.assertIsNotNone(model.attention_U)
        self.assertIsNotNone(model.attention_w)
        
    def test_initialization_hybrid_pooling(self):
        """Test model initialization with hybrid pooling modes."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="mean+cls_token"
        )
        
        self.assertEqual(model.pooling_mode, "mean+cls_token")
        self.assertTrue(model.use_cls_token)
        # Hybrid pooling should have 2x embedding dimension for heads
        for head in model.heads.values():
            self.assertEqual(head.in_features, 2 * self.embedding_dim)
            
    def test_forward_3d_input_mean_pooling(self):
        """Test forward pass with 3D input [B, N, D] using mean pooling."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="mean"
        )
        
        # Create input tensor [B, N, D]
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
        
        with torch.no_grad():
            outputs = model(x)
            
        # Check output structure
        self.assertIsInstance(outputs, dict)
        self.assertEqual(len(outputs), len(self.head_structure))
        
        for head_name, num_classes in self.head_structure.items():
            self.assertIn(head_name, outputs)
            self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
            
    def test_forward_3d_input_cls_token_pooling(self):
        """Test forward pass with 3D input using cls_token pooling."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="cls_token",
            separate_video_attention=False
        )
        
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
        
        with torch.no_grad():
            outputs = model(x)
            
        for head_name, num_classes in self.head_structure.items():
            self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
            
    def test_forward_4d_input_hierarchical_pooling(self):
        """Test forward pass with 4D input [B, N, L, D] for hierarchical processing."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="cls_token",
            separate_video_attention=False
        )
        
        # Create 4D input [B, N, L, D] 
        x = torch.randn(self.batch_size, self.num_instances, self.num_patches, self.embedding_dim)
        
        with torch.no_grad():
            outputs = model(x)
            
        for head_name, num_classes in self.head_structure.items():
            self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
            
    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="mean"
        )
        
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
        
        # Create mask where last 2 instances are invalid
        mask = torch.ones(self.batch_size, self.num_instances, dtype=torch.bool)
        mask[:, -2:] = False
        
        with torch.no_grad():
            outputs = model(x, mask=mask)
            
        for head_name, num_classes in self.head_structure.items():
            self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
            
    def test_forward_hybrid_pooling(self):
        """Test forward pass with hybrid pooling."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="attention+cls_token",
            separate_video_attention=False
        )
        
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
        
        with torch.no_grad():
            outputs = model(x)
            
        for head_name, num_classes in self.head_structure.items():
            self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
            
    def test_empty_sequence_handling(self):
        """Test handling of empty sequences (all instances masked)."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="mean"
        )
        
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
        
        # Create mask where all instances are invalid
        mask = torch.zeros(self.batch_size, self.num_instances, dtype=torch.bool)
        
        with torch.no_grad():
            outputs = model(x, mask=mask)
            
        # Should return zero tensors
        for head_name, num_classes in self.head_structure.items():
            self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
            self.assertTrue(torch.allclose(outputs[head_name], torch.zeros_like(outputs[head_name])))
            
    def test_invalid_pooling_mode(self):
        """Test initialization with invalid pooling mode."""
        with self.assertRaises(ValueError):
            MultiInstanceLinearProbing(
                embedding_dim=self.embedding_dim,
                head_structure=self.head_structure,
                pooling_mode="invalid_mode"
            )
            
    def test_empty_head_structure(self):
        """Test initialization with empty head structure."""
        with self.assertRaises(ValueError):
            MultiInstanceLinearProbing(
                embedding_dim=self.embedding_dim,
                head_structure={},
                pooling_mode="mean"
            )
            
    def test_invalid_input_shape(self):
        """Test forward pass with invalid input shape."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="mean"
        )
        
        # 2D input should fail
        x = torch.randn(self.batch_size, self.embedding_dim)
        
        with self.assertRaises(ValueError):
            model(x)
            
    def test_mismatched_embedding_dimension(self):
        """Test forward pass with mismatched embedding dimension."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="mean"
        )
        
        # Wrong embedding dimension
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim + 1)
        
        with self.assertRaises(ValueError):
            model(x)
            
    def test_different_pooling_modes(self):
        """Test all supported pooling modes."""
        pooling_modes = ["mean", "max", "attention", "cls_token", 
                        "mean+cls_token", "attention+cls_token"]
        
        for mode in pooling_modes:
            with self.subTest(pooling_mode=mode):
                model = MultiInstanceLinearProbing(
                    embedding_dim=self.embedding_dim,
                    head_structure=self.head_structure,
                    pooling_mode=mode,
                    separate_video_attention=False
                )
                
                x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
                
                with torch.no_grad():
                    outputs = model(x)
                    
                for head_name, num_classes in self.head_structure.items():
                    self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
                    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure={"test_head": 1},
            pooling_mode="cls_token",
            separate_video_attention=False
        )
        
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim, requires_grad=True)
        target = torch.randn(self.batch_size, 1)
        
        output = model(x)["test_head"]
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check that gradients exist and are non-zero
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        
        # Check model parameter gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                
    def test_separate_video_attention(self):
        """Test separate video attention functionality."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="cls_token",
            separate_video_attention=True
        )
        
        # Test with 4D input to trigger hierarchical processing
        x = torch.randn(self.batch_size, self.num_instances, self.num_patches, self.embedding_dim)
        
        with torch.no_grad():
            outputs = model(x)
            
        for head_name, num_classes in self.head_structure.items():
            self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
            
    def test_normalization_strategies(self):
        """Test different normalization strategies."""
        for norm_strategy in ["pre_norm", "post_norm"]:
            with self.subTest(normalization_strategy=norm_strategy):
                model = MultiInstanceLinearProbing(
                    embedding_dim=self.embedding_dim,
                    head_structure=self.head_structure,
                    pooling_mode="cls_token",
                    normalization_strategy=norm_strategy,
                    separate_video_attention=False
                )
                
                x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
                
                with torch.no_grad():
                    outputs = model(x)
                    
                for head_name, num_classes in self.head_structure.items():
                    self.assertEqual(outputs[head_name].shape, (self.batch_size, num_classes))
                    
    def test_dropout_functionality(self):
        """Test dropout during training and evaluation."""
        model = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure={"test_head": 1},
            pooling_mode="cls_token",
            dropout=0.5,
            separate_video_attention=False
        )
        
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
        
        # Training mode - should have some randomness due to dropout
        model.train()
        with torch.no_grad():
            output1 = model(x)["test_head"]
            output2 = model(x)["test_head"]
        
        # Evaluation mode - should be deterministic
        model.eval()
        with torch.no_grad():
            output3 = model(x)["test_head"]
            output4 = model(x)["test_head"]
            
        # Outputs in eval mode should be identical
        self.assertTrue(torch.allclose(output3, output4))
        
    def test_model_state_dict(self):
        """Test model state dict saving and loading."""
        model1 = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="cls_token",
            separate_video_attention=False
        )
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state dict
        model2 = MultiInstanceLinearProbing(
            embedding_dim=self.embedding_dim,
            head_structure=self.head_structure,
            pooling_mode="cls_token",
            separate_video_attention=False
        )
        model2.load_state_dict(state_dict)
        
        # Test that outputs are identical
        x = torch.randn(self.batch_size, self.num_instances, self.embedding_dim)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            outputs1 = model1(x)
            outputs2 = model2(x)
            
        for head_name in self.head_structure.keys():
            self.assertTrue(torch.allclose(outputs1[head_name], outputs2[head_name]))


if __name__ == '__main__':
    unittest.main() 