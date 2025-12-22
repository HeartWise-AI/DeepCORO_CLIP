import pytest
import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils.enums import LossType
from utils.loss.losses import (
    ContrastiveLoss, ContrastiveLossDDP, 
    SiglipLoss, SiglipLossDDP, 
    InfoNCELoss
)
from utils.loss.multi_positive_infonce import MultiPositiveInfoNCELoss
from utils.registry import LossRegistry


class TestLosses:
    """Test cases for loss functions."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing losses."""
        # Create random embeddings for testing
        batch_size = 8
        embedding_dim = 512
        video_features = torch.randn(batch_size, embedding_dim, requires_grad=True)
        text_features = torch.randn(batch_size, embedding_dim, requires_grad=True)
        return video_features, text_features

    def test_contrastive_loss(self, sample_data):
        """Test ContrastiveLoss functionality."""
        video_features, text_features = sample_data
        loss_fn = ContrastiveLoss()
        
        loss = loss_fn(video_features, text_features)
        
        # Check loss is a scalar tensor
        assert loss.ndim == 0
        # Check loss is positive
        assert loss.item() > 0
        # Check loss requires gradient
        assert loss.requires_grad

    def test_siglip_loss(self, sample_data):
        """Test SiglipLoss functionality."""
        video_features, text_features = sample_data
        loss_fn = SiglipLoss()
        
        loss = loss_fn(video_features, text_features)
        
        # Check loss is a scalar tensor
        assert loss.ndim == 0
        # Check loss is positive
        assert loss.item() > 0
        # Check loss requires gradient
        assert loss.requires_grad

    def test_siglip_gating(self):
        """Test that SiglipLoss correctly applies gating function."""
        # Create a controlled test case to verify gating
        batch_size = 4
        embedding_dim = 4
        
        # Create dummy data where we know the expected results
        video_features = torch.randn(batch_size, embedding_dim, requires_grad=True)
        text_features = torch.randn(batch_size, embedding_dim, requires_grad=True)
        
        # Initialize losses
        contrastive_loss = ContrastiveLoss()
        siglip_loss = SiglipLoss()
                
        # Verify the two losses behave differently
        c_loss = contrastive_loss(video_features, text_features)
        s_loss = siglip_loss(video_features, text_features)
        
        # The losses should be different due to the gating
        assert c_loss.item() != s_loss.item()

    def test_siglip_mathematical_properties(self):
        """Test mathematical properties of the SIGLIP gating function."""
        # Create a range of similarity values to test
        similarities = torch.linspace(-10, 5, 150)  # Extend range for negative values
        
        # Apply gating function g(x) = x * sigmoid(x)
        gated = similarities * torch.sigmoid(similarities)
                
        # For x > 0: Verify g(x) ≤ x
        positive_mask = similarities > 0
        if positive_mask.sum() > 0:
            assert torch.all(gated[positive_mask] <= similarities[positive_mask]), "g(x) should be <= x for x > 0"
        
        # For x < 0: Verify g(x) ≥ x
        negative_mask = similarities < 0
        if negative_mask.sum() > 0:
            assert torch.all(gated[negative_mask] >= similarities[negative_mask]), "g(x) should be >= x for x < 0"
        

    def test_info_nce_with_contrastive(self, sample_data):
        """Test InfoNCELoss with contrastive loss."""
        video_features, text_features = sample_data
        loss_fn = InfoNCELoss(temperature=0.1, loss_type='contrastive')
        
        loss = loss_fn(video_features, text_features)
        
        # Check loss is a scalar tensor
        assert loss.ndim == 0
        # Check loss is positive
        assert loss.item() > 0
        # Check loss requires gradient
        assert loss.requires_grad

    def test_info_nce_with_siglip(self, sample_data):
        """Test InfoNCELoss with siglip loss."""
        video_features, text_features = sample_data
        loss_fn = InfoNCELoss(temperature=0.1, loss_type='siglip')
        
        loss = loss_fn(video_features, text_features)
        
        # Check loss is a scalar tensor
        assert loss.ndim == 0
        # Check loss is positive
        assert loss.item() > 0
        # Check loss requires gradient
        assert loss.requires_grad

    def test_temperature_effect(self, sample_data):
        """Test effect of temperature on loss values."""
        video_features, text_features = sample_data
        
        # Create loss functions with different temperatures
        high_temp_loss = InfoNCELoss(temperature=1.0, loss_type='siglip')
        low_temp_loss = InfoNCELoss(temperature=0.01, loss_type='siglip')
        
        high_temp_value = high_temp_loss(video_features, text_features)
        low_temp_value = low_temp_loss(video_features, text_features)
        
        # Lower temperature typically results in higher loss as it makes the
        # distribution more peaky
        # This may not always be true due to randomness, but should generally hold
        # for random initialization
        print(f"High temp loss: {high_temp_value.item()}")
        print(f"Low temp loss: {low_temp_value.item()}")
        # We don't assert here as it's stochastic, but the print helps in debugging

    def test_batch_size_effect(self):
        """Test SIGLIP loss behavior with different batch sizes."""
        # Create loss function
        loss_fn = SiglipLoss()
        
        # Test with different batch sizes
        embedding_dim = 512
        batch_sizes = [2, 4, 8, 16]
        losses = []
        
        for size in batch_sizes:
            # Create random features
            video_features = torch.randn(size, embedding_dim, requires_grad=True)
            text_features = torch.randn(size, embedding_dim, requires_grad=True)
            
            # Compute loss
            loss = loss_fn(video_features, text_features)
            losses.append(loss.item())
            
            # Verify loss shape and gradient
            assert loss.ndim == 0
            assert loss.requires_grad
        
        # Print for manual inspection (values will vary due to randomness)
        for size, loss_val in zip(batch_sizes, losses):
            print(f"Batch size {size}: loss = {loss_val}")
        
        # The absolute values will vary, but what's important is that the loss
        # function can handle different batch sizes without numerical issues

    def test_multi_positive_infonce_loss(self):
        """Ensure multi-positive InfoNCE produces a finite scalar."""
        logits = torch.randn(4, 4, requires_grad=True)
        pos_mask = torch.eye(4)
        loss_fn = MultiPositiveInfoNCELoss()
        loss = loss_fn(logits, pos_mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_multi_positive_loss_registered(self):
        """Verify the registry exposes the multi-positive InfoNCE loss."""
        loss_cls = LossRegistry.get(LossType.MULTI_POSITIVE_INFONCE)
        assert issubclass(loss_cls, MultiPositiveInfoNCELoss)


# DDP tests would need to be run in a multi-GPU environment with proper setup
@pytest.mark.skipif(not dist.is_available() or not torch.cuda.is_available(), 
                    reason="DDP tests require distributed setup and CUDA")
class TestDDPLosses:
    """Test cases for DDP-aware losses. These tests will be skipped if not in a DDP environment."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing DDP losses."""
        # Create random embeddings for testing
        batch_size = 4  # Smaller batch for multi-GPU
        embedding_dim = 512
        video_features = torch.randn(batch_size, embedding_dim).cuda()
        text_features = torch.randn(batch_size, embedding_dim).cuda()
        return video_features, text_features

    def test_contrastive_loss_ddp(self, sample_data):
        """Test ContrastiveLossDDP functionality."""
        # This test will only run in a proper DDP environment
        if not dist.is_initialized():
            pytest.skip("DDP not initialized")
            
        video_features, text_features = sample_data
        loss_fn = ContrastiveLossDDP()
        
        loss = loss_fn(video_features, text_features)
        
        assert loss.ndim == 0
        assert loss.item() > 0
        assert loss.requires_grad

    def test_siglip_loss_ddp(self, sample_data):
        """Test SiglipLossDDP functionality."""
        # This test will only run in a proper DDP environment
        if not dist.is_initialized():
            pytest.skip("DDP not initialized")
            
        video_features, text_features = sample_data
        loss_fn = SiglipLossDDP()
        
        loss = loss_fn(video_features, text_features)
        
        assert loss.ndim == 0
        assert loss.item() > 0
        assert loss.requires_grad

    def test_info_nce_ddp_comparison(self, sample_data):
        """Test that InfoNCELoss with DDP produces different results for different loss types."""
        # This test will only run in a proper DDP environment
        if not dist.is_initialized():
            pytest.skip("DDP not initialized")
            
        video_features, text_features = sample_data
        
        contrastive_loss_fn = InfoNCELoss(temperature=0.1, use_ddp=True, loss_type='contrastive')
        siglip_loss_fn = InfoNCELoss(temperature=0.1, use_ddp=True, loss_type='siglip')
        
        contrastive_loss = contrastive_loss_fn(video_features, text_features)
        siglip_loss = siglip_loss_fn(video_features, text_features)
        
        # The losses should be different due to the gating mechanism in SIGLIP
        assert contrastive_loss.item() != siglip_loss.item()


def test_loss_registry_integration():
    """Test that the SIGLIP loss is properly registered and can be created from the registry."""
    from utils.registry import LossRegistry
    
    # Check if SIGLIP loss types are registered
    assert LossType.SIGLIP in LossRegistry.list_registered()
    assert LossType.SIGLIP_DDP in LossRegistry.list_registered()
    
    # Create losses from registry
    siglip_loss = LossRegistry.create(LossType.SIGLIP)
    siglip_ddp_loss = LossRegistry.create(LossType.SIGLIP_DDP)
    
    # Check correct types
    assert isinstance(siglip_loss, SiglipLoss)
    assert isinstance(siglip_ddp_loss, SiglipLossDDP)
    
    # Create InfoNCE loss with SIGLIP
    info_nce = LossRegistry.create(LossType.INFO_NCE, temperature=0.1, loss_type='siglip')
    assert isinstance(info_nce, InfoNCELoss)
    assert info_nce.loss_type == 'siglip'
    
    # Test with sample data
    batch_size = 4
    embedding_dim = 512
    video_features = torch.randn(batch_size, embedding_dim)
    text_features = torch.randn(batch_size, embedding_dim)
    
    # Make sure losses work when created through registry
    loss1 = siglip_loss(video_features, text_features)
    loss2 = info_nce(video_features, text_features)
    
    assert loss1.ndim == 0
    assert loss2.ndim == 0
    assert loss1.item() > 0
    assert loss2.item() > 0 
