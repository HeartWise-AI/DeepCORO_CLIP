import pytest
import torch
from utils.config import HeartWiseConfig

@pytest.fixture
def test_config():
    """Create a test configuration with default values."""
    return HeartWiseConfig(
        # Training parameters
        lr=1e-4,
        batch_size=32,
        epochs=10,
        num_workers=4,
        debug=True,
        temperature=0.07,
        mode="train",
        
        # Data parameters
        data_filename="test_data.csv",
        root="./data",
        target_label="test_label",
        datapoint_loc_label="video_path",
        frames=32,
        stride=2,
        multi_video=True,
        num_videos=4,
        groupby_column="group_id",
        shuffle_videos=True,
        
        # Model parameters
        model_name="mvit",
        pretrained=True,
        video_freeze_ratio=0.5,
        text_freeze_ratio=0.5,
        dropout=0.1,
        num_heads=8,
        aggregator_depth=2,
        
        # Optimization parameters
        optimizer="AdamW",
        scheduler_type="cosine",
        lr_step_period=1,
        factor=0.1,
        video_weight_decay=0.01,
        text_weight_decay=0.01,
        gradient_accumulation_steps=1,
        num_warmup_percent=0.1,
        
        # System parameters
        output_dir="./outputs",
        seed=42,
        use_amp=True,
        device="cuda",
        period=1,
        
        # Loss and metrics parameters
        loss_name="InfoNCE",
        recall_k=[1, 5, 10],
        ndcg_k=[5, 10],
        
        # Data augmentation parameters
        rand_augment=True,
        resize=224,
        apply_mask=False,
        view_count=None,
        
        # Checkpointing parameters
        save_best="loss",
        resume_training=False,
        checkpoint=None,
        
        # Logging parameters
        tag="test",
        name="test_run",
        project="test_project",
        entity="test_entity"
    )

@pytest.fixture
def mock_video_input():
    """Create a mock video input tensor."""
    return torch.randn(2, 32, 224, 224, 3)  # [batch_size, frames, height, width, channels]

@pytest.fixture
def mock_text_input():
    """Create mock text input tensors."""
    input_ids = torch.randint(0, 30522, (2, 512))  # [batch_size, seq_length]
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask 