import sys
import torch
import pytest

from utils.parser import HeartWiseParser


@pytest.fixture
def test_config():
    """Create a test configuration with default values."""
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], "--base_config", "tests/config/base_config.yaml"]
    config = HeartWiseParser.parse_config()
    sys.argv = original_argv
    return config

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

@pytest.fixture
def parser_config(monkeypatch):
    """Fixture to test HeartWiseParser with '--base_config config/clip/base_config' flag."""
    import sys
    from utils.parser import HeartWiseParser
    # Set sys.argv for the parser
    test_args = [sys.argv[0], "--base_config", "config/clip/base_config"]
    monkeypatch.setattr(sys, "argv", test_args)
    config = HeartWiseParser.parse_config()
    return config 