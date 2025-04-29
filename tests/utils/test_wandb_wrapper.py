import pytest
import unittest.mock as mock
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig


@pytest.fixture
def mock_config():
    """Create a mock HeartWiseConfig for testing."""
    config = mock.MagicMock(spec=HeartWiseConfig)
    config.project = "test-project"
    config.entity = "test-entity"
    config.loss_name = "test-loss"
    config.to_dict.return_value = {
        "project": "test-project",
        "entity": "test-entity",
        "loss_name": "test-loss",
        "parameter1": "value1",
        "parameter2": "value2"
    }
    return config


class TestWandbWrapper:
    """Tests for the WandbWrapper class."""

    @mock.patch("wandb.init")
    def test_init_not_initialized(self, mock_init):
        """Test initialization with initialized=False."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        wrapper = WandbWrapper(config, initialized=False)
        
        mock_init.assert_not_called()
        assert wrapper.initialized is False

    @mock.patch("wandb.init")
    def test_init_ref_device(self, mock_init):
        """Test initialization with is_ref_device=True."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        config.project = "test-project"
        config.entity = "test-entity"
        config.to_dict.return_value = {"param1": "value1", "param2": "value2"}
        
        wrapper = WandbWrapper(config, initialized=True, is_ref_device=True)
        
        mock_init.assert_called_once()
        init_kwargs = mock_init.call_args.kwargs
        assert init_kwargs["project"] == "test-project"
        assert init_kwargs["entity"] == "test-entity"
        assert init_kwargs["config"] == {"param1": "value1", "param2": "value2"}
        assert init_kwargs["allow_val_change"] is True
        assert wrapper.initialized is True

    @mock.patch("wandb.init")
    def test_init_not_ref_device(self, mock_init):
        """Test initialization with is_ref_device=False."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        
        wrapper = WandbWrapper(config, initialized=True, is_ref_device=False)
        
        mock_init.assert_called_once_with(mode="disabled")
        assert wrapper.initialized is True

    @mock.patch("wandb.init")
    def test_init_with_sweep_params(self, mock_init, mock_config):
        """Test initialization with sweep parameters."""
        sweep_params = ("parameter1", "parameter2")
        
        wrapper = WandbWrapper(
            mock_config,
            initialized=True,
            is_ref_device=True,
            sweep_params=sweep_params
        )
        
        mock_init.assert_called_once()
        init_kwargs = mock_init.call_args.kwargs
        # Verify sweep params were excluded but loss_name was included
        assert "parameter1" not in init_kwargs["config"]
        assert "parameter2" not in init_kwargs["config"]
        assert "loss_name" in init_kwargs["config"]

    def test_is_initialized(self):
        """Test is_initialized method."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        
        wrapper = WandbWrapper(config, initialized=True)
        assert wrapper.is_initialized() is True
        
        wrapper = WandbWrapper(config, initialized=False)
        assert wrapper.is_initialized() is False

    @mock.patch("wandb.log")
    def test_log(self, mock_log):
        """Test log method."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        wrapper = WandbWrapper(config)
        
        log_data = {"metric1": 0.5, "metric2": 0.8}
        wrapper.log(log_data)
        
        mock_log.assert_called_once_with(log_data)

    @mock.patch("wandb.log")
    @mock.patch("wandb.Image")
    def test_log_plot(self, mock_image, mock_log):
        """Test log_plot method."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        wrapper = WandbWrapper(config)
        
        # Create mock plot data
        plot_data = {"plot1": "plot_object1", "plot2": "plot_object2"}
        # Set up the mock to return specific values for each call
        mock_image.side_effect = lambda x: f"wandb_image_of_{x}"
        
        wrapper.log_plot(plot_data)
        
        # Check that Image was called for each plot
        assert mock_image.call_count == 2
        mock_image.assert_any_call("plot_object1")
        mock_image.assert_any_call("plot_object2")
        
        # Check that log was called with the correct transformed data
        expected_log_data = {
            "plot1": "wandb_image_of_plot_object1",
            "plot2": "wandb_image_of_plot_object2"
        }
        mock_log.assert_called_once_with(expected_log_data)

    @mock.patch("wandb.run")
    def test_get_run_id(self, mock_run):
        """Test get_run_id method."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        mock_run.id = "test-run-id"
        
        wrapper = WandbWrapper(config)
        run_id = wrapper.get_run_id()
        
        assert run_id == "test-run-id"

    @mock.patch("wandb.config")
    def test_config_update(self, mock_config):
        """Test config_update method."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        wrapper = WandbWrapper(config)
        
        update_data = {"param1": "new_value1", "param2": "new_value2"}
        wrapper.config_update(update_data)
        
        mock_config.update.assert_called_once_with(
            update_data, allow_val_change=True
        )

    @mock.patch("wandb.finish")
    def test_finish(self, mock_finish):
        """Test finish method."""
        config = mock.MagicMock(spec=HeartWiseConfig)
        wrapper = WandbWrapper(config)
        
        wrapper.finish()
        
        mock_finish.assert_called_once() 