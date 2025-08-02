import unittest
from unittest.mock import Mock, MagicMock
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import GradScaler

from runners.linear_probing_runner import LinearProbingRunner
from utils.config.linear_probing_config import LinearProbingConfig
from utils.wandb_wrapper import WandbWrapper
from utils.loss.typing import Loss


class TestLinearProbingRunner(unittest.TestCase):
    """Test cases for LinearProbingRunner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.config = Mock(spec=LinearProbingConfig)
        self.config.world_size = 1
        self.config.device = 0
        self.config.is_ref_device = True
        self.config.scheduler_name = "linear_with_warmup"
        self.config.head_task = {"test_head": "classification"}
        self.config.head_structure = {"test_head": 2}
        
        # Create mock components
        self.wandb_wrapper = Mock(spec=WandbWrapper)
        self.wandb_wrapper.is_initialized.return_value = False
        
        self.train_loader = Mock(spec=DataLoader)
        self.val_loader = Mock(spec=DataLoader)
        
        self.linear_probing = Mock()
        self.linear_probing.train = Mock()
        
        self.optimizer = Mock(spec=Adam)
        self.scaler = Mock(spec=GradScaler)
        self.lr_scheduler = Mock()
        self.loss_fn = Mock(spec=Loss)
        
        self.output_dir = "/tmp/test_output"
        
    def test_scheduler_is_per_iteration_with_warmup(self):
        """Test _scheduler_is_per_iteration method with warmup scheduler."""
        # Test case where scheduler contains "warmup"
        self.config.scheduler_name = "linear_with_warmup"
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return True because "warmup" is in the scheduler name
        self.assertTrue(runner._scheduler_is_per_iteration())
        
    def test_scheduler_is_per_iteration_with_cosine_warmup(self):
        """Test _scheduler_is_per_iteration method with cosine warmup scheduler."""
        # Test case where scheduler contains "with_warmup"
        self.config.scheduler_name = "cosine_with_warmup"
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return True because "with_warmup" is in the scheduler name
        self.assertTrue(runner._scheduler_is_per_iteration())
        
    def test_scheduler_is_per_iteration_without_warmup(self):
        """Test _scheduler_is_per_iteration method without warmup scheduler."""
        # Test case where scheduler doesn't contain warmup keywords
        self.config.scheduler_name = "step_lr"
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return False because no warmup keywords are present
        self.assertFalse(runner._scheduler_is_per_iteration())
        
    def test_scheduler_is_per_iteration_empty_scheduler_name(self):
        """Test _scheduler_is_per_iteration method with empty scheduler name."""
        # Test case where scheduler name is empty
        self.config.scheduler_name = ""
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return False for empty scheduler name
        self.assertFalse(runner._scheduler_is_per_iteration())


if __name__ == '__main__':
    unittest.main() 