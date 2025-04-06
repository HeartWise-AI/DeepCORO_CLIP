import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataloaders.video_dataset import custom_collate_fn
from models.video_encoder import VideoEncoder
from models.linear_probing import LinearProbing
from utils.parser import HeartWiseParser
from utils.enums import RunMode
from utils.loss.typing import Loss
from utils.registry import (
    LossRegistry, 
    ModelRegistry, 
    RunnerRegistry
)
from utils.wandb_wrapper import WandbWrapper
from utils.registry import register_submodules 
from utils.enums import SubmoduleType, LossType
from runners.typing import Runner
from runners.linear_probing_runner import LinearProbingRunner
from utils.config import load_config

# Register all submodules
register_submodules(SubmoduleType.RUNNER)
register_submodules(SubmoduleType.MODEL)
register_submodules(SubmoduleType.PROJECT)
register_submodules(SubmoduleType.CONFIG)
register_submodules(SubmoduleType.LOSS)
    
class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.video_shape = (16, 224, 224, 3)  # T, H, W, C
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            "videos": torch.randn(*self.video_shape),
            "targets": {
                "contrast_agent": torch.randint(0, 2, (1,)),
                "main_structure": torch.randint(0, 5, (1,)),
                "stent_presence": torch.randint(0, 2, (1,))
            },
            "video_fname": f"video_{idx}.mp4"
        }


class TestLinearProbing(unittest.TestCase):
    @patch('wandb.init')
    @patch('wandb.log')
    def setUp(self, mock_log, mock_init):
        """Set up test environment before each test method."""
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temp directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), "config/linear_probing_base_config.yaml")
        self.config = load_config(config_path)
        self.config.output_dir = self.temp_dir
        self.config.world_size = 1
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model components
        self.video_encoder = VideoEncoder()
        self.linear_probing = LinearProbing(
            video_encoder=self.video_encoder,
            num_classes=5,
            hidden_dim=512
        )
        
        # Create dummy datasets
        self.train_dataset = DummyDataset(num_samples=10)
        self.val_dataset = DummyDataset(num_samples=5)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=2,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=2,
            shuffle=False
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.linear_probing.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        
        # Initialize runner
        self.runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=None,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=None,
            lr_scheduler=self.scheduler,
            loss_fn=None,
            output_dir=self.temp_dir
        )
        
    @patch('wandb.log')
    def test_train_step(self, mock_log):
        """Test if training step runs without errors."""
        batch = next(iter(self.train_loader))
        batch_video = batch["videos"].unsqueeze(1).to(self.config.device)  # Move to device
        batch_targets = {k: v.to(self.config.device) for k, v in batch["targets"].items()}  # Move targets to device
        outputs = self.runner._train_step(
            batch_video=batch_video, 
            batch_targets=batch_targets
        )
        self.assertIsInstance(outputs, dict)
        self.assertIsInstance(outputs["losses"], dict)
        self.assertIsInstance(outputs["logits"], dict)
                
    def test_val_step(self):
        """Test if validation step runs without errors."""
        batch = next(iter(self.val_loader))
        batch_video = batch["videos"].unsqueeze(1)
        batch_targets = batch["targets"]
        outputs = self.runner._val_step(
            batch_video=batch_video, 
            batch_targets=batch_targets
        )
        self.assertIsInstance(outputs, dict)
        self.assertIsInstance(outputs["losses"], dict)
        self.assertIsInstance(outputs["logits"], dict)
        
    @patch('wandb.log')
    def test_full_epoch(self, mock_log):
        """Test if full training epoch runs without errors."""
        train_metrics = self.runner._run_epoch(mode=RunMode.TRAIN, epoch=0)
        val_metrics = self.runner._run_epoch(mode=RunMode.VALIDATION, epoch=0)
        print(f"train_metrics: {train_metrics}")
        print(f"val_metrics: {val_metrics}")
        
        # Check that both return dictionaries with expected metrics
        self.assertIsInstance(train_metrics, dict)
        self.assertIsInstance(val_metrics, dict)
        
        # Check for essential metrics in training
        # Binary classification heads (contrast_agent and stent_presence)
        self.assertIn('train/contrast_agent_auc', train_metrics)
        self.assertIn('train/contrast_agent_auprc', train_metrics)
        self.assertIn('train/stent_presence_auc', train_metrics)
        self.assertIn('train/stent_presence_auprc', train_metrics)
        
        # Multi-class head (main_structure)
        self.assertIn('train/main_structure_auc', train_metrics)
        
        # Check for essential metrics in validation
        # Binary classification heads
        self.assertIn('val/contrast_agent_auc', val_metrics)
        self.assertIn('val/contrast_agent_auprc', val_metrics)
        self.assertIn('val/stent_presence_auc', val_metrics)
        self.assertIn('val/stent_presence_auprc', val_metrics)
        
        # Multi-class head
        self.assertIn('val/main_structure_auc', val_metrics)
        
        # Check for loss metrics
        self.assertIn('train/contrast_agent_loss', train_metrics)
        self.assertIn('train/main_structure_loss', train_metrics)
        self.assertIn('train/stent_presence_loss', train_metrics)
        self.assertIn('train/main_loss', train_metrics)
        
        self.assertIn('val/contrast_agent_loss', val_metrics)
        self.assertIn('val/main_structure_loss', val_metrics)
        self.assertIn('val/stent_presence_loss', val_metrics)
        self.assertIn('val/main_loss', val_metrics)
                    
    def test_scheduler_per_iteration(self):
        """Test scheduler per iteration detection."""
        # Test with warmup scheduler
        self.runner.config.scheduler_name = "linear_warmup"
        self.assertTrue(self.runner._scheduler_is_per_iteration())
        
        # Test with non-warmup scheduler
        self.runner.config.scheduler_name = "step"
        self.assertFalse(self.runner._scheduler_is_per_iteration())
        
    def test_save_predictions(self):
        """Test saving predictions."""
        # Create dummy data
        accumulated_names = ["video1", "video2"]
        accumulated_preds = {
            "contrast_agent": [torch.tensor([[0.8], [0.2]])],
            "main_structure": [torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0], [0.0, 0.1, 0.2, 0.3, 0.4]])],
            "stent_presence": [torch.tensor([[0.9], [0.1]])]
        }
        accumulated_targets = {
            "contrast_agent": [torch.tensor([[1], [0]])],
            "main_structure": [torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])],
            "stent_presence": [torch.tensor([[1], [0]])]
        }
        
        # Test saving predictions
        self.runner._save_predictions(
            mode=RunMode.VALIDATION,
            accumulated_names=accumulated_names,
            accumulated_preds=accumulated_preds,
            accumulated_targets=accumulated_targets,
            epoch=0
        )
        
        # Verify files were created
        predictions_dir = os.path.join(self.temp_dir, "predictions")
        self.assertTrue(os.path.exists(predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(predictions_dir, "val_predictions_epoch_0.csv")))
        
    def test_inference(self):
        """Test inference method."""
        with self.assertRaises(NotImplementedError):
            self.runner.inference()
            
    def test_preprocess_inputs(self):
        """Test input preprocessing."""
        batch = {
            "videos": torch.randn(2, 16, 224, 224, 3),
            "targets": {
                "contrast_agent": torch.randint(0, 2, (2, 1)),
                "main_structure": torch.randint(0, 5, (2, 1)),
                "stent_presence": torch.randint(0, 2, (2, 1))
            }
        }
        
        processed = self.runner._preprocess_inputs(batch)
        self.assertIn("batch_video", processed)
        self.assertIn("batch_targets", processed)
        self.assertEqual(processed["batch_video"].shape, (2, 1, 16, 224, 224, 3))
        
    def test_train_with_scheduler(self):
        """Test training with scheduler."""
        # Create a simple scheduler
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(self.runner.optimizer, step_size=1, gamma=0.1)
        self.runner.lr_scheduler = scheduler
        
        # Run one epoch
        train_metrics = self.runner._run_epoch(mode=RunMode.TRAIN, epoch=0)
        self.assertIn("train/lr/linear_probing", train_metrics)
        
    def test_validation_with_scheduler(self):
        """Test validation with scheduler."""
        # Create a simple scheduler
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(self.runner.optimizer, step_size=1, gamma=0.1)
        self.runner.lr_scheduler = scheduler
        
        # Run one epoch
        val_metrics = self.runner._run_epoch(mode=RunMode.VALIDATION, epoch=0)
        self.assertIn("val/lr/linear_probing", val_metrics)
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up datasets
        if hasattr(self, 'train_dataset'):
            self.train_dataset.cleanup()
        if hasattr(self, 'val_dataset'):
            self.val_dataset.cleanup()
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main() 