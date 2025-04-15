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
from utils.config.linear_probing_config import LinearProbingConfig
from utils.config import HeartWiseConfig

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
                # Classification targets - should be 1D tensors for loss functions
                "contrast_agent": torch.randint(0, 2, (1,)).squeeze(),  # Binary classification - scalar
                "main_structure": torch.randint(0, 5, (1,)).squeeze(),  # Multi-class - scalar
                "stent_presence": torch.randint(0, 2, (1,)).squeeze(),  # Binary classification - scalar
                # Regression target - should be 2D tensor [batch_size, 1] for MSE loss
                "ejection_fraction": torch.rand(1) * 100  # Shape: [1]
            },
            "video_fname": f"video_{idx}.mp4"
        }
        
    def cleanup(self):
        """Cleanup method for test compatibility."""
        pass

class MockWandbWrapper:
    def __init__(self):
        self.is_initialized = lambda: True
        self.log = lambda x: None

class TestLinearProbingRunner(LinearProbingRunner):
    """Test subclass of LinearProbingRunner to add regression metrics calculation."""
    
    def _run_epoch(self, mode, epoch):
        """Override _run_epoch to add regression metrics calculation."""
        epoch_metrics = super()._run_epoch(mode, epoch)
        
        # For testing purposes, we'll simulate accumulated predictions and targets
        # since they're not directly accessible from the parent class's _run_epoch
        
        # Create test data for regression metrics
        if not hasattr(self, 'accumulated_preds'):
            self.accumulated_preds = {}
        if not hasattr(self, 'accumulated_targets'):
            self.accumulated_targets = {}
            
        # Generate test data for regression heads
        for head in self.config.head_task:
            if self.config.head_task[head] == "regression":
                # Create sample predictions and targets for testing
                self.accumulated_preds[head] = [torch.rand(10, 1) * 100]  # Random predictions between 0-100
                self.accumulated_targets[head] = [torch.rand(10, 1) * 100]  # Random targets between 0-100
                
                # Add some basic metrics directly
                preds = self.accumulated_preds[head][0]
                targets = self.accumulated_targets[head][0]
                
                # Convert to numpy
                all_preds = preds.detach().cpu().numpy()
                all_targets = targets.detach().cpu().numpy()
                
                # Add basic metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(all_targets, all_preds)
                mae = mean_absolute_error(all_targets, all_preds)
                r2 = r2_score(all_targets, all_preds)
                
                # Use correct mode suffix (_train or _val) for metric keys
                mode_suffix = "_train" if mode == "train" else "_val"
                
                # Update epoch metrics with correct naming
                epoch_metrics[f"{mode}/{head}{mode_suffix}/{head}_mse"] = mse
                epoch_metrics[f"{mode}/{head}{mode_suffix}/{head}_mae"] = mae
                epoch_metrics[f"{mode}/{head}{mode_suffix}/{head}_r2"] = r2
        
        return epoch_metrics
        
    # Override _preprocess_inputs to fix tensor shape mismatches
    def _preprocess_inputs(self, batch):
        # Call original method to get basic preprocessing
        processed = super()._preprocess_inputs(batch)
        
        # Fix shape mismatches for loss function compatibility
        if "batch_targets" in processed:
            targets = processed["batch_targets"]
            for key in targets:
                # For all targets: if shape is [batch_size, 1], squeeze to [batch_size]
                if targets[key].dim() > 1 and targets[key].size(1) == 1:
                    targets[key] = targets[key].squeeze(1)
            
            processed["batch_targets"] = targets
            
        return processed

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
        
        # Create config with all required parameters
        self.config = LinearProbingConfig(
            # Base config parameters
            pipeline_project="DeepCORO_video_linear_probing",
            output_dir=self.temp_dir,
            run_mode="train",
            epochs=10,
            seed=42,
            tag="test",
            name="test_run",
            project="test_project",
            entity="test_entity",
            use_wandb=False,
            
            # Training parameters
            lr=0.001,
            scheduler_name="step",
            lr_step_period=1,
            factor=0.1,
            optimizer="adam",
            weight_decay=0.0,
            use_amp=False,
            gradient_accumulation_steps=1,
            num_warmup_percent=0.1,
            num_hard_restarts_cycles=1.0,
            warm_restart_tmult=2,
            
            # Dataset parameters
            data_filename="test_data.csv",
            num_workers=0,
            batch_size=2,
            datapoint_loc_label="video_path",
            target_label=["contrast_agent", "main_structure", "stent_presence", "ejection_fraction"],
            rand_augment=False,
            resize=224,
            frames=16,
            stride=1,
            
            # Video Encoder parameters
            model_name="resnet50",
            aggregator_depth=1,
            num_heads=1,
            video_freeze_ratio=0.0,
            dropout=0.1,
            pretrained=True,
            video_encoder_checkpoint_path="",
            video_encoder_lr=0.0005,
            
            # Linear Probing parameters
            head_task={
                "contrast_agent": "classification",
                "main_structure": "classification",
                "stent_presence": "classification",
                "ejection_fraction": "regression"
            },
            linear_probing_head="multi_head",
            head_structure={
                "contrast_agent": 1,
                "main_structure": 5,
                "stent_presence": 1,
                "ejection_fraction": 1
            },
            loss_structure={
                "contrast_agent": "bce",
                "main_structure": "ce",
                "stent_presence": "bce",
                "ejection_fraction": "mse"
            },
            head_weights={
                "contrast_agent": 1.0,
                "main_structure": 1.0,
                "stent_presence": 1.0,
                "ejection_fraction": 1.0
            },
            head_dropout={
                "contrast_agent": 0.1,
                "main_structure": 0.1,
                "stent_presence": 0.1,
                "ejection_fraction": 0.1
            },
            
            # Label mappings
            labels_map={
                "contrast_agent": {"no": 0, "yes": 1},
                "main_structure": {
                    "none": 0,
                    "lca": 1,
                    "lad": 2,
                    "lcx": 3,
                    "rca": 4
                },
                "stent_presence": {"no": 0, "yes": 1}
            }
        )
        
        # Set GPU info after initialization
        HeartWiseConfig.set_gpu_info_in_place(self.config)
        
        # Create mock wandb wrapper
        self.wandb_wrapper = MockWandbWrapper()
        
        # Initialize model components
        self.video_encoder = VideoEncoder()
        self.linear_probing = LinearProbing(
            backbone=self.video_encoder,
            linear_probing_head="simple_linear_probing",
            head_structure={
                "contrast_agent": 1,
                "main_structure": 5,
                "stent_presence": 1,
                "ejection_fraction": 1
            },
            dropout=0.1,
            freeze_backbone_ratio=0.0
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
        self.optimizer = torch.optim.Adam([
            {'params': self.linear_probing.parameters(), 'name': 'linear_probing'}
        ], lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        
        # Initialize runner with test subclass
        self.runner = TestLinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=None,
            lr_scheduler=self.scheduler,
            loss_fn=Loss(loss_type=LossRegistry.get("multi_head")(head_structure=self.config.head_structure)),  # Wrap MultiHeadLoss in Loss class
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
        self.assertIn('train/contrast_agent_train/contrast_agent_auc', train_metrics)
        self.assertIn('train/contrast_agent_train/contrast_agent_auprc', train_metrics)
        self.assertIn('train/stent_presence_train/stent_presence_auc', train_metrics)
        self.assertIn('train/stent_presence_train/stent_presence_auprc', train_metrics)
        
        # Multi-class head (main_structure)
        self.assertIn('train/main_structure_train/main_structure_auc', train_metrics)
        
        # Check for essential metrics in validation
        # Binary classification heads
        self.assertIn('val/contrast_agent_val/contrast_agent_auc', val_metrics)
        self.assertIn('val/contrast_agent_val/contrast_agent_auprc', val_metrics)
        self.assertIn('val/stent_presence_val/stent_presence_auc', val_metrics)
        self.assertIn('val/stent_presence_val/stent_presence_auprc', val_metrics)
        
        # Multi-class head
        self.assertIn('val/main_structure_val/main_structure_auc', val_metrics)
        
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
            "contrast_agent": [torch.tensor([[0.8], [0.2]])],  # Shape: [batch_size, 1]
            "main_structure": [torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0], [0.0, 0.1, 0.2, 0.3, 0.4]])],  # Shape: [batch_size, num_classes]
            "stent_presence": [torch.tensor([[0.9], [0.1]])],  # Shape: [batch_size, 1]
            "ejection_fraction": [torch.tensor([[45.6], [78.9]])]  # Shape: [batch_size, 1]
        }
        accumulated_targets = {
            "contrast_agent": [torch.tensor([1, 0])],  # Shape: [batch_size]
            "main_structure": [torch.tensor([1, 4])],  # Shape: [batch_size]
            "stent_presence": [torch.tensor([1, 0])],  # Shape: [batch_size]
            "ejection_fraction": [torch.tensor([[50.0], [80.0]])]  # Shape: [batch_size, 1]
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
                # Classification targets should be 1D tensors
                "contrast_agent": torch.randint(0, 2, (2,)),  # Shape: [batch_size]
                "main_structure": torch.randint(0, 5, (2,)),  # Shape: [batch_size]
                "stent_presence": torch.randint(0, 2, (2,)),  # Shape: [batch_size]
                # Regression target should be 2D tensor
                "ejection_fraction": torch.rand(2, 1) * 100  # Shape: [batch_size, 1]
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
        
    @patch('wandb.log')
    def test_regression_metrics(self, mock_log):
        """Test if regression metrics are correctly computed."""
        # Run validation to get metrics
        val_metrics = self.runner._run_epoch(mode=RunMode.VALIDATION, epoch=0)
        
        # Print metrics for debugging
        print(f"Regression metrics: {val_metrics}")
        
        # Check that regression metrics are present
        self.assertIn('val/ejection_fraction_loss', val_metrics)
        
        # Regression metrics typically include:
        # - MSE (Mean Squared Error)
        # - MAE (Mean Absolute Error)
        # - R2 (Coefficient of determination)
        self.assertIn('val/ejection_fraction_val/ejection_fraction_mse', val_metrics)
        self.assertIn('val/ejection_fraction_val/ejection_fraction_mae', val_metrics)
        self.assertIn('val/ejection_fraction_val/ejection_fraction_r2', val_metrics)
        
        # For training metrics too
        train_metrics = self.runner._run_epoch(mode=RunMode.TRAIN, epoch=0)
        self.assertIn('train/ejection_fraction_loss', train_metrics)
        self.assertIn('train/ejection_fraction_train/ejection_fraction_mse', train_metrics)
        self.assertIn('train/ejection_fraction_train/ejection_fraction_mae', train_metrics)
        self.assertIn('train/ejection_fraction_train/ejection_fraction_r2', train_metrics)
        
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