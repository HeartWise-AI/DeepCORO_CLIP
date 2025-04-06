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

# Register all submodules
register_submodules(SubmoduleType.RUNNER)
register_submodules(SubmoduleType.MODEL)
register_submodules(SubmoduleType.PROJECT)
register_submodules(SubmoduleType.CONFIG)
register_submodules(SubmoduleType.LOSS)
    
class DummyDataset(Dataset):
    """Dummy dataset that generates random video and text data."""
    def __init__(self, num_samples=100, video_frames=16, height=224, width=224):
        self.num_samples = num_samples
        self.video_frames = video_frames
        self.height = height
        self.width = width
        
        # Create temporary directory for dummy videos
        self.temp_dir = tempfile.mkdtemp()
        self.video_paths = []
        
        # Create dummy MP4 files
        for i in range(num_samples):
            temp_path = os.path.join(self.temp_dir, f"dummy_video_{i}.mp4")
            # Create an empty MP4 file
            with open(temp_path, 'wb') as f:
                f.write(b'dummy mp4 content')
            self.video_paths.append(temp_path)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random video tensor [N, T, H, W, C]
        video = np.random.randn(self.video_frames, self.height, self.width, 3).astype(np.float32)
                        
        return (
            video, 
            {
                "contrast_agent": torch.randint(0, 2, (1,)),
                "main_structure": torch.randint(0, 5, (1,)),
                "stent_presence": torch.randint(0, 2, (1,)),
            },
            self.video_paths[idx]
        )
            
    def cleanup(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)


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
        
        # Create test config by simulating CLI arguments
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], "--base_config", "tests/config/linear_probing_base_config.yaml", "--output_dir", self.temp_dir]
        self.test_config = HeartWiseParser.parse_config()
        sys.argv = original_argv
        print(f"test_config: {self.test_config}")
        
        # Set device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_config.device = self.device
        self.test_config.world_size = 1  # Set to 1 for testing
        
        # Initialize video encoder backbone for linear probing
        video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone=self.test_config.model_name,
            num_frames=self.test_config.frames,
            pretrained=self.test_config.pretrained,
            freeze_ratio=self.test_config.video_freeze_ratio,
            dropout=self.test_config.dropout,
            num_heads=self.test_config.num_heads,
            aggregator_depth=self.test_config.aggregator_depth,
        ).to(self.device)  # Move to device
        
        # Create models with test settings
        self.linear_probing: LinearProbing = ModelRegistry.get(
            name=self.test_config.pipeline_project
        )(
            backbone=video_encoder,
            linear_probing_head=self.test_config.linear_probing_head,
            head_structure=self.test_config.head_structure,
            dropout=self.test_config.dropout,
            freeze_backbone_ratio=self.test_config.video_freeze_ratio,
        )
        
        # Create dummy datasets and dataloaders
        self.train_dataset = DummyDataset(num_samples=4)
        self.val_dataset = DummyDataset(num_samples=2)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=custom_collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=custom_collate_fn
        )
        
        param_groups = [
            {
                'params': self.linear_probing.parameters(),  # Main video backbone
                'lr': self.test_config.lr,
                'name': 'linear_probing',
                'weight_decay': self.test_config.weight_decay
            }
        ]
        optimizer_class: torch.optim.Optimizer = getattr(torch.optim, self.test_config.optimizer)
        optimizer: torch.optim.Optimizer = optimizer_class(param_groups)
        
        # Create loss function
        loss_fn: Loss = Loss(
            loss_type=LossRegistry.get(
                name=LossType.MULTI_HEAD
            )(
                head_structure=self.test_config.head_structure,
                loss_structure=self.test_config.loss_structure,
                head_weights=self.test_config.head_weights,
            )
        )
        
        # Create runner
        wandb_wrapper = WandbWrapper(
            config=self.test_config,
            initialized=False,
            is_ref_device=True,
            sweep_params=()
        )
        self.runner: Runner = Runner(
            runner_type = RunnerRegistry.get(
                name=self.test_config.pipeline_project
            )(
                config=self.test_config,
                wandb_wrapper=wandb_wrapper,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                linear_probing=self.linear_probing,
                optimizer=optimizer,
                scaler=None,  # No mixed precision for testing
                lr_scheduler=None,  # No scheduler for testing
                loss_fn=loss_fn,  # Add loss function
                output_dir=self.temp_dir
            )
        )
        
    @patch('wandb.log')
    def test_train_step(self, mock_log):
        """Test if training step runs without errors."""
        batch = next(iter(self.train_loader))
        batch_video = batch["videos"].unsqueeze(1).to(self.device)  # Move to device
        batch_targets = {k: v.to(self.device) for k, v in batch["targets"].items()}  # Move targets to device
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