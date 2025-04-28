import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from utils.parser import HeartWiseParser
from utils.loss.typing import Loss
from utils.registry import (
    LossRegistry, 
    ModelRegistry, 
    RunnerRegistry
)
from utils.wandb_wrapper import WandbWrapper
from runners.typing import Runner
from utils.registry import register_submodules
from utils.enums import (
    SubmoduleType, 
    RunMode
)

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
            
        self.reports = [f"Report for video {i}" for i in range(num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random video tensor [N, T, H, W, C]
        video = np.random.randn(2, self.video_frames, self.height, self.width, 3).astype(np.float32)
        
        # Generate random input_ids and attention_mask
        input_ids = torch.randint(0, 1000, (128,))
        attention_mask = torch.ones_like(input_ids)
        
        # Instead of a single "encoded_texts" tensor, store them as a dict
        encoded_texts = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        return {
            "videos": torch.from_numpy(video),
            "encoded_texts": encoded_texts,
            "attention_mask": attention_mask,
            "paths": self.video_paths[idx]
        }
    
    def get_video_paths(self, sid=None):
        """Return list of video paths for a given sample ID or all paths."""
        if sid is not None:
            # If sid is actually a path, return it as a single-item list
            if isinstance(sid, str) and sid.endswith('.mp4'):
                return [sid]
            # For numeric sid case
            try:
                if isinstance(sid, (int, str)) and int(sid) < len(self.video_paths):
                    return [self.video_paths[int(sid)]]
            except ValueError:
                pass
            return []
        return self.video_paths
    
    def get_reports(self, paths_or_sids):
        """Return reports for given paths or sample IDs."""
        def extract_index(path_or_sid):
            """Helper to extract index from path or sid."""
            if isinstance(path_or_sid, int):
                return path_or_sid if path_or_sid < len(self.reports) else 0
            if isinstance(path_or_sid, str):
                try:
                    if path_or_sid.endswith('.mp4'):
                        # Extract index from filename like 'dummy_video_2.mp4'
                        return int(os.path.basename(path_or_sid).split('_')[-1].split('.')[0])
                    return int(path_or_sid)
                except (ValueError, IndexError):
                    return 0
            return 0

        if isinstance(paths_or_sids, (str, int)):
            # Single path/sid case
            idx = extract_index(paths_or_sids)
            return [self.reports[idx]]
        elif isinstance(paths_or_sids, list):
            # Multiple paths/sids case
            return [self.reports[extract_index(p)] for p in paths_or_sids]
        return self.reports
    
    def cleanup(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)


class TestVideoContrastiveLearning(unittest.TestCase):
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
        sys.argv = [sys.argv[0], "--base_config", "tests/config/clip_base_config.yaml", "--base_checkpoint_path", self.temp_dir]
        self.test_config = HeartWiseParser.parse_config()
        sys.argv = original_argv
        print(f"test_config: {self.test_config}")
        # Create models with test settings
        self.video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone="mvit",
            num_frames=16,
            pretrained=False,
            output_dim=512
        )
        self.text_encoder: TextEncoder = ModelRegistry.get(
            name="text_encoder"
        )(
            output_dim=512,
            freeze_ratio=0.8
        )
        
        # Create dummy datasets and dataloaders
        self.train_dataset = DummyDataset(num_samples=4)
        self.val_dataset = DummyDataset(num_samples=2)
        self.train_loader = DataLoader(self.train_dataset, batch_size=2, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=2)
        
        # Create optimizer and scheduler
        self.optimizer = AdamW([
            {"params": self.video_encoder.parameters(), "name": "video_encoder"},
            {"params": self.text_encoder.parameters(), "name": "text_encoder"}
        ], lr=1e-4)
        
        # Create loss function with test settings
        kwargs = {
            "temperature": 0.07,
            "use_ddp": False
        }
        self.loss_fn: Loss = Loss(
            loss_type=LossRegistry.get(self.test_config.loss_name)(**kwargs)
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
                video_encoder=self.video_encoder,
                text_encoder=self.text_encoder,
                optimizer=self.optimizer,
                scaler=None,  # No mixed precision for testing
                log_temp=torch.tensor(0.07).log().requires_grad_(),
                lr_scheduler=None,  # No scheduler for testing
                loss_fn=self.loss_fn,  # Add loss function
                output_dir=self.temp_dir
            )
        )
        
    @patch('wandb.log')
    def test_train_step(self, mock_log):
        """Test if training step runs without errors."""
        batch = next(iter(self.train_loader))
        videos = batch["videos"]
        input_ids = batch["encoded_texts"]["input_ids"]
        attention_mask = batch["encoded_texts"]["attention_mask"]
        metrics, embeddings = self.runner._train_step(
            videos=videos, 
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(metrics["loss"], float)
        self.assertIsInstance(embeddings, dict)
        self.assertIn("video_embeddings", embeddings)
        self.assertIn("text_embeddings", embeddings)
        
    def test_val_step(self):
        """Test if validation step runs without errors."""
        batch = next(iter(self.val_loader))
        videos = batch["videos"]
        input_ids = batch["encoded_texts"]["input_ids"]
        attention_mask = batch["encoded_texts"]["attention_mask"]
        metrics, embeddings = self.runner._val_step(
            videos=videos, 
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(embeddings, dict)
        
    @patch('wandb.log')
    def test_full_epoch(self, mock_log):
        """Test if full training epoch runs without errors."""
        train_metrics = self.runner._run_epoch(mode=RunMode.TRAIN, epoch=0)
        val_metrics = self.runner._run_epoch(mode=RunMode.VALIDATION, epoch=0)
        
        # Check that both return dictionaries with expected metrics
        self.assertIsInstance(train_metrics, dict)
        self.assertIsInstance(val_metrics, dict)
        
        # Check for essential metrics in training
        self.assertIn('train/loss', train_metrics)
        self.assertIsInstance(train_metrics['train/loss'], float)
        self.assertIn('train/video_norm', train_metrics)
        self.assertIn('train/text_norm', train_metrics)
        self.assertIn('train/alignment_score', train_metrics)
        
        # Check for essential metrics in validation
        self.assertIn('val/loss', val_metrics)
        self.assertIsInstance(val_metrics['val/loss'], float)
        self.assertIn('val/video_norm', val_metrics)
        self.assertIn('val/text_norm', val_metrics)
        self.assertIn('val/alignment_score', val_metrics)
        
    def test_gather_tensor_along_batch(self):
        """Test tensor gathering across batch dimension."""
        local_tensor = torch.randn(2, 512)  # [local_batch_size, dim]
        gathered = self.runner.runner_type._gather_tensor_along_batch(local_tensor, world_size=1)
        self.assertEqual(gathered.shape, (2, 512))  # Single GPU case
        
    def test_gather_strings_across_gpus(self):
        """Test string gathering across GPUs."""
        local_strings = ["text1", "text2"]
        gathered = self.runner.runner_type._gather_strings_across_gpus(local_strings, world_size=1, device=torch.device("cpu"))
        self.assertEqual(gathered, ["text1", "text2"])  # Single GPU case
        
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