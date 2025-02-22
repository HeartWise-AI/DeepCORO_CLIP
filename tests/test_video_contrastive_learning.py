import unittest
import tempfile
import shutil
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader

from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from runners.video_constrative_learning import VideoContrastiveLearningRunner
from utils.config import HeartWiseConfig
from utils.enums import RunMode


class DummyDataset(Dataset):
    """Dummy dataset that generates random video and text data."""
    def __init__(self, num_samples=100, video_frames=32, height=224, width=224):
        self.num_samples = num_samples
        self.video_frames = video_frames
        self.height = height
        self.width = width
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random video tensor [T, H, W, C]
        video = np.random.randn(self.video_frames, self.height, self.width, 3).astype(np.float32)
        
        # Generate random encoded text
        input_ids = torch.randint(0, 1000, (128,))  # Random token IDs
        attention_mask = torch.ones_like(input_ids)  # All tokens attended to
        
        return {
            "videos": torch.from_numpy(video),
            "encoded_texts": {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            },
            "paths": f"dummy_video_{idx}"
        }


class TestVideoContrastiveLearning(unittest.TestCase):
    def setUp(self):
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temp directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy config
        self.config = HeartWiseConfig()
        self.config.recall_k = [1, 5, 10]
        self.config.ndcg_k = [1, 5, 10]
        self.config.world_size = 1
        self.config.is_ref_device = True
        
        # Create dummy datasets
        self.train_dataset = DummyDataset(num_samples=8)  # Small dataset for testing
        self.val_dataset = DummyDataset(num_samples=4)
        
        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=2, shuffle=False)
        
        # Initialize models
        self.video_encoder = VideoEncoder()
        self.text_encoder = TextEncoder()
        
        # Initialize optimizer and other components
        self.optimizer = AdamW(
            [
                {"params": self.video_encoder.parameters(), "name": "video_encoder"},
                {"params": self.text_encoder.parameters(), "name": "text_encoder"}
            ],
            lr=1e-4
        )
        self.scaler = GradScaler()
        self.log_temp = torch.nn.Parameter(torch.ones(1) * np.log(1/0.07))
        
        # Create runner
        self.runner = VideoContrastiveLearningRunner(
            config=self.config,
            device=0,
            world_size=1,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            video_encoder=self.video_encoder,
            text_encoder=self.text_encoder,
            optimizer=self.optimizer,
            scaler=self.scaler,
            log_temp=self.log_temp,
            lr_scheduler=None,
            loss_fn=self._dummy_loss_fn,
            output_dir=self.temp_dir
        )

    def tearDown(self):
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _dummy_loss_fn(video_embeddings, text_embeddings, log_temp):
        """Simple dummy loss function for testing."""
        return torch.mean(video_embeddings * text_embeddings) + torch.exp(log_temp)

    def test_train_step(self):
        """Test a single training step."""
        batch = next(iter(self.train_loader))
        metrics, embeddings = self.runner._train_step(
            videos=batch["videos"].float(),
            input_ids=batch["encoded_texts"]["input_ids"],
            attention_mask=batch["encoded_texts"]["attention_mask"]
        )
        
        # Check metrics
        self.assertIn("loss", metrics)
        self.assertIn("temperature", metrics)
        
        # Check embeddings
        self.assertIn("video_embeddings", embeddings)
        self.assertIn("text_embeddings", embeddings)
        
        # Check shapes
        self.assertEqual(embeddings["video_embeddings"].shape[0], 4)  # Batch size
        self.assertEqual(embeddings["text_embeddings"].shape[0], 4)  # Batch size

    def test_val_step(self):
        """Test a single validation step."""
        batch = next(iter(self.val_loader))
        metrics, embeddings = self.runner._val_step(
            videos=batch["videos"].float(),
            input_ids=batch["encoded_texts"]["input_ids"],
            attention_mask=batch["encoded_texts"]["attention_mask"]
        )
        
        # Check metrics
        self.assertIn("loss", metrics)
        self.assertIn("temperature", metrics)
        
        # Check embeddings
        self.assertIn("video_embeddings", embeddings)
        self.assertIn("text_embeddings", embeddings)
        
        # Check shapes
        self.assertEqual(embeddings["video_embeddings"].shape[0], 2)  # Batch size
        self.assertEqual(embeddings["text_embeddings"].shape[0], 2)  # Batch size

    def test_full_epoch(self):
        """Test running a full epoch."""
        # Run training epoch
        train_metrics = self.runner._run_epoch(mode=RunMode.TRAIN, epoch=0)
        
        # Check training metrics
        self.assertIsInstance(train_metrics, dict)
        self.assertIn("train/loss", train_metrics)
        
        # Run validation epoch
        val_metrics = self.runner._run_epoch(mode=RunMode.VALIDATION, epoch=0)
        
        # Check validation metrics
        self.assertIsInstance(val_metrics, dict)
        self.assertIn("val/loss", val_metrics)

    def test_gather_tensor_along_batch(self):
        """Test tensor gathering functionality."""
        local_tensor = torch.randn(4, 512)  # 4 samples, 512 features
        gathered = self.runner._gather_tensor_along_batch(local_tensor, world_size=1)
        
        # For single GPU case, should return same tensor
        self.assertTrue(torch.equal(local_tensor, gathered))

    def test_gather_strings_across_gpus(self):
        """Test string gathering functionality."""
        local_strings = ["test1", "test2", "test3"]
        gathered = self.runner._gather_strings_across_gpus(
            local_strings, 
            world_size=1,
            device=torch.device("cpu")
        )
        
        # For single GPU case, should return same strings
        self.assertEqual(local_strings, gathered)


if __name__ == '__main__':
    unittest.main() 