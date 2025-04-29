import os
import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from unittest.mock import patch, MagicMock
from dataloaders.stats_dataset import StatsDataset, stats_collate_fn
from tests.templates import DatasetTestsMixin


class TestStatsDataset(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv_path = os.path.join(self.temp_dir.name, "test_data.csv")
        
        # Create dummy video paths (we'll mock the actual video loading)
        self.video_paths = [
            os.path.join(self.temp_dir.name, f"video_{i}.mp4") 
            for i in range(10)  # Create more videos to test max_samples
        ]
        
        # Create dummy CSV data
        data = {
            "target_video_path": self.video_paths,
            "Split": ["train"] * 6 + ["val"] * 4,  # 6 train, 4 val
            "target_label": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.temp_csv_path, sep="α", index=False)
        
        # Mock video loading and validation
        self.patcher = patch('dataloaders.stats_dataset.load_video')
        self.mock_load_video = self.patcher.start()
        self.mock_load_video.return_value = np.zeros((32, 224, 224, 3), dtype=np.float32)
        
        # Mock file existence check to return True for our test files
        self.file_exists_patcher = patch('os.path.exists')
        self.mock_exists = self.file_exists_patcher.start()
        self.mock_exists.return_value = True
        
    def tearDown(self):
        """Clean up after each test method."""
        self.patcher.stop()
        self.file_exists_patcher.stop()
        self.temp_dir.cleanup()
        
    def test_init(self):
        """Test initialization of StatsDataset."""
        dataset = StatsDataset(
            data_filename=self.temp_csv_path,
            split="train",
            target_label="target_label",
            datapoint_loc_label="target_video_path",
            num_frames=32,
            backbone="default"
        )
        
        # Check internal state
        self.assertEqual(len(dataset.fnames), 6)  # 6 train videos
        self.assertEqual(len(dataset.outcome), 6)
        
    def test_max_samples(self):
        """Test max_samples parameter."""
        max_samples = 3
        dataset = StatsDataset(
            data_filename=self.temp_csv_path,
            split="train",
            target_label="target_label",
            max_samples=max_samples
        )
        
        # Should limit to max_samples
        self.assertEqual(len(dataset.fnames), max_samples)
        self.assertEqual(len(dataset.outcome), max_samples)
        
    def test_load_data(self):
        """Test data loading functionality."""
        # Test with 'all' split
        dataset = StatsDataset(
            data_filename=self.temp_csv_path,
            split="all",  # Load all data
            target_label="target_label",
            max_samples=None  # Load all samples
        )
        
        # Should load all videos
        self.assertEqual(len(dataset.fnames), 10)
        
        # Test with specific split
        dataset = StatsDataset(
            data_filename=self.temp_csv_path,
            split="val",
            target_label="target_label"
        )
        
        # Should load only val videos
        self.assertEqual(len(dataset.fnames), 4)
        
    def test_getitem(self):
        """Test __getitem__ functionality."""
        dataset = StatsDataset(
            data_filename=self.temp_csv_path,
            split="train",
            target_label="target_label"
        )
        
        # Test normal video loading
        video, _, path = dataset[0]
        
        # Check the types and shapes
        self.assertIsInstance(video, np.ndarray)
        self.assertEqual(video.shape, (32, 224, 224, 3))
        self.assertIsInstance(path, str)
        
        # Test error handling with failing video
        self.mock_load_video.side_effect = Exception("Mock video loading failure")
        video, _, path = dataset[0]
        self.assertIsNone(video)
        self.assertIsInstance(path, str)
        
    def test_backbone_frame_adjustment(self):
        """Test frame adjustment for different backbones."""
        # Test with default backbone
        dataset_default = StatsDataset(
            data_filename=self.temp_csv_path,
            split="train",
            target_label="target_label",
            num_frames=32,
            backbone="default"
        )
        self.assertEqual(dataset_default.num_frames, 32)
        
        # Test with MVit backbone (should force 16 frames)
        dataset_mvit = StatsDataset(
            data_filename=self.temp_csv_path,
            split="train",
            target_label="target_label",
            num_frames=32,  # This should be overridden
            backbone="mvit"
        )
        self.assertEqual(dataset_mvit.num_frames, 16)
        
    def test_stats_collate_fn(self):
        """Test stats_collate_fn functionality."""
        # Create sample batch data
        batch = [
            (
                np.zeros((32, 224, 224, 3), dtype=np.float32),
                None,
                "video1.mp4"
            ),
            (
                np.ones((32, 224, 224, 3), dtype=np.float32),
                None,
                "video2.mp4"
            )
        ]
        
        # Apply collate function
        collated = stats_collate_fn(batch)
        
        # Check result
        self.assertIsInstance(collated, torch.Tensor)
        self.assertEqual(collated.shape, torch.Size([2, 32, 224, 224, 3]))
        
        # Test with invalid samples
        batch_with_invalid = [
            (None, None, "video1.mp4"),
            (None, None, "video2.mp4")
        ]
        
        # Should raise RuntimeError because no valid samples
        with self.assertRaises(RuntimeError):
            stats_collate_fn(batch_with_invalid)
        
        # Test with mixed valid/invalid samples
        batch_mixed = [
            (None, None, "video1.mp4"),
            (np.ones((32, 224, 224, 3), dtype=np.float32), None, "video2.mp4")
        ]
        
        # Should only use the valid sample
        collated = stats_collate_fn(batch_mixed)
        self.assertEqual(collated.shape, torch.Size([1, 32, 224, 224, 3]))


if __name__ == '__main__':
    unittest.main() 