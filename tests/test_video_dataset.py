import os
import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from unittest.mock import patch, MagicMock
from dataloaders.video_dataset import VideoDataset, custom_collate_fn
from tests.templates import DatasetTestsMixin
from utils.video import load_video


class TestVideoDataset(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv_path = os.path.join(self.temp_dir.name, "test_data.csv")
        
        # Create dummy video paths (we'll mock the actual video loading)
        self.video_paths = [
            os.path.join(self.temp_dir.name, f"video_{i}.mp4") 
            for i in range(3)
        ]
        
        # Create dummy CSV data
        data = {
            "target_video_path": self.video_paths,
            "Split": ["train", "train", "val"],
            "target_label": [1.0, 0.0, 0.5]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.temp_csv_path, sep="α", index=False)
        
        # Default mean and std values for tests
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Mock video loading and validation
        self.patcher = patch('dataloaders.video_dataset.load_video')
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
        """Test initialization of VideoDataset."""
        dataset = VideoDataset(
            data_filename=self.temp_csv_path,
            split="train",
            target_label=["target_label"],
            datapoint_loc_label="target_video_path",
            num_frames=32,
            backbone="default",
            debug_mode=False,
            mean=self.mean,
            std=self.std
        )
        
        # Check internal state
        self.assertEqual(len(dataset.fnames), 2)  # 2 train videos
        self.assertEqual(len(dataset.outcomes), 2)
        self.assertEqual(len(dataset.valid_indices), 2)
        
    def test_load_data(self):
        """Test data loading functionality."""
        dataset = VideoDataset(
            data_filename=self.temp_csv_path,
            split="all",  # Load all data
            target_label=["target_label"],
            datapoint_loc_label="target_video_path",
            mean=self.mean,
            std=self.std
        )
        
        # Should load all videos
        self.assertEqual(len(dataset.fnames), 3)
        
        # Test with specific split
        dataset = VideoDataset(
            data_filename=self.temp_csv_path,
            split="val",
            target_label=["target_label"],
            datapoint_loc_label="target_video_path",
            mean=self.mean,
            std=self.std
        )
        
        # Should load only val videos
        self.assertEqual(len(dataset.fnames), 1)
        
    def test_getitem(self):
        """Test __getitem__ functionality."""
        dataset = VideoDataset(
            data_filename=self.temp_csv_path,
            split="train",
            target_label=["target_label"],
            datapoint_loc_label="target_video_path",
            mean=self.mean,
            std=self.std
        )
        
        # Get the first item
        video, outcomes, path = dataset[0]
        
        # Check the types and shapes
        self.assertIsInstance(video, np.ndarray)
        self.assertEqual(video.shape, (32, 224, 224, 3))
        self.assertIsInstance(outcomes, dict)
        self.assertIn("target_label", outcomes)
        self.assertIsInstance(path, str)
        
    def test_validate_videos(self):
        """Test video validation functionality."""
        # Create a dataset with debug_mode=True to trigger validation
        with patch('cv2.VideoCapture') as mock_video_capture:
            # Mock the VideoCapture methods
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.read.return_value = (True, np.zeros((224, 224, 3), dtype=np.uint8))
            mock_instance.release.return_value = None
            mock_video_capture.return_value = mock_instance
            
            dataset = VideoDataset(
                data_filename=self.temp_csv_path,
                split="train",
                target_label=["target_label"],
                datapoint_loc_label="target_video_path",
                debug_mode=True,
                mean=self.mean,
                std=self.std
            )
            
            # All videos should be valid
            self.assertEqual(len(dataset.valid_indices), 2)
            
            # Test with a failing video
            mock_instance.isOpened.side_effect = [True, False]
            
            dataset = VideoDataset(
                data_filename=self.temp_csv_path,
                split="train",
                target_label=["target_label"],
                datapoint_loc_label="target_video_path",
                debug_mode=True,
                mean=self.mean,
                std=self.std
            )
            
            # Only one video should be valid
            self.assertEqual(len(dataset.valid_indices), 1)
    
    def test_custom_collate_fn(self):
        """Test custom collate function."""
        # Create sample batch data
        batch = [
            (
                np.zeros((32, 224, 224, 3), dtype=np.float32),
                {"target_label": 0.5},
                "video1.mp4"
            ),
            (
                np.ones((32, 224, 224, 3), dtype=np.float32),
                {"target_label": 0.7},
                "video2.mp4"
            )
        ]
        
        # Apply collate function
        collated = custom_collate_fn(batch)
        
        # Check the collated batch
        self.assertIn("videos", collated)
        self.assertIn("targets", collated)
        self.assertIn("video_fname", collated)
        
        # Check shapes and types
        self.assertEqual(collated["videos"].shape, torch.Size([2, 32, 224, 224, 3]))
        self.assertIsInstance(collated["videos"], torch.Tensor)
        self.assertIsInstance(collated["targets"], dict)
        self.assertIn("target_label", collated["targets"])
        self.assertEqual(collated["targets"]["target_label"].shape, torch.Size([2]))
        self.assertEqual(len(collated["video_fname"]), 2)


if __name__ == '__main__':
    unittest.main() 