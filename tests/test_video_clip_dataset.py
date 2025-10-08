import os
import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from unittest.mock import patch, MagicMock
from dataloaders.video_clip_dataset import VideoClipDataset, custom_collate_fn
from tests.templates import DatasetTestsMixin


class TestVideoClipDataset(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory and CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv_path = os.path.join(self.temp_dir.name, "test_data.csv")
        
        # Create dummy video paths (we'll mock the actual video loading)
        self.video_paths = [
            os.path.join(self.temp_dir.name, f"video_{i}.mp4") 
            for i in range(5)
        ]
        
        # Create dummy CSV data
        data = {
            "target_video_path": self.video_paths,
            "Split": ["train", "train", "val", "val", "test"],
            "Report": [
                "This is report 1.",
                "This is report 2.",
                "This is report 3.",
                "This is report 4.",
                "This is report 5."
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.temp_csv_path, sep="α", index=False)
        
        # Default mean and std values for tests
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Mock tokenizer
        self.tokenizer_patcher = patch('models.text_encoder.get_tokenizer')
        self.mock_tokenizer = self.tokenizer_patcher.start()
        
        # Create a mock tokenizer with the necessary functionality
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.zeros((512,), dtype=torch.long),
            "attention_mask": torch.ones((512,), dtype=torch.long)
        }
        self.mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock video loading
        self.load_video_patcher = patch('dataloaders.video_clip_dataset.load_video')
        self.mock_load_video = self.load_video_patcher.start()
        self.mock_load_video.return_value = np.zeros((32, 224, 224, 3), dtype=np.float32)
        
        # Mock file existence check
        self.file_exists_patcher = patch('os.path.exists')
        self.mock_exists = self.file_exists_patcher.start()
        self.mock_exists.return_value = True
        
        # Mock VideoCapture for video validation
        self.cv2_patcher = patch('cv2.VideoCapture')
        self.mock_video_capture = self.cv2_patcher.start()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.release.return_value = None
        mock_cap.open.return_value = True
        self.mock_video_capture.return_value = mock_cap
        
    def tearDown(self):
        """Clean up after each test method."""
        self.tokenizer_patcher.stop()
        self.load_video_patcher.stop()
        self.file_exists_patcher.stop()
        self.cv2_patcher.stop()
        self.temp_dir.cleanup()
        
    def test_init(self):
        """Test initialization of VideoClipDataset."""
        dataset = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="target_video_path",
            num_frames=32,
            backbone="default",
            debug_mode=False,
            mean=self.mean,
            std=self.std
        )
        
        # Check internal state
        self.assertEqual(len(dataset.fnames), 2)  # 2 train videos
        self.assertEqual(len(dataset.outcome), 2)
        self.assertEqual(len(dataset.valid_indices), 2)
        
    def test_load_data(self):
        """Test data loading functionality."""
        # Test with 'all' split
        dataset = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="all",  # Load all data
            target_label="Report",
            datapoint_loc_label="target_video_path",
            mean=self.mean,
            std=self.std
        )
        
        # Should load all videos
        self.assertEqual(len(dataset.fnames), 5)
        
        # Test with specific split
        dataset = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="val",
            target_label="Report",
            datapoint_loc_label="target_video_path",
            mean=self.mean,
            std=self.std
        )
        
        # Should load only val videos
        self.assertEqual(len(dataset.fnames), 2)
        
        # Test missing external path
        with self.assertRaises(ValueError):
            dataset = VideoClipDataset(
                root=self.temp_dir.name,
                data_filename=os.path.basename(self.temp_csv_path),
                split="nonexistent",  # This split doesn't exist
                target_label="Report",
                datapoint_loc_label="target_video_path",
                mean=self.mean,
                std=self.std
            )
        
    def test_validate_videos(self):
        """Test video validation functionality."""
        # Test with debug_mode=True to trigger validation
        dataset = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="target_video_path",
            debug_mode=True,
            mean=self.mean,
            std=self.std
        )
        
        # All videos should be valid
        self.assertEqual(len(dataset.valid_indices), 2)
        
        # Test with a failing video
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]  # First video ok, second fails
        mock_cap.release.return_value = None
        mock_cap.open.return_value = True
        self.mock_video_capture.return_value = mock_cap
        
        dataset = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="target_video_path",
            debug_mode=True,
            mean=self.mean,
            std=self.std
        )
        
        # Only one video should be valid
        self.assertEqual(len(dataset.valid_indices), 1)
        
    def test_getitem(self):
        """Test __getitem__ functionality."""
        # For default backbone, return 32 frames
        self.mock_load_video.return_value = np.zeros((32, 224, 224, 3), dtype=np.float32)
        
        dataset = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="target_video_path",
            mean=self.mean,
            std=self.std
        )
        
        # Test successful item retrieval
        video, encoded, path, raw_text = dataset[0]
        
        # Check the types and shapes
        self.assertIsInstance(video, np.ndarray)
        self.assertEqual(video.shape, (32, 224, 224, 3))
        self.assertIsInstance(encoded, dict)
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertIsInstance(path, str)
        self.assertIsInstance(raw_text, str)
        
        # Test with MVit backbone (should force 16 frames)
        # Create a new dataset with mvit backbone
        dataset_mvit = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="target_video_path",
            backbone="mvit",
            mean=self.mean,
            std=self.std
        )
        
        # Mock load_video to return 16 frames when the dataset has mvit backbone
        def mock_load_video_side_effect(*args, **kwargs):
            if kwargs.get('backbone', '').lower() == 'mvit':
                return np.zeros((16, 224, 224, 3), dtype=np.float32)
            return np.zeros((32, 224, 224, 3), dtype=np.float32)
        
        self.mock_load_video.side_effect = mock_load_video_side_effect
        
        # Get item from MVit dataset, should receive 16 frames
        video, _, _, _ = dataset_mvit[0]
        self.assertEqual(video.shape[0], 16)
        
        # Test the error case where video shape doesn't match backbone requirements
        # Reset the mock to always return 32 frames, which will cause error with MVit
        self.mock_load_video.side_effect = None
        self.mock_load_video.return_value = np.zeros((32, 224, 224, 3), dtype=np.float32)
        
        # Should raise RuntimeError (not ValueError) because the VideoClipDataset.__getitem__ 
        # method wraps all exceptions in RuntimeError
        with self.assertRaises(RuntimeError):
            video, _, _ = dataset_mvit[0]
    
    def test_utility_methods(self):
        """Test utility methods like get_reports and get_all_reports."""
        dataset = VideoClipDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="target_video_path",
            mean=self.mean,
            std=self.std
        )
        
        # Test get_reports
        report = dataset.get_reports([dataset.fnames[0]])
        self.assertEqual(len(report), 1)
        self.assertIsInstance(report[0], str)
        
        # Test get_all_reports
        all_reports = dataset.get_all_reports()
        self.assertEqual(len(all_reports), 2)  # 2 train videos
        
        # Test get_reports with missing path
        missing_report = dataset.get_reports(["nonexistent_path"])
        self.assertEqual(len(missing_report), 1)
        self.assertEqual(missing_report[0], "")
    
    def test_custom_collate_fn(self):
        """Test custom_collate_fn functionality."""
        # Create sample batch data
        batch = [
            (
                np.zeros((32, 224, 224, 3), dtype=np.float32),
                {"input_ids": torch.zeros(512), "attention_mask": torch.ones(512)},
                "video1.mp4",
                "report 1",
            ),
            (
                np.ones((32, 224, 224, 3), dtype=np.float32),
                {"input_ids": torch.ones(512), "attention_mask": torch.ones(512)},
                "video2.mp4",
                "report 2",
            ),
        ]
        
        # Apply collate function
        collated = custom_collate_fn(batch)
        
        # Check the collated batch
        self.assertIn("videos", collated)
        self.assertIn("encoded_texts", collated)
        self.assertIn("paths", collated)
        
        # Check shapes and types
        self.assertEqual(collated["videos"].shape, torch.Size([2, 32, 224, 224, 3]))
        self.assertIsInstance(collated["videos"], torch.Tensor)
        
        self.assertIn("input_ids", collated["encoded_texts"])
        self.assertIn("attention_mask", collated["encoded_texts"])
        self.assertEqual(collated["encoded_texts"]["input_ids"].shape, torch.Size([2, 512]))
        
        self.assertEqual(len(collated["paths"]), 2)
        self.assertEqual(collated["reports"], ["report 1", "report 2"])
        
        # Test with None encoded_texts
        batch_with_none = [
            (
                np.zeros((32, 224, 224, 3), dtype=np.float32),
                None,
                "video1.mp4",
                "report 1",
            ),
            (
                np.ones((32, 224, 224, 3), dtype=np.float32),
                None,
                "video2.mp4",
                "report 2",
            ),
        ]

        collated = custom_collate_fn(batch_with_none)
        self.assertIsNone(collated["encoded_texts"])
        self.assertEqual(collated["reports"], ["report 1", "report 2"])


if __name__ == '__main__':
    unittest.main() 
