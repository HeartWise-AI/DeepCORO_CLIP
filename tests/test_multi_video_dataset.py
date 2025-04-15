import os
import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from unittest.mock import patch, MagicMock
from dataloaders.multi_video_dataset import MultiVideoDataset, multi_video_collate_fn
from tests.templates import DatasetTestsMixin


class TestMultiVideoDataset(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory and CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv_path = os.path.join(self.temp_dir.name, "test_data.csv")
        
        # Create dummy video paths (we'll mock the actual video loading)
        self.video_paths = []
        
        # Create study data with multiple videos per study
        studies = {
            "study1": 3,  # 3 videos
            "study2": 2,  # 2 videos
            "study3": 4   # 4 videos
        }
        
        # Create dummy CSV data
        rows = []
        for study_id, num_videos in studies.items():
            for i in range(num_videos):
                video_path = os.path.join(self.temp_dir.name, f"{study_id}_video_{i}.mp4")
                self.video_paths.append(video_path)
                rows.append({
                    "FileName": video_path,
                    "Split": "train" if study_id != "study3" else "val",
                    "StudyInstanceUID": study_id,
                    "Report": f"Report for {study_id} video {i}"
                })
        
        df = pd.DataFrame(rows)
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
        self.load_video_patcher = patch('dataloaders.multi_video_dataset.load_video')
        self.mock_load_video = self.load_video_patcher.start()
        # We'll set the return value in each test as needed
        
        # Mock file existence check
        self.file_exists_patcher = patch('os.path.exists')
        self.mock_exists = self.file_exists_patcher.start()
        self.mock_exists.return_value = True
        
    def tearDown(self):
        """Clean up after each test method."""
        self.tokenizer_patcher.stop()
        self.load_video_patcher.stop()
        self.file_exists_patcher.stop()
        self.temp_dir.cleanup()
        
    def test_init(self):
        """Test initialization and data grouping."""
        dataset = MultiVideoDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            groupby_column="StudyInstanceUID",
            num_videos=4,
            backbone="mvit",
            mean=self.mean,
            std=self.std
        )
        
        # Check internal state
        self.assertEqual(len(dataset.study_ids), 2)  # Only train studies (study1, study2)
        self.assertEqual(len(dataset.study_to_videos["study1"]), 3)
        self.assertEqual(len(dataset.study_to_videos["study2"]), 2)
        self.assertIn("study1", dataset.study_to_text)
        self.assertIn("study2", dataset.study_to_text)
        
    def test_getitem(self):
        """Test __getitem__ functionality."""
        # Set up mock to return non-zero values for video content
        self.mock_load_video.return_value = np.ones((16, 224, 224, 3), dtype=np.float32)
        
        dataset = MultiVideoDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            groupby_column="StudyInstanceUID",
            num_videos=4,
            backbone="mvit",
            mean=self.mean,
            std=self.std
        )
        
        # Get the first item
        multi_stack, encoded, sid = dataset[0]
        
        # Check the types and shapes
        self.assertIsInstance(multi_stack, np.ndarray)
        self.assertEqual(multi_stack.shape, (4, 16, 224, 224, 3))  # 4 videos, each with 16 frames
        self.assertIsInstance(encoded, dict)
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertIsInstance(sid, str)
        
        # Test padding when fewer videos available than num_videos
        # Check zero padding for study with fewer videos
        if len(dataset.study_to_videos["study2"]) < 4:
            # Get the second study item
            multi_stack, _, _ = dataset[1]  # study2 index
            
            # The first videos should be non-zero (actual videos)
            first_videos_sum = np.sum(multi_stack[:len(dataset.study_to_videos["study2"])])
            self.assertGreater(first_videos_sum, 0)
            
            # The padded videos should be all zeros
            padded_videos_sum = np.sum(multi_stack[len(dataset.study_to_videos["study2"]):])
            self.assertEqual(padded_videos_sum, 0)
        
    def test_shuffle_videos(self):
        """Test video shuffling functionality."""
        # Set up mock to return non-zero values
        self.mock_load_video.return_value = np.ones((16, 224, 224, 3), dtype=np.float32)
        
        # Create two datasets - one with shuffling, one without
        dataset_no_shuffle = MultiVideoDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            groupby_column="StudyInstanceUID",
            shuffle_videos=False,
            seed=42,  # Fixed seed for reproducibility
            mean=self.mean,
            std=self.std
        )
        
        # With fixed seed, we expect deterministic shuffling
        dataset_with_shuffle = MultiVideoDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            groupby_column="StudyInstanceUID",
            shuffle_videos=True,
            seed=42,  # Fixed seed for reproducibility
            mean=self.mean,
            std=self.std
        )
        
        # Get the video paths for verification
        study_id = dataset_no_shuffle.study_ids[0]
        original_paths = dataset_no_shuffle.study_to_videos[study_id].copy()
        
        # Check if order was maintained in the non-shuffled dataset
        # This is tricky to validate since we're just mocking the videos,
        # but we can at least verify the call order to load_video
        
        # Reset mock for first dataset getitem call
        self.mock_load_video.reset_mock()
        dataset_no_shuffle[0]  # This will load videos in original order
        
        # Get the calls made to load_video
        calls_no_shuffle = self.mock_load_video.call_args_list
        paths_no_shuffle = [call[0][0] for call in calls_no_shuffle]
        
        # Reset mock for second dataset getitem call  
        self.mock_load_video.reset_mock()
        dataset_with_shuffle[0]  # This will load videos in shuffled order
        
        # Get the calls made to load_video
        calls_with_shuffle = self.mock_load_video.call_args_list
        paths_with_shuffle = [call[0][0] for call in calls_with_shuffle]
        
        # If shuffling is working, the paths should be in different orders
        # Note: With a fixed seed, this should be deterministic
        if len(original_paths) > 1:  # Only meaningful with multiple videos
            # Check if orders are different (if shuffling actually occurred)
            # Note: There's a small chance they could be the same by random chance
            # especially with small numbers of videos
            self.assertNotEqual(paths_no_shuffle, paths_with_shuffle)
    
    def test_utility_methods(self):
        """Test utility methods like get_reports and get_video_paths."""
        dataset = MultiVideoDataset(
            root=self.temp_dir.name,
            data_filename=os.path.basename(self.temp_csv_path),
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            groupby_column="StudyInstanceUID",
            mean=self.mean,
            std=self.std
        )
        
        # Test get_reports
        study_ids = dataset.study_ids
        reports = dataset.get_reports(study_ids)
        self.assertEqual(len(reports), len(study_ids))
        for report in reports:
            self.assertIsInstance(report, str)
            
        # Test get_all_reports
        all_reports = dataset.get_all_reports()
        self.assertEqual(len(all_reports), len(study_ids))
        
        # Test get_video_paths
        video_paths = dataset.get_video_paths(study_ids[0])
        self.assertIsInstance(video_paths, list)
        self.assertGreater(len(video_paths), 0)
        
    def test_collate_fn(self):
        """Test the multi_video_collate_fn function."""
        # Create sample batch data with shape expected for multi-video
        batch = [
            (
                np.zeros((4, 16, 224, 224, 3), dtype=np.float32),
                {"input_ids": torch.zeros(512), "attention_mask": torch.ones(512)},
                "study1"
            ),
            (
                np.ones((4, 16, 224, 224, 3), dtype=np.float32),
                {"input_ids": torch.ones(512), "attention_mask": torch.ones(512)},
                "study2"
            )
        ]
        
        # Apply collate function
        collated = multi_video_collate_fn(batch)
        
        # Check the collated batch
        self.assertIn("videos", collated)
        self.assertIn("encoded_texts", collated)
        self.assertIn("paths", collated)
        
        # Check shapes and types
        self.assertEqual(collated["videos"].shape, torch.Size([2, 4, 16, 224, 224, 3]))
        self.assertIsInstance(collated["videos"], torch.Tensor)
        
        self.assertIn("input_ids", collated["encoded_texts"])
        self.assertIn("attention_mask", collated["encoded_texts"])
        self.assertEqual(collated["encoded_texts"]["input_ids"].shape, torch.Size([2, 512]))
        
        self.assertEqual(len(collated["paths"]), 2)
        self.assertEqual(collated["paths"][0], "study1")


if __name__ == '__main__':
    unittest.main() 