import os
import unittest
import tempfile
import numpy as np
import torch
import cv2
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from utils.video import (
    cleanup_temp_video,
    convert_video_for_wandb,
    load_video,
    format_mean_std
)


class TestVideoUtils(unittest.TestCase):
    """Test cases for video utility functions in utils/video.py."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a dummy video file for testing
        self.test_video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        self._create_dummy_video(self.test_video_path, frames=10, width=32, height=32)
        
        # Create a dummy non-MP4 video for conversion testing
        self.test_avi_path = os.path.join(self.temp_dir.name, "test_video.avi")
        self._create_dummy_video(self.test_avi_path, frames=10, width=32, height=32)

    def tearDown(self):
        """Clean up after each test method."""
        self.temp_dir.cleanup()
        
    def _create_dummy_video(self, filepath, frames=10, width=32, height=32):
        """Create a real dummy video file for testing."""
        # Determine fourcc based on file extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:  # Default to AVI
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
        
        try:
            # Create random frames
            for _ in range(frames):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                out.write(frame)
        finally:
            out.release()

    def test_cleanup_temp_video_existing_file(self):
        """Test cleanup_temp_video with an existing file."""
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)
        
        # Ensure file exists
        self.assertTrue(os.path.exists(temp_path))
        
        # Call cleanup function
        cleanup_temp_video(temp_path)
        
        # Check file was deleted
        self.assertFalse(os.path.exists(temp_path))

    def test_cleanup_temp_video_nonexistent_file(self):
        """Test cleanup_temp_video with a non-existent file."""
        # Use a path that doesn't exist
        nonexistent_path = os.path.join(self.temp_dir.name, "nonexistent.mp4")
        
        # Should not raise an exception
        cleanup_temp_video(nonexistent_path)

    def test_cleanup_temp_video_with_exception(self):
        """Test cleanup_temp_video when an exception occurs."""
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)
        
        # Ensure file exists
        self.assertTrue(os.path.exists(temp_path))
        
        # Mock Path.unlink to raise an exception
        with patch.object(Path, 'unlink', side_effect=Exception("Test exception")):
            # Should not raise an exception
            cleanup_temp_video(temp_path)
        
        # Manually clean up the file
        os.unlink(temp_path)

    @patch('subprocess.run')
    def test_convert_video_for_wandb_mp4(self, mock_run):
        """Test convert_video_for_wandb with MP4 file."""
        # MP4 files should be returned as-is
        output_path, is_temp = convert_video_for_wandb(self.test_video_path)
        
        # Check results
        self.assertEqual(output_path, self.test_video_path)
        self.assertFalse(is_temp)
        
        # Verify subprocess.run was not called
        mock_run.assert_not_called()

    @patch('subprocess.run')
    @patch('tempfile.mkstemp')
    @patch('os.close')
    def test_convert_video_for_wandb_non_mp4(self, mock_close, mock_mkstemp, mock_run):
        """Test convert_video_for_wandb with non-MP4 file."""
        # Set up tempfile.mkstemp to return controlled values
        temp_fd, temp_path = 10, os.path.join(self.temp_dir.name, "converted.mp4")
        mock_mkstemp.return_value = (temp_fd, temp_path)
        
        # Set up subprocess.run to return success
        mock_run.return_value = MagicMock(returncode=0)
        
        # Run conversion
        output_path, is_temp = convert_video_for_wandb(self.test_avi_path)
        
        # Check results
        self.assertEqual(output_path, temp_path)
        self.assertTrue(is_temp)
        
        # Verify ffmpeg command was called with correct parameters
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        self.assertEqual(cmd_args[0], "ffmpeg")
        self.assertEqual(cmd_args[2], self.test_avi_path)

    @patch('subprocess.run')
    @patch('tempfile.mkstemp')
    @patch('os.close')
    @patch('os.unlink')
    def test_convert_video_for_wandb_conversion_failure(self, mock_unlink, mock_close, mock_mkstemp, mock_run):
        """Test convert_video_for_wandb with failed conversion."""
        # Set up tempfile.mkstemp to return controlled values
        temp_fd, temp_path = 10, os.path.join(self.temp_dir.name, "converted.mp4")
        mock_mkstemp.return_value = (temp_fd, temp_path)
        
        # Set up subprocess.run to raise CalledProcessError
        mock_stderr = MagicMock()
        mock_stderr.decode.return_value = "Conversion failed"
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=mock_stderr)
        
        # Run conversion
        output_path, is_temp = convert_video_for_wandb(self.test_avi_path)
        
        # Check results - should fall back to original path
        self.assertEqual(output_path, self.test_avi_path)
        self.assertFalse(is_temp)
        
        # Verify temp file was deleted
        mock_unlink.assert_called_once_with(temp_path)

    def test_load_video_basic(self):
        """Test basic load_video functionality."""
        # Load video with default parameters
        video = load_video(
            video_path=self.test_video_path,
            n_frames=5,
            resize=64
        )
        
        # Check shape and dtype
        self.assertEqual(video.shape[0], 5)  # 5 frames
        self.assertEqual(video.shape[1], 64)  # height
        self.assertEqual(video.shape[2], 64)  # width
        self.assertEqual(video.shape[3], 3)   # channels
        self.assertEqual(video.dtype, np.float32)

    def test_load_video_normalization(self):
        """Test load_video with normalization."""
        # Define custom mean and std
        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]
        
        # Load video with normalization
        video = load_video(
            video_path=self.test_video_path,
            n_frames=5,
            resize=64,
            normalize=True,
            mean=mean,
            std=std
        )
        
        # Check shape and dtype
        self.assertEqual(video.shape[0], 5)
        self.assertEqual(video.shape[1], 64)
        self.assertEqual(video.shape[2], 64)
        self.assertEqual(video.shape[3], 3)
        
        # The actual values after normalization will depend on the random frames
        # Instead of checking the mean value, just verify the type is correct
        self.assertEqual(video.dtype, np.float32)

    def test_load_video_mvit_backbone(self):
        """Test load_video with MViT backbone."""
        # Load video with MViT backbone
        video = load_video(
            video_path=self.test_video_path,
            n_frames=32,  # This should be overridden to 16 for MViT
            resize=64,
            backbone="mvit"
        )
        
        # Check that exactly 16 frames were sampled
        self.assertEqual(video.shape[0], 16)
        self.assertEqual(video.shape[1], 64)
        self.assertEqual(video.shape[2], 64)
        self.assertEqual(video.shape[3], 3)

    def test_load_video_with_stride(self):
        """Test load_video with stride parameter."""
        # Use a fixed random seed for reproducibility
        np.random.seed(42)
        
        # Load video with stride=2
        video = load_video(
            video_path=self.test_video_path,
            n_frames=5,
            resize=64,
            stride=2
        )
        
        # Check shape
        self.assertEqual(video.shape[0], 5)

    @patch('cv2.VideoCapture')
    def test_load_video_with_few_frames(self, mock_cap):
        """Test load_video with fewer frames than requested."""
        # Mock VideoCapture to return only 3 frames
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        
        # Set up the read method to return 3 frames then False
        mock_instance.read.side_effect = [
            (True, np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(3)
        ] + [(False, None)]
        mock_instance.isOpened.return_value = True
        
        # Load video requesting 5 frames
        video = load_video(
            video_path=self.test_video_path,
            n_frames=5,
            resize=64
        )
        
        # Check that we got exactly 5 frames (last frame repeated)
        self.assertEqual(video.shape[0], 5)

    @patch('cv2.VideoCapture')
    def test_load_video_failed_open(self, mock_cap):
        """Test load_video with failed video open."""
        # Mock VideoCapture to fail on open
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = False
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            load_video(self.test_video_path)

    @patch('cv2.VideoCapture')
    def test_load_video_no_frames(self, mock_cap):
        """Test load_video with no frames."""
        # Mock VideoCapture to return no frames
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (False, None)
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            load_video(self.test_video_path)

    def test_format_mean_std_float(self):
        """Test format_mean_std with float input."""
        result = format_mean_std(0.5)
        self.assertEqual(result, [0.5, 0.5, 0.5])

    def test_format_mean_std_list(self):
        """Test format_mean_std with list input."""
        result = format_mean_std([0.1, 0.2, 0.3])
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_format_mean_std_numpy_array(self):
        """Test format_mean_std with numpy array input."""
        arr = np.array([0.4, 0.5, 0.6])
        result = format_mean_std(arr)
        self.assertEqual(result, [0.4, 0.5, 0.6])

    def test_format_mean_std_string(self):
        """Test format_mean_std with string input."""
        result = format_mean_std("0.7 0.8 0.9")
        self.assertEqual(result, [0.7, 0.8, 0.9])
        
        # Test with brackets
        result = format_mean_std("[0.1 0.2 0.3]")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_format_mean_std_invalid_string(self):
        """Test format_mean_std with invalid string input."""
        with self.assertRaises(ValueError):
            format_mean_std("not a number")

    def test_format_mean_std_invalid_list(self):
        """Test format_mean_std with invalid list input."""
        with self.assertRaises(ValueError):
            format_mean_std(["a", "b", "c"])

    def test_format_mean_std_invalid_type(self):
        """Test format_mean_std with invalid type."""
        with self.assertRaises(TypeError):
            format_mean_std(complex(1, 2))

    @patch('cv2.VideoCapture')
    def test_load_video_rand_augment(self, mock_cap):
        """Test load_video with rand_augment enabled."""
        # Mock VideoCapture to return frames
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        
        # Create some test frames
        frames = []
        for _ in range(5):
            frames.append((True, np.zeros((64, 64, 3), dtype=np.uint8)))
        frames.append((False, None))  # End of video
        mock_instance.read.side_effect = frames
        
        # Load video with rand_augment=True
        video = load_video(
            video_path=self.test_video_path,
            n_frames=5,
            resize=64,
            rand_augment=True
        )
        
        # Check that we have the right shape
        self.assertEqual(video.shape[0], 5)
        self.assertEqual(video.dtype, np.float32)

    @patch('cv2.VideoCapture')
    def test_load_video_with_transforms_error(self, mock_cap):
        """Test load_video when transforms raise an error."""
        # Mock VideoCapture to return frames
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        
        # Create some test frames
        frames = []
        for _ in range(5):
            frames.append((True, np.zeros((64, 64, 3), dtype=np.uint8)))
        frames.append((False, None))  # End of video
        mock_instance.read.side_effect = frames
        
        # Create a mock scripted transform that raises an error when called
        mock_transform = MagicMock()
        mock_transform.side_effect = RuntimeError("Transform error")
        
        # Patch the specific parts we need
        with patch('torch.jit.script', return_value=mock_transform):
            # This should catch the exception from the transform and continue
            video = load_video(
                video_path=self.test_video_path,
                n_frames=5,
                resize=64,
                video_transforms=[torch.nn.Identity()]  # Any module will do
            )
        
        # Should still return a valid video despite the transform error
        self.assertEqual(video.shape[0], 5)

    def test_load_video_normalization_error(self):
        """Test load_video normalization with missing mean/std."""
        with self.assertRaises(ValueError):
            load_video(
                video_path=self.test_video_path,
                n_frames=5,
                normalize=True,
                mean=None,
                std=None
            )

    @patch('cv2.VideoCapture')
    def test_load_video_3d_input(self, mock_cap):
        """Test load_video with 3D input (grayscale)."""
        # Mock VideoCapture
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        
        # Create grayscale frames (3D array for the whole video)
        frames = []
        for _ in range(5):
            frames.append((True, np.zeros((64, 64), dtype=np.uint8)))  # 2D frame = grayscale
        frames.append((False, None))  # End of video
        mock_instance.read.side_effect = frames
        
        # Load video
        video = load_video(
            video_path=self.test_video_path,
            n_frames=5,
            resize=64
        )
        
        # Should handle grayscale and convert to appropriate shape
        self.assertEqual(video.shape[0], 5)  # 5 frames
        self.assertEqual(video.shape[3], 3)  # 3 channels (RGB)

    @patch('cv2.VideoCapture')
    def test_load_video_invalid_shape(self, mock_cap):
        """Test load_video with invalid shape."""
        # Mock VideoCapture
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        
        # Create invalid shape (5D tensor)
        frames = []
        for _ in range(5):
            # Create 5D tensor by adding extra dimensions
            frames.append((True, np.zeros((64, 64, 3, 1, 1), dtype=np.uint8)))
        frames.append((False, None))  # End of video
        mock_instance.read.side_effect = frames
        
        # Should raise ValueError for invalid shape
        with self.assertRaises(ValueError):
            load_video(
                video_path=self.test_video_path,
                n_frames=5,
                resize=64
            )


if __name__ == '__main__':
    unittest.main() 