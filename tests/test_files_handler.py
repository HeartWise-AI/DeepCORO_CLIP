import os
import yaml
import shutil
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import patch, mock_open
from typing import Dict, Any
from dataclasses import dataclass

from utils.files_handler import load_yaml, generate_output_dir_name, backup_config


@dataclass
class MockConfig:
    """Mock configuration class for testing"""
    pipeline_project: str
    base_checkpoint_path: str
    project: str


class TestFilesHandler(TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample config for testing as an object with attributes
        self.sample_config = MockConfig(
            pipeline_project="test_project",
            base_checkpoint_path=self.temp_dir,
            project="test_wandb_project"
        )
        
        # Also create a dict config for tests that need it
        self.sample_config_dict = {
            "pipeline_project": "test_project",
            "base_checkpoint_path": self.temp_dir,
            "project": "test_wandb_project"
        }
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_load_yaml(self):
        """Test loading YAML configuration."""
        # Create a sample YAML file
        yaml_content = """
        pipeline_project: test_project
        base_checkpoint_path: /tmp/test
        project: test_wandb_project
        """
        
        # Mock the open function to return our yaml content
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = load_yaml("dummy_path.yaml")
            
        # Check if the YAML was loaded correctly
        self.assertEqual(config["pipeline_project"], "test_project")
        self.assertEqual(config["base_checkpoint_path"], "/tmp/test")
        self.assertEqual(config["project"], "test_wandb_project")
        
    @patch("time.strftime")
    def test_generate_output_dir_name_with_run_id(self, mock_strftime):
        """Test generating output directory name with wandb run ID."""
        # Mock time.strftime to return a fixed timestamp
        mock_strftime.return_value = "20230101-123456"
        
        # Test with run_id
        run_id = "test_run_123"
        output_dir = generate_output_dir_name(self.sample_config, run_id)
        
        # Expected path with run_id
        expected_path = os.path.join(
            self.temp_dir,
            "test_project",
            "test_wandb_project",
            f"{run_id}_20230101-123456"
        )
        
        self.assertEqual(output_dir, expected_path)
        
    @patch("time.strftime")
    def test_generate_output_dir_name_without_run_id(self, mock_strftime):
        """Test generating output directory name without wandb run ID."""
        # Mock time.strftime to return a fixed timestamp
        mock_strftime.return_value = "20230101-123456"
        
        # Test without run_id
        output_dir = generate_output_dir_name(self.sample_config)
        
        # Expected path without run_id
        expected_path = os.path.join(
            self.temp_dir,
            "test_project",
            "test_wandb_project",
            "20230101-123456_no_wandb"
        )
        
        self.assertEqual(output_dir, expected_path)
        
    def test_backup_config(self):
        """Test backing up configuration to output directory."""
        # Create output directory
        output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test backup_config using dictionary config
        with patch("builtins.open", mock_open()) as mock_file:
            backup_config(self.sample_config_dict, output_dir)
            
            # Check if file was opened with correct path
            expected_path = os.path.join(output_dir, "config.yaml")
            mock_file.assert_called_once_with(expected_path, "w")
            
            # Check if yaml.dump was called with correct arguments
            mock_file().write.assert_called()
            
        # Alternative test that actually writes to disk
        backup_config(self.sample_config_dict, output_dir)
        config_path = os.path.join(output_dir, "config.yaml")
        
        # Verify the file exists
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify contents
        with open(config_path, "r") as f:
            saved_config = yaml.safe_load(f)
            
        self.assertEqual(saved_config, self.sample_config_dict)


if __name__ == "__main__":
    unittest.main() 