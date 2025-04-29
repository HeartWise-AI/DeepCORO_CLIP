import unittest
import torch
import random
import numpy as np
from unittest.mock import patch, MagicMock

from utils.seed import set_seed, seed_worker


class TestSeed(unittest.TestCase):
    def test_set_seed_with_valid_seed(self):
        """Test that set_seed properly sets seeds for all random number generators"""
        test_seed = 42
        
        # Call set_seed with our test seed
        set_seed(test_seed)
        
        # Generate random numbers with each library to verify consistent seeding
        torch_rand = torch.rand(1).item()
        numpy_rand = np.random.rand()
        python_rand = random.random()
        
        # Reset and reseed with the same seed
        set_seed(test_seed)
        
        # Generate random numbers again and verify they're the same
        self.assertEqual(torch_rand, torch.rand(1).item())
        self.assertEqual(numpy_rand, np.random.rand())
        self.assertEqual(python_rand, random.random())
        
    def test_set_seed_with_none(self):
        """Test that set_seed handles None seed value correctly"""
        # This should not raise any errors
        set_seed(None)
        
        # Cannot easily test functionality since it should do nothing,
        # but we can verify it doesn't crash
        
    @patch('torch.cuda.manual_seed_all')
    @patch('torch.manual_seed')
    @patch('numpy.random.seed')
    @patch('random.seed')
    def test_set_seed_calls_all_seed_functions(self, mock_random_seed, 
                                             mock_np_seed, 
                                             mock_torch_seed, 
                                             mock_cuda_seed):
        """Test that set_seed calls all the appropriate seeding functions"""
        test_seed = 123
        set_seed(test_seed)
        
        # Verify each seeding function was called with the correct seed
        mock_random_seed.assert_called_once_with(test_seed)
        mock_np_seed.assert_called_once_with(test_seed)
        mock_torch_seed.assert_called_once_with(test_seed)
        mock_cuda_seed.assert_called_once_with(test_seed)
        
    def test_set_seed_configures_cudnn(self):
        """Test that set_seed configures CuDNN settings correctly"""
        test_seed = 456
        
        # Save original settings to restore later
        original_deterministic = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark
        
        try:
            # First set them to opposite values
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            # Call set_seed which should change them
            set_seed(test_seed)
            
            # Verify they're changed
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            # Restore original settings
            torch.backends.cudnn.deterministic = original_deterministic
            torch.backends.cudnn.benchmark = original_benchmark
        
    def test_seed_worker(self):
        """Test that seed_worker sets numpy and random seeds based on initial seed"""
        # Mock torch.initial_seed() to return a known value
        with patch('torch.initial_seed', return_value=12345):
            # Capture the seeds that numpy and random are set to
            with patch('numpy.random.seed') as mock_np_seed:
                with patch('random.seed') as mock_random_seed:
                    seed_worker(1)  # Worker ID doesn't matter for this test
                    
                    # Both should be set to the same derived seed
                    expected_seed = 12345 % 2**32
                    mock_np_seed.assert_called_once_with(expected_seed)
                    mock_random_seed.assert_called_once_with(expected_seed)
    
    def test_seed_worker_reproducibility(self):
        """Test that seed_worker produces reproducible results"""
        # Create a mock for torch.initial_seed that always returns the same value
        with patch('torch.initial_seed', return_value=54321):
            # Call seed_worker
            seed_worker(0)
            
            # Generate some random numbers
            np_val = np.random.rand()
            py_val = random.random()
            
            # Reset seeds with the same worker seed
            seed_worker(0)
            
            # Verify we get the same random numbers
            self.assertEqual(np_val, np.random.rand())
            self.assertEqual(py_val, random.random())


if __name__ == '__main__':
    unittest.main() 