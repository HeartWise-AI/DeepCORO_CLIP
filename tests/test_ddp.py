import unittest
import torch
import numpy as np
from utils.ddp import DistributedUtils

class TestDistributedUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.world_size = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_gather_object(self):
        """Test gathering objects across GPUs."""
        # Test with list
        local_list = [1, 2, 3]
        gathered = DistributedUtils.gather_object(local_list, self.world_size)
        self.assertEqual(len(gathered), self.world_size)
        self.assertEqual(gathered[0], local_list)
        
        # Test with dict
        local_dict = {"a": 1, "b": 2}
        gathered = DistributedUtils.gather_object(local_dict, self.world_size)
        self.assertEqual(len(gathered), self.world_size)
        self.assertEqual(gathered[0], local_dict)
        
    def test_gather_tensor(self):
        """Test gathering tensors across GPUs."""
        # Test 1D tensor
        local_tensor = torch.tensor([1, 2, 3], device=self.device)
        gathered = DistributedUtils.gather_tensor(local_tensor, self.world_size)
        self.assertEqual(gathered.shape[0], local_tensor.shape[0] * self.world_size)
        
        # Test 2D tensor
        local_tensor = torch.randn(3, 4, device=self.device)
        gathered = DistributedUtils.gather_tensor(local_tensor, self.world_size)
        self.assertEqual(gathered.shape[0], local_tensor.shape[0] * self.world_size)
        self.assertEqual(gathered.shape[1], local_tensor.shape[1])
        
    def test_sync_process_group(self):
        """Test process group synchronization."""
        # Test with single process
        DistributedUtils.sync_process_group(world_size=1, device_ids=0)
        
        # Test with multiple processes (simulated)
        DistributedUtils.sync_process_group(world_size=self.world_size, device_ids=0)
        
    def test_distributed_setup(self):
        """Test distributed setup."""
        # Test initialization
        DistributedUtils.distributed_setup(rank=0, world_size=self.world_size)
        
        # Test cleanup
        DistributedUtils.cleanup_distributed()
        
if __name__ == '__main__':
    unittest.main() 