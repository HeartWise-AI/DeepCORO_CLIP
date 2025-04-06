import os
import torch
import torch.distributed as dist
from unittest import TestCase
from utils.ddp import DistributedUtils

class TestDistributedUtils(TestCase):
    def setUp(self):
        """Set up test environment."""
        # Initialize distributed environment for testing
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
        
        self.world_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tearDown(self):
        """Clean up after tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_gather_object(self):
        """Test gathering objects across GPUs."""
        # Test with list
        local_list = [1, 2, 3]
        gathered = DistributedUtils.gather_object(local_list, self.world_size)
        self.assertEqual(gathered, [local_list])

        # Test with dict
        local_dict = {'a': 1, 'b': 2}
        gathered = DistributedUtils.gather_object(local_dict, self.world_size)
        self.assertEqual(gathered, [local_dict])

    def test_gather_tensor(self):
        """Test gathering tensors across GPUs."""
        # Test 1D tensor
        local_tensor = torch.tensor([1, 2, 3], device=self.device)
        gathered = DistributedUtils.gather_tensor(local_tensor)
        self.assertTrue(torch.equal(gathered, local_tensor))

        # Test 2D tensor
        local_tensor = torch.tensor([[1, 2], [3, 4]], device=self.device)
        gathered = DistributedUtils.gather_tensor(local_tensor)
        self.assertTrue(torch.equal(gathered, local_tensor))

    def test_sync_process_group(self):
        """Test process group synchronization."""
        # Test with single process
        DistributedUtils.sync_process_group(self.world_size, self.device)

    def test_distributed_setup(self):
        """Test distributed setup and cleanup."""
        # Test setup
        DistributedUtils.distributed_setup(rank=0, world_size=1)
        self.assertTrue(dist.is_initialized())
        
        # Test cleanup
        DistributedUtils.distributed_cleanup()
        self.assertFalse(dist.is_initialized())

if __name__ == '__main__':
    TestCase.main() 