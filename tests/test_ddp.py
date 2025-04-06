import os
import torch
import torch.distributed as dist
from unittest import TestCase
from utils.ddp import DistributedUtils

class TestDistributedUtils(TestCase):
    def setUp(self):
        """Set up test environment."""
        self.world_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tearDown(self):
        """Clean up after tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_distributed_setup(self):
        """Test distributed setup."""
        # Initialize distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # Use gloo backend for CPU
        backend = 'gloo' if not torch.cuda.is_available() else 'nccl'
        DistributedUtils.ddp_setup(gpu_id=0, world_size=1)
        
        # Test if distributed is initialized
        self.assertTrue(DistributedUtils.dist.is_initialized())
        
        # Cleanup
        DistributedUtils.ddp_cleanup()

    def test_gather_object(self):
        """Test gathering objects across processes."""
        # Initialize distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # Use gloo backend for CPU
        backend = 'gloo' if not torch.cuda.is_available() else 'nccl'
        DistributedUtils.ddp_setup(gpu_id=0, world_size=1)
        
        # Test gathering objects
        obj = {"test": 123}
        gathered = DistributedUtils.gather_object(obj, world_size=1)
        self.assertEqual(gathered, [obj])
        
        # Cleanup
        DistributedUtils.ddp_cleanup()

    def test_gather_tensor(self):
        """Test gathering tensors across processes."""
        # Initialize distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # Use gloo backend for CPU
        backend = 'gloo' if not torch.cuda.is_available() else 'nccl'
        DistributedUtils.ddp_setup(gpu_id=0, world_size=1)
        
        # Test gathering tensors
        tensor = torch.tensor([1, 2, 3])
        gathered = DistributedUtils.gather_tensor(tensor, world_size=1)
        self.assertTrue(torch.equal(gathered, tensor))
        
        # Cleanup
        DistributedUtils.ddp_cleanup()

    def test_sync_process_group(self):
        """Test synchronizing process group."""
        # Initialize distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # Use gloo backend for CPU
        backend = 'gloo' if not torch.cuda.is_available() else 'nccl'
        DistributedUtils.ddp_setup(gpu_id=0, world_size=1)
        
        # Test synchronization
        DistributedUtils.sync_process_group(world_size=1, device_ids=0)
        self.assertTrue(DistributedUtils.dist.is_initialized())
        
        # Cleanup
        DistributedUtils.ddp_cleanup()

if __name__ == '__main__':
    TestCase.main() 