import torch
import torch.distributed as dist
import torch.multiprocessing as MP
import torch.utils.data.distributed as DS
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Any
import os


class DistributedUtils:
    """Utility class for Distributed Data Parallel (DDP) operations"""

    MP = MP
    DS = DS
    DDP = DDP
    dist = dist

    @staticmethod
    def is_initialized():
        return dist.is_initialized()

    @staticmethod
    def ddp_setup(gpu_id: int, world_size: int):
        """Initialize distributed environment."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = str(gpu_id)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Use gloo backend for CPU
        backend = 'gloo'
        
        # Only set CUDA device if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            backend = 'nccl'
        
        # Initialize process group
        DistributedUtils.dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=gpu_id
        )
        
    @staticmethod
    def ddp_cleanup():
        """
        Cleanup the DistributedDataParallel.
        """
        destroy_process_group()

    @staticmethod
    def sync_process_group(
        world_size: int, 
        device_ids: int
    ):
        """
        Synchronize the process group across all devices.
        """
        if world_size > 1:
            torch.distributed.barrier(device_ids=[device_ids])
    
    @staticmethod
    def gather_loss(
        loss: list, 
        device: int
    ) -> float:
        """
        Gather the loss from all devices and return the mean loss.
        """
        loss_tensor = torch.tensor(loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor.mean().item() / dist.get_world_size()

    @staticmethod
    def gather_object(
        obj: Any,
        world_size: int
    ) -> list:
        """
        Gather an object from all devices and return a list of gathered objects.
        
        Args:
            obj: The object to gather
            world_size: The total number of GPUs
            
        Returns:
            list: A list of objects gathered from all devices
        """
        if world_size <= 1:
            return [obj]
            
        gathered_objects = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_objects, obj)
        return gathered_objects

    @staticmethod
    def gather_tensor(
        tensor: torch.Tensor,
        world_size: int
    ) -> torch.Tensor:
        """
        Gather a tensor from all devices and return the concatenated tensor.
        
        Args:
            tensor: The tensor to gather
            world_size: The total number of GPUs
            
        Returns:
            torch.Tensor: The concatenated tensor from all devices
        """
        if world_size <= 1:
            return tensor
            
        # Create tensors for gathering
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)