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
        """Initialise the default process group with a robust rank/device mapping."""
        # Resolve ranks from the launcher when available. torchrun exports both
        # LOCAL_RANK (device index on the node) and RANK (global process id).
        env_local_rank = os.environ.get("LOCAL_RANK")
        env_rank = os.environ.get("RANK")

        if env_local_rank is not None:
            local_rank = int(env_local_rank)
        elif isinstance(gpu_id, int):
            local_rank = gpu_id
        else:
            local_rank = 0

        if env_rank is not None:
            rank = int(env_rank)
        elif isinstance(gpu_id, int):
            rank = gpu_id
        else:
            rank = 0

        # Use gloo backend when CUDA is not available, otherwise NCCL.
        backend = 'gloo'

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = 'nccl'
            print(f"Using CUDA device {local_rank} for global rank {rank}")

        # Initialise the process group using the resolved rank information.
        DistributedUtils.dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        
    @staticmethod
    def ddp_cleanup():
        """
        Cleanup the DistributedDataParallel.
        """
        if dist.is_initialized():
            destroy_process_group()

    @staticmethod
    def sync_process_group(
        world_size: int,
        device_ids: int
    ):
        """
        Synchronize the process group across all devices.
        """
        if world_size > 1 and dist.is_initialized():
            torch.distributed.barrier()
    
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
