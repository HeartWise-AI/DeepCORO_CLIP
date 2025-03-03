import os
import torch
import torch.distributed as dist
import torch.multiprocessing as MP
import torch.utils.data.distributed as DS
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

MP = MP
DS = DS
DDP = DDP
dist = dist


def ddp_setup(gpu_id: int, world_size: int):
    """
    Setup DistributedDataParallel with explicit device ID
    
    Args:
        gpu_id: The GPU ID for this process
        world_size: The total number of GPUs
    """
    # Set the device
    torch.cuda.set_device(gpu_id)
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=gpu_id,
        world_size=world_size
    )
    
def ddp_cleanup():
    destroy_process_group()
    
def gather_loss(loss: list, device: int) -> float:
    loss_tensor = torch.tensor(loss, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.mean().item() / dist.get_world_size()