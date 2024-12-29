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


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")
    
def ddp_cleanup():
    destroy_process_group()
    
def gather_loss(loss: list, device: int) -> float:
    loss_tensor = torch.tensor(loss, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor.mean().item() / dist.get_world_size()
    return avg_loss