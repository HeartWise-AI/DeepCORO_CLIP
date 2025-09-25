#!/usr/bin/env python
import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.seed import set_seed
from utils.ddp import DistributedUtils
from utils.parser import HeartWiseParser

print("=" * 50)
print("Starting debug script")
print("=" * 50)

# Get config
config = HeartWiseParser.parse_config()
print(f"Config loaded. Device: {config.device}, World size: {config.world_size}, Is ref: {config.is_ref_device}")

# Set seed
set_seed(config.seed)
print(f"Seed set to {config.seed}")

# Initialize DDP
print(f"Initializing DDP with device={config.device}, world_size={config.world_size}")
DistributedUtils.ddp_setup(gpu_id=config.device, world_size=config.world_size)
print("DDP initialized successfully")

# Try first barrier
print("Attempting first barrier...")
import torch.distributed as dist
if dist.is_initialized():
    print(f"Process group is initialized. Backend: {dist.get_backend()}")
    print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

    # Try a simple tensor operation first
    test_tensor = torch.tensor([dist.get_rank()], dtype=torch.float32).cuda()
    print(f"Created test tensor on device: {test_tensor.device}")

    # Try barrier
    dist.barrier()
    print("Barrier passed!")

# Cleanup
DistributedUtils.ddp_cleanup()
print("Cleanup complete")