import os
import torch
import torch.distributed as dist
from typing import Any
from abc import ABC, abstractmethod

from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig
from utils.files_handler import generate_output_dir_name, backup_config
from utils.ddp import DistributedUtils

class BaseProject(ABC):
    def __init__(
        self, 
        config: HeartWiseConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: HeartWiseConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def _setup_inference_objects(self)->dict[str, Any]:
        pass
    
    @abstractmethod
    def _setup_training_objects(self)->dict[str, Any]:
        pass
    
    def _setup_project(self):
        # Get synchronized run ID across all GPUs
        if self.wandb_wrapper.is_initialized():
            run_id = self.wandb_wrapper.get_synchronized_run_id()
        else:
            run_id = None
        
        # Only rank 0 generates the output directory name
        if self.config.is_ref_device:
            self.config.output_dir = generate_output_dir_name(
                config=self.config, 
                run_id=run_id
            )
            print(f"Output directory: {self.config.output_dir}")
            
            # Create the output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Backup the configuration file
            backup_config(
                config=self.config,
                output_dir=self.config.output_dir
            )
        
        # Broadcast the output directory path to all GPUs
        if dist.is_initialized() and self.config.world_size > 1:
            if self.config.is_ref_device:
                # Encode output directory path
                output_dir_encoded = self.config.output_dir.encode('utf-8')
                output_dir_len = len(output_dir_encoded)
                output_dir_len_tensor = torch.tensor([output_dir_len], dtype=torch.long).cuda()
            else:
                output_dir_len_tensor = torch.tensor([0], dtype=torch.long).cuda()
            
            # Broadcast the length
            dist.broadcast(output_dir_len_tensor, src=0)
            output_dir_len = output_dir_len_tensor.item()
            
            # Create buffer for the path string
            if self.config.is_ref_device:
                output_dir_bytes = torch.tensor(list(output_dir_encoded), dtype=torch.uint8).cuda()
            else:
                output_dir_bytes = torch.zeros(output_dir_len, dtype=torch.uint8).cuda()
            
            # Broadcast the path bytes
            dist.broadcast(output_dir_bytes, src=0)
            
            # Decode the path on non-ref devices
            if not self.config.is_ref_device:
                output_dir_encoded = bytes(output_dir_bytes.cpu().numpy())
                self.config.output_dir = output_dir_encoded.decode('utf-8')
                print(f"[GPU {self.config.device}] Using output directory: {self.config.output_dir}")
        
        # Sync all processes
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
    
    def _load_checkpoint(
        self, 
        checkpoint_path: str
    )->dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        print(
            f"[BaseProject] Loading checkpoint: {checkpoint_path}"
        )
        
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)