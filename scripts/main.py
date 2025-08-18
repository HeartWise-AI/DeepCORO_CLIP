"""Training script for DeepCORO_CLIP model."""


import os
import sys
import gc
import wandb
import torch
from typing import Tuple
from pprint import pprint

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.enums import SubmoduleType
from utils.ddp import DistributedUtils
from utils.parser import HeartWiseParser
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig
from utils.registry import register_submodules, ProjectRegistry
from projects.typing import Project


# Register all submodules
register_submodules(SubmoduleType.RUNNER)
register_submodules(SubmoduleType.MODEL)
register_submodules(SubmoduleType.PROJECT)
register_submodules(SubmoduleType.CONFIG)
register_submodules(SubmoduleType.LOSS)


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def main(config: HeartWiseConfig):
    
    # Memory and performance optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul with slightly lower precision
        torch.backends.cudnn.allow_tf32 = True  # Faster convolutions
        torch.cuda.empty_cache()  # Clear cache at startup
        torch.cuda.reset_peak_memory_stats()  # Reset memory stats
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize process group with explicit device ID and world size
    DistributedUtils.ddp_setup(
        gpu_id=config.device, 
        world_size=config.world_size
    )
    
    try:
        # List of parameters controlled by sweep
        sweep_params: Tuple[str] = (
            'lr', 'batch_size', 'temperature', 'video_freeze_ratio',
            'text_freeze_ratio', 'dropout', 'num_heads', 'aggregator_depth',
            'optimizer', 'scheduler_name', 'lr_step_period', 'factor',
            'weight_decay', 'loss_name', 'tag', 'name', 'project', 'entity',
            'gradient_accumulation_steps', 'num_warmup_percent',
            # Multitask-specific parameters
            'text_lr', 'captioning_lr', 'mvm_lr',
            'video_weight_decay', 'text_weight_decay', 'captioning_weight_decay', 'mvm_weight_decay',
            'decoder_layers', 'decoder_heads', 'decoder_intermediate_size',
            'max_generation_length', 'mvm_decoder_hidden_size', 'mvm_decoder_layers',
            'mvm_decoder_heads', 'mask_ratio', 'label_smoothing',
            'frames', 'stride', 'max_grad_norm',
            'loss_weights', 'loss_weights.contrastive', 'loss_weights.captioning', 
            'loss_weights.masked_modeling'
        )
        
        wandb_wrapper: WandbWrapper = WandbWrapper(
            config=config,
            initialized=config.use_wandb,
            is_ref_device=config.is_ref_device, 
            sweep_params=sweep_params
        )
                    
        # Synchronize the updated config across all GPUs
        DistributedUtils.sync_process_group(
            world_size=config.world_size,
            device_ids=config.device
        )
            
        # Create and run the project
        project: Project = Project(
            project_type=ProjectRegistry.get(
                name=config.pipeline_project
            )(
                config=config,
                wandb_wrapper=wandb_wrapper
            )
        )
        project.run()
        
    except Exception as e:
        print(f"Error: {e}")
        if config.is_ref_device:
            wandb_wrapper.finish()
        DistributedUtils.ddp_cleanup()
        raise e
        
    finally:
        # Cleanup for sweep runs to prevent memory accumulation
        print("\n🧹 Performing cleanup after run...")
        
        # Clear CUDA cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Print final memory stats
            allocated_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            reserved_gb = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print(f"   Final GPU memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")
        
        # Force garbage collection
        gc.collect()
        
        # Finish wandb run
        if config.is_ref_device:
            wandb_wrapper.finish()
        
        # Cleanup distributed training
        DistributedUtils.ddp_cleanup()
        
        # Additional cleanup for sweep runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        print("   Cleanup completed ✓")
    
if __name__ == "__main__":
    # Get HeartWiseConfig with GPU info already set
    config: HeartWiseConfig = HeartWiseParser.parse_config()

    # Run the main function
    main(config=config)
