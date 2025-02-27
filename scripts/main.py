"""Training script for DeepCORO_CLIP model."""


import os
import sys
import wandb
from typing import Tuple
from pprint import pprint

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
            'gradient_accumulation_steps', 'num_warmup_percent'
        )
        
        wandb_wrapper = WandbWrapper(
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
        if config.is_ref_device:
            wandb_wrapper.finish()
        DistributedUtils.ddp_cleanup()
    
if __name__ == "__main__":
    # Get HeartWiseConfig with GPU info already set
    config: HeartWiseConfig = HeartWiseParser.parse_config()
        
    # Run the main function
    main(config=config)
