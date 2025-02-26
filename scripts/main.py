"""Training script for DeepCORO_CLIP model."""


import os
import sys
import wandb
from pprint import pprint

from utils.seed import set_seed
from utils.enums import SubmoduleType
from utils.parser import HeartWiseParser
from utils.config.heartwise_config import HeartWiseConfig
from utils.registry import register_submodules, ProjectRegistry
from utils.ddp import DistributedUtils
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
        if config.is_ref_device:
            wandb.init()
            pprint(f"Config: {config.to_dict()}")
            
            # List of parameters controlled by sweep
            sweep_params = {
                'lr', 'batch_size', 'temperature', 'video_freeze_ratio',
                'text_freeze_ratio', 'dropout', 'num_heads', 'aggregator_depth',
                'optimizer', 'scheduler_name', 'lr_step_period', 'factor',
                'weight_decay', 'loss_name', 'tag', 'name', 'project', 'entity',
                'gradient_accumulation_steps', 'num_warmup_percent'
            }
            
            # Filter out sweep-controlled parameters
            config_dict = {k: v for k, v in config.to_dict().items() if k not in sweep_params}
            
            wandb.config.update(
                config_dict,
                allow_val_change=True
            )        
        else:
            wandb.init(mode='disabled')
            
        # Synchronize the updated config across all GPUs
        DistributedUtils.sync_process_group(
            world_size=config.world_size,
            device_ids=config.device
        )
            
        # Create and run the project
        project: Project = Project(
            project_type=ProjectRegistry.get(
                name=config.pipeline_project
            )(config=config)
        )
        project.run()
        
    except Exception as e:
        print(f"Error: {e}")
        if config.is_ref_device:
            wandb.finish()
        DistributedUtils.ddp_cleanup()
        raise e
        
    finally:
        if config.is_ref_device:
            wandb.finish()
        DistributedUtils.ddp_cleanup()
    
if __name__ == "__main__":
    # Get HeartWiseConfig with GPU info already set
    config: HeartWiseConfig = HeartWiseParser.parse_config()
        
    # Run the main function
    main(config=config)
