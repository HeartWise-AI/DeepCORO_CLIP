"""Training script for DeepCORO_CLIP model."""


import os
import sys
import yaml
import wandb
import torch
from pprint import pprint

from utils.seed import set_seed
from utils.parser import HeartWiseParser
from utils.config import HeartWiseConfig
from utils.registry import ProjectRegistry
from utils.ddp import ddp_setup, ddp_cleanup
from projects import ContrastivePretraining

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_yaml_config(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)

def run_contrastive_pretraining(config: HeartWiseConfig):
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize process group with explicit device ID and world size
    ddp_setup(
        gpu_id=config.gpu, 
        world_size=config.world_size
    )
    
    try:            
        if config.is_ref_device:
            wandb.init()
            pprint(f"Config: {config.to_dict()}")
        else:
            wandb.init(mode='disabled')
            
        # Synchronize the updated config across all GPUs
        if config.world_size > 1:
            torch.distributed.barrier(device_ids=[config.gpu])
            
        # Create and run the project
        project: ContrastivePretraining = ProjectRegistry.get(
            "contrastive_pretraining"
        )(config=config)
        project.run()
        
    except Exception as e:
        print(f"Error: {e}")
        if config.is_ref_device:
            wandb.finish()
        ddp_cleanup()
        raise e
        
    finally:
        if config.is_ref_device:
            wandb.finish()
        ddp_cleanup()

def main(config: HeartWiseConfig):
    """Main function to either run training or hyperparameter sweep."""
    print(f"running main with gpu {config.gpu}")
    
    # Always run training directly - sweep is managed by shell script
    run_contrastive_pretraining(config)

if __name__ == "__main__":
    # Get HeartWiseConfig with GPU info already set
    config: HeartWiseConfig = HeartWiseParser.parse_config()
        
    # Run the main function
    main(config=config)
