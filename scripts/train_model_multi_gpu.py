"""Training script for DeepCORO_CLIP model."""


import os
import sys

from projects import ContrastivePretraining

from utils.parser import HeartWiseParser
from utils.config import HeartWiseConfig
from utils.registry import ProjectRegistry
from utils.ddp import ddp_setup, ddp_cleanup


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


def main(config: HeartWiseConfig):
    
    ddp_setup()
    
    try:
        project: ContrastivePretraining = ProjectRegistry.get(
            name="contrastive_pretraining"
        )(config=config)
        
        project.run()
    except Exception as e:
        print(f"Error on GPU {config.gpu}: {str(e)}")
        ddp_cleanup()
        raise e
    finally:
        ddp_cleanup()  



if __name__ == "__main__":
    # Get HeartWiseConfig
    config: HeartWiseConfig = HeartWiseParser.parse_config()
    
    # Setup GPUs information
    config.gpu = int(os.environ["LOCAL_RANK"])
    config.world_size = int(os.environ["WORLD_SIZE"])
    config.is_ref_device = (int(os.environ["LOCAL_RANK"]) == 0)
    
    # Run the main function
    main(config=config)
