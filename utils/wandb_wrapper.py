import wandb
from typing import Any
from utils.config.heartwise_config import HeartWiseConfig

class WandbWrapper:
    """
    A wrapper class for integrating Weights & Biases (wandb) logging with an optional initialization
    strategy based on device roles.

    Attributes:
        config (HeartWiseConfig): The configuration instance containing necessary wandb settings.
        initialized (bool): Indicates if wandb should be initialized.
    """    
    def __init__(
        self, 
        config: HeartWiseConfig,
        initialized: bool = False,
        is_ref_device: bool = False
    ):
        """
        Initializes wandb logging based on the provided flags.

        Args:
            config (HeartWiseConfig): Configuration settings.
            initialized (bool): Whether to initialize wandb.
            is_ref_device (bool): If True, initializes wandb for full logging; 
                                    otherwise, wandb is set into a disabled mode.
           """        
        self.config = config
        if initialized:
            if is_ref_device:
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=config,
                )
            else:
                wandb.init(mode="disabled")
        self.initialized: bool = initialized
        
    def is_initialized(self)->bool:
        return self.initialized
        
    def log(self, kwargs: dict[str, Any]):
        wandb.log(kwargs)

    def finish(self):
        wandb.finish()