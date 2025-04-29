import wandb
from typing import Any, Tuple
from utils.config.heartwise_config import HeartWiseConfig

class WandbWrapper:
    """
    A wrapper class for integrating Weights & Biases (wandb) logging with an optional initialization
    strategy based on device roles.

    Attributes:
        config (HeartWiseConfig): The configuration instance containing necessary wandb settings.
        initialized (bool): Indicates if wandb should be initialized.
        is_ref_device (bool): If True, initializes wandb for full logging; 
                                otherwise, wandb is set into a disabled mode.
        sweep_params (Tuple[str]): List of parameters to be excluded from wandb logging.
    """    
    def __init__(
        self, 
        config: HeartWiseConfig,
        initialized: bool = False,
        is_ref_device: bool = False,
        sweep_params: Tuple[str] = ()
    ):
        """
        Initializes wandb logging based on the provided flags.

        Args:
            config (HeartWiseConfig): Configuration settings.
            initialized (bool): Whether to initialize wandb.
            is_ref_device (bool): If True, initializes wandb for full logging; 
                                    otherwise, wandb is set into a disabled mode.
            sweep_params (Tuple[str]): List of parameters to be excluded from wandb logging.
        """        
        self.config = config
        if initialized:
            if is_ref_device:
                # Filter out sweep-controlled parameters
                config_dict = {
                    k: v for k, v in config.to_dict().items() 
                    if k not in sweep_params
                }
                
                # Ensure loss_name is included even if it's controlled by sweep
                if hasattr(config, 'loss_name'):
                    config_dict['loss_name'] = config.loss_name
                    
                wandb.init(
                    project=config.project,
                    entity=config.entity,
                    config=config_dict,
                    allow_val_change=True
                )
            else:
                wandb.init(mode="disabled")
        self.initialized: bool = initialized
        
    def is_initialized(self)->bool:
        return self.initialized
        
    def log(self, kwargs: dict[str, Any]):
        wandb.log(kwargs)
    
    def log_plot(self, kwargs: dict[str, Any]):
        wandb.log({k: wandb.Image(v) for k, v in kwargs.items()})

    def get_run_id(self)->str:
        return wandb.run.id

    def config_update(self, kwargs: dict[str, Any]):
        wandb.config.update(kwargs, allow_val_change=True)

    def finish(self):
        wandb.finish()