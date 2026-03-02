import wandb
import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple
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
        self._synchronized_run_id = None
        
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
        
    def log(self, kwargs: dict[str, Any], step: Optional[int] = None):
        if step is not None:
            wandb.log(kwargs, step=step)
        else:
            wandb.log(kwargs)
    
    def log_plot(self, kwargs: dict[str, Any]):
        wandb.log({k: wandb.Image(v) for k, v in kwargs.items()})

    def get_run_id(self)->str:
        return wandb.run.id if wandb.run else None
    
    def get_synchronized_run_id(self)->str:
        """
        Get a synchronized run ID across all GPUs.
        Rank 0 broadcasts its run ID to all other ranks.
        """
        if self._synchronized_run_id is not None:
            return self._synchronized_run_id
            
        # Get world size and rank
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            # Not in distributed mode
            self._synchronized_run_id = self.get_run_id()
            return self._synchronized_run_id
        
        # Get run ID from rank 0
        if rank == 0:
            run_id = self.get_run_id()
            if run_id is None:
                run_id = "no_wandb"
            # Convert string to tensor for broadcasting
            run_id_encoded = run_id.encode('utf-8')
            run_id_len = len(run_id_encoded)
            run_id_tensor = torch.tensor([run_id_len], dtype=torch.long).cuda()
        else:
            run_id_tensor = torch.tensor([0], dtype=torch.long).cuda()
        
        # Broadcast the length first
        dist.broadcast(run_id_tensor, src=0)
        run_id_len = run_id_tensor.item()
        
        # Create buffer for the run ID string
        if rank == 0:
            run_id_bytes = torch.tensor(list(run_id_encoded), dtype=torch.uint8).cuda()
        else:
            run_id_bytes = torch.zeros(run_id_len, dtype=torch.uint8).cuda()
        
        # Broadcast the run ID bytes
        dist.broadcast(run_id_bytes, src=0)
        
        # Decode the run ID
        run_id_encoded = bytes(run_id_bytes.cpu().numpy())
        self._synchronized_run_id = run_id_encoded.decode('utf-8')
        
        return self._synchronized_run_id

    def config_update(self, kwargs: dict[str, Any]):
        wandb.config.update(kwargs, allow_val_change=True)

    def finish(self):
        wandb.finish()