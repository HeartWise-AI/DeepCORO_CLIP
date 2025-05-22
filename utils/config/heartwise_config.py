import os
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from utils.files_handler import load_yaml
from utils.registry import ConfigRegistry


@dataclass
class HeartWiseConfig:
    """
    Base configuration class for all HeartWise projects.
    """
    # Pipeline parameters
    pipeline_project: str
    base_checkpoint_path: str    
    run_mode: str
    epochs: int
    seed: int
    
    # wandb parameters
    name: str
    project: str
    entity: str
    use_wandb: bool
    
    @classmethod
    def update_config_with_args(
        cls, 
        base_config: 'HeartWiseConfig', 
        args: Any
    ) -> 'HeartWiseConfig':  
        """Update a HeartWiseConfig instance with command line arguments."""
        data_parameters: Dict[str, Any] = base_config.to_dict().copy()
        registered_config: Any = ConfigRegistry.get(base_config.pipeline_project)
        if registered_config is None:
            raise ValueError(f"No registered config found for pipeline_project: {base_config.pipeline_project}")
        
        for key, value in vars(args).items():
            if value is not None and key in registered_config.__dataclass_fields__:
                data_parameters[key] = value
                
        return registered_config(**data_parameters)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HeartWiseConfig':
        """Create a HeartWiseConfig instance from a YAML file."""
        yaml_data: Dict[str, Any] = load_yaml(yaml_path)
        pipeline_project = yaml_data.get('pipeline_project')
        
        if pipeline_project is None:
            raise ValueError("pipeline_project is not set in the yaml file")
            
        registered_config = ConfigRegistry.get(pipeline_project)
        if registered_config is None:
            raise ValueError(f"No registered config found for pipeline_project: {pipeline_project}")
        
        # Prepare data_parameters, ensuring all fields required by __init__ are present.
        data_parameters: Dict[str, Any] = {}
        for field_name in registered_config.__dataclass_fields__:
            if field_name in yaml_data:
                data_parameters[field_name] = yaml_data[field_name]

        return registered_config(**data_parameters)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return asdict(self)

    @classmethod
    def set_gpu_info_in_place(cls, config: 'HeartWiseConfig') -> None:
        """Set GPU information from environment variables."""
        if "LOCAL_RANK" in os.environ:
            config.device = int(os.environ["LOCAL_RANK"])
            config.world_size = int(os.environ["WORLD_SIZE"])
            config.is_ref_device = (int(os.environ["LOCAL_RANK"]) == 0)
        else: 
            print("No GPU info found in environment variables, defaulting to CPU.")
            config.device = "cpu"
            config.world_size = 1
            config.is_ref_device = True 
