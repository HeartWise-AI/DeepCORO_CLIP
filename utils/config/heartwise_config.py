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
        yaml_config: Dict[str, Any] = load_yaml(yaml_path)
        pipeline_project = yaml_config.get('pipeline_project', None)
        
        # Better error handling
        if pipeline_project is None:
            raise ValueError("pipeline_project is not set in the yaml file")
            
        registered_config = ConfigRegistry.get(pipeline_project)
        if registered_config is None:
            raise ValueError(f"No registered config found for pipeline_project: {pipeline_project}")
        data_parameters: Dict[str, Any] = {}
        for key, value in yaml_config.items():
            if key in registered_config.__dataclass_fields__:
                data_parameters[key] = value
        return registered_config(**data_parameters)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return asdict(self) 
