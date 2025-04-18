import os
import yaml
import time
from typing import Dict, Any

def load_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def generate_output_dir_name(
    config: dict[str, Any], 
    run_id: str | None = None
)->str:
    """
    Generates a directory name for output based on the provided configuration.
    """
    current_time: str = time.strftime("%Y%m%d-%H%M%S")

    run_folder: str = f"{run_id}_{current_time}" if run_id is not None else f"{current_time}_no_wandb"

    model_dir: str = os.path.join(
        config.base_checkpoint_path, 
        config.pipeline_project,
        config.wandb_project,
        run_folder
    )
    return model_dir

def backup_config(
    config: dict[str, Any],
    output_dir: str
) -> None:
    """
    Backup the configuration file to the output directory.
    """
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)