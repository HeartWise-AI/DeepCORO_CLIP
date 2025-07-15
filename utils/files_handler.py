import os
import json
import time
import yaml
from typing import Dict, Any


def read_api_key(path: str) -> dict[str, str]:
    with open(path) as f:
        api_key = json.load(f)
    return api_key


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

    wandb_project: str = config.project
    model_dir: str = os.path.join(
        config.base_checkpoint_path, 
        config.pipeline_project,
        wandb_project,
        run_folder
    )
    return model_dir

# ---------------------------------------------------------
# New utility
# ---------------------------------------------------------
def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `updates` into `target` (nested‐dict friendly).
    Values in `updates` override those in `target`.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            target[k] = _deep_update(target[k], v)
        else:
            target[k] = v
    return target

# ---------------------------------------------------------
# Re-worked backup_config
# ---------------------------------------------------------
def backup_config(
    config: Any,
    output_dir: str
) -> None:
    """
    Create a *fully-resolved* copy of the configuration used for this run.

    Steps
    -----
    1. Load the base YAML pointed to by `config.base_config_path` (if present).
    2. Convert the runtime `config` object to a plain dict.
    3. Merge the runtime values over the base YAML (runtime wins).
    4. Write the merged result to `<output_dir>/config.yaml`.
    """
    # 1. Load base config if available
    base_cfg_path: str | None = getattr(config, "base_config_path", None)
    base_cfg: Dict[str, Any] = load_yaml(base_cfg_path) if base_cfg_path and os.path.isfile(base_cfg_path) else {}

    # 2. Convert runtime config to dict
    if hasattr(config, "to_dict"):
        runtime_cfg: Dict[str, Any] = config.to_dict()
    elif hasattr(config, "__dict__"):
        runtime_cfg = dict(config.__dict__)
    else:  # already a mapping
        runtime_cfg = dict(config)

    # 3. Merge (runtime overrides base)
    merged_cfg: Dict[str, Any] = _deep_update(base_cfg, runtime_cfg)

    # 4. Persist
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(merged_cfg, f, default_flow_style=False, sort_keys=False)