import yaml

import wandb


def load_yaml_config(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)


sweep_conf_file_path = "config/sweep_config.yaml"
sweep_conf = load_yaml_config(sweep_conf_file_path)
count = 25  # number of runs to execute

sweep_id = wandb.sweep(sweep_conf, project=sweep_conf["name"], entity="jacques-delfrate")

wandb.agent(sweep_id=sweep_id, entity="jacques-delfrate", project=sweep_conf["name"], count=count)
