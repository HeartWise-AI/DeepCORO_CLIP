import os
import torch
from typing import Any
from abc import ABC, abstractmethod

from utils.enums import RunMode
from utils.registry import RunnerRegistry
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig
from runners.typing import Runner

class BaseProject(ABC):
    def __init__(
        self, 
        config: HeartWiseConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: HeartWiseConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
            
    @abstractmethod
    def _setup_inference_objects(self)->dict[str, Any]:
        pass
    
    @abstractmethod
    def _setup_training_objects(self)->dict[str, Any]:
        pass
    
    @abstractmethod
    def _update_training_setup_with_checkpoint(
        self, 
        training_setup: dict[str, Any], 
        checkpoint: dict[str, Any]
    )->dict[str, Any]:
        pass
    
    def _load_checkpoint(
        self, 
        checkpoint_path: str
    )->dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        print(
            f"[BaseProject] Loading checkpoint: {checkpoint_path}"
        )
        
        return torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    def run(self):
        training_setup: dict[str, Any] = self._setup_training_objects()

        start_epoch = 0
        if self.config.resume_training:
            checkpoint = self._load_checkpoint(self.config.checkpoint)
            training_setup = self._update_training_setup_with_checkpoint(training_setup, checkpoint)
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch: {start_epoch}")

        runner: Runner = Runner(
            runner_type=RunnerRegistry.get(
                name=self.config.pipeline_project
            )(
                config=self.config,
                wandb_wrapper=self.wandb_wrapper,
                **training_setup,
            )
        )
        if self.config.run_mode == RunMode.TRAIN:
            end_epoch = start_epoch + self.config.epochs
            runner.train(
                start_epoch=start_epoch, 
                end_epoch=end_epoch
            ) 
        elif self.config.run_mode == RunMode.INFERENCE:
            runner.inference()
        else:
            raise ValueError(f"Invalid run mode: {self.config.run_mode}, must be one of {RunMode.TRAIN} or {RunMode.INFERENCE}")

