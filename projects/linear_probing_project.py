from typing import Any

from utils.enums import RunMode
from runners.typing import Runner
from projects.base_project import BaseProject
from utils.registry import ProjectRegistry, RunnerRegistry
from utils.config.linear_probing_config import LinearProbingConfig


@ProjectRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingProject(BaseProject):
    def __init__(
        self, 
        config: LinearProbingConfig,
    ):
        self.config: LinearProbingConfig = config

    def _setup_training_objects(self)->dict[str, Any]:
        print(f"Model checkpoint path: {self.config.model_checkpoint_path}")

    def run(self):
        training_setup: dict[str, Any] = self._setup_training_objects()
        
        start_epoch = 0
        
        runner: Runner = Runner(
            runner_type=RunnerRegistry.get(
                name=self.config.pipeline_project
            )(
                config=self.config,
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

