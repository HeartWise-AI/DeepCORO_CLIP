from typing import Any

from runners.typing import Runner
from models.video_encoder import VideoEncoder
from projects.base_project import BaseProject
from utils.registry import (
    ProjectRegistry, 
    RunnerRegistry, 
    ModelRegistry
)
from utils.enums import RunMode
from utils.ddp import DistributedUtils
from utils.config.linear_probing_config import LinearProbingConfig


@ProjectRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingProject(BaseProject):
    def __init__(
        self, 
        config: LinearProbingConfig,
    ):
        self.config: LinearProbingConfig = config

    def _setup_training_objects(
        self
    )->dict[str, Any]:
        
        # Create models
        video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone=self.config.backbone,
            num_frames=self.config.num_frames,
            pretrained=self.config.pretrained,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
        )        
        
        video_encoder = video_encoder.to(self.config.device).float()
        print(video_encoder)
        video_encoder = DistributedUtils.DDP(
            video_encoder, 
            device_ids=[self.config.device],
            find_unused_parameters=True
        )
        
        print(f"Video encoder checkpoint path: {self.config.video_encoder_checkpoint_path}")
        checkpoint: dict[str, Any] = self._load_checkpoint(self.config.video_encoder_checkpoint_path)
        print(f"Checkpoint loaded: {checkpoint.keys()}")
        
        video_encoder.module.load_state_dict(checkpoint["video_encoder"])
        print("model loaded!!!!")
        # return {
        #     "video_encoder": video_encoder
        # }

    def _setup_inference_objects(
        self
    )->dict[str, Any]:
        raise NotImplementedError("Inference is not implemented for this project")

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

