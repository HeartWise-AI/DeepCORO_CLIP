from typing import Any

from runners.typing import Runner
from models.video_encoder import VideoEncoder
from models.linear_probing import LinearProbing
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
        
        # Initialize video encoder backbone for linear probing
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

        # Load video encoder checkpoint
        video_encoder = video_encoder.to(self.config.device).float()        
        checkpoint: dict[str, Any] = self._load_checkpoint(self.config.video_encoder_checkpoint_path)       
        video_encoder.load_state_dict(checkpoint["video_encoder"])

        # Initialize linear probing model
        linear_probing: LinearProbing = ModelRegistry.get(
            name=self.config.pipeline_project
        )(
            backbone=video_encoder,
            linear_probing_head=self.config.linear_probing_head,
            head_structure=self.config.head_structure,
            dropout=self.config.dropout,
            freeze_backbone_ratio=self.config.video_freeze_ratio,
        )
        linear_probing = linear_probing.to(self.config.device).float()

        # Distribute linear probing model
        linear_probing = DistributedUtils.DDP(
            linear_probing, 
            device_ids=[self.config.device]
        )

    def _setup_inference_objects(
        self
    )->dict[str, Any]:
        raise NotImplementedError("Inference is not implemented for this project")

    def run(self):
        training_setup: dict[str, Any] = self._setup_training_objects()
        
        start_epoch: int = 0
        
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

