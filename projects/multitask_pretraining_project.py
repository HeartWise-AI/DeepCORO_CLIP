from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from models.text_encoder import TextEncoder
from models.video_encoder import VideoEncoder
from models.captioning_decoder import CaptioningDecoder
from models.masked_video_modeling import MaskedVideoModeling
from projects.base_project import BaseProject
from runners.typing import Runner
from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.enums import RunMode
from utils.config.clip_config import ClipConfig
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry, 
    RunnerRegistry, 
    ProjectRegistry, 
    LossRegistry
)
from utils.wandb_wrapper import WandbWrapper
from utils.video_project import calculate_dataset_statistics_ddp
from dataloaders.video_clip_dataset import get_distributed_video_clip_dataloader


@ProjectRegistry.register('DeepCORO_multitask')
class MultitaskPretrainingProject(BaseProject):
    """
    Multitask pretraining project combining:
    - Contrastive learning (video ↔ text)
    - Captioning (autoregressive report generation)
    - Masked video modeling (self-supervised learning)
    """
    
    def __init__(
        self, 
        config: ClipConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)
        
    def _setup_training_objects(
        self,
    ) -> dict:
        """
        Load training objects for multitask learning.

        Returns:
            dict: Dictionary containing training objects
        """
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)

        train_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="train", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=True,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=True,
        )
        val_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="val", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
        )

        # Create shared video encoder
        video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone=self.config.model_name,
            input_channels=3,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            output_dim=512,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
            aggregate_videos_tokens=False,  # We need token-level features for captioning
        )
        video_encoder = video_encoder.to(self.config.device).float()

        # Create text encoder
        text_encoder: TextEncoder = ModelRegistry.get(
            name="text_encoder"
        )(
            freeze_ratio=self.config.text_freeze_ratio,
            dropout=self.config.dropout,
        )
        text_encoder = text_encoder.to(self.config.device).float()

        # Create captioning decoder
        captioning_decoder: CaptioningDecoder = ModelRegistry.get(
            name="captioning_decoder"
        )(
            vocab_size=getattr(self.config, 'vocab_size', 30522),
            hidden_size=512,
            num_layers=getattr(self.config, 'decoder_layers', 6),
            num_heads=getattr(self.config, 'decoder_heads', 8),
            intermediate_size=getattr(self.config, 'decoder_intermediate_size', 2048),
            max_position_embeddings=getattr(self.config, 'max_position_embeddings', 512),
            dropout=self.config.dropout,
            use_biomed_tokenizer=getattr(self.config, 'use_biomed_tokenizer', True),
        )
        captioning_decoder = captioning_decoder.to(self.config.device).float()

        # Create masked video modeling module
        masked_video_modeling: MaskedVideoModeling = ModelRegistry.get(
            name="masked_video_modeling"
        )(
            hidden_size=512,
            decoder_hidden_size=getattr(self.config, 'mvm_decoder_hidden_size', 256),
            decoder_layers=getattr(self.config, 'mvm_decoder_layers', 2),
            decoder_heads=getattr(self.config, 'mvm_decoder_heads', 8),
            mask_ratio=getattr(self.config, 'mask_ratio', 0.75),
            mask_token_learnable=getattr(self.config, 'mask_token_learnable', True),
            norm_predict_loss=getattr(self.config, 'norm_predict_loss', True),
        )
        masked_video_modeling = masked_video_modeling.to(self.config.device).float()

        # Apply DDP
        video_encoder = DistributedUtils.DDP(
            video_encoder, 
            device_ids=[self.config.device], 
        )
        text_encoder = DistributedUtils.DDP(
            text_encoder, 
            device_ids=[self.config.device], 
        )
        captioning_decoder = DistributedUtils.DDP(
            captioning_decoder, 
            device_ids=[self.config.device], 
        )
        masked_video_modeling = DistributedUtils.DDP(
            masked_video_modeling, 
            device_ids=[self.config.device], 
        )

        # Make temperature a trainable parameter
        log_temperature: nn.Parameter = nn.Parameter(
            torch.log(
                torch.tensor(
                    [self.config.temperature], 
                    dtype=torch.float32, 
                    device=self.config.device
                )
            )
        )

        # Different learning rates for different components
        param_groups = [
            {
                'params': video_encoder.module.model.parameters(),  # Main video backbone
                'lr': self.config.lr,
                'name': 'video_backbone',
                'weight_decay': self.config.video_weight_decay
            },
            {
                'params': video_encoder.module.aggregator.parameters(),  # Multihead attention aggregator
                'lr': self.config.lr * 2.0,  # Higher learning rate for aggregator
                'name': 'video_aggregator',
                'weight_decay': self.config.video_weight_decay
            },
            {
                'params': text_encoder.module.parameters(),  # Entire text encoder
                'lr': getattr(self.config, 'text_lr', 0.00002),  # Lower learning rate for text encoder
                'name': 'text_encoder',
                'weight_decay': self.config.text_weight_decay
            },
            {
                'params': captioning_decoder.module.parameters(),  # Captioning decoder
                'lr': getattr(self.config, 'captioning_lr', self.config.lr),
                'name': 'captioning_decoder',
                'weight_decay': getattr(self.config, 'captioning_weight_decay', 0.01)
            },
            {
                'params': masked_video_modeling.module.parameters(),  # Masked video modeling
                'lr': getattr(self.config, 'mvm_lr', self.config.lr * 0.1),  # Lower learning rate for MVM
                'name': 'masked_video_modeling',
                'weight_decay': getattr(self.config, 'mvm_weight_decay', 0.01)
            },
            {
                'params': [log_temperature],  # Temperature parameter
                'lr': self.config.lr,
                'name': 'temperature'
            }
        ]

        # Include the temperature parameter in the optimizer
        optimizer_class: torch.optim.Optimizer = getattr(torch.optim, self.config.optimizer)
        optimizer: torch.optim.Optimizer = optimizer_class(
            param_groups,
            lr=self.config.lr # act as a default learning rate for unset learning rates in param_groups
        )

        scheduler: LRScheduler = get_scheduler(
            scheduler_name=self.config.scheduler_name,
            optimizer=optimizer,
            num_epochs=self.config.epochs,
            train_dataloader=train_loader,
            factor=self.config.factor,
            step_size=self.config.lr_step_period,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_warmup_percent=self.config.num_warmup_percent,
            num_hard_restarts_cycles=self.config.num_hard_restarts_cycles,
            warm_restart_tmult=self.config.warm_restart_tmult,
        )

        scaler: GradScaler = GradScaler() if self.config.use_amp else None

        # Create multitask loss function
        loss_weights = getattr(self.config, 'loss_weights', {
            'contrastive': 1.0,
            'captioning': 1.0,
            'masked_modeling': 0.1,
            'distillation': 0.0,
        })
        
        loss_fn: Loss = Loss(
            loss_type=LossRegistry.get('multitask')(
                loss_weights=loss_weights,
                contrastive_loss_type=getattr(self.config, 'contrastive_loss_type', 'sigmoid'),
                captioning_loss_type=getattr(self.config, 'captioning_loss_type', 'cross_entropy'),
                masked_modeling_loss_type=getattr(self.config, 'masked_modeling_loss_type', 'mse'),
                temperature=self.config.temperature,
                label_smoothing=getattr(self.config, 'label_smoothing', 0.1),
                ignore_index=getattr(self.config, 'ignore_index', -100),
            )
        )

        if self.config.is_ref_device:
            if self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.config_update(
                    {
                        "train_dataset_size": len(train_loader),
                        "val_dataset_size": len(val_loader),
                        "loss_weights": loss_weights,
                    },
                )        
            print("\n=== Dataset Information ===")
            print(f"Training:   {len(train_loader):,} batches per GPU")
            print(f"Validation: {len(val_loader):,} batches per GPU")
            print(f"Total:      {(len(train_loader) + len(val_loader)):,} batches per GPU")
            print(f"\nBatch Size: {self.config.batch_size}")
            print(f"Training: {len(train_loader) * self.config.batch_size:,} videos per GPU")
            print(f"Validation: {len(val_loader) * self.config.batch_size:,} videos per GPU")
            print(f"Total: {(len(train_loader) + len(val_loader)) * self.config.batch_size:,} videos per GPU")
            print("===========================\n")

        return {
            "video_encoder": video_encoder,
            "text_encoder": text_encoder,
            "captioning_decoder": captioning_decoder,
            "masked_video_modeling": masked_video_modeling,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "scaler": scaler,
            "log_temp": log_temperature,
            "loss_fn": loss_fn,
            "output_dir": self.config.output_dir if self.config.is_ref_device else None,
        }    

    def _setup_inference_objects(
        self,
    ) -> dict[str, Any]:
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)
        
        val_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="inference", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            drop_last=False,
        )
        
        # Create models
        video_encoder: VideoEncoder = ModelRegistry.get(
            name="video_encoder"
        )(
            backbone=self.config.model_name,
            input_channels=3,
            num_frames=self.config.frames,
            pretrained=self.config.pretrained,
            output_dim=512,
            freeze_ratio=self.config.video_freeze_ratio,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            aggregator_depth=self.config.aggregator_depth,
        )        
        video_encoder = video_encoder.to(self.config.device).float()
        
        video_encoder = DistributedUtils.DDP(
            video_encoder, 
            device_ids=[self.config.device], 
        )
        
        checkpoint: dict[str, Any] = self._load_checkpoint(self.config.checkpoint)
        video_encoder.module.load_state_dict(checkpoint["video_encoder"], weight_only=True)
        log_temp: float = checkpoint["train/temperature"]

        return {
            "val_loader": val_loader,
            "video_encoder": video_encoder,
            "log_temp": log_temp,
            "output_dir": self.config.inference_results_path,
        }

    def _update_training_setup_with_checkpoint(
        self, 
        training_setup: dict[str, Any], 
        checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        print(f"Resuming from checkpoint: {checkpoint.keys()}")
        
        # Load model states
        if "video_encoder" in checkpoint:
            training_setup["video_encoder"].module.load_state_dict(checkpoint["video_encoder"])
        if "text_encoder" in checkpoint:
            training_setup["text_encoder"].module.load_state_dict(checkpoint["text_encoder"])
        if "captioning_decoder" in checkpoint:
            training_setup["captioning_decoder"].module.load_state_dict(checkpoint["captioning_decoder"])
        if "masked_video_modeling" in checkpoint:
            training_setup["masked_video_modeling"].module.load_state_dict(checkpoint["masked_video_modeling"])
        
        # Load optimizer and scheduler
        if "optimizer" in checkpoint:
            training_setup["optimizer"].load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            training_setup["lr_scheduler"].load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            training_setup["scaler"].load_state_dict(checkpoint["scaler"])
        if "train/temperature" in checkpoint:
            training_setup["log_temp"].data.copy_(checkpoint["train/temperature"])
        
        return training_setup
        
    def run(self):
        self._setup_project()
        
        if self.config.run_mode == RunMode.TRAIN:
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
            end_epoch = start_epoch + self.config.epochs
            runner.train(start_epoch=start_epoch, end_epoch=end_epoch)
        elif self.config.run_mode == RunMode.INFERENCE:
            inference_setup: dict[str, Any] = self._setup_inference_objects()
            runner: Runner = Runner(
                runner_type=RunnerRegistry.get(
                    name=self.config.pipeline_project
                )(
                    config=self.config,
                    wandb_wrapper=self.wandb_wrapper,
                    **inference_setup,
                )
            )
            runner.inference()
        else:
            raise ValueError(
                f"Invalid run mode: {self.config.run_mode}, must be one of {RunMode.TRAIN} or {RunMode.INFERENCE}"
            )