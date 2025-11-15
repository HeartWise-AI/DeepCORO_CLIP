
from typing import Any

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from models.text_encoder import TextEncoder
from models.video_encoder import VideoEncoder
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

@ProjectRegistry.register('DeepCORO_clip')
class ContrastivePretrainingProject(BaseProject):
    def __init__(
        self, 
        config: ClipConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)
        
    def _setup_training_objects(
        self,
    )->dict:
        """
        Load training objects.

        Args:
            config (HeartWiseConfig): Configuration object

        Returns:
            dict: Dictionary containing training objects
        """
        cached_mean = getattr(self.config, "data_mean", None)
        cached_std = getattr(self.config, "data_std", None)
        if cached_mean is not None and cached_std is not None:
            mean = torch.tensor(cached_mean, dtype=torch.float32)
            std = torch.tensor(cached_std, dtype=torch.float32)
            if self.config.is_ref_device:
                print("\n=== Using cached dataset statistics ===")
                print(f"Mean: {mean.tolist()}")
                print(f"Std:  {std.tolist()}")
                print("===========================\n")
        else:
            mean, std = calculate_dataset_statistics_ddp(self.config)

        is_distributed = DistributedUtils.is_initialized()
        world_size = DistributedUtils.get_world_size() if is_distributed else 1
        global_rank = DistributedUtils.get_rank() if is_distributed else 0
        force_cpu = isinstance(self.config.device, str) and self.config.device == "cpu"
        use_cuda = torch.cuda.is_available() and not force_cpu
        default_local_rank = int(self.config.device) if isinstance(self.config.device, int) else 0
        if use_cuda:
            local_rank = DistributedUtils.get_local_rank() if is_distributed else default_local_rank
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            local_rank = 0
            device = torch.device("cpu")

        train_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="train", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=True,
            num_replicas=world_size,
            rank=global_rank,
            drop_last=True,
        )
        val_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="val", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=world_size,
            rank=global_rank,
            drop_last=True,
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
        video_encoder = video_encoder.to(device).float()

        text_encoder: TextEncoder = ModelRegistry.get(
            name="text_encoder"
        )(
            freeze_ratio=self.config.text_freeze_ratio,
            dropout=self.config.dropout,
        )
        text_encoder = text_encoder.to(device).float()

        # Make temperature a trainable parameter directly on the device
        log_temperature: nn.Parameter = nn.Parameter(
            torch.log(
                torch.tensor(
                    [self.config.temperature], 
                    dtype=torch.float32, 
                    device=device,
                )
            )
        )
        video_encoder.register_parameter("log_temperature", log_temperature)

        self._synchronize_trainable_params(video_encoder, module_name="video_encoder")
        self._synchronize_trainable_params(text_encoder, module_name="text_encoder")

        if DistributedUtils.is_initialized():
            vid_params = sum(p.numel() for p in video_encoder.parameters())
            txt_params = sum(p.numel() for p in text_encoder.parameters())
            print(f"[Rank {global_rank}] video params={vid_params}, text params={txt_params}")

        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        video_encoder = DistributedUtils.DDP(
            video_encoder,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        text_encoder = DistributedUtils.DDP(
            text_encoder,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        log_temperature = video_encoder.module.log_temperature

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
                'lr': 0.00002,  # Lower learning rate for text encoder
                'name': 'text_encoder',
                'weight_decay': self.config.text_weight_decay
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

        # Create loss function
        loss_fn: Loss = Loss(
            loss_type=LossRegistry.get(self.config.loss_name)()
        )

        if self.config.is_ref_device:
            if self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.config_update(
                    {
                        "train_dataset_size": len(train_loader),
                        "val_dataset_size": len(val_loader),
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
    )->dict[str, Any]:
        # Calculate dataset statistics
        mean, std = calculate_dataset_statistics_ddp(self.config)

        is_distributed = DistributedUtils.is_initialized()
        world_size = DistributedUtils.get_world_size() if is_distributed else 1
        global_rank = DistributedUtils.get_rank() if is_distributed else 0
        force_cpu = isinstance(self.config.device, str) and self.config.device == "cpu"
        use_cuda = torch.cuda.is_available() and not force_cpu
        default_local_rank = int(self.config.device) if isinstance(self.config.device, int) else 0
        if use_cuda:
            local_rank = DistributedUtils.get_local_rank() if is_distributed else default_local_rank
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            local_rank = 0
            device = torch.device("cpu")
        
        val_loader: DataLoader = get_distributed_video_clip_dataloader(
            self.config, 
            split="inference", 
            mean=mean.tolist(),
            std=std.tolist(),
            shuffle=False,
            num_replicas=world_size,
            rank=global_rank,
            drop_last=True,
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
        video_encoder = video_encoder.to(device).float()
        log_temperature: nn.Parameter = nn.Parameter(
            torch.log(
                torch.tensor(
                    [self.config.temperature],
                    dtype=torch.float32,
                    device=device,
                )
            )
        )
        video_encoder.register_parameter("log_temperature", log_temperature)
        self._synchronize_trainable_params(video_encoder, module_name="video_encoder")
        
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        video_encoder = DistributedUtils.DDP(
            video_encoder,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        log_temperature = video_encoder.module.log_temperature
        
        checkpoint: dict[str, Any] = self._load_checkpoint(self.config.checkpoint)
        video_state = checkpoint["video_encoder"]
        has_log_temp = "log_temperature" in video_state
        video_encoder.module.load_state_dict(video_state, strict=has_log_temp)
        if not has_log_temp and "train/temperature" in checkpoint:
            temp_tensor = torch.tensor(
                checkpoint["train/temperature"],
                dtype=log_temperature.dtype,
                device=log_temperature.device,
            )
            log_temperature.data.copy_(temp_tensor.log())
        log_temp: torch.Tensor = video_encoder.module.log_temperature

        return {
            "val_loader": val_loader,
            "video_encoder": video_encoder,
            "log_temp": log_temp,
            "output_dir": self.config.inference_results_path,
        }
    def _synchronize_trainable_params(self, module: nn.Module, module_name: str) -> None:
        """
        Ensure every rank exposes the same set of trainable parameters before wrapping
        the module with DistributedDataParallel. This avoids mismatches when partial
        freezing logic toggles different subsets across ranks (e.g., due to sweep
        overrides or floating-point rounding).
        """
        if not DistributedUtils.is_initialized():
            return

        params = list(module.parameters())
        if not params:
            return

        # Build a binary mask indicating which parameters require gradients.
        if torch.cuda.is_available():
            local_rank = DistributedUtils.get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        mask = torch.tensor(
            [1 if p.requires_grad else 0 for p in params],
            dtype=torch.int32,
            device=device,
        )

        # All-reduce with MAX so a parameter stays trainable if it is enabled on any rank.
        DistributedUtils.dist.all_reduce(mask, op=DistributedUtils.dist.ReduceOp.MAX)
        mask_list = mask.tolist()

        changed = False
        for desired, param in zip(mask_list, params):
            desired_bool = bool(desired)
            if param.requires_grad != desired_bool:
                param.requires_grad = desired_bool
                changed = True

        if changed and self.config.is_ref_device:
            print(
                f"[DDP Sync] Normalized trainable mask for {module_name} "
                "across ranks to keep DDP initialization consistent."
            )

    def _update_training_setup_with_checkpoint(
        self, 
        training_setup: dict[str, Any], 
        checkpoint: dict[str, Any]
    )->dict[str, Any]:
        print(f"Resuming from checkpoint: {checkpoint.keys()}")
        video_state = checkpoint["video_encoder"]
        has_log_temp = "log_temperature" in video_state
        training_setup["video_encoder"].module.load_state_dict(video_state, strict=has_log_temp)
        training_setup["text_encoder"].module.load_state_dict(checkpoint["text_encoder"])
        training_setup["optimizer"].load_state_dict(checkpoint["optimizer"])
        training_setup["lr_scheduler"].load_state_dict(checkpoint["scheduler"])
        training_setup["scaler"].load_state_dict(checkpoint["scaler"])
        if not has_log_temp and "train/temperature" in checkpoint:
            temp_tensor = torch.tensor(
                checkpoint["train/temperature"],
                dtype=training_setup["log_temp"].dtype,
                device=training_setup["log_temp"].device,
            )
            training_setup["log_temp"].data.copy_(temp_tensor.log())
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
