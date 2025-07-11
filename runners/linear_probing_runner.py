import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict,  Callable

from tqdm import tqdm
from torch.optim import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from collections import defaultdict

from utils.metrics import (
    compute_regression_metrics,
    compute_binary_classification_metrics, 
    compute_multiclass_classification_metrics
)
from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.enums import RunMode, MetricTask
from utils.wandb_wrapper import WandbWrapper
from utils.config.linear_probing_config import LinearProbingConfig

# Type aliases Definitions
ProcessedBatch = Dict[str, torch.Tensor]
StepFnResults = Dict[str, Dict[str, torch.Tensor]]

@dataclass
class BatchResult:
    outputs: StepFnResults
    processed_batch: ProcessedBatch
    

@RunnerRegistry.register("DeepCORO_video_linear_probing")
@RunnerRegistry.register("DeepCORO_video_linear_probing_test")
@RunnerRegistry.register("DeepCORO_video_linear_probing_cardio_syntax")
class LinearProbingRunner:
    """
    This class runs a linear probing pipeline using a VideoEncoder and TextEncoder.
    It handles both training and validation loops in a distributed data-parallel setting.
    """
    def __init__(
        self,
        loss_fn: Loss,
        output_dir: str,
        val_loader: DataLoader,
        config: LinearProbingConfig,
        wandb_wrapper: WandbWrapper,
        linear_probing,
        scaler: Optional[GradScaler] = None,
        optimizer: Optional[Optimizer] = None,
        train_loader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None
    ):
        """
        Initialize the runner with provided configurations, data loaders, and modules.

        :param config: LinearProbingConfig object with run/training configuration.
        :param wandb_wrapper: WandbWrapper instance.
        :param train_loader: DataLoader for training dataset.
        :param val_loader: DataLoader for validation dataset.
        :param linear_probing: LinearProbing model.
        :param optimizer: optimizer instance.
        :param scaler: GradScaler for automatic mixed precision.
        :param scheduler: Learning rate scheduler.
        :param loss_fn: Contrastive loss function callable.
        :param output_dir: Directory where checkpoints and outputs will be saved.
        """
        self.config: LinearProbingConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        self.world_size: int = config.world_size
        self.device: int = config.device
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.linear_probing = linear_probing  # VideoMILWrapper
        self.optimizer: Optimizer = optimizer
        self.scaler: GradScaler = scaler
        self.lr_scheduler: LRScheduler = lr_scheduler
        self.loss_fn: Loss = loss_fn
        self.output_dir: str = output_dir
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        # For simplicity: check the config for a known scheduler_name
        # If it includes "_warmup" or is from HF, we treat it as per-iteration
        self.scheduler_per_iteration = self._scheduler_is_per_iteration()
        self.step = 0

    def _scheduler_is_per_iteration(self) -> bool:
        """
        Returns True if the chosen scheduler is a Hugging Face style that
        expects a call to .step() per iteration (batch). Otherwise, False.
        """
        # We do a simpler check to change scheduler update to per epoch or per batch:
        # Example keywords: "linear_warmup", "cosine_with_warmup", 
        # "cosine_with_hard_restarts_with_warmup", etc.
        sched_name = getattr(self.config, "scheduler_name", "").lower()
        # If it matches typical HF warmup schedulers, return True
        HF_KEYS = ["warmup", "with_warmup"]
        return any(k in sched_name for k in HF_KEYS)

    def train(
        self, 
        start_epoch: int, 
        end_epoch: int
    ) -> None:
        for epoch in range(start_epoch, end_epoch):
            # Synchronize before epoch starts
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )            
            
            # Training phase
            train_metrics = self._run_epoch(
                mode=RunMode.TRAIN, 
                epoch=epoch
            )   
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                # Let wandb auto-increment steps
                self.wandb_wrapper.log(train_metrics)
                print(f"[DEBUG] rank={self.device} => Logged train metrics to W&B")
            
            # Sync before validation
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )
            
            # Validation phase
            val_metrics = self._run_epoch(
                mode=RunMode.VALIDATE, 
                epoch=epoch
            )
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )
            
            # Update best model
            if val_metrics["val/main_loss"] < self.best_val_loss:
                prev_best = self.best_val_loss
                self.best_val_loss = val_metrics["val/main_loss"]
                self.best_epoch = epoch
                if self.config.is_ref_device:
                    print(
                        f"\nNew best model! Val Loss: {val_metrics['val/main_loss']:.4f} "
                        f"(previous: {prev_best:.4f})"
                    )
                    self._save_checkpoint(
                        epoch=epoch,
                        metrics={
                            **train_metrics, 
                            **val_metrics
                        },
                        is_best=True,
                    )

            # Always save "latest" checkpoint
            if self.config.is_ref_device:
                self._save_checkpoint(
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_best=False,
                )
            
            # Sync after validation
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )
            
            if self.lr_scheduler and (not self.scheduler_per_iteration):
                self.lr_scheduler.step()
                
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                val_metrics['best_val_loss'] = self.best_val_loss
                print(f"[DEBUG] rank={self.device} => Val metrics: {val_metrics}")
                self.wandb_wrapper.log(val_metrics)
                print(f"[DEBUG] rank={self.device} => Logged val metrics to W&B")  
                
            # Sync after logging
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )                                 

            # ------------------------------------------------------------------
            # Memory cleanup to avoid GPU OOM across epochs
            # ------------------------------------------------------------------
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def _run_epoch(
        self, 
        mode: str, 
        epoch: int
    ) -> dict[str, float]:
        assert mode in [RunMode.TRAIN, RunMode.VALIDATE]
        
        self.linear_probing.train(mode == RunMode.TRAIN)

        epoch_metrics: dict[str, float] = {}

        dataloader: DataLoader = self.train_loader if mode == RunMode.TRAIN else self.val_loader
        step_fn: Callable[..., StepFnResults] = self._train_step if mode == RunMode.TRAIN else self._val_step

        tqdm_desc: str = f"[GPU {self.device}] Running {mode} Epoch {epoch + 1}"
        if self.config.is_ref_device:
            data_iter: tqdm = tqdm(dataloader, desc=tqdm_desc)
        else:
            data_iter: DataLoader = dataloader
        
        gathered_metrics: Dict[str, float] = {}
        accumulated_preds: Dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_targets: Dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_names: list[str] = []
        
        for batch_idx, batch in enumerate(data_iter, start=1):
            try:
                run_batch: BatchResult = self._run_batch(
                    batch=batch,
                    step_fn=step_fn
                )

                # Accumulate batch metrics for each head
                self._accumulate_batch_metrics(
                    run_batch=run_batch,
                    accumulated_preds=accumulated_preds,
                    accumulated_targets=accumulated_targets,
                    accumulated_names=accumulated_names,
                    gathered_metrics=gathered_metrics,
                    batch=batch
                )
                
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in step_fn: {e}")
            
            # Note: lr_scheduler.step() for per_iteration schedulers is handled in _train_step()
            # after optimizer.step() to maintain the correct PyTorch 1.1.0+ order

            # Update progress bar with gathered losses
            if self.config.is_ref_device:
                data_iter.set_postfix({
                    f"{k}": f'{(v / batch_idx):.4f}' if 'mean' in k else f'{v:.4f}' 
                    for k, v in gathered_metrics.items()
                })
            
            # Sync after each batch
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )

        # Gather all predictions and targets across GPUs if using distributed training
        if self.config.world_size > 1:
            self._gather_distributed_predictions(
                accumulated_preds=accumulated_preds,
                accumulated_targets=accumulated_targets,
                accumulated_names=accumulated_names
            )

        # Sync after gathering predictions and targets
        DistributedUtils.sync_process_group(
            world_size=self.world_size,
            device_ids=self.device
        )

        # Gather losses across GPUs if using distributed training
        if self.config.world_size > 1:
            # Convert accumulated losses to the format expected by gather_loss
            for k, v in gathered_metrics.items():
                if 'mean_' in k:
                    # Convert accumulated loss sum to a list and gather across GPUs
                    gathered_loss_sum = DistributedUtils.gather_loss([v], self.config.device)
                    # Compute average using total batches across all GPUs
                    epoch_metrics[f"{mode}/{k.replace('mean_', '')}_loss"] = gathered_loss_sum / batch_idx
        else:
            # Single GPU - compute averages normally
            for k, v in gathered_metrics.items():
                if 'mean' in k:
                    epoch_metrics[f"{mode}/{k.replace('mean_', '')}_loss"] = v / batch_idx
        
        # Compute AUC metrics for each head
        if self.config.is_ref_device:
            self._compute_heads_metrics(
                mode=mode,
                accumulated_preds=accumulated_preds,
                accumulated_targets=accumulated_targets,
                validation_metrics=epoch_metrics,
                compute_ci=False
            )
            
        # Sync after computing heads metrics
        DistributedUtils.sync_process_group(
            world_size=self.world_size,
            device_ids=self.device
        )

        if self.config.is_ref_device:
            print(f"[DEBUG] rank={self.device} => {mode} epoch metrics: {epoch_metrics}")

        # Save predictions for validation mode
        if mode == RunMode.VALIDATE and self.config.is_ref_device:
            print(f"[DEBUG] rank={self.device} => Saving predictions for validation mode")
            self._save_predictions(
                mode=mode,
                epoch=epoch,
                accumulated_names=accumulated_names,
                accumulated_preds=accumulated_preds,
                accumulated_targets=accumulated_targets,
            )

        # ------------------------------------------------------------------
        # Memory cleanup: delete large local variables & free GPU cache
        # ------------------------------------------------------------------
        import gc

        # Explicitly delete variables that exist in scope
        variables_to_delete = [
            'accumulated_preds',
            'accumulated_targets',
            'preds',
            'targets',
            'outputs',
            'losses',
        ]

        for var_name in variables_to_delete:
            if var_name in locals():
                del locals()[var_name]
        
        torch.cuda.empty_cache()
        gc.collect()

        return epoch_metrics

    def _preprocess_inputs(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Preprocess batch inputs for the model.
        
        Args:
            batch: Dictionary containing:
                videos: Tensor of shape [B * num_videos, C, F, H, W] for multi-video
                       or [B, C, F, H, W] for single-video
                targets: Dict of tensors [B, ...]
                video_indices: Optional tensor [B * num_videos] mapping videos to batch indices
                video_fname: List of file paths
                
        Returns:
            Dictionary containing processed tensors ready for model input
        """
        # Move videos to device
        batch_video = batch['videos'].to(self.config.device)
        
        # Move targets to device
        batch_targets = batch['targets']
        for k, v in batch_targets.items():
            batch_targets[k] = v.to(self.config.device)
            
        return {
            "batch_video": batch_video,
            "batch_targets": batch_targets,
        }
    
    def _run_batch(
        self,
        batch: Dict[str, torch.Tensor],
        step_fn: Callable[..., StepFnResults],
    ) -> BatchResult:
        processed_batch: ProcessedBatch = self._preprocess_inputs(batch)
        outputs: StepFnResults = step_fn(**processed_batch)
        
        return BatchResult(
            outputs=outputs,
            processed_batch=processed_batch
        )
    
    def _train_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: Dict[str, torch.Tensor]
    ) -> StepFnResults:
        # Clear gradients only if this is the first step in accumulation
        if self.step % self.config.gradient_accumulation_steps == 0:
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                
        # Forward pass with autocast for mixed precision
        with torch.amp.autocast('cuda', enabled=self.config.use_amp, dtype=torch.float16):
            try:
                if self.config.use_amp:
                    batch_video = batch_video.to(dtype=torch.float16)
                     
                outputs_dict: dict[str, torch.Tensor] = self.linear_probing(
                    batch_video
                )
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in linear_probing: {e} for batch with video shape {batch_video.shape}")

        try:
            if self.config.use_amp:
                for head_name, target in batch_targets.items():
                    if self.config.head_task[head_name] == MetricTask.REGRESSION:
                        batch_targets[head_name] = target.to(dtype=torch.float16)
                    # For classification tasks, keep targets as integers (long)
                    # No conversion needed as they should remain as long for loss functions
            
            losses: dict[str, torch.Tensor] = self.loss_fn.run(
                outputs=outputs_dict, 
                targets=batch_targets
            )
        except Exception as e:
            raise Exception(f"[DEBUG] rank={self.device} => Error in loss_fn: {e} for batch with outputs_dict {outputs_dict} and batch_targets {batch_targets}")

        # Scale loss by gradient accumulation steps
        scaled_loss = losses['main'] / self.config.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Only step optimizer and update scaler if this is the last step in accumulation
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler and self.optimizer is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            # Step the learning rate scheduler after optimizer step
            if self.lr_scheduler and self.scheduler_per_iteration:
                self.lr_scheduler.step()

        # Increment step counter
        self.step += 1

        # Get learning rate metrics
        lr_metrics = {}
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                lr_metrics[f"lr/{pg['name']}"] = pg["lr"]

        # Convert losses to scalar values
        scalar_losses = {k: v.item() for k, v in losses.items()}
        scalar_outputs = {k: v.detach() for k, v in outputs_dict.items()}

        return {
            "losses": scalar_losses,
            "logits": scalar_outputs,
            **lr_metrics
        }
        
    def _val_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: Dict[str, torch.Tensor],
    ) -> StepFnResults:                
        # Forward pass with autocast for mixed precision
        with torch.no_grad():
            try:
                # Convert inputs to float16 if AMP is enabled
                if self.config.use_amp:
                    batch_video = batch_video.to(dtype=torch.float16)
                
                outputs_dict: dict[str, torch.Tensor] = self.linear_probing(
                    batch_video
                )
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in linear_probing: {e} for batch with video shape {batch_video.shape}")

        try:
            # Only convert regression targets to float16 if AMP is enabled, keep classification targets as integers
            if self.config.use_amp:
                for head_name, target in batch_targets.items():
                    if self.config.head_task[head_name] == MetricTask.REGRESSION:
                        batch_targets[head_name] = target.to(dtype=torch.float16)
                    # For classification tasks, keep targets as integers (long)
                    # No conversion needed as they should remain as long for loss functions
            
            losses: dict[str, torch.Tensor] = self.loss_fn.run(
                outputs=outputs_dict, 
                targets=batch_targets
            )
        except Exception as e:
            raise Exception(f"[DEBUG] rank={self.device} => Error in loss_fn: {e} for batch with outputs shape {outputs_dict} and targets shape {batch_targets}")
                
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )

        # Get learning rate metrics
        lr_metrics = {}
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                lr_metrics[f"lr/{pg['name']}"] = pg["lr"]

        # Convert losses to scalar values
        scalar_losses = {k: v.item() for k, v in losses.items()}
        scalar_outputs = {k: v.detach() for k, v in outputs_dict.items()}
                
        return {
            "losses": scalar_losses,
            "logits": scalar_outputs,
            **lr_metrics
        }

    def _inference_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: Dict[str, torch.Tensor] # Unused - parsed to match signature of _val_step and _train_step
    ) -> StepFnResults:        
        # Forward pass with autocast for mixed precision
        with torch.no_grad():
            try:
                # Convert inputs to float16 if AMP is enabled
                if self.config.use_amp:
                    batch_video = batch_video.to(dtype=torch.float16)
                
                outputs_dict: dict[str, torch.Tensor] = self.linear_probing(
                    batch_video
                )
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in linear_probing: {e} for batch with video shape {batch_video.shape}")
  
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )

        # Convert losses to scalar values
        scalar_outputs = {k: v.detach() for k, v in outputs_dict.items()}
                
        return {
            "logits": scalar_outputs
        }

    def validate(self) -> None:
        """
        Run validation on the validation dataset and return metrics with confidence intervals.
        
        Returns:
            Dictionary containing computed metrics for all heads with CIs
        """
        # Set model to evaluation mode
        assert len(self.val_loader) > 0, "Validation loader is empty"
        
        self.linear_probing.train(False)
        
        # Synchronize before inference starts
        DistributedUtils.sync_process_group(
            world_size=self.world_size,
            device_ids=self.device
        )
        
        # Initialize metrics and accumulation containers
        validation_metrics: Dict[str, float] = {}
        accumulated_preds: Dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_targets: Dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_names: list[str] = []
        gathered_metrics: Dict[str, float] = {}
        
        # Set up progress bar
        tqdm_desc: str = f"[GPU {self.device}] Running Validation"
        if self.config.is_ref_device:
            data_iter: tqdm = tqdm(self.val_loader, desc=tqdm_desc, total=len(self.val_loader))
        else:
            data_iter = self.val_loader
        
        step_fn: Callable[..., StepFnResults] = self._val_step
        
        # Process all batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_iter, start=1):
                try:
                    # Forward pass
                    run_batch: BatchResult = self._run_batch(
                        batch=batch,
                        step_fn=step_fn
                    )
                    
                    # Accumulate batch metrics for each head
                    self._accumulate_batch_metrics(
                        run_batch=run_batch,
                        accumulated_preds=accumulated_preds,
                        accumulated_targets=accumulated_targets,
                        accumulated_names=accumulated_names,
                        gathered_metrics=gathered_metrics,
                        batch=batch
                    )
                    
                except Exception as e:
                    raise Exception(f"[DEBUG] rank={self.device} => Error in validation step: {e}")
                
                # Update progress bar with gathered losses
                if self.config.is_ref_device:
                    data_iter.set_postfix({
                        f"{k}": f'{(v / (batch_idx)):.4f}' if 'mean' in k else f'{v:.4f}' 
                        for k, v in gathered_metrics.items()
                    })
                
                # Sync after each batch
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )
        
        # Gather all predictions and targets across GPUs if using distributed training
        if self.config.world_size > 1:
            self._gather_distributed_predictions(
                accumulated_preds=accumulated_preds,
                accumulated_targets=accumulated_targets,
                accumulated_names=accumulated_names
            )
            
        # Compute metrics for each head WITH CONFIDENCE INTERVALS
        if self.config.is_ref_device:
            print(f"[DEBUG] rank={self.device} => Computing metrics with CI - This might take a while...")
            self._compute_heads_metrics(
                mode=self.config.run_mode,
                accumulated_preds=accumulated_preds,
                accumulated_targets=accumulated_targets,
                validation_metrics=validation_metrics,
                compute_ci=True
            )
        
        # Save predictions if on reference device
        if self.config.is_ref_device:
            self._save_predictions(
                mode=self.config.run_mode,
                accumulated_names=accumulated_names,
                accumulated_preds=accumulated_preds,
                accumulated_targets=accumulated_targets,
                epoch=-1  # Use -1 to indicate inference mode
            )
        
        # Log metrics to wandb if available
        if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            self.wandb_wrapper.log(validation_metrics)
            print(f"[DEBUG] rank={self.device} => Logged validation metrics to W&B")
        
        print(f"[DEBUG] rank={self.device} => Validation metrics: {validation_metrics}")
        
        # Final sync
        DistributedUtils.sync_process_group(
            world_size=self.world_size,
            device_ids=self.device
        )
        
        # Save validation metrics with CI to JSON file
        if self.config.is_ref_device:                
            self._save_validation_metrics_to_json(validation_metrics)
        
        # Memory cleanup for GPU tensors
        torch.cuda.empty_cache()
        
    def test(self) -> None:
        """
        Run test on the test dataset and return metrics with confidence intervals.
        """
        self.validate()
    
    def inference(self) -> None:
        """
        Run inference on the test dataset and return metrics.
        """
        assert len(self.val_loader) > 0, "Test loader is empty"
        
        self.linear_probing.train(False)
        
        # Synchronize before inference starts
        DistributedUtils.sync_process_group(
            world_size=self.world_size,
            device_ids=self.device
        )
        
        # Initialize metrics and accumulation containers
        accumulated_preds: Dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_names: list[str] = []
        
        # Set up progress bar
        tqdm_desc: str = f"[GPU {self.device}] Running Inference"
        if self.config.is_ref_device:
            data_iter: tqdm = tqdm(self.val_loader, desc=tqdm_desc, total=len(self.val_loader))
        else:
            data_iter = self.val_loader
        
        step_fn: Callable[..., StepFnResults] = self._inference_step
        
        # Process all batches
        with torch.no_grad():
            for _, batch in enumerate(data_iter, start=1):
                    # Forward pass
                    run_batch: BatchResult = self._run_batch(
                        batch=batch,
                        step_fn=step_fn
                    )
                    
                    # Accumulate batch metrics for each head
                    self._accumulate_batch_metrics_inference(
                        run_batch=run_batch,
                        accumulated_preds=accumulated_preds,
                        accumulated_names=accumulated_names,
                        batch=batch
                    )

        # Sync after inference
        DistributedUtils.sync_process_group(
            world_size=self.world_size,
            device_ids=self.device
        )

        # Gather all predictions across GPUs if using distributed training
        if self.config.world_size > 1:
            self._gather_distributed_predictions(
                accumulated_preds=accumulated_preds,
                accumulated_names=accumulated_names,
                accumulated_targets=None
            )
            
        # Save predictions if on reference device
        if self.config.is_ref_device:
            self._save_predictions(
                mode=self.config.run_mode,
                epoch=-1,
                accumulated_names=accumulated_names,
                accumulated_preds=accumulated_preds,
                accumulated_targets=None,
            )

    def _save_checkpoint(
        self, 
        epoch: int, 
        metrics: dict[str, float], 
        is_best: bool = False
    ) -> None:
        """
        Saves model checkpoint (including model weights, optimizer, scheduler, and metrics).

        :param epoch: Current epoch index.
        :param metrics: Dictionary of metrics to be saved.
        :param is_best: If True, saves as 'best_epoch.pt'. Otherwise, saves as 'checkpoint.pt'.
        """
        checkpoint_dir = os.path.join(self.output_dir, "models")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "linear_probing": self.linear_probing.module.state_dict()
            if hasattr(self.linear_probing, "module")
            else self.linear_probing.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "epoch": epoch,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        
        # Save regular checkpoint for current epoch
        checkpoint_path: str = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Delete the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_checkpoint_path: str = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch - 1}.pt')
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
                print(f"Deleted old checkpoint: {prev_checkpoint_path}")
        

        # If this is the best model, save it separately
        if is_best:
            # Delete previous best model if it exists
            for file in os.listdir(checkpoint_dir):
                if file.startswith('best_model_epoch_') and file.endswith('.pt'):
                    old_best_path = os.path.join(checkpoint_dir, file)
                    if os.path.exists(old_best_path):
                        os.remove(old_best_path)
                        print(f"Deleted previous best model: {old_best_path}")
            
            best_model_path: str = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch}.pt')
            torch.save(checkpoint, best_model_path)
            
        if self.wandb_wrapper.is_initialized():
            self.wandb_wrapper.log({
                "checkpoint/epoch": epoch,
                "checkpoint/loss": metrics["val/main_loss"],
            })

    def _save_predictions(
        self,
        mode: str,
        epoch: int,
        accumulated_names: list[str],
        accumulated_preds: dict[str, torch.Tensor],
        accumulated_targets: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Save model predictions and ground truth values to a CSV file.
        Saves for every epoch and keeps best epoch predictions.
        
        Args:
            mode: The current run mode (train/validation)
            accumulated_names: List of video filenames
            accumulated_targets: Dictionary of ground truth values for each head
            epoch: Current epoch number
        """
        try:
            # Create predictions dictionary with epoch column
            predictions_dict: dict[str, list] = {
                'epoch': [epoch] * len(accumulated_names),
                'video_name': accumulated_names
            }
            
            # Add predictions and ground truth for each head
            for head in accumulated_preds.keys():
                # Handle both tensor and list of tensors format
                if isinstance(accumulated_preds[head], list):
                    preds_tensor: torch.Tensor = accumulated_preds[head][0]
                    targets_tensor: torch.Tensor = accumulated_targets[head][0] if accumulated_targets is not None else None
                else:
                    preds_tensor: torch.Tensor = accumulated_preds[head]
                    targets_tensor: torch.Tensor = accumulated_targets[head] if accumulated_targets is not None else None
                                                    
                # Handle binary classification
                if self.config.head_task[head] == MetricTask.BINARY_CLASSIFICATION:
                    preds: np.ndarray = preds_tensor.squeeze().detach().cpu().float().numpy()
                    targets: np.ndarray = targets_tensor.squeeze().detach().cpu().int().numpy() if targets_tensor is not None else None
                    predictions_dict[f'{head}_pred'] = preds
                    predictions_dict[f'{head}_true'] = targets
                    
                # Handle regression
                elif self.config.head_task[head] == MetricTask.REGRESSION:
                    preds: np.ndarray = preds_tensor.squeeze().detach().cpu().float().numpy()
                    targets: np.ndarray = targets_tensor.squeeze().detach().cpu().float().numpy() if targets_tensor is not None else None
                    predictions_dict[f'{head}_pred'] = preds
                    predictions_dict[f'{head}_true'] = targets
                    
                # Handle multi-class classification
                elif self.config.head_task[head] == MetricTask.MULTICLASS_CLASSIFICATION:                        
                    # For multi-class, get both raw probabilities and predicted class
                    pred_labels: np.ndarray = preds_tensor.squeeze().detach().cpu().float().numpy()
                    target_labels: np.ndarray = targets_tensor.squeeze().detach().cpu().int().numpy() if targets_tensor is not None else None
                    
                    # Create index_to_label mapping
                    index_to_label = {v: k for k, v in self.config.labels_map[head].items()}
                    
                    # Store probabilities for each class
                    for class_idx in range(pred_labels.shape[1]):
                        if hasattr(self.config, 'labels_map') and head in self.config.labels_map:
                            label = f'{head}_prob_{index_to_label[class_idx]}'
                        else:
                            label = f'{head}_prob_{class_idx}'
                        
                        # Store probabilities for each class
                        predictions_dict[label] = pred_labels[:, class_idx]    
                        
                    # Store predicted class
                    predictions_dict[f'{head}_pred_class'] = pred_labels.argmax(axis=1)
                    # Store true class
                    predictions_dict[f'{head}_true_class'] = target_labels if target_labels is not None else None
                    
            # Create and save DataFrame
            df: pd.DataFrame = pd.DataFrame(predictions_dict)
            predictions_dir: str = os.path.join(self.output_dir, "predictions") 
            os.makedirs(predictions_dir, exist_ok=True)
            
            # Save current epoch predictions
            current_epoch_file: str = os.path.join(
                predictions_dir, 
                f'{mode}_predictions_epoch_{epoch}.csv'
            )
            df.to_csv(current_epoch_file, index=False)
            print(f"[DEBUG] rank={self.device} => Saved epoch {epoch} predictions to {current_epoch_file}")
            
            # If this is the best epoch, also save as best predictions
            if epoch == self.best_epoch:
                best_file: str = os.path.join(
                    predictions_dir, 
                    f'{mode}_predictions_best_epoch_{epoch}.csv'
                )
                df.to_csv(best_file, index=False)
                print(f"[DEBUG] rank={self.device} => Saved best epoch predictions to {best_file}")
            
        except Exception as e:
            print(f"[DEBUG] rank={self.device} => Error saving predictions to CSV: {e}")
            print(f"[DEBUG] rank={self.device} => Accumulated names length: {len(accumulated_names)}")
            for head in accumulated_preds.keys():
                print(f"[DEBUG] rank={self.device} => Head {head} - Final shapes:")
                if isinstance(accumulated_preds[head], list):
                    print(f"[DEBUG] rank={self.device} =>   Predictions: {accumulated_preds[head][0].shape}")
                    print(f"[DEBUG] rank={self.device} =>   Targets: {accumulated_targets[head][0].shape}")
                else:
                    print(f"[DEBUG] rank={self.device} =>   Predictions: {accumulated_preds[head].shape}")
                    print(f"[DEBUG] rank={self.device} =>   Targets: {accumulated_targets[head].shape}")
            if 'predictions_dict' in locals():
                print(f"[DEBUG] rank={self.device} => Predictions dictionary keys and lengths:")
                for k, v in predictions_dict.items():
                    print(f"[DEBUG] rank={self.device} =>   {k}: {len(v)}")

    def _save_validation_metrics_to_json(self, validation_metrics: dict[str, float]) -> None:
        """
        Save validation metrics with confidence intervals to a JSON file.
        
        Args:
            validation_metrics: Dictionary containing all validation metrics including CIs
        """
        def convert_to_serializable(obj):
            """Convert numpy/torch types to Python native types for JSON serialization."""
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif torch.is_tensor(obj):
                return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        def organize_metrics_by_head(metrics: dict) -> dict:
            """Organize flat metrics dictionary by head."""
            organized = {}
            
            for head in self.config.head_structure.keys():
                metric_values = {}
                for metric_name, metric_value in metrics.items():  
                    # More precise matching to avoid partial matches
                    if f"/{head}_" in metric_name: 
                        metric_name_filtered = metric_name.split("/")[-1].replace(f"{head}_", "")
                        metric_values[metric_name_filtered] = metric_value  
            
                organized[head] = metric_values            
            return organized
        
        try:
            # Convert all metrics to JSON-serializable format
            serializable_metrics = convert_to_serializable(validation_metrics)
            
            # Organize metrics by head
            organized_metrics = organize_metrics_by_head(serializable_metrics)
            
            # Add metadata
            metrics_with_metadata = {
                "model_config": {
                    "head_task": dict(self.config.head_task) if hasattr(self.config, 'head_task') else {},
                    "ci_confidence_level": getattr(self.config, 'ci_confidence_level', 0.95),
                    "ci_n_bootstrap": getattr(self.config, 'ci_n_bootstrap', 1000),
                },
                "metrics_by_head": organized_metrics,
            }
            
            # Create metrics directory
            metrics_dir = os.path.join(self.output_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Save with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = os.path.join(metrics_dir, f"validation_metrics_with_ci_{timestamp_str}.json")
            
            with open(json_file, 'w') as f:
                json.dump(metrics_with_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] rank={self.device} => Saved validation metrics with CI to {json_file}")
            
            # Also save as latest (overwrites previous)
            latest_json_file = os.path.join(metrics_dir, "validation_metrics_with_ci_latest.json")
            with open(latest_json_file, 'w') as f:
                json.dump(metrics_with_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] rank={self.device} => Saved latest validation metrics with CI to {latest_json_file}")
                        
        except Exception as e:
            print(f"[DEBUG] rank={self.device} => Error saving validation metrics to JSON: {e}")

    def _accumulate_batch_metrics(
        self,
        run_batch: BatchResult,
        accumulated_preds: Dict[str, list[torch.Tensor]], # leverage ref. ptr to avoid reference return by Tuple
        accumulated_targets: Dict[str, list[torch.Tensor]], # leverage ref. ptr to avoid reference return by Tuple
        accumulated_names: list[str], # leverage ref. ptr to avoid reference return by Tuple
        gathered_metrics: Dict[str, float], # leverage ref. ptr to avoid reference return by Tuple
        batch: Dict[str, torch.Tensor]
    ) -> None:
        """
        Accumulate metrics from a single batch for later computation.
        
        Args:
            outputs: Model outputs containing losses, logits, and lr metrics
            processed_batch: Preprocessed batch data
            accumulated_preds: Dictionary to accumulate predictions for each head
            accumulated_targets: Dictionary to accumulate targets for each head
            accumulated_names: List to accumulate video names
            gathered_metrics: Dictionary to accumulate scalar metrics
            batch: Original batch data containing video filenames
            
        Note:
            This function modifies the input containers in-place and returns None.            
        """
        # Accumulate metrics
        for k, v in run_batch.outputs.items():
            if k.startswith('lr/'):
                # Learning rate metrics are already scalar values
                gathered_metrics[k] = v
            elif k == 'losses':
                # Losses are already scalar values
                for loss_name, loss_val in v.items():
                    # Store current batch loss
                    gathered_metrics[f"{loss_name}"] = loss_val
                    
                    # Accumulate sum for averaging over all batches
                    if f'mean_{loss_name}' not in gathered_metrics:
                        gathered_metrics[f'mean_{loss_name}'] = 0.0
                    gathered_metrics[f'mean_{loss_name}'] += loss_val

            elif k == 'logits':
                # Handle logits for classification/regression metrics
                for head_name, logits in v.items():
                    # Get targets
                    targets: torch.Tensor = run_batch.processed_batch['batch_targets'][head_name]
                    
                    # Get predictions based on task type
                    if self.config.head_task[head_name] == MetricTask.BINARY_CLASSIFICATION:
                        if logits.ndim != 2 or logits.shape[1] != 1:  # Expected shape: [B, 1]
                            raise ValueError(
                                f"[DEBUG] rank={self.device} => Binary classification head {head_name} "
                                f"should have logits of shape [B, 1], but got {logits.shape}"
                            )
                        preds: torch.Tensor = torch.sigmoid(logits.float())
                        targets: torch.Tensor = targets.long()
                        
                    elif self.config.head_task[head_name] == MetricTask.MULTICLASS_CLASSIFICATION:
                        if logits.ndim != 2 or logits.shape[1] < 2:  # Expected shape: [B, C] with C > 1
                            raise ValueError(
                                f"[DEBUG] rank={self.device} => Multiclass classification head {head_name} "
                                f"should have logits of shape [B, C] where C > 1 (Nb of classes), but got {logits.shape}"
                            )
                        preds: torch.Tensor = torch.softmax(logits.float(), dim=1)
                        targets: torch.Tensor = targets.long()
                        
                    elif self.config.head_task[head_name] == MetricTask.REGRESSION:
                        preds: torch.Tensor = logits
                        targets: torch.Tensor = targets.float()
                        
                    else:
                        raise ValueError(
                            f"[DEBUG] rank={self.device} => Unknown head task: {self.config.head_task[head_name]} "
                            f"Supported tasks: {', '.join([metric.value for metric in MetricTask])}"
                        )
                    
                    # Accumulate predictions and targets
                    accumulated_preds[head_name].append(preds)
                    accumulated_targets[head_name].append(targets)
        
        # Accumulate video names
        accumulated_names.extend(batch['video_fname'])

    def _accumulate_batch_metrics_inference(
        self,
        run_batch: BatchResult,
        accumulated_preds: Dict[str, list[torch.Tensor]], # leverage ref. ptr to avoid reference return by Tuple
        accumulated_names: list[str], # leverage ref. ptr to avoid reference return by Tuple
        batch: Dict[str, torch.Tensor]
    ) -> None:
        """
        Accumulate metrics from a single batch for later computation.
        
        Args:
            outputs: Model outputs containing losses, logits, and lr metrics
            processed_batch: Preprocessed batch data
            accumulated_preds: Dictionary to accumulate predictions for each head
            accumulated_targets: Dictionary to accumulate targets for each head
            accumulated_names: List to accumulate video names
            gathered_metrics: Dictionary to accumulate scalar metrics
            batch: Original batch data containing video filenames
            
        Note:
            This function modifies the input containers in-place and returns None.            
        """
        # Accumulate metrics
        for k, v in run_batch.outputs.items():
            if k == 'logits':
                # Handle logits for classification/regression metrics
                for head_name, logits in v.items():                    
                    # Get predictions based on task type
                    if self.config.head_task[head_name] == MetricTask.BINARY_CLASSIFICATION:
                        if logits.ndim != 2 or logits.shape[1] != 1:  # Expected shape: [B, 1]
                            raise ValueError(
                                f"[DEBUG] rank={self.device} => Binary classification head {head_name} "
                                f"should have logits of shape [B, 1], but got {logits.shape}"
                            )
                        preds: torch.Tensor = torch.sigmoid(logits.float())
                        
                    elif self.config.head_task[head_name] == MetricTask.MULTICLASS_CLASSIFICATION:
                        if logits.ndim != 2 or logits.shape[1] < 2:  # Expected shape: [B, C] with C > 1
                            raise ValueError(
                                f"[DEBUG] rank={self.device} => Multiclass classification head {head_name} "
                                f"should have logits of shape [B, C] where C > 1 (Nb of classes), but got {logits.shape}"
                            )
                        preds: torch.Tensor = torch.softmax(logits.float(), dim=1)
                        
                    elif self.config.head_task[head_name] == MetricTask.REGRESSION:
                        preds: torch.Tensor = logits
                        
                    else:
                        raise ValueError(
                            f"[DEBUG] rank={self.device} => Unknown head task: {self.config.head_task[head_name]} "
                            f"Supported tasks: {', '.join([metric.value for metric in MetricTask])}"
                        )
                    
                    # Accumulate predictions and targets
                    accumulated_preds[head_name].append(preds)
        
        # Accumulate video names
        accumulated_names.extend(batch['video_fname'])

    def _gather_distributed_predictions(
        self,
        accumulated_names: list[str],  # leverage ref. ptr to avoid reference return by Tuple
        accumulated_preds: Dict[str, list[torch.Tensor]],  # leverage ref. ptr to avoid reference return by Tuple
        accumulated_targets: Optional[Dict[str, list[torch.Tensor]]] = None  # leverage ref. ptr to avoid reference return by Tuple
    ) -> None:
        """
        Gather predictions, targets, and video names across all distributed processes.
        
        Args:
            accumulated_preds: Dictionary of predictions for each head (modified in-place)
            accumulated_targets: Dictionary of targets for each head (modified in-place)
            accumulated_names: List of video names (modified in-place)
        
        Note:
            This function modifies the input containers in-place and returns None.
        """
        # Gather video names across all processes
        gathered_names = DistributedUtils.gather_object(accumulated_names, self.config.world_size)
        gathered_names_flat = [name for sublist in gathered_names for name in sublist]
        
        # Clear and update accumulated_names in-place
        accumulated_names.clear()
        accumulated_names.extend(gathered_names_flat)
        
        # Gather predictions and targets for each head
        for head in accumulated_preds.keys():
            # Convert lists to tensors (concatenate batches from this process)
            local_preds: torch.Tensor = torch.cat(accumulated_preds[head], dim=0)
            local_targets: torch.Tensor = torch.cat(accumulated_targets[head], dim=0) if accumulated_targets is not None else None
            
            # Gather tensors across all processes
            gathered_preds: torch.Tensor = DistributedUtils.gather_tensor(local_preds, self.config.world_size)
            gathered_targets: torch.Tensor = DistributedUtils.gather_tensor(local_targets, self.config.world_size) if accumulated_targets is not None else None
            
            # Update accumulated containers in-place (replace list of many tensors with list of one complete tensor)
            accumulated_preds[head] = [gathered_preds]
            if accumulated_targets is not None:
                accumulated_targets[head] = [gathered_targets]
    
    def _compute_heads_metrics(
        self,
        mode: str,
        accumulated_preds: Dict[str, list[torch.Tensor]],
        accumulated_targets: Dict[str, list[torch.Tensor]],
        validation_metrics: Dict[str, float], # leverage ref. ptr to avoid copying
        compute_ci: bool = False
    ) -> None:
        """
        Compute metrics for the given predictions and targets.
        
        Args:
            accumulated_preds: Dictionary of predictions for each head
            accumulated_targets: Dictionary of targets for each head
            validation_metrics: Dictionary to accumulate validation metrics
            
        Note:
            This function modifies the input validation_metrics dictionary in-place.
        """
        for head in tqdm(accumulated_preds.keys(), desc=f"Computing {mode} heads metrics {'with CI' if compute_ci else ''}", total=len(accumulated_preds)):
            # Gather accumulated predictions and targets for each head
            preds = torch.cat(accumulated_preds[head], dim=0)
            targets = torch.cat(accumulated_targets[head], dim=0)
            
            if self.config.head_task[head] == MetricTask.BINARY_CLASSIFICATION:
                # Compute classification metrics WITH CI
                head_metrics = compute_binary_classification_metrics(
                    preds=preds.squeeze(-1) if preds.ndim == 2 else preds, # squeeze to remove the last dimension of the preds tensor - currently [B, 1] and expects [B,]
                    targets=targets.squeeze(-1) if targets.ndim == 2 else targets, # squeeze to remove the last dimension of the targets tensor - currently [B, 1] and expects [B,]
                    head_name=head,
                    labels_map=self.config.labels_map,
                    mode=mode, 
                    wandb_wrapper=self.wandb_wrapper,
                    is_ref_device=self.config.is_ref_device,
                    confidence_level=getattr(self.config, 'ci_confidence_level', 0.95),
                    n_bootstrap=getattr(self.config, 'ci_n_bootstrap', 1000),
                    compute_ci=compute_ci
                )
            elif self.config.head_task[head] == MetricTask.MULTICLASS_CLASSIFICATION:
                # Compute classification metrics WITH CI for multiclass
                head_metrics = compute_multiclass_classification_metrics(
                    preds=preds,
                    targets=targets,
                    head_name=head,
                    labels_map=self.config.labels_map,
                    mode=mode, 
                    wandb_wrapper=self.wandb_wrapper,
                    is_ref_device=self.config.is_ref_device,
                    confidence_level=getattr(self.config, 'ci_confidence_level', 0.95),
                    n_bootstrap=getattr(self.config, 'ci_n_bootstrap', 1000),
                    compute_ci=compute_ci
                )
            elif self.config.head_task[head] == MetricTask.REGRESSION:
                # Compute regression metrics WITH CI
                head_metrics = compute_regression_metrics(
                    preds=preds,
                    targets=targets,
                    head_name=head,
                    mode=mode, 
                    wandb_wrapper=self.wandb_wrapper,
                    is_ref_device=self.config.is_ref_device,
                    confidence_level=getattr(self.config, 'ci_confidence_level', 0.95),
                    n_bootstrap=getattr(self.config, 'ci_n_bootstrap', 1000),
                    compute_ci=compute_ci
                )
            else:
                raise ValueError(
                    f"[DEBUG] rank={self.device} => Unknown head task: {self.config.head_task[head]} "
                    f"Supported tasks: {', '.join([metric.value for metric in MetricTask])}"
                )
            
            # Update inference metrics with computed metrics for the current head
            for k, v in head_metrics.items():
                validation_metrics[k] = v