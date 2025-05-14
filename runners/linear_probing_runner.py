import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

from tqdm import tqdm
from scipy.stats import pearsonr
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)
from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.enums import RunMode, MetricTask
from utils.wandb_wrapper import WandbWrapper
from utils.metrics import compute_best_threshold
from utils.loss.losses import LossRegistry, LossType
from utils.config.linear_probing_config import LinearProbingConfig
from models.linear_probing import LinearProbing


@RunnerRegistry.register("DeepCORO_video_linear_probing")
@RunnerRegistry.register("DeepCORO_video_linear_probing_test")
class LinearProbingRunner:
    """
    This class runs a linear probing pipeline using a VideoEncoder and TextEncoder.
    It handles both training and validation loops in a distributed data-parallel setting.
    """

    def __init__(
        self,
        config: LinearProbingConfig,
        wandb_wrapper: WandbWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        linear_probing: LinearProbing,
        optimizer: Optimizer,
        scaler: GradScaler,
        lr_scheduler: LRScheduler,
        loss_fn: Loss,
        output_dir: str,
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
        self.linear_probing: LinearProbing = linear_probing
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
            train_metrics = self._run_epoch(mode=RunMode.TRAIN, epoch=epoch)   
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
                mode=RunMode.VALIDATION, 
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
                self.wandb_wrapper.log(val_metrics)
                print(f"[DEBUG] rank={self.device} => Logged val metrics to W&B")  
                
            # Sync after logging
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )                                 

    def _run_epoch(
        self, 
        mode: str, 
        epoch: int
    ) -> dict[str, float]:
        assert mode in [RunMode.TRAIN, RunMode.VALIDATION]
        
        self.linear_probing.train(mode == RunMode.TRAIN)

        epoch_metrics: dict[str, float] = {}

        dataloader: DataLoader = self.train_loader if mode == RunMode.TRAIN else self.val_loader
        step_fn: callable = self._train_step if mode == RunMode.TRAIN else self._val_step

        tqdm_desc: str = f"[GPU {self.device}] Running {mode} Epoch {epoch + 1}"
        if self.config.is_ref_device:
            data_iter: tqdm = tqdm(dataloader, desc=tqdm_desc)
        else:
            data_iter: DataLoader = dataloader
        
        gathered_metrics: dict[str, float] = {}
        accumulated_preds: dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_targets: dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_names: list[str] = []
        for batch_idx, batch in enumerate(data_iter, start=1):
            try:
                processed_batch: dict[str, torch.Tensor] = self._preprocess_inputs(batch)
                outputs: dict[str, dict[str, torch.Tensor]] = step_fn(**processed_batch)
                
                # Accumulate metrics
                for k, v in outputs.items():
                    if k.startswith('lr/'):
                        # Learning rate metrics are already scalar values
                        gathered_metrics[k] = v
                    elif k == 'losses':
                        # Losses are already scalar values
                        for loss_name, loss_val in v.items():
                            gathered_metrics[f"{loss_name}"] = loss_val
                            gathered_metrics[f'mean_{loss_name}'] = gathered_metrics.get(f"mean_{loss_name}", 0.0) + loss_val
                    elif k == 'logits':
                        # Handle logits for classification metrics
                        for head_name, logits in v.items():
                            if self.config.head_task[head_name] == MetricTask.CLASSIFICATION:
                                preds = torch.sigmoid(logits.float())
                                targets = processed_batch['batch_targets'][head_name].long()
                                if preds.ndim > 1 and preds.shape[1] > 1:
                                    one_hot_targets = torch.zeros_like(preds)
                                    one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
                                    targets = one_hot_targets
                                accumulated_preds[head_name].append(preds)
                                accumulated_targets[head_name].append(targets)
                            elif self.config.head_task[head_name] == MetricTask.REGRESSION:
                                accumulated_preds[head_name].append(logits)
                                accumulated_targets[head_name].append(processed_batch['batch_targets'][head_name].float())
                            else:
                                raise ValueError(
                                    f"[DEBUG] rank={self.device} => Unknown head task: {self.config.head_task[head_name]} "
                                    f"Supported tasks: {MetricTask.CLASSIFICATION}, {MetricTask.REGRESSION}"
                                )
                
                accumulated_names.extend(batch['video_fname'])
                
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in step_fn: {e}")
            
            if self.lr_scheduler and self.scheduler_per_iteration and (mode == RunMode.TRAIN):
                self.lr_scheduler.step()

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

        # Gather all predictions and targets across GPUs
        if self.config.world_size > 1:
            # Gather video names
            gathered_names = DistributedUtils.gather_object(accumulated_names, self.config.world_size)
            accumulated_names = [name for sublist in gathered_names for name in sublist]

            # Gather predictions and targets for each head
            for head in accumulated_preds.keys():
                # Convert lists to tensors
                local_preds: torch.Tensor = torch.cat(accumulated_preds[head], dim=0)
                local_targets: torch.Tensor = torch.cat(accumulated_targets[head], dim=0)

                # Gather tensors
                gathered_preds: torch.Tensor = DistributedUtils.gather_tensor(local_preds, self.config.world_size)
                gathered_targets: torch.Tensor = DistributedUtils.gather_tensor(local_targets, self.config.world_size)

                # Update accumulated tensors
                accumulated_preds[head] = [gathered_preds]
                accumulated_targets[head] = [gathered_targets]

        # Get mean epoch losses
        for k, v in gathered_metrics.items():
            if 'mean' in k:
                epoch_metrics[f"{mode}/{k.replace('mean_', '')}_loss"] = v / batch_idx
        
        # Add learning rate metrics to epoch metrics
        for k, v in outputs.items():
            if k.startswith('lr/'):
                epoch_metrics[f"{mode}/{k}"] = v

        # Compute AUC metrics for each head
        if mode == RunMode.TRAIN or mode == RunMode.VALIDATION:
            for head in accumulated_preds.keys():
                # Gather accumulated predictions and targets for each head
                preds = torch.cat(accumulated_preds[head], dim=0)
                targets = torch.cat(accumulated_targets[head], dim=0)
                
                if self.config.head_task[head] == MetricTask.CLASSIFICATION:                
                    # Compute metrics using the helper function
                    head_metrics = compute_classification_metrics(
                        preds=preds,
                        targets=targets,
                        head_name=head,
                        head_structure=self.config.head_structure,
                        labels_map=self.config.labels_map,
                        mode=mode,
                        wandb_wrapper=self.wandb_wrapper,
                        is_ref_device=self.config.is_ref_device
                    )
                    
                elif self.config.head_task[head] == MetricTask.REGRESSION:
                    # Compute metrics using the helper function
                    head_metrics = compute_regression_metrics(
                        preds=preds,
                        targets=targets,
                        head_name=head,
                        mode=mode,
                        wandb_wrapper=self.wandb_wrapper,
                        is_ref_device=self.config.is_ref_device
                    )
                    
                # Update epoch metrics with the computed metrics for the current head
                for k, v in head_metrics.items():
                    epoch_metrics[f"{mode}/{head}_{k}"] = v
                
            # Sync after logging metrics and confusion matrix
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )

        print(f"[DEBUG] rank={self.device} => {mode} epoch metrics: {epoch_metrics}")

        # Save predictions for validation mode
        if mode == RunMode.VALIDATION and self.config.is_ref_device:
            self._save_predictions(
                mode=mode,
                accumulated_names=accumulated_names,
                accumulated_preds=accumulated_preds,
                accumulated_targets=accumulated_targets,
                epoch=epoch
            )

        return epoch_metrics

    def _preprocess_inputs(
        self, 
        batch: dict[str, torch.Tensor]
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
            
        # Move video indices to device if present
        video_indices = batch.get('video_indices')
        if video_indices is not None:
            video_indices = video_indices.to(self.config.device)
            
        return {
            "batch_video": batch_video,
            "batch_targets": batch_targets,
            "video_indices": video_indices
        }
        
    def _train_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: dict[str, torch.Tensor],
        video_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # Clear gradients only if this is the first step in accumulation
        if self.step % self.config.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
                
        # Forward pass with autocast for mixed precision
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            try:
                outputs_dict: dict[str, torch.Tensor] = self.linear_probing(
                    batch_video, 
                    video_indices=video_indices
                )
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in linear_probing: {e} for batch with video shape {batch_video.shape}")

        try:
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
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        # Increment step counter
        self.step += 1

        # Get learning rate metrics
        lr_metrics = {}
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
        batch_targets: dict[str, torch.Tensor],
        video_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, dict[str, torch.Tensor]]:                
        # Forward pass with autocast for mixed precision
        with torch.no_grad():
            try:
                outputs_dict: dict[str, torch.Tensor] = self.linear_probing(
                    batch_video,
                    video_indices=video_indices
                )
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in linear_probing: {e} for batch with video shape {batch_video.shape}")

        try:
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

    def inference(self) -> None:
        raise NotImplementedError("Linear probing inference not implemented")
    
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
            "optimizer": self.optimizer.state_dict(),
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
        accumulated_names: list[str],
        accumulated_preds: dict[str, torch.Tensor],
        accumulated_targets: dict[str, torch.Tensor],
        epoch: int
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
        # Only save for validation mode
        if mode != RunMode.VALIDATION:
            return

        try:
            # Debug array lengths
            print(f"[DEBUG] rank={self.device} => Number of accumulated names: {len(accumulated_names)}")
            for head in accumulated_preds.keys():
                print(f"[DEBUG] rank={self.device} => Head {head} - Predictions shape: {accumulated_preds[head][0].shape}, Targets shape: {accumulated_targets[head][0].shape}")

            # Create predictions dictionary with epoch column
            predictions_dict: dict[str, list] = {
                'epoch': [epoch] * len(accumulated_names),
                'video_name': accumulated_names
            }
            
            # Add predictions and ground truth for each head
            for head in accumulated_preds.keys():
                preds: np.ndarray = accumulated_preds[head][0].squeeze().detach().cpu().float().numpy()
                targets: np.ndarray = accumulated_targets[head][0].squeeze().detach().cpu().float().numpy()
                
                # Debug shapes after conversion
                print(f"[DEBUG] rank={self.device} => Head {head} - After conversion - Preds shape: {preds.shape}, Targets shape: {targets.shape}")
                
                # Handle binary vs multi-class classification
                if self.config.head_structure[head] == 1:
                    predictions_dict[f'{head}_pred'] = preds
                    predictions_dict[f'{head}_true'] = targets
                else:
                    # For multi-class, get both raw probabilities and predicted class
                    pred_labels: np.ndarray = preds.argmax(axis=1)
                    target_labels: np.ndarray = targets.argmax(axis=1)
                    
                    # Debug shapes after argmax
                    print(f"[DEBUG] rank={self.device} => Head {head} - After argmax - Pred labels shape: {pred_labels.shape}, Target labels shape: {target_labels.shape}")
                    
                    # Map numeric labels to actual class names
                    label_map: dict[str, int] = self.config.labels_map[head]
                    rev_label_map: dict[int, str] = {v: k for k, v in label_map.items()}
                    
                    pred_classes: list[str] = [rev_label_map[i] for i in pred_labels]
                    true_classes: list[str] = [rev_label_map[i] for i in target_labels]
                    
                    # Debug lengths after class mapping
                    print(f"[DEBUG] rank={self.device} => Head {head} - After class mapping - Pred classes len: {len(pred_classes)}, True classes len: {len(true_classes)}")
                    
                    predictions_dict[f'{head}_pred_class'] = pred_classes
                    predictions_dict[f'{head}_true_class'] = true_classes
                    
                    # Add probabilities for each class
                    for class_name, class_idx in label_map.items():
                        predictions_dict[f'{head}_prob_{class_name}'] = preds[:, class_idx]
            
            # Debug final dictionary lengths
            print(f"[DEBUG] rank={self.device} => Final dictionary lengths:")
            for k, v in predictions_dict.items():
                print(f"[DEBUG] rank={self.device} => {k}: {len(v)}")
            
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
                print(f"[DEBUG] rank={self.device} =>   Predictions: {accumulated_preds[head][0].shape}")
                print(f"[DEBUG] rank={self.device} =>   Targets: {accumulated_targets[head][0].shape}")
            if 'predictions_dict' in locals():
                print(f"[DEBUG] rank={self.device} => Predictions dictionary keys and lengths:")
                for k, v in predictions_dict.items():
                    print(f"[DEBUG] rank={self.device} =>   {k}: {len(v)}")

def compute_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    mode: str = "val",
    wandb_wrapper = None,
    is_ref_device: bool = False
) -> dict:
    """
    Compute regression metrics for the given predictions and targets,
    and log a regression plot if on the reference device and W&B is initialized.
    """
    metrics = {}

    with torch.no_grad():
        # Compute MAE
        metrics[f"{mode}/{head_name}_mae"] = LossRegistry.get(LossType.MAE)()(
            outputs=preds,
            targets=targets
        ).item()

        # Compute MSE
        metrics[f"{mode}/{head_name}_mse"] = LossRegistry.get(LossType.MSE)()(
            outputs=preds,
            targets=targets
        ).item()

        # Compute RMSE
        metrics[f"{mode}/{head_name}_rmse"] = LossRegistry.get(LossType.RMSE)()(
            outputs=preds,
            targets=targets
        ).item()

    # Convert tensors to numpy arrays
    preds_np = preds.detach().cpu().float().numpy().squeeze()
    targets_np = targets.detach().cpu().float().numpy().squeeze()

    # Calculate Pearson correlation using numpy arrays
    try:
        r, _ = pearsonr(preds_np, targets_np)
        metrics[f"{mode}/{head_name}_pearson_r"] = r
    except ValueError as e:
        print(f"Could not compute Pearson correlation for {head_name}: {e}")
        r, p_value = np.nan, np.nan # Assign NaN if calculation fails

    # Generate and log regression plot if wandb is initialized and on ref device
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized():
        try:
            plt.figure(figsize=(10, 8))
            sns.regplot(x=targets_np, y=preds_np, line_kws={'color': 'red'})
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plot_title = (
                f'{mode.capitalize()} Regression Plot - {head_name}\n'
                f'Pearson r: {r:.3f}'
            )
            plt.title(plot_title)
            plt.grid(True)

            # Log to wandb
            wandb_wrapper.log_plot({
                f"regression_plot/{mode}/{head_name}": plt
            })
            plt.close() # Close the plot to free memory

        except Exception as e:
            print(f"Error generating/logging regression plot for {head_name}: {e}")
            plt.close() # Ensure plot is closed even if error occurs

    return metrics

def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    head_structure: dict,
    labels_map: dict = None,
    mode: str = "val",
    wandb_wrapper = None,
    is_ref_device: bool = False
) -> dict:
    """
    Compute classification metrics for the given predictions and targets.
    
    Args:
        preds: Tensor of model predictions
        targets: Tensor of ground truth labels
        head_name: Name of the model head being evaluated
        head_structure: Dictionary containing output dimensions for each head
        labels_map: Dictionary mapping class names to indices
        mode: Current mode (train or validation)
        device: Current device ID
        wandb_wrapper: WandbWrapper instance for logging
        is_ref_device: Whether this device is the reference device for logging
        
    Returns:
        Dictionary of computed metrics
    """
    
    metrics = {}
    
    # Convert to numpy for sklearn metrics
    all_preds = preds.detach().cpu().numpy()
    all_targets = targets.detach().cpu().numpy()
    
    # Compute AUC and AUPRC
    try:
        auc = roc_auc_score(all_targets.tolist(), all_preds.tolist(), average="micro")
        metrics[f"{mode}/{head_name}_auc"] = auc
    except Exception as e:
        print(f"Error computing AUC: {e}")
    
    try:
        auprc = average_precision_score(all_targets.tolist(), all_preds.tolist(), average="micro")
        metrics[f"{mode}/{head_name}_auprc"] = auprc
    except Exception as e:
        print(f"Error computing AUPRC: {e}")
    
    # Compute and log confusion matrix if wandb is initialized
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized():
        try:
            # For binary classification
            if head_structure[head_name] == 1:
                try:
                    # Compute best threshold using Youden's J statistic
                    best_threshold = compute_best_threshold(
                        all_targets.tolist(), 
                        all_preds.tolist()
                    )
                except Exception:
                    # If error, use default threshold 0.5
                    best_threshold = 0.5
                
                # Binarize predictions
                pred_labels = (all_preds > best_threshold).astype(int)
                
                # Log best threshold
                metrics[f"{mode}/{head_name}_best_threshold"] = best_threshold
            
            # For multi-class classification
            else:
                pred_labels = all_preds.argmax(axis=1)
                all_targets = all_targets.argmax(axis=1)
            
            # Create confusion matrix if labels_map is provided
            if labels_map:
                # Create labels list
                labels = [''] * len(labels_map[head_name])
                for k, v in labels_map[head_name].items():
                    labels[v] = k
                
                # Compute confusion matrix
                cm = confusion_matrix(y_true=all_targets, y_pred=pred_labels)
                
                # Create confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=labels, 
                    yticklabels=labels
                )
                plt.title(f'{mode.capitalize()} Confusion Matrix - {head_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                # Log to wandb
                wandb_wrapper.log_plot({
                    f"confusion_matrix/{mode}/{head_name}": plt
                })
                plt.close()
        
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
    
    return metrics