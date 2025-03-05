import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)
import pandas as pd

from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.enums import RunMode, MetricTask
from utils.wandb_wrapper import WandbWrapper
from utils.metrics import compute_best_threshold
from utils.config.linear_probing_config import LinearProbingConfig

from models.linear_probing import LinearProbing


@RunnerRegistry.register("DeepCORO_video_linear_probing")
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

    def train(
        self, 
        start_epoch: int, 
        end_epoch: int
    ):
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
            
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log(val_metrics)
                print(f"[DEBUG] rank={self.device} => Logged val metrics to W&B")

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

    def _run_epoch(
        self, 
        mode: str, 
        epoch: int
    ) -> dict[str, float]:
        assert mode in [RunMode.TRAIN, RunMode.VALIDATION]
        
        self.linear_probing.train(mode == RunMode.TRAIN)

        epoch_metrics: dict[str, float] = {}

        dataloader = self.train_loader if mode == RunMode.TRAIN else self.val_loader
        step_fn = self._train_step if mode == RunMode.TRAIN else self._val_step

        tqdm_desc = f"[GPU {self.device}] Running {mode} Epoch {epoch + 1}"
        if self.config.is_ref_device:
            data_iter = tqdm(dataloader, desc=tqdm_desc)
        else:
            data_iter = dataloader
        
        gathered_metrics: dict[str, float] = {}
        accumulated_preds: dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_targets: dict[str, list[torch.Tensor]] = defaultdict(list)
        accumulated_names: list[str] = []
        for batch_idx, batch in enumerate(data_iter, start=1):
            try:
                processed_batch: dict[str, torch.Tensor] = self._preprocess_inputs(batch)
                outputs: dict[str, torch.Tensor] = step_fn(**processed_batch)
                if self.config.task == MetricTask.CLASSIFICATION:
                    for k, v in outputs['logits'].items():
                        preds = torch.sigmoid(v.float())
                        targets = processed_batch['batch_targets'][k].long()
                        if preds.ndim > 1 and preds.shape[1] > 1:
                            one_hot_targets = torch.zeros_like(preds)
                            one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
                            targets = one_hot_targets
                        accumulated_preds[k].append(preds)
                        accumulated_targets[k].append(targets)
                accumulated_names.extend(batch['video_fname'])
                
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in step_fn: {e}")
            
            for k, v in outputs['losses'].items():
                gathered_metrics[f"{k}"] = DistributedUtils.gather_loss(
                    [v], 
                    self.config.device
                )
                # Update total loss
                gathered_metrics[f'mean_{k}'] = gathered_metrics.get(f"mean_{k}", 0.0) + gathered_metrics[f"{k}"]
                
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
        
        # Gather and compute AUC and AUPRC metrics only if task is classification
        if self.config.task == MetricTask.CLASSIFICATION:
            # Convert lists into concatenated tensors per head and gather from all GPUs
            for head in list(accumulated_preds.keys()):
                local_preds = torch.cat(accumulated_preds[head], dim=0)
                local_targets = torch.cat(accumulated_targets[head], dim=0)

                # If distributed is initialized and more than one process exists, gather values
                if torch.distributed.is_available() and torch.distributed.is_initialized() and self.config.world_size > 1:
                    preds_list = [torch.zeros_like(local_preds) for _ in range(self.config.world_size)]
                    targets_list = [torch.zeros_like(local_targets) for _ in range(self.config.world_size)]
                    torch.distributed.all_gather(preds_list, local_preds)
                    torch.distributed.all_gather(targets_list, local_targets)
                    global_preds = torch.cat(preds_list, dim=0)
                    global_targets = torch.cat(targets_list, dim=0)
                else:
                    global_preds = local_preds
                    global_targets = local_targets

                accumulated_preds[head] = global_preds
                accumulated_targets[head] = global_targets

            # Sync after gathering predictions and targets
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )

            # Save predictions to CSV if on reference device
            if self.config.is_ref_device:
                try:
                    # Create predictions dictionary
                    predictions_dict = {
                        'video_name': accumulated_names
                    }
                    
                    # Add predictions and ground truth for each head
                    for head in accumulated_preds.keys():
                        preds = accumulated_preds[head].squeeze().detach().cpu().numpy()
                        targets = accumulated_targets[head].squeeze().detach().cpu().numpy()
                        
                        # Handle binary vs multi-class classification
                        if self.config.head_structure[head] == 1:
                            predictions_dict[f'{head}_pred'] = preds
                            predictions_dict[f'{head}_true'] = targets
                        else:
                            # For multi-class, get both raw probabilities and predicted class
                            pred_labels = preds.argmax(axis=1)
                            target_labels = targets.argmax(axis=1)
                            
                            # Map numeric labels to actual class names
                            label_map = self.config.labels_map[head]
                            rev_label_map = {v: k for k, v in label_map.items()}
                            
                            pred_classes = [rev_label_map[i] for i in pred_labels]
                            true_classes = [rev_label_map[i] for i in target_labels]
                            
                            predictions_dict[f'{head}_pred_class'] = pred_classes
                            predictions_dict[f'{head}_true_class'] = true_classes
                            
                            # Add probabilities for each class
                            for class_name, class_idx in label_map.items():
                                predictions_dict[f'{head}_prob_{class_name}'] = preds[:, class_idx]
                    
                    # Create and save DataFrame
                    df = pd.DataFrame(predictions_dict)
                    output_file = os.path.join(
                        self.output_dir, 
                        f'{mode}_predictions.csv'
                    )
                    df.to_csv(output_file, index=False)
                    print(f"[DEBUG] rank={self.device} => Saved predictions to {output_file}")
                    
                except Exception as e:
                    print(f"[DEBUG] rank={self.device} => Error saving predictions to CSV: {e}")

            # Compute metrics for each head
            for head in accumulated_preds.keys():
                all_preds: np.ndarray = accumulated_preds[head].squeeze().detach().cpu().numpy()
                all_targets: np.ndarray = accumulated_targets[head].squeeze().detach().cpu().numpy()
                print(f"[DEBUG] rank={self.device} => head={head} all_preds shape: {all_preds.shape}, all_targets shape: {all_targets.shape}")
                try:
                    auc: float = roc_auc_score(all_targets.tolist(), all_preds.tolist(), average="micro")
                    print(f"[DEBUG] rank={self.device} => head={head} AUC: {auc}")
                except Exception as e:
                    raise Exception(f"[DEBUG] rank={self.device} => Error in AUC: {e}")
                    
                try:
                    auprc: float = average_precision_score(all_targets.tolist(), all_preds.tolist(), average="micro")
                    print(f"[DEBUG] rank={self.device} => {mode} head={head} AUPRC: {auprc}")
                except Exception as e:
                    raise Exception(f"[DEBUG] rank={self.device} => Error in AUPRC: {e}")
                    
                epoch_metrics[f"{mode}/{head}_auc"] = auc
                epoch_metrics[f"{mode}/{head}_auprc"] = auprc

                # Compute and log confusion matrix
                if self.config.is_ref_device and self.wandb_wrapper.is_initialized():
                    try:
                        # For binary classification
                        if self.config.head_structure[head] == 1:
                            try:
                                # Compute best threshold using Youden's J statistic
                                best_threshold: float = compute_best_threshold(
                                    all_targets.tolist(), 
                                    all_preds.tolist()
                                )
                                print(f"[DEBUG] rank={self.device} => head={head} best_threshold: {best_threshold}")
                            except Exception as e:
                                # If error, use default threshold 0.5
                                print(f"[DEBUG] rank={self.device} => Error computing threshold: {e} using default threshold 0.5")
                                best_threshold = 0.5
                                
                            # Binarize predictions
                            pred_labels: np.ndarray = (all_preds > best_threshold).astype(int)
                            
                            # Log best threshold
                            epoch_metrics[f"{mode}/{head}_best_threshold"] = best_threshold
                            
                        # For multi-class classification
                        else:
                            pred_labels: np.ndarray = all_preds.argmax(axis=1)
                            all_targets: np.ndarray = all_targets.argmax(axis=1)
                        
                        # Create confusion matrix using preprocessed targets and predictions
                        labels: list[str] = [''] * len(self.config.labels_map[head])
                        for k, v in self.config.labels_map[head].items():
                            labels[v] = k
                        
                        cm: np.ndarray = confusion_matrix(y_true=all_targets, y_pred=pred_labels)

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
                        plt.title(f'{mode.capitalize()} Confusion Matrix - {head}')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        
                        # Log to wandb
                        self.wandb_wrapper.log_plot({
                            f"confusion_matrix/{mode}/{head}": plt
                        })
                        plt.close()
                        
                    except Exception as e:
                        print(f"[DEBUG] rank={self.device} => Error in confusion matrix: {e}")
                        print(f"[DEBUG] rank={self.device} => all_preds shape: {all_preds.shape}, all_targets shape: {all_targets.shape}")

            # Sync after logging metrics and confusion matrix
            DistributedUtils.sync_process_group(
                world_size=self.world_size,
                device_ids=self.device
            )
                
        print(f"[DEBUG] rank={self.device} => {mode} Epoch metrics: {epoch_metrics}")
        # Get mean epoch losses
        for k, v in gathered_metrics.items():
            if 'mean' in k:
                epoch_metrics[f"{mode}/{k.replace('mean_', '')}_loss"] = v / batch_idx
        
        return epoch_metrics

    def _preprocess_inputs(
        self, 
        batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        batch_video = batch['videos'].to(self.device)
        batch_targets = batch['targets']
        for k, v in batch_targets.items():
            batch_targets[k] = v.to(self.device)
        # [B, 1, T, H, W, C] -> 1 is the aggregator dimension
        batch_video = batch_video.unsqueeze(1)
        return {
            "batch_video": batch_video,
            "batch_targets": batch_targets
        }
        
    def _train_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        # Clear gradients
        self.optimizer.zero_grad()
                
        # Forward pass with autocast for mixed precision
        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16
        ):
            try:
                outputs_dict: dict[str, torch.Tensor] = self.linear_probing(batch_video)
            except Exception as e:
                raise Exception(f"[DEBUG] rank={self.device} => Error in linear_probing: {e} for batch with video shape {batch_video.shape}")

        try:
            losses: dict[str, torch.Tensor] = self.loss_fn.run(
                outputs=outputs_dict, 
                targets=batch_targets
            )
        except Exception as e:
            raise Exception(f"[DEBUG] rank={self.device} => Error in loss_fn: {e} for batch with outputs_dict {outputs_dict} and batch_targets {batch_targets}")

        
        # Backward pass with gradient scaling
        self.scaler.scale(losses['main']).backward()
        
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "losses": losses,
            "logits": outputs_dict
        }
        
    def _val_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: dict[str, torch.Tensor]
    ) -> dict[str, float]:                
        # Forward pass with autocast for mixed precision
        with torch.no_grad():
            try:
                outputs_dict: dict[str, torch.Tensor] = self.linear_probing(batch_video)
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
                
        return {
            "losses": losses,
            "logits": outputs_dict
        }

    def inference(self):
        raise NotImplementedError("Linear probing inference not implemented")
    
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Saves model checkpoint (including model weights, optimizer, scheduler, and metrics).

        :param epoch: Current epoch index.
        :param metrics: Dictionary of metrics to be saved.
        :param is_best: If True, saves as 'best_epoch.pt'. Otherwise, saves as 'checkpoint.pt'.
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_dict = {
            "linear_probing": self.linear_probing.module.state_dict()
            if hasattr(self.linear_probing, "module")
            else self.linear_probing.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "epoch": epoch,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

        checkpoint = {
            **model_dict,
            **metrics,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }

        save_path = os.path.join(
            checkpoint_dir, "best_epoch.pt" if is_best else "checkpoint.pt"
        )
        torch.save(checkpoint, save_path)
        print(
            f"\nSaved {'best' if is_best else 'latest'} checkpoint at epoch {epoch + 1} to {save_path}"
        )    