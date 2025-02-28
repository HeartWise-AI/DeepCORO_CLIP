import os
import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from utils.enums import RunMode
from utils.loss.typing import Loss
from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.wandb_wrapper import WandbWrapper
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
        for batch_idx, batch in enumerate(data_iter, start=1):
            batch_video = batch['videos'].to(self.device)
            batch_targets = batch['targets']
            for k, v in batch_targets.items():
                batch_targets[k] = v.to(self.device)
            # [B, 1, T, H, W, C] -> 1 is the aggregator dimension
            batch_video = batch_video.unsqueeze(1)
            try:
                outputs = step_fn(batch_video, batch_targets)
            except Exception as e:
                print(f"[DEBUG] rank={self.device} => Error in step_fn: {e} for batch {batch['video_fname']} - {batch_targets}")
                continue
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
        
        # Get mean epoch losses
        for k, v in gathered_metrics.items():
            if 'mean' in k:
                epoch_metrics[f"{mode}/{k.replace('mean_', '')}_loss"] = v / batch_idx
        
        return epoch_metrics
        
    def _train_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, float]:
        # Clear gradients
        self.optimizer.zero_grad()
                
        # Forward pass with autocast for mixed precision
        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16
        ):
            outputs_dict: dict[str, torch.Tensor] = self.linear_probing(batch_video)

        losses: dict[str, torch.Tensor] = self.loss_fn.run(
            outputs=outputs_dict, 
            targets=batch_targets
        )
        
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
            "losses": losses
        }
        
    def _val_step(
        self, 
        batch_video: torch.Tensor,
        batch_targets: torch.Tensor
    ) -> dict[str, float]:                
        # Forward pass with autocast for mixed precision
        with torch.no_grad():
            outputs_dict: dict[str, torch.Tensor] = self.linear_probing(batch_video)

        losses: dict[str, torch.Tensor] = self.loss_fn.run(
            outputs=outputs_dict, 
            targets=batch_targets
        )
                
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
                
        return {
            "losses": losses
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