"""Enhanced video contrastive learning runner with two-optimizer setup and LLRD."""

import os
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
from tqdm.auto import tqdm
import math

from utils.config.clip_config import ClipConfig
from utils.loss.typing import Loss
from utils.registry import RunnerRegistry
from utils.wandb_logger import WandbWrapper
from utils.enums import RunMode
from runners.video_constrative_learning_runner import VideoContrastiveLearningRunner
from utils.optimizer_utils import clamp_logit_scale, PhasedTrainingScheduler


@RunnerRegistry.register("DeepCORO_clip_v2")
class VideoContrastiveLearningRunnerV2(VideoContrastiveLearningRunner):
    """Enhanced runner with two-optimizer setup, LLRD, and phased training."""
    
    def __init__(
        self,
        config: ClipConfig,
        wandb_wrapper: WandbWrapper,
        video_encoder: nn.Module,
        text_encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Loss,
        log_temp: nn.Parameter,
        output_dir: Optional[str] = None,
        # Original optimizer setup (for compatibility)
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        scaler: Optional[GradScaler] = None,
        # New two-optimizer setup
        video_optimizer: Optional[torch.optim.Optimizer] = None,
        text_optimizer: Optional[torch.optim.Optimizer] = None,
        video_scheduler: Optional[Any] = None,
        text_scheduler: Optional[Any] = None,
        video_scaler: Optional[GradScaler] = None,
        text_scaler: Optional[GradScaler] = None,
        # Projection heads
        video_proj: Optional[nn.Module] = None,
        text_proj: Optional[nn.Module] = None,
        # Phased training
        phased_scheduler: Optional[PhasedTrainingScheduler] = None,
        **kwargs
    ):
        """Initialize enhanced runner.
        
        Args:
            config: Configuration object
            wandb_wrapper: Wandb wrapper
            video_encoder: Video encoder model
            text_encoder: Text encoder model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            loss_fn: Loss function
            log_temp: Logit scale parameter
            output_dir: Output directory for checkpoints
            optimizer: Single optimizer (fallback)
            lr_scheduler: Single scheduler (fallback)
            scaler: Single grad scaler (fallback)
            video_optimizer: Video optimizer (two-opt setup)
            text_optimizer: Text optimizer (two-opt setup)
            video_scheduler: Video scheduler (two-opt setup)
            text_scheduler: Text scheduler (two-opt setup)
            video_scaler: Video grad scaler (two-opt setup)
            text_scaler: Text grad scaler (two-opt setup)
            video_proj: Video projection head
            text_proj: Text projection head
            phased_scheduler: Phased training scheduler
        """
        # Initialize base class
        super().__init__(
            config=config,
            wandb_wrapper=wandb_wrapper,
            video_encoder=video_encoder,
            text_encoder=text_encoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=scaler,
            log_temp=log_temp,
            loss_fn=loss_fn,
            output_dir=output_dir,
        )
        
        # Store two-optimizer components
        self.use_two_optimizers = config.use_two_optimizers
        self.video_optimizer = video_optimizer
        self.text_optimizer = text_optimizer
        self.video_scheduler = video_scheduler
        self.text_scheduler = text_scheduler
        self.video_scaler = video_scaler
        self.text_scaler = text_scaler
        
        # Store projection heads
        self.video_proj = video_proj
        self.text_proj = text_proj
        
        # Store phased training scheduler
        self.phased_scheduler = phased_scheduler
        self.current_phase = None
        
        # Track global step for phased training and wandb logging
        # This counts the total number of training steps across all epochs
        self.global_step = 0
        
        # Mixed precision settings
        self.text_amp_enabled = config.text_amp_enabled if hasattr(config, 'text_amp_enabled') else True
        self.video_amp_enabled = config.video_amp_enabled if hasattr(config, 'video_amp_enabled') else True
        self.keep_norm_fp32 = config.keep_norm_fp32 if hasattr(config, 'keep_norm_fp32') else True
        self.grad_clip_norm = config.grad_clip_norm if hasattr(config, 'grad_clip_norm') else 1.0
        
        # Logit scale clamping settings
        self.clamp_logit_scale = config.clamp_logit_scale if hasattr(config, 'clamp_logit_scale') else True
        self.logit_scale_min = config.logit_scale_min if hasattr(config, 'logit_scale_min') else 0.0
        self.logit_scale_max = config.logit_scale_max if hasattr(config, 'logit_scale_max') else 3.912
    
    def _train_step_v2(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[Dict, Dict]:
        """Enhanced training step with two-optimizer support.
        
        Args:
            videos: Video tensor
            input_ids: Text input IDs
            attention_mask: Text attention mask
            
        Returns:
            Tuple of (metrics, embeddings)
        """
        # Phase updates are now handled per epoch in train() method
        
        # Clear gradients
        if self.use_two_optimizers:
            if self.global_step % self.config.gradient_accumulation_steps == 0:
                self.video_optimizer.zero_grad(set_to_none=True)
                self.text_optimizer.zero_grad(set_to_none=True)
        else:
            if self.step % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        if self.use_two_optimizers:
            # Video forward with optional AMP
            with torch.amp.autocast(device_type="cuda", enabled=self.video_amp_enabled):
                video_embeddings = self.video_encoder(videos)
                if self.video_proj is not None:
                    video_embeddings = self.video_proj(video_embeddings)
                video_embeddings = nn.functional.normalize(video_embeddings, dim=-1)
            
            # Text forward with optional AMP and fp32 for LayerNorm
            with torch.amp.autocast(device_type="cuda", enabled=self.text_amp_enabled):
                text_embeddings = self.text_encoder(input_ids, attention_mask)
                if self.text_proj is not None:
                    text_embeddings = self.text_proj(text_embeddings)
                text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)
            
            # Compute loss (always in fp32 for stability)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                video_embeddings_fp32 = video_embeddings.float()
                text_embeddings_fp32 = text_embeddings.float()
                
                # Clamp logit scale if enabled
                if self.clamp_logit_scale:
                    clamp_logit_scale(self.log_temp, self.logit_scale_min, self.logit_scale_max)
                
                temperature = torch.exp(self.log_temp)
                loss = self.loss_fn.run(
                    video_features=video_embeddings_fp32,
                    text_features=text_embeddings_fp32,
                    log_temp=self.log_temp
                )
                # Compute alignment score separately
                from utils.retrieval_metrics import compute_alignment_score
                alignment_score = compute_alignment_score(video_embeddings_fp32, text_embeddings_fp32)
        else:
            # Original single optimizer path
            amp_enabled = self.scaler is not None
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                video_embeddings = self.video_encoder(videos)
                text_embeddings = self.text_encoder(input_ids, attention_mask)
                
                if self.video_proj is not None:
                    video_embeddings = self.video_proj(video_embeddings)
                if self.text_proj is not None:
                    text_embeddings = self.text_proj(text_embeddings)
                
                video_embeddings = nn.functional.normalize(video_embeddings, dim=-1)
                text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)
                
                temperature = torch.exp(self.log_temp)
                loss = self.loss_fn.run(
                    video_features=video_embeddings,
                    text_features=text_embeddings,
                    log_temp=self.log_temp
                )
                # Compute alignment score separately
                from utils.retrieval_metrics import compute_alignment_score
                alignment_score = compute_alignment_score(video_embeddings, text_embeddings)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with two optimizers
        if self.use_two_optimizers:
            # Single backward pass (gradients flow to all parameters that require grad)
            if self.video_scaler is not None and self.video_scaler == self.text_scaler:
                # Use same scaler for both
                self.video_scaler.scale(loss).backward()
            elif self.video_scaler is not None:
                # Use video scaler as primary
                self.video_scaler.scale(loss).backward()
            else:
                # No scaling
                loss.backward()
            
            # Gradient clipping and optimizer step
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Unscale gradients for both optimizers
                if self.video_scaler is not None:
                    self.video_scaler.unscale_(self.video_optimizer)
                    if self.text_scaler is not None and self.text_scaler != self.video_scaler:
                        self.video_scaler.unscale_(self.text_optimizer)
                
                # Collect parameters that require gradients
                video_params = [p for p in self.video_encoder.parameters() if p.requires_grad]
                text_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                
                # Add projection head parameters if they exist
                if self.video_proj is not None:
                    video_params.extend([p for p in self.video_proj.parameters() if p.requires_grad])
                if self.text_proj is not None:
                    text_params.extend([p for p in self.text_proj.parameters() if p.requires_grad])
                
                # Add logit scale to video params if it requires grad
                if hasattr(self, 'log_temp') and self.log_temp.requires_grad:
                    video_params.append(self.log_temp)
                
                # Gradient clipping only for parameters with gradients
                video_grad_norm = torch.tensor(0.0)
                text_grad_norm = torch.tensor(0.0)
                
                if video_params:
                    video_grad_norm = torch.nn.utils.clip_grad_norm_(
                        video_params,
                        self.grad_clip_norm
                    )
                
                if text_params:
                    text_grad_norm = torch.nn.utils.clip_grad_norm_(
                        text_params,
                        self.grad_clip_norm
                    )
                
                # Optimizer steps
                if self.video_scaler is not None:
                    self.video_scaler.step(self.video_optimizer)
                    self.video_scaler.update()
                else:
                    self.video_optimizer.step()
                
                # Only step text optimizer if there are text parameters to update
                if text_params:
                    if self.text_scaler is not None and self.text_scaler != self.video_scaler:
                        self.text_scaler.step(self.text_optimizer)
                        self.text_scaler.update()
                    elif self.text_scaler is None:
                        self.text_optimizer.step()
                
                # Scheduler steps (if per-iteration)
                if self.video_scheduler and self.scheduler_per_iteration:
                    self.video_scheduler.step()
                if self.text_scheduler and self.scheduler_per_iteration:
                    self.text_scheduler.step()
        else:
            # Original single optimizer backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Original optimizer step
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.video_encoder.parameters()) + list(self.text_encoder.parameters()),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.video_encoder.parameters()) + list(self.text_encoder.parameters()),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                if self.lr_scheduler and self.scheduler_per_iteration:
                    self.lr_scheduler.step()
        
        # Increment global step (only during training, not validation)
        self.global_step += 1
        self.step += 1
        
        # Log metrics periodically during training (every 100 steps)
        if self.global_step % 100 == 0 and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            periodic_metrics = {
                f"train/periodic_loss": loss.item() * self.config.gradient_accumulation_steps,
                f"train/periodic_alignment": alignment_score if isinstance(alignment_score, float) else alignment_score.item(),
                f"train/periodic_temp": temperature.item(),
            }
            self.wandb_wrapper.log(periodic_metrics, step=self.global_step)
        
        # Collect metrics
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "alignment_score": alignment_score if isinstance(alignment_score, float) else alignment_score.item(),
            "temperature": temperature.item(),
            "logit_scale": self.log_temp.item(),
        }
        
        # Add optimizer-specific metrics
        if self.use_two_optimizers:
            metrics["video_lr"] = self.video_optimizer.param_groups[0]["lr"]
            metrics["text_lr"] = self.text_optimizer.param_groups[0]["lr"]
            if 'video_grad_norm' in locals():
                metrics["video_grad_norm"] = video_grad_norm.item()
            if 'text_grad_norm' in locals():
                metrics["text_grad_norm"] = text_grad_norm.item()
        else:
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        
        # Add phase info if using phased training
        if hasattr(self, 'phase_info') and self.phase_info is not None:
            # Add phase info (only numeric values for metrics)
            for k, v in self.phase_info.items():
                if isinstance(v, (int, float, bool)):
                    metrics[f"phase/{k}"] = float(v)
            # Add freeze ratios for wandb logging
            metrics["phase/video_freeze_ratio"] = self.phase_info.get("video_freeze_ratio", 0.0)
            text_freeze_layers = self.phase_info.get("text_freeze_layers")
            if text_freeze_layers is not None:
                if text_freeze_layers == -1:  # All layers unfrozen
                    metrics["phase/text_freeze_layers"] = 12
                    metrics["phase/text_freeze_ratio"] = 1.0
                else:
                    metrics["phase/text_freeze_layers"] = text_freeze_layers
                    # Calculate text freeze ratio (12 total BERT layers)
                    text_freeze_ratio = text_freeze_layers / 12.0 if text_freeze_layers > 0 else 0.0
                    metrics["phase/text_freeze_ratio"] = text_freeze_ratio
            else:
                metrics["phase/text_freeze_layers"] = 0
                metrics["phase/text_freeze_ratio"] = 0.0
        
        embeddings = {
            "video_embeddings": video_embeddings.detach(),
            "text_embeddings": text_embeddings.detach(),
        }
        
        return metrics, embeddings
    
    def _train_step(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[Dict, Dict]:
        """Override base train step to use enhanced version."""
        if self.use_two_optimizers:
            return self._train_step_v2(videos, input_ids, attention_mask)
        else:
            return super()._train_step(videos, input_ids, attention_mask)
    
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False, is_highest_alignment: bool = False):
        """Save checkpoint with two-optimizer support."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Base model dict
        model_dict = {
            "video_encoder": self.video_encoder.module.state_dict()
            if hasattr(self.video_encoder, "module")
            else self.video_encoder.state_dict(),
            "text_encoder": self.text_encoder.module.state_dict()
            if hasattr(self.text_encoder, "module")
            else self.text_encoder.state_dict(),
            "epoch": epoch,
        }
        
        # Add projection heads if they exist
        if self.video_proj is not None:
            model_dict["video_proj"] = self.video_proj.module.state_dict() \
                if hasattr(self.video_proj, "module") else self.video_proj.state_dict()
        if self.text_proj is not None:
            model_dict["text_proj"] = self.text_proj.module.state_dict() \
                if hasattr(self.text_proj, "module") else self.text_proj.state_dict()
        
        # Add optimizer states
        if self.use_two_optimizers:
            model_dict["video_optimizer"] = self.video_optimizer.state_dict()
            model_dict["text_optimizer"] = self.text_optimizer.state_dict()
            if self.video_scheduler:
                model_dict["video_scheduler"] = self.video_scheduler.state_dict()
            if self.text_scheduler:
                model_dict["text_scheduler"] = self.text_scheduler.state_dict()
            if self.video_scaler:
                model_dict["video_scaler"] = self.video_scaler.state_dict()
            if self.text_scaler:
                model_dict["text_scaler"] = self.text_scaler.state_dict()
        else:
            model_dict["optimizer"] = self.optimizer.state_dict()
            if self.lr_scheduler:
                model_dict["scheduler"] = self.lr_scheduler.state_dict()
            if self.scaler:
                model_dict["scaler"] = self.scaler.state_dict()
        
        # Add training state
        model_dict["global_step"] = self.global_step
        model_dict["current_phase"] = self.current_phase
        
        # Create full checkpoint
        checkpoint = {
            **model_dict,
            **metrics,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "highest_alignment_score": self.highest_alignment_score,
            "highest_alignment_epoch": self.highest_alignment_epoch,
        }
        
        # Save checkpoint
        if is_best:
            save_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt")
        elif is_highest_alignment:
            save_path = os.path.join(checkpoint_dir, f"highest_alignment_epoch_{epoch}.pt")
        else:
            save_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        
        torch.save(checkpoint, save_path)
        print(f"\nSaved checkpoint to {save_path}")
    
    def train(self, start_epoch: int = 0, end_epoch: int = 10):
        """Enhanced training loop with two-optimizer support and proper wandb logging."""
        if self.use_two_optimizers:
            print("\n=== Two-Optimizer Training Mode ===")
            print(f"Video Optimizer: {type(self.video_optimizer).__name__}")
            print(f"Text Optimizer: {type(self.text_optimizer).__name__}")
            print(f"Phased Training: {self.phased_scheduler is not None}")
            print("===================================\n")
        
        # Import needed modules
        from utils.enums import RunMode
        from utils.ddp import DistributedUtils
        
        try:
            for epoch in range(start_epoch, end_epoch):
                # Update phase if using phased training (at epoch boundary)
                if self.phased_scheduler is not None:
                    self.phase_info = self.phased_scheduler.update_for_epoch(epoch)
                    if self.phase_info["phase_name"] != self.current_phase:
                        self.current_phase = self.phase_info["phase_name"]
                        if self.config.is_ref_device:
                            print(f"\n[Epoch {epoch+1}] Entering phase: {self.current_phase}")
                            print(f"  Text freeze layers: {self.phase_info['text_freeze_layers']}")
                            print(f"  Video freeze ratio: {self.phase_info['video_freeze_ratio']}")
                            print(f"  Logit scale trainable: {self.phase_info['logit_scale_trainable']}")
                else:
                    self.phase_info = None
                
                # Synchronize before epoch starts
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )
                
                # Apply temperature schedule
                scheduled_temp = self._compute_temperature_schedule(epoch)
                self.log_temp = torch.log(torch.tensor(scheduled_temp, device=self.device))
                if self.config.is_ref_device:
                    print(f"\n[Epoch {epoch+1}] Temperature scheduled: {scheduled_temp:.4f}")
                
                # Apply freeze ratio schedule for video encoder
                scheduled_video_freeze = self._compute_freeze_schedule(epoch)
                if hasattr(self.video_encoder, 'update_freeze_ratio'):
                    self.video_encoder.update_freeze_ratio(scheduled_video_freeze)
                    if self.config.is_ref_device:
                        print(f"[Epoch {epoch+1}] Video freeze ratio scheduled: {scheduled_video_freeze:.2f}")
                
                # Apply freeze ratio schedule for text encoder
                scheduled_text_freeze = self._compute_text_freeze_schedule(epoch)
                if hasattr(self.text_encoder, 'update_freeze_ratio'):
                    self.text_encoder.update_freeze_ratio(scheduled_text_freeze)
                    if self.config.is_ref_device:
                        print(f"[Epoch {epoch+1}] Text freeze ratio scheduled: {scheduled_text_freeze:.2f}")
                
                # Set epoch for both train and val samplers (ensures deterministic behavior)
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)
                if hasattr(self.val_loader.sampler, 'set_epoch'):
                    self.val_loader.sampler.set_epoch(epoch)  # Important for validation stability

                # Training phase
                train_metrics: dict[str, float] = self._run_epoch(mode=RunMode.TRAIN, epoch=epoch)
                
                # Add scheduled values to metrics
                train_metrics["train/scheduled_temperature"] = scheduled_temp
                train_metrics["train/scheduled_video_freeze_ratio"] = scheduled_video_freeze
                train_metrics["train/scheduled_text_freeze_ratio"] = scheduled_text_freeze
                
                # Log train metrics with explicit step
                if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                    # Use the actual global step counter which increments during training
                    train_wandb_step = self.global_step
                    self.wandb_wrapper.log(train_metrics, step=train_wandb_step)
                    print(f"[DEBUG] rank={self.device} => Logged train metrics to W&B at step {train_wandb_step}")

                # Sync before validation
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )

                # Validation phase
                val_metrics: dict[str, float] = self._run_epoch(
                    mode=RunMode.VALIDATE, 
                    epoch=epoch
                )
                
                # Sync after validation
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )

                # Handle checkpointing and best model tracking
                if self.config.is_ref_device:
                    # Debug: print available keys
                    if epoch == 0:  # Only print once
                        print(f"\n[DEBUG] Available validation metrics keys: {list(val_metrics.keys())}")
                    
                    # Check for both possible loss keys (main_loss from multitask, loss from single task)
                    current_val_loss = val_metrics.get("val/main_loss", val_metrics.get("val/loss", float("inf")))
                    if current_val_loss == float("inf"):
                        print(f"\n[WARNING] Validation loss not found in metrics! Keys available: {list(val_metrics.keys())}")
                        # Try without prefix
                        current_val_loss = val_metrics.get("loss", float("inf"))
                    
                    current_alignment_score = val_metrics.get("val/alignment_score", 0.0)
                    
                    # Check for best loss
                    is_best = current_val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = current_val_loss
                        self.best_epoch = epoch
                        print(f"\n🎯 New best validation loss: {self.best_val_loss:.4f} at epoch {epoch+1}")
                        
                    # Check for highest alignment score 
                    is_highest_alignment = current_alignment_score > self.highest_alignment_score
                    if is_highest_alignment:
                        self.highest_alignment_score = current_alignment_score
                        self.highest_alignment_epoch = epoch
                        print(f"\n🎯 New highest alignment score: {self.highest_alignment_score:.4f} at epoch {epoch+1}")
                    
                    # Save checkpoints
                    if self.config.save_best == "loss" and is_best:
                        self._save_checkpoint(epoch, val_metrics, is_best=True)
                    elif self.config.save_best == "alignment" and is_highest_alignment:
                        self._save_checkpoint(epoch, val_metrics, is_highest_alignment=True)
                    elif epoch % self.config.period == 0:
                        self._save_checkpoint(epoch, val_metrics)
                        
                    # Always save latest checkpoint
                    self._save_checkpoint(epoch, val_metrics)
                            
                # Log validation metrics with explicit step
                if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                    # Don't log inf values to wandb
                    if self.best_val_loss != float("inf"):
                        val_metrics['best_val_loss'] = self.best_val_loss
                    else:
                        val_metrics['best_val_loss'] = current_val_loss  # Use current if no best yet
                    val_metrics['highest_alignment_score'] = self.highest_alignment_score
                    # Use current global step for validation (after training epoch)
                    val_wandb_step = self.global_step
                    self.wandb_wrapper.log(val_metrics, step=val_wandb_step)
                    print(f"[DEBUG] rank={self.device} => Logged val metrics to W&B at step {val_wandb_step}")
                
                # Sync after logging
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )

                # LR Scheduling (if not per-iteration)
                if not self.scheduler_per_iteration:
                    if self.use_two_optimizers:
                        if self.video_scheduler:
                            self.video_scheduler.step()
                        if self.text_scheduler:
                            self.text_scheduler.step()
                    else:
                        if self.lr_scheduler:
                            self.lr_scheduler.step()

                # Sync at end of epoch
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )
                
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            if self.config.is_ref_device:
                self._save_checkpoint(epoch, {"interrupted": True})
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            raise