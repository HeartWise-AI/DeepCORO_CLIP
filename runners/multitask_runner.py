import os
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Dict, Tuple, List, Any, Optional

from tqdm import tqdm

from utils.enums import RunMode
from utils.ddp import DistributedUtils
from utils.config.clip_config import ClipConfig
from utils.registry import RunnerRegistry
from utils.retrieval_metrics import (
    compute_mrr,
    compute_map,
    compute_ndcg_at_k,
    compute_median_rank,
    compute_recall_at_k,
    compute_embedding_norms,
    compute_alignment_score,
)
from utils.wandb_logger import (
    log_best_worst_retrievals,
    log_gradient_norms,
    save_retrieval_results,
)
import wandb
from utils.loss.typing import Loss
from utils.wandb_wrapper import WandbWrapper
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from models.captioning_decoder import CaptioningDecoder
from models.masked_video_modeling import MaskedVideoModeling
from dataloaders.video_clip_dataset import VideoClipDataset
import itertools
from torch.nn.utils import clip_grad_norm_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@RunnerRegistry.register("DeepCORO_multitask")
class MultitaskRunner:
    """
    Multitask runner for DeepCORO-CLIP with captioning and masked video modeling.
    
    This runner handles:
    - Contrastive learning (video ↔ text)
    - Captioning (autoregressive report generation)
    - Masked video modeling (self-supervised learning)
    """
    
    def __init__(
        self,
        config: ClipConfig = None,
        wandb_wrapper: WandbWrapper = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        video_encoder: VideoEncoder = None,
        text_encoder: TextEncoder = None,
        captioning_decoder: CaptioningDecoder = None,
        masked_video_modeling: MaskedVideoModeling = None,
        optimizer: Optimizer = None,
        scaler: GradScaler = None,
        log_temp: torch.Tensor = None,
        lr_scheduler: LRScheduler = None,
        loss_fn: Loss = None,
        output_dir: str = None,
    ):
        """
        Initialize the multitask runner.
        
        Args:
            config: ClipConfig object with run/training configuration
            wandb_wrapper: WandbWrapper instance
            train_loader: DataLoader for training dataset
            val_loader: DataLoader for validation dataset
            video_encoder: VideoEncoder model
            text_encoder: TextEncoder model
            captioning_decoder: CaptioningDecoder model
            masked_video_modeling: MaskedVideoModeling model
            optimizer: Optimizer instance
            scaler: GradScaler for automatic mixed precision
            log_temp: Logarithm of temperature used in contrastive loss
            lr_scheduler: Learning rate scheduler
            loss_fn: Multitask loss function
            output_dir: Directory where checkpoints and outputs will be saved
        """
        if not isinstance(config.recall_k, list):
            raise ValueError(
                f"config.recall_k must be a list of ints, got {type(config.recall_k)}"
            )
        if not isinstance(config.ndcg_k, list):
            raise ValueError(
                f"config.ndcg_k must be a list of ints, got {type(config.ndcg_k)}"
            )

        self.config: ClipConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        self.device: int = config.device
        self.world_size: int = config.world_size
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.video_encoder: VideoEncoder = video_encoder
        self.text_encoder: TextEncoder = text_encoder
        self.captioning_decoder: CaptioningDecoder = captioning_decoder
        self.masked_video_modeling: MaskedVideoModeling = masked_video_modeling
        self.optimizer: Optimizer = optimizer
        self.scaler: GradScaler = scaler
        self.log_temp: torch.Tensor = log_temp
        self.lr_scheduler: LRScheduler = lr_scheduler
        self.loss_fn: Loss = loss_fn
        self.output_dir: str = output_dir
        
        # Get tokenizer from captioning decoder
        if self.captioning_decoder is not None:
            self.tokenizer = self.captioning_decoder.module.tokenizer
        else:
            self.tokenizer = None
        
        # Training state
        self.best_loss = float('inf')
        self.best_alignment = 0.0
        self.current_epoch = 0
        self.global_step = 0  # Global step counter for consistent wandb logging
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        
        # Loss weight scheduler (optional)
        self.loss_weight_scheduler = None
        if hasattr(config, 'use_loss_weight_scheduler') and config.use_loss_weight_scheduler:
            from utils.loss.multitask_loss import LossWeightScheduler
            self.loss_weight_scheduler = LossWeightScheduler(
                initial_weights=getattr(config, 'initial_loss_weights', {
                    'contrastive': 1.0,
                    'captioning': 0.5,
                    'masked_modeling': 0.1,
                }),
                final_weights=getattr(config, 'final_loss_weights', {
                    'contrastive': 1.0,
                    'captioning': 1.0,
                    'masked_modeling': 0.1,
                }),
                warmup_steps=getattr(config, 'loss_warmup_steps', 1000),
                total_steps=getattr(config, 'loss_total_steps', 10000),
                schedule_type=getattr(config, 'loss_schedule_type', 'linear'),
            )
    
    def _scheduler_is_per_iteration(self) -> bool:
        """Check if the scheduler is per-iteration or per-epoch."""
        scheduler_name = self.config.scheduler_name.lower()
        return scheduler_name in ['cosine', 'linear', 'exponential', 'step']
    
    def train(
        self, 
        start_epoch: int = 0, 
        end_epoch: int = 10,
    ):
        """
        Train the multitask model.
        
        Args:
            start_epoch: Starting epoch
            end_epoch: Ending epoch
        """
        print(f"Starting training from epoch {start_epoch} to {end_epoch}")
        
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._run_epoch("train", epoch)
            
            # Validation phase
            val_metrics = self._run_epoch("val", epoch)
            
            # Track best epoch
            if val_metrics.get("total_loss", float("inf")) < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                self.best_epoch = epoch
                if self.config.is_ref_device:
                    print(f"New best model at epoch {epoch} with val loss: {self.best_val_loss:.4f}")
            
            # Update learning rate
            if self.lr_scheduler is not None:
                if self._scheduler_is_per_iteration():
                    # Scheduler is updated per iteration, so we don't update here
                    pass
                else:
                    self.lr_scheduler.step()
            
            # Log metrics
            if self.config.is_ref_device:
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if self.config.is_ref_device:
                self._save_checkpoint(epoch, val_metrics)
    
    def _run_epoch(
        self, 
        mode: str, 
        epoch: int
    ) -> dict[str, float]:
        """
        Run a single epoch.
        
        Args:
            mode: "train" or "val"
            epoch: Current epoch number
            
        Returns:
            Dictionary of metrics
        """
        if mode == "train":
            self.video_encoder.train()
            self.text_encoder.train()
            self.captioning_decoder.train()
            self.masked_video_modeling.train()
            dataloader = self.train_loader
        else:
            self.video_encoder.eval()
            self.text_encoder.eval()
            self.captioning_decoder.eval()
            self.masked_video_modeling.eval()
            dataloader = self.val_loader
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_captioning_loss = 0.0
        total_masked_modeling_loss = 0.0
        num_batches = 0
        
        # For contrastive learning metrics
        all_video_features = []
        all_text_features = []
        all_texts = []
        all_paths = []  # Store paths for best/worst retrieval logging
        
        # For captioning metrics
        all_generated_texts = []
        all_target_texts = []
        
        with tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Preprocess inputs
                processed_batch, texts = self._preprocess_inputs(batch)
                videos = processed_batch["videos"].to(self.device)
                input_ids = processed_batch["input_ids"].to(self.device)
                attention_mask = processed_batch["attention_mask"].to(self.device)
                
                if mode == "train":
                    # Training step
                    batch_metrics, batch_outputs = self._train_step(
                        videos, input_ids, attention_mask, texts, batch
                    )
                    
                    # Log step-wise metrics to W&B
                    if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                        self.global_step += 1  # Increment global step counter
                        step_log_dict = {
                            "train/step_loss": batch_metrics["total_loss"],
                            "train/step_contrastive_loss": batch_metrics.get("contrastive_loss", 0.0),
                            "train/step_captioning_loss": batch_metrics.get("captioning_loss", 0.0),
                            "train/step_masked_modeling_loss": batch_metrics.get("masked_modeling_loss", 0.0),
                            "train/grad_norm": batch_metrics.get("grad_norm", 0.0),
                            "train/step": self.global_step,
                            "train/epoch": epoch,
                        }
                        
                        # Log learning rates for each param group with clear labels
                        if self.optimizer is not None:
                            for i, param_group in enumerate(self.optimizer.param_groups):
                                # Use the name if available, otherwise use index
                                group_name = param_group.get('name', f'group_{i}')
                                step_log_dict[f"lr/{group_name}"] = param_group['lr']
                        
                        # Log temperature
                        if self.log_temp is not None:
                            step_log_dict["train/temperature"] = torch.exp(self.log_temp).item()
                        
                        # Log loss weight if using scheduler
                        if self.loss_weight_scheduler is not None:
                            current_weights = self.loss_weight_scheduler.get_weights(self.global_step)
                            for key, value in current_weights.items():
                                step_log_dict[f"loss_weight/{key}"] = value
                        
                        self.wandb_wrapper.log(step_log_dict, step=self.global_step)
                else:
                    # Validation step
                    batch_metrics, batch_outputs = self._val_step(
                        videos, input_ids, attention_mask, texts, batch
                    )
                    
                    # Log validation step metrics to W&B
                    if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                        # Don't increment global step during validation, just use current value
                        val_step_log_dict = {
                            "val/step_loss": batch_metrics["total_loss"],
                            "val/step_contrastive_loss": batch_metrics.get("contrastive_loss", 0.0),
                            "val/step_captioning_loss": batch_metrics.get("captioning_loss", 0.0),
                            "val/step_masked_modeling_loss": batch_metrics.get("masked_modeling_loss", 0.0),
                            "val/step": self.global_step,
                            "val/epoch": epoch,
                        }
                        self.wandb_wrapper.log(val_step_log_dict, step=self.global_step)
                
                # Accumulate metrics
                total_loss += batch_metrics["total_loss"]
                total_contrastive_loss += batch_metrics.get("contrastive_loss", 0.0)
                total_captioning_loss += batch_metrics.get("captioning_loss", 0.0)
                total_masked_modeling_loss += batch_metrics.get("masked_modeling_loss", 0.0)
                num_batches += 1
                
                # Store features for metrics
                if "video_features" in batch_outputs:
                    all_video_features.append(batch_outputs["video_features"].detach())
                if "text_features" in batch_outputs:
                    all_text_features.append(batch_outputs["text_features"].detach())
                if "texts" in batch_outputs:
                    all_texts.extend(batch_outputs["texts"])
                if "paths" in batch_outputs:
                    all_paths.extend(batch_outputs["paths"])
                
                # Store captioning outputs
                if "generated_texts" in batch_outputs:
                    all_generated_texts.extend(batch_outputs["generated_texts"])
                if "target_texts" in batch_outputs:
                    all_target_texts.extend(batch_outputs["target_texts"])
                
                # Sync after each validation batch
                if mode == "val":
                    DistributedUtils.sync_process_group(
                        world_size=self.world_size,
                        device_ids=self.device
                    )
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{batch_metrics['total_loss']:.4f}",
                    'contrastive': f"{batch_metrics.get('contrastive_loss', 0.0):.4f}",
                    'captioning': f"{batch_metrics.get('captioning_loss', 0.0):.4f}",
                    'mvm': f"{batch_metrics.get('masked_modeling_loss', 0.0):.4f}",
                })
        
        # Gather losses across GPUs
        if self.world_size > 1:
            # Gather loss sums from all GPUs
            total_loss_gathered = DistributedUtils.gather_loss([total_loss], self.device)
            total_contrastive_gathered = DistributedUtils.gather_loss([total_contrastive_loss], self.device)
            total_captioning_gathered = DistributedUtils.gather_loss([total_captioning_loss], self.device)
            total_mvm_gathered = DistributedUtils.gather_loss([total_masked_modeling_loss], self.device)
            
            # Gather batch counts
            batch_count_tensor = torch.tensor([num_batches], dtype=torch.float32, device=self.device)
            batch_counts = [torch.zeros_like(batch_count_tensor) for _ in range(self.world_size)]
            dist.all_gather(batch_counts, batch_count_tensor)
            total_batches = sum(b.item() for b in batch_counts)
            
            # Compute average metrics across all GPUs
            metrics = {
                "total_loss": total_loss_gathered / total_batches,
                "contrastive_loss": total_contrastive_gathered / total_batches,
                "captioning_loss": total_captioning_gathered / total_batches,
                "masked_modeling_loss": total_mvm_gathered / total_batches,
            }
        else:
            # Single GPU - compute averages normally
            metrics = {
                "total_loss": total_loss / num_batches,
                "contrastive_loss": total_contrastive_loss / num_batches,
                "captioning_loss": total_captioning_loss / num_batches,
                "masked_modeling_loss": total_masked_modeling_loss / num_batches,
            }
        
        # Gather all predictions and compute metrics
        if all_video_features and all_text_features:
            # Gather distributed predictions
            (video_features, text_features, 
             gathered_texts, gathered_paths, 
             gathered_generated, gathered_targets) = self._gather_distributed_predictions(
                all_video_features, all_text_features,
                all_texts, all_paths,
                all_generated_texts, all_target_texts
            )
            
            # Compute metrics on rank 0
            if self.config.is_ref_device:
                # Compute contrastive metrics
                contrastive_metrics = self._compute_contrastive_metrics(
                    video_features, text_features, gathered_texts, gathered_paths, epoch
                )
                metrics.update(contrastive_metrics)
                
                # Compute captioning metrics if available
                if gathered_generated and gathered_targets:
                    captioning_metrics = self._compute_captioning_metrics(
                        gathered_generated, gathered_targets, epoch
                    )
                    metrics.update(captioning_metrics)
                
                # Save predictions for validation
                if mode == "val":
                    self._save_predictions(
                        epoch=epoch,
                        video_paths=gathered_paths,
                        texts=gathered_texts,
                        generated_texts=gathered_generated,
                        video_features=video_features,
                        text_features=text_features,
                    )
        
        # Final sync after metrics computation
        DistributedUtils.sync_process_group(
            world_size=self.world_size,
            device_ids=self.device
        )
        
        # Print validation metrics to console
        if mode == "val" and self.config.is_ref_device:
            print(f"\n{'='*60}")
            print(f"Validation Metrics - Epoch {epoch}")
            print(f"{'='*60}")
            print(f"Loss: {metrics.get('total_loss', 0.0):.4f}")
            print(f"  - Contrastive: {metrics.get('contrastive_loss', 0.0):.4f}")
            print(f"  - Captioning: {metrics.get('captioning_loss', 0.0):.4f}")
            print(f"  - MVM: {metrics.get('masked_modeling_loss', 0.0):.4f}")
            
            if 'Recall@1' in metrics:
                print(f"\nRetrieval Metrics:")
                for k in [1, 5, 10, 50]:
                    recall_key = f"Recall@{k}"
                    if recall_key in metrics:
                        print(f"  {recall_key}: {metrics[recall_key]:.4f}")
                
                if 'MRR_V2T' in metrics:
                    print(f"  MRR: {metrics['MRR_V2T']:.4f}")
                if 'NDCG@5_V2T' in metrics:
                    print(f"  NDCG@5: {metrics['NDCG@5_V2T']:.4f}")
                if 'alignment_score' in metrics:
                    print(f"  Alignment Score: {metrics['alignment_score']:.4f}")
                if 'median_rank' in metrics:
                    print(f"  Median Rank: {metrics['median_rank']}")
            
            if 'BLEU' in metrics:
                print(f"\nCaptioning Metrics:")
                print(f"  BLEU: {metrics.get('BLEU', 0.0):.4f}")
                print(f"  ROUGE-1: {metrics.get('ROUGE-1', 0.0):.4f}")
                print(f"  ROUGE-2: {metrics.get('ROUGE-2', 0.0):.4f}")
                print(f"  ROUGE-L: {metrics.get('ROUGE-L', 0.0):.4f}")
                if 'METEOR' in metrics:
                    print(f"  METEOR: {metrics['METEOR']:.4f}")
            print(f"{'='*60}\n")
        
        return metrics
    
    def _train_step(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: list[str],
        batch: dict,
    ) -> tuple[dict, dict]:
        """
        Single training step.
        
        Args:
            videos: Video tensors [batch_size, num_frames, height, width, channels]
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            texts: List of text strings
            
        Returns:
            Tuple of (metrics, outputs)
        """
        self.optimizer.zero_grad()
        
        # Get video features (token-level for captioning and masked modeling)
        # Handle DistributedDataParallel wrapper
        if hasattr(self.video_encoder, 'module'):
            video_tokens = self.video_encoder.module.get_tokens(videos, mode="patch")  # [B, num_tokens, hidden_size]
        else:
            video_tokens = self.video_encoder.get_tokens(videos, mode="patch")  # [B, num_tokens, hidden_size]
        
        # Get aggregated video features for contrastive learning
        if hasattr(self.video_encoder, 'module'):
            video_features = self.video_encoder.module.get_tokens(videos, mode="video")  # [B, hidden_size]
        else:
            video_features = self.video_encoder.get_tokens(videos, mode="video")  # [B, hidden_size]
        
        # Get text features
        text_features = self.text_encoder(input_ids, attention_mask)  # [B, hidden_size]
        
        # Captioning: prepare targets (shift input_ids for autoregressive training)
        caption_targets = input_ids.clone()
        caption_input_ids = input_ids[:, :-1].contiguous()
        caption_targets = caption_targets[:, 1:].contiguous()
        
        # Captioning forward pass
        caption_outputs = self.captioning_decoder(
            input_ids=caption_input_ids,
            attention_mask=attention_mask[:, :-1],
            video_features=video_tokens,
        )
        caption_logits = caption_outputs["logits"]  # [B, seq_len-1, vocab_size]
        
        # Masked video modeling forward pass
        mvm_outputs = self.masked_video_modeling(video_tokens)
        masked_pred = mvm_outputs["pred"]
        masked_target = video_tokens
        masked_mask = mvm_outputs["mask"]
        
        # Update loss weights if using scheduler
        if self.loss_weight_scheduler is not None:
            current_weights = self.loss_weight_scheduler.get_weights(self.global_step)
            self.loss_fn.loss_type.loss_weights = current_weights
        
        # Compute multitask loss
        loss_outputs = self.loss_fn.run(
            video_features=video_features,
            text_features=text_features,
            caption_logits=caption_logits,
            caption_targets=caption_targets,
            masked_pred=masked_pred,
            masked_target=masked_target,
            masked_mask=masked_mask,
            log_temp=self.log_temp,
        )
        
        total_loss = loss_outputs["total"]
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()
        
        # Gradient clipping and norm logging
        grad_norm = None
        if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
            # Get all parameters from all param groups
            all_params = []
            for group in self.optimizer.param_groups:
                all_params.extend(group['params'])
            total_norm = clip_grad_norm_(all_params, self.config.max_grad_norm)
            grad_norm = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Update learning rate if per-iteration scheduler
        if self.lr_scheduler is not None and self._scheduler_is_per_iteration():
            self.lr_scheduler.step()
        
        # Prepare metrics
        metrics = {
            "total_loss": total_loss.item(),
            "contrastive_loss": loss_outputs.get("contrastive", torch.tensor(0.0)).item(),
            "captioning_loss": loss_outputs.get("captioning", torch.tensor(0.0)).item(),
            "masked_modeling_loss": loss_outputs.get("masked_modeling", torch.tensor(0.0)).item(),
        }
        
        # Add gradient norm if calculated
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        
        # Prepare outputs
        outputs = {
            "video_features": video_features,
            "text_features": text_features,
            "texts": texts,
            "paths": batch.get("paths", batch.get("sids", [])),  # Get paths or SIDs from batch
        }
        
        return metrics, outputs
    
    def _val_step(
        self, 
        videos: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        texts: list[str],
        batch: dict,
    ) -> tuple[dict, dict]:
        """
        Single validation step.
        
        Args:
            videos: Video tensors [batch_size, num_frames, height, width, channels]
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            texts: List of text strings
            
        Returns:
            Tuple of (metrics, outputs)
        """
        with torch.no_grad():
            # Get video features
            if hasattr(self.video_encoder, 'module'):
                video_tokens = self.video_encoder.module.get_tokens(videos, mode="patch")
                video_features = self.video_encoder.module.get_tokens(videos, mode="video")
            else:
                video_tokens = self.video_encoder.get_tokens(videos, mode="patch")
                video_features = self.video_encoder.get_tokens(videos, mode="video")
            
            # Get text features
            text_features = self.text_encoder(input_ids, attention_mask)
            
            # Captioning: prepare targets
            caption_targets = input_ids.clone()
            caption_input_ids = input_ids[:, :-1].contiguous()
            caption_targets = caption_targets[:, 1:].contiguous()
            
            # Captioning forward pass
            caption_outputs = self.captioning_decoder(
                input_ids=caption_input_ids,
                attention_mask=attention_mask[:, :-1],
                video_features=video_tokens,
            )
            caption_logits = caption_outputs["logits"]
            
            # Generate captions for evaluation
            # Handle DistributedDataParallel wrapper
            if hasattr(self.captioning_decoder, 'module'):
                generated_ids = self.captioning_decoder.module.generate(
                    video_features=video_tokens,
                    max_length=getattr(self.config, 'max_generation_length', 128),
                    do_sample=getattr(self.config, 'captioning_do_sample', False),
                    temperature=getattr(self.config, 'captioning_temperature', 1.0),
                )
            else:
                generated_ids = self.captioning_decoder.generate(
                    video_features=video_tokens,
                    max_length=getattr(self.config, 'max_generation_length', 128),
                    do_sample=getattr(self.config, 'captioning_do_sample', False),
                    temperature=getattr(self.config, 'captioning_temperature', 1.0),
                )
            
            # Decode generated texts
            generated_texts = []
            target_texts = []
            if self.tokenizer is not None:
                for i in range(len(generated_ids)):
                    generated_text = self.tokenizer.decode(
                        generated_ids[i], 
                        skip_special_tokens=True
                    )
                    target_text = self.tokenizer.decode(
                        input_ids[i], 
                        skip_special_tokens=True
                    )
                    generated_texts.append(generated_text)
                    target_texts.append(target_text)
            
            # Masked video modeling forward pass
            mvm_outputs = self.masked_video_modeling(video_tokens)
            masked_pred = mvm_outputs["pred"]
            masked_target = video_tokens
            masked_mask = mvm_outputs["mask"]
            
            # Compute multitask loss
            loss_outputs = self.loss_fn.run(
                video_features=video_features,
                text_features=text_features,
                caption_logits=caption_logits,
                caption_targets=caption_targets,
                masked_pred=masked_pred,
                masked_target=masked_target,
                masked_mask=masked_mask,
                log_temp=self.log_temp,
            )
            
            total_loss = loss_outputs["total"]
            
            # Prepare metrics
            metrics = {
                "total_loss": total_loss.item(),
                "contrastive_loss": loss_outputs.get("contrastive", torch.tensor(0.0)).item(),
                "captioning_loss": loss_outputs.get("captioning", torch.tensor(0.0)).item(),
                "masked_modeling_loss": loss_outputs.get("masked_modeling", torch.tensor(0.0)).item(),
            }
            
            # Prepare outputs
            outputs = {
                "video_features": video_features,
                "text_features": text_features,
                "texts": texts,
                "generated_texts": generated_texts,
                "target_texts": target_texts,
                "paths": batch.get("paths", batch.get("sids", [])),  # Get paths or SIDs from batch
            }
            
            return metrics, outputs
    
    def _compute_contrastive_metrics(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        texts: list[str],
        paths: list[str] = None,
        epoch: int = 0,
    ) -> dict[str, float]:
        """Compute contrastive learning metrics."""
        metrics = {}
        
        # Compute retrieval metrics
        if len(video_features) > 0 and len(text_features) > 0:
            # Normalize features
            video_features_norm = torch.nn.functional.normalize(video_features, dim=1)
            text_features_norm = torch.nn.functional.normalize(text_features, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(video_features_norm, text_features_norm.t())
            
            # Get unique texts and create mapping
            unique_texts = list(dict.fromkeys(texts))  # Preserve order while removing duplicates
            text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
            
            # Create ground truth indices for unique texts
            ground_truth_indices = torch.tensor([text_to_idx[text] for text in texts], 
                                               device=similarity_matrix.device)
            
            if len(unique_texts) < len(texts):
                # Remap similarity matrix to unique texts
                unique_text_features = []
                for text in unique_texts:
                    idx = texts.index(text)
                    unique_text_features.append(text_features_norm[idx])
                unique_text_features = torch.stack(unique_text_features)
                similarity_matrix = torch.matmul(video_features_norm, unique_text_features.t())
            
            # Compute retrieval metrics with proper ground truth indices
            recall_metrics = compute_recall_at_k(similarity_matrix, ground_truth_indices, self.config.recall_k)
            metrics.update(recall_metrics)
            
            # Compute NDCG metrics
            ndcg_metrics = compute_ndcg_at_k(similarity_matrix, ground_truth_indices, self.config.ndcg_k)
            for metric_name, value in ndcg_metrics.items():
                metrics[metric_name] = value
            
            # Compute alignment score
            alignment_score = compute_alignment_score(video_features_norm, text_features_norm)
            metrics["alignment_score"] = alignment_score
            
            # Compute embedding norms
            embedding_norms = compute_embedding_norms(video_features_norm, text_features_norm)
            metrics.update(embedding_norms)
            
            # Compute median rank
            median_rank = compute_median_rank(similarity_matrix, ground_truth_indices)
            metrics["median_rank"] = median_rank
            
            # Compute MRR
            mrr_dict = compute_mrr(similarity_matrix, ground_truth_indices)
            metrics.update(mrr_dict)
            
            # Log best/worst retrievals if wandb is initialized and we have paths
            if self.wandb_wrapper.is_initialized() and paths is not None and self.val_loader is not None:
                try:
                    # Get the dataset object
                    dataset = self.val_loader.dataset
                    log_best_worst_retrievals(
                        similarity_matrix=similarity_matrix,
                        all_paths=paths,
                        unique_texts=unique_texts,
                        ground_truth_indices=ground_truth_indices,
                        epoch=self.global_step,  # Use global_step instead of epoch
                        wandb_wrapper=self.wandb_wrapper,
                        dataset_obj=dataset
                    )
                except Exception as e:
                    print(f"Warning: Failed to log best/worst retrievals: {e}")
        
        return metrics
    
    def _compute_captioning_metrics(
        self,
        generated_texts: list[str],
        target_texts: list[str],
        epoch: int,
    ) -> dict[str, float]:
        """Compute captioning metrics including BLEU, ROUGE, and METEOR."""
        metrics = {}
        
        if not generated_texts or not target_texts:
            return metrics
        
        try:
            # Import metrics libraries
            from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
            from nltk.translate.meteor_score import meteor_score
            from rouge_score import rouge_scorer
            import nltk
            
            # Ensure NLTK data is downloaded
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            except:
                pass
            
            # Calculate BLEU scores
            bleu_scores = []
            for gen, tgt in zip(generated_texts, target_texts):
                reference = [tgt.split()]
                hypothesis = gen.split()
                bleu_scores.append(sentence_bleu(reference, hypothesis))
            
            metrics["BLEU"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for gen, tgt in zip(generated_texts, target_texts):
                scores = scorer.score(tgt, gen)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            metrics["ROUGE-1"] = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
            metrics["ROUGE-2"] = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
            metrics["ROUGE-L"] = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
            
            # Calculate METEOR scores
            meteor_scores = []
            for gen, tgt in zip(generated_texts, target_texts):
                # METEOR requires word tokenization
                reference = tgt.split()
                hypothesis = gen.split()
                try:
                    score = meteor_score([reference], hypothesis)
                    meteor_scores.append(score)
                except:
                    # If METEOR fails, skip this pair
                    pass
            
            if meteor_scores:
                metrics["METEOR"] = sum(meteor_scores) / len(meteor_scores)
            
            # Log best, random, and worst caption examples to W&B
            if self.wandb_wrapper.is_initialized() and len(generated_texts) >= 5:
                # Calculate BLEU scores for all samples to identify best/worst
                sample_scores = bleu_scores if bleu_scores else [0.0] * len(generated_texts)
                
                # Get indices for best, worst, and random samples
                sorted_indices = sorted(range(len(sample_scores)), key=lambda i: sample_scores[i], reverse=True)
                best_indices = sorted_indices[:5]
                worst_indices = sorted_indices[-5:]
                
                # Get random 5 samples
                import random
                random_indices = random.sample(range(len(generated_texts)), min(5, len(generated_texts)))
                
                # Log best examples
                best_html = "<h3>Best Generated Captions (Top 5 BLEU)</h3>"
                for i, idx in enumerate(best_indices):
                    best_html += f"<p><b>Sample {i+1} (BLEU: {sample_scores[idx]:.3f}):</b><br>"
                    best_html += f"<b>Generated:</b> {generated_texts[idx]}<br>"
                    best_html += f"<b>Target:</b> {target_texts[idx]}</p>"
                
                # Log worst examples
                worst_html = "<h3>Worst Generated Captions (Bottom 5 BLEU)</h3>"
                for i, idx in enumerate(worst_indices):
                    worst_html += f"<p><b>Sample {i+1} (BLEU: {sample_scores[idx]:.3f}):</b><br>"
                    worst_html += f"<b>Generated:</b> {generated_texts[idx]}<br>"
                    worst_html += f"<b>Target:</b> {target_texts[idx]}</p>"
                
                # Log random examples
                random_html = "<h3>Random Generated Captions (5 samples)</h3>"
                for i, idx in enumerate(random_indices):
                    random_html += f"<p><b>Sample {i+1} (BLEU: {sample_scores[idx]:.3f}):</b><br>"
                    random_html += f"<b>Generated:</b> {generated_texts[idx]}<br>"
                    random_html += f"<b>Target:</b> {target_texts[idx]}</p>"
                
                self.wandb_wrapper.log({
                    "captioning/best_examples": wandb.Html(best_html),
                    "captioning/worst_examples": wandb.Html(worst_html),
                    "captioning/random_examples": wandb.Html(random_html),
                    "epoch": epoch,
                }, step=self.global_step)
            
        except ImportError as e:
            print(f"Warning: Could not compute caption metrics due to missing dependencies: {e}")
            print("Install with: pip install nltk rouge-score")
            metrics["num_generated"] = len(generated_texts)
            metrics["num_targets"] = len(target_texts)
        
        return metrics
    
    def _preprocess_inputs(self, batch: dict) -> tuple[dict, list[str]]:
        """Preprocess inputs for the model."""
        videos = batch["videos"]
        # Get raw texts from reports if available, otherwise use texts
        texts = batch.get("reports", batch.get("texts", None))
        if texts is None and "encoded_texts" in batch and batch["encoded_texts"] is not None:
            # If we only have encoded texts, we'll need to decode them or use placeholder
            texts = [""] * len(videos)
        
        # Tokenize texts
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=getattr(self.config, 'max_text_length', 512),
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
        else:
            # Fallback to existing text processing
            input_ids = batch.get("input_ids", torch.zeros(len(texts), 1))
            attention_mask = batch.get("attention_mask", torch.ones(len(texts), 1))
        
        processed_batch = {
            "videos": videos,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        return processed_batch, texts
    
    def _gather_tensor_along_batch(self, local_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        """Gather tensor from all GPUs."""
        if world_size == 1:
            return local_tensor
        
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, local_tensor)
        return torch.cat(gathered_tensors, dim=0)
    
    def _gather_strings_across_gpus(
        self, local_strings: list[str], world_size: int, device: torch.device
    ) -> list[str]:
        """Gather strings from all GPUs using all_gather_object."""
        if world_size == 1:
            return local_strings
        
        # Use all_gather_object to properly gather strings from all ranks
        all_strings_list = [None] * world_size
        torch.distributed.all_gather_object(all_strings_list, local_strings)
        
        # Flatten the list of lists
        all_strings = []
        for strings in all_strings_list:
            if strings is not None:
                all_strings.extend(strings)
        
        return all_strings
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics for the epoch."""
        if self.wandb_wrapper.is_initialized():
            # Log training metrics
            train_log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            train_log_dict["epoch"] = epoch
            
            # Log validation metrics
            val_log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
            
            # Log learning rate with clear labels
            if self.optimizer is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    # Use the name if available, otherwise use index
                    group_name = param_group.get('name', f'group_{i}')
                    train_log_dict[f"lr/{group_name}"] = param_group['lr']
            
            # Log temperature
            if self.log_temp is not None:
                train_log_dict["train/temperature"] = torch.exp(self.log_temp).item()
            
            # Combine and log
            log_dict = {**train_log_dict, **val_log_dict}
            self.wandb_wrapper.log(log_dict, step=self.global_step)
    
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False, is_highest_alignment: bool = False):
        """Save checkpoint."""
        if self.output_dir is None:
            return
        
        checkpoint = {
            "epoch": epoch,
            "video_encoder": self.video_encoder.module.state_dict(),
            "text_encoder": self.text_encoder.module.state_dict(),
            "captioning_decoder": self.captioning_decoder.module.state_dict(),
            "masked_video_modeling": self.masked_video_modeling.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "train/temperature": self.log_temp.data.clone(),
            "metrics": metrics,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.output_dir, "checkpoints", "latest.pt")
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoints", "best.pt")
            torch.save(checkpoint, best_path)
        
        # Save highest alignment checkpoint
        if is_highest_alignment:
            alignment_path = os.path.join(self.output_dir, "checkpoints", "best_alignment.pt")
            torch.save(checkpoint, alignment_path)
    
    def validate(self):
        """Run validation."""
        val_metrics = self._run_epoch("val", 0)
        return val_metrics
    
    def inference(self):
        """Run inference."""
        # TODO: Implement inference logic
        pass
    
    def _save_predictions(
        self,
        epoch: int,
        video_paths: List[str],
        texts: List[str],
        generated_texts: List[str],
        video_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> None:
        """
        Save validation predictions to CSV file with top-5 retrieval results.
        
        Args:
            epoch: Current epoch number
            video_paths: List of video file paths
            texts: List of ground truth texts
            generated_texts: List of generated captions
            video_features: Video feature embeddings [N, D]
            text_features: Text feature embeddings [M, D]
        """
        if not self.config.is_ref_device:
            return
        
        # Create predictions directory
        predictions_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Normalize features for similarity computation
        video_features_norm = torch.nn.functional.normalize(video_features, dim=1)
        text_features_norm = torch.nn.functional.normalize(text_features, dim=1)
        
        # Compute similarity matrix [N_videos, N_texts]
        similarity_matrix = torch.matmul(video_features_norm, text_features_norm.t())
        
        # Get unique texts for retrieval
        unique_texts = list(dict.fromkeys(texts))  # Preserve order while removing duplicates
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
        
        # If we have duplicates, remap similarity matrix
        if len(unique_texts) < len(texts):
            unique_text_features = []
            for text in unique_texts:
                idx = texts.index(text)
                unique_text_features.append(text_features_norm[idx])
            unique_text_features = torch.stack(unique_text_features)
            similarity_matrix = torch.matmul(video_features_norm, unique_text_features.t())
        
        # Get top-5 retrievals for each video
        topk_values, topk_indices = torch.topk(similarity_matrix, k=min(5, len(unique_texts)), dim=1)
        
        # Prepare data for CSV
        predictions_data = []
        for i, video_path in enumerate(video_paths):
            row = {
                'epoch': epoch,
                'video_path': video_path,
                'ground_truth_text': texts[i] if i < len(texts) else "",
                'predicted_text': generated_texts[i] if i < len(generated_texts) else "",
            }
            
            # Add top-5 retrieval results (indices and scores only)
            for k in range(min(5, len(unique_texts))):
                if k < topk_indices.shape[1]:
                    idx = topk_indices[i, k].item()
                    score = topk_values[i, k].item()
                    
                    row[f'top{k+1}_idx'] = idx
                    row[f'top{k+1}_score'] = score
                else:
                    row[f'top{k+1}_idx'] = -1
                    row[f'top{k+1}_score'] = 0.0
            
            predictions_data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(predictions_data)
        
        # Save current epoch predictions
        epoch_file = os.path.join(predictions_dir, f'val_predictions_epoch_{epoch}.csv')
        df.to_csv(epoch_file, index=False)
        print(f"Saved predictions to {epoch_file}")
        
        # If this is the best epoch, save separately
        if hasattr(self, 'best_epoch') and epoch == self.best_epoch:
            best_file = os.path.join(predictions_dir, f'val_predictions_best_epoch_{epoch}.csv')
            df.to_csv(best_file, index=False)
            print(f"Saved best epoch predictions to {best_file}")
    
    def _gather_distributed_predictions(
        self,
        video_features: List[torch.Tensor],
        text_features: List[torch.Tensor],
        texts: List[str],
        paths: List[str],
        generated_texts: List[str],
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], List[str], List[str]]:
        """
        Gather all predictions across GPUs.
        
        Returns:
            Gathered tensors and lists from all GPUs
        """
        # Concatenate local tensors
        local_video_features = torch.cat(video_features, dim=0) if video_features else torch.empty(0)
        local_text_features = torch.cat(text_features, dim=0) if text_features else torch.empty(0)
        
        # Gather across GPUs if distributed
        if self.world_size > 1:
            gathered_video_features = self._gather_tensor_along_batch(local_video_features, self.world_size)
            gathered_text_features = self._gather_tensor_along_batch(local_text_features, self.world_size)
            gathered_texts = self._gather_strings_across_gpus(texts, self.world_size, self.device)
            gathered_paths = self._gather_strings_across_gpus(paths, self.world_size, self.device)
            gathered_generated = self._gather_strings_across_gpus(generated_texts, self.world_size, self.device)
            gathered_targets = self._gather_strings_across_gpus(target_texts, self.world_size, self.device)
        else:
            gathered_video_features = local_video_features
            gathered_text_features = local_text_features
            gathered_texts = texts
            gathered_paths = paths
            gathered_generated = generated_texts
            gathered_targets = target_texts
        
        return (gathered_video_features, gathered_text_features, 
                gathered_texts, gathered_paths, gathered_generated, gathered_targets)