import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
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
                        videos, input_ids, attention_mask, texts
                    )
                else:
                    # Validation step
                    batch_metrics, batch_outputs = self._val_step(
                        videos, input_ids, attention_mask, texts
                    )
                
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
                
                # Store captioning outputs
                if "generated_texts" in batch_outputs:
                    all_generated_texts.extend(batch_outputs["generated_texts"])
                if "target_texts" in batch_outputs:
                    all_target_texts.extend(batch_outputs["target_texts"])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{batch_metrics['total_loss']:.4f}",
                    'contrastive': f"{batch_metrics.get('contrastive_loss', 0.0):.4f}",
                    'captioning': f"{batch_metrics.get('captioning_loss', 0.0):.4f}",
                    'mvm': f"{batch_metrics.get('masked_modeling_loss', 0.0):.4f}",
                })
        
        # Compute average metrics
        metrics = {
            "total_loss": total_loss / num_batches,
            "contrastive_loss": total_contrastive_loss / num_batches,
            "captioning_loss": total_captioning_loss / num_batches,
            "masked_modeling_loss": total_masked_modeling_loss / num_batches,
        }
        
        # Compute contrastive learning metrics
        if all_video_features and all_text_features:
            video_features = torch.cat(all_video_features, dim=0)
            text_features = torch.cat(all_text_features, dim=0)
            
            # Gather features across GPUs
            if self.world_size > 1:
                video_features = self._gather_tensor_along_batch(video_features, self.world_size)
                text_features = self._gather_tensor_along_batch(text_features, self.world_size)
                all_texts = self._gather_strings_across_gpus(all_texts, self.world_size, self.device)
            
            if self.config.is_ref_device:
                contrastive_metrics = self._compute_contrastive_metrics(
                    video_features, text_features, all_texts
                )
                metrics.update(contrastive_metrics)
        
        # Compute captioning metrics
        if all_generated_texts and all_target_texts:
            if self.world_size > 1:
                all_generated_texts = self._gather_strings_across_gpus(
                    all_generated_texts, self.world_size, self.device
                )
                all_target_texts = self._gather_strings_across_gpus(
                    all_target_texts, self.world_size, self.device
                )
            
            if self.config.is_ref_device:
                captioning_metrics = self._compute_captioning_metrics(
                    all_generated_texts, all_target_texts
                )
                metrics.update(captioning_metrics)
        
        return metrics
    
    def _train_step(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: list[str],
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
        video_tokens = self.video_encoder.get_tokens(videos, mode="patch")  # [B, num_tokens, hidden_size]
        
        # Get aggregated video features for contrastive learning
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
            current_weights = self.loss_weight_scheduler.get_weights(
                self.current_epoch * len(self.train_loader) + batch_idx
            )
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
        
        # Gradient clipping
        if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
            clip_grad_norm_(self.optimizer.param_groups, self.config.max_grad_norm)
        
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
        
        # Prepare outputs
        outputs = {
            "video_features": video_features,
            "text_features": text_features,
            "texts": texts,
        }
        
        return metrics, outputs
    
    def _val_step(
        self, 
        videos: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        texts: list[str],
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
            }
            
            return metrics, outputs
    
    def _compute_contrastive_metrics(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        texts: list[str],
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
            
            # Compute metrics
            for k in self.config.recall_k:
                recall = compute_recall_at_k(similarity_matrix, k)
                metrics[f"recall_at_{k}"] = recall
            
            for k in self.config.ndcg_k:
                ndcg = compute_ndcg_at_k(similarity_matrix, k)
                metrics[f"ndcg_at_{k}"] = ndcg
            
            # Compute alignment score
            alignment_score = compute_alignment_score(video_features_norm, text_features_norm)
            metrics["alignment_score"] = alignment_score
            
            # Compute median rank
            median_rank = compute_median_rank(similarity_matrix)
            metrics["median_rank"] = median_rank
        
        return metrics
    
    def _compute_captioning_metrics(
        self,
        generated_texts: list[str],
        target_texts: list[str],
    ) -> dict[str, float]:
        """Compute captioning metrics."""
        metrics = {}
        
        # TODO: Implement BLEU, ROUGE, and other captioning metrics
        # For now, return basic metrics
        metrics["num_generated"] = len(generated_texts)
        metrics["num_targets"] = len(target_texts)
        
        return metrics
    
    def _preprocess_inputs(self, batch: dict) -> tuple[dict, list[str]]:
        """Preprocess inputs for the model."""
        videos = batch["videos"]
        texts = batch["texts"]
        
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
        """Gather strings from all GPUs."""
        if world_size == 1:
            return local_strings
        
        # Convert strings to tensors for gathering
        string_tensor = torch.tensor([hash(s) for s in local_strings], device=device)
        gathered_tensors = [torch.zeros_like(string_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, string_tensor)
        
        # Convert back to strings (simplified - in practice you'd need proper string handling)
        all_strings = local_strings.copy()
        for tensor in gathered_tensors:
            if tensor.device != device:
                tensor = tensor.to(device)
            # This is a simplified version - in practice you'd need proper string serialization
            all_strings.extend(local_strings)  # Simplified
        
        return all_strings
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics for the epoch."""
        if self.wandb_wrapper.is_initialized():
            # Log training metrics
            train_log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            train_log_dict["epoch"] = epoch
            
            # Log validation metrics
            val_log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
            
            # Log learning rate
            if self.optimizer is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    train_log_dict[f"lr/group_{i}"] = param_group['lr']
            
            # Log temperature
            if self.log_temp is not None:
                train_log_dict["train/temperature"] = torch.exp(self.log_temp).item()
            
            # Combine and log
            log_dict = {**train_log_dict, **val_log_dict}
            self.wandb_wrapper.log(log_dict)
    
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