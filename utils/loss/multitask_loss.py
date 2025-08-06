import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from typing import Dict, Optional, Any

from utils.registry import LossRegistry
from utils.enums import LossType


@LossRegistry.register(LossType.MULTITASK)
class MultitaskLoss(nn.Module):
    """
    Multitask loss combining contrastive, captioning, and masked video modeling losses.
    
    This loss function supports:
    - Contrastive loss (video ↔ text alignment)
    - Captioning loss (autoregressive report generation)
    - Masked video modeling loss (self-supervised learning)
    - Optional distillation loss (future task)
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        contrastive_loss_type: str = "sigmoid",  # "sigmoid" or "softmax"
        captioning_loss_type: str = "cross_entropy",
        masked_modeling_loss_type: str = "mse",
        temperature: float = 0.1,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        
        # Default loss weights
        self.default_weights = {
            "contrastive": 1.0,
            "captioning": 1.0,
            "masked_modeling": 0.1,
            "distillation": 0.0,  # Future task
        }
        
        # Update with provided weights
        if loss_weights is not None:
            self.default_weights.update(loss_weights)
        
        self.loss_weights = self.default_weights
        self.contrastive_loss_type = contrastive_loss_type
        self.captioning_loss_type = captioning_loss_type
        self.masked_modeling_loss_type = masked_modeling_loss_type
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        # Initialize loss functions
        self._init_loss_functions()
    
    def _init_loss_functions(self):
        """Initialize the individual loss functions."""
        # Contrastive loss
        if self.contrastive_loss_type == "sigmoid":
            self.contrastive_loss = self._sigmoid_contrastive_loss
        elif self.contrastive_loss_type == "softmax":
            self.contrastive_loss = self._softmax_contrastive_loss
        else:
            raise ValueError(f"Unknown contrastive loss type: {self.contrastive_loss_type}")
        
        # Captioning loss
        if self.captioning_loss_type == "cross_entropy":
            self.captioning_loss = self._cross_entropy_loss
        else:
            raise ValueError(f"Unknown captioning loss type: {self.captioning_loss_type}")
        
        # Masked modeling loss
        if self.masked_modeling_loss_type == "mse":
            self.masked_modeling_loss = self._mse_loss
        else:
            raise ValueError(f"Unknown masked modeling loss type: {self.masked_modeling_loss_type}")
    
    def _sigmoid_contrastive_loss(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sigmoid contrastive loss (SigLIP-style).
        
        Args:
            video_features: [batch_size, hidden_size]
            text_features: [batch_size, hidden_size]
            log_temp: Temperature parameter
            
        Returns:
            Contrastive loss
        """
        # Perform the entire loss computation in full precision to avoid
        # overflow/underflow issues when AMP is enabled. This has negligible
        # memory overhead because the tensors involved are only of size [B, D]
        # and [B, B].
        with autocast('cuda', enabled=False):
            # Ensure features are 2D
            if video_features.dim() == 3:
                video_features = video_features.squeeze(1)
            if text_features.dim() == 3:
                text_features = text_features.squeeze(1)
            
            # Normalize features in fp32 for numerical stability
            video_features_fp32 = F.normalize(video_features.float(), dim=-1)
            text_features_fp32 = F.normalize(text_features.float(), dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(video_features_fp32, text_features_fp32.t())
            
            # Apply temperature
            temp = torch.exp(log_temp.float())
            logits = similarity / temp
            
            # Create labels (diagonal = positive pairs)
            labels = torch.eye(logits.size(0), device=logits.device)
            
            # Sigmoid cross-entropy loss
            loss_v2t = F.binary_cross_entropy_with_logits(logits, labels)
            loss_t2v = F.binary_cross_entropy_with_logits(logits.t(), labels)
            
            return 0.5 * (loss_v2t + loss_t2v)
    
    def _softmax_contrastive_loss(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Softmax contrastive loss (CLIP-style).
        
        Args:
            video_features: [batch_size, hidden_size]
            text_features: [batch_size, hidden_size]
            log_temp: Temperature parameter
            
        Returns:
            Contrastive loss
        """
        # Perform the entire loss computation in full precision to avoid
        # overflow/underflow issues when AMP is enabled. This has negligible
        # memory overhead because the tensors involved are only of size [B, D]
        # and [B, B].
        with autocast('cuda', enabled=False):
            # Ensure features are 2D
            if video_features.dim() == 3:
                video_features = video_features.squeeze(1)
            if text_features.dim() == 3:
                text_features = text_features.squeeze(1)
            
            # Normalize features in fp32 for numerical stability
            video_features_fp32 = F.normalize(video_features.float(), dim=-1)
            text_features_fp32 = F.normalize(text_features.float(), dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(video_features_fp32, text_features_fp32.t())
            
            # Apply temperature
            temp = torch.exp(log_temp.float())
            logits = similarity / temp
            
            # Create labels
            labels = torch.arange(logits.size(0), device=logits.device)
            
            # Cross-entropy loss
            loss_v2t = F.cross_entropy(logits, labels)
            loss_t2v = F.cross_entropy(logits.t(), labels)
            
            return 0.5 * (loss_v2t + loss_t2v)
    
    def _cross_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss for captioning.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
            
        Returns:
            Captioning loss
        """
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            return F.cross_entropy(
                logits_flat, 
                targets_flat, 
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing
            )
        else:
            return F.cross_entropy(
                logits_flat, 
                targets_flat, 
                ignore_index=self.ignore_index
            )
    
    def _mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss for masked video modeling.
        
        Args:
            pred: Predicted tokens [batch_size, num_tokens, hidden_size]
            target: Target tokens [batch_size, num_tokens, hidden_size]
            mask: Mask indicating which tokens were masked [batch_size, num_tokens]
            
        Returns:
            Masked modeling loss
        """
        # Compute MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over hidden dimension
        
        # Apply mask to get loss only on masked tokens
        mask = mask.bool()
        if mask.sum() > 0:
            loss = loss[mask].mean()
        else:
            loss = loss.mean()  # Fallback if no masked tokens
        
        return loss
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        caption_logits: Optional[torch.Tensor] = None,
        caption_targets: Optional[torch.Tensor] = None,
        masked_pred: Optional[torch.Tensor] = None,
        masked_target: Optional[torch.Tensor] = None,
        masked_mask: Optional[torch.Tensor] = None,
        log_temp: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        student_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of multitask loss.
        
        Args:
            video_features: Video features [batch_size, hidden_size]
            text_features: Text features [batch_size, hidden_size]
            caption_logits: Caption logits [batch_size, seq_len, vocab_size]
            caption_targets: Caption targets [batch_size, seq_len]
            masked_pred: Masked modeling predictions [batch_size, num_tokens, hidden_size]
            masked_target: Masked modeling targets [batch_size, num_tokens, hidden_size]
            masked_mask: Mask for masked modeling [batch_size, num_tokens]
            log_temp: Temperature parameter for contrastive loss
            teacher_logits: Teacher logits for distillation [batch_size, seq_len, vocab_size]
            student_logits: Student logits for distillation [batch_size, seq_len, vocab_size]
            
        Returns:
            Dictionary containing total loss and individual losses
        """
        losses = {}
        total_loss = 0.0
        
        # Contrastive loss
        if self.loss_weights["contrastive"] > 0:
            if log_temp is None:
                log_temp = torch.log(torch.tensor(self.temperature, device=video_features.device))
            
            contrastive_loss = self.contrastive_loss(video_features, text_features, log_temp)
            losses["contrastive"] = contrastive_loss
            total_loss += self.loss_weights["contrastive"] * contrastive_loss
        
        # Captioning loss
        if self.loss_weights["captioning"] > 0 and caption_logits is not None and caption_targets is not None:
            captioning_loss = self.captioning_loss(caption_logits, caption_targets)
            losses["captioning"] = captioning_loss
            total_loss += self.loss_weights["captioning"] * captioning_loss
        
        # Masked modeling loss
        if self.loss_weights["masked_modeling"] > 0 and masked_pred is not None and masked_target is not None:
            masked_modeling_loss = self.masked_modeling_loss(masked_pred, masked_target, masked_mask)
            losses["masked_modeling"] = masked_modeling_loss
            total_loss += self.loss_weights["masked_modeling"] * masked_modeling_loss
        
        # Distillation loss (future task)
        if self.loss_weights["distillation"] > 0 and teacher_logits is not None and student_logits is not None:
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            losses["distillation"] = distillation_loss
            total_loss += self.loss_weights["distillation"] * distillation_loss
        
        losses["total"] = total_loss
        
        return losses


class LossWeightScheduler:
    """
    Scheduler for loss weights during training.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        schedule_type: str = "linear",  # "linear", "cosine", "step"
    ):
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0
    
    def step(self) -> Dict[str, float]:
        """
        Update loss weights based on current step.
        
        Returns:
            Current loss weights
        """
        if self.current_step < self.warmup_steps:
            # Warmup phase - use initial weights
            weights = self.initial_weights.copy()
        else:
            # Main training phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.schedule_type == "linear":
                alpha = progress
            elif self.schedule_type == "cosine":
                alpha = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            elif self.schedule_type == "step":
                alpha = 1.0 if progress > 0.5 else 0.0
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")
            
            # Interpolate between initial and final weights
            weights = {}
            for key in self.initial_weights:
                if key in self.final_weights:
                    weights[key] = (1 - alpha) * self.initial_weights[key] + alpha * self.final_weights[key]
                else:
                    weights[key] = self.initial_weights[key]
        
        self.current_step += 1
        return weights
    
    def get_weights(self, step: int) -> Dict[str, float]:
        """
        Get loss weights for a specific step.
        
        Args:
            step: Current training step
            
        Returns:
            Loss weights for the given step
        """
        self.current_step = step
        return self.step()