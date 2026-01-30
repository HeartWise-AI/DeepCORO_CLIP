import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from typing import Dict, Optional, Any, List

from utils.registry import LossRegistry
from utils.enums import LossType
from utils.loss.losses import InfoNCELoss
from utils.stenosis_extractor import (
    extract_stenosis_from_report,
    get_stenosis_feature_vector,
    compute_stenosis_metrics,
)


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
        contrastive_loss_type: str = "siglip",  # "siglip" or "softmax"
        captioning_loss_type: str = "cross_entropy",
        masked_modeling_loss_type: str = "mse",
        temperature: float = 0.1,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
        patch_contrastive_weight: float = 0.4,  # Weight for patch-level contrastive loss
        # Stenosis-aware loss parameters
        use_stenosis_loss: bool = True,
        stenosis_mse_weight: float = 0.3,
        critical_bce_weight: float = 0.2,
        critical_threshold: float = 70.0,
        numeric_token_weight_multiplier: float = 5.0,
    ):
        super().__init__()

        # Default loss weights
        self.default_weights = {
            "contrastive": 1.0,
            "captioning": 1.0,
            "masked_modeling": 0.1,
            "distillation": 0.0,  # Future task
            "patch_contrastive": 0.0,  # Patch-level contrastive (computed separately)
            "stenosis_mse": stenosis_mse_weight,  # Per-artery stenosis MSE
            "critical_bce": critical_bce_weight,  # Critical finding detection
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
        self.patch_contrastive_weight = patch_contrastive_weight

        # Stenosis-aware loss parameters
        self.use_stenosis_loss = use_stenosis_loss
        self.critical_threshold = critical_threshold
        self.numeric_token_weight_multiplier = numeric_token_weight_multiplier

        # Initialize loss functions
        self._init_loss_functions()
    
    def _init_loss_functions(self):
        """Initialize the individual loss functions."""
        # Always use SigLIP-style contrastive loss (gated similarity + sigmoid)
        self.contrastive_loss_fn = InfoNCELoss(
            temperature=self.temperature,
            use_ddp=True,
            loss_type='siglip'
        )
        
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

    def _compute_stenosis_losses(
        self,
        generated_texts: List[str],
        target_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stenosis-aware auxiliary losses.

        Args:
            generated_texts: List of generated caption texts (decoded from logits)
            target_texts: List of target caption texts (ground truth)

        Returns:
            Dictionary with stenosis_mse and critical_bce losses
        """
        batch_size = len(generated_texts)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')

        # Extract stenosis vectors for all samples
        gen_stenosis_list = []
        tgt_stenosis_list = []
        has_critical_gen = []
        has_critical_tgt = []

        for gen_text, tgt_text in zip(generated_texts, target_texts):
            # Get stenosis feature vectors [17 arteries]
            gen_vec = get_stenosis_feature_vector(gen_text)
            tgt_vec = get_stenosis_feature_vector(tgt_text)

            gen_stenosis_list.append(gen_vec)
            tgt_stenosis_list.append(tgt_vec)

            # Check for critical stenoses (≥70%)
            has_critical_gen.append(1.0 if (gen_vec >= self.critical_threshold).any() else 0.0)
            has_critical_tgt.append(1.0 if (tgt_vec >= self.critical_threshold).any() else 0.0)

        # Convert to tensors (numpy arrays -> torch tensors)
        import numpy as np
        gen_stenosis = torch.from_numpy(
            np.stack(gen_stenosis_list)
        ).to(device=device, dtype=torch.float32)
        tgt_stenosis = torch.from_numpy(
            np.stack(tgt_stenosis_list)
        ).to(device=device, dtype=torch.float32)

        has_critical_gen_tensor = torch.tensor(has_critical_gen, dtype=torch.float32, device=device)
        has_critical_tgt_tensor = torch.tensor(has_critical_tgt, dtype=torch.float32, device=device)

        # Compute per-artery MSE loss (normalized by 100 to match percentage scale)
        stenosis_mse = F.mse_loss(gen_stenosis / 100.0, tgt_stenosis / 100.0)

        # Compute critical finding detection loss (binary cross-entropy with logits)
        # Use sigmoid to convert probabilities to logits range
        # Add epsilon to avoid log(0)
        eps = 1e-7
        has_critical_gen_clipped = torch.clamp(has_critical_gen_tensor, eps, 1.0 - eps)
        critical_bce = F.binary_cross_entropy(
            has_critical_gen_clipped,
            has_critical_tgt_tensor,
            reduction='mean'
        )

        return {
            'stenosis_mse': stenosis_mse,
            'critical_bce': critical_bce,
        }
    
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
        patch_features: Optional[torch.Tensor] = None,  # [B, N, D] patch-level features
        # Stenosis-aware loss inputs
        generated_texts: Optional[List[str]] = None,  # Decoded generated captions
        target_texts: Optional[List[str]] = None,  # Target caption texts
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
        
        # Study-level contrastive loss
        if self.loss_weights["contrastive"] > 0:
            if log_temp is None:
                log_temp = torch.log(torch.tensor(self.temperature, device=video_features.device))

            contrastive_loss = self.contrastive_loss_fn(video_features, text_features, log_temp)
            losses["contrastive"] = contrastive_loss
            total_loss += self.loss_weights["contrastive"] * contrastive_loss

            # Patch-level contrastive loss (if patch features provided)
            if patch_features is not None and self.patch_contrastive_weight > 0:
                # Pool patch features to study level for alignment
                patch_pooled = patch_features.mean(dim=1)  # [B, D]

                # Compute patch-level contrastive loss
                patch_contrastive_loss = self.contrastive_loss_fn(patch_pooled, text_features, log_temp)
                losses["patch_contrastive"] = patch_contrastive_loss

                # Add weighted patch contrastive loss
                total_loss += self.loss_weights["contrastive"] * self.patch_contrastive_weight * patch_contrastive_loss
        
        # Captioning loss
        if self.loss_weights["captioning"] > 0 and caption_logits is not None and caption_targets is not None:
            captioning_loss = self.captioning_loss(caption_logits, caption_targets)
            losses["captioning"] = captioning_loss
            total_loss += self.loss_weights["captioning"] * captioning_loss

            # Stenosis-aware auxiliary losses (only if texts are provided)
            if self.use_stenosis_loss and generated_texts is not None and target_texts is not None:
                try:
                    stenosis_losses = self._compute_stenosis_losses(generated_texts, target_texts)

                    # Add stenosis MSE loss
                    if self.loss_weights["stenosis_mse"] > 0:
                        losses["stenosis_mse"] = stenosis_losses["stenosis_mse"]
                        total_loss += self.loss_weights["stenosis_mse"] * stenosis_losses["stenosis_mse"]

                    # Add critical BCE loss
                    if self.loss_weights["critical_bce"] > 0:
                        losses["critical_bce"] = stenosis_losses["critical_bce"]
                        total_loss += self.loss_weights["critical_bce"] * stenosis_losses["critical_bce"]

                except Exception as e:
                    # Gracefully handle extraction failures (e.g., malformed text)
                    # Don't crash training, just skip stenosis loss for this batch
                    pass
        
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
