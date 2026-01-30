"""
Unified Contrastive Losses: CLIP and SigLIP

Two losses, both auto-detect DDP:
- CLIP: Softmax cross-entropy (1 positive per video)
- SigLIP: Sigmoid BCE (multi-positive per video)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp.autocast_mode import autocast

from utils.enums import LossType
from utils.registry import LossRegistry


def compute_entropy_regularization(
    logits: torch.Tensor,
    min_entropy_threshold: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute entropy regularization to prevent embedding collapse.

    When all video embeddings collapse to the same point, the prediction
    distribution becomes very peaked (low entropy). This regularization
    encourages the model to maintain diverse predictions.

    Args:
        logits: [B, T] similarity logits
        min_entropy_threshold: Only penalize if entropy falls below this

    Returns:
        entropy_loss: Penalty term (0 if entropy is above threshold)
        diagnostics: Dict with entropy statistics
    """
    # Compute softmax probabilities for each video (over texts)
    probs = F.softmax(logits, dim=1)  # [B, T]

    # Compute entropy for each video: H = -sum(p * log(p))
    # Add small epsilon to prevent log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy_per_video = -(probs * log_probs).sum(dim=1)  # [B]

    mean_entropy = entropy_per_video.mean()
    min_entropy = entropy_per_video.min()

    # Maximum possible entropy for uniform distribution over T texts
    max_entropy = torch.log(torch.tensor(logits.size(1), dtype=logits.dtype, device=logits.device))

    # Normalized entropy (0 = peaked, 1 = uniform)
    normalized_entropy = mean_entropy / max_entropy

    # Penalty: encourage entropy to be above threshold
    # If entropy is below threshold, add penalty proportional to shortfall
    entropy_deficit = F.relu(min_entropy_threshold - mean_entropy)
    entropy_loss = entropy_deficit

    diagnostics = {
        "entropy_mean": mean_entropy.item(),
        "entropy_min": min_entropy.item(),
        "entropy_max": entropy_per_video.max().item(),
        "entropy_normalized": normalized_entropy.item(),
        "entropy_deficit": entropy_deficit.item(),
    }

    return entropy_loss, diagnostics


# =============================================================================
# DDP Utilities
# =============================================================================

class _GatherWithGradient(torch.autograd.Function):
    """Gather tensors from all DDP processes while preserving gradients."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return grad_output
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        grad_list = grad_output.chunk(world_size, dim=0)
        return grad_list[rank]


def gather_with_gradient(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all processes. No-op if DDP not initialized."""
    if not dist.is_initialized():
        return tensor
    return _GatherWithGradient.apply(tensor.contiguous())


# =============================================================================
# CLIP Loss (Softmax Cross-Entropy)
# =============================================================================

@LossRegistry.register(LossType.CLIP)
@LossRegistry.register(LossType.CONTRASTIVE)
@LossRegistry.register(LossType.CONTRASTIVE_DDP)
class CLIPLoss(nn.Module):
    """
    CLIP-style contrastive loss using softmax cross-entropy.

    Automatically handles DDP (gathers features from all GPUs).
    Assumes 1-to-1 video-text matching (diagonal positives).

    Loss = 0.5 * (CE(video->text) + CE(text->video))
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_features: [B, D] video embeddings
            text_features: [B, D] text embeddings
            log_temp: Learnable log temperature

        Returns:
            Scalar loss
        """
        with autocast("cuda", enabled=False):
            # Gather from all GPUs if DDP
            video_features = gather_with_gradient(video_features)
            text_features = gather_with_gradient(text_features)

            # Normalize
            video_features = F.normalize(video_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)

            # Compute similarity [N, N]
            similarity = torch.matmul(video_features, text_features.t())

            # Apply temperature
            temp = torch.exp(log_temp.float()).clamp(min=1e-4)
            logits = similarity / temp

            # Diagonal targets (1-to-1 matching)
            batch_size = logits.size(0)
            targets = torch.arange(batch_size, device=logits.device)

            # Bidirectional cross-entropy
            loss_v2t = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
            loss_t2v = F.cross_entropy(logits.t(), targets, label_smoothing=self.label_smoothing)

        return 0.5 * (loss_v2t + loss_t2v)


# =============================================================================
# SigLIP Loss (Sigmoid BCE - Multi-Positive)
# =============================================================================

@LossRegistry.register(LossType.SIGLIP)
@LossRegistry.register(LossType.SIGLIP_PAIRWISE)
@LossRegistry.register(LossType.SIGLIP2_BCE)
@LossRegistry.register(LossType.SIGLIP2_BCE_DDP)
@LossRegistry.register(LossType.SIGLIP2_MULTI_POSITIVE)
class SigLIPLoss(nn.Module):
    """
    SigLIP-style loss using sigmoid BCE for independent pair classification.

    Automatically handles DDP (gathers features from all GPUs).
    Supports multi-positive (multiple texts per video) via pos_mask.

    Key advantages over CLIP:
    - Each pair classified independently (no softmax competition)
    - Works with any batch size (no need for large batches)
    - Supports multiple positives per video
    - Optional severity weighting for medical imaging
    - Entropy regularization to prevent embedding collapse
    """

    def __init__(
        self,
        bias_init: float = -10.0,
        learnable_bias: bool = True,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        use_severity_weights: bool = True,
        auto_balance: bool = False,
        entropy_regularization: bool = False,
        entropy_weight: float = 0.1,
        min_entropy_threshold: float = 2.0,
    ):
        """
        Args:
            bias_init: Initial bias (negative = predict "no match" by default)
            learnable_bias: Whether bias is trainable
            positive_weight: Base weight for positive pairs
            negative_weight: Base weight for negative pairs
            use_severity_weights: Whether to use per-pair severity weights
            auto_balance: Auto-compute pos/neg weight ratio per sample
            entropy_regularization: Add entropy penalty to prevent collapse
            entropy_weight: Weight for entropy loss term
            min_entropy_threshold: Only penalize if entropy drops below this
        """
        super().__init__()
        self.positive_weight = max(float(positive_weight), 1e-6)
        self.negative_weight = max(float(negative_weight), 1e-6)
        self.use_severity_weights = use_severity_weights
        self.auto_balance = auto_balance
        self.entropy_regularization = entropy_regularization
        self.entropy_weight = entropy_weight
        self.min_entropy_threshold = min_entropy_threshold
        self._last_entropy_diagnostics: dict = {}

        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("bias", torch.tensor(bias_init))

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor,
        pos_mask: torch.Tensor | None = None,
        pos_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            video_features: [B, D] video embeddings
            text_features: [T, D] text embeddings (T can differ from B)
            log_temp: Learnable log temperature
            pos_mask: [B, T] binary mask (1 = positive pair).
                     If None, assumes diagonal (1-to-1 matching).
            pos_weights: [B, T] optional per-pair weights (e.g., severity)

        Returns:
            Scalar loss
        """
        with autocast("cuda", enabled=False):
            # Gather from all GPUs if DDP
            video_features = gather_with_gradient(video_features)
            if pos_mask is not None:
                pos_mask = gather_with_gradient(pos_mask)
            if pos_weights is not None:
                pos_weights = gather_with_gradient(pos_weights)

            # Normalize
            video_features = F.normalize(video_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)

            # Compute similarity [B_global, T]
            similarity = torch.matmul(video_features, text_features.t())

            # Apply temperature and bias
            temp = torch.exp(log_temp.float()).clamp(min=1e-4)
            logits = similarity / temp + self.bias

            # Clamp for numerical stability
            logits = torch.clamp(logits, min=-30.0, max=30.0)

            # Build targets
            B, T = logits.shape
            if pos_mask is None:
                # Default: diagonal matching (1-to-1)
                min_dim = min(B, T)
                targets = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
                targets[:min_dim, :min_dim] = torch.eye(min_dim, device=logits.device, dtype=logits.dtype)
            else:
                targets = pos_mask.float().clamp(0.0, 1.0)

            # Build weight matrix
            weight_matrix = torch.full_like(logits, self.negative_weight)

            if self.use_severity_weights and pos_weights is not None:
                # Use provided per-pair weights for positives
                positive_contrib = pos_weights * self.positive_weight
            else:
                positive_contrib = torch.full_like(logits, self.positive_weight)

            if self.auto_balance:
                # Auto-balance: weight positives by neg/pos ratio per row
                pos_counts = targets.sum(dim=1, keepdim=True).clamp(min=1.0)
                neg_counts = T - pos_counts
                ratio = (neg_counts / pos_counts).clamp(min=1.0)
                positive_contrib = ratio.expand_as(logits)

            weight_matrix = torch.where(targets > 0.5, positive_contrib, weight_matrix)

            # Compute BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weight_matrix, reduction="mean"
            )

            # Add entropy regularization if enabled
            if self.entropy_regularization:
                entropy_loss, self._last_entropy_diagnostics = compute_entropy_regularization(
                    logits, self.min_entropy_threshold
                )
                total_loss = bce_loss + self.entropy_weight * entropy_loss
                self._last_entropy_diagnostics["bce_loss"] = bce_loss.item()
                self._last_entropy_diagnostics["entropy_loss"] = (self.entropy_weight * entropy_loss).item()
                return total_loss

        return bce_loss

    def get_entropy_diagnostics(self) -> dict:
        """Get entropy diagnostics from last forward pass."""
        return self._last_entropy_diagnostics


# =============================================================================
# Convenience aliases
# =============================================================================

# These are the same class, just different names for clarity
CLIP = CLIPLoss
SigLIP = SigLIPLoss
