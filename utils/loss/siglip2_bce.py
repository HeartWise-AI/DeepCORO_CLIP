"""
SigLIP 2 BCE Loss Implementation.

SigLIP (Sigmoid Loss for Language Image Pre-training) treats image-text matching
as a binary classification problem rather than a contrastive softmax problem.
Each (image, text) pair is independently classified as matching (1) or not matching (0).

Reference: https://arxiv.org/abs/2303.15343 (SigLIP paper)
Reference: https://arxiv.org/abs/2502.14786 (SigLIP 2 paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp.autocast_mode import autocast

# NOTE: The unified SigLIPLoss in utils/loss/contrastive.py is now registered
# for all SigLIP variants. These classes are kept for backwards compatibility.


class SigLIP2BCELoss(nn.Module):
    """
    SigLIP 2 BCE Loss: Binary cross-entropy loss for image-text matching.

    Unlike CLIP's contrastive loss which uses softmax over the entire batch,
    SigLIP treats each image-text pair as an independent binary classification:
    - Positive pairs (diagonal): label = 1
    - Negative pairs (off-diagonal): label = 0

    Loss = -sum(y * log(sigmoid(logit)) + (1-y) * log(1 - sigmoid(logit)))

    The key advantages:
    1. No need for large batch sizes (each pair is independent)
    2. More stable training with mixed precision
    3. Better handling of noisy image-text pairs
    """

    def __init__(
        self,
        bias_init: float = -10.0,
        learnable_bias: bool = True,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize SigLIP 2 BCE Loss.

        Args:
            bias_init: Initial value for the learnable bias term.
                       Negative values make the model initially predict low probabilities,
                       which is desirable since most pairs are negative.
            learnable_bias: Whether to make the bias learnable.
            label_smoothing: Label smoothing factor (0.0 = no smoothing).
        """
        super().__init__()
        self.label_smoothing = label_smoothing

        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("bias", torch.tensor(bias_init))

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SigLIP BCE loss between video and text features.

        Args:
            video_features: [B, D] video/image embeddings
            text_features: [B, D] text embeddings
            log_temp: Learnable log temperature parameter

        Returns:
            Scalar loss value
        """
        with autocast("cuda", enabled=False):
            # Normalize embeddings
            video_features = F.normalize(video_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)

            # Compute similarity matrix [B, B]
            similarity = torch.matmul(video_features, text_features.t())

            # Apply temperature scaling and bias
            temp = torch.exp(log_temp.float())
            logits = similarity / temp + self.bias

            # Create binary labels: 1 on diagonal (matching pairs), 0 elsewhere
            batch_size = video_features.shape[0]
            labels = torch.eye(batch_size, device=video_features.device, dtype=video_features.dtype)

            # Apply label smoothing if specified
            if self.label_smoothing > 0:
                labels = labels * (1 - self.label_smoothing) + self.label_smoothing / 2

            # Compute binary cross-entropy loss for all pairs
            # Using BCE with logits for numerical stability
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        return loss


class SigLIP2BCELossRegistered(SigLIP2BCELoss):
    """Legacy registered version - use SigLIPLoss from contrastive.py instead."""
    pass


class SigLIP2BCELossDDP(nn.Module):
    """
    DDP-aware SigLIP 2 BCE Loss.

    Gathers features from all GPUs to compute the loss over the full global batch,
    while maintaining gradient flow back to local features.
    """

    def __init__(
        self,
        bias_init: float = -10.0,
        learnable_bias: bool = True,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize DDP SigLIP 2 BCE Loss.

        Args:
            bias_init: Initial value for the learnable bias term.
            learnable_bias: Whether to make the bias learnable.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        self.label_smoothing = label_smoothing

        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("bias", torch.tensor(bias_init))

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DDP-aware SigLIP BCE loss.

        Args:
            video_features: [B_local, D] local video embeddings
            text_features: [B_local, D] local text embeddings
            log_temp: Learnable log temperature parameter

        Returns:
            Scalar loss value
        """
        with autocast("cuda", enabled=False):
            # Gather features from all GPUs
            video_features_all = _gather_all_with_gradient(video_features)
            text_features_all = _gather_all_with_gradient(text_features)

            # Normalize embeddings
            video_features_all = F.normalize(video_features_all.float(), dim=-1)
            text_features_all = F.normalize(text_features_all.float(), dim=-1)

            # Compute global similarity matrix [N, N] where N = global batch size
            similarity = torch.matmul(video_features_all, text_features_all.t())

            # Apply temperature scaling and bias
            temp = torch.exp(log_temp.float())
            logits = similarity / temp + self.bias

            # Create binary labels
            global_batch_size = video_features_all.shape[0]
            labels = torch.eye(global_batch_size, device=video_features.device, dtype=video_features.dtype)

            # Apply label smoothing
            if self.label_smoothing > 0:
                labels = labels * (1 - self.label_smoothing) + self.label_smoothing / 2

            # Compute BCE loss
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        return loss


class SigLIP2BCELossDDPRegistered(SigLIP2BCELossDDP):
    """Legacy registered version - use SigLIPLoss from contrastive.py instead."""
    pass


class _GatherAllGradients(torch.autograd.Function):
    """
    Autograd function that gathers tensors from all processes while preserving gradients.
    """

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


def _gather_all_with_gradient(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all processes with gradient preservation."""
    if not dist.is_initialized():
        return tensor
    return _GatherAllGradients.apply(tensor)


class SigLIP2MultiPositiveBCELoss(nn.Module):
    """
    SigLIP 2 BCE Loss with support for multiple positive pairs.

    This extends the standard SigLIP BCE loss to handle cases where
    each video/image may have multiple matching text descriptions,
    or multiple videos may share the same text description.

    Useful for:
    - Multi-caption datasets (one image, multiple descriptions)
    - Semantic similarity (multiple images with similar meaning)
    """

    def __init__(
        self,
        bias_init: float = -10.0,
        learnable_bias: bool = True,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize multi-positive SigLIP 2 BCE Loss.

        Args:
            bias_init: Initial value for the learnable bias term.
            learnable_bias: Whether to make the bias learnable.
            positive_weight: Weight for positive pairs in the loss.
            negative_weight: Weight for negative pairs in the loss.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.label_smoothing = label_smoothing

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
        Compute multi-positive SigLIP BCE loss.

        Args:
            video_features: [B_v, D] video embeddings
            text_features: [B_t, D] text embeddings
            log_temp: Learnable log temperature parameter
            pos_mask: [B_v, B_t] binary mask indicating positive pairs.
                     If None, assumes diagonal (standard 1-to-1 matching).
            pos_weights: [B_v, B_t] optional weights for positive pairs.

        Returns:
            Scalar loss value
        """
        with autocast("cuda", enabled=False):
            # Normalize embeddings
            video_features = F.normalize(video_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)

            # Compute similarity matrix [B_v, B_t]
            similarity = torch.matmul(video_features, text_features.t())

            # Apply temperature scaling and bias
            temp = torch.exp(log_temp.float())
            logits = similarity / temp + self.bias

            # Clamp logits for numerical stability
            logits = torch.clamp(logits, min=-30.0, max=30.0)

            # Create or use provided positive mask
            if pos_mask is None:
                # Default: diagonal matching
                B_v, B_t = video_features.shape[0], text_features.shape[0]
                min_size = min(B_v, B_t)
                labels = torch.zeros(B_v, B_t, device=video_features.device, dtype=video_features.dtype)
                labels[:min_size, :min_size] = torch.eye(min_size, device=video_features.device, dtype=video_features.dtype)
            else:
                labels = pos_mask.float()

            # Apply label smoothing
            if self.label_smoothing > 0:
                labels = labels * (1 - self.label_smoothing) + self.label_smoothing / 2

            # Compute per-element weights
            weights = torch.where(
                labels > 0.5,
                torch.full_like(logits, self.positive_weight),
                torch.full_like(logits, self.negative_weight),
            )

            # Apply optional per-pair positive weights
            if pos_weights is not None:
                weights = torch.where(labels > 0.5, weights * pos_weights, weights)

            # Compute weighted BCE loss
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, weight=weights, reduction="mean"
            )

        return loss


class SigLIP2MultiPositiveBCELossRegistered(SigLIP2MultiPositiveBCELoss):
    """Legacy registered version - use SigLIPLoss from contrastive.py instead."""
    pass
