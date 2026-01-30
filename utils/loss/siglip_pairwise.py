import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp.autocast_mode import autocast


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


def _gather_all(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all processes with gradient preservation."""
    if not dist.is_initialized():
        return tensor
    return _GatherWithGradient.apply(tensor)


# =============================================================================
# NOTE: The unified SigLIPLoss in utils/loss/contrastive.py is now registered
# for all SigLIP variants. This class is kept for backwards compatibility.
# =============================================================================


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


class SiglipPairwiseLoss(nn.Module):
    """
    True SigLIP-style BCE loss for multi-positive video-text alignment.

    NOTE: For new code, use SigLIPLoss from utils.loss.contrastive instead.
    It provides the same functionality with auto-DDP support.

    Key features:
    - Independent BCE per pair (not softmax over batch)
    - Multi-positive support via pos_mask
    - Per-pair weighting via pos_weights
    - Auto positive weighting to handle class imbalance
    - Entropy regularization to prevent embedding collapse
    """

    def __init__(
        self,
        *,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        reduction: str = "mean",
        use_positive_weights: bool = True,
        auto_positive_weight: bool = False,
        entropy_regularization: bool = False,
        entropy_weight: float = 0.1,
        min_entropy_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction '{reduction}'. Expected 'mean' or 'sum'.")

        self.positive_weight = max(float(positive_weight), 1e-6)
        self.negative_weight = max(float(negative_weight), 0.0)
        self.reduction = reduction
        self.use_positive_weights = use_positive_weights
        self.auto_positive_weight = auto_positive_weight
        self.entropy_regularization = entropy_regularization
        self.entropy_weight = entropy_weight
        self.min_entropy_threshold = min_entropy_threshold
        self._last_entropy_diagnostics: dict = {}

    def forward(
        self,
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape {tuple(logits.shape)}")
        if pos_mask.shape != logits.shape:
            raise ValueError("pos_mask must match logits shape.")
        if pos_weights is not None and pos_weights.shape != logits.shape:
            raise ValueError("pos_weights must match logits shape.")

        targets = pos_mask.clamp_min(0.0).clamp_max(1.0)

        weight_matrix = torch.full_like(logits, self.negative_weight)
        if self.use_positive_weights and pos_weights is not None:
            positive_contrib = pos_weights * self.positive_weight
        else:
            positive_contrib = torch.full_like(logits, self.positive_weight)

        if self.auto_positive_weight:
            total_texts = logits.size(1)
            pos_counts = pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            neg_counts = total_texts - pos_counts
            ratio = (neg_counts / pos_counts).clamp_min(1.0)
            positive_contrib = ratio.expand_as(logits)

        weight_matrix = torch.where(pos_mask > 0, positive_contrib, weight_matrix)

        # Clamp logits to prevent numerical instability in BCE
        # sigmoid(-30) ≈ 0, sigmoid(30) ≈ 1, so gradients are well-behaved
        logits_clamped = torch.clamp(logits, min=-30.0, max=30.0)

        loss = F.binary_cross_entropy_with_logits(
            logits_clamped,
            targets,
            weight=weight_matrix,
            reduction="none",
        )

        if self.reduction == "mean":
            bce_loss = loss.mean()
        else:
            bce_loss = loss.sum()

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


class SiglipPairwiseLossDDP(nn.Module):
    """
    DDP-compatible SigLIP pairwise BCE loss.

    This version gathers features from all GPUs before computing the loss,
    ensuring correct gradient flow in distributed training.

    For single-GPU training, use SiglipPairwiseLoss directly.
    """

    def __init__(
        self,
        *,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        reduction: str = "mean",
        use_positive_weights: bool = True,
        auto_positive_weight: bool = False,
        entropy_regularization: bool = False,
        entropy_weight: float = 0.1,
        min_entropy_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.loss_fn = SiglipPairwiseLoss(
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            reduction=reduction,
            use_positive_weights=use_positive_weights,
            auto_positive_weight=auto_positive_weight,
            entropy_regularization=entropy_regularization,
            entropy_weight=entropy_weight,
            min_entropy_threshold=min_entropy_threshold,
        )

    def forward(
        self,
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute DDP-aware pairwise BCE loss.

        Note: The logits should already be computed from gathered features
        in the runner. This wrapper ensures the loss computation is
        consistent across all GPUs.
        """
        # Gather pos_mask and pos_weights if in DDP mode
        if dist.is_initialized():
            pos_mask = _gather_all(pos_mask.contiguous())
            if pos_weights is not None:
                pos_weights = _gather_all(pos_weights.contiguous())
            # Logits are gathered via all_gather in the runner's _scaled_logits
            logits = _gather_all(logits.contiguous())

        return self.loss_fn(logits, pos_mask, pos_weights)

    def get_entropy_diagnostics(self) -> dict:
        """Get entropy diagnostics from last forward pass."""
        return self.loss_fn.get_entropy_diagnostics()


class SiglipPairwiseFeatureLoss(nn.Module):
    """
    Full-featured DDP SigLIP loss that takes raw features.

    This is the most complete implementation - it takes video/text features
    directly and handles:
    1. DDP gathering with gradient preservation
    2. Similarity computation with temperature scaling
    3. Multi-positive BCE loss with severity weighting
    4. Entropy regularization to prevent embedding collapse

    Use this when you want the loss to handle everything.
    """

    def __init__(
        self,
        *,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        use_positive_weights: bool = True,
        auto_positive_weight: bool = False,
        entropy_regularization: bool = False,
        entropy_weight: float = 0.1,
        min_entropy_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.positive_weight = max(float(positive_weight), 1e-6)
        self.negative_weight = max(float(negative_weight), 0.0)
        self.use_positive_weights = use_positive_weights
        self.auto_positive_weight = auto_positive_weight
        self.entropy_regularization = entropy_regularization
        self.entropy_weight = entropy_weight
        self.min_entropy_threshold = min_entropy_threshold
        self._last_entropy_diagnostics: dict = {}

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute SigLIP loss from raw features.

        Args:
            video_features: [B, D] video embeddings (local batch in DDP)
            text_features: [T, D] text embeddings
            log_temp: Learnable log temperature
            pos_mask: [B, T] binary mask indicating positive pairs
            pos_weights: [B, T] optional weights for positive pairs

        Returns:
            Scalar loss value
        """
        with autocast("cuda", enabled=False):
            # Gather features in DDP mode
            if dist.is_initialized():
                video_features = _gather_all(video_features.contiguous())
                pos_mask = _gather_all(pos_mask.contiguous())
                if pos_weights is not None:
                    pos_weights = _gather_all(pos_weights.contiguous())

            # Normalize embeddings
            video_features = F.normalize(video_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)

            # Compute similarity matrix [B_global, T]
            similarity = torch.matmul(video_features, text_features.t())

            # Apply temperature scaling
            temp = torch.exp(log_temp.float())
            logits = similarity / temp

            # Clamp logits for numerical stability
            logits = torch.clamp(logits, min=-30.0, max=30.0)

            # Build targets and weights
            targets = pos_mask.clamp(0.0, 1.0).float()

            weight_matrix = torch.full_like(logits, self.negative_weight)
            if self.use_positive_weights and pos_weights is not None:
                positive_contrib = pos_weights * self.positive_weight
            else:
                positive_contrib = torch.full_like(logits, self.positive_weight)

            if self.auto_positive_weight:
                total_texts = logits.size(1)
                pos_counts = pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                neg_counts = total_texts - pos_counts
                ratio = (neg_counts / pos_counts).clamp_min(1.0)
                positive_contrib = ratio.expand_as(logits)

            weight_matrix = torch.where(pos_mask > 0, positive_contrib, weight_matrix)

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
