import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp.autocast_mode import autocast

from typing import Any
from utils.enums import LossType
from utils.registry import LossRegistry


@LossRegistry.register(LossType.CONTRASTIVE)
class ContrastiveLoss(nn.Module):
    """
    CLIP-style bidirectional contrastive loss for video and text embeddings.
    """
    def __init__(self):
        """
        Initialize contrastive loss.
        """
        super().__init__()
    
    def forward(
        self,
        video_features: torch.Tensor, 
        text_features: torch.Tensor,
        log_temp: torch.Tensor = torch.log(torch.tensor(0.1))
    ) -> torch.Tensor:
        """
        Compute contrastive loss between video and text features.
        
        Args:
            video_features (torch.Tensor): [B, D] video embeddings
            text_features (torch.Tensor): [B, D] text embeddings
            log_temp (torch.Tensor, optional): float torch tensor value
            
        Returns:
            torch.Tensor: Scalar loss value
        """        
        # Perform the entire loss computation in full precision to avoid
        # overflow/underflow issues when AMP is enabled. This has negligible
        # memory overhead because the tensors involved are only of size [B, D]
        # and [B, B].
        with autocast(video_features.device.type, enabled=False):
            # Normalize embeddings in fp32 for numerical stability.
            video_features_fp32 = F.normalize(video_features.float(), dim=1)
            text_features_fp32 = F.normalize(text_features.float(), dim=1)

            # Compute similarity matrix of shape [B, B].
            similarity_matrix = torch.matmul(video_features_fp32, text_features_fp32.t())

            # Apply temperature scaling.
            temp = torch.exp(log_temp.float())
            logits = similarity_matrix / temp

            # Targets: assume the i-th video corresponds to the i-th text.
            targets = torch.arange(logits.size(0), device=logits.device)

            # Cross-entropy loss in both directions.
            loss_v2t = F.cross_entropy(logits, targets)
            loss_t2v = F.cross_entropy(logits.t(), targets)
            loss = 0.5 * (loss_v2t + loss_t2v)

        return loss

###############################################################################
# DDP-aware version: uses a custom autograd gather to preserve gradients.
###############################################################################

class GatherLayer(torch.autograd.Function):
    """
    Custom autograd function that performs an all_gather on the input tensor
    while preserving gradients.
    """
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
        # Synchronously gather tensors from all processes.
        dist.all_gather(output, tensor)
        return torch.cat(output, dim=0)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return grad_output
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        # Split the gradient and return the chunk corresponding to this process.
        grad_list = grad_output.chunk(world_size, dim=0)
        return grad_list[local_rank]

def gather_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gathers a tensor from all DDP processes using the custom GatherLayer.
    Assumes tensor shape is [local_batch_size, ...].
    """
    if not dist.is_initialized():
        return tensor
    return GatherLayer.apply(tensor)

@LossRegistry.register(LossType.CONTRASTIVE_DDP)
class ContrastiveLossDDP(nn.Module):
    """
    CLIP-style bidirectional contrastive loss for video and text embeddings in DDP setting.
    """
    def __init__(self):
        """
        Initialize DDP contrastive loss.
        """
        super().__init__()
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor = torch.log(torch.tensor(0.1))
    ) -> torch.Tensor:
        """
        Compute contrastive loss between video and text features in DDP setting.
        
        Args:
            video_features (torch.Tensor): [B, D] video embeddings (local batch)
            text_features (torch.Tensor): [B, D] text embeddings (local batch)
            log_temp (torch.Tensor, optional): Override internal temperature
            
        Returns:
            torch.Tensor: Scalar loss value
        """        
        with autocast(video_features.device.type, enabled=False):
            # 1) Gather features from all GPUs.
            video_features_all = gather_all(video_features)
            text_features_all  = gather_all(text_features)

            # 2) Normalize the gathered features.
            video_features_all = F.normalize(video_features_all.float(), dim=1)
            text_features_all  = F.normalize(text_features_all.float(), dim=1)

            # 3) Compute global similarity matrix of shape [N, N],
            # where N is the total global batch size.
            similarity_matrix = torch.matmul(video_features_all, text_features_all.t())

            # 4) Apply temperature scaling.
            temp = torch.exp(log_temp.float())
            logits = similarity_matrix / temp

            # 5) Targets: assume matching pairs lie along the diagonal.
            n = logits.size(0)
            targets = torch.arange(n, device=logits.device)

            # Cross-entropy losses.
            loss_i2t = F.cross_entropy(logits, targets)
            loss_t2i = F.cross_entropy(logits.t(), targets)
            loss = 0.5 * (loss_i2t + loss_t2i)

        return loss

class SiglipLoss(nn.Module):
    """
    SIGLIP (Simple Gated Language-Image Pre-training) loss implementation.
    This loss applies a gating mechanism to the cosine similarity matrix 
    before computing the contrastive loss.
    """
    def __init__(self):
        """
        Initialize SIGLIP loss.
        """
        super().__init__()
    
    def forward(
        self,
        video_features: torch.Tensor, 
        text_features: torch.Tensor,
        log_temp: torch.Tensor = torch.log(torch.tensor(0.1))
    ) -> torch.Tensor:
        """
        Compute SIGLIP loss between video and text features.
        
        Args:
            video_features (torch.Tensor): [B, D] video embeddings
            text_features (torch.Tensor): [B, D] text embeddings
            log_temp (torch.Tensor, optional): float torch tensor value
            
        Returns:
            torch.Tensor: Scalar loss value
        """        
        # Normalize embeddings.
        video_features = F.normalize(video_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix of shape [B, B].
        similarity_matrix = torch.matmul(video_features, text_features.t())
        
        # Apply SIGLIP gating function: g(x) = x * sigmoid(x)
        gated_similarity = similarity_matrix * torch.sigmoid(similarity_matrix)
        
        # Apply temperature scaling.
        temp = torch.exp(log_temp)
        logits = gated_similarity / temp
        
        # Create targets: assume the i-th video corresponds to the i-th text.
        targets = torch.arange(logits.size(0), device=logits.device)
        
        # Compute cross-entropy loss in both directions.
        loss_v2t = F.cross_entropy(logits, targets)
        loss_t2v = F.cross_entropy(logits.t(), targets)
        loss = 0.5 * (loss_v2t + loss_t2v)
        return loss

@LossRegistry.register(LossType.SIGLIP_DDP)
class SiglipLossDDP(nn.Module):
    """
    SIGLIP loss implementation with DDP support.
    """
    def __init__(self):
        """
        Initialize DDP SIGLIP loss.
        """
        super().__init__()
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor = torch.log(torch.tensor(0.1))
    ) -> torch.Tensor:
        """
        Compute SIGLIP loss between video and text features in DDP setting.
        
        Args:
            video_features (torch.Tensor): [B, D] video embeddings (local batch)
            text_features (torch.Tensor): [B, D] text embeddings (local batch)
            log_temp (torch.Tensor, optional): Override internal temperature
            
        Returns:
            torch.Tensor: Scalar loss value
        """        
        with autocast(video_features.device.type, enabled=False):
            # 1) Gather features from all GPUs using fp16 to cut communication volume.
            video_features_half = gather_all(video_features.half())
            text_features_half  = gather_all(text_features.half())

            # 2) Normalize the gathered features in fp32 for numerical stability.
            video_features_all = F.normalize(video_features_half.float(), dim=1)
            text_features_all  = F.normalize(text_features_half.float(), dim=1)

            # 3) Compute global similarity matrix in manageable chunks to avoid long-running kernels.
            total_samples = video_features_all.size(0)
            chunk_size = 1024 if total_samples > 1024 else total_samples
            similarity_chunks = []
            for start in range(0, total_samples, chunk_size):
                end = start + chunk_size
                v_chunk = video_features_all[start:end]
                similarity_chunks.append(torch.matmul(v_chunk, text_features_all.t()))
            similarity_matrix = torch.cat(similarity_chunks, dim=0)

            # 4) Apply SIGLIP gating function: g(x) = x * sigmoid(x)
            gated_similarity = similarity_matrix * torch.sigmoid(similarity_matrix)
            
            # 5) Apply temperature scaling.
            temp = torch.exp(log_temp.float())
            logits = gated_similarity / temp

            # 6) Targets along the diagonal.
            n = logits.size(0)
            targets = torch.arange(n, device=logits.device)

            # Cross-entropy losses.
            loss_i2t = F.cross_entropy(logits, targets)
            loss_t2i = F.cross_entropy(logits.t(), targets)
            loss = 0.5 * (loss_i2t + loss_t2i)

        return loss

@LossRegistry.register(LossType.INFO_NCE)
class InfoNCELoss(nn.Module):
    """
    InfoNCE loss implementation for contrastive learning.
    This is a wrapper around the contrastive loss functions that maintains state
    and provides a more object-oriented interface.
    """
    def __init__(self, temperature: float = 0.07, use_ddp: bool = False, loss_type: str = 'contrastive'):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature (float): Temperature parameter for scaling logits
            use_ddp (bool): Whether to use DDP-aware version of the loss
            loss_type (str): Type of loss to use ('contrastive' or 'siglip')
        """
        super().__init__()
        self.temperature = temperature
        self.use_ddp = use_ddp
        self.loss_type = loss_type
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        log_temp: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss between video and text features.
        
        Args:
            video_features (torch.Tensor): [B, D] video embeddings
            text_features (torch.Tensor): [B, D] text embeddings
            log_temp (torch.Tensor, optional): Override internal temperature
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Use provided temperature if given, otherwise use internal one
        temp = log_temp if log_temp is not None else self.log_temp
        
        if self.loss_type == 'siglip':
            if self.use_ddp and dist.is_initialized():
                return SiglipLossDDP()(video_features, text_features, temp)
            else:
                return SiglipLoss()(video_features, text_features, temp)
        elif self.loss_type == 'contrastive':
            if self.use_ddp and dist.is_initialized():
                return ContrastiveLossDDP()(video_features, text_features, temp)
            else:
                return ContrastiveLoss()(video_features, text_features, temp)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        

@LossRegistry.register(LossType.MSE)
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(outputs.view(-1), targets)


@LossRegistry.register(LossType.HUBER)
class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return F.huber_loss(outputs.view(-1), targets, delta=self.delta)


@LossRegistry.register(LossType.MAE)
class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(outputs.view(-1), targets)


@LossRegistry.register(LossType.RMSE)
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(outputs.view(-1), targets))


@LossRegistry.register(LossType.BCE_LOGIT)
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, pos_weight=None):
        """
        Args:
            weight: Optional per-element rescaling weight (PyTorch ``weight``); this does
                NOT rebalance classes.
            pos_weight: Optional weight of the positive class (PyTorch ``pos_weight``),
                e.g. negatives / positives on the train split, for rare findings.
        """
        super().__init__()
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32).reshape(-1)
        self.criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        if outputs.dim() > 1 and outputs.size(1) == 2:
            outputs: torch.Tensor = outputs[:, 1]  # Select the second item for binary classification
        elif outputs.dim() > 1 and outputs.size(1) == 1:
            outputs: torch.Tensor = outputs.squeeze(1)  # Squeeze the dimension

        # Convert targets to float type
        targets: torch.Tensor = targets.float()

        # Keep the (buffer) weights on the outputs' device: MultiHeadLoss stores its
        # per-head criteria in a plain dict, so `.to(device)` never reaches them.
        for name in ("weight", "pos_weight"):
            value = getattr(self.criterion, name)
            if value is not None and value.device != outputs.device:
                setattr(self.criterion, name, value.to(outputs.device))

        return self.criterion(outputs, targets)


@LossRegistry.register(LossType.CE)
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        outputs: torch.Tensor = outputs.squeeze()
        targets: torch.Tensor = targets.squeeze().long()
        if self.criterion.weight is not None and self.criterion.weight.device != outputs.device:
            self.criterion.weight = self.criterion.weight.to(outputs.device)
        return self.criterion(outputs, targets)


@LossRegistry.register(LossType.MULTICLASS_FOCAL)
class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma: float = gamma
        self.ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Multi-class Focal Loss"""
        ce_loss: torch.Tensor = self.ce_loss(pred, target)
        pt: torch.Tensor = torch.exp(-ce_loss)
        focal_loss: torch.Tensor = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


@LossRegistry.register(LossType.BINARY_FOCAL)
class BinaryFocalLoss(nn.Module):
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0
    ):
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Binary Focal Loss"""
        pred: torch.Tensor = pred.squeeze()
        target: torch.Tensor = target.float()

        bce_loss: torch.Tensor = self.bce_loss(pred, target)
        probs: torch.Tensor = torch.sigmoid(pred)
        pt: torch.Tensor = torch.where(target == 1, probs, 1 - probs)
        focal_weight: torch.Tensor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight: torch.Tensor = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight: torch.Tensor = focal_weight * alpha_weight

        return (focal_weight * bce_loss).mean()


@LossRegistry.register(LossType.MULTI_HEAD)
class MultiHeadLoss(nn.Module):
    def __init__(
        self,
        head_structure: dict[str, int],
        loss_structure: dict[str, str] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        head_weights: dict[str, float] = None,
        loss_weights: dict[str, torch.Tensor] = None,
        pos_weights: dict[str, torch.Tensor] = None,
        target_normalization: dict[str, dict[str, float]] = None,
        target_scale: dict[str, float] = None,
        sample_weight_source: dict[str, str] = None,
        sample_weight_tables: dict[str, torch.Tensor] = None,
        sample_weight_bin_width: float = 1.0,
    ):
        """
        Initialize Multi-Head Loss with support for both standard and focal loss.

        Args:
            head_structure (dict): Dictionary mapping head names to number of classes
            loss_structure (dict): Dictionary mapping head names to loss function names
            alpha (float): Weighting factor for focal loss (used only for focal losses)
            gamma (float): Focusing parameter for focal loss (used only for focal losses)
            head_weights (dict): Optional weights for each head in the final loss computation
            loss_weights (dict): Optional dictionary mapping head names to loss weights tensor
                (BCE: per-element ``weight``; CE: per-class ``weight``)
            pos_weights (dict): Optional dictionary mapping binary (BCE) head names to their
                ``pos_weight`` (weight of the positive class)
            target_normalization (dict): Optional {head: {"mean": m, "std": s}} for regression
                heads trained on z-scored targets. The loss compares the head output to
                ``(target - m) / s``; use ``denormalize_outputs`` to map outputs back.
            target_scale (dict): Optional head -> lambda; regression loss is computed
                on (output / lambda, target / lambda)
            sample_weight_source (dict): Optional head -> score column whose value
                selects the per-sample weight from sample_weight_tables
            sample_weight_tables (dict): score column -> 1D tensor of weights indexed
                by floor(score / sample_weight_bin_width)
            sample_weight_bin_width (float): bin width used to index the tables
        """
        super().__init__()
        self.loss_structure: dict[str, str] = dict(loss_structure or {})
        self.target_scale: dict[str, float] = dict(target_scale or {})
        self.sample_weight_source: dict[str, str] = dict(sample_weight_source or {})
        self.sample_weight_tables: dict[str, torch.Tensor] = dict(sample_weight_tables or {})
        self.sample_weight_bin_width: float = sample_weight_bin_width
        self.head_structure: dict[str, int] = head_structure
        self.head_weights: dict[str, float] = head_weights or {head: 1.0 for head in head_structure.keys()}
        self.loss_weights: dict[str, torch.Tensor] = loss_weights or {}
        self.pos_weights: dict[str, torch.Tensor] = pos_weights or {}
        self.target_normalization: dict[str, dict[str, float]] = target_normalization or {}
        both = sorted(set(self.target_normalization) & set(self.target_scale))
        if both:
            raise ValueError(
                f"Heads {both} set both normalize_regression_targets and regression_target_scale; use one."
            )

        # Initialize loss functions for each head
        if loss_structure is None:
            # Default to BCE for binary and CE for multi-class
            self.loss_fns = {}
            for head, num_classes in head_structure.items():
                weight = self.loss_weights.get(head, None)
                if num_classes == 1:
                    self.loss_fns[head] = LossRegistry.create(
                        "bce_logit", weight=weight, pos_weight=self.pos_weights.get(head)
                    )
                else:
                    self.loss_fns[head] = LossRegistry.create("ce", weight=weight)
        else:
            # Create loss functions based on specified structure
            self.loss_fns: dict[str, nn.Module] = {}
            for head, loss_name in loss_structure.items():
                kwargs: dict[str, Any] = {}
                if loss_name == LossType.BINARY_FOCAL:
                    kwargs["alpha"] = alpha
                    kwargs["gamma"] = gamma
                elif loss_name == LossType.MULTICLASS_FOCAL:
                    kwargs["gamma"] = gamma
                elif loss_name in [LossType.BCE_LOGIT, LossType.CE]:
                    kwargs["weight"] = self.loss_weights.get(head)
                if loss_name == LossType.BCE_LOGIT:
                    kwargs["pos_weight"] = self.pos_weights.get(head)
                self.loss_fns[head] = LossRegistry.get(loss_name)(**kwargs)

    def denormalize_outputs(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map z-scored regression head outputs back to target units (for metrics/predictions)."""
        if not self.target_normalization:
            return outputs
        denormalized = dict(outputs)
        for head, stats in self.target_normalization.items():
            if head in denormalized:
                denormalized[head] = denormalized[head] * stats["std"] + stats["mean"]
        return denormalized

    def forward(
        self, 
        outputs: dict[str, torch.Tensor], 
        targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the combined loss for all heads.

        Args:
            outputs (dict): Dictionary of model outputs for each head
            targets (dict): Dictionary of target values for each head

        Returns:
            total_loss (torch.Tensor): Combined loss from all heads
            losses (dict): Individual losses per head for logging
        """
        losses: dict[str, torch.Tensor] = {}

        for head_name in self.head_structure.keys():
            target = targets[head_name]
            output = outputs[head_name]

            # Mask out NaN targets (missing data)
            valid_mask = ~torch.isnan(target)
            if valid_mask.sum() == 0:
                continue

            sample_weights = self._sample_weights(head_name, targets, valid_mask)

            if not valid_mask.all():
                target = target[valid_mask]
                output = output[valid_mask]

            # z-score regression targets (the head predicts in normalized units)
            stats = self.target_normalization.get(head_name)
            if stats is not None:
                target = (target - stats["mean"]) / stats["std"]
            scale = self.target_scale.get(head_name)
            if scale is not None:
                output = output / scale
                target = target / scale

            # Compute loss using the appropriate loss function
            if sample_weights is None:
                head_loss: torch.Tensor = self.loss_fns[head_name](output, target)
            else:
                per_sample = self._per_sample_loss(head_name, output, target)
                head_loss = (sample_weights * per_sample).mean()

            # Apply head-specific weight
            losses[head_name] = head_loss
            losses['main'] = losses.get('main', 0.0) + self.head_weights[head_name] * head_loss

        # If no heads had valid targets, set main loss to 0
        if 'main' not in losses:
            device = next(iter(outputs.values())).device
            losses['main'] = torch.tensor(0.0, device=device, requires_grad=True)

        return losses

    def _sample_weights(
        self,
        head_name: str,
        targets: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Inverse-frequency weights for the valid samples of ``head_name``."""
        source = self.sample_weight_source.get(head_name)
        if source is None:
            return None
        table = self.sample_weight_tables[source]
        score = targets[source].float().view(-1)[valid_mask.view(-1)]
        table = table.to(score.device)
        bins = torch.nan_to_num(score, nan=0.0).div(self.sample_weight_bin_width).floor().long()
        weights = table[bins.clamp(0, table.numel() - 1)]
        # Samples whose source score is missing keep weight 1.
        return torch.where(torch.isnan(score), torch.ones_like(weights), weights)

    def _per_sample_loss(
        self,
        head_name: str,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Unreduced loss for ``head_name`` (used when sample weights are set)."""
        loss_name = self.loss_structure.get(head_name)
        output = output.float()
        target = target.float()
        if loss_name == LossType.MSE:
            return (output.view(-1) - target.view(-1)) ** 2
        if loss_name == LossType.HUBER:
            return F.huber_loss(
                output.view(-1), target.view(-1),
                delta=self.loss_fns[head_name].delta, reduction="none",
            )
        if loss_name == LossType.MAE:
            return (output.view(-1) - target.view(-1)).abs()
        if loss_name == LossType.CE:
            weight = self.loss_weights.get(head_name)
            return F.cross_entropy(
                output.view(target.numel(), -1), target.view(-1).long(),
                weight=None if weight is None else weight.to(output.device), reduction="none",
            )
        if loss_name == LossType.BCE_LOGIT:
            if output.dim() > 1 and output.size(1) == 2:
                output = output[:, 1]
            pos_weight = self.pos_weights.get(head_name)
            return F.binary_cross_entropy_with_logits(
                output.reshape(-1), target.view(-1),
                pos_weight=None if pos_weight is None else pos_weight.to(output.device), reduction="none",
            )
        raise ValueError(f"Sample weighting is not supported for loss '{loss_name}' (head '{head_name}')")
