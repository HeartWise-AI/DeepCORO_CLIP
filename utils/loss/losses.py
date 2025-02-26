import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
        # Normalize embeddings.
        video_features = F.normalize(video_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix of shape [B, B].
        similarity_matrix = torch.matmul(video_features, text_features.t())
        
        # Apply temperature scaling.
        temp = torch.exp(log_temp)
        logits = similarity_matrix / temp

        # Create targets: assume the i-th video corresponds to the i-th text.
        targets = torch.arange(logits.size(0), device=logits.device)

        # Compute cross-entropy loss in both directions.
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
        # 1) Gather features from all GPUs.
        video_features_all = gather_all(video_features)
        text_features_all  = gather_all(text_features)

        # 2) Normalize the gathered features.
        video_features_all = F.normalize(video_features_all, dim=1)
        text_features_all  = F.normalize(text_features_all, dim=1)

        # 3) Compute global similarity matrix of shape [N, N],
        # where N is the total global batch size.
        similarity_matrix = torch.matmul(video_features_all, text_features_all.t())

        # 4) Apply temperature scaling.
        temp = torch.exp(log_temp)
        logits = similarity_matrix / temp

        # 5) Create targets: assume that the matching pairs lie along the diagonal.
        n = logits.size(0)  # global batch size
        targets = torch.arange(n, device=logits.device)

        # Compute cross-entropy losses in both directions.
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
    def __init__(self, temperature: float = 0.07, use_ddp: bool = False):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature (float): Temperature parameter for scaling logits
            use_ddp (bool): Whether to use DDP-aware version of the loss
        """
        super().__init__()
        self.temperature = temperature
        self.use_ddp = use_ddp
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
        
        if self.use_ddp and dist.is_initialized():
            return ContrastiveLossDDP()(video_features, text_features, temp)
        else:
            return ContrastiveLoss()(video_features, text_features, temp)
        

@LossRegistry.register(LossType.MSE)
class MSELoss(nn.Module):
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(outputs.view(-1), targets)


@LossRegistry.register(LossType.HUBER)
class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return F.huber_loss(outputs.view(-1), targets, delta=self.delta)


@LossRegistry.register(LossType.L1)
class L1Loss(nn.Module):
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(outputs.view(-1), targets)


@LossRegistry.register(LossType.RMSE)
class RMSELoss(nn.Module):
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(outputs.view(-1), targets))


@LossRegistry.register(LossType.BCE_LOGIT)
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        if weight is not None and isinstance(weight, torch.Tensor):
            weight = weight.to(weight.device)
        self.criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(weight=weight)

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        if outputs.dim() > 1 and outputs.size(1) == 2:
            outputs: torch.Tensor = outputs[:, 1]  # Select the second item for binary classification
        elif outputs.dim() > 1 and outputs.size(1) == 1:
            outputs: torch.Tensor = outputs.squeeze()  # Squeeze the dimension

        # Convert targets to float type
        targets: torch.Tensor = targets.float()

        return self.criterion(outputs, targets)


@LossRegistry.register(LossType.CE)
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        outputs: torch.Tensor = outputs.squeeze()
        targets: torch.Tensor = targets.squeeze().long()
        return self.criterion(outputs, targets)


@LossRegistry.register(LossType.MULTICLASS_FOCAL)
class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
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
        """
        super().__init__()
        self.head_structure: dict[str, int] = head_structure
        self.head_weights: dict[str, float] = head_weights or {head: 1.0 for head in head_structure.keys()}
        self.loss_weights: dict[str, torch.Tensor] = loss_weights or {}

        # Initialize loss functions for each head
        if loss_structure is None:
            # Default to BCE for binary and CE for multi-class
            self.loss_fns = {}
            for head, num_classes in head_structure.items():
                weight = self.loss_weights.get(head, None)
                if num_classes == 1:
                    self.loss_fns[head] = LossRegistry.create("bce_logit", weight=weight)
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
                self.loss_fns[head] = LossRegistry.get(loss_name)(**kwargs)

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
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
        total_loss: float = 0.0

        for head_name in self.head_structure.keys():
            # Compute loss using the appropriate loss function
            head_loss: torch.Tensor = self.loss_fns[head_name](outputs[head_name], targets[head_name])

            # Apply head-specific weight
            head_weight: float = self.head_weights[head_name]
            losses[head_name] = head_loss
            total_loss += head_weight * head_loss

        return total_loss, losses