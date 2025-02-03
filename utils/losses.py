import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def get_loss_fn(loss_name: str) -> nn.Module:
    """
    Returns the appropriate loss function by name.
    If running on a single GPU, use contrastive_loss;
    if using DDP, you might choose 'contrastive_ddp'.
    """
    if loss_name == "contrastive":
        return contrastive_loss
    elif loss_name == "contrastive_ddp":
        return contrastive_loss_ddp
    else:
        raise ValueError(f"Loss function {loss_name} not found")

def contrastive_loss(
    video_features: torch.Tensor, 
    text_features: torch.Tensor, 
    log_temp: torch.Tensor = torch.log(torch.tensor(0.1))
) -> torch.Tensor:
    """
    Compute CLIP-style bidirectional contrastive loss for video and text embeddings.
    
    Args:
        video_features (torch.Tensor): [B, D] video embeddings.
        text_features (torch.Tensor): [B, D] text embeddings.
        log_temp (torch.Tensor): Log temperature scalar.
        
    Returns:
        torch.Tensor: Scalar loss value.
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

def contrastive_loss_ddp(
    video_features: torch.Tensor,
    text_features: torch.Tensor,
    log_temp: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CLIP-style bidirectional contrastive loss in a DDP setting.
    This function gathers the local features from all GPUs (with gradient support),
    then computes a global [N, N] similarity matrix (N = global batch size).
    
    Args:
        video_features (torch.Tensor): [B, D] video embeddings (local batch).
        text_features (torch.Tensor): [B, D] text embeddings (local batch).
        log_temp (torch.Tensor): Log temperature scalar.
    
    Returns:
        torch.Tensor: Scalar loss value.
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