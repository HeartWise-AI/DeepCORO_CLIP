import torch
import torch.nn as nn

def get_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "contrastive":
        return contrastive_loss
    else:
        raise ValueError(f"Loss function {loss_name} not found")

def contrastive_loss(
    video_features: torch.Tensor, 
    text_features: torch.Tensor, 
    log_temp: torch.Tensor = torch.log(torch.tensor(0.1))
) -> torch.Tensor:
    """Compute contrastive loss between video and text embeddings.

    Args:
        video_features (torch.Tensor): Video embeddings of shape [batch_size, output_dim]
        text_features (torch.Tensor): Text embeddings of shape [batch_size, output_dim]
        temp (float): Temperature parameter for scaling logits

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Normalize features
    video_features = nn.functional.normalize(video_features, dim=1)
    text_features = nn.functional.normalize(text_features, dim=1)

    similarity_matrix = torch.matmul(video_features, text_features.t())

    # Scale by temperature (lower temp => sharper distribution)
    similarity_matrix = similarity_matrix / torch.exp(log_temp)

    # Create labels for matching pairs
    labels = torch.arange(len(video_features), device=video_features.device)

    # Compute loss in both directions and average
    loss_v = nn.CrossEntropyLoss()(similarity_matrix, labels)
    loss_t = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)
    loss = (loss_v + loss_t) / 2

    return loss