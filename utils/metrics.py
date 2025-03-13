import math
import torch
import numpy as np
import torch.nn as nn
from typing import List
from sklearn.metrics import roc_curve


def compute_recall_at_k(similarity_matrix, global_gt_indices, k_values=[1, 5]):
    """
    Compute recall@k for video->text retrieval.
    
    Args:
        similarity_matrix: Tensor of shape (n_videos, n_unique_texts) containing similarity scores.
        global_gt_indices: Tensor of shape (n_videos,) containing the index of the 
                         correct text for each video in the unique texts list.
        k_values: List of k values to compute recall for
        
    Returns:
        Dictionary containing recall scores for each k value
    """
    metrics = {}
    num_candidates = similarity_matrix.size(1)
    for k in k_values:
        # If there are fewer candidates than k, adjust k to avoid the error.
        if num_candidates < k:
            print(f"Warning: similarity matrix has only {num_candidates} candidates; adjusting Recall@{k} to Recall@{num_candidates}.")
            k_use = num_candidates
        else:
            k_use = k
        # Get the indices of the top-k candidates.
        v2t_topk = torch.topk(similarity_matrix, k_use, dim=1)[1]  # shape: [n_videos, k_use]
        # Compare with ground truth indices.
        v2t_correct = (v2t_topk == global_gt_indices.unsqueeze(1))
        recall = (v2t_correct.sum(dim=1) > 0).float().mean().item()
        metrics[f"Recall@{k}"] = recall
    return metrics



def compute_mrr(
    similarity_matrix: torch.Tensor, 
    global_gt_indices: torch.Tensor
) -> dict[str, float]:
    # Video to Text
    target_scores: torch.Tensor = similarity_matrix.gather(1, global_gt_indices.unsqueeze(1))
    v2t_ranks: torch.Tensor = (similarity_matrix >= target_scores).sum(1).float()
    v2t_mrr: float = (1 / v2t_ranks).mean().item()

    return {"MRR_V2T": v2t_mrr}

def compute_similarity_matrix(video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    normalized_video: torch.Tensor = nn.functional.normalize(video_features, dim=1)
    normalized_text: torch.Tensor = nn.functional.normalize(text_features, dim=1)
    return torch.matmul(normalized_video, normalized_text.T)

# Normalize embeddings
def compute_embedding_norms(video_features: torch.Tensor, text_features: torch.Tensor) -> dict:
    """Compute L2 norms of video and text embeddings."""
    video_norms: torch.Tensor = torch.norm(video_features, dim=1).mean().item()
    text_norms: torch.Tensor = torch.norm(text_features, dim=1).mean().item()
    return {"video_norm": video_norms, "text_norm": text_norms}


def compute_alignment_score(
    video_features: torch.Tensor,
    text_features: torch.Tensor,
    all_video_embeddings: torch.Tensor = None,
    all_text_embeddings: torch.Tensor = None,
    global_ground_truth_indices_tensor: torch.Tensor = None,
)-> float:
    """
    Compute average cosine similarity of positive pairs.

    Parameters:
    - video_features: torch.Tensor (batch local video embeddings)
    - text_features: torch.Tensor (batch local text embeddings)
    - all_video_embeddings: torch.Tensor of all validation video embeddings [N_videos, dim] (optional)
    - all_text_embeddings: torch.Tensor of all global text embeddings [N_texts, dim] (optional)
    - global_ground_truth_indices_tensor: torch.Tensor of global GT indices for each video (optional)

    If all_video_embeddings, all_text_embeddings, and global_ground_truth_indices_tensor
    are provided, compute global alignment using global embeddings.

    Otherwise, compute local alignment score assuming a one-to-one mapping between
    video_features[i] and text_features[i].
    """
    if (
        all_video_embeddings is not None
        and all_text_embeddings is not None
        and global_ground_truth_indices_tensor is not None
    ):
        # Global alignment scenario (for validation)
        correct_text_embeddings: torch.Tensor = all_text_embeddings[global_ground_truth_indices_tensor]
        normalized_video: torch.Tensor = nn.functional.normalize(all_video_embeddings, dim=1)
        normalized_text: torch.Tensor = nn.functional.normalize(correct_text_embeddings, dim=1)
        alignment_scores: torch.Tensor = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()
    else:
        # Local alignment scenario (for training)
        normalized_video: torch.Tensor = nn.functional.normalize(video_features, dim=1)
        normalized_text: torch.Tensor = nn.functional.normalize(text_features, dim=1)
        alignment_scores: torch.Tensor = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()

def compute_ndcg_at_k(
    similarity_matrix: torch.Tensor, 
    global_gt_indices: torch.Tensor, 
    k_values: List[int]
) -> dict[str, float]:
    """
    Compute NDCG@k for each query and average over all queries.
    Simplified assumption: one correct answer per query.

    Args:
        similarity_matrix (torch.Tensor): [num_queries, num_candidates]
        global_gt_indices (torch.Tensor): [num_queries], each entry is the index of the correct text
        k (int): Rank cutoff

    Returns:
        float: Average NDCG@k over all queries.
    """
    num_queries: int = similarity_matrix.size(0)
    num_candidates: int = similarity_matrix.size(1)
    if num_queries == 0:
        return 0.0

    # Sort candidates by similarity in descending order
    sorted_indices: torch.Tensor = torch.argsort(similarity_matrix, dim=1, descending=True)

    metrics: dict[str, float] = {}
    for k in k_values:
        # Adjust k if it's larger than number of candidates
        effective_k: int = min(k, num_candidates)
        
        ndcg_values: list[float] = []
        for i in range(num_queries):
            correct_idx: int = global_gt_indices[i].item()
            # Find the rank of the correct index
            ranking: torch.Tensor = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
            if ranking.numel() == 0:
                # Correct item not found (should not happen if all candidates included)
                ndcg_values.append(0.0)
                continue

            rank: int = ranking.item()
            if rank < effective_k:
                # DCG = 1 / log2(rank+2)
                dcg: float = 1.0 / math.log2(rank + 2)
            else:
                dcg: float = 0.0

            # Ideal DCG (IDCG) = 1 since there's only one relevant doc at best rank
            idcg: float = 1.0
            ndcg_values.append(dcg / idcg)

        metrics[f"NDCG@{k}_V2T"] = float(torch.tensor(ndcg_values).mean().item())

    return metrics


def compute_median_rank(
    similarity_matrix: torch.Tensor, 
    global_gt_indices: torch.Tensor
) -> int:
    """
    Compute the median rank of the correct item over all queries.
    Lower is better.
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    ranks = []
    for i in range(num_queries):
        correct_idx = global_gt_indices[i].item()
        ranking = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
        if ranking.numel() == 0:
            # Not found, assign large rank
            ranks.append(similarity_matrix.size(1))
        else:
            rank = ranking.item() + 1  # +1 because ranks are 1-based
            ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)
    median_rank = int(ranks.median().item())  # Convert to int before returning
    return median_rank


def compute_map(similarity_matrix: torch.Tensor, global_gt_indices: torch.Tensor) -> float:
    """
    Compute mean average precision (MAP).
    Assuming exactly one relevant doc per query.
    AP = 1/rank_of_correct_item
    MAP = average of AP over all queries
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0.0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    aps = []
    for i in range(num_queries):
        correct_idx = global_gt_indices[i].item()
        ranking = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
        if ranking.numel() == 0:
            # Correct not found, AP=0
            aps.append(0.0)
        else:
            rank = ranking.item() + 1
            ap = 1.0 / rank
            aps.append(ap)

    return float(torch.tensor(aps).mean().item())


def compute_best_threshold(
    df_gt_col: list[float], 
    df_pred_col: list[float]
) -> float:
    """
    Compute the best threshold using the Youden Index.

    Args:
        df_gt_col (pd.Series): Ground truth values for a specific column.
        df_pred_col (pd.Series): Predicted values for a specific column.

    Returns:
        float: The best threshold value.
    """
    fpr, tpr, roc_thresholds = roc_curve(df_gt_col, df_pred_col)
    
    # Compute Youden Index
    youden_index = tpr - fpr
    best_threshold = roc_thresholds[np.argmax(youden_index)]
    
    return float(best_threshold) # convert to float instead of numpy.float32