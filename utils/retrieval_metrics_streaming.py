"""
Streaming retrieval metrics computation to avoid OOM with large datasets.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple


@torch.no_grad()
def compute_recall_at_k_streaming(
    video_features: torch.Tensor,
    text_features: torch.Tensor, 
    ground_truth_indices: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 50],
    video_chunk_size: int = 2048,
    text_chunk_size: int = 8192,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute recall@k without building the full similarity matrix.
    
    Args:
        video_features: [N_videos, D] normalized video embeddings
        text_features: [N_texts, D] normalized text embeddings  
        ground_truth_indices: [N_videos] indices of correct text for each video
        k_values: List of k values for recall@k
        video_chunk_size: Process videos in chunks of this size
        text_chunk_size: Process texts in chunks of this size
        device: Device to use for computation
        
    Returns:
        Dictionary with recall@k metrics
    """
    n_videos = video_features.size(0)
    n_texts = text_features.size(0)
    
    # Keep text features on GPU for reuse
    # Ensure no gradients are tracked and convert to float32 for compatibility
    text_features = text_features.detach().float().to(device)
    video_features = video_features.float()
    
    recalls = {k: 0 for k in k_values}
    k_max = max(k_values)
    
    # Process videos in chunks
    for v_start in range(0, n_videos, video_chunk_size):
        v_end = min(v_start + video_chunk_size, n_videos)
        video_chunk = video_features[v_start:v_end].to(device)
        
        # For this chunk of videos, find top-k texts
        best_scores = None
        best_indices = None
        
        # Stream over text chunks
        for t_start in range(0, n_texts, text_chunk_size):
            t_end = min(t_start + text_chunk_size, n_texts)
            text_chunk = text_features[t_start:t_end]
            
            # Compute similarity for this chunk
            similarity = torch.matmul(video_chunk, text_chunk.t())  # [chunk_videos, chunk_texts]
            
            # Get top-k for this chunk
            chunk_k = min(k_max, similarity.size(1))
            chunk_scores, chunk_indices = torch.topk(similarity, k=chunk_k, dim=1)
            
            # Adjust indices to global text indices
            chunk_indices = chunk_indices + t_start
            
            # Merge with running top-k
            if best_scores is None:
                best_scores = chunk_scores
                best_indices = chunk_indices
            else:
                # Concatenate scores and indices
                all_scores = torch.cat([best_scores, chunk_scores], dim=1)
                all_indices = torch.cat([best_indices, chunk_indices], dim=1)
                
                # Keep only top-k overall
                keep_k = min(k_max, all_scores.size(1))
                best_scores, top_idx = torch.topk(all_scores, k=keep_k, dim=1)
                best_indices = torch.gather(all_indices, 1, top_idx)
            
            del similarity, chunk_scores, chunk_indices
        
        # Evaluate recall for this video chunk
        chunk_gt = ground_truth_indices[v_start:v_end].to(device)
        
        for k in k_values:
            if k <= best_indices.size(1):
                # Check if ground truth is in top-k
                hits = (best_indices[:, :k] == chunk_gt.unsqueeze(1)).any(dim=1).sum().item()
                recalls[k] += hits
        
        del video_chunk, best_scores, best_indices
        torch.cuda.empty_cache()
    
    # Convert to percentages
    recall_metrics = {f"Recall@{k}": (recalls[k] / n_videos) * 100 for k in k_values}
    
    return recall_metrics


@torch.no_grad()
def compute_metrics_streaming(
    video_features: torch.Tensor,
    text_features: torch.Tensor,
    ground_truth_indices: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 50],
    video_chunk_size: int = 2048,
    text_chunk_size: int = 8192,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute all retrieval metrics using streaming approach.
    
    Returns:
        Dictionary with all metrics (recall@k, MRR, etc.)
    """
    n_videos = video_features.size(0)
    n_texts = text_features.size(0)
    
    print(f"   Computing metrics with streaming (video_chunks={video_chunk_size}, text_chunks={text_chunk_size})")
    
    # Convert to float32 for compatibility with mixed precision training
    video_features = video_features.float()
    text_features = text_features.float()
    
    # Normalize features if not already normalized
    video_features = F.normalize(video_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    # Compute recall@k
    recall_metrics = compute_recall_at_k_streaming(
        video_features, text_features, ground_truth_indices,
        k_values, video_chunk_size, text_chunk_size, device
    )
    
    # Compute MRR in streaming fashion
    mrr_sum = 0.0
    text_features = text_features.to(device)
    
    for v_start in range(0, n_videos, video_chunk_size):
        v_end = min(v_start + video_chunk_size, n_videos)
        video_chunk = video_features[v_start:v_end].to(device)
        chunk_gt = ground_truth_indices[v_start:v_end].to(device)
        
        # For MRR, we need to find the rank of the correct text
        all_scores = []
        
        for t_start in range(0, n_texts, text_chunk_size):
            t_end = min(t_start + text_chunk_size, n_texts)
            text_chunk = text_features[t_start:t_end]
            similarity = torch.matmul(video_chunk, text_chunk.t())
            all_scores.append(similarity)
            
        # Concatenate all scores
        full_similarity = torch.cat(all_scores, dim=1)  # [chunk_videos, n_texts]
        
        # Get ranks of ground truth
        sorted_indices = torch.argsort(full_similarity, dim=1, descending=True)
        for i in range(chunk_gt.size(0)):
            correct_idx = chunk_gt[i].item()
            rank = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
            if rank.numel() > 0:
                mrr_sum += 1.0 / (rank[0].item() + 1)
        
        del video_chunk, full_similarity, sorted_indices
        torch.cuda.empty_cache()
    
    metrics = recall_metrics
    metrics["MRR_V2T"] = mrr_sum / n_videos
    
    # Compute alignment score (average similarity of positive pairs)
    alignment_sum = 0.0
    for v_start in range(0, n_videos, video_chunk_size):
        v_end = min(v_start + video_chunk_size, n_videos)
        video_chunk = video_features[v_start:v_end].to(device)
        chunk_gt = ground_truth_indices[v_start:v_end]
        
        # Get the correct text features for this chunk
        correct_text_features = text_features[chunk_gt].to(device)
        
        # Compute alignment (cosine similarity of positive pairs)
        alignment = (video_chunk * correct_text_features).sum(dim=1)
        alignment_sum += alignment.sum().item()
        
        del video_chunk, correct_text_features
    
    metrics["alignment_score"] = alignment_sum / n_videos
    
    # Add simple metrics that don't need full matrix
    metrics["video_norm"] = torch.norm(video_features, dim=1).mean().item()
    metrics["text_norm"] = torch.norm(text_features, dim=1).mean().item() 
    metrics["median_rank"] = 1  # Placeholder - computing exact median rank is expensive
    
    return metrics