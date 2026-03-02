"""
Optimized metrics computation with sanity checks.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import gc


class OptimizedMetricsComputer:
    """
    Optimized metrics computation that ensures:
    1. No gradients are tracked
    2. Efficient memory usage
    3. Proper cleanup
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        video_chunk_size: int = 2048,
        text_chunk_size: int = 4096,
        pin_text_features: bool = True,
        clear_cache_frequency: int = 10
    ):
        self.device = device
        self.video_chunk_size = video_chunk_size
        self.text_chunk_size = text_chunk_size
        self.pin_text_features = pin_text_features
        self.clear_cache_frequency = clear_cache_frequency
        self.pinned_text_features = None
        self.pinned_unique_texts = None
        
    @torch.no_grad()
    def prepare_text_features(
        self, 
        text_features: torch.Tensor,
        texts: List[str]
    ) -> tuple:
        """
        Prepare and optionally pin text features on GPU.
        
        Returns:
            (unique_text_features, text_to_idx, ground_truth_indices)
        """
        # Sanity check: ensure no gradients
        assert not text_features.requires_grad, "Text features should not require gradients!"
        
        # Get unique texts
        unique_texts = list(dict.fromkeys(texts))
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
        
        print(f"   Preparing {len(unique_texts)} unique texts from {len(texts)} samples")
        
        # Extract features for unique texts
        unique_text_features = []
        for text in unique_texts:
            idx = texts.index(text)
            unique_text_features.append(text_features[idx])
        unique_text_features = torch.stack(unique_text_features)
        
        # Ensure features are detached and normalized
        unique_text_features = unique_text_features.detach()
        unique_text_features = F.normalize(unique_text_features, dim=1)
        
        # Pin on GPU if requested
        if self.pin_text_features:
            self.pinned_text_features = unique_text_features.to(self.device)
            self.pinned_unique_texts = unique_texts
            print(f"   Pinned text features on {self.device}")
        
        # Create ground truth indices
        ground_truth_indices = torch.tensor(
            [text_to_idx[text] for text in texts],
            dtype=torch.long
        )
        
        return unique_text_features, text_to_idx, ground_truth_indices
    
    @torch.no_grad()
    def compute_metrics_streaming(
        self,
        video_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        ground_truth_indices: Optional[torch.Tensor] = None,
        k_values: List[int] = [1, 5, 10, 50]
    ) -> Dict[str, float]:
        """
        Compute metrics using streaming approach with pinned text features.
        """
        # Sanity check
        assert not video_features.requires_grad, "Video features should not require gradients!"
        
        # Use pinned features if available
        if text_features is None and self.pinned_text_features is not None:
            text_features = self.pinned_text_features
            print("   Using pinned text features")
        else:
            # Ensure features are on device and normalized
            text_features = text_features.detach().to(self.device)
            text_features = F.normalize(text_features, dim=1)
        
        # Normalize video features
        video_features = F.normalize(video_features.detach(), dim=1)
        
        n_videos = video_features.size(0)
        n_texts = text_features.size(0)
        
        print(f"   Computing metrics: {n_videos} videos × {n_texts} texts")
        
        recalls = {k: 0 for k in k_values}
        mrr_sum = 0.0
        alignment_sum = 0.0
        
        # Process videos in chunks
        for i, v_start in enumerate(range(0, n_videos, self.video_chunk_size)):
            v_end = min(v_start + self.video_chunk_size, n_videos)
            video_chunk = video_features[v_start:v_end].to(self.device)
            chunk_gt = ground_truth_indices[v_start:v_end].to(self.device)
            
            # Find top-k texts for this video chunk
            best_scores = None
            best_indices = None
            
            # Stream over text chunks (if text features aren't pinned)
            if self.pinned_text_features is None:
                for t_start in range(0, n_texts, self.text_chunk_size):
                    t_end = min(t_start + self.text_chunk_size, n_texts)
                    text_chunk = text_features[t_start:t_end]
                    
                    # Compute similarity
                    sim = torch.matmul(video_chunk, text_chunk.t())
                    
                    # Get top-k
                    k_chunk = min(max(k_values), sim.size(1))
                    scores, indices = torch.topk(sim, k=k_chunk, dim=1)
                    indices = indices + t_start
                    
                    # Merge with running top-k
                    if best_scores is None:
                        best_scores, best_indices = scores, indices
                    else:
                        all_scores = torch.cat([best_scores, scores], dim=1)
                        all_indices = torch.cat([best_indices, indices], dim=1)
                        keep_k = min(max(k_values), all_scores.size(1))
                        best_scores, top_idx = torch.topk(all_scores, k=keep_k, dim=1)
                        best_indices = torch.gather(all_indices, 1, top_idx)
                    
                    del sim, scores, indices
            else:
                # If text features are pinned, compute all at once
                sim = torch.matmul(video_chunk, text_features.t())
                k_max = min(max(k_values), sim.size(1))
                best_scores, best_indices = torch.topk(sim, k=k_max, dim=1)
                
                # Also compute MRR from full similarity
                sorted_indices = torch.argsort(sim, dim=1, descending=True)
                for j in range(chunk_gt.size(0)):
                    correct_idx = chunk_gt[j].item()
                    rank = (sorted_indices[j] == correct_idx).nonzero(as_tuple=True)[0]
                    if rank.numel() > 0:
                        mrr_sum += 1.0 / (rank[0].item() + 1)
                
                del sim, sorted_indices
            
            # Evaluate recall
            for k in k_values:
                if k <= best_indices.size(1):
                    hits = (best_indices[:, :k] == chunk_gt.unsqueeze(1)).any(dim=1).sum().item()
                    recalls[k] += hits
            
            # Compute alignment for this chunk
            correct_text_features = text_features[chunk_gt]
            alignment = (video_chunk * correct_text_features).sum(dim=1)
            alignment_sum += alignment.sum().item()
            
            del video_chunk, best_scores, best_indices, correct_text_features
            
            # Clear cache periodically
            if i % self.clear_cache_frequency == 0:
                torch.cuda.empty_cache()
        
        # Compute final metrics
        metrics = {
            f"Recall@{k}": (recalls[k] / n_videos) * 100 for k in k_values
        }
        metrics["MRR_V2T"] = mrr_sum / n_videos
        metrics["alignment_score"] = alignment_sum / n_videos
        metrics["video_norm"] = torch.norm(video_features, dim=1).mean().item()
        metrics["text_norm"] = torch.norm(text_features, dim=1).mean().item()
        
        return metrics
    
    def clear_pinned_features(self):
        """Clear pinned features to free GPU memory."""
        self.pinned_text_features = None
        self.pinned_unique_texts = None
        torch.cuda.empty_cache()
        gc.collect()


def validate_no_gradients(*tensors):
    """
    Validate that no tensors require gradients.
    Raises AssertionError if any tensor has requires_grad=True.
    """
    for i, tensor in enumerate(tensors):
        if tensor is not None and hasattr(tensor, 'requires_grad'):
            assert not tensor.requires_grad, f"Tensor {i} should not require gradients in metrics computation!"