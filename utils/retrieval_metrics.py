import math
import torch
import torch.nn as nn

from typing import Iterable, List, Sequence, Set, Union


def _normalize_ground_truth_sets(
    ground_truth_indices: Union[torch.Tensor, Sequence, Iterable],
    num_queries: int,
) -> List[Set[int]]:
    """
    Convert ground-truth specifications into a list of integer index sets.

    Accepts tensors (1D or 2D), iterables of indices, or iterables of iterables.
    Each query is represented by a set of acceptable text indices.
    """
    gt_sets: List[Set[int]] = []

    if isinstance(ground_truth_indices, torch.Tensor):
        if ground_truth_indices.ndim == 1:
            gt_sets = [{int(idx)} for idx in ground_truth_indices.tolist()]
        elif ground_truth_indices.ndim == 2:
            gt_sets = [
                {int(x) for x in row if x is not None and int(x) >= 0}
                for row in ground_truth_indices.tolist()
            ]
        else:
            raise ValueError(
                "ground_truth_indices tensor must be 1D or 2D for multi-label support"
            )
    elif isinstance(ground_truth_indices, (list, tuple)):
        for entry in ground_truth_indices:
            if isinstance(entry, (list, tuple, set)):
                normalized = {
                    int(x)
                    for x in entry
                    if x is not None and int(x) >= 0
                }
                gt_sets.append(normalized)
            elif entry is None:
                gt_sets.append(set())
            else:
                gt_sets.append({int(entry)})
    else:
        raise TypeError(
            f"Unsupported ground_truth_indices type: {type(ground_truth_indices)}"
        )

    if len(gt_sets) < num_queries:
        gt_sets.extend([set() for _ in range(num_queries - len(gt_sets))])
    elif len(gt_sets) > num_queries:
        gt_sets = gt_sets[:num_queries]

    cleaned: List[Set[int]] = []
    for gt in gt_sets:
        cleaned.append({idx for idx in gt if idx is not None and idx >= 0})

    if not cleaned:
        cleaned = [set() for _ in range(num_queries)]

    return cleaned


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
    k_values: List[int] = [1, 5],
) -> dict[str, float]:
    """
    Compute recall@k for video→text retrieval with multi-label support.
    """
    gt_sets = _normalize_ground_truth_sets(global_gt_indices, similarity_matrix.size(0))

    metrics = {}
    num_candidates = similarity_matrix.size(1)
    for k in k_values:
        if num_candidates < k:
            print(
                f"Warning: similarity matrix has only {num_candidates} candidates; "
                f"adjusting Recall@{k} to Recall@{num_candidates}."
            )
            k_use = num_candidates
        else:
            k_use = k

        v2t_topk = torch.topk(similarity_matrix, k_use, dim=1)[1]

        hits = []
        for row_idx in range(v2t_topk.size(0)):
            gt = gt_sets[row_idx] if row_idx < len(gt_sets) else set()
            if not gt:
                hits.append(0.0)
                continue
            topk_indices = v2t_topk[row_idx].tolist()
            hits.append(1.0 if any(idx in gt for idx in topk_indices) else 0.0)

        metrics[f"Recall@{k}"] = float(sum(hits) / len(hits)) if hits else 0.0
    return metrics


def compute_mrr(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
) -> dict[str, float]:
    """Compute Mean Reciprocal Rank for video-to-text retrieval."""
    try:
        if similarity_matrix.dim() != 2:
            print(
                f"Warning: similarity_matrix has {similarity_matrix.dim()} "
                "dimensions, expected 2"
            )
            return {"MRR_V2T": 0.0}

        num_videos = similarity_matrix.size(0)
        num_texts = similarity_matrix.size(1)

        similarity_matrix = torch.nan_to_num(
            similarity_matrix, nan=0.0, posinf=1e4, neginf=-1e4
        )

        if num_texts == 1:
            return {"MRR_V2T": 1.0}

        gt_sets = _normalize_ground_truth_sets(global_gt_indices, num_videos)

        ranking = torch.argsort(similarity_matrix, dim=1, descending=True)
        mrr_values = []
        for i in range(num_videos):
            gt_set = gt_sets[i] if i < len(gt_sets) else set()
            if not gt_set:
                mrr_values.append(0.0)
                continue

            best_rank = None
            for gt_idx in gt_set:
                matches = (ranking[i] == gt_idx).nonzero(as_tuple=True)[0]
                if matches.numel() > 0:
                    candidate_rank = matches[0].item() + 1  # 1-based
                    if best_rank is None or candidate_rank < best_rank:
                        best_rank = candidate_rank

            if best_rank is None or best_rank <= 0:
                mrr_values.append(0.0)
            else:
                mrr_values.append(1.0 / best_rank)

        v2t_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
        return {"MRR_V2T": v2t_mrr}

    except Exception as e:
        print(f"Error in compute_mrr: {e}")
        print(f"similarity_matrix shape: {similarity_matrix.shape}")
        if isinstance(global_gt_indices, torch.Tensor):
            print(f"global_gt_indices shape: {global_gt_indices.shape}")
        else:
            print("global_gt_indices is not a tensor")
        return {"MRR_V2T": 0.0}


def compute_similarity_matrix(video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    normalized_video: torch.Tensor = nn.functional.normalize(video_features, dim=1)
    normalized_text: torch.Tensor = nn.functional.normalize(text_features, dim=1)
    return torch.matmul(normalized_video, normalized_text.T)


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
) -> float:
    """
    Compute average cosine similarity of positive pairs.
    """
    if (
        all_video_embeddings is not None
        and all_text_embeddings is not None
        and global_ground_truth_indices_tensor is not None
    ):
        correct_text_embeddings: torch.Tensor = all_text_embeddings[
            global_ground_truth_indices_tensor
        ]
        normalized_video: torch.Tensor = nn.functional.normalize(
            all_video_embeddings, dim=1
        )
        normalized_text: torch.Tensor = nn.functional.normalize(
            correct_text_embeddings, dim=1
        )
        alignment_scores: torch.Tensor = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()
    else:
        normalized_video: torch.Tensor = nn.functional.normalize(video_features, dim=1)
        normalized_text: torch.Tensor = nn.functional.normalize(text_features, dim=1)
        alignment_scores: torch.Tensor = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()


def compute_ndcg_at_k(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
    k_values: List[int],
) -> dict[str, float]:
    """
    Compute NDCG@k for each query with multi-label support.
    """
    num_queries: int = similarity_matrix.size(0)
    num_candidates: int = similarity_matrix.size(1)
    if num_queries == 0:
        return {}

    gt_sets = _normalize_ground_truth_sets(global_gt_indices, num_queries)
    sorted_indices: torch.Tensor = torch.argsort(
        similarity_matrix, dim=1, descending=True
    )

    metrics: dict[str, float] = {}
    for k in k_values:
        effective_k: int = min(k, num_candidates)
        ndcg_values: List[float] = []

        for i in range(num_queries):
            gt_set = gt_sets[i] if i < len(gt_sets) else set()
            if not gt_set:
                ndcg_values.append(0.0)
                continue

            dcg = 0.0
            for rank_idx in range(effective_k):
                candidate_idx = sorted_indices[i, rank_idx].item()
                if candidate_idx in gt_set:
                    dcg += 1.0 / math.log2(rank_idx + 2)

            ideal_hits = min(len(gt_set), effective_k)
            if ideal_hits == 0:
                ndcg_values.append(0.0)
                continue

            idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f"NDCG@{k}_V2T"] = float(torch.tensor(ndcg_values).mean().item())

    return metrics


def compute_median_rank(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
) -> int:
    """
    Compute the median rank of the best-matching relevant item over all queries.
    """
    num_queries = similarity_matrix.size(0)
    num_candidates = similarity_matrix.size(1)
    if num_queries == 0:
        return 0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    gt_sets = _normalize_ground_truth_sets(global_gt_indices, num_queries)
    ranks = []
    for i in range(num_queries):
        gt_set = gt_sets[i] if i < len(gt_sets) else set()
        if not gt_set:
            ranks.append(num_candidates)
            continue

        best_rank = None
        for gt_idx in gt_set:
            ranking = (sorted_indices[i] == gt_idx).nonzero(as_tuple=True)[0]
            if ranking.numel() > 0:
                candidate_rank = ranking[0].item() + 1  # 1-based
                if best_rank is None or candidate_rank < best_rank:
                    best_rank = candidate_rank

        ranks.append(best_rank if best_rank is not None else num_candidates)

    ranks_tensor = torch.tensor(ranks, dtype=torch.float)
    return int(ranks_tensor.median().item())


def compute_map(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
) -> float:
    """
    Compute mean average precision with support for multiple relevant items.
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0.0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    gt_sets = _normalize_ground_truth_sets(global_gt_indices, num_queries)
    aps: List[float] = []

    for i in range(num_queries):
        gt_set = gt_sets[i] if i < len(gt_sets) else set()
        if not gt_set:
            aps.append(0.0)
            continue

        hits = 0
        precision_sum = 0.0
        for rank_idx, cand_idx_tensor in enumerate(sorted_indices[i], start=1):
            cand_idx = cand_idx_tensor.item()
            if cand_idx in gt_set:
                hits += 1
                precision_sum += hits / rank_idx

        if hits > 0:
            aps.append(precision_sum / hits)
        else:
            aps.append(0.0)

    return float(torch.tensor(aps).mean().item())
