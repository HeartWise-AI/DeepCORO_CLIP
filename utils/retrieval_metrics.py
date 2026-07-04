"""Retrieval evaluation metrics for DeepCORO-CLIP.

NOTE: All retrieval metrics in this module are 0-1 scaled (fractions, NOT
percentages). Recall@K, MRR, NDCG, MAP, RecallAny@K, PositiveCoverage@K, etc.
are returned in [0, 1]; multiply by 100 at the reporting layer if a percentage
is desired.
"""

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


def count_valid_rows(
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
    num_queries: int,
) -> tuple[int, int]:
    """
    Count rows with a non-empty ground-truth set vs. rows with an empty one.

    Returns a ``(valid, invalid)`` tuple where ``valid`` is the number of rows
    with at least one ground-truth index and ``invalid`` is the number of rows
    whose ground-truth set is empty. ``valid + invalid == num_queries``.
    """
    gt_sets = _normalize_ground_truth_sets(global_gt_indices, num_queries)
    valid = sum(1 for gt in gt_sets if len(gt) > 0)
    invalid = num_queries - valid
    return valid, invalid


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
    k_values: List[int] = [1, 5],
    valid_only: bool = False,
) -> dict[str, float]:
    """
    Compute recall@k for video→text retrieval with multi-label support.

    Returns Recall@K as a fraction in [0, 1].

    If ``valid_only`` is False (default, backward-compatible) every row counts
    toward the denominator and rows with an empty ground-truth set contribute a
    0.0 miss (mixing data-quality penalty with model performance). If
    ``valid_only`` is True, rows with an empty ground-truth set are excluded
    from both numerator and denominator so the score reflects model
    performance only.
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
                if valid_only:
                    continue
                hits.append(0.0)
                continue
            topk_indices = v2t_topk[row_idx].tolist()
            hits.append(1.0 if any(idx in gt for idx in topk_indices) else 0.0)

        metrics[f"Recall@{k}"] = float(sum(hits) / len(hits)) if hits else 0.0
    return metrics


def compute_recall_at_k_valid_only(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
    k_values: List[int] = [1, 5],
) -> dict[str, float]:
    """
    Recall@k for video→text retrieval excluding rows with an empty
    ground-truth set from the denominator. Thin wrapper around
    ``compute_recall_at_k(..., valid_only=True)``. Returns fractions in [0, 1].
    """
    return compute_recall_at_k(
        similarity_matrix, global_gt_indices, k_values, valid_only=True
    )


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

        # NOTE: The previous early-return for ``num_texts == 1`` (returning a
        # fake perfect MRR_V2T == 1.0) was removed. The normal ranking logic
        # below correctly handles the single-candidate case and only credits a
        # rank when the ground-truth set is actually valid.
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


def _build_relevance_matrix(
    gt_sets: List[set], num_queries: int, num_candidates: int, device
) -> torch.Tensor:
    """Build a `[num_queries, num_candidates]` 0/1 relevance tensor."""
    rel = torch.zeros((num_queries, num_candidates), dtype=torch.bool, device=device)
    rows: List[int] = []
    cols: List[int] = []
    for q in range(num_queries):
        gt_set = gt_sets[q] if q < len(gt_sets) else set()
        for idx in gt_set:
            if 0 <= idx < num_candidates:
                rows.append(q)
                cols.append(idx)
    if rows:
        row_t = torch.as_tensor(rows, dtype=torch.long, device=device)
        col_t = torch.as_tensor(cols, dtype=torch.long, device=device)
        rel[row_t, col_t] = True
    return rel


def compute_median_rank(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
) -> int:
    """
    Vectorized median rank of the best-matching relevant item per query.
    """
    num_queries = similarity_matrix.size(0)
    num_candidates = similarity_matrix.size(1)
    if num_queries == 0:
        return 0

    device = similarity_matrix.device
    gt_sets = _normalize_ground_truth_sets(global_gt_indices, num_queries)
    relevance = _build_relevance_matrix(gt_sets, num_queries, num_candidates, device)
    if not relevance.any():
        return num_candidates

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    relevance_sorted = relevance.gather(1, sorted_indices)
    # Position (1-indexed) of the first hit per row; if no hit, fall back to num_candidates.
    has_hit = relevance_sorted.any(dim=1)
    first_hit = torch.argmax(relevance_sorted.to(torch.int8), dim=1) + 1
    ranks = torch.where(
        has_hit,
        first_hit,
        torch.full_like(first_hit, num_candidates),
    ).to(torch.float32)
    return int(ranks.median().item())


def compute_map(
    similarity_matrix: torch.Tensor,
    global_gt_indices: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
) -> float:
    """
    Vectorized mean average precision with support for multiple relevant items.
    """
    num_queries = similarity_matrix.size(0)
    num_candidates = similarity_matrix.size(1)
    if num_queries == 0:
        return 0.0

    device = similarity_matrix.device
    gt_sets = _normalize_ground_truth_sets(global_gt_indices, num_queries)
    relevance = _build_relevance_matrix(gt_sets, num_queries, num_candidates, device).to(torch.float32)
    if relevance.sum() == 0:
        return 0.0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    relevance_sorted = relevance.gather(1, sorted_indices)
    cum_hits = relevance_sorted.cumsum(dim=1)
    ranks = torch.arange(1, num_candidates + 1, device=device, dtype=torch.float32)
    precision_at_k = cum_hits / ranks.unsqueeze(0)
    # Average precision per query = sum of (precision@k * relevant@k) / total_relevant.
    total_relevant = relevance.sum(dim=1)
    per_query_ap = torch.where(
        total_relevant > 0,
        (precision_at_k * relevance_sorted).sum(dim=1) / total_relevant.clamp(min=1.0),
        torch.zeros_like(total_relevant),
    )
    valid_mask = total_relevant > 0
    if valid_mask.any():
        return float(per_query_ap[valid_mask].mean().item())
    return 0.0


def weighted_ndcg_at_k(
    similarity: torch.Tensor,
    positive_weights: torch.Tensor,
    k_values: List[int],
) -> dict[str, float]:
    """
    Severity-weighted NDCG@k (vectorized, on ``similarity.device``).

    Args:
        similarity: ``[N, C]`` similarity scores (video -> candidate).
        positive_weights: ``[N, C]`` non-negative relevance/severity gains.
            Negative entries are clamped to 0.
        k_values: cutoffs.

    For each k: take topk indices of ``similarity``, gather the corresponding
    gains, ``dcg = sum(gain * 1/log2(i+2))``. The ideal DCG (``idcg``) uses the
    per-row gains sorted descending. ``ndcg = dcg / idcg`` averaged over rows
    whose total gain is > 0; ``nan`` if no such row exists.

    Returns ``{f"WeightedNDCG@{k}": float}`` in [0, 1] (or nan).
    """
    if similarity.dim() != 2:
        return {f"WeightedNDCG@{k}": float("nan") for k in k_values}

    device = similarity.device
    sim = torch.nan_to_num(similarity, nan=0.0, posinf=1e4, neginf=-1e4)
    gains = positive_weights.to(device=device, dtype=torch.float32).clamp_min(0.0)

    num_candidates = sim.size(1)
    row_has_gain = gains.sum(dim=1) > 0

    metrics: dict[str, float] = {}
    for k in k_values:
        k_use = max(1, min(int(k), num_candidates))
        discounts = 1.0 / torch.log2(
            torch.arange(2, k_use + 2, device=device, dtype=torch.float32)
        )

        topk_idx = torch.topk(sim, k_use, dim=1)[1]
        topk_gains = torch.gather(gains, 1, topk_idx)
        dcg = (topk_gains * discounts.unsqueeze(0)).sum(dim=1)

        ideal_gains = torch.sort(gains, dim=1, descending=True)[0][:, :k_use]
        idcg = (ideal_gains * discounts.unsqueeze(0)).sum(dim=1)

        ndcg = torch.where(
            idcg > 0,
            dcg / idcg.clamp_min(torch.finfo(torch.float32).eps),
            torch.zeros_like(dcg),
        )
        if row_has_gain.any():
            metrics[f"WeightedNDCG@{k}"] = float(ndcg[row_has_gain].mean().item())
        else:
            metrics[f"WeightedNDCG@{k}"] = float("nan")
    return metrics


def compute_recall_at_k_t2v(
    similarity_v2t: torch.Tensor,
    gt_video_sets: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
    k_values: List[int] = [1, 5],
) -> dict[str, float]:
    """
    Text-to-video RecallAny@K (multi-positive).

    ``similarity_v2t`` is the ``[num_videos, num_texts]`` video->text matrix; it
    is transposed internally to ``[num_texts, num_videos]`` so each text query
    is ranked against all videos.

    ``gt_video_sets`` gives, per text, the SET of acceptable ground-truth video
    indices (multi-positive). Accepted formats mirror
    ``_normalize_ground_truth_sets``: a 1D/2D tensor, an iterable of indices, or
    an iterable of iterables; entry ``i`` corresponds to text ``i``.

    A text is a hit at k if ANY of its ground-truth videos appears in the top-k
    ranked videos. Rows with an empty ground-truth set contribute 0.0. Returns
    ``{f"RecallAny@{k}": float}`` in [0, 1].
    """
    if similarity_v2t.dim() != 2:
        return {f"RecallAny@{k}": 0.0 for k in k_values}

    similarity_t2v = similarity_v2t.transpose(0, 1).contiguous()
    num_texts = similarity_t2v.size(0)
    num_videos = similarity_t2v.size(1)
    gt_sets = _normalize_ground_truth_sets(gt_video_sets, num_texts)

    metrics: dict[str, float] = {}
    for k in k_values:
        k_use = max(1, min(int(k), num_videos)) if num_videos > 0 else 0
        if k_use == 0:
            metrics[f"RecallAny@{k}"] = 0.0
            continue
        topk = torch.topk(similarity_t2v, k_use, dim=1)[1]
        hits = []
        for row_idx in range(num_texts):
            gt = gt_sets[row_idx] if row_idx < len(gt_sets) else set()
            if not gt:
                hits.append(0.0)
                continue
            topk_indices = topk[row_idx].tolist()
            hits.append(1.0 if any(idx in gt for idx in topk_indices) else 0.0)
        metrics[f"RecallAny@{k}"] = float(sum(hits) / len(hits)) if hits else 0.0
    return metrics


def _clamp_k(k: int, num_candidates: int) -> int:
    """Clamp a cutoff ``k`` into ``[1, num_candidates]`` (0 if no candidates)."""
    if num_candidates <= 0:
        return 0
    return max(1, min(int(k), num_candidates))


def recall_any_at_k(
    similarity: torch.Tensor,
    positive_mask: torch.Tensor,
    k_values: List[int],
) -> dict[str, float]:
    """
    Multi-positive RecallAny@K.

    Args:
        similarity: ``[N, C]`` similarity scores.
        positive_mask: ``[N, C]`` boolean/0-1 mask of positive candidates.
        k_values: cutoffs.

    A row is a hit at k if ANY positive falls in the top-k. The mean is taken
    over rows with at least one positive; ``nan`` if no row has a positive.

    Returns ``{f"RecallAny@{k}": float}`` in [0, 1] (or nan).
    """
    if similarity.dim() != 2:
        return {f"RecallAny@{k}": float("nan") for k in k_values}

    device = similarity.device
    sim = torch.nan_to_num(similarity, nan=0.0, posinf=1e4, neginf=-1e4)
    pos = positive_mask.to(device=device, dtype=torch.bool)
    num_candidates = sim.size(1)
    row_has_pos = pos.any(dim=1)

    metrics: dict[str, float] = {}
    for k in k_values:
        k_use = _clamp_k(k, num_candidates)
        if k_use == 0 or not row_has_pos.any():
            metrics[f"RecallAny@{k}"] = float("nan")
            continue
        topk_idx = torch.topk(sim, k_use, dim=1)[1]
        topk_pos = torch.gather(pos, 1, topk_idx)
        hit = topk_pos.any(dim=1).to(torch.float32)
        metrics[f"RecallAny@{k}"] = float(hit[row_has_pos].mean().item())
    return metrics


def positive_coverage_at_k(
    similarity: torch.Tensor,
    positive_mask: torch.Tensor,
    k_values: List[int],
) -> dict[str, float]:
    """
    Fraction of a row's positives recovered within the top-k.

    For each valid row (>=1 positive): ``recovered_positives_in_topk / total_positives``,
    averaged over valid rows. ``nan`` if no valid row.

    Returns ``{f"PositiveCoverage@{k}": float}`` in [0, 1] (or nan).
    """
    if similarity.dim() != 2:
        return {f"PositiveCoverage@{k}": float("nan") for k in k_values}

    device = similarity.device
    sim = torch.nan_to_num(similarity, nan=0.0, posinf=1e4, neginf=-1e4)
    pos = positive_mask.to(device=device, dtype=torch.bool)
    num_candidates = sim.size(1)
    total_pos = pos.sum(dim=1).to(torch.float32)
    row_has_pos = total_pos > 0

    metrics: dict[str, float] = {}
    for k in k_values:
        k_use = _clamp_k(k, num_candidates)
        if k_use == 0 or not row_has_pos.any():
            metrics[f"PositiveCoverage@{k}"] = float("nan")
            continue
        topk_idx = torch.topk(sim, k_use, dim=1)[1]
        recovered = torch.gather(pos, 1, topk_idx).sum(dim=1).to(torch.float32)
        coverage = recovered / total_pos.clamp_min(1.0)
        metrics[f"PositiveCoverage@{k}"] = float(coverage[row_has_pos].mean().item())
    return metrics


def masked_alignment_score(
    video_features: torch.Tensor,
    text_features: torch.Tensor,
    positive_mask: torch.Tensor,
    positive_weights: torch.Tensor = None,
) -> float:
    """
    Mean (optionally weighted) cosine similarity of positive video-text pairs.

    Features are L2-normalized, ``sim = video @ text.T``. For each video the
    per-video score is the positive-masked weighted mean
    ``(sim * weights).sum(1) / weights.sum(1)`` where ``weights`` defaults to
    the boolean ``positive_mask``. The final score is the mean over videos that
    have at least one positive; ``nan`` if none.
    """
    if video_features.dim() != 2 or text_features.dim() != 2:
        return float("nan")

    device = video_features.device
    norm_v = nn.functional.normalize(video_features, dim=1)
    norm_t = nn.functional.normalize(text_features, dim=1)
    sim = torch.matmul(norm_v, norm_t.T)

    pos = positive_mask.to(device=device, dtype=torch.float32)
    if positive_weights is not None:
        weights = positive_weights.to(device=device, dtype=torch.float32).clamp_min(0.0)
        weights = weights * pos
    else:
        weights = pos

    weight_sum = weights.sum(dim=1)
    row_valid = weight_sum > 0
    if not row_valid.any():
        return float("nan")

    per_video = (sim * weights).sum(dim=1) / weight_sum.clamp_min(
        torch.finfo(torch.float32).eps
    )
    return float(per_video[row_valid].mean().item())


def compute_multi_positive_recall_at_k(
    similarity: torch.Tensor,
    positive_mask: torch.Tensor,
    k_values: List[int],
) -> dict[str, float]:
    """
    Multi-positive Recall@K: any-positive-hit in the top-k.

    For each row with >=1 positive, the row scores 1 if ANY positive is within
    the top-k, else 0; the mean is taken over those rows. ``nan`` if no row has
    a positive.

    Returns ``{f"Recall@{k}": float}`` in [0, 1] (or nan).
    """
    if similarity.dim() != 2:
        return {f"Recall@{k}": float("nan") for k in k_values}

    device = similarity.device
    sim = torch.nan_to_num(similarity, nan=0.0, posinf=1e4, neginf=-1e4)
    pos = positive_mask.to(device=device, dtype=torch.bool)
    num_candidates = sim.size(1)
    row_has_pos = pos.any(dim=1)

    metrics: dict[str, float] = {}
    for k in k_values:
        k_use = _clamp_k(k, num_candidates)
        if k_use == 0 or not row_has_pos.any():
            metrics[f"Recall@{k}"] = float("nan")
            continue
        topk_idx = torch.topk(sim, k_use, dim=1)[1]
        topk_pos = torch.gather(pos, 1, topk_idx)
        hit = topk_pos.any(dim=1).to(torch.float32)
        metrics[f"Recall@{k}"] = float(hit[row_has_pos].mean().item())
    return metrics


def best_positive_rank(
    similarity_row: torch.Tensor,
    positive_mask_row: torch.Tensor,
) -> int:
    """
    1-based rank of the first (highest-similarity) positive for a single row.

    Returns ``num_candidates + 1`` if the row has no positive (or no
    candidates).
    """
    sim = similarity_row.reshape(-1)
    pos = positive_mask_row.reshape(-1).to(torch.bool)
    num_candidates = sim.size(0)
    if num_candidates == 0 or not pos.any():
        return num_candidates + 1

    sim = torch.nan_to_num(sim, nan=0.0, posinf=1e4, neginf=-1e4)
    order = torch.argsort(sim, descending=True)
    pos_sorted = pos[order]
    first = int(torch.argmax(pos_sorted.to(torch.int8)).item())
    return first + 1
