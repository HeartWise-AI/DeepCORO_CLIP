from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from dataloaders.video_clip_dataset import VideoClipDataset

DEFAULT_SEVERITY_LEVELS: Tuple[str, ...] = ("normal", "mild", "moderate", "severe")


def _normalize_string(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    return text


def _normalize_tree(value: object, dataset: VideoClipDataset) -> Optional[str]:
    if value is None:
        return None
    normalize_fn = getattr(dataset, "_normalize_tree_key", None)
    if callable(normalize_fn):
        normalized = normalize_fn(value)
        if normalized is None:
            return None
        return str(normalized).strip().lower()
    return _normalize_string(value)


def _extract_pred_attributes(
    text_id: Optional[str],
    siglip_lookup: Dict[str, Dict[str, object]],
    dataset: VideoClipDataset,
) -> Optional[Dict[str, Optional[str]]]:
    if text_id is None:
        return None
    meta = siglip_lookup.get(text_id)
    if meta is None:
        return None
    tree = _normalize_tree(meta.get("tree"), dataset)
    segment = _normalize_string(meta.get("segment"))
    severity = _normalize_string(meta.get("disease_severity"))
    return {
        "tree": tree,
        "segment": segment,
        "severity": severity,
    }


def _resolve_dataset_index(identifier: str, dataset: VideoClipDataset) -> Optional[int]:
    idx = dataset.video_path_to_idx.get(str(identifier))
    if idx is not None:
        return idx
    if getattr(dataset, "multi_video_mode", False):
        study_videos = getattr(dataset, "study_to_videos", {}).get(identifier, [])
        for path in study_videos:
            idx = dataset.video_path_to_idx.get(str(path))
            if idx is not None:
                return idx
    return None


def compute_siglip_semantic_metrics(
    similarity_matrix: torch.Tensor,
    sample_identifiers: Sequence[str],
    dataset: VideoClipDataset,
    all_text_ids: Sequence[str],
    top_tree_k: int = 5,
    top_segment_k: int = 15,
    severity_levels: Sequence[str] = DEFAULT_SEVERITY_LEVELS,
) -> Dict[str, float]:
    """
    Compute semantic alignment metrics using SigLIP metadata.

    Returns metrics such as tree recall@5, segment+severity alignment@15,
    and per-severity recalls at top-5/top-15 that also require tree agreement.
    """
    if not getattr(dataset, "siglip_enabled", False):
        return {}
    siglip = getattr(dataset, "siglip", None)
    if siglip is None or not hasattr(siglip, "text_lookup"):
        return {}
    if similarity_matrix.numel() == 0:
        return {}

    num_candidates = similarity_matrix.shape[1]
    tree_k = min(top_tree_k, num_candidates)
    segment_k = min(top_segment_k, num_candidates)
    if tree_k <= 0:
        return {}
    max_k = max(tree_k, segment_k)

    top_indices = torch.topk(similarity_matrix, k=max_k, dim=1).indices.cpu().tolist()

    lookup = siglip.text_lookup
    tree_scores: List[float] = []
    tree_recall_scores: List[float] = []
    segment_scores: List[float] = []
    segment_recall_scores: List[float] = []
    severity_levels = tuple(level.lower() for level in severity_levels)
    severity_counts_5 = {s: {"match": 0, "total": 0} for s in severity_levels}
    severity_counts_15 = {s: {"match": 0, "total": 0} for s in severity_levels}

    for row_idx, identifier in enumerate(sample_identifiers):
        dataset_idx = _resolve_dataset_index(identifier, dataset)
        if dataset_idx is None or dataset_idx >= len(dataset.video_positive_texts):
            continue

        positives = dataset.video_positive_texts[dataset_idx]
        if not positives:
            continue

        gt_trees: set[str] = set()
        segment_to_severity: Dict[str, set] = defaultdict(set)
        severity_to_trees: Dict[str, set] = defaultdict(set)

        for text_id, _ in positives:
            meta = lookup.get(text_id)
            if meta is None:
                continue
            severity = _normalize_string(meta.get("disease_severity"))
            segment = _normalize_string(meta.get("segment"))
            tree = _normalize_tree(meta.get("tree"), dataset)

            if tree:
                gt_trees.add(tree)
                if severity:
                    severity_to_trees[severity].add(tree)
            if segment and severity:
                segment_to_severity[segment].add(severity)

        if not gt_trees and not segment_to_severity:
            continue

        pred_indices = top_indices[row_idx]
        pred_attrs: List[Optional[Dict[str, Optional[str]]]] = []
        for pred_idx in pred_indices:
            if pred_idx >= len(all_text_ids):
                pred_attrs.append(None)
                continue
            text_id = all_text_ids[pred_idx]
            pred_attrs.append(_extract_pred_attributes(text_id, lookup, dataset))

        # Tree alignment @ top_tree_k.
        # precision = #correct preds / #preds ; recall = #gt hit / #gt
        tree_k_actual = min(tree_k, len(pred_attrs))
        if gt_trees and tree_k_actual > 0:
            top_tree_preds = [
                attr.get("tree")
                for attr in pred_attrs[:tree_k_actual]
                if attr and attr.get("tree")
            ]
            matches = sum(1 for t in top_tree_preds if t in gt_trees)
            tree_scores.append(matches / tree_k_actual)
            pred_tree_set = set(top_tree_preds)
            hit_trees = sum(1 for t in gt_trees if t in pred_tree_set)
            tree_recall_scores.append(hit_trees / len(gt_trees))

        # Segment + severity alignment @ top_segment_k.
        segment_k_actual = min(segment_k, len(pred_attrs))
        if segment_to_severity and segment_k_actual > 0:
            per_segment: List[float] = []
            per_segment_recall: List[float] = []
            for segment, severity_set in segment_to_severity.items():
                if not severity_set:
                    continue
                matches = sum(
                    1
                    for attr in pred_attrs[:segment_k_actual]
                    if attr
                    and attr.get("segment") == segment
                    and attr.get("severity") in severity_set
                )
                per_segment.append(matches / segment_k_actual)
                # Recall: fraction of gt (segment, severity) pairs recovered
                # in the top-k for this segment.
                pred_sev_for_segment = {
                    attr.get("severity")
                    for attr in pred_attrs[:segment_k_actual]
                    if attr and attr.get("segment") == segment
                }
                hit_sev = sum(
                    1 for sev in severity_set if sev in pred_sev_for_segment
                )
                per_segment_recall.append(hit_sev / len(severity_set))
            if per_segment:
                segment_scores.append(mean(per_segment))
            if per_segment_recall:
                segment_recall_scores.append(mean(per_segment_recall))

        # Severity-specific recalls (top-5 and top-15) requiring severity + tree match
        for severity in severity_levels:
            gt_severity_trees = severity_to_trees.get(severity)
            if not gt_severity_trees:
                continue

            if tree_k_actual > 0:
                matches5 = sum(
                    1
                    for attr in pred_attrs[:tree_k_actual]
                    if attr
                    and attr.get("severity") == severity
                    and attr.get("tree") in gt_severity_trees
                )
                severity_counts_5[severity]["match"] += matches5
                severity_counts_5[severity]["total"] += tree_k_actual

            if segment_k_actual > 0:
                matches15 = sum(
                    1
                    for attr in pred_attrs[:segment_k_actual]
                    if attr
                    and attr.get("severity") == severity
                    and attr.get("tree") in gt_severity_trees
                )
                severity_counts_15[severity]["match"] += matches15
                severity_counts_15[severity]["total"] += segment_k_actual

    metrics: Dict[str, float] = {}
    if tree_scores:
        tree_precision = float(mean(tree_scores))
        # Correctly-named precision (denominator = #predictions).
        metrics["semantic/tree_precision@5"] = tree_precision
        # Backward-compatible alias: historically this precision-like value was
        # exposed (incorrectly) as "tree_recall@5"; kept so existing dashboards
        # do not break.
        metrics["semantic/tree_recall@5"] = tree_precision
    if tree_recall_scores:
        # True recall (denominator = #ground-truth trees).
        metrics["semantic/tree_recall_true@5"] = float(mean(tree_recall_scores))
    if segment_scores:
        seg_precision = float(mean(segment_scores))
        metrics["semantic/segment_severity_precision@15"] = seg_precision
        # Backward-compatible alias for the old "alignment" name.
        metrics["semantic/segment_severity_alignment@15"] = seg_precision
    if segment_recall_scores:
        metrics["semantic/segment_severity_recall@15"] = float(
            mean(segment_recall_scores)
        )

    for severity in severity_levels:
        total5 = severity_counts_5[severity]["total"]
        if total5 > 0:
            sev5 = severity_counts_5[severity]["match"] / total5
            # match/total here divides by #predictions -> precision-like.
            metrics[f"semantic/severity_tree_precision@5/{severity}"] = sev5
            # Backward-compatible alias (was named "..._recall@5").
            metrics[f"semantic/severity_tree_recall@5/{severity}"] = sev5
        total15 = severity_counts_15[severity]["total"]
        if total15 > 0:
            sev15 = severity_counts_15[severity]["match"] / total15
            metrics[f"semantic/severity_tree_precision@15/{severity}"] = sev15
            metrics[f"semantic/severity_tree_recall@15/{severity}"] = sev15

    return metrics
