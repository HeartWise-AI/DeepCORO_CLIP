"""
Single-head SigLIP retrieval sampler with hamper-aware negative construction.

Implements the policy described in the system brief:
    - Per-phase positive budgets with type scaling
    - Exam severity priors and class balancing
    - Mandatory normal/mild negatives when diseased segments exist
    - Optional summary handling (CAVS-Lite)
    - Soft adjacency between adjacent disease bins (excluding <30)

The sampler consumes lightweight batch descriptions and returns:
    * Ordered text ids used this step
    * Dense label and weight matrices (Y and W)
    * Per-text metadata for downstream encoding/logit adjustment
    * An audit trail describing sampled negatives for debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict
import math
import random

import torch


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextMetadata:
    """Canonical metadata for a text entry."""

    text_id: str
    prompt_text: str
    prompt_type: Optional[str]
    category: Optional[str]
    segment: Optional[str]
    bin: Optional[str]
    tree: Optional[str]
    stent: Optional[str]
    soft_weight: float
    disease_severity: Optional[str]
    tags: Dict[str, str] = field(default_factory=dict)
    prompt_bucket: Optional[str] = None
    class_key: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = None
    logit_bias: float = 0.0
    class_weight: float = 1.0


@dataclass
class VideoBatchEntry:
    """Input description for a single video in the batch."""

    video_id: str
    exam_severity: str  # NORMAL | MILD | SEVERE
    tree: Optional[str]
    positive_pairs: List[Tuple[str, float]]  # (text_id, base_weight)
    negative_candidates: Optional[List[str]] = None


@dataclass
class SamplerOutput:
    """Structured output from the sampler."""

    text_ids: List[str]
    labels: torch.Tensor
    weights: torch.Tensor
    text_metadata: List[Dict[str, Any]]
    audit: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
\
@dataclass(frozen=True)
class NegativeCandidate:
    meta: TextMetadata
    bucket: str
    weight_key: str
    reason: str


class SingleHeadRetrievalSampler:
    """
    Construct SigLIP retrieval targets with severity-aware positive capping and
    contrastive negative sampling.
    """

    ABNORMAL_CATEGORIES = {
        "stenosis",
        "in_stent",
        "thrombus",
        "calcification",
        "cto",
        "medina",
    }
    ABNORMAL_PROMPT_BUCKETS = {"abnormal"}
    SUMMARY_BUCKETS = {"summary", "other_summary"}
    DEFAULT_POSITIVE_SEVERITY_WEIGHTS = {
        "normal": 0.25,
        "mild": 0.5,
        "moderate": 1.0,
        "severe": 1.5,
        "critical": 1.5,
        "cto": 1.5,
    }
    SEVERITY_ORDER = {
        "normal": 0,
        "mild": 1,
        "moderate": 2,
        "severe": 3,
    }

    def __init__(
        self,
        text_catalog: Dict[str, TextMetadata],
        class_weight: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float],
        logit_bias: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float],
        *,
        alpha_neg: float = 2.0,
        rng: Optional[random.Random] = None,
        max_negatives: int = 0,
        base_negative_weight: float = 0.04,
        round_robin: bool = False,
        min_pos_weight: float = 0.0,
        positive_severity_weights: Optional[Dict[str, float]] = None,
        neg_normal_scale: float = 0.25,
        neg_mild_scale: float = 0.75,
        neg_abnormal_scale: float = 1.5,
        same_segment_boost: float = 1.5,
        same_tree_boost: float = 1.25,
        contradiction_boost: float = 1.0,
        contradiction_min_severity: str = "moderate",
    ) -> None:
        self.text_catalog = text_catalog
        self.class_weight_lookup = class_weight
        self.logit_bias_lookup = logit_bias
        self.alpha_neg = alpha_neg
        self._rng = rng or random.Random(0)
        self.max_negatives = max(0, int(max_negatives))
        self.base_negative_weight = max(0.0, float(base_negative_weight))
        self.round_robin = bool(round_robin)
        self.min_pos_weight = max(0.0, float(min_pos_weight))
        self.positive_severity_weights = self._merge_positive_weights(positive_severity_weights or {})
        self.neg_normal_scale = max(0.0, float(neg_normal_scale))
        self.neg_mild_scale = max(0.0, float(neg_mild_scale))
        self.neg_abnormal_scale = max(0.0, float(neg_abnormal_scale))
        self.same_segment_boost = max(0.0, float(same_segment_boost))
        self.same_tree_boost = max(0.0, float(same_tree_boost))
        self.contradiction_boost = max(0.0, float(contradiction_boost))
        self.contradiction_min_severity = str(contradiction_min_severity or "").strip().lower()
        self._contradiction_min_rank = self._severity_rank(self.contradiction_min_severity)

        self._round_robin_state: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._pos_round_robin: Dict[str, int] = {}

        self._segment_index: Dict[str, List[TextMetadata]] = defaultdict(list)
        self._tree_meta_index: Dict[str, List[TextMetadata]] = defaultdict(list)
        self._summary_texts: List[TextMetadata] = []
        self._all_texts: List[TextMetadata] = list(self.text_catalog.values())

        self._build_indices()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def prepare_batch(
        self,
        batch_videos: List[VideoBatchEntry],
        *,
        epoch: int,
        phase: str,
    ) -> SamplerOutput:
        """Return the SigLIP matrices and metadata for the batch."""

        B = len(batch_videos)
        text_ids: List[str] = []
        per_video_entries: List[List[Tuple[str, float, float]]] = []
        audit_records: Dict[str, Dict[str, Any]] = {}

        for vid_entry in batch_videos:
            entries, audit = self._prepare_video_entries(
                vid_entry,
                epoch=epoch,
            )
            per_video_entries.append(entries)
            audit_records[vid_entry.video_id] = audit
            for text_id, _, _ in entries:
                if text_id not in text_ids:
                    text_ids.append(text_id)

        text_to_col = {tid: idx for idx, tid in enumerate(text_ids)}
        labels = torch.zeros(B, len(text_ids), dtype=torch.float32)
        weights = torch.zeros_like(labels)

        for row_idx, entries in enumerate(per_video_entries):
            for text_id, y_val, w_val in entries:
                col = text_to_col[text_id]
                labels[row_idx, col] = max(labels[row_idx, col].item(), y_val)
                weights[row_idx, col] += w_val

        text_meta = [self._metadata_dict(self.text_catalog[tid]) for tid in text_ids]
        return SamplerOutput(
            text_ids=text_ids,
            labels=labels,
            weights=weights,
            text_metadata=text_meta,
            audit={
                "videos": audit_records,
                "phase": phase,
                "epoch": epoch,
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _prepare_video_entries(
        self,
        video: VideoBatchEntry,
        *,
        epoch: int,
    ) -> Tuple[List[Tuple[str, float, float]], Dict[str, Any]]:
        """
        Build the list of (text_id, label, weight) entries for one video.

        Returns both the entries and a debug record describing the sampling.
        """

        audit: Dict[str, Any] = {"positives": [], "negatives": []}

        positive_meta_pairs: List[Tuple[TextMetadata, float]] = []
        for text_id, base_weight in video.positive_pairs:
            meta = self.text_catalog.get(text_id)
            if meta is None:
                continue
            positive_meta_pairs.append((meta, self._safe_float(base_weight, 1.0)))

        if not positive_meta_pairs:
            return [], audit

        selected_pairs, skipped_pairs = self._select_positive_pairs(video.video_id, positive_meta_pairs)
        for meta, _ in skipped_pairs:
            audit["positives"].append(
                {
                    "text_id": meta.text_id,
                    "weight": 0.0,
                    "category": meta.category,
                    "severity": self._severity_label(meta),
                    "note": "capped_normal",
                }
            )

        positives: List[Tuple[str, float, float]] = []
        positive_ids: Set[str] = set()
        pos_meta_list: List[TextMetadata] = []

        for meta, base_weight in selected_pairs:
            weight = self._positive_weight(meta, base_weight)
            positives.append((meta.text_id, 1.0, weight))
            positive_ids.add(meta.text_id)
            pos_meta_list.append(meta)
            audit["positives"].append(
                {
                    "text_id": meta.text_id,
                    "weight": weight,
                    "category": meta.category,
                    "severity": self._severity_label(meta),
                }
            )

        if not positives:
            return [], audit

        neg_entries, neg_audit = self._build_negatives(
            video=video,
            pos_meta_list=pos_meta_list,
            positive_ids=positive_ids,
        )
        audit["negatives"].extend(neg_audit)

        entries = positives + neg_entries
        return entries, audit

    # ------------------------------------------------------------------ #

    def _select_positive_pairs(
        self,
        video_id: str,
        pairs: List[Tuple[TextMetadata, float]],
    ) -> Tuple[List[Tuple[TextMetadata, float]], List[Tuple[TextMetadata, float]]]:
        """Split positive prompts into selected vs skipped according to severity policy."""

        if not pairs:
            return [], []

        normal_pairs: List[Tuple[TextMetadata, float]] = []
        abnormal_pairs: List[Tuple[TextMetadata, float]] = []

        for meta, weight in pairs:
            if self._severity_label(meta) == "normal":
                normal_pairs.append((meta, weight))
            else:
                abnormal_pairs.append((meta, weight))

        selected: List[Tuple[TextMetadata, float]] = list(abnormal_pairs)
        skipped: List[Tuple[TextMetadata, float]] = []

        if not normal_pairs:
            return selected, skipped

        if not abnormal_pairs:
            selected_normals, skipped_normals = self._select_normals(video_id, normal_pairs, max_count=None, max_weight=None)
        else:
            abnormal_weight = sum(self._severity_weight(meta) for meta, _ in abnormal_pairs)
            weight_limit = max(abnormal_weight / 3.0, 0.0)
            selected_normals, skipped_normals = self._select_normals(video_id, normal_pairs, max_count=None, max_weight=weight_limit)

        selected.extend(selected_normals)
        skipped.extend(skipped_normals)

        return selected, skipped

    def _severity_label(self, meta: TextMetadata) -> str:
        severity = (meta.disease_severity or "").strip().lower()
        if severity in {"critical", "cto"}:
            return "severe"
        if severity in self.SEVERITY_ORDER:
            return severity

        bin_label = self._normalize_bin(meta.bin)
        if bin_label in {"0", "<30"}:
            return "normal"
        if bin_label in {"30-49"}:
            return "mild"
        if bin_label in {"50-69"}:
            return "moderate"
        if bin_label in {"70-89", ">=90", "100", "cto"}:
            return "severe"

        category = (meta.category or "").lower()
        if category == "normal":
            return "normal"
        if category in {"stenosis", "in_stent", "medina", "thrombus", "calcification", "cto"}:
            return "severe"
        return "unknown"

    def _same_segment_targets(self, severity: str) -> Set[str]:
        if severity == "normal":
            return {"mild", "moderate", "severe"}
        if severity == "mild":
            return {"moderate", "severe"}
        if severity == "moderate":
            return {"mild", "severe"}
        if severity == "severe":
            return {"mild", "moderate"}
        return set()

    def _preferred_negative_severities(self, severity: str) -> Set[str]:
        if severity in {"normal", "mild"}:
            return {"severe"}
        if severity in {"moderate", "severe"}:
            return {"normal"}
        return set()

    def _severity_rank(self, severity: str) -> int:
        if not severity:
            return -1
        base = severity.strip().lower()
        if base in {"critical", "cto"}:
            base = "severe"
        return self.SEVERITY_ORDER.get(base, -1)

    def _severity_weight(self, meta: TextMetadata) -> float:
        return self.positive_severity_weights.get(self._severity_label(meta), 1.0)

    def _select_normals(
        self,
        video_id: str,
        normal_pairs: List[Tuple[TextMetadata, float]],
        *,
        max_count: Optional[int],
        max_weight: Optional[float],
    ) -> Tuple[List[Tuple[TextMetadata, float]], List[Tuple[TextMetadata, float]]]:
        if not normal_pairs:
            return [], []

        pairs_sorted = sorted(normal_pairs, key=lambda pair: pair[0].text_id)
        total = len(pairs_sorted)
        count_limit = max_count if max_count is not None and max_count > 0 else total
        if max_weight is None:
            weight_limit = float("inf")
        else:
            weight_limit = max(max_weight, 0.0)

        start = self._pos_round_robin.get(video_id, 0) % total
        picks: List[Tuple[TextMetadata, float]] = []
        used_weight = 0.0
        visited = 0

        while visited < total and len(picks) < count_limit:
            pair = pairs_sorted[(start + visited) % total]
            visited += 1
            weight = self._severity_weight(pair[0])
            if used_weight + weight <= weight_limit + 1e-6 or not picks or math.isinf(weight_limit):
                picks.append(pair)
                used_weight += weight
            if used_weight >= weight_limit - 1e-6 and not math.isinf(weight_limit):
                break

        if not picks and pairs_sorted:
            picks.append(pairs_sorted[start])
            visited = max(visited, 1)

        self._pos_round_robin[video_id] = (start + max(visited, 1)) % total
        picked_ids = {meta.text_id for meta, _ in picks}
        skipped = [pair for pair in pairs_sorted if pair[0].text_id not in picked_ids]
        return picks, skipped

    def _gather_same_segment_candidates(
        self,
        pos_meta: TextMetadata,
        positive_ids: Set[str],
    ) -> List[NegativeCandidate]:
        segment = pos_meta.segment
        if not segment:
            return []

        pos_severity = self._severity_label(pos_meta)
        target_severities = self._same_segment_targets(pos_severity)
        preferred_severities = self._preferred_negative_severities(pos_severity)

        preferred: List[NegativeCandidate] = []
        fallback: List[NegativeCandidate] = []
        for meta in self._segment_index.get(segment, []):
            if meta.text_id in positive_ids or meta.text_id == pos_meta.text_id:
                continue
            if self._is_summary(meta):
                continue
            cand_severity = self._severity_label(meta)
            if cand_severity in preferred_severities:
                preferred.append(
                    NegativeCandidate(
                        meta=meta,
                        bucket="same_segment",
                        weight_key="same_segment",
                        reason=f"segment:{segment}|severity:{cand_severity}",
                    )
                )
                continue
            if cand_severity in target_severities:
                fallback.append(
                    NegativeCandidate(
                        meta=meta,
                        bucket="same_segment",
                        weight_key="same_segment",
                        reason=f"segment:{segment}|severity:{cand_severity}",
                    )
                )
        if preferred:
            return self._dedupe_candidates(preferred)
        if fallback:
            return self._dedupe_candidates(fallback)
        return []

    def _gather_same_tree_candidates(
        self,
        pos_meta: TextMetadata,
        positive_ids: Set[str],
    ) -> List[NegativeCandidate]:
        tree = (pos_meta.tree or "").lower()
        if not tree:
            return []

        pos_severity = self._severity_label(pos_meta)
        preferred_severities = self._preferred_negative_severities(pos_severity)

        preferred: List[NegativeCandidate] = []
        fallback: List[NegativeCandidate] = []
        for meta in self._tree_meta_index.get(tree, []):
            if meta.text_id in positive_ids or meta.text_id == pos_meta.text_id:
                continue
            if self._is_summary(meta):
                continue
            # Exclude same segment - we want different segments within same tree
            if meta.segment == pos_meta.segment or not meta.segment:
                continue
            cand_severity = self._severity_label(meta)
            if cand_severity == "unknown":
                continue
            # RELAXED: Allow same severity if it's the only option to prefer same_tree over cross_tree
            # Still prefer different severities when available, but don't exclude same severity completely
            # The old logic: if cand_severity == pos_severity: continue
            # New: We'll allow it, but weight preferences will handle it
            if pos_severity == "normal" and cand_severity == "normal":
                continue  # Keep this exclusion - two normals from same tree are too similar
            candidate = NegativeCandidate(
                meta=meta,
                bucket="same_tree",
                weight_key="same_tree",
                reason=f"tree:{tree}|segment:{meta.segment}|severity:{cand_severity}",
            )
            if cand_severity in preferred_severities:
                preferred.append(candidate)
                continue
            if cand_severity != pos_severity:
                fallback.append(candidate)
        if preferred:
            return self._dedupe_candidates(preferred)
        if fallback:
            return self._dedupe_candidates(fallback)
        return []

    def _gather_cross_tree_candidates(
        self,
        pos_meta: TextMetadata,
        positive_ids: Set[str],
    ) -> List[NegativeCandidate]:
        pos_tree = (pos_meta.tree or "").lower()
        pos_severity = self._severity_label(pos_meta)
        preferred_severities = self._preferred_negative_severities(pos_severity)
        target_trees = [t for t in self._tree_meta_index.keys() if t != pos_tree] if pos_tree else list(self._tree_meta_index.keys())
        preferred: List[NegativeCandidate] = []
        fallback: List[NegativeCandidate] = []
        for tree in target_trees:
            for meta in self._tree_meta_index.get(tree, []):
                if meta.text_id in positive_ids:
                    continue
                if self._is_summary(meta):
                    continue
                cand_severity = self._severity_label(meta)
                if cand_severity == "unknown":
                    continue
                if cand_severity == pos_severity and pos_severity != "unknown":
                    continue
                if pos_severity == "normal" and cand_severity == "normal":
                    continue
                candidate = NegativeCandidate(
                    meta=meta,
                    bucket="cross_tree",
                    weight_key="cross_tree",
                    reason=f"tree:{tree}|segment:{meta.segment}|severity:{cand_severity}",
                )
                if cand_severity in preferred_severities:
                    preferred.append(candidate)
                    continue
                if cand_severity != pos_severity:
                    fallback.append(candidate)
        if preferred:
            return self._dedupe_candidates(preferred)
        if fallback:
            return self._dedupe_candidates(fallback)
        return []

    def _bucket_key(self, pos_meta: TextMetadata, bucket: str) -> str:
        segment = pos_meta.segment or "none"
        tree = (pos_meta.tree or "unknown").lower()
        severity = self._severity_label(pos_meta)
        return f"{bucket}|{tree}|{segment}|{severity}"

    def _pop_candidate(
        self,
        video_id: str,
        bucket_key: str,
        candidates: List[NegativeCandidate],
        used_ids: Set[str],
    ) -> Optional[NegativeCandidate]:
        if not candidates:
            return None

        pool = [cand for cand in candidates if cand.meta.text_id not in used_ids]
        if not pool:
            return None

        picks = self._round_robin_pick(video_id, bucket_key, pool, 1)
        if not picks:
            return None

        pick = picks[0]
        used_ids.add(pick.meta.text_id)
        candidates[:] = [cand for cand in candidates if cand.meta.text_id != pick.meta.text_id]
        return pick

    def _build_negatives(
        self,
        video: VideoBatchEntry,
        *,
        pos_meta_list: List[TextMetadata],
        positive_ids: Set[str],
    ) -> Tuple[List[Tuple[str, float, float]], List[Dict[str, Any]]]:
        if self.max_negatives <= 0:
            return [], []

        used_ids: Set[str] = set(positive_ids)
        candidate_groups: List[Tuple[TextMetadata, Dict[str, List[NegativeCandidate]]]] = []
        for meta in pos_meta_list:
            group = {
                "same_segment": self._gather_same_segment_candidates(meta, positive_ids),
                "same_tree": self._gather_same_tree_candidates(meta, positive_ids),
                "cross_tree": self._gather_cross_tree_candidates(meta, positive_ids),
            }
            candidate_groups.append((meta, group))

        negatives: List[Tuple[str, float, float]] = []
        audit: List[Dict[str, Any]] = []
        
        # PRIORITIZED SAMPLING: Exhaust buckets in order across all positives before moving to next bucket
        # This ensures we prioritize same_tree + same_vessel negatives before cross_tree
        bucket_order = ["same_segment", "same_tree", "cross_tree"]
        
        for bucket in bucket_order:
            if len(negatives) >= self.max_negatives:
                break
                
            # Try to pick from this bucket for all positives before moving to next bucket
            progress = True
            while len(negatives) < self.max_negatives and progress:
                progress = False
                for meta, group in candidate_groups:
                    if len(negatives) >= self.max_negatives:
                        break
                    candidates = group[bucket]
                    if not candidates:
                        continue
                    bucket_key = self._bucket_key(meta, bucket)
                    pick = self._pop_candidate(video.video_id, bucket_key, candidates, used_ids)
                    if pick is None:
                        continue
                    weight = self._negative_weight_for(pick.meta, meta)
                    negatives.append((pick.meta.text_id, 0.0, weight))
                    audit.append(
                        {
                            "text_id": pick.meta.text_id,
                            "weight": weight,
                            "bucket": bucket,
                            "weight_key": bucket,
                            "reason": pick.reason,
                            "positive_ref": meta.text_id,
                        }
                    )
                    progress = True

        if len(negatives) < self.max_negatives:
            deficit = self.max_negatives - len(negatives)
            fallback = self._fallback_negatives(video, deficit, used_ids, positive_ids)
            for cand in fallback:
                weight = self._negative_weight_for(cand.meta, None)
                negatives.append((cand.meta.text_id, 0.0, weight))
                audit.append(
                    {
                        "text_id": cand.meta.text_id,
                        "weight": weight,
                        "bucket": "fallback",
                        "weight_key": "fallback",
                        "reason": cand.reason,
                        "positive_ref": None,
                    }
                )

        if len(negatives) > self.max_negatives:
            negatives = negatives[: self.max_negatives]
            audit = audit[: self.max_negatives]

        return negatives, audit

    def _round_robin_pick(
        self,
        video_id: str,
        bucket_key: str,
        candidates: List[NegativeCandidate],
        count: int,
    ) -> List[NegativeCandidate]:
        if count <= 0 or not candidates:
            return []

        if not self.round_robin:
            start = self._rng.randrange(len(candidates)) if candidates else 0
            ordered = candidates[start:] + candidates[:start]
            return ordered[:count]

        pool_size = len(candidates)
        state = self._round_robin_state[video_id]
        offset = state.get(bucket_key, 0) % pool_size
        state[bucket_key] = (offset + count) % pool_size

        result: List[NegativeCandidate] = []
        idx = offset
        while len(result) < count:
            result.append(candidates[idx])
            idx = (idx + 1) % pool_size
        return result

    def _fallback_negatives(
        self,
        video: VideoBatchEntry,
        deficit: int,
        used_ids: Set[str],
        positive_ids: Set[str],
    ) -> List[NegativeCandidate]:
        if deficit <= 0:
            return []

        fallback: List[NegativeCandidate] = []
        for meta in self._all_texts:
            if len(fallback) >= deficit:
                break
            if meta.text_id in used_ids or meta.text_id in positive_ids:
                continue
            if self._is_summary(meta):
                continue
            used_ids.add(meta.text_id)
            fallback.append(
                NegativeCandidate(
                    meta=meta,
                    bucket="fallback",
                    weight_key="fallback",
                    reason="global_pool",
                )
            )
        return fallback

    def _merge_positive_weights(self, overrides: Dict[str, float]) -> Dict[str, float]:
        merged = dict(self.DEFAULT_POSITIVE_SEVERITY_WEIGHTS)
        for key, value in overrides.items():
            try:
                merged[str(key).lower()] = max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
        return merged

    def _dedupe_candidates(self, candidates: List[NegativeCandidate]) -> List[NegativeCandidate]:
        seen: Dict[str, NegativeCandidate] = {}
        for cand in candidates:
            if cand.meta.text_id not in seen:
                seen[cand.meta.text_id] = cand
        return list(seen.values())

    def _negative_weight(self) -> float:
        return self.base_negative_weight

    def _negative_weight_for(
        self,
        candidate: TextMetadata,
        reference: Optional[TextMetadata] = None,
    ) -> float:
        weight = self.base_negative_weight
        severity = (candidate.disease_severity or "").strip().lower()
        category = (candidate.category or "").strip().lower()
        is_abnormal = self._is_abnormal(candidate)

        if not is_abnormal:
            scale = self.neg_normal_scale
        elif severity in {"mild"} or category in {"calcification"}:
            scale = self.neg_mild_scale
        else:
            scale = self.neg_abnormal_scale
        weight *= max(scale, 0.0)

        if reference is not None:
            same_tree = bool(candidate.tree and reference.tree and candidate.tree == reference.tree)
            same_segment = bool(candidate.segment and reference.segment and candidate.segment == reference.segment)
            if same_segment:
                weight *= max(self.same_segment_boost, 0.0)
                if (
                    self.contradiction_boost > 0.0
                    and self._contradiction_min_rank >= 0
                    and self._severity_label(candidate) == "normal"
                ):
                    ref_severity_rank = self._severity_rank(self._severity_label(reference))
                    if ref_severity_rank >= self._contradiction_min_rank:
                        weight *= max(self.contradiction_boost, 0.0)
            elif same_tree:
                weight *= max(self.same_tree_boost, 0.0)

        return weight

    def _positive_weight(self, meta: TextMetadata, base_weight: float) -> float:
        soft_weight = float(meta.soft_weight or 1.0)
        class_weight = float(meta.class_weight or 1.0)
        severity_scale = self.positive_severity_weights.get(self._severity_label(meta), 1.0)
        weight = soft_weight * class_weight * max(base_weight, 0.0) * max(severity_scale, 1e-3)
        return max(weight, self.min_pos_weight)

    @staticmethod
    def _safe_float(value: Any, default: float = 1.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _metadata_dict(self, meta: TextMetadata) -> Dict[str, Any]:
        return {
            "text_id": meta.text_id,
            "prompt_text": meta.prompt_text,
            "prompt_type": meta.prompt_type,
            "segment": meta.segment,
            "tree": meta.tree,
            "category": meta.category,
            "bin": meta.bin,
            "prompt_bucket": meta.prompt_bucket,
            "is_abnormal": self._is_abnormal(meta),
            "class_weight": float(meta.class_weight or 1.0),
        }

    def _is_abnormal(self, meta: TextMetadata) -> bool:
        category = (meta.category or "").lower()
        if category in self.ABNORMAL_CATEGORIES:
            return True
        bucket = (meta.prompt_bucket or "").lower()
        if bucket in self.ABNORMAL_PROMPT_BUCKETS:
            return True
        severity = (meta.disease_severity or "").lower()
        return severity in {"mild", "moderate", "severe", "critical", "cto"}

    def _is_summary(self, meta: TextMetadata) -> bool:
        bucket = (meta.prompt_bucket or "").lower()
        category = (meta.category or "").lower()
        return bucket in self.SUMMARY_BUCKETS or category == "summary"

    def _normalize_bin(self, bin_label: Optional[str]) -> str:
        if bin_label is None:
            return ""
        if isinstance(bin_label, float):
            if math.isnan(bin_label):
                return ""
            bin_str = f"{bin_label:.0f}" if bin_label.is_integer() else str(bin_label)
            return bin_str.strip().lower()
        text = str(bin_label)
        return text.strip().lower()

    def _build_indices(self) -> None:
        """Pre-compute lookup indices for sampling."""

        self._segment_index.clear()
        self._tree_meta_index.clear()
        self._summary_texts.clear()

        for meta in self.text_catalog.values():
            if meta.segment:
                self._segment_index[meta.segment].append(meta)
            if meta.tree:
                self._tree_meta_index[meta.tree].append(meta)
            if self._is_summary(meta):
                self._summary_texts.append(meta)


# ---------------------------------------------------------------------------
# Utility functions for catalog construction
# ---------------------------------------------------------------------------


def build_text_catalog(
    texts: Iterable[Dict[str, Any]],
    class_weight: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float],
    logit_bias: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float],
) -> Dict[str, TextMetadata]:
    """Convert raw dict entries to TextMetadata with default fields."""

    catalog: Dict[str, TextMetadata] = {}
    for entry in texts:
        text_id = str(entry["text_id"])
        tags = dict(entry.get("tags") or {})
        segment = entry.get("segment") or tags.get("segment")
        bin_label = entry.get("bin") or tags.get("bin")
        stent = entry.get("stent") or tags.get("stent") or "n"
        tree = (entry.get("tree") or tags.get("tree") or "").lower() or None
        class_key = (segment, bin_label, stent)
        catalog[text_id] = TextMetadata(
            text_id=text_id,
            prompt_text=str(entry.get("prompt_text", "")),
            prompt_type=entry.get("prompt_type"),
            category=entry.get("category"),
            segment=segment,
            bin=bin_label,
            tree=tree,
            stent=stent,
            soft_weight=float(entry.get("soft_weight", 1.0)),
            disease_severity=entry.get("disease_severity"),
            tags=tags,
            prompt_bucket=entry.get("prompt_bucket"),
            class_key=class_key,
            logit_bias=logit_bias.get(class_key, 0.0),
            class_weight=class_weight.get(class_key, 1.0),
        )
    return catalog


def compute_class_statistics(
    texts: Iterable[Dict[str, Any]],
    beta: float = 0.999,
) -> Tuple[
    Dict[Tuple[Optional[str], Optional[str], Optional[str]], float],
    Dict[Tuple[Optional[str], Optional[str], Optional[str]], float],
]:
    """
    Return (effective-number weights, logit biases) keyed by (segment, bin, stent).
    """

    counts: Dict[Tuple[Optional[str], Optional[str], Optional[str]], int] = {}
    for entry in texts:
        tags = dict(entry.get("tags") or {})
        segment = entry.get("segment") or tags.get("segment")
        bin_label = entry.get("bin") or tags.get("bin")
        stent = entry.get("stent") or tags.get("stent") or "n"
        key = (segment, bin_label, stent)
        counts[key] = counts.get(key, 0) + 1

    total = max(1, sum(counts.values()))
    class_weight: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float] = {}
    logit_bias: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float] = {}

    for key, freq in counts.items():
        effective = (1 - beta) / (1 - math.pow(beta, freq))
        class_weight[key] = effective
        pi = freq / total
        epsilon = 1e-6
        pi = min(max(pi, epsilon), 1 - epsilon)
        logit_bias[key] = math.log((1 - pi) / pi)

    return class_weight, logit_bias


# ---------------------------------------------------------------------------
