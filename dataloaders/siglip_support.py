import collections
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import pandas as pd
import torch

from utils.siglip.single_head_sampler import (
    SingleHeadRetrievalSampler,
    TextMetadata,
    VideoBatchEntry,
    build_text_catalog,
    compute_class_statistics,
)
from models.text_encoder import get_tokenizer


class SiglipSupport:
    """
    Encapsulates SigLIP-specific resource loading, caching and sampling logic so that
    ``VideoClipDataset`` can remain agnostic to the SigLIP metadata layout.
    """

    LEFT_TREE_SEGMENTS: Set[str] = {
        "left_main",
        "prox_lad",
        "mid_lad",
        "dist_lad",
        "d1",
        "d2",
        "om1",
        "om2",
        "ramus",
        "dist_lcx",
    }
    RIGHT_TREE_SEGMENTS: Set[str] = {
        "prox_rca",
        "mid_rca",
        "dist_rca",
        "pda",
        "posterolateral",
        "lvp",
    }
    SEGMENTS_BY_TREE: Dict[str, Set[str]] = {
        "left": LEFT_TREE_SEGMENTS,
        "right": RIGHT_TREE_SEGMENTS,
    }
    SEGMENT_SYNONYMS: Dict[str, str] = {
        "proximal rca": "prox_rca",
        "mid rca": "mid_rca",
        "distal rca": "dist_rca",
        "proximal lad": "prox_lad",
        "mid lad": "mid_lad",
        "distal lad": "dist_lad",
        "left main": "left_main",
        "pda": "pda",
        "posterior descending": "pda",
        "right ventricular posterior": "lvp",
        "rvp": "lvp",
        "posterolateral": "posterolateral",
        "om": "om1",
        "om branch": "om1",
        "obtuse marginal": "om1",
    }

    def _normalize_tree_hint(self, tree_hint: Optional[str]) -> Optional[str]:
        if tree_hint is None:
            return None
        normalizer = getattr(self.dataset, "_normalize_tree_key", None)
        if callable(normalizer):
            return normalizer(tree_hint)
        if not isinstance(tree_hint, str):
            return None
        text = tree_hint.strip().lower()
        if not text:
            return None
        if any(token in text for token in ["left", "lad", "lcx", "circ", "diag", "ramus", "om"]):
            return "left"
        if any(token in text for token in ["right", "rca", "pda", "posterolateral", "rvp", "lvp"]):
            return "right"
        return text

    @classmethod
    def _canonical_segment(cls, segment: Optional[str]) -> Optional[str]:
        if not isinstance(segment, str):
            return None
        key = segment.strip().lower()
        if not key:
            return None
        return cls.SEGMENT_SYNONYMS.get(key, key)

    @classmethod
    def _allowed_segments_for_tree(cls, tree_hint: Optional[str]) -> Optional[Set[str]]:
        if not isinstance(tree_hint, str):
            return None
        return cls.SEGMENTS_BY_TREE.get(tree_hint.strip().lower())

    def __init__(self, dataset: "VideoClipDataset", kwargs: Dict[str, Any]) -> None:
        self.dataset = dataset

        self.texts_path: Optional[str] = kwargs.pop("siglip_texts_path", None)
        self.video_id_column: str = kwargs.pop("siglip_video_id_column", "video_id")
        self.text_id_column: str = kwargs.pop("siglip_text_id_column", "text_id")
        self.prompt_text_column: str = kwargs.pop("siglip_prompt_text_column", "prompt_text")
        self.prompt_type_column: str = kwargs.pop("siglip_prompt_type_column", "prompt_type")
        self.soft_weight_column: str = kwargs.pop("siglip_soft_weight_column", "soft_weight")
        self.edge_weight_column: str = kwargs.pop("siglip_edge_weight_column", "weight")

        self.negatives_per_video: int = int(kwargs.pop("siglip_negatives_per_video", 256))
        self.round_robin_sampling: bool = bool(kwargs.pop("siglip_round_robin_sampling", False))
        self.alpha_neg: float = float(kwargs.pop("siglip_alpha_neg", 2.0))
        self.max_segments_per_video: int = int(kwargs.pop("siglip_max_segments_per_video", 15))

        kwargs.pop("siglip_pos_samples_per_video", None)
        kwargs.pop("siglip_negative_mix", None)
        kwargs.pop("siglip_negative_weights", None)

        neg_weight_override = kwargs.pop("siglip_negative_weight", None)
        if neg_weight_override is not None:
            try:
                self.negative_weight = float(neg_weight_override)
            except (TypeError, ValueError):
                print(f"Warning: Invalid siglip_negative_weight: {neg_weight_override}")
                self.negative_weight = 0.04
        else:
            self.negative_weight = 0.04

        self.positive_severity_weights: Dict[str, float] = {
            "normal": 0.75,
            "mild": 1.25,
            "moderate": 1.75,
            "severe": 2.5,
            "critical": 2.5,
            "cto": 2.5,
        }
        pos_weight_override = kwargs.pop("siglip_positive_severity_weights", None)
        if isinstance(pos_weight_override, dict) and pos_weight_override:
            for key, value in pos_weight_override.items():
                norm_key = str(key).lower()
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                self.positive_severity_weights[norm_key] = max(val, 0.0)

        min_pos_weight_override = kwargs.pop("siglip_min_pos_weight", None)
        if min_pos_weight_override is not None:
            try:
                self.min_pos_weight = float(min_pos_weight_override)
            except (TypeError, ValueError):
                self.min_pos_weight = 0.5
        else:
            self.min_pos_weight = 0.5
        self.neg_normal_scale: float = float(kwargs.pop("siglip_neg_normal_scale", 0.25))
        self.neg_mild_scale: float = float(kwargs.pop("siglip_neg_mild_scale", 0.75))
        self.neg_abnormal_scale: float = float(kwargs.pop("siglip_neg_abnormal_scale", 1.5))
        self.same_segment_boost: float = float(kwargs.pop("siglip_same_segment_boost", 1.5))
        self.same_tree_boost: float = float(kwargs.pop("siglip_same_tree_boost", 1.25))
        self.contradiction_boost: float = float(kwargs.pop("siglip_contradiction_boost", 1.0))
        self.contradiction_min_severity: str = str(
            kwargs.pop("siglip_contradiction_min_severity", "moderate")
        ).lower()

        self.enabled: bool = bool(self.texts_path)
        self._is_primary_rank: bool = (
            torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True
        )

        # Resource caches
        self.text_lookup: Dict[str, Dict[str, Any]] = {}
        self.video_to_text_ids: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        self.tree_to_texts: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self.all_text_entries: List[Dict[str, Any]] = []
        self.text_label_lookup: Dict[str, str] = {}
        self.text_catalog: Dict[str, TextMetadata] = {}
        self.text_id_to_index: Dict[str, int] = {}
        self.text_input_ids: Optional[torch.Tensor] = None
        self.text_attention_mask: Optional[torch.Tensor] = None
        self.exam_severity: Dict[str, str] = {}
        self.class_weight_lookup: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float] = {}
        self.logit_bias_lookup: Dict[Tuple[Optional[str], Optional[str], Optional[str]], float] = {}
        self.prompt_to_bias: Dict[str, float] = {}
        self.prompt_to_metadata: Dict[str, TextMetadata] = {}
        self.sampler: Optional[SingleHeadRetrievalSampler] = None
        self._phase: str = "A"
        self._epoch: int = 0

        if self.enabled:
            self._load_resources()

    # ------------------------------------------------------------------ #
    # Resource loading
    # ------------------------------------------------------------------ #

    def reload_resources(self) -> None:
        if self.enabled:
            self._load_resources()

    def _load_resources(self) -> None:
        assert self.texts_path is not None

        texts_path = Path(self.texts_path).expanduser().resolve()
        if not texts_path.exists():
            raise FileNotFoundError(f"SigLIP texts.csv not found at {texts_path}")

        texts_df = pd.read_csv(texts_path)
        required_text_cols = {self.text_id_column, self.prompt_text_column}
        missing_text_cols = required_text_cols - set(texts_df.columns)
        if missing_text_cols:
            raise ValueError(f"texts.csv is missing required columns: {sorted(missing_text_cols)}")

        self.text_lookup.clear()
        self.tree_to_texts.clear()
        self.all_text_entries.clear()
        self.text_catalog.clear()
        self.text_id_to_index.clear()
        self.text_input_ids = None
        self.text_attention_mask = None
        self.prompt_to_metadata.clear()

        raw_text_entries: List[Dict[str, Any]] = []
        for _, row in texts_df.iterrows():
            text_id = str(row[self.text_id_column])
            tags = self.dataset._parse_tags(row.get("tags", ""))
            tree_hint = self.dataset._normalize_tree_key(row.get("tree") or tags.get("tree"))
            prompt_type_val = row.get(self.prompt_type_column)
            prompt_text_val = row.get(self.prompt_text_column)
            category_val = row.get("category")
            segment_val = row.get("segment") or tags.get("segment")
            bin_val = row.get("bin") or tags.get("bin")
            stent_val = row.get("stent") or tags.get("stent") or "n"
            disease_severity = row.get("disease_severity")
            if isinstance(disease_severity, str):
                disease_severity = disease_severity.strip().lower()
            else:
                disease_severity = None

            is_abnormal = self.dataset._is_abnormal_prompt(prompt_type_val, prompt_text_val)

            soft_weight_val = row.get(self.soft_weight_column, 1.0)
            try:
                soft_weight_val = float(soft_weight_val)
            except (TypeError, ValueError):
                soft_weight_val = 1.0

            text_info = {
                "text_id": text_id,
                "prompt_text": prompt_text_val,
                "prompt_type": prompt_type_val,
                "soft_weight": soft_weight_val,
                "tree": tree_hint,
                "tags": tags,
                "is_abnormal": is_abnormal,
                "category": category_val,
                "segment": segment_val,
                "bin": bin_val,
                "stent": stent_val,
                "disease_severity": disease_severity,
                "prompt_bucket": row.get("prompt_bucket"),
            }

            self.text_lookup[text_id] = text_info
            self.text_label_lookup[text_id] = "abnormal" if is_abnormal else "normal"
            if tree_hint:
                self.tree_to_texts.setdefault(tree_hint, []).append(text_info)
            self.all_text_entries.append(text_info)
            raw_text_entries.append(text_info)

        class_weight_lookup, logit_bias_lookup = compute_class_statistics(raw_text_entries)
        self.text_catalog = build_text_catalog(raw_text_entries, class_weight_lookup, logit_bias_lookup)
        self.class_weight_lookup = class_weight_lookup
        self.logit_bias_lookup = logit_bias_lookup

        self._precompute_text_encodings()

        self.video_to_text_ids.clear()
        self.exam_severity.clear()

        videos_path = Path(self.dataset.filename)
        if not videos_path.is_absolute():
            videos_path = Path(self.dataset.folder or "") / videos_path
        videos_path = videos_path.resolve()
        if not videos_path.exists():
            raise FileNotFoundError(
                f"SigLIP videos.csv not found at {videos_path}; required to resolve positive_text_ids."
            )
        expected_cols = [
            self.dataset.datapoint_loc_label,
            getattr(self.dataset, "siglip_video_id_column", "video_id"),
            "Split",
            "positive_text_ids",
        ]
        videos_df = self.dataset._read_metadata_csv(videos_path, expected_columns=expected_cols)

        for _, row in videos_df.iterrows():
            video_id_val = row.get(self.video_id_column)
            if video_id_val is None or pd.isna(video_id_val):
                continue
            video_key = str(video_id_val)
            pos_ids_str = row.get("positive_text_ids", "")
            if not isinstance(pos_ids_str, str) or not pos_ids_str.strip():
                continue

            pairs: List[Tuple[str, float]] = []
            severity_votes: set[str] = set()
            for tid in pos_ids_str.split("|"):
                tid = tid.strip()
                if not tid or tid not in self.text_lookup:
                    continue
                if tid not in self.text_catalog:
                    continue
                text_meta = self.text_lookup.get(tid, {})
                pairs.append((tid, 1.0))
                sev = text_meta.get("disease_severity")
                if isinstance(sev, str) and sev and sev != "summary":
                    severity_votes.add(sev.lower())

            if not pairs:
                continue

            self.video_to_text_ids[video_key] = list(pairs)
            self.exam_severity[video_key] = self.dataset._resolve_exam_severity(severity_votes)

        self.prompt_to_bias = {}
        self.prompt_to_metadata = {}
        for meta in self.text_catalog.values():
            prompt_text = meta.prompt_text
            if prompt_text:
                self.prompt_to_bias[prompt_text] = meta.logit_bias
                self.prompt_to_metadata.setdefault(prompt_text, meta)

        self.sampler = SingleHeadRetrievalSampler(
            text_catalog=self.text_catalog,
            class_weight=class_weight_lookup,
            logit_bias=logit_bias_lookup,
            alpha_neg=self.alpha_neg,
            max_negatives=self.negatives_per_video,
            base_negative_weight=self.negative_weight,
            round_robin=self.round_robin_sampling,
            min_pos_weight=self.min_pos_weight,
            positive_severity_weights=dict(self.positive_severity_weights),
            neg_normal_scale=self.neg_normal_scale,
            neg_mild_scale=self.neg_mild_scale,
            neg_abnormal_scale=self.neg_abnormal_scale,
            same_segment_boost=self.same_segment_boost,
            same_tree_boost=self.same_tree_boost,
            contradiction_boost=self.contradiction_boost,
            contradiction_min_severity=self.contradiction_min_severity,
        )
        if self._is_primary_rank:
            print(
                f"[VideoClipDataset] Loaded SigLIP resources: "
                f"{len(self.text_lookup)} texts, "
                f"{len(self.video_to_text_ids)} videos with positives"
            )

    # ------------------------------------------------------------------ #
    # Public API surface for dataset
    # ------------------------------------------------------------------ #
    def _precompute_text_encodings(self) -> None:
        if not self.text_lookup:
            return

        tokenizer = getattr(self.dataset, "tokenizer", None)
        if tokenizer is None:
            tokenizer = get_tokenizer()
            self.dataset.tokenizer = tokenizer

        prompts: List[str] = []
        text_ids: List[str] = []
        for text_id, meta in self.text_lookup.items():
            prompts.append(str(meta.get("prompt_text", "")))
            text_ids.append(text_id)
            self.text_id_to_index[text_id] = len(text_ids) - 1

        batch_size = max(1, int(getattr(self.dataset, "siglip_encoding_batch_size", 1024)))
        input_ids_chunks: List[torch.Tensor] = []
        attention_chunks: List[torch.Tensor] = []

        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            batch_prompts = prompts[start:end]
            encoding = tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=self.dataset.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids_chunks.append(encoding["input_ids"])
            attention_chunks.append(encoding["attention_mask"])

        self.text_input_ids = torch.cat(input_ids_chunks, dim=0).contiguous()
        self.text_attention_mask = torch.cat(attention_chunks, dim=0).contiguous()

    def get_positive_pairs(self, video_key: Optional[str]) -> List[Tuple[str, float]]:
        if not video_key:
            return []
        return list(self.video_to_text_ids.get(video_key, []))

    def get_exam_severity(self, video_key: Optional[str]) -> str:
        return self.exam_severity.get(video_key or "", "NORMAL")

    def encode_siglip_text(
        self,
        text_id: str,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.text_input_ids is None or self.text_attention_mask is None:
            raise RuntimeError("SigLIP text encodings are not initialised")

        idx = self.text_id_to_index.get(text_id)
        if idx is None:
            raise KeyError(f"Unknown SigLIP text_id '{text_id}'")

        input_ids = self.text_input_ids[idx]
        attention_mask = self.text_attention_mask[idx]

        if device is not None:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_text_batch(
        self,
        text_ids: List[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        if not text_ids:
            empty = torch.empty((0, 1), dtype=torch.long)
            if device is not None:
                empty = empty.to(device)
            return {"input_ids": empty, "attention_mask": empty.clone()}

        if self.text_input_ids is None or self.text_attention_mask is None:
            raise RuntimeError("SigLIP text encodings are not initialised")

        indices = []
        for tid in text_ids:
            idx = self.text_id_to_index.get(tid)
            if idx is None:
                raise KeyError(f"Unknown SigLIP text_id '{tid}'")
            indices.append(idx)

        index_tensor = torch.tensor(indices, dtype=torch.long, device=self.text_input_ids.device)
        input_ids = torch.index_select(self.text_input_ids, 0, index_tensor)
        attention_mask = torch.index_select(self.text_attention_mask, 0, index_tensor)

        if device is not None:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def sample_positive_entries(self, video_idx: int) -> List[Dict[str, Any]]:
        if video_idx < 0 or video_idx >= len(self.dataset.video_positive_texts):
            return []

        raw_pairs = self.dataset.video_positive_texts[video_idx]
        if not raw_pairs:
            return []

        tree_hint = None
        if 0 <= video_idx < len(self.dataset.video_trees):
            tree_hint = self.dataset.video_trees[video_idx]

        filtered_pairs = self.filter_positive_pairs(raw_pairs, tree_hint=tree_hint)
        if not filtered_pairs:
            filtered_pairs = raw_pairs

        meta_items: List[Tuple[TextMetadata, float]] = []
        for text_id, weight in filtered_pairs:
            meta = self.text_catalog.get(text_id)
            if meta is None:
                continue
            meta_items.append((meta, weight))
        if not meta_items:
            return []

        pruned_meta_items = self._prune_meta_items(meta_items)
        if not pruned_meta_items:
            pruned_meta_items = meta_items

        entries: List[Dict[str, Any]] = []
        for meta, weight in pruned_meta_items:
            soft_weight = self._compute_positive_weight(meta, weight)
            entries.append(
                {
                    "text_id": meta.text_id,
                    "prompt_text": meta.prompt_text,
                    "prompt_type": meta.prompt_type,
                    "soft_weight": soft_weight,
                    "tags": meta.tags,
                    "edge_weight": float(weight),
                    "is_abnormal": self._is_abnormal(meta),
                    "disease_severity": meta.disease_severity,
                    "prompt_bucket": meta.prompt_bucket,
                    "category": meta.category,
                    "segment": meta.segment,
                    "tree": meta.tree,
                    "bin": meta.bin,
                    "encoding": self.encode_siglip_text(meta.text_id),
                }
            )
        return entries

    def filter_positive_pairs(
        self,
        pairs: List[Tuple[str, float]],
        tree_hint: Optional[str] = None,
        max_segments: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Prune contradictory or overly generic positives for a video."""
        if not pairs:
            return []

        normalized_tree = self._normalize_tree_hint(tree_hint)
        allowed_segments = self._allowed_segments_for_tree(normalized_tree)
        if max_segments is None:
            max_segments = self.max_segments_per_video

        meta_items: List[Tuple[TextMetadata, float]] = []
        for text_id, weight in pairs:
            meta = self.text_catalog.get(text_id)
            if meta is None:
                continue
            meta_tree = self._normalize_tree_hint(meta.tree)
            seg = self._canonical_segment(meta.segment)
            if normalized_tree and meta_tree and meta_tree != normalized_tree:
                continue
            if normalized_tree and allowed_segments and seg and seg not in allowed_segments:
                continue
            meta_items.append((meta, weight))

        if not meta_items:
            return []

        pruned_meta = self._prune_meta_items(meta_items)
        if not pruned_meta:
            pruned_meta = meta_items

        if max_segments and max_segments > 0 and len(pruned_meta) > max_segments:
            pruned_meta = sorted(
                pruned_meta,
                key=lambda item: (
                    self._severity_rank(self._severity_label(item[0])),
                    self._specificity_score(item[0]),
                    float(item[1]),
                ),
                reverse=True,
            )[:max_segments]

        return [(meta.text_id, weight) for meta, weight in pruned_meta]

    @staticmethod
    def _severity_label(meta: TextMetadata) -> str:
        severity = (meta.disease_severity or "").strip().lower()
        if severity in {"critical", "cto"}:
            return "severe"
        if severity in {"normal", "mild", "moderate", "severe"}:
            return severity
        bin_label = str(meta.bin or "").strip().lower()
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

    @staticmethod
    def _is_abnormal(meta: TextMetadata) -> bool:
        category = (meta.category or "").lower()
        if category in {"stenosis", "in_stent", "thrombus", "calcification", "cto", "medina"}:
            return True
        bucket = (meta.prompt_bucket or "").lower()
        if bucket in {"abnormal"}:
            return True
        severity = (meta.disease_severity or "").lower()
        return severity not in {"", "normal"}

    def _compute_positive_weight(self, meta: TextMetadata, base_edge_weight: float) -> float:
        """
        Combine CSV edge weight, per-text soft weight, and severity scaling.
        Ensures that abnormal prompts keep higher emphasis than normal findings.
        """
        severity_label = self._severity_label(meta)
        severity_scale = self.positive_severity_weights.get(severity_label, 1.0)
        soft_weight = float(meta.soft_weight or 1.0)
        edge_weight = float(base_edge_weight or 1.0)

        # Abnormal prompts should not be down-weighted even if metadata is missing
        if self._is_abnormal(meta):
            severity_scale = max(severity_scale, 1.0)
        else:
            # Mild / normal findings should not overpower abnormal ones
            severity_scale = min(severity_scale, self.positive_severity_weights.get("mild", severity_scale))

        combined = soft_weight * edge_weight * severity_scale
        if severity_label == "normal":
            combined = min(max(combined, 0.5), 1.0)
        elif severity_label == "mild":
            combined = max(combined, 1.0)
        elif severity_label == "moderate":
            combined = max(combined, 1.5)
        elif severity_label == "severe":
            combined = max(combined, 2.0)
        return float(max(combined, 1e-6))

    def compute_positive_weight(self, text_id: str, base_edge_weight: float) -> float:
        """Public helper to fetch severity-aware positive weights for a text id."""
        meta = self.text_catalog.get(str(text_id))
        if meta is None:
            try:
                return float(base_edge_weight or 1.0)
            except (TypeError, ValueError):
                return 1.0
        return self._compute_positive_weight(meta, base_edge_weight)

    @staticmethod
    def _specificity_score(meta: TextMetadata) -> int:
        score = 0
        if meta.segment:
            score += 3
        if meta.bin:
            score += 2
        if meta.category and meta.category.lower() not in {"", "normal"}:
            score += 1
        if meta.stent and str(meta.stent).strip().lower() not in {"", "n", "no"}:
            score += 1
        return score

    @staticmethod
    def _severity_rank(label: str) -> int:
        mapped = (label or "").strip().lower()
        if mapped in {"critical", "cto"}:
            mapped = "severe"
        rank_map = {
            "normal": 0,
            "mild": 1,
            "moderate": 2,
            "severe": 3,
        }
        return rank_map.get(mapped, -1)

    def _prune_meta_items(
        self,
        meta_items: List[Tuple[TextMetadata, float]],
    ) -> List[Tuple[TextMetadata, float]]:
        if not meta_items:
            return []

        severity_labels = [self._severity_label(meta) for meta, _ in meta_items]
        groups: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
        for idx, (meta, _) in enumerate(meta_items):
            key = (
                (meta.tree or "").lower(),
                (meta.segment or "").lower(),
            )
            groups[key].append(idx)

        keep_indices: Set[int] = set()
        for indices in groups.values():
            if not indices:
                continue
            best_idx = max(
                indices,
                key=lambda i: (
                    self._severity_rank(self._severity_label(meta_items[i][0])),
                    self._specificity_score(meta_items[i][0]),
                    -i,
                ),
            )
            keep_indices.add(best_idx)

        if not keep_indices:
            best_idx = max(
                range(len(meta_items)),
                key=lambda i: (
                    self._severity_rank(self._severity_label(meta_items[i][0])),
                    self._specificity_score(meta_items[i][0]),
                    -i,
                ),
            )
            keep_indices.add(best_idx)

        all_non_diseased = all(
            self._severity_rank(label) <= self._severity_rank("normal") for label in severity_labels
        )
        if all_non_diseased:
            segmented = [
                idx
                for idx in keep_indices
                if (meta_items[idx][0].segment or "").strip()
            ]
            if segmented:
                keep_indices = set(segmented)

        ordered_indices = sorted(keep_indices)
        unique_results: List[Tuple[TextMetadata, float]] = []
        seen_ids: Set[str] = set()
        for idx in ordered_indices:
            meta, weight = meta_items[idx]
            if meta.text_id in seen_ids:
                continue
            seen_ids.add(meta.text_id)
            unique_results.append((meta, weight))

        if not unique_results:
            unique_results = [meta_items[ordered_indices[0]]] if ordered_indices else meta_items[:1]

        return unique_results

    def build_negative_candidates(
        self,
        positive_ids: Iterable[str],
        tree_hint: Optional[str],
    ) -> List[str]:
        positive_ids_set = set(positive_ids)
        if tree_hint and tree_hint in self.tree_to_texts:
            candidate_ids = [
                t.get("text_id")
                for t in self.tree_to_texts[tree_hint]
                if t.get("text_id") not in positive_ids_set
            ]
        else:
            candidate_ids = []

        if candidate_ids:
            return candidate_ids

        pos_metadata = [
            self.text_lookup.get(tid)
            for tid in positive_ids_set
            if tid in self.text_lookup
        ]
        pos_segments = {
            (
                (meta.get("tree") or "").lower(),
                (meta.get("segment") or "").lower(),
            )
            for meta in pos_metadata
            if meta is not None
        }
        tree_norm = (tree_hint or "").lower()
        same_segment: List[str] = []
        same_tree_diff_segment: List[str] = []
        global_rest: List[str] = []
        for text_entry in self.all_text_entries:
            tid = text_entry.get("text_id")
            if not tid or tid in positive_ids_set:
                continue
            entry_tree = (text_entry.get("tree") or "").lower()
            entry_segment = (text_entry.get("segment") or "").lower()
            key = (entry_tree, entry_segment)
            if key in pos_segments and key != ("", ""):
                same_segment.append(tid)
            elif entry_tree and entry_tree == tree_norm:
                same_tree_diff_segment.append(tid)
            else:
                global_rest.append(tid)
        candidate_ids = same_segment + same_tree_diff_segment + global_rest
        if not candidate_ids:
            candidate_ids = [
                t.get("text_id")
                for t in self.all_text_entries
                if t.get("text_id") not in positive_ids_set
            ]
        return candidate_ids

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def epoch(self) -> int:
        return self._epoch

    def get_logit_bias_for_prompt(self, prompt_text: str) -> Optional[float]:
        return self.prompt_to_bias.get(prompt_text)

    def get_metadata_for_prompt(self, prompt_text: str) -> Optional[Dict[str, Any]]:
        meta = self.prompt_to_metadata.get(prompt_text)
        if meta is None:
            return None
        return {
            "text_id": meta.text_id,
            "prompt_text": meta.prompt_text,
            "tree": meta.tree,
            "segment": meta.segment,
            "bin": meta.bin,
            "stent": meta.stent,
            "category": meta.category,
            "prompt_bucket": meta.prompt_bucket,
            "disease_severity": meta.disease_severity,
            "is_abnormal": meta.prompt_bucket == "abnormal" or meta.disease_severity not in {"normal", "mild"},
        }
