import os
import cv2
import torch
import random
import pathlib
import collections

import numpy as np
import pandas as pd

from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from torch.utils.data import DataLoader, get_worker_info

from utils.enums import RunMode
from utils.seed import seed_worker
from utils.ddp import DistributedUtils
from models.text_encoder import get_tokenizer
from utils.video import load_video, format_mean_std
from utils.config.heartwise_config import HeartWiseConfig
from dataloaders.csv_utils import read_csv_with_fallback
from dataloaders.siglip_support import SiglipSupport


class VideoClipDataset(torch.utils.data.Dataset):
    """
    Unified dataset class for single- and multi-video (grouped) video-text pairs.
    """

    def __init__(
        self,
        root: str,
        data_filename: str,
        split: str,
        target_label: Optional[str],
        datapoint_loc_label: str = "target_video_path",
        num_frames: int = 16,
        backbone: str = "default",
        debug_mode: bool = False,
        normalize: bool = True,
        mean: Optional[Any] = None,
        std: Optional[Any] = None,
        stride: int = 1,
        groupby_column: Optional[str] = None,
        num_videos: int = 4,
        shuffle_videos: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.split = split
        self.datapoint_loc_label = datapoint_loc_label
        self.debug_mode = debug_mode
        self.backbone = backbone
        self.num_frames = num_frames
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.normalize = normalize
        self.stride = np.random.randint(1, stride + 1) if stride > 1 else 1 if split == RunMode.TRAIN else stride
        self.groupby_column = groupby_column
        self.num_videos = num_videos
        self.shuffle_videos = shuffle_videos
        self.seed = seed
        self.multi_video_mode = kwargs.pop("multi_video", False)
        self.video_positive_texts: List[List[Tuple[str, float]]] = []
        self.video_negative_pool: List[List[str]] = []
        self.video_ids: List[Optional[str]] = []
        self.video_path_to_idx: Dict[str, int] = {}
        self.video_trees: List[Optional[str]] = []
        self.main_structure_labels: List[int] = []
        self.labels: List[str] = []
        self._siglip_neg_cursors: Dict[int, int] = collections.defaultdict(int)
        self._warned_missing_target_label: bool = False

        self.video_transforms = kwargs.pop("video_transforms", None)
        self.rand_augment = kwargs.pop("rand_augment", False) if split == RunMode.TRAIN else False
        self.resize = kwargs.pop("resize", 224)
        self.max_length = kwargs.pop("max_length", 250)

        self.siglip_texts_path = kwargs.get("siglip_texts_path")
        self.siglip_enabled = bool(self.siglip_texts_path)
        self.siglip_max_positive_per_video = int(kwargs.get("siglip_max_positive_per_video", 8))
        self.siglip_negatives_per_video = int(kwargs.get("siglip_negatives_per_video", 0))
        siglip_kwargs = dict(kwargs)
        self.siglip: Optional[SiglipSupport] = (
            SiglipSupport(self, siglip_kwargs) if self.siglip_enabled else None
        )

        # Early validation for multi-video mode
        if self.multi_video_mode and (self.groupby_column is None or not self.groupby_column):
            raise ValueError(
                "groupby_column must be specified when multi_video is True. "
                "This is required to group videos by study instance."
            )

        if self.seed is not None:
            random.seed(self.seed)
            print(f"[VideoClipDataset] seed={self.seed} for random video sampling")
        else:
            print(f"[VideoClipDataset] no seed for random video sampling")

        if self.siglip_enabled and self.siglip is not None:
            self.siglip.reload_resources()

        if self.split != "inference":
            target_label = (
                [target_label]
                if target_label and not isinstance(target_label, list)
                else target_label
            )
            self.target_label = target_label
            self.external_test_location = kwargs.pop("external_test_location", None)

            if self.multi_video_mode:
                print("Initializing multi-video mode")
                self._init_multi_video()
            else:
                print("Initializing single-video mode")
                self.fnames, self.outcome, self.target_index = self.load_data(
                    self.split, self.target_label
                )

            # Initialize tokenizer only once
            if not hasattr(self, 'tokenizer'):
                try:
                    self.tokenizer = get_tokenizer()
                    print("Tokenizer initialized successfully")
                except Exception as e:
                    print(f"Error initializing tokenizer: {str(e)}")
                    raise RuntimeError("Failed to initialize tokenizer") from e

        if self.debug_mode and not self.multi_video_mode:
            print("Validating all videos in single-video mode with groupby_column", self.groupby_column)
            self.valid_indices = self._validate_all_videos()
        elif not self.multi_video_mode:
            print("Initializing single-video mode")
            self.valid_indices = list(range(len(self.fnames)))
            # For compatibility with unit tests that access `study_ids` even
            # in single-video mode, expose it as the list of file names.
            self.study_ids = [str(f) for f in self.fnames]

    def _init_multi_video(self):
        self.study_to_videos = collections.defaultdict(list)
        self.study_to_text = {}
        csv_path = self.folder / self.filename
        df = pd.read_csv(csv_path, sep="α", engine="python")
        df_split = df[df["Split"].str.lower() == self.split.lower()].copy()
        missing_studies = 0
        # Determine the text column name
        text_col = None
        if self.target_label is not None:
            if isinstance(self.target_label, list):
                text_col = self.target_label[0]
            else:
                text_col = self.target_label
        for _, row in df_split.iterrows():
            group_val = row.get(self.groupby_column, None)
            if group_val is None or not pd.notna(group_val):
                continue
            sid = str(group_val)
            fpath = row[self.datapoint_loc_label]
            if not os.path.exists(fpath):
                missing_studies += 1
                continue
            self.study_to_videos[sid].append(fpath)
            text_val = row.get(text_col, None) if text_col is not None else None
            if text_val is not None and pd.notna(text_val):
                val = text_val
            else:
                val = "No Report"
            self.study_to_text[sid] = str(val)
        # Filter out None keys
        self.study_ids = sorted([k for k in self.study_to_videos.keys() if k is not None])
        print(f"[VideoClipDataset] Found {len(self.study_ids)} studies in split='{self.split}'")
        print(f"[VideoClipDataset] Missing {missing_studies} studies in split='{self.split}'")

    def __len__(self):
        if self.multi_video_mode:
            return len(self.study_ids)
        return len(self.valid_indices)

    def load_data(self, split: str, target_label: Optional[List[str]]):
        file_path = os.path.join(self.folder, self.filename)
        expected_cols = [self.datapoint_loc_label, "Split"]
        if target_label and len(target_label) > 0 and target_label[0]:
            expected_cols.append(target_label[0])
        data = read_csv_with_fallback(file_path, expected_columns=expected_cols)

        print(f"\nAvailable splits in dataset:")
        print(data["Split"].value_counts())

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        desired_split = str(split).strip().lower() if split is not None else "all"
        if not desired_split or desired_split == "nan":
            desired_split = "all"

        target_index = None
        if target_label and len(target_label) > 0 and target_label[0]:
            target_index = data.columns.get_loc(target_label[0])

        fnames: List[str] = []
        outcome: List[str] = []

        total_rows = 0
        valid_files = 0
        split_matches = 0

        self.video_positive_texts.clear()
        self.video_negative_pool.clear()
        self.video_ids.clear()
        self.video_path_to_idx.clear()
        self.video_trees.clear()
        self.main_structure_labels.clear()
        self.labels.clear()

        for row_idx, row in data.iterrows():
            total_rows += 1
            file_name = row.iloc[filename_index]
            raw_split_value = row.iloc[split_index]
            if pd.isna(raw_split_value) or (isinstance(raw_split_value, str) and not raw_split_value.strip()):
                raise ValueError(
                    f"Row {row_idx} in '{file_path}' has an empty Split value for file '{file_name}'. "
                    "Populate the Split column with a valid split name."
                )
            file_mode = str(raw_split_value).lower().strip()

            if self.external_test_location and self.split == "external_test":
                full_path = os.path.join(self.external_test_location, file_name)
            else:
                full_path = file_name

            if not os.path.exists(full_path):
                continue

            valid_files += 1
            if desired_split != "all" and file_mode not in {"all", desired_split}:
                continue

            split_matches += 1
            fnames.append(full_path)
            if target_index is not None:
                outcome.append(str(row.iloc[target_index]))
            else:
                outcome.append("")

            video_id_str = None
            main_structure_value = self._extract_main_structure_value(row)
            tree_hint = self._normalize_tree_key(main_structure_value)
            if self.siglip_enabled and self.siglip is not None:
                video_id_val = row.get(self.siglip.video_id_column)
                if video_id_val is not None and not pd.isna(video_id_val):
                    video_id_str = str(video_id_val)

                positive_pairs = self.siglip.get_positive_pairs(video_id_str)
                if not positive_pairs:
                    positive_pairs = self._fallback_positive_pairs(row)

                valid_pairs: List[Tuple[str, float]] = []
                positive_ids: set[str] = set()
                for text_id, weight in positive_pairs:
                    if text_id not in self.siglip.text_lookup:
                        continue
                    scaled_weight = self.siglip.compute_positive_weight(text_id, weight)
                    valid_pairs.append((text_id, scaled_weight))
                    positive_ids.add(text_id)

                if valid_pairs and not tree_hint:
                    tree_hint = self._resolve_tree_from_positive_pairs(valid_pairs)
                tree_hint = self._normalize_tree_key(tree_hint)

                filtered_pairs = self.siglip.filter_positive_pairs(
                    valid_pairs,
                    tree_hint=tree_hint,
                )
                if filtered_pairs:
                    valid_pairs = filtered_pairs
                else:
                    dedup_pairs = self.siglip.filter_positive_pairs(valid_pairs)
                    if dedup_pairs:
                        valid_pairs = dedup_pairs

                negative_ids = self.siglip.build_negative_candidates(positive_ids, tree_hint)
                if not negative_ids:
                    negative_ids = self._fallback_negative_ids(row, positive_ids)

                has_abnormal = any(
                    self.siglip.text_label_lookup.get(text_id, "normal") == "abnormal"
                    for text_id, _ in valid_pairs
                )
                self.labels.append("abnormal" if has_abnormal else "normal")
                self.video_positive_texts.append(valid_pairs)
                self.video_negative_pool.append(negative_ids)
            else:
                self.labels.append("unknown")
                self.video_positive_texts.append([])
                self.video_negative_pool.append([])

            self.video_ids.append(video_id_str)
            self.video_path_to_idx[full_path] = len(fnames) - 1
            if tree_hint is None:
                tree_hint = self._infer_tree_from_structure(main_structure_value)
            tree_hint = self._normalize_tree_key(tree_hint)
            self.video_trees.append(tree_hint)
            self.main_structure_labels.append(self._tree_to_label(tree_hint))

        print(f"\nDataset loading statistics for split '{split}':")
        print(f"Total rows in CSV: {total_rows}")
        print(f"Valid files found: {valid_files}")
        print(f"Matching split '{split}': {split_matches}")
        print(f"Final dataset size: {len(fnames)}")

        if len(fnames) == 0:
            raise ValueError(
                f"No samples found for split '{split}'. "
                f"Available splits: {data['Split'].unique()}. "
                "Check your data split assignments."
            )

        return fnames, outcome, target_index

    def _validate_all_videos(self):
        print("Validating all videos in dataset...")
        valid_indices = []
        self.failed_videos = []

        for idx, fname in enumerate(self.fnames):
            try:
                cap = cv2.VideoCapture(fname)
                if not cap.isOpened():
                    raise ValueError(f"Unable to open video {fname}")
                cap.release()
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Failed to load video {fname}: {str(e)}")
                self.failed_videos.append((fname, str(e)))

        print(f"Found {len(valid_indices)} valid videos out of {len(self.fnames)}")
        if self.failed_videos:
            print(f"Failed to load {len(self.failed_videos)} videos")

        return valid_indices

    def __getitem__(self, index: int) -> tuple:
        if self.multi_video_mode:
            sid = self.study_ids[index]
            assert isinstance(sid, str), f"sid must be a string, got {type(sid)}"
            vid_paths = self.study_to_videos[sid]
            text_report = self.study_to_text[sid]
            main_structure_label = -1
            for vp in vid_paths:
                mapped_idx = self.video_path_to_idx.get(vp)
                if mapped_idx is not None and mapped_idx < len(self.main_structure_labels):
                    main_structure_label = self.main_structure_labels[mapped_idx]
                    break
            if self.shuffle_videos:
                vid_paths = random.sample(vid_paths, len(vid_paths))
            if len(vid_paths) > self.num_videos:
                chose_paths = vid_paths[:self.num_videos]
            else:
                chose_paths = vid_paths
            loaded = []
            for vp in chose_paths:
                try:
                    arr = load_video(
                        vp,
                        n_frames=16 if self.backbone.lower() == "mvit" else self.num_frames,
                        resize=self.resize,
                        normalize=self.normalize,
                        mean=self.mean[0] if isinstance(self.mean, list) else self.mean,
                        std=self.std[0] if isinstance(self.std, list) else self.std,
                        video_transforms=self.video_transforms,
                        rand_augment=self.rand_augment,
                        backbone=self.backbone,
                        stride=self.stride,
                    )
                except Exception as e:
                    print(f"Warning: {vp} load error: {e}")
                    arr = np.zeros((16 if self.backbone.lower() == "mvit" else self.num_frames, self.resize, self.resize, 3), dtype=np.float32)
                loaded.append(arr)
            while len(loaded) < self.num_videos:
                arr = np.zeros((16 if self.backbone.lower() == "mvit" else self.num_frames, self.resize, self.resize, 3), dtype=np.float32)
                loaded.append(arr)
            multi_stack = np.stack(loaded, axis=0)

            encoded = self.tokenizer(
                text_report,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.squeeze(0) for k, v in encoded.items()}
            return multi_stack, encoded, sid, main_structure_label
        else:
            actual_idx = self.valid_indices[index]
            video_fname = self.fnames[actual_idx]
            main_structure_label = (
                self.main_structure_labels[actual_idx]
                if actual_idx < len(self.main_structure_labels)
                else -1
            )

            try:
                video = load_video(
                    video_fname,
                    n_frames=16 if self.backbone.lower() == "mvit" else self.num_frames,
                    resize=self.resize,
                    normalize=self.normalize,
                    mean=self.mean[0] if isinstance(self.mean, list) else self.mean,
                    std=self.std[0] if isinstance(self.std, list) else self.std,
                    video_transforms=self.video_transforms,
                    rand_augment=self.rand_augment,
                    backbone=self.backbone,
                    stride=self.stride,
                )

                if self.backbone.lower() == "mvit" and video.shape[0] != 16:
                    raise ValueError(f"Expected 16 frames for MViT, got {video.shape[0]}")
 
                encoded = None
                if self.split != "inference" and self.target_label is not None and self.target_index is not None:
                    text = self.outcome[actual_idx]
                    if not isinstance(text, str):
                        text = str(text)
                    encoded = self.tokenizer(
                        text,
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoded = {k: v.squeeze(0) for k, v in encoded.items()}
                    # If you want them on GPU, do it in the training loop, not here
                else:
                    if not self._warned_missing_target_label:
                        worker = get_worker_info()
                        if worker is None or worker.id == 0:
                            print("Warning: target_label is not set; using external text sources for training.")
                        self._warned_missing_target_label = True
                    encoded = None
                    if self.siglip_enabled and self.siglip is not None:
                        siglip_pairs = self.video_positive_texts[actual_idx] if actual_idx < len(self.video_positive_texts) else []
                        if siglip_pairs:
                            first_text_id = siglip_pairs[0][0]
                            meta = self.siglip.text_lookup.get(first_text_id, {})
                            prompt_text = meta.get("prompt_text", "")
                            if prompt_text:
                                encoded = self.tokenizer(
                                    prompt_text,
                                    padding="max_length",
                                    max_length=512,
                                    truncation=True,
                                    return_tensors="pt",
                                )
                                encoded = {k: v.squeeze(0) for k, v in encoded.items()}
                    if encoded is None:
                        blank = self.tokenizer(
                            "",
                            padding="max_length",
                            max_length=512,
                            truncation=True,
                            return_tensors="pt",
                        )
                        encoded = {k: v.squeeze(0) for k, v in blank.items()}

                return video, encoded, video_fname, main_structure_label

            except Exception as e:
                raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e

    def get_reports(self, ids: List[str]) -> List[str]:
        filtered_ids = [x for x in ids if x is not None]
        if self.multi_video_mode:
            return [self.study_to_text.get(sid, "") for sid in filtered_ids]
        else:
            reports = []
            for path in filtered_ids:
                idx = self.video_path_to_idx.get(str(path))
                if idx is None:
                    try:
                        idx = self.fnames.index(str(path))
                    except ValueError:
                        print(f"Warning: No report found for video {path}")
                        reports.append("")
                        continue
                reports.append(self._get_report_for_index(idx))
            return reports

    def get_all_reports(self):
        if self.multi_video_mode:
            return [self.study_to_text[sid] for sid in self.study_ids]
        else:
            return [self._get_report_for_index(idx) for idx in range(len(self.fnames))]

    def _get_report_for_index(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.outcome):
            return ""
        base_text = str(self.outcome[idx]) if idx < len(self.outcome) else ""
        normalized = base_text.strip()
        if normalized and normalized.lower() not in {"nan", "none"}:
            return normalized
        siglip_report = self._compose_siglip_report(idx)
        return siglip_report if siglip_report else ""

    def _compose_siglip_report(self, dataset_idx: int) -> str:
        if (
            not self.siglip_enabled
            or self.siglip is None
            or dataset_idx < 0
            or dataset_idx >= len(self.video_positive_texts)
        ):
            return ""

        pairs = self.video_positive_texts[dataset_idx]
        if not pairs:
            return ""

        cap = getattr(self, "siglip_max_positive_per_video", None)
        if isinstance(cap, int) and cap > 0:
            pairs = pairs[:cap]

        prompt_texts: List[str] = []
        seen: Set[str] = set()
        for text_id, _ in pairs:
            meta = self.siglip.text_lookup.get(text_id)
            if meta is None:
                continue
            prompt_text = meta.get("prompt_text")
            if not isinstance(prompt_text, str):
                continue
            cleaned = prompt_text.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            prompt_texts.append(cleaned)

        return "; ".join(prompt_texts)

    def get_video_paths(self, sid: Optional[str]) -> List[str]:
        if sid is None:
            return []
        if self.multi_video_mode:
            return self.study_to_videos.get(sid, [])
        else:
            return [sid] if sid in self.fnames else []

    @staticmethod
    def _parse_tags(tag_str: Any) -> Dict[str, str]:
        if not isinstance(tag_str, str) or not tag_str.strip():
            return {}
        tags: Dict[str, str] = {}
        for chunk in str(tag_str).split("|"):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key:
                tags[key] = value
        return tags

    @staticmethod
    def _is_abnormal_prompt(prompt_type: Optional[str], prompt_text: Optional[str]) -> bool:
        if isinstance(prompt_type, str) and prompt_type.strip().lower() == "abnormal":
            return True
        if not isinstance(prompt_text, str):
            return False
        lowered = prompt_text.lower()
        for keyword in ["stenosis", "calcification", "thrombus", "in-stent", "occlusion", "cto"]:
            if keyword in lowered:
                return True
        return False

    @staticmethod
    def _extract_main_structure_value(row: pd.Series) -> Optional[str]:
        def _map_numeric(value: Any) -> Optional[str]:
            try:
                num = int(value)
            except (TypeError, ValueError):
                return None
            if num == 0:
                return "left"
            if num == 1:
                return "right"
            return None

        primary = row.get("main_structure")
        if primary is not None and not pd.isna(primary):
            if isinstance(primary, str) and primary.strip():
                return primary
            mapped = _map_numeric(primary)
            if mapped:
                return mapped

        secondary = row.get("main_structure_name")
        if secondary is not None and not pd.isna(secondary):
            if isinstance(secondary, str) and secondary.strip():
                return secondary
            mapped = _map_numeric(secondary)
            if mapped:
                return mapped

        return None

    @staticmethod
    def _normalize_tree_key(tree_value: Optional[str]) -> Optional[str]:
        if not isinstance(tree_value, str):
            return None
        text = tree_value.strip().lower()
        if not text:
            return None
        mapping = {
            "left": {"left", "lad", "lcx", "diagonal", "circumflex"},
            "right": {"right", "rca"},
        }
        for canonical, aliases in mapping.items():
            if text in aliases or any(alias in text for alias in aliases):
                return canonical
        return text

    @staticmethod
    def _infer_tree_from_structure(main_structure: Optional[str]) -> Optional[str]:
        if not isinstance(main_structure, str):
            return None
        lowered = main_structure.lower()
        if "left" in lowered or "lad" in lowered or "circ" in lowered:
            return "left"
        if "right" in lowered or "rca" in lowered:
            return "right"
        return None

    @staticmethod
    def _tree_to_label(tree_hint: Optional[str]) -> int:
        normalized = VideoClipDataset._normalize_tree_key(tree_hint)
        if normalized == "left":
            return 0
        if normalized == "right":
            return 1
        return -1

    def _resolve_tree_from_positive_pairs(
        self,
        pairs: List[Tuple[str, float]],
    ) -> Optional[str]:
        if not self.siglip:
            return None
        for text_id, _ in pairs:
            meta = self.siglip.text_lookup.get(text_id)
            if meta is None:
                continue
            tree = meta.get("tree")
            if tree:
                normalized = self._normalize_tree_key(tree)
                if normalized:
                    return normalized
        return None

    def _fallback_positive_pairs(self, row: pd.Series) -> List[Tuple[str, float]]:
        pos_ids_str = row.get("positive_text_ids", "")
        if not isinstance(pos_ids_str, str) or not pos_ids_str.strip():
            return []
        pairs: List[Tuple[str, float]] = []
        for tid in pos_ids_str.split("|"):
            tid = tid.strip()
            if not tid:
                continue
            pairs.append((tid, 1.0))
        return pairs

    def _fallback_negative_ids(
        self,
        row: pd.Series,
        exclude_ids: Iterable[str],
    ) -> List[str]:
        exclude = {tid for tid in exclude_ids}
        neg_ids_str = row.get("negative_text_ids", "")
        if isinstance(neg_ids_str, str) and neg_ids_str.strip():
            candidates = [
                tid.strip()
                for tid in neg_ids_str.split("|")
                if tid.strip() and tid.strip() not in exclude
            ]
            if candidates:
                return candidates
        if self.siglip is None:
            return []
        fallback = [
            text_id
            for text_id in self.siglip.text_lookup.keys()
            if text_id not in exclude
        ]
        return fallback

    def _cap_positive_pairs_by_segment(
        self,
        pairs: List[Tuple[str, float]],
        cap: int,
    ) -> List[Tuple[str, float]]:
        """
        Keep at most one positive per segment (based on SigLIP metadata) up to `cap`.
        Falls back to naive truncation if metadata is unavailable.
        """
        if not pairs or cap <= 0:
            return []
        if self.siglip is None or not self.siglip.text_lookup:
            return pairs[:cap]

        kept: List[Tuple[str, float]] = []
        seen_segments: Set[str] = set()
        for text_id, weight in pairs:
            meta = self.siglip.text_lookup.get(text_id) or {}
            seg_value = str(meta.get("segment") or "").strip().lower()
            if seg_value:
                if seg_value in seen_segments:
                    continue
                seen_segments.add(seg_value)
            kept.append((text_id, weight))
            if len(kept) >= cap:
                break
        return kept

    @staticmethod
    def _resolve_exam_severity(severity_votes: Iterable[str]) -> str:
        votes = {str(v).lower() for v in severity_votes if v}
        if "severe" in votes or "critical" in votes:
            return "SEVERE"
        if "moderate" in votes:
            return "MODERATE"
        if "mild" in votes:
            return "MILD"
        return "NORMAL"

    @staticmethod
    def _read_metadata_csv(csv_path: pathlib.Path, expected_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return read_csv_with_fallback(csv_path, expected_columns=expected_columns)

    def _sample_siglip_negatives(self, dataset_index: int, pool: List[str], limit: int) -> List[str]:
        if limit <= 0:
            limit = self.siglip_negatives_per_video
        if limit <= 0 or not pool:
            return []
        cursor = self._siglip_neg_cursors[dataset_index]
        sampled: List[str] = []
        pool_len = len(pool)
        for _ in range(min(limit, pool_len)):
            sampled.append(pool[cursor % pool_len])
            cursor += 1
        self._siglip_neg_cursors[dataset_index] = cursor
        return sampled

    def build_siglip_batch(self, sample_paths: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        if (
            not self.siglip_enabled
            or self.siglip is None
            or not sample_paths
            or self.multi_video_mode
        ):
            return None
        video_indices: List[Optional[int]] = [self.video_path_to_idx.get(str(p)) for p in sample_paths]
        if all(idx is None for idx in video_indices):
            return None

        positive_lists: List[List[Tuple[str, float]]] = []
        negative_lists: List[List[str]] = []
        unique_text_ids: List[str] = []
        text_to_col: Dict[str, int] = {}

        for vid_idx, dataset_idx in enumerate(video_indices):
            if dataset_idx is None or dataset_idx >= len(self.video_positive_texts):
                positive_lists.append([])
                negative_lists.append([])
                continue

            raw_pos = self.video_positive_texts[dataset_idx]
            cap = max(1, int(self.siglip_max_positive_per_video))
            pos_subset = self._cap_positive_pairs_by_segment(raw_pos, cap)
            positive_lists.append(pos_subset)
            neg_pool = self.video_negative_pool[dataset_idx]
            neg_limit = self.siglip_negatives_per_video or getattr(self.siglip, "negatives_per_video", 0)
            negative_lists.append(self._sample_siglip_negatives(dataset_idx, neg_pool, neg_limit))

            for text_id, _ in pos_subset:
                if text_id not in text_to_col:
                    text_to_col[text_id] = len(unique_text_ids)
                    unique_text_ids.append(text_id)
            for text_id in negative_lists[-1]:
                if text_id not in text_to_col:
                    text_to_col[text_id] = len(unique_text_ids)
                    unique_text_ids.append(text_id)

        if not unique_text_ids:
            return None

        encodings = self.siglip.encode_text_batch(unique_text_ids)
        pos_mask = torch.zeros(len(sample_paths), len(unique_text_ids), dtype=torch.float32)
        pos_weights = torch.zeros_like(pos_mask)

        for row_idx, pairs in enumerate(positive_lists):
            for text_id, weight in pairs:
                col = text_to_col.get(text_id)
                if col is None:
                    continue
                pos_mask[row_idx, col] = 1.0
                pos_weights[row_idx, col] += float(weight)

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "positive_mask": pos_mask,
            "positive_weights": pos_weights,
        }

def custom_collate_fn(batch):
    """Custom collate function to handle video, text, and tree labels."""
    import numpy as np
    import torch

    videos_list = []
    encoded_list = []
    paths_list = []
    main_structures: List[int] = []

    for item in batch:
        if len(item) == 4:
            video, encoded_text, path, tree_label = item
        else:
            video, encoded_text, path = item
            tree_label = -1
        videos_list.append(video)
        encoded_list.append(encoded_text)
        paths_list.append(path)
        main_structures.append(tree_label if tree_label is not None else -1)

    first_video = videos_list[0]
    if isinstance(first_video, np.ndarray) and first_video.ndim == 5:
        videos = torch.from_numpy(np.stack(videos_list, axis=0))
    else:
        videos = torch.stack([torch.from_numpy(v) for v in videos_list])

    if encoded_list[0] is not None:
        combined_texts = {
            "input_ids": torch.stack([text["input_ids"] for text in encoded_list]),
            "attention_mask": torch.stack([text["attention_mask"] for text in encoded_list]),
        }
    else:
        combined_texts = None

    main_structure_tensor: Optional[torch.Tensor]
    if all(label == -1 for label in main_structures):
        main_structure_tensor = None
    else:
        main_structure_tensor = torch.tensor(main_structures, dtype=torch.long)

    return {
        "videos": videos,
        "encoded_texts": combined_texts,
        "paths": paths_list,
        "main_structure": main_structure_tensor,
    }

def get_distributed_video_clip_dataloader(
    config: HeartWiseConfig,
    split: str,
    mean: List[float],
    std: List[float],
    shuffle: bool,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    drop_last: bool = True,
) -> DataLoader:
    # Create the video dataset
    video_dataset = VideoClipDataset(
        root=getattr(config, 'root', '') or '',
        data_filename=getattr(config, 'data_filename', '') or '',
        split=split,
        target_label=getattr(config, 'target_label', '') or '',
        datapoint_loc_label=getattr(config, 'datapoint_loc_label', '') or '',
        num_frames=getattr(config, 'frames', 32),
        backbone=getattr(config, 'model_name', 'default'),
        mean=mean,
        std=std,
        rand_augment=getattr(config, 'rand_augment', False),
        stride=getattr(config, 'stride', 1),
        groupby_column=getattr(config, 'groupby_column', None),
        num_videos=getattr(config, 'num_videos', 4),
        shuffle_videos=getattr(config, 'shuffle_videos', False),
        seed=getattr(config, 'seed', None),
        multi_video=getattr(config, 'multi_video', False),
        video_transforms=getattr(config, 'video_transforms', None),
        resize=getattr(config, 'resize', 224),
        max_length=getattr(config, 'max_length', 250),
        siglip_texts_path=getattr(config, 'siglip_texts_path', None),
        siglip_max_positive_per_video=getattr(config, 'siglip_max_positive_per_video', 8),
        siglip_negatives_per_video=getattr(config, 'siglip_negatives_per_video', 0),
        siglip_round_robin_sampling=getattr(config, 'siglip_round_robin_sampling', False),
        siglip_max_segments_per_video=getattr(config, 'siglip_max_segments_per_video', 15),
        siglip_positive_severity_weights=getattr(config, 'siglip_positive_severity_weights', None),
    )
    # Create a sampler for distributed training
    sampler = DistributedUtils.DS.DistributedSampler(
        video_dataset, 
        shuffle=shuffle, 
        num_replicas=num_replicas, 
        rank=rank
    )
    # Create the dataloader
    return DataLoader(
        video_dataset,
        batch_size=getattr(config, 'batch_size', 1),
        sampler=sampler,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
    )
