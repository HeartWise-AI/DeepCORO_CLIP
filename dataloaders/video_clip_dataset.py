import os
import torch
import random
import pathlib
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

from utils.seed import seed_worker
from utils.ddp import DistributedUtils
from models.text_encoder import get_tokenizer
from utils.video import load_video, format_mean_std, create_video_capture
from utils.config.heartwise_config import HeartWiseConfig


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
        self.stride = stride
        self.groupby_column = groupby_column
        self.num_videos = num_videos
        self.shuffle_videos = shuffle_videos
        self.seed = seed
        self.multi_video_mode = kwargs.pop("multi_video", False)

        # SigLIP negatives configuration
        self.siglip_negatives_per_video: int = int(kwargs.pop("siglip_negatives_per_video", 0))
        self.siglip_pos_samples_per_video: int = max(1, int(kwargs.pop("siglip_pos_samples_per_video", 1)))
        self.siglip_round_robin_sampling: bool = bool(kwargs.pop("siglip_round_robin_sampling", False))

        # Placeholders populated after metadata loading
        self.video_positive_texts: List[List[Dict[str, Any]]] = []
        self.video_ids: List[Optional[str]] = []
        self.video_path_to_idx: Dict[str, int] = {}
        self.main_structures: List[str] = []
        self.video_negative_pool: List[List[Dict[str, Any]]] = []
        self._siglip_pos_cursors: List[int] = []

        # Optional SigLIP resources (videos/texts/edges manifest)
        self.siglip_texts_path: Optional[str] = kwargs.pop("siglip_texts_path", None)
        self.siglip_edges_path: Optional[str] = kwargs.pop("siglip_edges_path", None)
        self.siglip_video_id_column: str = kwargs.pop("siglip_video_id_column", "video_id")
        self.siglip_text_id_column: str = kwargs.pop("siglip_text_id_column", "text_id")
        self.siglip_prompt_text_column: str = kwargs.pop("siglip_prompt_text_column", "prompt_text")
        self.siglip_prompt_type_column: str = kwargs.pop("siglip_prompt_type_column", "prompt_type")
        self.siglip_soft_weight_column: str = kwargs.pop("siglip_soft_weight_column", "soft_weight")
        self.siglip_edge_weight_column: str = kwargs.pop("siglip_edge_weight_column", "weight")
        self.siglip_enabled: bool = bool(self.siglip_texts_path and self.siglip_edges_path)

        if self.siglip_enabled and self.multi_video_mode:
            raise ValueError(
                "SigLIP multiprompt sampling is not supported when multi_video=True."
            )

        # Early validation for multi-video mode
        if self.multi_video_mode and (self.groupby_column is None or not self.groupby_column):
            raise ValueError(
                "groupby_column must be specified when multi_video is True. "
                "This is required to group videos by study instance."
            )

        self.video_transforms = kwargs.pop("video_transforms", None)
        self.rand_augment = kwargs.pop("rand_augment", False)
        self.resize = kwargs.pop("resize", 224)
        self.max_length = kwargs.pop("max_length", 250)

        if self.seed is not None:
            random.seed(self.seed)
            print(f"[VideoClipDataset] seed={self.seed} for random video sampling")
        else:
            print(f"[VideoClipDataset] no seed for random video sampling")

        # SigLIP resources
        self._siglip_text_lookup: Dict[str, Dict[str, Any]] = {}
        self._siglip_video_to_texts: Dict[str, List[Dict[str, Any]]] = {}
        self._siglip_tree_to_texts: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self._siglip_all_text_entries: List[Dict[str, Any]] = []
        if self.siglip_enabled:
            self._load_siglip_resources()

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

    def _read_metadata_csv(self, csv_path: Path | str) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

        try:
            df_alpha = pd.read_csv(csv_path, sep="α", engine="python")
            if df_alpha.shape[1] > 1:
                return df_alpha
        except Exception:
            pass

        return pd.read_csv(csv_path)

    def _load_siglip_resources(self) -> None:
        texts_path = Path(self.siglip_texts_path).expanduser().resolve()
        edges_path = Path(self.siglip_edges_path).expanduser().resolve()

        if not texts_path.exists():
            raise FileNotFoundError(f"SigLIP texts.csv not found at {texts_path}")
        if not edges_path.exists():
            raise FileNotFoundError(f"SigLIP edges.csv not found at {edges_path}")

        texts_df = pd.read_csv(texts_path)
        edges_df = pd.read_csv(edges_path)

        required_text_cols = {self.siglip_text_id_column, self.siglip_prompt_text_column}
        missing_text_cols = required_text_cols - set(texts_df.columns)
        if missing_text_cols:
            raise ValueError(
                f"texts.csv is missing required columns: {sorted(missing_text_cols)}"
            )

        required_edge_cols = {self.siglip_video_id_column, self.siglip_text_id_column}
        missing_edge_cols = required_edge_cols - set(edges_df.columns)
        if missing_edge_cols:
            raise ValueError(
                f"edges.csv is missing required columns: {sorted(missing_edge_cols)}"
            )

        self._siglip_text_lookup.clear()
        self._siglip_tree_to_texts.clear()
        self._siglip_all_text_entries.clear()
        for _, row in texts_df.iterrows():
            text_id = str(row[self.siglip_text_id_column])
            tags = self._parse_tags(row.get("tags", ""))
            tree = tags.get("tree")
            text_info = {
                "text_id": text_id,
                "prompt_text": row[self.siglip_prompt_text_column],
                "prompt_type": row.get(self.siglip_prompt_type_column),
                "soft_weight": float(row.get(self.siglip_soft_weight_column, 1.0)),
                "tree": tree,
                "tags": tags,
            }
            self._siglip_text_lookup[text_id] = text_info
            if tree:
                self._siglip_tree_to_texts[tree].append(text_info)
            self._siglip_all_text_entries.append(text_info)

        self._siglip_video_to_texts = collections.defaultdict(list)
        for _, row in edges_df.iterrows():
            video_id = str(row[self.siglip_video_id_column])
            text_id = str(row[self.siglip_text_id_column])
            text_info = self._siglip_text_lookup.get(text_id)
            if not text_info:
                continue
            entry = dict(text_info)
            entry["edge_weight"] = float(row.get(self.siglip_edge_weight_column, 1.0))
            self._siglip_video_to_texts[video_id].append(entry)

        print(
            f"[VideoClipDataset] Loaded SigLIP resources: "
            f"{len(self._siglip_text_lookup)} texts, "
            f"{len(self._siglip_video_to_texts)} videos with positives"
        )

    @staticmethod
    def _parse_tags(tag_str: Any) -> Dict[str, str]:
        tags: Dict[str, str] = {}
        if isinstance(tag_str, str):
            for kv in tag_str.split("|"):
                if ":" in kv:
                    key, value = kv.split(":", 1)
                    tags[key.strip()] = value.strip()
        return tags

    @staticmethod
    def _infer_tree_from_structure(main_structure: Optional[str]) -> Optional[str]:
        if not isinstance(main_structure, str):
            return None
        lower = main_structure.lower()
        if "left" in lower:
            return "left"
        if "right" in lower:
            return "right"
        return None

    def _init_multi_video(self):
        self.study_to_videos = collections.defaultdict(list)
        self.study_to_text = {}
        csv_path = self.folder / self.filename
        df = self._read_metadata_csv(csv_path)
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

    def load_data(self, split, target_label):
        file_path = os.path.join(self.folder, self.filename)
        data = self._read_metadata_csv(file_path)

        print(f"\nAvailable splits in dataset:")
        print(data["Split"].value_counts())

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")

        target_index = None
        if target_label is not None:
            first_label = target_label[0]
            if first_label in data.columns:
                target_index = data.columns.get_loc(first_label)
            elif not self.siglip_enabled:
                raise KeyError(f"Target label '{first_label}' not found in dataset columns")

        fnames: List[str] = []
        outcome: List[Any] = []
        self.video_ids = []
        self.video_positive_texts = []
        self.video_path_to_idx = {}
        self.main_structures = []
        self.video_negative_pool = []

        total_rows = 0
        valid_files = 0
        split_matches = 0
        self._siglip_pos_cursors = []

        for _, row in data.iterrows():
            total_rows += 1
            file_name = row.iloc[filename_index]
            file_mode = str(row.iloc[split_index]).lower().strip()

            if self.external_test_location and self.split == "external_test":
                full_path = os.path.join(self.external_test_location, file_name)
            else:
                full_path = file_name

            if not os.path.exists(full_path):
                continue

            valid_files += 1
            if split not in ["all", file_mode]:
                continue

            video_id_val = row.get(self.siglip_video_id_column) if self.siglip_enabled else row.get("video_id")
            video_id_str = str(video_id_val) if video_id_val is not None and pd.notna(video_id_val) else None

            main_structure = str(row.get("main_structure", ""))

            if self.siglip_enabled:
                positives_raw = self._siglip_video_to_texts.get(video_id_str or "", [])
                if not positives_raw:
                    continue  # Skip videos without positives in SigLIP mode
                positives = [dict(p) for p in positives_raw]
                positive_ids = {p.get("text_id") for p in positives if p.get("text_id") is not None}
                tree = self._infer_tree_from_structure(main_structure)
                neg_candidates: List[Dict[str, Any]] = []
                if tree and tree in self._siglip_tree_to_texts:
                    neg_candidates = [dict(t) for t in self._siglip_tree_to_texts[tree] if t.get("text_id") not in positive_ids]
                if not neg_candidates:
                    neg_candidates = [dict(t) for t in self._siglip_all_text_entries if t.get("text_id") not in positive_ids]

                self.video_positive_texts.append(positives)
                self.video_negative_pool.append(neg_candidates)
                fallback_text = positives[0].get("prompt_text", "") if positives else ""
                outcome.append(fallback_text)
                self._siglip_pos_cursors.append(0)
            else:
                if target_index is not None:
                    outcome.append(row.iloc[target_index])
                else:
                    outcome.append("")
                self.video_positive_texts.append([])
                self.video_negative_pool.append([])
                self._siglip_pos_cursors.append(0)

            fnames.append(full_path)
            self.video_ids.append(video_id_str)
            self.video_path_to_idx[full_path] = len(fnames) - 1
            self.main_structures.append(main_structure)
            split_matches += 1

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
                cap = create_video_capture(fname)
                if cap is None:
                    raise ValueError(f"Unable to open video {fname}")
                try:
                    if not cap.isOpened():
                        raise ValueError(f"Unable to open video {fname}")
                finally:
                    cap.release()
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Failed to load video {fname}: {str(e)}")
                self.failed_videos.append((fname, str(e)))

        print(f"Found {len(valid_indices)} valid videos out of {len(self.fnames)}")
        if self.failed_videos:
            print(f"Failed to load {len(self.failed_videos)} videos")

        return valid_indices

    def _sample_siglip_positive_entries(self, idx: int) -> List[Dict[str, Any]]:
        positives: List[Dict[str, Any]] = []
        if 0 <= idx < len(self.video_positive_texts):
            positives = self.video_positive_texts[idx]

        if not positives:
            return []

        k = max(1, self.siglip_pos_samples_per_video)

        if len(positives) == 1 or (k == 1 and not self.siglip_round_robin_sampling):
            chosen = [random.choice(positives)]
        elif self.siglip_round_robin_sampling:
            cursor = self._siglip_pos_cursors[idx] if idx < len(self._siglip_pos_cursors) else 0
            take = min(k, len(positives))
            chosen = [positives[(cursor + offset) % len(positives)] for offset in range(take)]
            if idx < len(self._siglip_pos_cursors):
                self._siglip_pos_cursors[idx] = (cursor + take) % len(positives)
        elif len(positives) <= k:
            chosen = list(positives)
        else:
            chosen = random.sample(positives, k)

        entries: List[Dict[str, Any]] = []
        for pos in chosen:
            prompt_text = str(pos.get("prompt_text", ""))
            encoding = self.tokenizer(
                prompt_text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            entries.append(
                {
                    "text_id": pos.get("text_id"),
                    "prompt_text": prompt_text,
                    "prompt_type": pos.get("prompt_type"),
                    "soft_weight": float(pos.get("soft_weight", 1.0)),
                    "tags": pos.get("tags"),
                    "encoding": {k: v.squeeze(0) for k, v in encoding.items()},
                }
            )

        return entries

    def __getitem__(self, index: int) -> tuple:
        if self.multi_video_mode:
            sid = self.study_ids[index]
            assert isinstance(sid, str), f"sid must be a string, got {type(sid)}"
            vid_paths = self.study_to_videos[sid]
            text_report = self.study_to_text[sid]
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
                    if np.isnan(arr).any():
                        print(f"Warning: NaN frames detected in {vp}; replacing with zeros")
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
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
            return multi_stack, encoded, sid, text_report
        else:
            actual_idx = self.valid_indices[index]
            video_fname = self.fnames[actual_idx]
            video_id = self.video_ids[actual_idx] if actual_idx < len(self.video_ids) else None

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
                if np.isnan(video).any():
                    print(f"Warning: NaN frames detected in {video_fname}; replacing with zeros")
                video = np.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)

                if self.backbone.lower() == "mvit" and video.shape[0] != 16:
                    raise ValueError(f"Expected 16 frames for MViT, got {video.shape[0]}")

                if self.siglip_enabled:
                    positive_entries = self._sample_siglip_positive_entries(actual_idx)
                    if not positive_entries:
                        positive_entries = [
                            {
                                "text_id": None,
                                "prompt_text": "",
                                "prompt_type": None,
                                "soft_weight": 1.0,
                                "tags": None,
                                "encoding": {
                                    k: v.squeeze(0)
                                    for k, v in self.tokenizer(
                                        "",
                                        padding="max_length",
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt",
                                    ).items()
                                },
                            }
                        ]

                    payload = {
                        "positive_entries": positive_entries,
                    }
                    primary_text = positive_entries[0].get("prompt_text", "") if positive_entries else ""
                    return video, payload, video_fname, primary_text
                else:
                    encoded = None
                    raw_text = ""
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
                        raw_text = text

                return video, encoded, video_fname, raw_text

            except Exception as e:
                raise RuntimeError(f"Failed to load video {video_fname}: {str(e)}") from e

    def get_reports(self, ids: List[str]) -> List[str]:
        filtered_ids = [x for x in ids if x is not None]
        if self.multi_video_mode:
            return [self.study_to_text.get(sid, "") for sid in filtered_ids]
        else:
            if self.siglip_enabled:
                reports = []
                for path in filtered_ids:
                    idx = self.video_path_to_idx.get(str(path))
                    if idx is None or idx >= len(self.video_positive_texts):
                        reports.append("")
                        continue
                    positives = self.video_positive_texts[idx]
                    if positives:
                        reports.append(positives[0].get("prompt_text", ""))
                    else:
                        reports.append("")
                return reports
            reports = []
            for path in filtered_ids:
                try:
                    idx = self.fnames.index(str(path))
                    reports.append(str(self.outcome[idx]))
                except ValueError:
                    print(f"Warning: No report found for video {path}")
                    reports.append("")
            return reports

    def get_all_reports(self):
        if self.multi_video_mode:
            return [self.study_to_text[sid] for sid in self.study_ids]
        else:
            return [str(o) for o in self.outcome]

    def get_video_paths(self, sid: Optional[str]) -> List[str]:
        if sid is None:
            return []
        if self.multi_video_mode:
            return self.study_to_videos.get(sid, [])
        else:
            return [sid] if sid in self.fnames else []

    def sample_negative_pack(self, paths: List[str], k: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self.siglip_enabled or k <= 0:
            return None

        if not hasattr(self, 'tokenizer'):
            self.tokenizer = get_tokenizer()

        neg_text_batches: List[str] = []
        valid_mask_rows: List[List[float]] = []

        for path in paths:
            idx = self.video_path_to_idx.get(str(path))
            pool = []
            if idx is not None and idx < len(self.video_negative_pool):
                pool = self.video_negative_pool[idx]

            selected: List[Dict[str, Any]] = []
            if pool:
                if len(pool) >= k:
                    selected = random.sample(pool, k)
                else:
                    selected = pool.copy()
                    if len(selected) < k and self._siglip_all_text_entries:
                        filler = random.choices(self._siglip_all_text_entries, k=k - len(selected))
                        selected.extend([dict(entry) for entry in filler])

            texts = [entry.get("prompt_text", "") for entry in selected[:k]]
            mask = [1.0] * len(texts)

            if len(texts) < k:
                deficit = k - len(texts)
                texts.extend(["" for _ in range(deficit)])
                mask.extend([0.0 for _ in range(deficit)])

            neg_text_batches.extend(texts)
            valid_mask_rows.append(mask)

        if not neg_text_batches:
            return None

        encoded = self.tokenizer(
            neg_text_batches,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        B = len(paths)
        K = k
        input_ids = encoded["input_ids"].view(B, K, -1)
        attention_mask = encoded["attention_mask"].view(B, K, -1)
        valid_mask_tensor = torch.tensor(valid_mask_rows, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "valid_mask": valid_mask_tensor,
        }

def custom_collate_fn(batch):
    """Custom collate function to handle video and text data."""
    videos, payloads, paths, raw_texts = zip(*batch)
    import numpy as np
    import torch

    multi_pos_mode = (
        isinstance(payloads[0], dict)
        and payloads[0] is not None
        and "positive_entries" in payloads[0]
    )

    if multi_pos_mode:
        videos_tensor = torch.stack([torch.from_numpy(v).float() for v in videos])

        text_id_to_idx: Dict[str, int] = {}
        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        text_meta: List[Dict[str, Any]] = []

        for payload in payloads:
            for entry in payload.get("positive_entries", []):
                text_id = str(entry.get("text_id"))
                if text_id is None:
                    continue
                if text_id not in text_id_to_idx:
                    idx = len(text_id_to_idx)
                    text_id_to_idx[text_id] = idx
                    encoding = entry["encoding"]
                    input_ids_list.append(encoding["input_ids"].clone())
                    attention_mask_list.append(encoding["attention_mask"].clone())
                    text_meta.append(
                        {
                            "text_id": text_id,
                            "prompt_text": entry.get("prompt_text", ""),
                            "prompt_type": entry.get("prompt_type"),
                            "soft_weight": float(entry.get("soft_weight", 1.0)),
                            "tags": entry.get("tags"),
                        }
                    )

        if not text_id_to_idx:
            # Fallback: create a single blank entry
            text_id_to_idx["__blank__"] = 0
            input_ids_list.append(payloads[0]["positive_entries"][0]["encoding"]["input_ids"].clone())
            attention_mask_list.append(payloads[0]["positive_entries"][0]["encoding"]["attention_mask"].clone())
            text_meta.append(
                {
                    "text_id": "__blank__",
                    "prompt_text": "",
                    "prompt_type": None,
                    "soft_weight": 1.0,
                    "tags": None,
                }
            )

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        B = len(videos)
        M = input_ids.size(0)
        positive_mask = torch.zeros(B, M, dtype=torch.float32)
        positive_weights = torch.zeros(B, M, dtype=torch.float32)

        for vid_idx, payload in enumerate(payloads):
            for entry in payload.get("positive_entries", []):
                text_id = str(entry.get("text_id"))
                if text_id is None:
                    continue
                col = text_id_to_idx.get(text_id)
                if col is None:
                    continue
                positive_mask[vid_idx, col] = 1.0
                positive_weights[vid_idx, col] = float(entry.get("soft_weight", 1.0))

        reports = []
        for payload, fallback in zip(payloads, raw_texts):
            positives = payload.get("positive_entries", [])
            if positives:
                reports.append(positives[0].get("prompt_text", fallback))
            else:
                reports.append(fallback)

        return {
            "videos": videos_tensor,
            "encoded_texts": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "positive_mask": positive_mask,
            "positive_weights": positive_weights,
            "text_ids": [tid if tid != "__blank__" else None for tid in text_id_to_idx.keys()],
            "text_metadata": text_meta,
            "paths": list(paths),
            "reports": reports,
        }

    if isinstance(videos[0], np.ndarray) and videos[0].ndim == 5:
        videos_tensor = torch.from_numpy(np.stack(videos, axis=0)).float()
    else:
        videos_tensor = torch.stack([torch.from_numpy(v).float() for v in videos])

    encoded_payloads = payloads
    if (
        encoded_payloads[0] is not None
        and isinstance(encoded_payloads[0], dict)
        and "input_ids" in encoded_payloads[0]
    ):
        combined_texts = {
            "input_ids": torch.stack([text["input_ids"] for text in encoded_payloads]),
            "attention_mask": torch.stack([text["attention_mask"] for text in encoded_payloads]),
        }
    else:
        combined_texts = None

    return {
        "videos": videos_tensor,
        "encoded_texts": combined_texts,
        "paths": list(paths),
        "reports": list(raw_texts),
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
    # Determine if this is validation/test split for deterministic behavior
    is_validation = split in ['val', 'validation', 'test', 'inference']
    
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
        rand_augment=False if is_validation else getattr(config, 'rand_augment', False),  # NEVER augment validation
        stride=getattr(config, 'stride', 1),
        groupby_column=getattr(config, 'groupby_column', None),
        num_videos=getattr(config, 'num_videos', 4),
        shuffle_videos=False if is_validation else getattr(config, 'shuffle_videos', False),  # NEVER shuffle videos in validation
        seed=getattr(config, 'seed', None),
        multi_video=getattr(config, 'multi_video', False),
        video_transforms=None if is_validation else getattr(config, 'video_transforms', None),  # No transforms for validation
        resize=getattr(config, 'resize', 224),
        max_length=getattr(config, 'max_length', 250),
        siglip_texts_path=getattr(config, 'siglip_texts_path', None),
        siglip_edges_path=getattr(config, 'siglip_edges_path', None),
        siglip_video_id_column=getattr(config, 'siglip_video_id_column', 'video_id'),
        siglip_text_id_column=getattr(config, 'siglip_text_id_column', 'text_id'),
        siglip_prompt_text_column=getattr(config, 'siglip_prompt_text_column', 'prompt_text'),
        siglip_prompt_type_column=getattr(config, 'siglip_prompt_type_column', 'prompt_type'),
        siglip_soft_weight_column=getattr(config, 'siglip_soft_weight_column', 'soft_weight'),
        siglip_edge_weight_column=getattr(config, 'siglip_edge_weight_column', 'weight'),
        siglip_negatives_per_video=getattr(config, 'siglip_negatives_per_video', 0),
        siglip_pos_samples_per_video=getattr(config, 'siglip_pos_samples_per_video', 1),
        siglip_round_robin_sampling=getattr(config, 'siglip_round_robin_sampling', False),
    )
    # Create a sampler for distributed training
    sampler = DistributedUtils.DS.DistributedSampler(
        video_dataset, 
        shuffle=shuffle, 
        num_replicas=num_replicas, 
        rank=rank
    )
    # Use the same batch size for all splits
    batch_size = getattr(config, 'batch_size', 1)
    
    # Use deterministic worker init for validation
    if is_validation:
        def deterministic_worker_init(worker_id):
            """Fixed seed for validation workers to ensure determinism."""
            fixed_seed = 42 + worker_id  # Fixed seed per worker
            np.random.seed(fixed_seed)
            random.seed(fixed_seed)
            torch.manual_seed(fixed_seed)
        worker_init = deterministic_worker_init
    else:
        worker_init = seed_worker
    
    # Create the dataloader with optimizations
    return DataLoader(
        video_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_init,
        persistent_workers=getattr(config, 'persistent_workers', False) and getattr(config, 'num_workers', 0) > 0,
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if getattr(config, 'num_workers', 0) > 0 else None,
    )
