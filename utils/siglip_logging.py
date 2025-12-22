"""
Enhanced SigLIP logging utilities for multi-positive contrastive learning.

Provides detailed logging similar to ECG-style reports with:
- All positive text IDs and their texts for each video
- Top-K predictions with probabilities and text IDs
- Per-sample alignment scores (row positive/negative mass)
- Generated reports from LocCa decoder
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import wandb

from dataloaders.video_clip_dataset import VideoClipDataset


@dataclass
class SiglipRetrievalRecord:
    """
    Detailed retrieval record for a single video sample.
    Mirrors the ECG logging format for comprehensive analysis.
    """
    # Identifier
    video_id: str
    video_path: str
    study_id: Optional[str] = None

    # Ground truth (all positives for multi-positive SigLIP)
    ground_truth_text_ids: List[str] = field(default_factory=list)
    ground_truth_texts: List[str] = field(default_factory=list)
    ground_truth_severities: List[str] = field(default_factory=list)
    ground_truth_segments: List[str] = field(default_factory=list)

    # Concatenated ground truth report (for generation comparison)
    ground_truth_report: str = ""

    # Generated report (from LocCa decoder if available)
    generated_report: Optional[str] = None

    # Alignment metrics
    alignment_score: float = 0.0  # Mean similarity to all positives
    row_pos_mass: float = 0.0  # Sum of probabilities on positive pairs
    row_neg_mass: float = 0.0  # Sum of probabilities on negative pairs
    max_pos_similarity: float = 0.0  # Max similarity to any positive
    min_pos_similarity: float = 0.0  # Min similarity to any positive

    # Top predictions
    top_pred_text_ids: List[str] = field(default_factory=list)
    top_pred_texts: List[str] = field(default_factory=list)
    top_pred_probs: List[float] = field(default_factory=list)
    top_pred_segments: List[str] = field(default_factory=list)
    top_pred_severities: List[str] = field(default_factory=list)

    # Recall metrics for this sample
    hit_at_1: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    first_positive_rank: int = -1  # Rank of first positive in predictions


def compute_sample_alignment_metrics(
    similarity_row: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute detailed alignment metrics for a single video sample.

    Args:
        similarity_row: [num_texts] similarity scores to all texts
        positive_mask: [num_texts] binary mask (1 for positives, 0 for negatives)
        temperature: Temperature for softmax normalization

    Returns:
        Dictionary with alignment metrics
    """
    # Get positive and negative similarities
    pos_indices = torch.where(positive_mask > 0)[0]
    neg_indices = torch.where(positive_mask == 0)[0]

    pos_sims = similarity_row[pos_indices] if len(pos_indices) > 0 else torch.tensor([0.0])
    neg_sims = similarity_row[neg_indices] if len(neg_indices) > 0 else torch.tensor([0.0])

    # Compute softmax probabilities
    probs = F.softmax(similarity_row / temperature, dim=0)

    pos_mass = probs[pos_indices].sum().item() if len(pos_indices) > 0 else 0.0
    neg_mass = probs[neg_indices].sum().item() if len(neg_indices) > 0 else 0.0

    return {
        "alignment_score": pos_sims.mean().item() if len(pos_sims) > 0 else 0.0,
        "row_pos_mass": pos_mass,
        "row_neg_mass": neg_mass,
        "max_pos_similarity": pos_sims.max().item() if len(pos_sims) > 0 else 0.0,
        "min_pos_similarity": pos_sims.min().item() if len(pos_sims) > 0 else 0.0,
    }


def build_siglip_retrieval_records(
    similarity_matrix: torch.Tensor,
    ground_truth_matrix: torch.Tensor,
    all_video_ids: List[str],
    all_video_paths: List[str],
    all_text_ids: List[str],
    text_lookup: Dict[str, Dict[str, Any]],
    dataset: VideoClipDataset,
    top_k: int = 10,
    temperature: float = 1.0,
    generated_reports: Optional[List[str]] = None,
) -> List[SiglipRetrievalRecord]:
    """
    Build comprehensive retrieval records for all samples.

    Args:
        similarity_matrix: [num_videos, num_texts] similarity scores
        ground_truth_matrix: [num_videos, num_texts] binary ground truth (multi-positive)
        all_video_ids: List of video IDs
        all_video_paths: List of video paths
        all_text_ids: List of text IDs corresponding to columns
        text_lookup: Dictionary mapping text_id -> metadata
        dataset: VideoClipDataset instance
        top_k: Number of top predictions to include
        temperature: Temperature for probability computation
        generated_reports: Optional list of generated reports from LocCa

    Returns:
        List of SiglipRetrievalRecord objects
    """
    num_videos = similarity_matrix.shape[0]
    records = []

    for video_idx in range(num_videos):
        video_id = all_video_ids[video_idx] if video_idx < len(all_video_ids) else f"video_{video_idx}"
        video_path = all_video_paths[video_idx] if video_idx < len(all_video_paths) else ""

        # Get ground truth positive indices
        pos_indices = torch.where(ground_truth_matrix[video_idx] > 0)[0]

        # Collect all ground truth texts
        gt_text_ids = []
        gt_texts = []
        gt_severities = []
        gt_segments = []

        for pos_idx in pos_indices:
            text_id = all_text_ids[pos_idx.item()]
            gt_text_ids.append(text_id)

            meta = text_lookup.get(text_id, {})
            gt_texts.append(meta.get("prompt_text", text_id))
            gt_severities.append(meta.get("disease_severity", "unknown"))
            gt_segments.append(meta.get("segment", "unknown"))

        # Build concatenated report from all positives
        gt_report = " ".join(gt_texts) if gt_texts else "No findings."

        # Compute alignment metrics
        alignment = compute_sample_alignment_metrics(
            similarity_matrix[video_idx],
            ground_truth_matrix[video_idx],
            temperature,
        )

        # Get top-K predictions
        sim_row = similarity_matrix[video_idx]
        top_k_actual = min(top_k, len(sim_row))
        top_values, top_indices = torch.topk(sim_row, k=top_k_actual)

        top_text_ids = []
        top_texts = []
        top_probs = []
        top_segments = []
        top_severities = []

        # Compute softmax probabilities
        probs = F.softmax(sim_row / temperature, dim=0)

        for rank, (val, idx) in enumerate(zip(top_values, top_indices)):
            text_id = all_text_ids[idx.item()]
            top_text_ids.append(text_id)

            meta = text_lookup.get(text_id, {})
            top_texts.append(meta.get("prompt_text", text_id))
            top_probs.append(probs[idx].item())
            top_segments.append(meta.get("segment", "unknown"))
            top_severities.append(meta.get("disease_severity", "unknown"))

        # Compute per-sample recall metrics
        top_set_1 = set(top_text_ids[:1])
        top_set_5 = set(top_text_ids[:5])
        top_set_10 = set(top_text_ids[:10])
        gt_set = set(gt_text_ids)

        hit_at_1 = bool(top_set_1 & gt_set)
        hit_at_5 = bool(top_set_5 & gt_set)
        hit_at_10 = bool(top_set_10 & gt_set)

        # Find first positive rank
        first_pos_rank = -1
        for rank, tid in enumerate(top_text_ids):
            if tid in gt_set:
                first_pos_rank = rank + 1
                break

        record = SiglipRetrievalRecord(
            video_id=video_id,
            video_path=video_path,
            study_id=getattr(dataset, "groupby_column", None),
            ground_truth_text_ids=gt_text_ids,
            ground_truth_texts=gt_texts,
            ground_truth_severities=gt_severities,
            ground_truth_segments=gt_segments,
            ground_truth_report=gt_report,
            generated_report=generated_reports[video_idx] if generated_reports else None,
            alignment_score=alignment["alignment_score"],
            row_pos_mass=alignment["row_pos_mass"],
            row_neg_mass=alignment["row_neg_mass"],
            max_pos_similarity=alignment["max_pos_similarity"],
            min_pos_similarity=alignment["min_pos_similarity"],
            top_pred_text_ids=top_text_ids,
            top_pred_texts=top_texts,
            top_pred_probs=top_probs,
            top_pred_segments=top_segments,
            top_pred_severities=top_severities,
            hit_at_1=hit_at_1,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            first_positive_rank=first_pos_rank,
        )
        records.append(record)

    return records


def log_siglip_qualitative_to_wandb(
    records: List[SiglipRetrievalRecord],
    epoch: int,
    wandb_wrapper: Any,
    num_best: int = 3,
    num_worst: int = 3,
) -> None:
    """
    Log qualitative examples to wandb with full multi-positive detail.

    Args:
        records: List of SiglipRetrievalRecord objects
        epoch: Current epoch number
        wandb_wrapper: Wandb wrapper instance
        num_best: Number of best examples to log
        num_worst: Number of worst examples to log
    """
    if not wandb_wrapper.is_initialized() or not records:
        return

    # Sort by alignment score
    sorted_records = sorted(records, key=lambda r: r.alignment_score, reverse=True)

    best_records = sorted_records[:num_best]
    worst_records = sorted_records[-num_worst:]

    def format_record_html(record: SiglipRetrievalRecord, is_best: bool) -> str:
        """Format a single record as HTML for wandb."""
        prefix = "Good" if is_best else "Bad"

        # Format all ground truth texts (multi-positive)
        gt_lines = []
        for i, (tid, text, sev, seg) in enumerate(zip(
            record.ground_truth_text_ids,
            record.ground_truth_texts,
            record.ground_truth_severities,
            record.ground_truth_segments,
        )):
            gt_lines.append(f"  {i+1}. [{tid}] [{seg}] [{sev}] {text}")
        gt_html = "<br>".join(gt_lines) if gt_lines else "  None"

        # Format top predictions
        pred_lines = []
        for i, (tid, text, prob, seg, sev) in enumerate(zip(
            record.top_pred_text_ids[:5],
            record.top_pred_texts[:5],
            record.top_pred_probs[:5],
            record.top_pred_segments[:5],
            record.top_pred_severities[:5],
        )):
            is_hit = tid in record.ground_truth_text_ids
            marker = " [HIT]" if is_hit else ""
            pred_lines.append(f"  {i+1}. [{tid}] [{seg}] p={prob:.3f}{marker} {text}")
        pred_html = "<br>".join(pred_lines)

        html = f"""
        <div style="font-family: monospace; font-size: 12px;">
        <b>{prefix} Retrieval - {record.video_id}</b><br>
        <br>
        <b>Alignment Metrics:</b><br>
        &nbsp;&nbsp;Score: {record.alignment_score:.4f}<br>
        &nbsp;&nbsp;Pos Mass: {record.row_pos_mass:.4f}<br>
        &nbsp;&nbsp;Neg Mass: {record.row_neg_mass:.4f}<br>
        &nbsp;&nbsp;First Pos Rank: {record.first_positive_rank}<br>
        &nbsp;&nbsp;Hit@1: {record.hit_at_1}, Hit@5: {record.hit_at_5}<br>
        <br>
        <b>Ground Truth ({len(record.ground_truth_text_ids)} positives):</b><br>
        {gt_html}<br>
        <br>
        <b>Concatenated Report:</b><br>
        &nbsp;&nbsp;{record.ground_truth_report[:500]}...<br>
        <br>
        <b>Top 5 Predictions:</b><br>
        {pred_html}
        </div>
        """

        if record.generated_report:
            html += f"""
            <br>
            <b>Generated Report (LocCa):</b><br>
            &nbsp;&nbsp;{record.generated_report[:500]}...
            """

        return html

    # Log best examples
    for i, record in enumerate(best_records):
        html = format_record_html(record, is_best=True)
        wandb_wrapper.log({
            f"qualitative/good_retrieval_text_{i}": wandb.Html(html),
            f"qualitative/good_alignment_{i}": record.alignment_score,
            f"qualitative/good_pos_mass_{i}": record.row_pos_mass,
            "epoch": epoch,
        })

    # Log worst examples
    for i, record in enumerate(worst_records):
        html = format_record_html(record, is_best=False)
        wandb_wrapper.log({
            f"qualitative/bad_retrieval_text_{i}": wandb.Html(html),
            f"qualitative/bad_alignment_{i}": record.alignment_score,
            f"qualitative/bad_pos_mass_{i}": record.row_pos_mass,
            "epoch": epoch,
        })


def save_siglip_detailed_csv(
    records: List[SiglipRetrievalRecord],
    output_path: str,
) -> None:
    """
    Save detailed retrieval results to CSV in ECG-style format.

    Args:
        records: List of SiglipRetrievalRecord objects
        output_path: Path to output CSV file
    """
    fieldnames = [
        "video_id",
        "video_path",
        "ground_truth_report",
        "generated_report",
        "alignment_score",
        "row_pos_mass",
        "row_neg_mass",
        "max_pos_similarity",
        "min_pos_similarity",
        "hit_at_1",
        "hit_at_5",
        "hit_at_10",
        "first_positive_rank",
        "num_positives",
        "ground_truth_text_ids",
        "ground_truth_texts",
        "ground_truth_severities",
        "ground_truth_segments",
        "top_pred_text_ids",
        "top_pred_texts",
        "top_pred_probs",
        "top_pred_segments",
        "top_pred_severities",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = {
                "video_id": record.video_id,
                "video_path": record.video_path,
                "ground_truth_report": record.ground_truth_report,
                "generated_report": record.generated_report or "",
                "alignment_score": f"{record.alignment_score:.6f}",
                "row_pos_mass": f"{record.row_pos_mass:.6f}",
                "row_neg_mass": f"{record.row_neg_mass:.6f}",
                "max_pos_similarity": f"{record.max_pos_similarity:.6f}",
                "min_pos_similarity": f"{record.min_pos_similarity:.6f}",
                "hit_at_1": record.hit_at_1,
                "hit_at_5": record.hit_at_5,
                "hit_at_10": record.hit_at_10,
                "first_positive_rank": record.first_positive_rank,
                "num_positives": len(record.ground_truth_text_ids),
                "ground_truth_text_ids": "|".join(record.ground_truth_text_ids),
                "ground_truth_texts": "|".join(record.ground_truth_texts),
                "ground_truth_severities": "|".join(record.ground_truth_severities),
                "ground_truth_segments": "|".join(record.ground_truth_segments),
                "top_pred_text_ids": "|".join(record.top_pred_text_ids),
                "top_pred_texts": "|".join(record.top_pred_texts),
                "top_pred_probs": ", ".join(f"{p:.4f}" for p in record.top_pred_probs),
                "top_pred_segments": "|".join(record.top_pred_segments),
                "top_pred_severities": "|".join(record.top_pred_severities),
            }
            writer.writerow(row)

    print(f"Saved detailed SigLIP retrieval results to {output_path}")


def save_siglip_detailed_json(
    records: List[SiglipRetrievalRecord],
    output_path: str,
) -> None:
    """
    Save detailed retrieval results to JSON for further analysis.

    Args:
        records: List of SiglipRetrievalRecord objects
        output_path: Path to output JSON file
    """
    data = []
    for record in records:
        entry = {
            "video_id": record.video_id,
            "video_path": record.video_path,
            "ground_truth": {
                "text_ids": record.ground_truth_text_ids,
                "texts": record.ground_truth_texts,
                "severities": record.ground_truth_severities,
                "segments": record.ground_truth_segments,
                "concatenated_report": record.ground_truth_report,
            },
            "generated_report": record.generated_report,
            "alignment_metrics": {
                "alignment_score": record.alignment_score,
                "row_pos_mass": record.row_pos_mass,
                "row_neg_mass": record.row_neg_mass,
                "max_pos_similarity": record.max_pos_similarity,
                "min_pos_similarity": record.min_pos_similarity,
            },
            "recall_metrics": {
                "hit_at_1": record.hit_at_1,
                "hit_at_5": record.hit_at_5,
                "hit_at_10": record.hit_at_10,
                "first_positive_rank": record.first_positive_rank,
            },
            "top_predictions": [
                {
                    "rank": i + 1,
                    "text_id": tid,
                    "text": text,
                    "probability": prob,
                    "segment": seg,
                    "severity": sev,
                    "is_hit": tid in record.ground_truth_text_ids,
                }
                for i, (tid, text, prob, seg, sev) in enumerate(zip(
                    record.top_pred_text_ids,
                    record.top_pred_texts,
                    record.top_pred_probs,
                    record.top_pred_segments,
                    record.top_pred_severities,
                ))
            ],
        }
        data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved detailed SigLIP retrieval results to {output_path}")


def build_report_from_positive_text_ids(
    positive_text_ids: List[str],
    text_lookup: Dict[str, Dict[str, Any]],
    separator: str = " ",
    order_by_severity: bool = True,
) -> str:
    """
    Build a concatenated report from multiple positive text IDs.

    This is the TARGET for LocCa generation - the model should learn to
    generate this concatenated report from the video features.

    Args:
        positive_text_ids: List of text IDs that are positive for this video
        text_lookup: Dictionary mapping text_id -> metadata
        separator: Separator between texts
        order_by_severity: Whether to order by severity (severe first)

    Returns:
        Concatenated report string
    """
    if not positive_text_ids:
        return "No findings."

    # Collect texts with metadata
    texts_with_meta = []
    for text_id in positive_text_ids:
        meta = text_lookup.get(text_id, {})
        text = meta.get("prompt_text", text_id)
        severity = meta.get("disease_severity", "normal")
        segment = meta.get("segment", "")

        # Severity ordering
        severity_rank = {
            "severe": 0, "critical": 0, "cto": 0,
            "moderate": 1,
            "mild": 2,
            "normal": 3,
        }.get(severity.lower() if isinstance(severity, str) else "normal", 3)

        texts_with_meta.append({
            "text": text,
            "severity_rank": severity_rank,
            "segment": segment,
        })

    if order_by_severity:
        texts_with_meta.sort(key=lambda x: (x["severity_rank"], x["segment"]))

    return separator.join(t["text"] for t in texts_with_meta)
