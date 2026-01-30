"""
Validation Logger for SigLIP Training.

Provides utilities for:
1. Logging generated text vs ground truth (first batch each epoch)
2. Saving detailed retrieval results to CSV
3. Saving recall metrics per sample and summary
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn.functional as F


def log_text_comparison_samples(
    video_ids: List[str],
    ground_truth_texts: List[str],
    predicted_texts: List[str],
    similarities: List[float],
    epoch: int,
    batch_idx: int = 0,
    output_dir: Optional[str] = None,
    max_samples: int = 5,
    print_to_console: bool = True,
) -> str:
    """
    Log comparison between ground truth and predicted texts for debugging.

    Args:
        video_ids: List of video identifiers
        ground_truth_texts: List of ground truth texts
        predicted_texts: List of predicted (top-1) texts
        similarities: List of similarity scores for top-1 predictions
        epoch: Current epoch number
        batch_idx: Batch index (0 = first batch)
        output_dir: Directory to save log file
        max_samples: Maximum samples to display
        print_to_console: Whether to print to console

    Returns:
        Formatted string of comparisons
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"TEXT COMPARISON - Epoch {epoch}, Batch {batch_idx}")
    lines.append(f"(Samples selected from diverse exams and ground truth texts)")
    lines.append(f"{'='*80}")

    n_samples = min(len(video_ids), max_samples)

    # Track unique predictions for diagnostic
    unique_predictions = set()

    for i in range(n_samples):
        vid = video_ids[i] if i < len(video_ids) else "N/A"
        gt = ground_truth_texts[i] if i < len(ground_truth_texts) else "N/A"
        pred = predicted_texts[i] if i < len(predicted_texts) else "N/A"
        sim = similarities[i] if i < len(similarities) else 0.0

        unique_predictions.add(pred.strip())

        # Extract StudyInstanceUID from video path for clarity
        study_uid = "N/A"
        if vid and vid != "N/A":
            basename = os.path.basename(vid)
            if "_" in basename:
                study_uid = basename.split("_")[0][-12:]  # Last 12 chars for brevity

        # Truncate long texts for display
        gt_display = gt[:200] + "..." if len(gt) > 200 else gt
        pred_display = pred[:200] + "..." if len(pred) > 200 else pred

        match = "✓ MATCH" if gt.strip() == pred.strip() else "✗ MISMATCH"

        lines.append(f"\n--- Sample {i+1} (Exam: ...{study_uid}) [{match}] ---")
        lines.append(f"Video: {os.path.basename(vid) if vid else 'N/A'}")
        lines.append(f"GT:   {gt_display}")
        lines.append(f"PRED: {pred_display}")
        lines.append(f"Similarity: {sim:.4f}")

    # Add diagnostic summary
    lines.append(f"\n{'-'*80}")
    lines.append(f"DIAGNOSTIC SUMMARY:")
    lines.append(f"  Unique predictions: {len(unique_predictions)}/{n_samples}")
    if len(unique_predictions) == 1 and n_samples > 1:
        lines.append(f"  ⚠️  WARNING: All samples predicted the SAME text!")
        lines.append(f"     This may indicate model collapse or poor video feature diversity.")
    elif len(unique_predictions) < n_samples // 2:
        lines.append(f"  ⚠️  WARNING: Low prediction diversity ({len(unique_predictions)} unique out of {n_samples})")

    lines.append(f"{'='*80}\n")

    output = "\n".join(lines)

    if print_to_console:
        print(output)

    # Save to file if output_dir provided
    if output_dir:
        log_file = os.path.join(output_dir, f"text_comparisons_epoch{epoch}.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(output)

    return output


def save_detailed_retrieval_csv(
    similarity_matrix: torch.Tensor,
    video_ids: List[str],
    unique_texts: List[str],
    ground_truth_indices: torch.Tensor,
    epoch: int,
    output_dir: str,
    recall_k_values: List[int] = [1, 5, 10],
    top_k_save: int = 5,
    ground_truth_matrix: Optional[torch.Tensor] = None,
    text_ids: Optional[List[str]] = None,
) -> str:
    """
    Save detailed retrieval results to CSV with per-sample recall.

    Args:
        similarity_matrix: [N_videos, N_texts] similarity scores
        video_ids: List of video identifiers
        unique_texts: List of unique text descriptions
        ground_truth_indices: [N_videos] indices of ground truth texts
        epoch: Current epoch
        output_dir: Directory to save CSV
        recall_k_values: List of k values for recall computation
        top_k_save: Number of top predictions to save per sample
        ground_truth_matrix: Optional [N_videos, N_texts] multi-positive mask
        text_ids: Optional list of text IDs

    Returns:
        Path to saved CSV file
    """
    csv_path = os.path.join(output_dir, f"retrieval_detailed_epoch{epoch}.csv")

    n_videos = similarity_matrix.shape[0]
    n_texts = similarity_matrix.shape[1]

    # Get top-k predictions for each video
    top_k = min(top_k_save, n_texts)
    topk_sims, topk_indices = torch.topk(similarity_matrix, top_k, dim=1)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Build header
        header = ["video_id", "ground_truth_text", "ground_truth_idx"]

        # Add recall columns
        for k in recall_k_values:
            header.append(f"recall@{k}_hit")

        # Add top-k prediction columns
        for rank in range(1, top_k + 1):
            header.append(f"pred_{rank}_text")
            header.append(f"pred_{rank}_sim")
            if text_ids:
                header.append(f"pred_{rank}_text_id")

        writer.writerow(header)

        # Write data for each video
        for i in range(n_videos):
            vid = video_ids[i] if i < len(video_ids) else f"video_{i}"
            gt_idx = ground_truth_indices[i].item() if isinstance(ground_truth_indices, torch.Tensor) else ground_truth_indices[i]
            gt_text = unique_texts[gt_idx] if gt_idx < len(unique_texts) else "N/A"

            row = [vid, gt_text, gt_idx]

            # Compute recall hits for this sample
            pred_indices = topk_indices[i].tolist()

            for k in recall_k_values:
                k_use = min(k, top_k)
                if ground_truth_matrix is not None:
                    # Multi-positive: check if any positive in top-k
                    gt_positives = ground_truth_matrix[i]
                    hit = any(gt_positives[pred_indices[j]] > 0.5 for j in range(k_use))
                else:
                    # Single-positive: check if gt_idx in top-k
                    hit = gt_idx in pred_indices[:k_use]
                row.append(1 if hit else 0)

            # Add top-k predictions
            for j in range(top_k):
                pred_idx = pred_indices[j]
                pred_text = unique_texts[pred_idx] if pred_idx < len(unique_texts) else "N/A"
                pred_sim = topk_sims[i, j].item()

                row.append(pred_text[:500])  # Truncate long texts
                row.append(f"{pred_sim:.4f}")
                if text_ids:
                    row.append(text_ids[pred_idx] if pred_idx < len(text_ids) else "N/A")

            writer.writerow(row)

    print(f"[ValidationLogger] Saved detailed retrieval CSV: {csv_path}")
    return csv_path


def save_recall_summary_csv(
    recall_metrics: Dict[str, float],
    epoch: int,
    output_dir: str,
    additional_metrics: Optional[Dict[str, float]] = None,
    append: bool = True,
) -> str:
    """
    Save or append recall metrics summary to CSV.

    Args:
        recall_metrics: Dictionary of recall metrics (e.g., {"Recall@1": 0.5, "Recall@5": 0.8})
        epoch: Current epoch
        output_dir: Directory to save CSV
        additional_metrics: Optional additional metrics to include
        append: Whether to append to existing file

    Returns:
        Path to saved CSV file
    """
    csv_path = os.path.join(output_dir, "recall_summary.csv")

    # Combine all metrics
    all_metrics = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
    all_metrics.update(recall_metrics)
    if additional_metrics:
        all_metrics.update(additional_metrics)

    # Check if file exists and get existing headers
    file_exists = os.path.exists(csv_path) and append
    existing_headers = []

    if file_exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, [])

    # Merge headers
    if existing_headers:
        headers = existing_headers
        for key in all_metrics.keys():
            if key not in headers:
                headers.append(key)
    else:
        headers = list(all_metrics.keys())

    # Write data
    mode = "a" if file_exists else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')

        if not file_exists:
            writer.writeheader()

        writer.writerow(all_metrics)

    print(f"[ValidationLogger] Saved recall summary: {csv_path}")
    return csv_path


def _extract_study_uid(video_path: str) -> str:
    """
    Extract StudyInstanceUID from video path.

    Video paths typically look like:
    .../1.2.392.200036.9116.1467.20170720125102258.4.2_1.2.392...dcm.avi

    The StudyInstanceUID is the part before the underscore.
    """
    import os
    basename = os.path.basename(video_path)
    # Split by underscore - StudyInstanceUID is before first underscore
    if "_" in basename:
        return basename.split("_")[0]
    return basename


def _sample_diverse_indices(
    video_ids: List[str],
    ground_truth_indices: torch.Tensor,
    max_samples: int,
) -> List[int]:
    """
    Sample indices that maximize diversity in both exams (StudyInstanceUID) and ground truth texts.

    Strategy:
    1. First, try to get samples from different exams
    2. Within that constraint, try to get samples with different ground truth texts

    Returns:
        List of indices into the video_ids list
    """
    if len(video_ids) == 0:
        return []

    n_total = len(video_ids)

    # Extract study UIDs and GT indices
    study_uids = [_extract_study_uid(vid) for vid in video_ids]
    gt_indices = [
        ground_truth_indices[i].item() if isinstance(ground_truth_indices, torch.Tensor) else ground_truth_indices[i]
        for i in range(min(len(ground_truth_indices), n_total))
    ]

    # Build a map: study_uid -> list of (idx, gt_idx)
    study_to_samples: Dict[str, List[tuple]] = {}
    for i, (study_uid, gt_idx) in enumerate(zip(study_uids, gt_indices)):
        if study_uid not in study_to_samples:
            study_to_samples[study_uid] = []
        study_to_samples[study_uid].append((i, gt_idx))

    selected_indices = []
    seen_gt_indices = set()
    seen_study_uids = set()

    # First pass: select one sample per unique study UID, prioritizing unique GT texts
    for study_uid, samples in study_to_samples.items():
        if len(selected_indices) >= max_samples:
            break

        # Sort samples by GT index to get variety
        # Prefer samples with GT indices we haven't seen yet
        unseen_gt = [(idx, gt) for idx, gt in samples if gt not in seen_gt_indices]

        if unseen_gt:
            # Pick the first sample with unseen GT
            chosen_idx, chosen_gt = unseen_gt[0]
        else:
            # All GT indices seen, just pick the first sample from this study
            chosen_idx, chosen_gt = samples[0]

        selected_indices.append(chosen_idx)
        seen_gt_indices.add(chosen_gt)
        seen_study_uids.add(study_uid)

    # Second pass: if we still need more samples, add more from seen studies
    # but prioritize unseen GT indices
    if len(selected_indices) < max_samples:
        all_candidates = []
        for study_uid, samples in study_to_samples.items():
            for idx, gt in samples:
                if idx not in selected_indices:
                    all_candidates.append((idx, gt, gt not in seen_gt_indices))

        # Sort: prioritize unseen GT indices
        all_candidates.sort(key=lambda x: (not x[2], x[1]))

        for idx, gt, _ in all_candidates:
            if len(selected_indices) >= max_samples:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)
                seen_gt_indices.add(gt)

    return selected_indices


def log_first_batch_comparisons(
    video_features: torch.Tensor,
    text_features: torch.Tensor,
    video_ids: List[str],
    unique_texts: List[str],
    ground_truth_indices: torch.Tensor,
    epoch: int,
    output_dir: str,
    max_samples: int = 5,
) -> None:
    """
    Log text comparisons for first batch of validation.

    Samples are selected to maximize diversity:
    - Different exams (StudyInstanceUID) are prioritized
    - Different ground truth texts are prioritized

    Args:
        video_features: [B, D] video embeddings
        text_features: [N_texts, D] text embeddings
        video_ids: List of video identifiers
        unique_texts: List of unique text descriptions
        ground_truth_indices: [B] ground truth text indices
        epoch: Current epoch
        output_dir: Output directory
        max_samples: Max samples to log
    """
    # Compute similarities
    video_norm = F.normalize(video_features.float(), dim=-1)
    text_norm = F.normalize(text_features.float(), dim=-1)
    similarities = torch.matmul(video_norm, text_norm.t())

    # Get top-1 predictions
    top_sims, top_indices = similarities.max(dim=1)

    # Get diverse sample indices (different exams, different GT texts)
    sample_indices = _sample_diverse_indices(
        video_ids=video_ids,
        ground_truth_indices=ground_truth_indices,
        max_samples=max_samples,
    )

    if not sample_indices:
        print("[ValidationLogger] No samples to log")
        return

    # Prepare data for selected samples
    gt_texts = []
    pred_texts = []
    sims = []
    selected_video_ids = []
    selected_video_feats = []

    for i in sample_indices:
        gt_idx = ground_truth_indices[i].item() if isinstance(ground_truth_indices, torch.Tensor) else ground_truth_indices[i]
        pred_idx = top_indices[i].item()

        selected_video_ids.append(video_ids[i])
        gt_texts.append(unique_texts[gt_idx] if gt_idx < len(unique_texts) else "N/A")
        pred_texts.append(unique_texts[pred_idx] if pred_idx < len(unique_texts) else "N/A")
        sims.append(top_sims[i].item())
        selected_video_feats.append(video_norm[i])

    # Log comparisons
    log_text_comparison_samples(
        video_ids=selected_video_ids,
        ground_truth_texts=gt_texts,
        predicted_texts=pred_texts,
        similarities=sims,
        epoch=epoch,
        batch_idx=0,
        output_dir=output_dir,
        max_samples=max_samples,
        print_to_console=True,
    )

    # Log video feature diversity diagnostic
    if len(selected_video_feats) > 1:
        _log_video_feature_diversity(
            selected_video_feats=selected_video_feats,
            selected_video_ids=selected_video_ids,
            epoch=epoch,
            output_dir=output_dir,
        )


def _log_video_feature_diversity(
    selected_video_feats: List[torch.Tensor],
    selected_video_ids: List[str],
    epoch: int,
    output_dir: str,
) -> None:
    """Log diagnostic info about video feature diversity."""
    feats_stack = torch.stack(selected_video_feats)  # [N, D]

    # Compute pairwise cosine similarities between selected video features
    video_video_sim = torch.matmul(feats_stack, feats_stack.t())  # [N, N]

    lines = []
    lines.append(f"\n{'-'*80}")
    lines.append(f"VIDEO FEATURE DIVERSITY (Epoch {epoch}):")

    n = len(selected_video_feats)
    # Get off-diagonal similarities
    mask = ~torch.eye(n, dtype=torch.bool, device=video_video_sim.device)
    off_diag_sims = video_video_sim[mask]

    if off_diag_sims.numel() > 0:
        mean_sim = off_diag_sims.mean().item()
        min_sim = off_diag_sims.min().item()
        max_sim = off_diag_sims.max().item()

        lines.append(f"  Pairwise video-video cosine similarity:")
        lines.append(f"    Mean: {mean_sim:.4f}, Min: {min_sim:.4f}, Max: {max_sim:.4f}")

        if mean_sim > 0.95:
            lines.append(f"  ⚠️  WARNING: Video features are VERY similar (mean={mean_sim:.4f})")
            lines.append(f"     This indicates potential feature collapse in the video encoder!")
        elif mean_sim > 0.85:
            lines.append(f"  ⚠️  CAUTION: Video features have low diversity (mean={mean_sim:.4f})")

        # Show pairwise matrix for small N
        if n <= 5:
            lines.append(f"  Pairwise similarity matrix:")
            for i in range(n):
                row = [f"{video_video_sim[i, j].item():.3f}" for j in range(n)]
                lines.append(f"    [{', '.join(row)}]")

    lines.append(f"{'-'*80}\n")

    output = "\n".join(lines)
    print(output)

    # Append to file
    if output_dir:
        log_file = os.path.join(output_dir, f"text_comparisons_epoch{epoch}.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(output)


class ValidationLogger:
    """
    Centralized validation logger for SigLIP training.

    Tracks and saves:
    - Per-sample retrieval results
    - Recall metrics over epochs
    - Text comparisons for debugging
    """

    def __init__(self, output_dir: str, recall_k_values: List[int] = [1, 5, 10]):
        """
        Initialize validation logger.

        Args:
            output_dir: Directory to save all logs
            recall_k_values: K values for recall computation
        """
        self.output_dir = output_dir
        self.recall_k_values = recall_k_values
        self.epoch_metrics_history: List[Dict[str, Any]] = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def log_epoch_validation(
        self,
        similarity_matrix: torch.Tensor,
        video_ids: List[str],
        unique_texts: List[str],
        ground_truth_indices: torch.Tensor,
        epoch: int,
        recall_metrics: Dict[str, float],
        video_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        ground_truth_matrix: Optional[torch.Tensor] = None,
        text_ids: Optional[List[str]] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
        log_text_samples: bool = True,
        save_detailed_csv: bool = True,
    ) -> None:
        """
        Log all validation results for an epoch.

        Args:
            similarity_matrix: [N_videos, N_texts] similarity scores
            video_ids: List of video identifiers
            unique_texts: List of unique text descriptions
            ground_truth_indices: [N_videos] ground truth text indices
            epoch: Current epoch
            recall_metrics: Dictionary of recall metrics
            video_features: Optional video features for text comparison
            text_features: Optional text features for text comparison
            ground_truth_matrix: Optional multi-positive ground truth mask
            text_ids: Optional text IDs
            additional_metrics: Optional additional metrics
            log_text_samples: Whether to log text comparison samples
            save_detailed_csv: Whether to save detailed CSV
        """
        # Save recall summary
        save_recall_summary_csv(
            recall_metrics=recall_metrics,
            epoch=epoch,
            output_dir=self.output_dir,
            additional_metrics=additional_metrics,
            append=True,
        )

        # Save detailed retrieval CSV
        if save_detailed_csv:
            save_detailed_retrieval_csv(
                similarity_matrix=similarity_matrix,
                video_ids=video_ids,
                unique_texts=unique_texts,
                ground_truth_indices=ground_truth_indices,
                epoch=epoch,
                output_dir=self.output_dir,
                recall_k_values=self.recall_k_values,
                ground_truth_matrix=ground_truth_matrix,
                text_ids=text_ids,
            )

        # Log text comparison samples
        if log_text_samples and video_features is not None and text_features is not None:
            log_first_batch_comparisons(
                video_features=video_features,
                text_features=text_features,
                video_ids=video_ids,
                unique_texts=unique_texts,
                ground_truth_indices=ground_truth_indices,
                epoch=epoch,
                output_dir=self.output_dir,
            )

        # Store metrics history
        metrics_entry = {
            "epoch": epoch,
            **recall_metrics,
            **(additional_metrics or {}),
        }
        self.epoch_metrics_history.append(metrics_entry)

    def get_best_epoch(self, metric_name: str = "Recall@1") -> Dict[str, Any]:
        """Get the best epoch based on a metric."""
        if not self.epoch_metrics_history:
            return {}

        best = max(self.epoch_metrics_history, key=lambda x: x.get(metric_name, 0))
        return best
