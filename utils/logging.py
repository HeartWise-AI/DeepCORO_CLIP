"""Logging utilities for training and evaluation."""

import math
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import csv
import wandb

from typing import List, Dict

class WandbLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(
        self,
        project_name: str = "deepcoro_clip",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "online",
    ):
        """Initialize WandB logger.

        Args:
            project_name: Name of the project on WandB
            experiment_name: Name of this specific run. If None, will be auto-generated
            config: Dictionary of hyperparameters to log
            mode: WandB mode ('online', 'offline', or 'disabled')
        """
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gpu_info = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
            experiment_name = f"train_{timestamp}_gpu{gpu_info}"

        # Initialize wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            mode=mode,
        )

        self.step = 0
        self.epoch = 0
        self.mode = mode

    def log_batch(
        self,
        loss: float,
        batch_idx: int,
        epoch: int,
        learning_rate: float,
        batch_size: int,
        num_batches: int,
        **kwargs,
    ) -> None:
        """Log metrics for a training batch.

        Only logs essential training metrics (loss, lr) for continuous monitoring.
        """
        if self.mode == "disabled":
            return

        self.step += 1
        metrics = {
            "train/loss": loss,
            "train/epoch": epoch,
            "train/batch": batch_idx,
            "train/progress": (batch_idx + 1) / num_batches,
            "train/learning_rate": learning_rate,
            "train/batch_size": batch_size,
            "train/global_step": self.step,
        }

        # Add any additional continuous metrics
        for key, value in kwargs.items():
            if key.startswith("continuous_"):
                metrics[f"train/{key.replace('continuous_', '')}"] = value

        wandb.log(metrics, step=self.step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        learning_rate: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log comprehensive metrics at the end of each epoch for training.

        Args:
            epoch: Current epoch number
            train_loss: Average training loss for the epoch
            learning_rate: Optional current learning rate
        """
    
        if self.mode == "disabled":
            return
        
        # Skip logging epoch=0 so we don't pollute the chart
        if epoch == 0:
            return

        
        self.epoch = epoch
        metrics = {
            "epoch": epoch,
            "epoch/train_loss": train_loss,
        }

        if learning_rate is not None:
            metrics["epoch/learning_rate"] = learning_rate

        # Add any additional epoch-level metrics here
        for key, value in kwargs.items():
            metrics[f"epoch/{key}"] = value

        wandb.log(metrics, step=self.step)

    def log_val_only_metrics(
        self,
        epoch: int,
        val_loss: float,
        recall_metrics: Optional[Dict[str, float]] = None,
        mrr_metrics: Optional[Dict[str, float]] = None,
        video_norm: Optional[float] = None,
        text_norm: Optional[float] = None,
        alignment_score: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log metrics from the 'val_only' scenario.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss (val-only scenario)
            recall_metrics: Dictionary containing Recall@K metrics
            mrr_metrics: Dictionary containing MRR metrics
            video_norm: Average L2 norm of video embeddings (val_only)
            text_norm: Average L2 norm of text embeddings (val_only)
            alignment_score: Average cosine similarity of positive pairs (val_only)

        Recommended additional metrics for retrieval:
        - NDCG@K
        - Median Rank
        - R@10, R@50
        - MAP (Mean Average Precision)

        If computed, log them similarly to recall and MRR below.
        """
        if self.mode == "disabled":
            return

        metrics = {
            "val_only/loss": val_loss,
            "val_only/epoch": epoch,
        }

        # Log retrieval metrics
        if recall_metrics:
            for metric_name, value in recall_metrics.items():
                metrics[f"val_only/{metric_name}"] = value

        if mrr_metrics:
            for metric_name, value in mrr_metrics.items():
                metrics[f"val_only/{metric_name}"] = value

        # Embedding stats
        if video_norm is not None:
            metrics["val_only/embeddings/video_norm"] = video_norm
        if text_norm is not None:
            metrics["val_only/embeddings/text_norm"] = text_norm
        if alignment_score is not None:
            metrics["val_only/embeddings/alignment_score"] = alignment_score

        # Additional metrics
        for key, value in kwargs.items():
            metrics[f"val_only/{key}"] = value

        wandb.log(metrics, step=self.step)

    def log_global_pool_val_metrics(
        self,
        epoch: int,
        val_loss: float,
        recall_metrics: Optional[Dict[str, float]] = None,
        mrr_metrics: Optional[Dict[str, float]] = None,
        video_norm: Optional[float] = None,
        text_norm: Optional[float] = None,
        alignment_score: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log metrics from the 'global_pool_val' scenario.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss (global pool scenario)
            recall_metrics: Dictionary containing Recall@K metrics (global pool)
            mrr_metrics: Dictionary containing MRR metrics (global pool)
            video_norm: Average L2 norm of video embeddings (global pool)
            text_norm: Average L2 norm of text embeddings (global pool)
            alignment_score: Average cosine similarity of positive pairs (global pool)

        Recommended additional metrics for retrieval:
        - NDCG@K
        - Median Rank
        - R@10, R@50
        - MAP

        If computed, log them similarly.
        """
        if self.mode == "disabled":
            return

        metrics = {
            "global_pool_val/loss": val_loss,
            "global_pool_val/epoch": epoch,
        }

        # Log retrieval metrics
        if recall_metrics:
            for metric_name, value in recall_metrics.items():
                metrics[f"global_pool_val/{metric_name}"] = value

        if mrr_metrics:
            for metric_name, value in mrr_metrics.items():
                metrics[f"global_pool_val/{metric_name}"] = value

        # Embedding stats
        if video_norm is not None:
            metrics["global_pool_val/embeddings/video_norm"] = video_norm
        if text_norm is not None:
            metrics["global_pool_val/embeddings/text_norm"] = text_norm
        if alignment_score is not None:
            metrics["global_pool_val/embeddings/alignment_score"] = alignment_score

        # Additional metrics
        for key, value in kwargs.items():
            metrics[f"global_pool_val/{key}"] = value

        wandb.log(metrics, step=self.step)

    def log_media(
        self,
        video_path: str,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log video or image media."""
        if self.mode == "disabled":
            return

        wandb.log(
            {"media": wandb.Video(video_path, caption=caption)},
            step=step or self.step,
        )

    def finish(self) -> None:
        """Finish logging and close wandb run."""
        if self.mode != "disabled":
            wandb.finish()

    def log_validation_examples(
        self,
        scenario: str,  # 'val_only' or 'global_pool_val'
        val_best_videos,
        val_best_reports,
        val_worst_videos,
        val_worst_reports,
        epoch: int,
    ) -> None:
        """Log validation video examples (best/worst) under the chosen scenario (val_only or global_pool_val)."""
        if self.mode == "disabled":
            return

        from utils.logging import cleanup_temp_video, convert_video_for_wandb

        temp_files = []

        # Log best retrievals
        for i, (video_path, report_data) in enumerate(zip(val_best_videos, val_best_reports)):
            try:
                mp4_path, is_temp = convert_video_for_wandb(video_path)
                if is_temp:
                    temp_files.append(mp4_path)

                unique_predicted_reports = []
                seen_reports = set()
                for text in report_data["predicted"]:
                    if text not in seen_reports:
                        unique_predicted_reports.append(text)
                        seen_reports.add(text)

                report_html = "<br>".join(
                    [
                        f"<b>Ground Truth:</b> {report_data['ground_truth']}",
                        "<b>Top 5 Retrieved Reports:</b>",
                        *[f"{j+1}. {text}" for j, text in enumerate(unique_predicted_reports)],
                    ]
                )

                # Include epoch in the keys for uniqueness
                wandb.log(
                    {
                        f"{scenario}/qualitative/good_retrieval_epoch_{epoch}_{i}": wandb.Video(
                            mp4_path,
                            caption=f"Good {scenario} Retrieval {i+1} (Sim: {report_data['similarity_score']:.3f})",
                        ),
                        f"{scenario}/qualitative/good_reports_epoch_{epoch}_{i}": wandb.Html(
                            report_html
                        ),
                        f"{scenario}/qualitative/good_similarity_epoch_{epoch}_{i}": report_data[
                            "similarity_score"
                        ],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            except Exception as e:
                print(f"Warning: Failed to log good {scenario} video {video_path}: {str(e)}")

        # Log worst retrievals
        for i, (video_path, report_data) in enumerate(zip(val_worst_videos, val_worst_reports)):
            try:
                mp4_path, is_temp = convert_video_for_wandb(video_path)
                if is_temp:
                    temp_files.append(mp4_path)

                unique_predicted_reports = []
                seen_reports = set()
                for text in report_data["predicted"]:
                    if text not in seen_reports:
                        unique_predicted_reports.append(text)
                        seen_reports.add(text)

                report_html = "<br>".join(
                    [
                        f"<b>Ground Truth:</b> {report_data['ground_truth']}",
                        "<b>Top 5 Retrieved Reports:</b>",
                        *[f"{j+1}. {text}" for j, text in enumerate(unique_predicted_reports)],
                    ]
                )

                wandb.log(
                    {
                        f"{scenario}/qualitative/bad_retrieval_epoch_{epoch}_{i}": wandb.Video(
                            mp4_path,
                            caption=f"Bad {scenario} Retrieval {i+1} (Sim: {report_data['similarity_score']:.3f})",
                        ),
                        f"{scenario}/qualitative/bad_reports_epoch_{epoch}_{i}": wandb.Html(
                            report_html
                        ),
                        f"{scenario}/qualitative/bad_similarity_epoch_{epoch}_{i}": report_data[
                            "similarity_score"
                        ],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            except Exception as e:
                print(f"Warning: Failed to log bad {scenario} video {video_path}: {str(e)}")

        for temp_file in temp_files:
            cleanup_temp_video(temp_file)


def create_logger(
    args,
    project_name: str = "deepcoro_clip",
    experiment_name: Optional[str] = None,
) -> WandbLogger:
    """Create a WandbLogger instance with configuration from args."""

    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "gpu": args.gpu,
    }

    # Add any additional args to config
    for key, value in vars(args).items():
        if key not in config:
            config[key] = value

    return WandbLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        config=config,
    )


def convert_video_for_wandb(video_path):
    """Convert video to MP4 format for wandb logging if needed.

    Args:
        video_path: Path to input video

    Returns:
        tuple: (output_path, is_temp) where is_temp indicates if the file needs cleanup
    """
    # If already MP4, return as is
    if video_path.lower().endswith(".mp4"):
        return video_path, False

    import subprocess
    import tempfile

    # Create temporary MP4 file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)

    try:
        # Convert to MP4 using ffmpeg
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-c:v", "libx264", "-preset", "fast", "-y", temp_path],
            check=True,
            capture_output=True,
        )
        return temp_path, True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to convert video {video_path}: {e.stderr.decode()}")
        os.unlink(temp_path)
        return video_path, False


def cleanup_temp_video(video_path):
    """Delete temporary video file if it exists."""
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception as e:
        print(f"Warning: Failed to delete temporary video {video_path}: {str(e)}")


def get_best_and_worst_retrievals(similarity_matrix, paths, reports, k=2):
    """Get the best and worst retrievals based on similarity scores, along with their top text matches.

    Args:
        similarity_matrix: Tensor of shape (num_videos, num_queries)
        paths: List of video paths
        reports: List of report texts
        k: Number of best/worst examples to return

    Returns:
        tuple: (best_indices, worst_indices, best_scores, worst_scores, best_text_indices, worst_text_indices)
    """
    # Get mean similarity score for each video-query pair
    mean_similarities = similarity_matrix.mean(dim=1)

    # Adjust k to not exceed batch size
    k = min(k, len(mean_similarities))

    # Get indices of best and worst k videos
    best_values, best_indices = torch.topk(mean_similarities, k=k)
    worst_values, worst_indices = torch.topk(mean_similarities, k=k, largest=False)

    # Get top-5 text matches for each video
    best_text_indices = []
    worst_text_indices = []

    for idx in best_indices:
        # Get top N text matches for this video, where N is min(5, batch_size)
        n_texts = min(5, similarity_matrix.size(1))
        top_n_texts = torch.topk(similarity_matrix[idx], k=n_texts)[1]
        unique_texts = []
        seen_texts = set()
        for text_idx in top_n_texts:
            if text_idx.item() not in seen_texts:
                unique_texts.append(text_idx)
                seen_texts.add(text_idx.item())
            if len(unique_texts) == 5:
                break
        best_text_indices.append(torch.tensor(unique_texts))

    for idx in worst_indices:
        # Get top N text matches for this video, where N is min(5, batch_size)
        n_texts = min(5, similarity_matrix.size(1))
        top_n_texts = torch.topk(similarity_matrix[idx], k=n_texts)[1]
        unique_texts = []
        seen_texts = set()
        for text_idx in top_n_texts:
            if text_idx.item() not in seen_texts:
                unique_texts.append(text_idx)
                seen_texts.add(text_idx.item())
            if len(unique_texts) == 5:
                break
        worst_text_indices.append(torch.tensor(unique_texts))

    return (
        best_indices,
        worst_indices,
        best_values,
        worst_values,
        best_text_indices,
        worst_text_indices,
    )


def compute_ndcg(similarity_matrix: torch.Tensor, global_gt_indices: torch.Tensor, k=5) -> float:
    """
    Compute NDCG@k for each query and average over all queries.
    Simplified assumption: one correct answer per query.

    Args:
        similarity_matrix (torch.Tensor): [num_queries, num_candidates]
        global_gt_indices (torch.Tensor): [num_queries], each entry is the index of the correct text
        k (int): Rank cutoff

    Returns:
        float: Average NDCG@k over all queries.
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0.0

    # Sort candidates by similarity in descending order
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

    ndcg_values = []
    for i in range(num_queries):
        correct_idx = global_gt_indices[i].item()
        # Find the rank of the correct index
        ranking = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
        if ranking.numel() == 0:
            # Correct item not found (should not happen if all candidates included)
            ndcg_values.append(0.0)
            continue

        rank = ranking.item()
        if rank < k:
            # DCG = 1 / log2(rank+2)
            dcg = 1.0 / math.log2(rank + 2)
        else:
            dcg = 0.0

        # Ideal DCG (IDCG) = 1 since there's only one relevant doc at best rank
        idcg = 1.0
        ndcg_values.append(dcg / idcg)

    return float(torch.tensor(ndcg_values).mean().item())


def compute_median_rank(
    similarity_matrix: torch.Tensor, global_gt_indices: torch.Tensor
) -> float:
    """
    Compute the median rank of the correct item over all queries.
    Lower is better.
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0.0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    ranks = []
    for i in range(num_queries):
        correct_idx = global_gt_indices[i].item()
        ranking = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
        if ranking.numel() == 0:
            # Not found, assign large rank
            ranks.append(similarity_matrix.size(1))
        else:
            rank = ranking.item() + 1  # +1 because ranks are 1-based
            ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)
    median_rank = ranks.median().item()
    return median_rank


def compute_map(similarity_matrix: torch.Tensor, global_gt_indices: torch.Tensor) -> float:
    """
    Compute mean average precision (MAP).
    Assuming exactly one relevant doc per query.
    AP = 1/rank_of_correct_item
    MAP = average of AP over all queries
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0.0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    aps = []
    for i in range(num_queries):
        correct_idx = global_gt_indices[i].item()
        ranking = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
        if ranking.numel() == 0:
            # Correct not found, AP=0
            aps.append(0.0)
        else:
            rank = ranking.item() + 1
            ap = 1.0 / rank
            aps.append(ap)

    return float(torch.tensor(aps).mean().item())

def log_val_only_retrievals(
    similarity_matrix: torch.Tensor,
    all_paths: List[str],
    all_ground_truth_reports: List[str],
    all_reports: List[str],
    epoch: int,
    wandb_run,
    output_dir: str,
    k: int = 1,
):
    """
    Log best and worst retrieval examples for val-only scenario.
    Writes out a CSV with top-5 predictions per video, logs best/worst samples to wandb.

    Args:
        similarity_matrix: [N, M] matrix (N = #videos, M = #text embeddings)
        all_paths: List of video file paths
        all_ground_truth_reports: Ground-truth report for each video
        all_reports: Full list of text reports (used for indexing)
        epoch: Current epoch
        wandb_run: The wandb.Run object for logging
        output_dir: Directory to save CSV and other artifacts
        k: # of best/worst items to highlight in the logs
    """
    # Create CSV file listing top-5 predictions for each video
    val_csv_path = os.path.join(output_dir, f"val_epoch{epoch}.csv")
    os.makedirs(os.path.dirname(val_csv_path), exist_ok=True)

    with open(val_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "FileName",
            "ground_truth_idx",
            "predicted_idx_1",
            "sim_1",
            "predicted_idx_2",
            "sim_2",
            "predicted_idx_3",
            "sim_3",
            "predicted_idx_4",
            "sim_4",
            "predicted_idx_5",
            "sim_5",
        ]
        writer.writerow(header)

        for i, path in enumerate(all_paths):
            top_5_text_indices = torch.argsort(similarity_matrix[i], descending=True)[:5]
            predicted_indices = [idx.item() for idx in top_5_text_indices]
            predicted_sims = [similarity_matrix[i, idx].item() for idx in top_5_text_indices]

            row_data = [path, i]
            for p_idx, p_sim in zip(predicted_indices, predicted_sims):
                row_data.append(p_idx)
                row_data.append(f"{p_sim:.4f}")

            writer.writerow(row_data)

    # Pick the best/worst examples based on max similarity
    max_scores, _ = similarity_matrix.max(dim=1)
    best_scores, best_indices = torch.topk(max_scores, k=k)
    worst_scores, worst_indices = torch.topk(max_scores, k=k, largest=False)

    # Prepare lists to store for logging
    val_best_videos = []
    val_best_reports = []
    val_worst_videos = []
    val_worst_reports = []

    # Gather best retrieval
    for i in range(best_indices.numel()):
        idx = best_indices[i].item()
        score = best_scores[i].item()
        top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
        predicted_reports = [all_reports[t_idx.item()] for t_idx in top_5_text_indices]

        val_best_videos.append(all_paths[idx])
        val_best_reports.append(
            {
                "ground_truth": all_ground_truth_reports[idx],
                "predicted": predicted_reports,
                "similarity_score": score,
            }
        )

    # Gather worst retrieval
    for i in range(worst_indices.numel()):
        idx = worst_indices[i].item()
        score = worst_scores[i].item()
        top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
        predicted_reports = [all_reports[t_idx.item()] for t_idx in top_5_text_indices]

        val_worst_videos.append(all_paths[idx])
        val_worst_reports.append(
            {
                "ground_truth": all_ground_truth_reports[idx],
                "predicted": predicted_reports,
                "similarity_score": score,
            }
        )

    # Log best/worst to wandb if available
    if wandb_run is not None:
        # -- log best example(s) --
        if val_best_videos:
            for i, (video_path, report_data) in enumerate(zip(val_best_videos, val_best_reports)):
                mp4_path, is_temp = convert_video_for_wandb(video_path)
                # Log the video
                wandb_run.log(
                    {
                        f"qualitative/good_retrieval": wandb.Video(
                            mp4_path,
                            caption=f"Sim: {report_data['similarity_score']:.3f}",
                            format="mp4",
                        ),
                        "epoch": epoch,
                    }
                )
                # Build HTML
                predicted_html = "<br>".join(
                    [f"{j+1}. {text}" for j, text in enumerate(report_data["predicted"])]
                )
                ground_truth_html = (
                    f"<b>Ground Truth:</b> {report_data['ground_truth']}<br>"
                    f"<b>Top 5 Predicted:</b><br>{predicted_html}"
                )
                wandb_run.log(
                    {
                        f"qualitative/good_retrieval_text": wandb.Html(ground_truth_html),
                        "epoch": epoch,
                    }
                )
                # Cleanup
                if is_temp:
                    cleanup_temp_video(mp4_path)

        # -- log worst example(s) --
        if val_worst_videos:
            for i, (video_path, report_data) in enumerate(zip(val_worst_videos, val_worst_reports)):
                mp4_path, is_temp = convert_video_for_wandb(video_path)
                # Log the video
                wandb_run.log(
                    {
                        f"qualitative/bad_retrieval": wandb.Video(
                            mp4_path,
                            caption=f"Sim: {report_data['similarity_score']:.3f}",
                            format="mp4",
                        ),
                        "epoch": epoch,
                    }
                )
                # Build HTML
                predicted_html = "<br>".join(
                    [f"{j+1}. {text}" for j, text in enumerate(report_data["predicted"])]
                )
                ground_truth_html = (
                    f"<b>Ground Truth:</b> {report_data['ground_truth']}<br>"
                    f"<b>Top 5 Predicted:</b><br>{predicted_html}"
                )
                wandb_run.log(
                    {
                        f"qualitative/bad_retrieval_text": wandb.Html(ground_truth_html),
                        "epoch": epoch,
                    }
                )
                # Cleanup
                if is_temp:
                    cleanup_temp_video(mp4_path)


def compute_recall_at_k(similarity_matrix, global_gt_indices, k_values=[1, 5]):
    batch_size = similarity_matrix.shape[0]
    metrics = {}

    for k in k_values:
        v2t_topk = torch.topk(similarity_matrix, k, dim=1)[1]
        v2t_correct = v2t_topk == global_gt_indices.unsqueeze(1)
        v2t_recall = (v2t_correct.sum(dim=1) > 0).float().mean().item()
        metrics[f"Recall@{k}_V2T"] = v2t_recall
        ## A report text can haev multiple video ground truths so cant calculate t2v
        # t2v_topk = torch.topk(similarity_matrix.t(), k, dim=1)[1]
        # t2v_correct = t2v_topk == global_gt_indices.unsqueeze(1)
        # t2v_recall = (t2v_correct.sum(dim=1) > 0).float().mean().item()
        # metrics[f"Recall@{k}_T2V"] = t2v_recall

    return metrics


def compute_mrr(similarity_matrix, global_gt_indices):
    batch_size = similarity_matrix.shape[0]
    device = similarity_matrix.device

    # Video to Text
    target_scores = similarity_matrix.gather(1, global_gt_indices.unsqueeze(1))
    v2t_ranks = (similarity_matrix >= target_scores).sum(1).float()
    v2t_mrr = (1 / v2t_ranks).mean().item()

    return {"MRR_V2T": v2t_mrr}


def compute_embedding_norms(video_features, text_features):
    """Compute L2 norms of video and text embeddings."""
    video_norms = torch.norm(video_features, dim=1).mean().item()
    text_norms = torch.norm(text_features, dim=1).mean().item()
    return {"video_norm": video_norms, "text_norm": text_norms}


def compute_alignment_score(
    video_features,
    text_features,
    all_video_embeddings=None,
    all_text_embeddings=None,
    global_ground_truth_indices_tensor=None,
):
    """
    Compute average cosine similarity of positive pairs.

    Parameters:
    - video_features: torch.Tensor (batch local video embeddings)
    - text_features: torch.Tensor (batch local text embeddings)
    - all_video_embeddings: torch.Tensor of all validation video embeddings [N_videos, dim] (optional)
    - all_text_embeddings: torch.Tensor of all global text embeddings [N_texts, dim] (optional)
    - global_ground_truth_indices_tensor: torch.Tensor of global GT indices for each video (optional)

    If all_video_embeddings, all_text_embeddings, and global_ground_truth_indices_tensor
    are provided, compute global alignment using global embeddings.

    Otherwise, compute local alignment score assuming a one-to-one mapping between
    video_features[i] and text_features[i].
    """
    if (
        all_video_embeddings is not None
        and all_text_embeddings is not None
        and global_ground_truth_indices_tensor is not None
    ):
        # Global alignment scenario (for validation)
        correct_text_embeddings = all_text_embeddings[global_ground_truth_indices_tensor]
        normalized_video = nn.functional.normalize(all_video_embeddings, dim=1)
        normalized_text = nn.functional.normalize(correct_text_embeddings, dim=1)
        alignment_scores = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()
    else:
        # Local alignment scenario (for training)
        normalized_video = nn.functional.normalize(video_features, dim=1)
        normalized_text = nn.functional.normalize(text_features, dim=1)
        alignment_scores = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()
