"""
Logging utilities for training and evaluation.
"""

import os
import csv
import wandb
import torch
import torch.nn as nn
from datetime import datetime
from typing import Any, Dict, Optional, List

from utils.config import HeartWiseConfig
from utils.video import convert_video_for_wandb, cleanup_temp_video

class WandbLogger:
    """
    Wrapper for Weights & Biases logging.
    """

    def __init__(
        self,
        project_name: str = "deepcoro_clip",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "online",  # "online", "offline", or "disabled"
    ):
        """
        Initialize WandB logger.

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
        """
        Log metrics for a training batch.

        Only logs essential training metrics (loss, lr) for continuous monitoring.
        
        Args:
            loss: Training loss value
            batch_idx: Current batch index
            epoch: Current epoch number
            learning_rate: Current learning rate
            batch_size: Batch size used
            num_batches: Total number of batches per epoch
            **kwargs: Additional metrics to log        
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
        """
        Log comprehensive metrics at the end of each epoch for training.

        Args:
            epoch: Current epoch number
            train_loss: Average training loss for the epoch
            learning_rate: Optional current learning rate
        """
        if self.mode == "disabled":
            return

        # Optionally skip logging epoch=0, if you don't want that first row
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
        """
        Log metrics from the 'val_only' scenario.

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
        """
        Log metrics from the 'global_pool_val' scenario.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss (global pool scenario)
            recall_metrics: Dictionary containing Recall@K metrics (global pool)
            mrr_metrics: Dictionary containing MRR metrics (global pool)
            video_norm: Average L2 norm of video embeddings (global pool)
            text_norm: Average L2 norm of text embeddings (global pool)
            alignment_score: Average cosine similarity of positive pairs (global pool)
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
        """
        Log video or image media to wandb.
        """
        if self.mode == "disabled":
            return

        mp4_path, is_temp = convert_video_for_wandb(video_path)
        wandb.log(
            {"media": wandb.Video(mp4_path, caption=caption)},
            step=step or self.step,
        )
        if is_temp:
            cleanup_temp_video(mp4_path)

    def log_validation_examples(
        self,
        scenario: str,  # 'val_only' or 'global_pool_val'
        val_best_videos: List[str],
        val_best_reports: List[Dict[str, Any]],
        val_worst_videos: List[str],
        val_worst_reports: List[Dict[str, Any]],
        epoch: int,
    ) -> None:
        """
        Log validation video examples (best/worst) under the chosen scenario.
        Each item in val_best_reports or val_worst_reports is a dict with keys like:
            {
                "ground_truth": str,
                "predicted": List[str],
                "similarity_score": float
            }
        """
        if self.mode == "disabled":
            return

        temp_files = []

        # Log best retrievals
        for i, (video_path, report_data) in enumerate(zip(val_best_videos, val_best_reports)):
            try:
                mp4_path, is_temp = convert_video_for_wandb(video_path)
                if is_temp:
                    temp_files.append(mp4_path)

                # Remove duplicates from predicted
                seen = set()
                unique_predicted = []
                for txt in report_data["predicted"]:
                    if txt not in seen:
                        unique_predicted.append(txt)
                        seen.add(txt)

                report_html = "<br>".join(
                    [
                        f"<b>Ground Truth:</b> {report_data['ground_truth']}",
                        "<b>Top 5 Retrieved Reports:</b>",
                        *[f"{j+1}. {txt}" for j, txt in enumerate(unique_predicted)],
                    ]
                )

                wandb.log(
                    {
                        f"{scenario}/qualitative/good_retrieval_epoch_{epoch}_{i}": wandb.Video(
                            mp4_path,
                            caption=(
                                f"Good {scenario} Retrieval {i+1} "
                                f"(Sim: {report_data['similarity_score']:.3f})"
                            ),
                        ),
                        f"{scenario}/qualitative/good_reports_epoch_{epoch}_{i}": wandb.Html(report_html),
                        f"{scenario}/qualitative/good_similarity_epoch_{epoch}_{i}": report_data["similarity_score"],
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

                # Remove duplicates from predicted
                seen = set()
                unique_predicted = []
                for txt in report_data["predicted"]:
                    if txt not in seen:
                        unique_predicted.append(txt)
                        seen.add(txt)

                report_html = "<br>".join(
                    [
                        f"<b>Ground Truth:</b> {report_data['ground_truth']}",
                        "<b>Top 5 Retrieved Reports:</b>",
                        *[f"{j+1}. {txt}" for j, txt in enumerate(unique_predicted)],
                    ]
                )

                wandb.log(
                    {
                        f"{scenario}/qualitative/bad_retrieval_epoch_{epoch}_{i}": wandb.Video(
                            mp4_path,
                            caption=(
                                f"Bad {scenario} Retrieval {i+1} "
                                f"(Sim: {report_data['similarity_score']:.3f})"
                            ),
                        ),
                        f"{scenario}/qualitative/bad_reports_epoch_{epoch}_{i}": wandb.Html(report_html),
                        f"{scenario}/qualitative/bad_similarity_epoch_{epoch}_{i}": report_data["similarity_score"],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            except Exception as e:
                print(f"Warning: Failed to log bad {scenario} video {video_path}: {str(e)}")

        # Clean up any temp files
        for temp_file in temp_files:
            cleanup_temp_video(temp_file)


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
    """
    Get the best and worst retrievals based on average similarity scores across the row,
    along with their top text matches.
    similarity_matrix: [N, M]
    """
    mean_similarities = similarity_matrix.mean(dim=1)
    k = min(k, len(mean_similarities))

    best_values, best_indices = torch.topk(mean_similarities, k=k)
    worst_values, worst_indices = torch.topk(mean_similarities, k=k, largest=False)

    best_text_indices = []
    worst_text_indices = []

    for idx in best_indices:
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

def log_gradient_norms(model: nn.Module, prefix: str = ""):
    """
    Logs the L2 norm of gradients across the model's parameters.
    You can adjust to log per-layer or per-parameter if needed.
    """
    total_norm = 0.0
    param_count = 0
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
            param_count += 1

    total_norm = total_norm**0.5
    wandb.log({f"{prefix}grad_norm": total_norm})



def create_logger(config: HeartWiseConfig):
    """Create logger with proper WandB configuration.

    Args:
        args: Parsed command line arguments with config values

    Returns:
        WandbLogger instance
    """
    # Create config dictionary from args
    wandb_config = {
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "num_workers": config.num_workers,
        "gpu": config.gpu,
        "model_name": config.model_name,
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "scheduler_type": config.scheduler_type,
        "lr_step_period": config.lr_step_period,
        "factor": config.factor,
        "frames": config.frames,
        "pretrained": config.pretrained,
    }

    print(f"Project: {config.project}, Entity: {config.entity}, Tag: {config.tag}")

    # Initialize wandb with proper project and entity
    wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.tag,
        config=wandb_config,
    )

    return wandb.run


def log_best_worst_retrievals(
    similarity_matrix: torch.Tensor,
    all_paths: List[str],
    unique_texts: List[str],
    ground_truth_indices: torch.Tensor,
    epoch: int
) -> None:
    """Log best and worst retrievals to wandb.
    
    Args:
        wandb_logger: Wandb logger instance to use for logging
        similarity_matrix: Tensor of shape [num_videos x num_unique_texts] containing similarity scores
        all_paths: List of video paths
        unique_texts: List of unique text descriptions
        ground_truth_indices: Tensor mapping each video to its ground truth text index
        epoch: Current epoch number
    """
    # Find best and worst retrievals based on maximum similarity scores
    max_scores, _ = similarity_matrix.max(dim=1)
    k = 1  # Only pick top 1 best and worst
    best_scores, best_indices = torch.topk(max_scores, k=k)
    worst_scores, worst_indices = torch.topk(max_scores, k=k, largest=False)
    
    # Process and log best retrieval
    if best_indices.numel() > 0:
        _log_retrieval(
            idx=best_indices[0].item(),
            score=best_scores[0].item(),
            similarity_matrix=similarity_matrix,
            all_paths=all_paths,
            unique_texts=unique_texts,
            ground_truth_indices=ground_truth_indices,
            epoch=epoch,
            is_best=True
        )

    # Process and log worst retrieval
    if worst_indices.numel() > 0:
        _log_retrieval(
            idx=worst_indices[0].item(),
            score=worst_scores[0].item(),
            similarity_matrix=similarity_matrix,
            all_paths=all_paths,
            unique_texts=unique_texts,
            ground_truth_indices=ground_truth_indices,
            epoch=epoch,
            is_best=False
        )

def _log_retrieval(
    idx: int,
    score: float,
    similarity_matrix: torch.Tensor,
    all_paths: List[str],
    unique_texts: List[str],
    ground_truth_indices: torch.Tensor,
    epoch: int,
    is_best: bool
) -> None:
    """Helper function to log a single retrieval example."""
    # Get top 5 predicted texts
    top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
    predicted_texts = [unique_texts[j.item()] for j in top_5_text_indices]
    
    # Convert video and log
    mp4_path, is_temp = convert_video_for_wandb(all_paths[idx])
    
    prefix = "good" if is_best else "bad"
    wandb.log({
        f"qualitative/{prefix}_retrieval": wandb.Video(
            mp4_path,
            caption=f"Sim: {score:.3f}",
            format="mp4"
        ),
        "epoch": epoch
    })
    
    # Log text information
    predicted_html = "<br>".join(
        [f"{i+1}. {text}" for i, text in enumerate(predicted_texts)]
    )
    ground_truth_html = (
        f"<b>Ground Truth:</b> {unique_texts[ground_truth_indices[idx]]}<br>"
        f"<b>Top 5 Predicted:</b><br>{predicted_html}"
    )
    wandb.log({
        f"qualitative/{prefix}_retrieval_text": wandb.Html(ground_truth_html),
        "epoch": epoch
    })
    
    if is_temp:
        cleanup_temp_video(mp4_path)



def save_retrieval_results(
    similarity_matrix: torch.Tensor,
    all_paths: List[str],
    all_ground_truth_reports: List[str],
    report_to_global_index: Optional[Dict[str, int]],
    epoch: int,
    output_dir: str,
    gpu_id: int
) -> None:
    """
    Save retrieval results to a CSV, showing top-5 predicted indices and their similarities
    for each sample. If report_to_global_index is None, we default to using row index i as the
    ground-truth index.
    """
    val_csv_path = os.path.join(output_dir, f"val_epoch{epoch}.csv")

    with open(val_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
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

            gt_text = all_ground_truth_reports[i]

            # If we have a mapping dict, use it. Otherwise just use row index i.
            if report_to_global_index is not None:
                gt_idx = report_to_global_index[gt_text]
            else:
                gt_idx = i

            row_data = [path, gt_idx]

            for p_idx, p_sim in zip(predicted_indices, predicted_sims):
                row_data.append(p_idx)
                row_data.append(f"{p_sim:.4f}")

            writer.writerow(row_data)

    print(f"Saved retrieval results to {val_csv_path}")