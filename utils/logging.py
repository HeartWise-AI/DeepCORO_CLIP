"""Logging utilities for training and evaluation."""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import wandb


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
