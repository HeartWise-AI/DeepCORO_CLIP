"""Logging utilities for training and evaluation."""

import os
import torch
import wandb

from datetime import datetime
from typing import Any, Dict, Optional, List

from utils.config import HeartWiseConfig
from utils.video import convert_video_for_wandb, cleanup_temp_video

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

        # Add any additional metrics that should be logged continuously
        for key, value in kwargs.items():
            if key.startswith("continuous_"):
                metrics[f"train/{key.replace('continuous_', '')}"] = value

        wandb.log(metrics, step=self.step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        recall_metrics: Optional[Dict[str, float]] = None,
        mrr_metrics: Optional[Dict[str, float]] = None,
        video_norm: Optional[float] = None,
        text_norm: Optional[float] = None,
        alignment_score: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log comprehensive metrics at the end of each epoch.

        Args:
            epoch: Current epoch number
            train_loss: Average training loss for the epoch
            val_loss: Optional validation loss
            learning_rate: Optional current learning rate
            recall_metrics: Optional dictionary of Recall@K metrics
            mrr_metrics: Optional dictionary of MRR metrics
            video_norm: Optional average L2 norm of video embeddings
            text_norm: Optional average L2 norm of text embeddings
            alignment_score: Optional average cosine similarity of positive pairs
            **kwargs: Additional metrics to log
        """
        if self.mode == "disabled":
            return

        self.epoch = epoch
        metrics = {
            "epoch": epoch,
            "epoch/train_loss": train_loss,
        }

        if val_loss is not None:
            metrics["epoch/val_loss"] = val_loss

        if learning_rate is not None:
            metrics["epoch/learning_rate"] = learning_rate

        # Add retrieval metrics if provided
        if recall_metrics:
            for metric_name, value in recall_metrics.items():
                metrics[f"epoch/retrieval/{metric_name}"] = value

        if mrr_metrics:
            for metric_name, value in mrr_metrics.items():
                metrics[f"epoch/retrieval/{metric_name}"] = value

        # Add embedding metrics if provided
        if video_norm is not None:
            metrics["epoch/embeddings/video_norm"] = video_norm
        if text_norm is not None:
            metrics["epoch/embeddings/text_norm"] = text_norm
        if alignment_score is not None:
            metrics["epoch/embeddings/alignment_score"] = alignment_score

        # Add any additional metrics
        for key, value in kwargs.items():
            metrics[f"epoch/{key}"] = value

        wandb.log(metrics, step=self.step)

    def log_validation(
        self,
        val_loss: float,
        epoch: int,
        **kwargs,
    ) -> None:
        """Log validation metrics.

        Args:
            val_loss: Validation loss value
            epoch: Current epoch number
            **kwargs: Additional validation metrics to log
        """
        if self.mode == "disabled":
            return

        metrics = {
            "val/loss": val_loss,
            "val/epoch": epoch,
        }

        # Add any additional validation metrics
        for key, value in kwargs.items():
            metrics[f"val/{key}"] = value

        wandb.log(metrics, step=self.step)

    def log_model_graph(self, model) -> None:
        """Log model architecture graph.

        Args:
            model: PyTorch model to visualize
        """
        if self.mode == "disabled":
            return

        wandb.watch(model, log="all")

    def log_media(
        self,
        video_path: str,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log video or image media.

        Args:
            video_path: Path to video/image file
            caption: Optional caption for the media
            step: Optional step number for logging
        """
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

    def log_retrieval_metrics(
        self,
        recall_metrics: Dict[str, float],
        mrr_metrics: Dict[str, float],
        split: str = "train",
        **kwargs,
    ) -> None:
        """Log detailed retrieval metrics at epoch end.

        Args:
            recall_metrics: Dictionary containing Recall@K metrics for V2T and T2V
            mrr_metrics: Dictionary containing MRR metrics for V2T and T2V
            split: Data split (train/val)
            **kwargs: Additional metrics to log
        """
        if self.mode == "disabled":
            return

        metrics = {}

        # Log Recall@K metrics
        for metric_name, value in recall_metrics.items():
            metrics[f"{split}/retrieval/{metric_name}"] = value

        # Log MRR metrics
        for metric_name, value in mrr_metrics.items():
            metrics[f"{split}/retrieval/{metric_name}"] = value

        # Add any additional metrics
        for key, value in kwargs.items():
            metrics[f"{split}/retrieval/{key}"] = value

        wandb.log(metrics, step=self.step)

    def log_embedding_stats(
        self,
        video_norm: float,
        text_norm: float,
        alignment_score: float,
        split: str = "train",
        **kwargs,
    ) -> None:
        """Log detailed embedding statistics at epoch end.

        Args:
            video_norm: Average L2 norm of video embeddings
            text_norm: Average L2 norm of text embeddings
            alignment_score: Average cosine similarity of positive pairs
            split: Data split (train/val)
            **kwargs: Additional embedding metrics to log
        """
        if self.mode == "disabled":
            return

        metrics = {
            f"{split}/embeddings/video_norm": video_norm,
            f"{split}/embeddings/text_norm": text_norm,
            f"{split}/embeddings/alignment_score": alignment_score,
        }

        # Add any additional metrics
        for key, value in kwargs.items():
            metrics[f"{split}/embeddings/{key}"] = value

        wandb.log(metrics, step=self.step)

    def log_validation_examples(
        self,
        val_best_videos,
        val_best_reports,
        val_worst_videos,
        val_worst_reports,
        epoch: int
    ) -> None:
        """Log validation video examples with their reports, keyed by epoch."""
        if self.mode == "disabled":
            return

        # Track temporary files for cleanup
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
                        f"qualitative/good_retrieval_epoch_{epoch}_{i}": wandb.Video(
                            mp4_path,
                            caption=f"Good Validation Retrieval {i+1} (Similarity: {report_data['similarity_score']:.3f})",
                        ),
                        f"qualitative/good_reports_epoch_{epoch}_{i}": wandb.Html(report_html),
                        f"qualitative/good_similarity_epoch_{epoch}_{i}": report_data["similarity_score"],
                        "epoch": epoch
                    },
                    step=epoch,  # Use epoch as step
                )

                print(
                    f"\nLogged good validation example {i+1} at epoch {epoch} (Similarity: {report_data['similarity_score']:.3f})"
                )

            except Exception as e:
                print(f"Warning: Failed to log good validation video {video_path}: {str(e)}")

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
                        f"qualitative/bad_retrieval_epoch_{epoch}_{i}": wandb.Video(
                            mp4_path,
                            caption=f"Bad Validation Retrieval {i+1} (Similarity: {report_data['similarity_score']:.3f})",
                        ),
                        f"qualitative/bad_reports_epoch_{epoch}_{i}": wandb.Html(report_html),
                        f"qualitative/bad_similarity_epoch_{epoch}_{i}": report_data["similarity_score"],
                        "epoch": epoch
                    },
                    step=epoch,  # Use epoch as step
                )

                print(
                    f"\nLogged bad validation example {i+1} at epoch {epoch} (Similarity: {report_data['similarity_score']:.3f})"
                )
            except Exception as e:
                print(f"Warning: Failed to log bad validation video {video_path}: {str(e)}")

        for temp_file in temp_files:
            cleanup_temp_video(temp_file)


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
    wandb_logger,
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
            wandb_logger=wandb_logger,
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
            wandb_logger=wandb_logger,
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
    wandb_logger,
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
    wandb_logger.log({
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
    wandb_logger.log({
        f"qualitative/{prefix}_retrieval_text": wandb.Html(ground_truth_html),
        "epoch": epoch
    })
    
    if is_temp:
        cleanup_temp_video(mp4_path)

