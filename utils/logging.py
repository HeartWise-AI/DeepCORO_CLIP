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


def create_logger(
    args,
    project_name: str = "deepcoro_clip",
    experiment_name: Optional[str] = None,
) -> WandbLogger:
    """Create a WandbLogger instance with configuration from args.

    Args:
        args: Arguments from argparse containing training configuration
        project_name: Name of the project on WandB
        experiment_name: Optional name for this specific run

    Returns:
        WandbLogger instance
    """
    # Create config dictionary from args
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
