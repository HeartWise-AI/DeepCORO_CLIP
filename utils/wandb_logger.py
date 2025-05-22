"""
Logging utilities for training and evaluation.
"""

import os
import csv
import wandb
import torch
import torch.nn as nn
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from PIL import Image
import numpy as np
import tempfile
import shutil
import cv2
import subprocess

from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig
from utils.video import convert_video_for_wandb, cleanup_temp_video
from dataloaders.video_clip_dataset import VideoClipDataset

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


def convert_video_for_wandb(video_path: str, max_frames: int = 50, fps: int = 10) -> Tuple[str, bool]:
    """Convert video to MP4 format for wandb logging if needed.

    Args:
        video_path: Path to input video
        max_frames: Maximum number of frames to include in the video
        fps: Frames per second to use in the output video

    Returns:
        tuple: (output_path, is_temp) where is_temp indicates if the file needs cleanup
    """
    if video_path.lower().endswith(".mp4"):
        return video_path, False

    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)

    try:
        # Simplified ffmpeg command for compatibility, ensure target scale is reasonable
        # Example: scale to width 320, preserve aspect ratio for height, limit frames
        # The original scaling logic was: f"scale=-1:{(max_frames * fps) // fps}"
        # This can lead to very large dimensions if fps is high. Let's use a fixed width.
        # It seems the user wants a grid, so convert_video_for_wandb will be for single videos or fallback.
        # The grid creation will handle its own resizing.
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps},scale=320:-1", # Scale to width 320, adjust fps
            "-frames:v", str(max_frames), # Limit total frames
            "-c:v", "libx264", "-y", temp_path # Removed -preset fast
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return temp_path, True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to convert video {video_path} with ffmpeg: {e.stderr}")
        if os.path.exists(temp_path):
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

def log_gradient_norms(
    model: nn.Module,
    wandb_wrapper: WandbWrapper,
    prefix: str = ""):
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
    wandb_wrapper.log({f"{prefix}grad_norm": total_norm})



def create_logger(config: HeartWiseConfig):
    """Create logger with proper WandB configuration.

    Args:
        args: Parsed command line arguments with config values

    Returns:
        WandbLogger instance
    """
    # Create config dictionary from args
    wandb_config = {
        # Training parameters
        "epochs": config.epochs,
        "num_workers": config.num_workers,
        "debug": config.debug,
        "use_amp": config.use_amp,
        "period": config.period,
        "run_mode": config.run_mode,
        
        # Dataset parameters
        "data_filename": config.data_filename,
        "root": config.root,
        "target_label": config.target_label,
        "datapoint_loc_label": config.datapoint_loc_label,
        "frames": config.frames,
        "stride": config.stride,
        "per_video_pool": config.per_video_pool,
        "aggregate_videos_tokens": config.aggregate_videos_tokens,
        "multi_video": config.multi_video,
        "num_videos": config.num_videos,
        "groupby_column": config.groupby_column,
        "shuffle_videos": config.shuffle_videos,
        "batch_size": config.batch_size,
        
        # Seed
        "seed": config.seed,
        
        # Model parameters
        "model_name": config.model_name,
        "pretrained": config.pretrained,
        
        # Optimizer parameters
        "optimizer": config.optimizer,
        "scheduler_name": config.scheduler_name,
        "lr": config.lr,
        "lr_step_period": config.lr_step_period,
        "factor": config.factor,
        "loss_name": config.loss_name,
        "video_weight_decay": config.video_weight_decay,
        "text_weight_decay": config.text_weight_decay,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_warmup_percent": config.num_warmup_percent,
        "num_hard_restarts_cycles": config.num_hard_restarts_cycles,
        "warm_restart_tmult": config.warm_restart_tmult,
        "max_grad_norm": config.max_grad_norm,
        
        # Model architecture parameters
        "num_heads": config.num_heads,
        "aggregator_depth": config.aggregator_depth,
        "temperature": config.temperature,
        "dropout": config.dropout,
        "video_freeze_ratio": config.video_freeze_ratio,
        "text_freeze_ratio": config.text_freeze_ratio,
        
        # Checkpointing
        "resume_training": config.resume_training,
        "checkpoint": config.checkpoint,
        "output_dir": config.output_dir,
        "save_best": config.save_best,
        
        # Metrics
        "recall_k": config.recall_k,
        "ndcg_k": config.ndcg_k,
        
        # Data augmentation
        "rand_augment": config.rand_augment,
        "resize": config.resize,
        "apply_mask": config.apply_mask,
        
        # wandb parameters
        "tag": config.tag,
        "name": config.name,
        "project": config.project,
        "entity": config.entity,
        "use_wandb": config.use_wandb,
        
        # Inference parameters
        "topk": config.topk,
        "text_embeddings_path": config.text_embeddings_path,
        "metadata_path": config.metadata_path,
        "inference_results_path": config.inference_results_path,
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
    epoch: int, 
    wandb_wrapper: WandbWrapper,
    dataset_obj: VideoClipDataset 
) -> None:
    """Log best and worst retrievals to wandb.
    
    Args:
        wandb_logger: Wandb logger instance to use for logging
        similarity_matrix: Tensor of shape [num_videos x num_unique_texts] containing similarity scores
        all_paths: List of video paths (or SIDs in multi-video mode)
        unique_texts: List of unique text descriptions
        ground_truth_indices: Tensor mapping each video to its ground truth text index
        epoch: Current epoch number
        dataset_obj: The VideoClipDataset instance for multi-video path resolution.
    """
    if not wandb_wrapper.is_initialized(): # Check if wandb is initialized
        return
        
    # Find best and worst retrievals based on maximum similarity scores for each video
    max_scores_per_video, _ = similarity_matrix.max(dim=1)
    
    num_examples_to_log = 2  # Log top 2 best and top 2 worst
    
    # Ensure k is not greater than the number of videos
    k_actual = min(num_examples_to_log, max_scores_per_video.numel())
    
    if k_actual == 0:
        print("Warning: No videos to log for best/worst retrievals.")
        return

    best_scores, best_indices = torch.topk(max_scores_per_video, k=k_actual)
    worst_scores, worst_indices = torch.topk(max_scores_per_video, k=k_actual, largest=False)
    
    # Process and log best retrievals
    for i in range(best_indices.numel()):
        video_idx = best_indices[i].item()
        score = best_scores[i].item()
        _log_retrieval(
            idx=video_idx,
            score=score,
            similarity_matrix=similarity_matrix,
            all_paths=all_paths,
            unique_texts=unique_texts,
            ground_truth_indices=ground_truth_indices,
            epoch=epoch,
            is_best=True,
            wandb_wrapper=wandb_wrapper,
            dataset_obj=dataset_obj
        )

    # Process and log worst retrievals
    for i in range(worst_indices.numel()):
        video_idx = worst_indices[i].item()
        score = worst_scores[i].item()
        _log_retrieval(
            idx=video_idx,
            score=score,
            similarity_matrix=similarity_matrix,
            all_paths=all_paths,
            unique_texts=unique_texts,
            ground_truth_indices=ground_truth_indices,
            epoch=epoch,
            is_best=False,
            wandb_wrapper=wandb_wrapper,
            dataset_obj=dataset_obj
        )

def _log_retrieval(
    idx: int,
    score: float,
    similarity_matrix: torch.Tensor,
    all_paths: List[str],
    unique_texts: List[str],
    ground_truth_indices: torch.Tensor,
    epoch: int,
    is_best: bool,
    wandb_wrapper: WandbWrapper,
    dataset_obj: VideoClipDataset
) -> None:
    """Helper function to log a single retrieval example."""
    top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
    predicted_texts = [unique_texts[j.item()] for j in top_5_text_indices]
    
    video_to_log_path: Optional[str] = None
    is_temp_video = False

    if dataset_obj.multi_video_mode:
        sid = all_paths[idx]
        actual_video_files = dataset_obj.get_video_paths(sid)
        videos_for_grid = [vp for vp in actual_video_files if os.path.exists(vp)][:4]
        
        if videos_for_grid:
            print(f"Creating video grid for SID: {sid} with videos: {videos_for_grid}")
            video_to_log_path, is_temp_video = _create_video_grid(videos_for_grid)
            if not video_to_log_path:
                print(f"Warning: Failed to create video grid for SID {sid}. Skipping video log.")
        else:
            print(f"Warning: No valid video files found for SID {sid} to create a grid. Skipping video log.")
    else:
        single_video_path = all_paths[idx]
        if os.path.exists(single_video_path):
            video_to_log_path, is_temp_video = convert_video_for_wandb(single_video_path)
        else:
            print(f"Warning: Single video path '{single_video_path}' does not exist. Skipping video log.")

    if video_to_log_path:
        prefix = "good" if is_best else "bad"
        wandb_wrapper.log({
            f"qualitative/{prefix}_retrieval": wandb.Video(
                video_to_log_path,
                caption=f"Sim: {score:.3f} (Identifier: {all_paths[idx]})",
                format="mp4"
            ),
            "epoch": epoch
        })
        
        # Log text information
        predicted_html = "<br>".join(
            [f"{i+1}. {text}" for i, text in enumerate(predicted_texts)]
        )
        gt_text_idx = ground_truth_indices[idx].item()
        ground_truth_text_display = "N/A"
        if 0 <= gt_text_idx < len(unique_texts):
            ground_truth_text_display = unique_texts[gt_text_idx]
        else:
            print(f"Warning: Ground truth index {gt_text_idx} is out of bounds for unique_texts (len: {len(unique_texts)}). Using N/A.")

        ground_truth_html = (
            f"<b>Identifier:</b> {all_paths[idx]}<br>"
            f"<b>Ground Truth:</b> {ground_truth_text_display}<br>"
            f"<b>Top 5 Predicted:</b><br>{predicted_html}"
        )
        wandb.log({
            f"qualitative/{prefix}_retrieval_text": wandb.Html(ground_truth_html),
            "epoch": epoch
        })
        
        if is_temp_video:
            cleanup_temp_video(video_to_log_path)
    # else: (warnings for not logging video are handled above)


def _create_video_grid(video_paths: List[str], output_fps: int = 10, cell_resolution: Tuple[int, int] = (224, 224), grid_dim: Tuple[int, int] = (2,2)) -> Tuple[Optional[str], bool]:
    """
    Creates a video grid from a list of up to 4 videos.

    Args:
        video_paths: List of paths to input videos (max 4).
        output_fps: FPS for the output grid video.
        cell_resolution: Tuple (width, height) for each cell in the grid.
        grid_dim: Tuple (rows, cols) for the grid layout (e.g., (2,2) for 4 videos).

    Returns:
        Tuple (path_to_grid_video, is_temp_file) or (None, False) if failed.
    """
    if not video_paths:
        return None, False

    videos_to_process = video_paths[:grid_dim[0] * grid_dim[1]]
    caps = []
    for vp in videos_to_process:
        if not os.path.exists(vp):
            print(f"Warning: Video path {vp} not found for grid. Skipping this video.")
            continue
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"Warning: Could not open video {vp} for grid. Skipping this video.")
            continue
        caps.append(cap)

    if not caps:
        print("No valid videos to create a grid.")
        return None, False

    # Determine minimum frame count across all valid videos to sync them
    min_frames = float('inf')
    original_fps_list = []
    for cap in caps:
        original_fps_list.append(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            min_frames = min(min_frames, frame_count)
    
    if min_frames == float('inf') or min_frames == 0:
        print("Warning: Could not determine frame counts or videos are empty. Cannot create grid.")
        for cap in caps:
            cap.release()
        return None, False

    # Ensure min_frames is an integer for range()
    min_frames = int(min_frames)

    grid_width = grid_dim[1] * cell_resolution[0]
    grid_height = grid_dim[0] * cell_resolution[1]

    temp_fd, temp_grid_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)

    # Define the codec and create VideoWriter object
    # Using 'avc1' (H.264) or 'mp4v' (MPEG-4) - mp4v is often more compatible
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_writer = cv2.VideoWriter(temp_grid_path, fourcc, output_fps, (grid_width, grid_height))

    if not out_writer.isOpened():
        print(f"Error: Could not open VideoWriter for path {temp_grid_path}. Check OpenCV/ffmpeg setup.")
        for cap in caps:
            cap.release()
        if os.path.exists(temp_grid_path):
            os.unlink(temp_grid_path)
        return None, False

    try:
        for frame_num in range(min_frames):
            grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8) # Black background
            for i, cap in enumerate(caps):
                if frame_num == 0: # Reset video to beginning if looping or re-reading (not strictly needed here due to min_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                ret, frame = cap.read()
                if not ret:
                    # If a video ends prematurely (shouldn't happen if min_frames is correct)
                    # Fill its cell with black or last good frame (here, simply skip updating its cell)
                    continue 
                
                resized_frame = cv2.resize(frame, cell_resolution)
                
                row = i // grid_dim[1]
                col = i % grid_dim[1]
                y_offset = row * cell_resolution[1]
                x_offset = col * cell_resolution[0]
                
                grid_frame[y_offset:y_offset+cell_resolution[1], x_offset:x_offset+cell_resolution[0]] = resized_frame
            out_writer.write(grid_frame)
    finally:
        for cap in caps:
            cap.release()
        out_writer.release()

    return temp_grid_path, True


def save_retrieval_results(
    similarity_matrix: torch.Tensor,
    all_identifiers: List[str],
    all_ground_truth_reports: List[str],
    report_to_global_index: Optional[Dict[str, int]],
    epoch: int,
    output_dir: str,
    dataset_obj: VideoClipDataset
) -> None:
    """
    Save retrieval results to a CSV, showing top-5 predicted indices and their similarities
    for each sample. If report_to_global_index is None, we default to using row index i as the
    ground-truth index.
    Handles different CSV structures for single vs. multi-video modes.
    """
    val_csv_path = os.path.join(output_dir, f"val_epoch{epoch}.csv")
    
    multi_video_mode = dataset_obj.multi_video_mode
    actual_groupby_col_name = dataset_obj.groupby_column if dataset_obj.groupby_column else "study_id"

    with open(val_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        header_base = [
            "ground_truth_idx",
            "predicted_idx_1", "sim_1",
            "predicted_idx_2", "sim_2",
            "predicted_idx_3", "sim_3",
            "predicted_idx_4", "sim_4",
            "predicted_idx_5", "sim_5",
        ]
        
        header_prefix = []
        if multi_video_mode:
            header_prefix = [actual_groupby_col_name, "VideoFileNames"]
        else:
            header_prefix = ["FileName"]
            
        writer.writerow(header_prefix + header_base)

        for i, identifier in enumerate(all_identifiers):
            top_5_sim_scores, top_5_text_indices = torch.topk(similarity_matrix[i], k=5)
            predicted_indices = [idx.item() for idx in top_5_text_indices]
            predicted_sims = [score.item() for score in top_5_sim_scores]

            gt_text = all_ground_truth_reports[i]

            if report_to_global_index is not None and gt_text in report_to_global_index:
                gt_idx = report_to_global_index[gt_text]
            else:
                print(f"Warning: Ground truth text '{gt_text}' not found in report_to_global_index for identifier '{identifier}'. Using row index {i} as fallback gt_idx.")
                gt_idx = i 

            current_row_prefix_data = []
            if multi_video_mode:
                sid = identifier
                video_files_list = dataset_obj.get_video_paths(sid)
                video_files_str = ";".join(video_files_list) if video_files_list else ""
                current_row_prefix_data.extend([sid, video_files_str])
            else:
                filename = identifier
                current_row_prefix_data.append(filename)
            
            row_data_suffix = [gt_idx]
            for p_idx, p_sim in zip(predicted_indices, predicted_sims):
                row_data_suffix.append(p_idx)
                row_data_suffix.append(f"{p_sim:.4f}")
            
            num_prediction_pairs = (len(header_base) -1) // 2
            while len(predicted_indices) < num_prediction_pairs:
                row_data_suffix.extend(["", ""])
                predicted_indices.append(-1) 
            writer.writerow(current_row_prefix_data + row_data_suffix)
    print(f"Saved retrieval results to {val_csv_path}")