import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Dict, Tuple, List, Any

from tqdm import tqdm

from utils.enums import RunMode
from utils.ddp import DistributedUtils
from utils.config.clip_config import ClipConfig
from utils.registry import RunnerRegistry
from utils.metrics import (
    compute_mrr,
    compute_map,
    compute_ndcg_at_k,
    compute_median_rank,
    compute_recall_at_k,
    compute_embedding_norms,
    compute_alignment_score,
)
from utils.wandb_logger import (
    log_best_worst_retrievals,
    log_gradient_norms,
    save_retrieval_results,
)
from utils.loss.typing import Loss
from utils.wandb_wrapper import WandbWrapper
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from dataloaders.video_clip_dataset import VideoClipDataset
import itertools
from torch.nn.utils import clip_grad_norm_

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@RunnerRegistry.register("DeepCORO_clip")
@RunnerRegistry.register("DeepCORO_clip_test")
class VideoContrastiveLearningRunner:
    """
    This class runs a video contrastive learning pipeline using a VideoEncoder and TextEncoder.
    It handles both training and validation loops in a distributed data-parallel setting.
    """

    def __init__(
        self,
        config: ClipConfig = None,
        wandb_wrapper: WandbWrapper = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        video_encoder: VideoEncoder = None,
        text_encoder: TextEncoder = None,
        optimizer: Optimizer = None,
        scaler: GradScaler = None,
        log_temp: torch.Tensor = None,
        lr_scheduler: LRScheduler = None,
        loss_fn: Loss = None,
        output_dir: str = None,
    ):
        """
        Initialize the runner with provided configurations, data loaders, and modules.

        :param config: ClipConfig object with run/training configuration.
        :param wandb_wrapper: WandbWrapper instance.
        :param train_loader: DataLoader for training dataset.
        :param val_loader: DataLoader for validation dataset.
        :param video_encoder: VideoEncoder model.
        :param text_encoder: TextEncoder model.
        :param optimizer: optimizer instance.
        :param scheduler: Learning rate scheduler.
        :param scaler: GradScaler for automatic mixed precision.
        :param log_temp: Logarithm of temperature used in contrastive loss.
        :param lr_scheduler: Learning rate scheduler.
        :param loss_fn: Contrastive loss function callable.
        :param output_dir: Directory where checkpoints and outputs will be saved.
        :raises ValueError: If recall_k or ndcg_k are not lists of ints.
        """
        if not isinstance(config.recall_k, list):
            raise ValueError(
                f"config.recall_k must be a list of ints, got {type(config.recall_k)}"
            )
        if not isinstance(config.ndcg_k, list):
            raise ValueError(
                f"config.ndcg_k must be a list of ints, got {type(config.ndcg_k)}"
            )

        self.config: ClipConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        self.device: int = config.device
        self.world_size: int = config.world_size
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.video_encoder: VideoEncoder = video_encoder
        self.text_encoder: TextEncoder = text_encoder
        self.optimizer: Optimizer = optimizer
        self.scaler: GradScaler = scaler
        self.lr_scheduler: LRScheduler = lr_scheduler
        self.loss_fn: Loss = loss_fn
        self.output_dir: str = output_dir
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1
        self.highest_alignment_score: float = float("-inf")
        self.highest_alignment_epoch: int = -1
        self.log_temp: torch.Tensor = log_temp
        self.step: int = 0  # Initialize step counter for gradient accumulation

        # For simplicity: check the config for a known scheduler_name
        # If it includes "_warmup" or is from HF, we treat it as per-iteration
        self.scheduler_per_iteration = self._scheduler_is_per_iteration()

    def _scheduler_is_per_iteration(self) -> bool:
        """
        Returns True if the chosen scheduler is a Hugging Face style that
        expects a call to .step() per iteration (batch). Otherwise, False.
        """
        # We do a simpler check to change scheduler update to per epoch or per batch:
        # Example keywords: "linear_warmup", "cosine_with_warmup", 
        # "cosine_with_hard_restarts_with_warmup", etc.
        sched_name = getattr(self.config, "scheduler_name", "").lower()
        # If it matches typical HF warmup schedulers, return True
        HF_KEYS = ["warmup", "with_warmup"]
        return any(k in sched_name for k in HF_KEYS)


    def train(
        self, 
        start_epoch: int = 0, 
        end_epoch: int = 10,
    ):
        """
        Main training loop. Iterates through epochs, performing:
          - Training step
          - Validation step
          - Checkpoint saving
          - LR scheduling

        :param start_epoch: Starting epoch index.
        :param end_epoch: Ending epoch index.
        :raises RuntimeError: If NaN loss is detected during training.
        """
        try:
            for epoch in range(start_epoch, end_epoch):
                # Synchronize before epoch starts
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )

                # Training phase
                train_metrics: dict[str, float] = self._run_epoch(mode=RunMode.TRAIN, epoch=epoch)
                if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                    # Let wandb auto-increment steps
                    self.wandb_wrapper.log(train_metrics)
                    print(f"[DEBUG] rank={self.device} => Logged train metrics to W&B")

                # Sync before validation
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )

                # Validation phase
                val_metrics: dict[str, float] = self._run_epoch(
                    mode=RunMode.VALIDATION, 
                    epoch=epoch
                )
                
                # Sync after validation
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )
                
                # If it's an epoch-based scheduler (like StepLR, CosineAnnealingLR, etc.),
                # call lr_scheduler.step() after each epoch
                if self.lr_scheduler and (not self.scheduler_per_iteration):
                    self.lr_scheduler.step()

                # Update best model
                if val_metrics["val/loss"] < self.best_val_loss:
                    prev_best: float = self.best_val_loss
                    self.best_val_loss = val_metrics["val/loss"]
                    self.best_epoch = epoch
                    if self.config.is_ref_device:
                        print(
                            f"\nNew best model! Val Loss: {val_metrics['val/loss']:.4f} "
                            f"(previous: {prev_best:.4f})"
                        )
                        self._save_checkpoint(
                            epoch=epoch,
                            metrics={
                                **train_metrics, 
                                **val_metrics
                            },
                            is_best=True,
                        )
                
                # Update and save model with highest alignment score
                if val_metrics["val/alignment_score"] > self.highest_alignment_score:
                    prev_highest: float = self.highest_alignment_score
                    self.highest_alignment_score = val_metrics["val/alignment_score"]
                    self.highest_alignment_epoch = epoch
                    if self.config.is_ref_device:
                        print(
                            f"\nNew highest alignment score model! Alignment Score: {val_metrics['val/alignment_score']:.4f} "
                            f"(previous: {prev_highest:.4f})"
                        )
                        self._save_checkpoint(
                            epoch=epoch,
                            metrics={
                                **train_metrics, 
                                **val_metrics
                            },
                            is_best=False,
                            is_highest_alignment=True,
                        )

                # Always save "latest" checkpoint
                if self.config.is_ref_device:
                    self._save_checkpoint(
                        epoch=epoch,
                        metrics={**train_metrics, **val_metrics},
                        is_best=False,
                    )
                
                # Sync after saving checkpoint
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )
                            
                if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                    val_metrics['best_val_loss'] = self.best_val_loss
                    val_metrics['highest_alignment_score'] = self.highest_alignment_score
                    self.wandb_wrapper.log(val_metrics)
                    print(f"[DEBUG] rank={self.device} => Logged val metrics to W&B")
                
                # Sync after logging
                DistributedUtils.sync_process_group(
                    world_size=self.world_size,
                    device_ids=self.device
                )

        except RuntimeError as e:
            if "NaN loss" in str(e):
                if self.config.is_ref_device:
                    print("\nTraining stopped due to NaN loss. Saving final checkpoint...")
                    # Save a checkpoint with the error state
                    self._save_checkpoint(
                        epoch=epoch,
                        metrics={
                            "error": str(e),
                            "train/loss": float("nan"),
                            "val/loss": float("nan"),
                        },
                        is_best=False,
                    )
            raise  # Re-raise the exception after saving checkpoint

    def _gather_tensor_along_batch(self, local_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        """
        Gathers tensors across multiple ranks along the batch dimension (dim=0).

        :param local_tensor: Local tensor to gather from current rank.
        :param world_size: Number of participating ranks.
        :return: Concatenated tensor containing data from all ranks.
        """
        if self.config.world_size < 2:
            return local_tensor

        device: int = local_tensor.device
        local_size: torch.Tensor = torch.tensor([local_tensor.shape[0]], device=device, dtype=torch.long)
        sizes_list: list[torch.Tensor] = [torch.zeros_like(local_size) for _ in range(world_size)]
        DistributedUtils.dist.all_gather(sizes_list, local_size)
        sizes_list: list[int] = [s.item() for s in sizes_list]
        max_size: int = max(sizes_list)

        # Pad to max_size along dim=0
        if local_tensor.dim() == 1:
            pad: tuple[int, int] = (0, max_size - local_tensor.shape[0])
            padded: torch.Tensor = torch.nn.functional.pad(local_tensor, pad, "constant", 0)
        else:
            pad_rows: int = max_size - local_tensor.shape[0]
            if pad_rows > 0:
                padded: torch.Tensor = torch.nn.functional.pad(local_tensor, (0, 0, 0, pad_rows))
            else:
                padded: torch.Tensor = local_tensor

        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        DistributedUtils.dist.all_gather(gathered, padded)

        cat: torch.Tensor = torch.stack(gathered, dim=0)
        out_list: list[torch.Tensor] = []
        for rank_idx in range(world_size):
            actual_size: int = sizes_list[rank_idx]
            if local_tensor.dim() == 1:
                out_list.append(cat[rank_idx, :actual_size])
            else:
                out_list.append(cat[rank_idx, :actual_size, :])
        return torch.cat(out_list, dim=0)

    def _gather_strings_across_gpus(
        self, local_strings: list[str], world_size: int, device: torch.device
    ) -> list[str]:
        """
        Gathers lists of strings from multiple ranks into one combined list.

        :param local_strings: List of strings on the current rank.
        :param world_size: Number of participating ranks.
        :param device: CUDA device.
        :return: Combined list of strings from all ranks.
        """
        if self.config.world_size < 2:
            return local_strings

        import pickle

        local_data_bytes: bytes = pickle.dumps(local_strings)
        local_size: torch.Tensor = torch.tensor([len(local_data_bytes)], dtype=torch.long, device=device)

        sizes_list: list[torch.Tensor] = [torch.zeros_like(local_size) for _ in range(world_size)]
        DistributedUtils.dist.all_gather(sizes_list, local_size)
        sizes_list: list[int] = [s.item() for s in sizes_list]
        max_size: int = max(sizes_list)

        local_buffer: torch.Tensor = torch.zeros(max_size, dtype=torch.uint8, device=device)
        local_buffer[: local_size.item()] = torch.as_tensor(
            list(local_data_bytes), dtype=torch.uint8, device=device
        )

        all_buffers: list[torch.Tensor] = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
        DistributedUtils.dist.all_gather(all_buffers, local_buffer)

        out_list = []
        for rank_idx, buf in enumerate(all_buffers):
            size: int = sizes_list[rank_idx]
            valid_bytes: bytes = buf[:size].cpu().numpy().tobytes()
            str_list: list[str] = pickle.loads(valid_bytes)
            out_list.extend(str_list)

        return out_list

    def _run_epoch(
        self, 
        mode: str, 
        epoch: int
    ) -> dict[str, float]:
        """
        Runs either a training or validation epoch. Gathers embeddings across ranks
        and optionally computes NxN retrieval metrics on rank 0.

        :param mode: One of ["train", "val"] indicating the mode.
        :param epoch: Current epoch index.
        :return: Dictionary of metrics, averaged over all batches and reduced across ranks.
        :raises RuntimeError: If NaN loss is detected during training.
        """
        assert mode in [RunMode.TRAIN, RunMode.VALIDATION]

        self.video_encoder.train(mode == RunMode.TRAIN)
        self.text_encoder.train(mode == RunMode.TRAIN)

        total_loss: float = 0.0
        epoch_metrics: dict[str, float] = {}

        dataloader: DataLoader = self.train_loader if mode == RunMode.TRAIN else self.val_loader
        # Explicitly cast dataset to VideoClipDataset for type safety and access to custom methods
        dataset: VideoClipDataset = dataloader.dataset # type: ignore
        
        # Improved type hint for step_fn
        step_fn: Callable[..., Tuple[Dict[str, Any], Dict[str, torch.Tensor]]] = self._train_step if mode == RunMode.TRAIN else self._val_step


        # Store local embeddings & text for retrieval computations
        all_video_embeddings_local: List[torch.Tensor] = []
        all_text_embeddings_local: List[torch.Tensor] = []
        all_paths_local: List[str] = []
        all_ground_truth_reports_local: List[str] = []

        tqdm_desc = f"[GPU {self.device}] Running {mode} Epoch {epoch + 1}"
        
        # Determine the iterator: tqdm-wrapped or raw dataloader
        iterator_obj: Any = dataloader # Use Any for iterator_obj as it can be DataLoader or tqdm
        if self.config.is_ref_device or (self.device == 0):
            iterator_obj = tqdm(dataloader, desc=tqdm_desc)

        batch_count: int = 0
        for _, batch in enumerate(iterator_obj, start=1): # Use iterator_obj
            if batch["videos"] is None or batch["encoded_texts"] is None:
                continue

            step_inputs, paths_or_sids = self._preprocess_inputs(batch)

            batch_metrics, embeddings = step_fn(**step_inputs)

            # Check for NaN loss
            if torch.isnan(torch.tensor(batch_metrics["loss"])):
                error_msg = f"NaN loss detected in {mode} at epoch {epoch + 1}, batch {batch_count + 1}"
                if self.config.is_ref_device:
                    print(f"\nERROR: {error_msg}")
                raise RuntimeError(error_msg)

            if self.lr_scheduler and self.scheduler_per_iteration and (mode == RunMode.TRAIN):
                self.lr_scheduler.step()

            # Store embeddings on CPU
            all_video_embeddings_local.append(embeddings["video_embeddings"].cpu())
            all_text_embeddings_local.append(embeddings["text_embeddings"].cpu())

            # Use dataset.multi_video_mode for consistency
            # paths_or_sids contains SIDs in multi-video mode, and direct file paths in single-video mode.
            # all_paths_local will store these identifiers directly.
            all_paths_local.extend(paths_or_sids)

            all_ground_truth_reports_local.extend(dataset.get_reports(paths_or_sids)) # Now type-safe

            # accumulate metrics
            for k, v in batch_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)

            total_loss += float(batch_metrics["loss"])
            batch_count += 1

            if (self.config.is_ref_device or (self.device == 0)) and mode == RunMode.TRAIN:
                if hasattr(iterator_obj, 'set_postfix'): # Check if it's the tqdm instance
                    iterator_obj.set_postfix(
                        {
                            f"{mode}_loss": f"{batch_metrics['loss']:.4f}", # Use mode variable
                            "avg_loss": f"{(total_loss / batch_count):.4f}",
                        }
                    )

        if 'batch_metrics' in locals():
            del batch_metrics
        if 'embeddings' in locals():
            del embeddings
        torch.cuda.empty_cache()

        # finalize epoch metrics
        if batch_count > 0:
            for k in epoch_metrics.keys():
                epoch_metrics[k] /= batch_count

        # 1) Concat local feats
        if len(all_video_embeddings_local) > 0:
            local_video_feats: torch.Tensor = torch.cat(all_video_embeddings_local, dim=0).to(self.device)
            local_text_feats: torch.Tensor = torch.cat(all_text_embeddings_local, dim=0).to(self.device)
        else:
            local_video_feats: torch.Tensor = torch.empty((0, 512), device=self.device)
            local_text_feats: torch.Tensor = torch.empty((0, 512), device=self.device)

        # 2) gather across GPUs
        global_video_feats: torch.Tensor = self._gather_tensor_along_batch(local_video_feats, self.world_size)

        # 3) gather paths & reports
        global_paths: list[str] = self._gather_strings_across_gpus(
            all_paths_local, self.world_size, device=local_video_feats.device
        )
        global_reports: list[str] = self._gather_strings_across_gpus(
            all_ground_truth_reports_local, self.world_size, device=local_video_feats.device
        )

        # Optionally compute NxM retrieval metrics on rank 0
        retrieval_metrics: dict[str, float] = {}
        if mode == "val" and self.config.is_ref_device:
            print(
                f"[DEBUG rank={self.device}] Starting retrieval computation with {global_video_feats.shape[0]} videos."
            )
            
            # Step 1: Deduplicate texts and create mapping
            unique_texts: List[str] = sorted(set(global_reports))
            text_to_index: Dict[str, int] = {text: idx for idx, text in enumerate(unique_texts)}
            
            # Step 2: Get ground truth indices for each video
            ground_truth_indices: torch.Tensor = torch.tensor(
                [text_to_index[text] for text in global_reports],
                device=self.device
            )
            
            print(f"[DEBUG rank={self.device}] Found {len(unique_texts)} unique texts out of {len(global_reports)} total.")
            
            # Step 3: Encode unique texts
            unique_text_embeddings_list: List[torch.Tensor] = [] # Renamed list
            batch_size: int = 64  # Process in batches to avoid OOM
            
            self.text_encoder.eval()
            with torch.no_grad():
                for start_idx in range(0, len(unique_texts), batch_size):
                    end_idx = min(start_idx + batch_size, len(unique_texts))
                    text_batch = unique_texts[start_idx:end_idx]
                    
                    # Tokenize batch, Handle both DDP and non-DDP cases for text encoder
                    tokenizer = (
                        self.text_encoder.module.tokenizer 
                        if hasattr(self.text_encoder, "module") 
                        else self.text_encoder.tokenizer
                    )
                    encoded = tokenizer(
                        text_batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    # Get embeddings
                    text_embs = self.text_encoder(
                        encoded["input_ids"],
                        encoded["attention_mask"]
                    )
                    unique_text_embeddings_list.append(text_embs) # Append to renamed list
            
            # Concatenate all batches into a new tensor variable
            unique_text_embeddings_tensor: torch.Tensor = torch.cat(unique_text_embeddings_list, dim=0)
            
            # Step 4: Normalize embeddings
            global_video_feats_norm: torch.Tensor = nn.functional.normalize(global_video_feats, dim=1)
            unique_text_embeddings_norm: torch.Tensor = nn.functional.normalize(unique_text_embeddings_tensor, dim=1) # Use normalized tensor
            
            # Step 5: Compute NxM similarity matrix
            similarity_matrix: torch.Tensor = torch.matmul(global_video_feats_norm, unique_text_embeddings_norm.t()) # Use normalized embeddings
            print(
                f"[DEBUG rank={self.device}] Computed NxM sim matrix with shape={similarity_matrix.shape}"
            )
            
            # Log best/worst retrievals using unique texts
            if self.wandb_wrapper.is_initialized():
                log_best_worst_retrievals(
                    similarity_matrix=similarity_matrix,
                    all_paths=global_paths,
                    unique_texts=unique_texts,
                    ground_truth_indices=ground_truth_indices,
                    epoch=epoch,
                    wandb_wrapper=self.wandb_wrapper,
                    dataset_obj=dataset # Pass the dataset object
                )

            # Save retrieval results with mapping
            save_retrieval_results(
                similarity_matrix=similarity_matrix,
                all_identifiers=global_paths, # Pass SIDs or file paths
                all_ground_truth_reports=global_reports,
                report_to_global_index=text_to_index,
                epoch=epoch,
                output_dir=self.output_dir,
                dataset_obj=dataset # Pass the dataset object
            )
            print(f"ground_truth_indices={ground_truth_indices}")
            # Compute retrieval metrics using ground truth indices
            recall_metrics: Dict[str, float] = compute_recall_at_k( # Ensure it's a Dict
                similarity_matrix, ground_truth_indices, k_values=self.config.recall_k
            )
            mrr_score_dict: Dict[str, float] = compute_mrr(similarity_matrix, ground_truth_indices) # Assign to dict
            map_score: float = compute_map(similarity_matrix, ground_truth_indices) # Assuming this returns float
            median_rank_score: float = compute_median_rank(similarity_matrix, ground_truth_indices) # Assuming this returns float
            ndcg_scores_dict: Dict[str, float] = compute_ndcg_at_k( # Assign to dict
                similarity_matrix, ground_truth_indices, k_values=self.config.ndcg_k
            )

            retrieval_metrics.update(recall_metrics)
            retrieval_metrics.update(mrr_score_dict) # Update with dict
            retrieval_metrics.update(ndcg_scores_dict) # Update with dict
            retrieval_metrics["MAP"] = map_score 
            retrieval_metrics["MedianRank_V2T"] = median_rank_score
            
            # Save unique texts and their indices
            df_texts: pd.DataFrame = pd.DataFrame({
                "Index": range(len(unique_texts)),
                "Text": unique_texts
            })
            texts_csv_path: str = os.path.join(self.output_dir, f"val_unique_texts.csv")
            df_texts.to_csv(texts_csv_path, index=False)
            print(f"[DEBUG rank={self.device}] Saved {len(unique_texts)} unique texts to {texts_csv_path}")
            
            # Also save text embeddings for future use
            embeddings_path: str = os.path.join(self.output_dir, f"val_text_embeddings_epoch_{epoch}.pt")
            torch.save({
                'text_embeddings': unique_text_embeddings_tensor.cpu(), # Save the concatenated tensor
                'text_to_index': text_to_index,
                'unique_texts': unique_texts
            }, embeddings_path)
            print(f"[DEBUG rank={self.device}] Saved text embeddings to {embeddings_path}")
            
            print(
                f"[DEBUG rank={self.device}] Completed retrieval metrics & logging for val at epoch={epoch}"
            )
        else:
            # Non-zero ranks: placeholders for the same metrics
            for k_val in self.config.recall_k:
                retrieval_metrics[f"Recall@{k_val}"] = 0.0
            retrieval_metrics.update({
                "MRR": 0.0,
                "MAP": 0.0,
                "MedianRank_V2T": 0.0,
            })
            for k_val in self.config.ndcg_k:
                retrieval_metrics[f"NDCG@{k_val}"] = 0.0

        epoch_metrics.update(retrieval_metrics)

        # 4) reduce final epoch metrics across ranks
        gathered_metrics: dict[str, float] = {}
        for k, v in epoch_metrics.items():
            gathered_metrics[f"{mode}/{k}"] = self._maybe_reduce_metric(k, v)

        return gathered_metrics

    def _maybe_reduce_metric(self, name: str, val: float) -> float:
        """
        Optionally reduces (averages) a metric value across all ranks in DDP.

        :param name: Metric name (unused here, but can be helpful for debug).
        :param val: Metric value on current rank.
        :return: Mean metric value across all ranks, if DDP is initialized. Otherwise, returns val.
        """
        if self.config.world_size > 1:
            t = torch.tensor([val], dtype=torch.float, device=self.device)
            DistributedUtils.dist.all_reduce(t, op=DistributedUtils.dist.ReduceOp.AVG)
            return t.item()
        else:
            return val

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False, is_highest_alignment: bool = False):
        """
        Saves model checkpoint (including model weights, optimizer, scheduler, and metrics).

        :param epoch: Current epoch index.
        :param metrics: Dictionary of metrics to be saved.
        :param is_best: If True, saves as 'best_model_epoch_{epoch}.pt'. Otherwise, saves as 'checkpoint.pt'.
        :param is_highest_alignment: If True, saves as 'highest_alignment_epoch_{epoch}.pt'.
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_dict = {
            "video_encoder": self.video_encoder.module.state_dict()
            if hasattr(self.video_encoder, "module")
            else self.video_encoder.state_dict(),
            "text_encoder": self.text_encoder.module.state_dict()
            if hasattr(self.text_encoder, "module")
            else self.text_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "epoch": epoch,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

        checkpoint = {
            **model_dict,
            **metrics,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "highest_alignment_score": self.highest_alignment_score,
            "highest_alignment_epoch": self.highest_alignment_epoch,
        }

        if is_best:
            save_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt")
        elif is_highest_alignment:
            save_path = os.path.join(checkpoint_dir, f"highest_alignment_epoch_{epoch}.pt")
        else:
            save_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            
        torch.save(checkpoint, save_path)
        print(
            f"\nSaved {('best' if is_best else 'highest alignment' if is_highest_alignment else 'latest')} checkpoint at epoch {epoch + 1} to {save_path}"
        )

    def _preview_checkpoint_for_resuming(self, checkpoint_path: str):
        """
        Loads minimal info (epoch, best_val_loss) from a checkpoint to preview
        before deciding on a full resume.

        :param checkpoint_path: Path to the checkpoint file.
        :return: (wandb_run, start_epoch, best_val_loss, best_epoch) tuple of checkpoint info.
        """
        if not os.path.isfile(checkpoint_path):
            print(f"Warning: checkpoint not found at {checkpoint_path}.")
            return None, 0, float("inf"), -1

        print(f"[Preview] Loading minimal info from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        wandb_run = checkpoint.get("wandb_run", None)
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_epoch = checkpoint.get("best_epoch", -1)

        print(
            f"[Preview] Found run_id={wandb_run}, start_epoch={start_epoch}, "
            f"best_val_loss={best_val_loss}, best_epoch={best_epoch}"
        )
        return wandb_run, start_epoch, best_val_loss, best_epoch

    def _load_full_checkpoint(self, checkpoint_path: str, device, training_setup):
        """
        Loads a full checkpoint including encoders, optimizer, and scheduler states.

        :param checkpoint_path: Path to the checkpoint file.
        :param device: Device on which to map the checkpoint.
        :param training_setup: Dict containing model/optimizer references, e.g.
          {
            "video_encoder": self.video_encoder,
            "text_encoder": self.text_encoder,
            "optimizer": self.optimizer,
            "scheduler": self.lr_scheduler
          }
        :return: None. Modifies models/optimizer in-place.
        """
        if not os.path.isfile(checkpoint_path):
            print(f"Warning: checkpoint not found at {checkpoint_path}. No loading done.")
            return

        print(f"[Full Load] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        video_encoder = training_setup["video_encoder"]
        text_encoder = training_setup["text_encoder"]
        optimizer = training_setup["optimizer"]
        scheduler = training_setup["scheduler"]

        if "video_encoder" in checkpoint:
            video_encoder.load_state_dict(checkpoint["video_encoder"], strict=False)
        if "text_encoder" in checkpoint:
            text_encoder.load_state_dict(checkpoint["text_encoder"], strict=False)

        if "optimizer" in checkpoint and checkpoint["optimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    def _preprocess_inputs(self, batch: dict) -> tuple[dict, list[str]]:
        """
        Moves raw batch data (videos, texts) to GPU and returns a dictionary suitable
        for the model step, along with a list of paths or IDs for each sample.

        :param batch: Dictionary containing 'videos', 'encoded_texts', and 'paths'.
        :return: (step_inputs, paths_or_sids)
        """
        return {
            "videos": batch["videos"].to(self.device).float(),
            "input_ids": batch["encoded_texts"]["input_ids"].to(self.device),
            "attention_mask": batch["encoded_texts"]["attention_mask"].to(self.device),
        }, batch["paths"]

    def _train_step(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[dict, dict]:
        """
        One training iteration: forward → backward → (optional) clip → optimizer step →
        metric computation.

        Returns
        -------
        batch_metrics : dict
            loss, LR(s), temperature, alignment score, embedding norms …
        embeddings : dict
            'video_embeddings' | 'text_embeddings' (fp32, detached only at caller's request)
        """
        # ------------------------------------------------------------------
        # 0. housekeeping
        # ------------------------------------------------------------------
        # Clear gradients only if this is the first step in accumulation
        if self.step % self.config.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        amp_enabled = self.scaler is not None
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=amp_enabled)

        # ------------------------------------------------------------------
        # 1. forward
        # ------------------------------------------------------------------
        with autocast_ctx:
            video_emb = self.video_encoder(videos)
            text_emb  = self.text_encoder(input_ids, attention_mask)
            loss      = self.loss_fn.run(
                video_features=video_emb,
                text_features=text_emb,
                log_temp=self.log_temp,
            )

        # Scale loss by gradient accumulation steps
        scaled_loss = loss / self.config.gradient_accumulation_steps

        # ------------------------------------------------------------------
        # 2. backward
        # ------------------------------------------------------------------
        if amp_enabled:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # ------------------------------------------------------------------
        # 3. (optional) gradient clipping
        # ------------------------------------------------------------------
        max_norm = getattr(self.config, "max_grad_norm", 5.0)  # 0 or None → disabled
        if max_norm and (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gather all trainable parameters once (store in __init__ if you prefer)
            params = itertools.chain(
                self.video_encoder.parameters(),
                self.text_encoder.parameters(),
            )
            if amp_enabled:
                self.scaler.unscale_(self.optimizer)  # Unscale before clipping
            clip_grad_norm_(params, max_norm=max_norm)

        # ------------------------------------------------------------------
        # 4. optimizer step
        # ------------------------------------------------------------------
        # Only step optimizer and update scaler if this is the last step in accumulation
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            if amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        # Increment step counter
        self.step += 1

        # ------------------------------------------------------------------
        # 5. logging (rank‑0 only)
        # ------------------------------------------------------------------
        if (
            self.config.is_ref_device
            and self.wandb_wrapper.is_initialized()
        ):
            # Norms already unscaled; no extra sync needed if you run this on rank‑0
            log_gradient_norms(
                model=self.video_encoder,
                wandb_wrapper=self.wandb_wrapper,
                prefix="train/video_encoder/",
            )
            log_gradient_norms(
                model=self.text_encoder,
                wandb_wrapper=self.wandb_wrapper,
                prefix="train/text_encoder/",
            )

        # ------------------------------------------------------------------
        # 6. metrics (no_grad)
        # ------------------------------------------------------------------
        with torch.no_grad():
            video_fp32 = video_emb.float()
            text_fp32  = text_emb.float()

            embedding_norms  = compute_embedding_norms(video_fp32, text_fp32)
            alignment_score  = compute_alignment_score(video_fp32, text_fp32)

        # LR per param‑group (torch‑native ≥ 2.2 supports names)
        lr_metrics = {
            f"lr/{pg.get('name', str(i))}": pg["lr"]
            for i, pg in enumerate(self.optimizer.param_groups)
        }

        batch_metrics = {
            "loss": loss.detach().item(),
            "temperature": self.log_temp.exp().detach().item(),
            "alignment_score": alignment_score,
            **embedding_norms,
            **lr_metrics,
        }

        embeddings = {
            "video_embeddings": video_fp32,
            "text_embeddings": text_fp32,
        }

        return batch_metrics, embeddings

    def _val_step(
        self, 
        videos: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> tuple[dict, dict]:
        """
        Performs a single validation step (forward pass + metric computation).

        :param videos: Tensor of shape [B, num_clips, T, H, W, C].
        :param input_ids: Encoded text token IDs.
        :param attention_mask: Attention mask for text.
        :return: (batch_metrics, embeddings) similar to _train_step, but without backprop.
        """
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                video_features = self.video_encoder(videos)
                text_features = self.text_encoder(input_ids, attention_mask)
                loss = self.loss_fn.run(
                    video_features=video_features, 
                    text_features=text_features, 
                    log_temp=self.log_temp
                )

            embedding_norms = compute_embedding_norms(video_features, text_features)
            alignment_score = compute_alignment_score(video_features, text_features)

        metrics = {"alignment_score": alignment_score, **embedding_norms}

        return (
            {
                "loss": loss.detach().item(),
                "temperature": self.log_temp.exp().detach().item(),
                **{
                    k: v.detach().item() if torch.is_tensor(v) else v
                    for k, v in metrics.items()
                },
            },
            {
                "video_embeddings": video_features.float(),
                "text_embeddings": text_features.float(),
            },
        )

    def validate(self):
        """
        Optional method for a dedicated validation-only routine.
        Currently unimplemented.
        """
        raise NotImplementedError("Validation is not implemented for this runner")

    def inference(self):
        """
        Method for a dedicated inference.
        """
        # Load text embeddings tensor
        text_embeddings: torch.Tensor = torch.load(self.config.text_embeddings_path, weights_only=True, map_location=torch.device(self.device))
        # Load metadata
        metadata: pd.DataFrame = pd.read_parquet(self.config.metadata_path)
        
        # Create a list to store all averaged metadata
        all_averaged_metadata = []
        
        # Get the dataset object to access get_video_paths and groupby_column
        # Assuming val_loader has a dataset attribute which is an instance of VideoClipDataset
        dataset = self.val_loader.dataset
        groupby_col_name = self.config.groupby_column if hasattr(self.config, 'groupby_column') and self.config.groupby_column else "study_id"

        for batch in tqdm(
            self.val_loader, 
            desc=f"[GPU {self.device}] Running inference", 
            disable=not self.config.is_ref_device
        ):
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    video_embeddings = self.video_encoder(batch["videos"]).float()
                    
            similarity_matrix = torch.matmul(video_embeddings, text_embeddings.t())
            _, topk_indices = torch.topk(similarity_matrix, k=self.config.topk, dim=1)
            
            # Compute the average of the topk metadata rows for each video embedding
            topk_indices_np = topk_indices.cpu().numpy()  # shape: [N, topk]

            # Get SIDs from batch (in multi-video mode, 'paths' contains SIDs)
            # In single-video mode, 'paths' would contain actual file paths.
            # The logic here assumes 'paths' from the batch correctly gives the identifier needed.
            identifiers_from_batch = batch["paths"]
            
            for idx, top_k_meta_indices in enumerate(topk_indices_np):
                current_identifier = identifiers_from_batch[idx]
                
                # Get the top-k metadata rows
                topk_metadata = metadata.iloc[top_k_meta_indices]
                
                averaged_row = {}
                
                # If in multi-video mode, add groupby column and all its video filenames
                print(f"Dataset multi_video_mode status: {getattr(dataset, 'multi_video_mode', False)}")
                if getattr(dataset, 'multi_video_mode', False):
                    actual_video_filenames = dataset.get_video_paths(current_identifier) # current_identifier is SID
                    video_filenames_str = ";".join(actual_video_filenames)
                    averaged_row[groupby_col_name] = current_identifier
                    averaged_row['video_filenames'] = video_filenames_str
                else: # Single video mode
                    averaged_row['video_name'] = current_identifier # current_identifier is filename

                for column in metadata.columns:
                    if pd.api.types.is_numeric_dtype(metadata[column]):
                        # For numeric columns, compute mean
                        averaged_row[column] = topk_metadata[column].mean()
                    elif pd.api.types.is_string_dtype(metadata[column]):
                        # For string columns, get most frequent value
                        # Ensure there's at least one mode, otherwise, handle appropriately
                        modes = topk_metadata[column].mode()
                        averaged_row[column] = modes.iloc[0] if not modes.empty else None
                    else:
                        # If not numeric or string, try to get the first value or handle as error
                        # This part might need adjustment based on expected non-numeric/non-string data
                        try:
                            averaged_row[column] = topk_metadata[column].iloc[0] 
                        except IndexError:
                             averaged_row[column] = None # Or some other placeholder
                        # For strictness: raise ValueError(f"Unsupported data type for averaging/aggregation: {metadata[column].dtype} in column {column}")
                
                all_averaged_metadata.append(averaged_row)
        
        # Convert list of averaged metadata to DataFrame
        averaged_metadata_df = pd.DataFrame(all_averaged_metadata)
        
        # Reorder columns to have identifier and filenames first
        if getattr(dataset, 'multi_video_mode', False):
            cols_prefix = [groupby_col_name, 'video_filenames']
        else:
            cols_prefix = ['video_name']
        
        remaining_cols = [col for col in averaged_metadata_df.columns if col not in cols_prefix]
        ordered_cols = cols_prefix + remaining_cols
        averaged_metadata_df = averaged_metadata_df[ordered_cols]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, "averaged_metadata.csv")
        averaged_metadata_df.to_csv(output_path, index=False)
        print(f"Saved averaged metadata to: {output_path}")        
        print("Inference completed")