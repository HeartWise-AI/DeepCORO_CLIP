import math
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Dict, Tuple, List, Any, Optional

from tqdm import tqdm

from utils.enums import RunMode
from utils.ddp import DistributedUtils
from utils.config.clip_config import ClipConfig
from utils.registry import RunnerRegistry
from utils.retrieval_metrics import (
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
from utils.semantic_metrics import compute_siglip_semantic_metrics
from utils.validation_logger import (
    ValidationLogger,
    log_first_batch_comparisons,
    save_recall_summary_csv,
)
from utils.loss.typing import Loss
# NOTE: SiglipPairwiseLoss is deprecated - use unified SigLIPLoss via loss_fn
from utils.loss.multi_positive_infonce import MultiPositiveInfoNCELoss
from utils.wandb_wrapper import WandbWrapper
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from dataloaders.video_clip_dataset import VideoClipDataset
import itertools
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NonFiniteLossError(RuntimeError):
    """Raised when a non-finite loss or tensor value is detected during training."""

    def __init__(self, message: str, diagnostics: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostics: Dict[str, Any] = diagnostics or {}

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
        loss_fns: Optional[Dict[str, Loss]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        locca_decoder: Optional[nn.Module] = None,
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
        self.loss_fns: Dict[str, Loss] = loss_fns or {}
        self.loss_weights: Dict[str, float] = loss_weights or {}
        self.locca_decoder: Optional[nn.Module] = locca_decoder
        self.output_dir: str = output_dir
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1
        self.highest_alignment_score: float = float("-inf")
        self.highest_alignment_epoch: int = -1
        self.log_temp: torch.Tensor = log_temp
        self.step: int = 0  # Step counter for logging/diagnostics
        self.siglip_active: bool = bool(
            isinstance(train_loader.dataset, VideoClipDataset)
            and getattr(train_loader.dataset, "siglip_enabled", False)
        )
        self.siglip_severity_weighting: bool = bool(
            getattr(self.config, "siglip_enable_severity_weighting", False)
        )
        self.siglip_loss_mode: str = str(getattr(self.config, "loss_name", "") or "").lower()
        # Recognize all siglip* variants (siglip, siglip_pairwise, siglip2_bce, etc.)
        self.use_siglip_loss = self.siglip_active and self.siglip_loss_mode.startswith("siglip")
        # Legacy flags for backward compatibility
        self.use_siglip_pairwise = self.use_siglip_loss  # Now all siglip* uses unified loss
        self.use_siglip_infonce = self.siglip_active and self.siglip_loss_mode == "infonce_loss_ddp"
        # NOTE: siglip_pairwise_loss is no longer used - the unified SigLIPLoss
        # from contrastive.py is accessed via self.loss_fn
        self.siglip_pairwise_loss = None  # Deprecated - use loss_fn instead
        self.multi_positive_loss = (
            MultiPositiveInfoNCELoss()
            if self.use_siglip_infonce
            else None
        )
        configured_tree_weight = getattr(self.config, "tree_loss_weight", None)
        legacy_tree_weight = getattr(self.config, "main_structure_loss_weight", 0.0)
        if configured_tree_weight is None:
            configured_tree_weight = legacy_tree_weight
        configured_tree_enabled = getattr(self.config, "tree_loss_enabled", None)
        if configured_tree_enabled is None:
            tree_loss_enabled = float(configured_tree_weight or 0.0) > 0.0
        else:
            tree_loss_enabled = bool(configured_tree_enabled)
        tree_loss_weight_value = float(configured_tree_weight or 0.0)
        self.tree_loss_weight: float = tree_loss_weight_value if tree_loss_enabled else 0.0
        self.tree_loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        self._last_grad_norm: float = 0.0
        self._last_video_grad_norm: float = 0.0
        self._last_text_grad_norm: float = 0.0
        self.progress_log_interval: int = max(
            1, int(getattr(self.config, "progress_log_interval", 200))
        )
        if getattr(self.config, "gradient_accumulation_steps", 1) != 1:
            if self.config.is_ref_device:
                print(
                    "[Runner] gradient_accumulation_steps>1 is no longer supported; "
                    "defaulting to 1 (micro-batching disabled)."
                )
            self.config.gradient_accumulation_steps = 1
        video_max_grad = getattr(self.config, "video_max_grad_norm", None)
        if video_max_grad is None:
            video_max_grad = getattr(self.config, "max_grad_norm", 1.0)
        self.video_max_grad_norm: float = (
            float(video_max_grad) if video_max_grad is not None else 0.0
        )
        text_max_grad = getattr(self.config, "text_max_grad_norm", None)
        self.text_max_grad_norm: float = (
            float(text_max_grad) if text_max_grad is not None else 0.0
        )

        # Debug mode for detailed numerical stability logging
        self.debug_numerical: bool = bool(getattr(self.config, "debug", False))
        self._debug_log_interval: int = 50  # Log every N steps when debug is enabled

        # LocCa captioning support
        self._locca_enabled: bool = (
            self.locca_decoder is not None
            and any("locca" in name for name in self.loss_fns.keys())
        )
        if self._locca_enabled and self.config.is_ref_device:
            print("[Runner] LocCa captioning enabled - will use SigLIP texts as captions")

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

    def _prepare_caption_tokens(
        self,
        siglip_batch: Optional[dict],
        video_paths: Optional[List[str]] = None,
        dataset: Optional[Any] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]]]:
        """
        Prepare caption tokens for LocCa decoder from SigLIP batch.

        Concatenates ALL positive texts for each video as the caption target,
        ordered by severity (most severe first). This creates comprehensive
        reports for captioning training.

        Args:
            siglip_batch: Dict with input_ids, attention_mask, positive_mask
            video_paths: List of video file paths for looking up captions
            dataset: Dataset with siglip support for building reports

        Returns:
            Tuple of (caption_input_ids, caption_labels, caption_attention_mask, caption_texts)
            where caption_texts is a list of decoded caption strings for debugging.
        """
        if not self._locca_enabled or siglip_batch is None:
            return None, None, None, None

        device = siglip_batch.get("input_ids").device if siglip_batch.get("input_ids") is not None else self.device

        # Try to build concatenated reports from all positives using dataset's siglip support
        if video_paths and dataset is not None and hasattr(dataset, "siglip") and dataset.siglip is not None:
            siglip = dataset.siglip
            caption_texts: List[str] = []
            all_input_ids: List[torch.Tensor] = []
            all_attention_masks: List[torch.Tensor] = []

            # Get video keys from paths using dataset's mapping
            for path in video_paths:
                video_idx = dataset.video_path_to_idx.get(str(path))
                video_key = None
                if video_idx is not None and video_idx < len(dataset.video_ids):
                    video_key = dataset.video_ids[video_idx]

                if video_key:
                    # Build concatenated report from all positive texts for this video
                    report_data = siglip.build_report_tokens(
                        video_key,
                        separator=" ",
                        order_by_severity=True,
                        device=device,
                    )
                    caption_texts.append(report_data.get("report_text", "No findings."))
                    all_input_ids.append(report_data["input_ids"])
                    all_attention_masks.append(report_data["attention_mask"])
                else:
                    # Fallback - use "No findings" placeholder
                    tokenizer = getattr(dataset, "tokenizer", None)
                    if tokenizer:
                        encoding = tokenizer(
                            "No findings.",
                            padding="max_length",
                            max_length=dataset.max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        caption_texts.append("No findings.")
                        all_input_ids.append(encoding["input_ids"].squeeze(0).to(device))
                        all_attention_masks.append(encoding["attention_mask"].squeeze(0).to(device))

            if all_input_ids:
                caption_tokens = torch.stack(all_input_ids, dim=0)  # [B, seq_len]
                attention_mask = torch.stack(all_attention_masks, dim=0)  # [B, seq_len]
            else:
                return None, None, None, None
        else:
            # Fallback: use first positive per video (original behavior)
            siglip_input_ids = siglip_batch.get("input_ids")
            siglip_attention_mask = siglip_batch.get("attention_mask")
            positive_mask = siglip_batch.get("positive_mask")
            caption_texts = None

            if siglip_input_ids is None:
                return None, None, None, None

            if positive_mask is not None and siglip_input_ids.ndim == 2:
                B = positive_mask.shape[0]
                first_positive_idx = positive_mask.argmax(dim=1)
                has_positive = positive_mask.sum(dim=1) > 0
                if not has_positive.all():
                    first_positive_idx = torch.where(
                        has_positive,
                        first_positive_idx,
                        torch.zeros_like(first_positive_idx)
                    )
                caption_tokens = siglip_input_ids[first_positive_idx]
                if siglip_attention_mask is not None:
                    attention_mask = siglip_attention_mask[first_positive_idx]
                else:
                    attention_mask = None
            elif siglip_input_ids.ndim == 3:
                caption_tokens = siglip_input_ids[:, 0, :]
                if siglip_attention_mask is not None and siglip_attention_mask.ndim == 3:
                    attention_mask = siglip_attention_mask[:, 0, :]
                else:
                    attention_mask = siglip_attention_mask
            else:
                caption_tokens = siglip_input_ids
                attention_mask = siglip_attention_mask

        # Autoregressive training setup:
        # BERT-style tokenizers (PubMedBERT) already prepend [CLS] which acts as BOS.
        # Tokenized sequence: [CLS, t1, t2, ..., tN, SEP, PAD, ...]
        #
        # Standard next-token prediction with causal masking:
        # - Input:  tokens[:-1] = [CLS, t1, t2, ..., tN, SEP, PAD[:-1]...]
        # - Labels: tokens[1:]  = [t1, t2, ..., tN, SEP, PAD[1:]...]
        #
        # This gives proper alignment:
        # - Position 0: input=[CLS] (BOS), predict=[t1]
        # - Position 1: input=[t1], predict=[t2]
        # - ...
        # - Position N: input=[tN], predict=[SEP] (EOS)
        # - Padding positions: labels=-100 (ignored in loss)
        #
        # For GPT-2 style tokenizers without BOS, prepend eos_token_id as BOS
        # (GPT-2 uses EOS as BOS). This is handled by locca_decoder's bos_token_id fallback.

        caption_input_ids = caption_tokens[:, :-1].contiguous()
        caption_labels = caption_tokens[:, 1:].contiguous()

        # Preserve padding: set labels to -100 where attention_mask is 0
        # This ensures padding positions are ignored in cross-entropy loss
        caption_attention_mask = None
        if attention_mask is not None:
            caption_attention_mask = attention_mask[:, :-1].contiguous()
            shifted_mask = attention_mask[:, 1:].contiguous()
            caption_labels = caption_labels.clone()
            caption_labels[shifted_mask == 0] = -100  # ignore_index for CrossEntropyLoss

        return caption_input_ids, caption_labels, caption_attention_mask, caption_texts

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

                train_sampler = getattr(self.train_loader, "sampler", None)
                if isinstance(train_sampler, DistributedUtils.DS.DistributedSampler):
                    train_sampler.set_epoch(epoch)
                val_sampler = getattr(self.val_loader, "sampler", None)
                if isinstance(val_sampler, DistributedUtils.DS.DistributedSampler):
                    val_sampler.set_epoch(epoch)

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
                    mode=RunMode.VALIDATE, 
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

                # ------------------------------------------------------------------
                # Memory cleanup to avoid GPU OOM across epochs
                # ------------------------------------------------------------------
                # Use locals() checks to prevent NameError in static analysis
                for _var in [
                    'similarity_matrix',
                    'global_video_feats_norm',
                    'unique_text_embeddings_norm',
                    'unique_text_embeddings_tensor',
                    'global_video_feats',
                ]:
                    if _var in locals():
                        del locals()[_var]
                torch.cuda.empty_cache()
                import gc
                gc.collect()

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
        assert mode in [RunMode.TRAIN, RunMode.VALIDATE]

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
        all_tree_logits_local: List[torch.Tensor] = []
        all_paths_local: List[str] = []
        all_ground_truth_reports_local: List[str] = []
        # Store vision_features for LocCa caption generation
        all_vision_features_local: List[torch.Tensor] = []
        all_caption_texts_local: List[str] = []  # Ground truth captions

        tqdm_desc = f"[GPU {self.device}] Running {mode} Epoch {epoch + 1}"
        
        # Determine the iterator: tqdm-wrapped or raw dataloader
        iterator_obj: Any = dataloader # Use Any for iterator_obj as it can be DataLoader or tqdm
        if self.config.is_ref_device or (self.device == 0):
            iterator_obj = tqdm(dataloader, desc=tqdm_desc)
        batch_count: int = 0
        for iteration_idx, batch in enumerate(iterator_obj, start=1): # Use iterator_obj
            if batch["videos"] is None or batch["encoded_texts"] is None:
                continue

            step_inputs, paths_or_sids = self._preprocess_inputs(batch)
            siglip_payload = self._build_siglip_payload(dataset, paths_or_sids)
            if siglip_payload is not None:
                step_inputs["siglip_batch"] = siglip_payload

            # Pass video_paths and dataset for LocCa caption building
            if self._locca_enabled:
                step_inputs["video_paths"] = paths_or_sids
                step_inputs["dataset"] = dataset
                step_inputs["is_first_batch"] = (iteration_idx == 1 and epoch == 0)

            try:
                batch_metrics, embeddings = step_fn(**step_inputs)
            except NonFiniteLossError as err:
                error_msg = (
                    f"NaN loss detected in {mode} at epoch {epoch + 1}, "
                    f"batch {iteration_idx}"
                )
                if self.config.is_ref_device:
                    print(f"\nERROR: {error_msg}")
                    if err.diagnostics:
                        print(f"Diagnostics: {err.diagnostics}")
                raise RuntimeError(error_msg) from err

            # Check for NaN loss
            loss_value = float(batch_metrics["loss"])
            if not math.isfinite(loss_value):
                error_msg = f"NaN loss detected in {mode} at epoch {epoch + 1}, batch {iteration_idx}"
                if self.config.is_ref_device:
                    print(f"\nERROR: {error_msg}")
                    print(f"Diagnostics: {{'loss': {loss_value}}}")
                raise RuntimeError(error_msg)

            # Store embeddings on CPU
            batch_video_emb = embeddings["video_embeddings"]
            batch_text_emb = embeddings["text_embeddings"]
            batch_size = batch_video_emb.shape[0]
            all_video_embeddings_local.append(batch_video_emb.cpu())
            all_text_embeddings_local.append(batch_text_emb.cpu())

            # Store vision_features for LocCa caption generation (validation only, limited samples)
            if mode == RunMode.VALIDATE and self._locca_enabled:
                vis_feats = embeddings.get("vision_features")
                if vis_feats is not None and len(all_vision_features_local) < 50:  # Limit to 50 samples
                    all_vision_features_local.append(vis_feats.cpu())
                    # Get ground truth captions for these samples
                    caption_input_ids, caption_labels, caption_attn_mask, caption_texts = self._prepare_caption_tokens(
                        siglip_batch=siglip_payload,
                        video_paths=paths_or_sids,
                        dataset=dataset,
                    )
                    if caption_texts:
                        all_caption_texts_local.extend(caption_texts[:batch_size])

            if mode == RunMode.VALIDATE:
                tree_logits_batch = embeddings.get("tree_logits")
                if tree_logits_batch is not None:
                    all_tree_logits_local.append(tree_logits_batch.detach().cpu())

            # Use dataset.multi_video_mode for consistency
            # paths_or_sids contains SIDs in multi-video mode, and direct file paths in single-video mode.
            # all_paths_local will store these identifiers directly.
            all_paths_local.extend(paths_or_sids)

            all_ground_truth_reports_local.extend(dataset.get_reports(paths_or_sids)) # Now type-safe

            # accumulate metrics
            for k, v in batch_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)
                if (
                    self.wandb_wrapper.is_initialized()
                    and self.config.is_ref_device
                    and (mode == RunMode.TRAIN or mode == RunMode.VALIDATE)
                ):
                    mode_name = mode.value if isinstance(mode, RunMode) else str(mode)
                    wandb_key = f"{mode_name}/{k}"
                    self.wandb_wrapper.log({wandb_key: float(v)})

            total_loss += float(batch_metrics["loss"])
            batch_count += 1


            if (self.config.is_ref_device or (self.device == 0)) and mode == RunMode.TRAIN:
                postfix_data = {
                    f"{mode}_loss": f"{batch_metrics['loss']:.4f}", # Use mode variable
                    "avg_loss": f"{(total_loss / batch_count):.4f}",
                    "grad_v": f"{self._last_video_grad_norm:.2f}",
                    "grad_t": f"{self._last_text_grad_norm:.2f}",
                }
                # Add individual loss values for each head (e.g., loss/siglip_pairwise, loss/locca_caption)
                for key, val in batch_metrics.items():
                    if key.startswith("loss/"):
                        # Create short name for tqdm display (e.g., "siglip_pairwise" -> "sig", "locca_caption" -> "loc")
                        short_name = key.replace("loss/", "")
                        if "siglip" in short_name:
                            short_name = "sig"
                        elif "locca" in short_name:
                            short_name = "loc"
                        elif "clip" in short_name or "contrastive" in short_name:
                            short_name = "clip"
                        else:
                            short_name = short_name[:4]  # First 4 chars
                        postfix_data[short_name] = f"{val:.4f}"
                if "train/Recall@5" in batch_metrics:
                    postfix_data["recall@5"] = f"{batch_metrics['train/Recall@5']:.3f}"
                iterator_obj.set_postfix(postfix_data)
                if hasattr(iterator_obj, 'set_postfix_str'):
                    iterator_obj.set_postfix_str(
                        " | ".join(f"{k}:{v}" for k, v in postfix_data.items())
                    )
                if batch_count % self.progress_log_interval == 0:
                    log_line = (
                        f"[Epoch {epoch + 1} | Step {batch_count}] "
                        + " | ".join(f"{k}={v}" for k, v in postfix_data.items())
                    )
                    tqdm.write(log_line)
                    if hasattr(iterator_obj, 'write'):
                        iterator_obj.write(log_line)

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

        local_tree_logits: Optional[torch.Tensor] = None
        if len(all_tree_logits_local) > 0:
            local_tree_logits = torch.cat(all_tree_logits_local, dim=0).to(self.device)

        # 2) gather across GPUs
        global_video_feats: torch.Tensor = self._gather_tensor_along_batch(local_video_feats, self.world_size)
        global_tree_logits: Optional[torch.Tensor] = None
        if local_tree_logits is not None and local_tree_logits.numel() > 0:
            gathered_tree = self._gather_tensor_along_batch(local_tree_logits, self.world_size)
            global_tree_logits = gathered_tree

        # 3) gather paths & reports
        global_paths: list[str] = self._gather_strings_across_gpus(
            all_paths_local, self.world_size, device=local_video_feats.device
        )
        global_reports: list[str] = self._gather_strings_across_gpus(
            all_ground_truth_reports_local, self.world_size, device=local_video_feats.device
        )

        tree_predictions: Optional[List[Optional[str]]] = None
        if (
            global_tree_logits is not None
            and global_tree_logits.numel() == len(global_paths)
            and len(global_paths) > 0
        ):
            logits_cpu = global_tree_logits.detach().cpu().tolist()
            tree_predictions = [
                (None if math.isnan(value) else ("right" if value >= 0.0 else "left"))
                for value in logits_cpu
            ]

        # Optionally compute NxM retrieval metrics on rank 0
        retrieval_metrics: dict[str, float] = {}
        if mode == RunMode.VALIDATE and len(global_video_feats) > 0:
            print(
                f"[DEBUG rank={self.device}] Starting retrieval computation with {global_video_feats.shape[0]} videos."
            )

        text_segments: Optional[List[Optional[str]]] = None
        text_trees: Optional[List[Optional[str]]] = None

        if mode == RunMode.VALIDATE:
            if self.config.is_ref_device and len(global_video_feats) > 0:
                # Check if we're using SigLIP-style multi-positive training
                all_text_ids: List[str] = []
                use_siglip_texts = (
                    hasattr(dataset, 'siglip_enabled') and
                    dataset.siglip_enabled and
                    dataset.siglip is not None and
                    hasattr(dataset.siglip, 'text_lookup') and
                    len(dataset.siglip.text_lookup) > 0
                )

                if use_siglip_texts:
                    # SigLIP mode: Load all unique texts from texts.csv
                    print(f"[DEBUG rank={self.device}] Using SigLIP mode - loading all texts from texts.csv")

                    # Get all text IDs and their corresponding prompt texts
                    all_text_ids = sorted(dataset.siglip.text_lookup.keys())
                    unique_texts: List[str] = [
                        dataset.siglip.text_lookup[tid].get("prompt_text", "")
                        for tid in all_text_ids
                    ]
                    siglip_lookup = dataset.siglip.text_lookup if dataset.siglip else {}
                    normalize_tree = getattr(dataset, "_normalize_tree_key", None)
                    text_segments = []
                    text_trees = []
                    for tid in all_text_ids:
                        meta = siglip_lookup.get(tid, {})
                        text_segments.append(meta.get("segment"))
                        raw_tree = meta.get("tree")
                        normalized_tree = None
                        if callable(normalize_tree):
                            normalized_tree = normalize_tree(raw_tree)
                        elif isinstance(raw_tree, str):
                            stripped = raw_tree.strip().lower()
                            normalized_tree = stripped if stripped else None
                        text_trees.append(normalized_tree)
                    text_id_to_index: Dict[str, int] = {tid: idx for idx, tid in enumerate(all_text_ids)}
                    # Also create text_to_index for save_retrieval_results compatibility
                    text_to_index: Dict[str, int] = {text: idx for idx, text in enumerate(unique_texts)}

                    print(f"[DEBUG rank={self.device}] Found {len(unique_texts)} unique texts out of {len(all_text_ids)} text IDs.")

                    # Step 2: Build ground truth matrix (videos x texts) with multiple positives per video
                    ground_truth_matrix = torch.zeros(
                        len(global_video_feats),
                        len(unique_texts),
                        dtype=torch.float32,
                        device=self.device
                    )

                    # For each video, mark all its positive texts
                    for video_idx, video_path in enumerate(global_paths):
                        # Get the dataset index for this video
                        dataset_idx = dataset.video_path_to_idx.get(str(video_path))
                        if dataset_idx is None:
                            continue

                        # Get positive text IDs for this video
                        if dataset_idx < len(dataset.video_positive_texts):
                            positive_pairs = dataset.video_positive_texts[dataset_idx]
                            for text_id, weight in positive_pairs:
                                if text_id in text_id_to_index:
                                    text_idx = text_id_to_index[text_id]
                                    ground_truth_matrix[video_idx, text_idx] = 1.0

                    # For single-positive metrics, use SEVERITY-PRIORITIZED selection
                    # (prefer severe/critical findings over normal segments)
                    severity_order = {
                        'severe': 0, 'critical': 0, 'cto': 0,
                        'moderate': 1,
                        'mild': 2,
                        'normal': 3,
                    }

                    # Build text_idx -> severity mapping
                    text_idx_to_severity: Dict[int, str] = {}
                    for tid, idx in text_id_to_index.items():
                        meta = siglip_lookup.get(tid, {})
                        sev = meta.get("disease_severity", "normal")
                        if not isinstance(sev, str):
                            sev = "normal"
                        text_idx_to_severity[idx] = sev.lower().strip()

                    ground_truth_indices: torch.Tensor = torch.zeros(
                        len(global_video_feats), dtype=torch.long, device=self.device
                    )
                    for video_idx in range(len(global_video_feats)):
                        positive_indices = torch.where(ground_truth_matrix[video_idx] > 0)[0]
                        if len(positive_indices) > 0:
                            # Sort by severity: severe first, then moderate, mild, normal
                            sorted_positives = sorted(
                                positive_indices.tolist(),
                                key=lambda i: severity_order.get(text_idx_to_severity.get(i, "normal"), 3)
                            )
                            ground_truth_indices[video_idx] = sorted_positives[0]

                elif len(global_reports) > 0:
                    # Original mode: Use concatenated reports per video
                    unique_texts: List[str] = sorted(set(global_reports))
                    if len(unique_texts) == 0:
                        print(f"[DEBUG rank={self.device}] No unique texts found; skipping retrieval metrics.")
                        retrieval_metrics = self._zero_retrieval_metrics()
                        unique_texts = []
                    else:
                        text_to_index: Dict[str, int] = {text: idx for idx, text in enumerate(unique_texts)}
                        text_segments = [None for _ in unique_texts]

                        # Step 2: Get ground truth indices for each video
                        ground_truth_indices: torch.Tensor = torch.tensor(
                            [text_to_index[text] for text in global_reports],
                            device=self.device
                        )
                        ground_truth_matrix = None  # Not used in this mode

                        print(f"[DEBUG rank={self.device}] Found {len(unique_texts)} unique texts out of {len(global_reports)} total.")
                        all_text_ids = list(unique_texts)
                else:
                        print(f"[DEBUG rank={self.device}] No reports or SigLIP texts found; skipping retrieval metrics.")
                        retrieval_metrics = self._zero_retrieval_metrics()
                        unique_texts = []
                        all_text_ids = []
                        text_segments = None

                # Continue with encoding if we have texts
                if len(unique_texts) > 0:
                    
                    # Step 3: Encode unique texts
                    unique_text_embeddings_list: List[torch.Tensor] = []
                    batch_size: int = 64  # Process in batches to avoid OOM
                    
                    self.text_encoder.eval()
                    with torch.no_grad():
                        for start_idx in range(0, len(unique_texts), batch_size):
                            end_idx = min(start_idx + batch_size, len(unique_texts))
                            text_batch = unique_texts[start_idx:end_idx]
                            
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
                            text_embs = self.text_encoder(
                                encoded["input_ids"],
                                encoded["attention_mask"]
                            )
                            unique_text_embeddings_list.append(text_embs)
                    
                    if len(unique_text_embeddings_list) == 0:
                        print(f"[DEBUG rank={self.device}] No text embeddings produced; skipping retrieval metrics.")
                        retrieval_metrics = self._zero_retrieval_metrics()
                    else:
                        unique_text_embeddings_tensor: torch.Tensor = torch.cat(unique_text_embeddings_list, dim=0)
                        
                        # Step 4: Normalize embeddings
                        global_video_feats_norm: torch.Tensor = nn.functional.normalize(global_video_feats, dim=1)
                        unique_text_embeddings_norm: torch.Tensor = nn.functional.normalize(unique_text_embeddings_tensor, dim=1)
                        
                        # Step 5: Compute NxM similarity matrix
                        similarity_matrix: torch.Tensor = torch.matmul(global_video_feats_norm, unique_text_embeddings_norm.t())
                        print(
                            f"[DEBUG rank={self.device}] Computed NxM sim matrix with shape={similarity_matrix.shape}"
                        )

                        semantic_metrics = compute_siglip_semantic_metrics(
                            similarity_matrix=similarity_matrix.detach().cpu(),
                            sample_identifiers=global_paths,
                            dataset=dataset,
                            all_text_ids=all_text_ids,
                            top_tree_k=5,
                            top_segment_k=15,
                        )
                        retrieval_metrics.update(semantic_metrics)
                        
                        if self.wandb_wrapper.is_initialized():
                            # Pass multi-positive ground truth matrix and text lookup for detailed logging
                            gt_matrix_for_logging = ground_truth_matrix if use_siglip_texts and 'ground_truth_matrix' in locals() else None
                            text_lookup_for_logging = siglip_lookup if use_siglip_texts and 'siglip_lookup' in locals() else None

                            log_best_worst_retrievals(
                                similarity_matrix=similarity_matrix,
                                all_paths=global_paths,
                                unique_texts=unique_texts,
                                ground_truth_indices=ground_truth_indices,
                                epoch=epoch,
                                wandb_wrapper=self.wandb_wrapper,
                                dataset_obj=dataset,
                                text_segments=text_segments,
                                text_trees=text_trees,
                                sample_tree_predictions=tree_predictions,
                                ground_truth_matrix=gt_matrix_for_logging,
                                all_text_ids=all_text_ids if 'all_text_ids' in locals() else None,
                                text_lookup=text_lookup_for_logging,
                            )

            # Save retrieval results with mapping
                        save_retrieval_results(
                            similarity_matrix=similarity_matrix,
                            all_identifiers=global_paths,
                            all_ground_truth_reports=global_reports,
                            report_to_global_index=text_to_index,
                            epoch=epoch,
                            output_dir=self.output_dir,
                            dataset_obj=dataset,
                            ground_truth_indices_override=ground_truth_indices,
                            top_k_predictions=int(getattr(self.config, "topk", 5) or 5),
                            text_segments=text_segments,
                            text_trees=text_trees,
                            sample_tree_predictions=tree_predictions,
                            index_to_texts=unique_texts,
                            index_to_text_ids=all_text_ids,
                            json_top_k=15,
                        )
                        # Compute retrieval metrics (use ground_truth_matrix if available for multi-positive)
                        if use_siglip_texts and 'ground_truth_matrix' in locals():
                            recall_metrics: Dict[str, float] = compute_recall_at_k(
                                similarity_matrix,
                                ground_truth_matrix=ground_truth_matrix,
                                k_values=self.config.recall_k
                            )
                        else:
                            recall_metrics: Dict[str, float] = compute_recall_at_k(
                                similarity_matrix, ground_truth_indices, k_values=self.config.recall_k
                            )
                        mrr_score_dict: Dict[str, float] = compute_mrr(similarity_matrix, ground_truth_indices)
                        map_score: float = compute_map(similarity_matrix, ground_truth_indices)
                        median_rank_score: float = compute_median_rank(similarity_matrix, ground_truth_indices)
                        ndcg_scores_dict: Dict[str, float] = compute_ndcg_at_k(
                            similarity_matrix, ground_truth_indices, k_values=self.config.ndcg_k
                        )

                        retrieval_metrics.update(recall_metrics)
                        retrieval_metrics.update(mrr_score_dict)
                        retrieval_metrics.update(ndcg_scores_dict)
                        retrieval_metrics["MAP"] = map_score
                        retrieval_metrics["MedianRank_V2T"] = median_rank_score

                        # === Validation Logging: Save detailed CSV and text comparisons ===
                        try:
                            # Save recall summary to CSV (append per epoch)
                            save_recall_summary_csv(
                                recall_metrics=recall_metrics,
                                epoch=epoch,
                                output_dir=self.output_dir,
                                additional_metrics={
                                    "MAP": map_score,
                                    "MRR_V2T": mrr_score_dict.get("MRR_V2T", 0.0),
                                    "MedianRank_V2T": median_rank_score,
                                    **ndcg_scores_dict,
                                },
                                append=True,
                            )

                            # Log first batch text comparisons (GT vs Predicted)
                            if len(unique_texts) > 0 and global_video_feats.shape[0] > 0:
                                print(f"\n[Epoch {epoch}] === Text Comparison Samples (First 5) ===")
                                log_first_batch_comparisons(
                                    video_features=global_video_feats[:min(10, global_video_feats.shape[0])],
                                    text_features=unique_text_embeddings_tensor,
                                    video_ids=global_paths[:min(10, len(global_paths))],
                                    unique_texts=unique_texts,
                                    ground_truth_indices=ground_truth_indices[:min(10, len(ground_truth_indices))],
                                    epoch=epoch,
                                    output_dir=self.output_dir,
                                    max_samples=5,
                                )
                        except Exception as e:
                            print(f"[ValidationLogger] Warning: Could not log validation details: {e}")
                        # === End Validation Logging ===

                        # === LocCa Caption Generation ===
                        if self._locca_enabled and self.config.is_ref_device:
                            try:
                                self._save_locca_captions(
                                    vision_features_list=all_vision_features_local,
                                    ground_truth_captions=all_caption_texts_local,
                                    video_paths=global_paths[:len(all_caption_texts_local)],
                                    epoch=epoch,
                                    dataset=dataset,
                                    max_samples=50,
                                )
                            except Exception as e:
                                print(f"[LocCa] Warning: Could not generate captions: {e}")
                        # === End LocCa Caption Generation ===

            else:
                retrieval_metrics = self._zero_retrieval_metrics()
        else:
            retrieval_metrics = {}

        epoch_metrics.update(retrieval_metrics)
        # 4) reduce final epoch metrics across ranks
        gathered_metrics: dict[str, float] = {}
        mode_prefix = mode.value if isinstance(mode, RunMode) else str(mode)
        for k, v in epoch_metrics.items():
            gathered_metrics[f"{mode_prefix}/{k}"] = self._maybe_reduce_metric(k, v)

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

    def _zero_retrieval_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for k_val in self.config.recall_k:
            metrics[f"Recall@{k_val}"] = 0.0
        metrics.update({
            "MRR": 0.0,
            "MAP": 0.0,
            "MedianRank_V2T": 0.0,
        })
        for k_val in self.config.ndcg_k:
            metrics[f"NDCG@{k_val}"] = 0.0
        return metrics

    def _save_locca_captions(
        self,
        vision_features_list: List[torch.Tensor],
        ground_truth_captions: List[str],
        video_paths: List[str],
        epoch: int,
        dataset: Any,
        max_samples: int = 50,
    ) -> None:
        """
        Generate and save LocCa captions for validation samples.

        Args:
            vision_features_list: List of vision features tensors [B, L, D]
            ground_truth_captions: List of ground truth caption strings
            video_paths: List of video file paths
            epoch: Current epoch
            dataset: Dataset with tokenizer for decoding
            max_samples: Maximum number of samples to generate
        """
        if not self._locca_enabled or self.locca_decoder is None:
            return

        if not vision_features_list:
            return

        import json

        # Get tokenizer for decoding
        tokenizer = getattr(dataset, "tokenizer", None)
        if tokenizer is None:
            print("[LocCa] Warning: No tokenizer found, skipping caption generation")
            return

        # Concatenate vision features
        all_vis_feats = torch.cat(vision_features_list, dim=0)[:max_samples]
        num_samples = min(all_vis_feats.shape[0], len(ground_truth_captions), len(video_paths))

        # Get LocCa module (handle DDP wrapper)
        locca_module = self.locca_decoder.module if hasattr(self.locca_decoder, 'module') else self.locca_decoder

        # Generate captions
        caption_records = []
        self.locca_decoder.eval()
        with torch.no_grad():
            # Move to device and ensure 3D
            vis_feats = all_vis_feats[:num_samples].to(self.device)
            if vis_feats.ndim == 2:
                vis_feats = vis_feats.unsqueeze(1)

            # Generate in batches of 8
            batch_size = 8
            all_generated = []
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_feats = vis_feats[start_idx:end_idx]

                generated_ids = locca_module.generate(
                    vision_features=batch_feats,
                    max_length=100,
                    temperature=0.8,
                    top_k=50,
                )
                all_generated.append(generated_ids.cpu())

            # Decode generated captions
            generated_ids = torch.cat(all_generated, dim=0)
            for i in range(num_samples):
                gen_ids = generated_ids[i].tolist()
                # Remove special tokens and decode
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                gt_text = ground_truth_captions[i] if i < len(ground_truth_captions) else ""
                video_path = video_paths[i] if i < len(video_paths) else ""

                caption_records.append({
                    "video_path": video_path,
                    "ground_truth": gt_text,
                    "generated": gen_text,
                    "epoch": epoch,
                })

        # Save to JSON
        output_path = os.path.join(self.output_dir, f"locca_captions_epoch{epoch}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(caption_records, f, indent=2)

        # Also save human-readable text file
        txt_path = os.path.join(self.output_dir, f"locca_captions_epoch{epoch}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"LocCa Generated Captions - Epoch {epoch}\n")
            f.write("=" * 80 + "\n\n")
            for i, record in enumerate(caption_records[:20]):  # First 20 for readability
                f.write(f"--- Sample {i+1} ---\n")
                f.write(f"Video: {record['video_path']}\n")
                f.write(f"GT:    {record['ground_truth']}\n")
                f.write(f"GEN:   {record['generated']}\n\n")

        print(f"[LocCa] Saved {len(caption_records)} generated captions to {output_path}")

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

    def _build_siglip_payload(
        self,
        dataset: VideoClipDataset,
        sample_ids: List[str],
    ) -> Optional[dict]:
        if not getattr(dataset, "siglip_enabled", False):
            return None
        if not hasattr(dataset, "build_siglip_batch"):
            return None
        siglip_batch = dataset.build_siglip_batch(sample_ids)
        if not siglip_batch:
            return None
        payload = {}
        for key in ("input_ids", "attention_mask", "positive_mask", "positive_weights"):
            tensor = siglip_batch.get(key)
            if tensor is None:
                continue
            payload[key] = tensor.to(self.device, non_blocking=True)
        return payload

    def _scaled_logits(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine-similarity logits with robust normalization, nan guards, and a CLIP-style cap.
        """
        v = F.normalize(v, dim=1, eps=1e-6)
        t = F.normalize(t, dim=1, eps=1e-6)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        logits = torch.matmul(v.float(), t.float().t()).float()

        # Clamp both lower and upper bounds to prevent underflow/overflow
        # Lower bound: exp(-4.6) ≈ 0.01, Upper bound: exp(4.6) ≈ 100
        logit_scale = torch.clamp((-self.log_temp).float(), min=math.log(0.01), max=math.log(100.0))
        scale = torch.exp(logit_scale)
        return logits * scale

    def _compute_multi_loss(
        self,
        video_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        siglip_batch: Optional[dict] = None,
        caption_input_ids: Optional[torch.Tensor] = None,
        caption_labels: Optional[torch.Tensor] = None,
        caption_attention_mask: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        caption_texts: Optional[List[str]] = None,
        is_first_batch: bool = False,
        dataset: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss from all registered losses.

        Args:
            video_embeddings: [B, D] pooled video embeddings for contrastive loss
            text_embeddings: [B, D] text embeddings for contrastive loss
            siglip_batch: SigLIP batch with multi-positive data
            caption_input_ids: [B, L-1] input tokens for captioning (shifted)
            caption_labels: [B, L-1] target tokens for captioning (shifted, -100 for padding)
            caption_attention_mask: [B, L-1] attention mask for decoder (1=attend, 0=ignore)
            vision_features: [B, N, D] pre-pooled vision features for cross-attention

        Returns:
            total_loss: Combined weighted loss
            loss_breakdown: Dict of individual loss values for logging
        """
        total_loss = video_embeddings.new_tensor(0.0)
        loss_breakdown: Dict[str, float] = {}

        for loss_name, loss_fn in self.loss_fns.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            if weight <= 0:
                continue

            loss_val = video_embeddings.new_tensor(0.0)

            # Contrastive losses (siglip, clip, etc.)
            if loss_name.startswith("siglip") or loss_name in ("clip", "contrastive"):
                if siglip_batch and self.siglip_active:
                    # Use SigLIP batch for multi-positive
                    loss_val = self._compute_siglip_pairwise_loss(
                        video_embeddings=video_embeddings,
                        siglip_batch=siglip_batch,
                    )
                else:
                    loss_val = loss_fn.run(
                        video_features=video_embeddings,
                        text_features=text_embeddings,
                        log_temp=self.log_temp,
                    )

            # LocCa captioning loss
            elif loss_name.startswith("locca") and self.locca_decoder is not None:
                if caption_input_ids is not None and caption_labels is not None and vision_features is not None:
                    # Ensure vision_features is 3D [B, L_vision, D] for cross-attention
                    # If it's 2D [B, D] (pooled embedding), unsqueeze to [B, 1, D]
                    vis_feats = vision_features
                    if vis_feats.ndim == 2:
                        vis_feats = vis_feats.unsqueeze(1)  # [B, D] -> [B, 1, D]

                    # Forward through LocCa decoder with causal masking
                    # The decoder applies causal self-attention mask internally
                    # We pass attention_mask to handle padding in the input
                    logits = self.locca_decoder(
                        input_ids=caption_input_ids,
                        vision_features=vis_feats,
                        attention_mask=caption_attention_mask,
                    )
                    # LocCa loss computes cross-entropy, ignoring -100 in labels
                    loss_val = loss_fn.loss_type(
                        logits=logits,
                        targets=caption_labels,
                    )

                    # Debug output for first batch: show target caption and model prediction
                    if is_first_batch and self.config.is_ref_device:
                        print("\n" + "=" * 80)
                        print("[LocCa Debug - First Batch]")
                        print("=" * 80)

                        # Get tokenizer for decoding
                        tokenizer = None
                        if dataset is not None and hasattr(dataset, "tokenizer"):
                            tokenizer = dataset.tokenizer

                        # Show first sample's target caption
                        if caption_texts and len(caption_texts) > 0:
                            print(f"\n[Target Caption (Video 0)]:")
                            print(f"  {caption_texts[0][:200]}..." if len(caption_texts[0]) > 200 else f"  {caption_texts[0]}")

                        # Get input and predicted token IDs for comparison
                        input_ids_sample = caption_input_ids[0]  # [seq_len]
                        label_ids_sample = caption_labels[0]  # [seq_len] - the target (shifted by 1)
                        pred_tokens = logits[0].argmax(dim=-1)  # [seq_len]

                        # Check if prediction matches input (accidental copy bug)
                        match_input = (pred_tokens == input_ids_sample).float().mean().item()
                        match_labels = (pred_tokens == label_ids_sample).float().mean().item()

                        print(f"\n[Token ID Comparison (first 20 tokens)]:")
                        print(f"  Input IDs:     {input_ids_sample[:20].tolist()}")
                        print(f"  Label IDs:     {label_ids_sample[:20].tolist()}")
                        print(f"  Predicted IDs: {pred_tokens[:20].tolist()}")
                        print(f"\n  Match with input: {match_input*100:.1f}%")
                        print(f"  Match with labels: {match_labels*100:.1f}%")

                        # Show logit statistics (should be ~uniform for random model)
                        logit_sample = logits[0, :10]  # First 10 positions
                        print(f"\n[Logit Stats (first 10 positions)]:")
                        print(f"  Mean: {logit_sample.mean().item():.4f}")
                        print(f"  Std:  {logit_sample.std().item():.4f}")
                        print(f"  Min:  {logit_sample.min().item():.4f}")
                        print(f"  Max:  {logit_sample.max().item():.4f}")

                        # Decode model's prediction
                        if tokenizer is not None:
                            try:
                                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                                print(f"\n[Model Prediction (greedy decode)]:")
                                print(f"  {pred_text[:200]}..." if len(pred_text) > 200 else f"  {pred_text}")
                            except Exception as e:
                                print(f"\n[Model Prediction decode error]: {e}")

                        # Show shapes
                        print(f"\n[Shapes]:")
                        print(f"  caption_input_ids: {caption_input_ids.shape}")
                        print(f"  caption_labels: {caption_labels.shape}")
                        print(f"  vision_features: {vis_feats.shape}")
                        print(f"  logits: {logits.shape}")
                        print(f"  loss: {loss_val.item():.4f} (random chance ~10.3)")
                        print("=" * 80 + "\n")

            # Add weighted loss
            weighted_loss = loss_val * weight
            total_loss = total_loss + weighted_loss
            loss_breakdown[f"loss/{loss_name}"] = float(loss_val.detach().item())

        return total_loss, loss_breakdown

    def _compute_siglip_pairwise_loss(
        self,
        video_embeddings: torch.Tensor,
        siglip_batch: dict,
    ) -> torch.Tensor:
        """Compute SigLIP loss using the unified SigLIPLoss from contrastive.py."""
        siglip_text = self.text_encoder(
            siglip_batch["input_ids"],
            siglip_batch["attention_mask"],
        )
        pos_mask: torch.Tensor = siglip_batch["positive_mask"]
        if pos_mask.sum().item() == 0:
            return video_embeddings.sum() * 0.0

        pos_weights: Optional[torch.Tensor] = siglip_batch.get("positive_weights")
        weights = pos_weights if self.siglip_severity_weighting else None
        if weights is not None:
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).float()
            # Clamp weights to prevent extreme gradients (max 10x - proven stable in vvuet8ad)
            weights = torch.clamp(weights, min=0.1, max=10.0)

        # Use unified SigLIPLoss via loss_fn (handles DDP, normalization, etc.)
        return self.loss_fn.run(
            video_features=video_embeddings,
            text_features=siglip_text,
            log_temp=self.log_temp,
            pos_mask=pos_mask,
            pos_weights=weights,
        )

    def _compute_siglip_infonce_loss(
        self,
        video_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        siglip_batch: dict,
    ) -> torch.Tensor:
        logits = self._scaled_logits(video_embeddings, text_embeddings)

        pos_mask: torch.Tensor = siglip_batch["positive_mask"]
        pos_weights: Optional[torch.Tensor] = siglip_batch.get("positive_weights")
        weights = pos_weights if self.siglip_severity_weighting else None
        if weights is not None:
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).float()
            # Clamp weights to prevent extreme gradients (max 10x - proven stable in vvuet8ad)
            weights = torch.clamp(weights, min=0.1, max=10.0)

        if pos_mask.sum().item() == 0:
            return logits.sum() * 0.0

        return self.multi_positive_loss(logits, pos_mask, weights)

    def _compute_tree_losses(
        self,
        video_embeddings: torch.Tensor,
        main_structure: Optional[torch.Tensor],
        tree_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        zero = video_embeddings.new_tensor(0.0)
        if (
            main_structure is None
            or self.tree_loss_weight <= 0
            or main_structure.numel() == 0
        ):
            return zero, zero, None, None

        valid_mask = main_structure >= 0
        if not valid_mask.any():
            return zero, zero, None, None

        if tree_logits is None:
            return zero, zero, None, None

        logits = tree_logits[valid_mask].view(-1).float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        logits = torch.clamp(logits, -20.0, 20.0)
        targets = main_structure[valid_mask].float()
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        tree_loss = self.tree_loss_fn(logits, targets)
        if not torch.isfinite(tree_loss):
            if getattr(self.config, "is_ref_device", False):
                print(
                    "[TreeLoss] Non-finite tree loss detected; skipping this batch.",
                )
            return zero, zero, None, None
        probs = torch.sigmoid(logits.detach())
        targets_detached = targets.detach()
        return tree_loss, tree_loss * self.tree_loss_weight, probs, targets_detached

    @staticmethod
    def _binary_auc(
        probs: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> Optional[float]:
        if probs is None or targets is None:
            return None
        if probs.numel() == 0 or targets.numel() == 0:
            return None
        targets = targets.float()
        pos = (targets > 0.5).sum()
        neg = targets.numel() - pos
        if pos == 0 or neg == 0:
            return None
        sorted_indices = torch.argsort(probs)
        sorted_targets = targets[sorted_indices]
        neg_cumulative = torch.cumsum((sorted_targets <= 0.5).float(), dim=0)
        auc = neg_cumulative[sorted_targets > 0.5].sum() / (pos * neg)
        return float(auc.item())

    @staticmethod
    def _grad_norm_from_params(params: list[torch.nn.Parameter]) -> float:
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            param_norm = p.grad.detach().data.norm(2)
            total += float(param_norm.item() ** 2)
        return total ** 0.5

    def _preprocess_inputs(self, batch: dict) -> tuple[dict, list[str]]:
        """
        Moves raw batch data (videos, texts) to GPU and returns a dictionary suitable
        for the model step, along with a list of paths or IDs for each sample.

        :param batch: Dictionary containing 'videos', 'encoded_texts', and 'paths'.
        :return: (step_inputs, paths_or_sids)
        """
        main_structure = batch.get("main_structure")
        if main_structure is not None:
            main_structure = main_structure.to(self.device)
        return {
            "videos": batch["videos"].to(self.device).float(),
            "input_ids": batch["encoded_texts"]["input_ids"].to(self.device),
            "attention_mask": batch["encoded_texts"]["attention_mask"].to(self.device),
            "main_structure": main_structure,
        }, batch["paths"]

    def _train_step(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        main_structure: Optional[torch.Tensor] = None,
        siglip_batch: Optional[dict] = None,
        video_paths: Optional[List[str]] = None,
        dataset: Optional[Any] = None,
        is_first_batch: bool = False,
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
        self.optimizer.zero_grad(set_to_none=True)

        amp_enabled: bool = self.scaler is not None
        autocast_ctx: torch.amp.autocast = torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        )

        need_tree_logits = (
            main_structure is not None
            and self.tree_loss_weight > 0
        )

        # Prepare caption tokens for LocCa if enabled (concatenates ALL positive texts per video)
        caption_input_ids, caption_labels, caption_attention_mask, caption_texts = self._prepare_caption_tokens(
            siglip_batch=siglip_batch,
            video_paths=video_paths,
            dataset=dataset,
        )

        with autocast_ctx:
            video_output = self.video_encoder(
                videos,
                compute_tree_logits=need_tree_logits,
                return_vision_features=self._locca_enabled,
            )
            video_emb = video_output["video_embeds"]
            tree_logits = video_output.get("tree_logits")
            vision_features = video_output.get("vision_features")  # For LocCa cross-attention
            text_emb = self.text_encoder(input_ids, attention_mask)

            # Debug logging for numerical stability monitoring
            if self.debug_numerical and self.step % self._debug_log_interval == 0:
                v_min, v_max = video_emb.min().item(), video_emb.max().item()
                v_mean, v_std = video_emb.mean().item(), video_emb.std().item()
                t_min, t_max = text_emb.min().item(), text_emb.max().item()
                t_mean, t_std = text_emb.mean().item(), text_emb.std().item()
                temp_val = torch.exp(-self.log_temp).item()
                print(
                    f"[Debug Step {self.step}] "
                    f"video_emb: min={v_min:.4f} max={v_max:.4f} mean={v_mean:.4f} std={v_std:.4f} | "
                    f"text_emb: min={t_min:.4f} max={t_max:.4f} mean={t_mean:.4f} std={t_std:.4f} | "
                    f"temp={temp_val:.4f}"
                )

            if (
                not torch.isfinite(video_emb).all()
                or not torch.isfinite(text_emb).all()
            ):
                diagnostics = {
                    "video_has_nan": bool(torch.isnan(video_emb).any().item()),
                    "video_has_inf": bool(torch.isinf(video_emb).any().item()),
                    "text_has_nan": bool(torch.isnan(text_emb).any().item()),
                    "text_has_inf": bool(torch.isinf(text_emb).any().item()),
                    "step": self.step,
                }
                raise NonFiniteLossError(
                    "Non-finite encoder outputs",
                    diagnostics=diagnostics,
                )
            # Multi-loss computation when loss_fns is configured
            loss_breakdown: Dict[str, float] = {}
            if self.loss_fns:
                contrastive_loss, loss_breakdown = self._compute_multi_loss(
                    video_embeddings=video_emb,
                    text_embeddings=text_emb,
                    siglip_batch=siglip_batch,
                    caption_input_ids=caption_input_ids,
                    caption_labels=caption_labels,
                    caption_attention_mask=caption_attention_mask,
                    vision_features=vision_features,
                    caption_texts=caption_texts,
                    is_first_batch=is_first_batch,
                    dataset=dataset,
                )
            elif siglip_batch and self.siglip_active:
                if self.use_siglip_pairwise:  # All siglip* losses use unified SigLIPLoss
                    contrastive_loss = self._compute_siglip_pairwise_loss(
                        video_embeddings=video_emb,
                        siglip_batch=siglip_batch,
                    )
                elif self.use_siglip_infonce and self.multi_positive_loss is not None:
                    contrastive_loss = self._compute_siglip_infonce_loss(
                        video_embeddings=video_emb,
                        text_embeddings=text_emb,
                        siglip_batch=siglip_batch,
                    )
                else:
                    contrastive_loss = self.loss_fn.run(
                        video_features=video_emb,
                        text_features=text_emb,
                        log_temp=self.log_temp,
                    )
            else:
                contrastive_loss = self.loss_fn.run(
                    video_features=video_emb,
                    text_features=text_emb,
                    log_temp=self.log_temp,
                )
            tree_loss_raw, tree_loss_weighted, tree_probs, tree_targets = self._compute_tree_losses(
                video_embeddings=video_emb,
                main_structure=main_structure,
                tree_logits=tree_logits,
            )
            loss = contrastive_loss + tree_loss_weighted

            if not torch.isfinite(loss):
                diagnostics = {
                    "contrastive_loss": float(contrastive_loss.detach().float()),
                    "tree_loss": float(tree_loss_weighted.detach().float()),
                    "step": self.step,
                }
                raise NonFiniteLossError(
                    "Detected non-finite loss before backward pass",
                    diagnostics=diagnostics,
                )

        if amp_enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        video_params = [p for p in self.video_encoder.parameters() if p.requires_grad]
        text_params = [p for p in self.text_encoder.parameters() if p.requires_grad]

        if amp_enabled:
            self.scaler.unscale_(self.optimizer)

        video_grad_norm = self._grad_norm_from_params(video_params)
        text_grad_norm = self._grad_norm_from_params(text_params)
        grad_norm_val: float = (video_grad_norm ** 2 + text_grad_norm ** 2) ** 0.5

        if (
            self.video_max_grad_norm
            and self.video_max_grad_norm > 0
            and video_params
        ):
            clip_grad_norm_(video_params, max_norm=self.video_max_grad_norm)
        if (
            self.text_max_grad_norm
            and self.text_max_grad_norm > 0
            and text_params
        ):
            clip_grad_norm_(text_params, max_norm=self.text_max_grad_norm)

        # Guard against NaN temperature (reset to default if corrupted)
        if hasattr(self, "log_temp") and not torch.isfinite(self.log_temp):
            print("[WARNING] log_temp became non-finite, resetting to default")
            self.log_temp.data.fill_(math.log(0.07))
            if self.log_temp.grad is not None:
                self.log_temp.grad.zero_()

        # Clip temperature gradient to prevent wild swings in logit scaling
        # Tightened from ±0.1 to ±0.01 to prevent temperature explosion
        if hasattr(self, "log_temp") and self.log_temp.grad is not None:
            self.log_temp.grad.clamp_(-0.01, 0.01)

        self._last_grad_norm = grad_norm_val
        self._last_video_grad_norm = video_grad_norm
        self._last_text_grad_norm = text_grad_norm

        if amp_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.lr_scheduler and self.scheduler_per_iteration:
            self.lr_scheduler.step()

        self.step += 1

        if (
            self.config.is_ref_device
            and self.wandb_wrapper.is_initialized()
        ):
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

        tree_auc = self._binary_auc(tree_probs, tree_targets)

        effective_logit_scale = float(
            torch.exp(
                torch.clamp((-self.log_temp).detach().float(), max=math.log(100.0))
            ).item()
        )

        batch_metrics = {
            "loss": loss.detach().item(),
            "contrastive_loss": contrastive_loss.detach().item(),
            "tree_loss": tree_loss_raw.detach().item(),
            "tree_auc": tree_auc if tree_auc is not None else float("nan"),
            "temperature": self.log_temp.exp().detach().item(),
            "logit_scale": effective_logit_scale,
            "grad_norm": grad_norm_val,
            "grad_norm/video": video_grad_norm,
            "grad_norm/text": text_grad_norm,
            "alignment_score": alignment_score,
            **embedding_norms,
            **lr_metrics,
            **loss_breakdown,  # Individual loss values from multi-loss
        }

        embeddings = {
            "video_embeddings": video_fp32,
            "text_embeddings": text_fp32,
        }
        if tree_logits is not None:
            embeddings["tree_logits"] = tree_logits.detach().float()

        return batch_metrics, embeddings

    def _val_step(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        main_structure: Optional[torch.Tensor] = None,
        siglip_batch: Optional[dict] = None,
        video_paths: Optional[List[str]] = None,
        dataset: Optional[Any] = None,
        is_first_batch: bool = False,
    ) -> tuple[dict, dict]:
        """
        Performs a single validation step (forward pass + metric computation).

        :param videos: Tensor of shape [B, num_clips, T, H, W, C].
        :param input_ids: Encoded text token IDs.
        :param attention_mask: Attention mask for text.
        :param video_paths: Optional list of video file paths for LocCa caption building.
        :param dataset: Optional dataset reference for LocCa caption building.
        :param is_first_batch: Whether this is the first batch (for debug output).
        :return: (batch_metrics, embeddings) similar to _train_step, but without backprop.
        """
        need_tree_logits = (
            main_structure is not None
            and self.tree_loss_weight > 0
        )

        # Prepare caption tokens for LocCa if enabled
        caption_input_ids, caption_labels, caption_attention_mask, caption_texts = self._prepare_caption_tokens(
            siglip_batch=siglip_batch,
            video_paths=video_paths,
            dataset=dataset,
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                video_output = self.video_encoder(
                    videos,
                    compute_tree_logits=need_tree_logits,
                    return_vision_features=self._locca_enabled,
                )
                video_features = video_output["video_embeds"]
                tree_logits = video_output.get("tree_logits")
                vision_features = video_output.get("vision_features")  # For LocCa cross-attention
                text_features = self.text_encoder(input_ids, attention_mask)
                # Multi-loss computation when loss_fns is configured
                loss_breakdown: Dict[str, float] = {}
                if self.loss_fns:
                    contrastive_loss, loss_breakdown = self._compute_multi_loss(
                        video_embeddings=video_features,
                        text_embeddings=text_features,
                        siglip_batch=siglip_batch,
                        vision_features=vision_features,
                        caption_input_ids=caption_input_ids,
                        caption_labels=caption_labels,
                        caption_attention_mask=caption_attention_mask,
                        caption_texts=caption_texts,
                        dataset=dataset,
                        is_first_batch=is_first_batch,
                    )
                elif siglip_batch and self.siglip_active:
                    if self.use_siglip_pairwise:  # All siglip* losses use unified SigLIPLoss
                        contrastive_loss = self._compute_siglip_pairwise_loss(
                            video_embeddings=video_features,
                            siglip_batch=siglip_batch,
                        )
                    elif self.use_siglip_infonce and self.multi_positive_loss is not None:
                        contrastive_loss = self._compute_siglip_infonce_loss(
                            video_embeddings=video_features,
                            text_embeddings=text_features,
                            siglip_batch=siglip_batch,
                        )
                    else:
                        contrastive_loss = self.loss_fn.run(
                            video_features=video_features,
                            text_features=text_features,
                            log_temp=self.log_temp
                        )
                else:
                    contrastive_loss = self.loss_fn.run(
                        video_features=video_features,
                        text_features=text_features,
                        log_temp=self.log_temp
                    )
                tree_loss_raw, tree_loss_weighted, tree_probs, tree_targets = self._compute_tree_losses(
                    video_embeddings=video_features,
                    main_structure=main_structure,
                    tree_logits=tree_logits,
                )
                loss = contrastive_loss + tree_loss_weighted

            embedding_norms = compute_embedding_norms(video_features, text_features)
            alignment_score = compute_alignment_score(video_features, text_features)

        metrics = {"alignment_score": alignment_score, **embedding_norms}
        tree_auc = self._binary_auc(tree_probs, tree_targets)

        embeddings_out = {
            "video_embeddings": video_features.float(),
            "text_embeddings": text_features.float(),
        }
        if tree_logits is not None:
            embeddings_out["tree_logits"] = tree_logits.float()
        if vision_features is not None:
            embeddings_out["vision_features"] = vision_features.float()

        return (
            {
                "loss": loss.detach().item(),
                "contrastive_loss": contrastive_loss.detach().item(),
                "tree_loss": tree_loss_raw.detach().item(),
                "tree_auc": tree_auc if tree_auc is not None else float("nan"),
                "temperature": self.log_temp.exp().detach().item(),
                **{
                    k: v.detach().item() if torch.is_tensor(v) else v
                    for k, v in metrics.items()
                },
                **loss_breakdown,  # Individual loss values from multi-loss
            },
            embeddings_out,
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
                    video_embeddings = self.video_encoder(batch["videos"])["video_embeds"].float()
                    
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
                        averaged_row[column] = None if modes.empty else modes.iloc[0]
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
