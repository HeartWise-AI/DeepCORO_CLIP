import os
import wandb
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from tqdm import tqdm

from utils.enums import RunMode
from utils.config import HeartWiseConfig
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
from utils.logging import (
    log_best_worst_retrievals,
    log_gradient_norms,
    save_retrieval_results,
)
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@RunnerRegistry.register("video_contrastive_learning")
class VideoContrastiveLearningRunner:
    """
    This class runs a video contrastive learning pipeline using a VideoEncoder and TextEncoder.
    It handles both training and validation loops in a distributed data-parallel setting.
    """

    def __init__(
        self,
        config: HeartWiseConfig,
        device: int,
        world_size: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        video_encoder: VideoEncoder,
        text_encoder: TextEncoder,
        optimizer: AdamW,
        scaler: GradScaler,
        log_temp: torch.Tensor,
        lr_scheduler: LRScheduler,
        loss_fn: callable,
        output_dir: str,
    ):
        """
        Initialize the runner with provided configurations, data loaders, and modules.

        :param config: HeartWiseConfig object with run/training configuration.
        :param device: Integer specifying the GPU index.
        :param world_size: Number of GPUs used in DDP.
        :param train_loader: DataLoader for training dataset.
        :param val_loader: DataLoader for validation dataset.
        :param video_encoder: VideoEncoder model.
        :param text_encoder: TextEncoder model.
        :param optimizer: AdamW optimizer instance.
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

        self.config: HeartWiseConfig = config
        self.device: int = device
        self.world_size: int = world_size
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.video_encoder: VideoEncoder = video_encoder
        self.text_encoder: TextEncoder = text_encoder
        self.optimizer: AdamW = optimizer
        self.scaler: GradScaler = scaler
        self.lr_scheduler: LRScheduler = lr_scheduler
        self.loss_fn: callable = loss_fn
        self.output_dir: str = output_dir
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.log_temp = log_temp

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
        """
        for epoch in range(start_epoch, end_epoch):
            # Synchronize before epoch starts (DDP barrier)
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])

            # Training phase
            train_metrics = self._run_epoch(mode=RunMode.TRAIN, epoch=epoch)
            if self.config.is_ref_device or (self.device == 0):
                # Let wandb auto-increment steps
                wandb.log(train_metrics)
                print(f"[DEBUG] rank={self.device} => Logged train metrics to W&B")

            # Sync before validation
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])

            # Validation phase
            val_metrics = self._run_epoch(
                mode=RunMode.VALIDATION, 
                epoch=epoch
            )
            
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])
            
            if self.config.is_ref_device:
                wandb.log(val_metrics)
                print(f"[DEBUG] rank={self.device} => Logged val metrics to W&B")

            # Sync after validation
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])
            
            # If it's an epoch-based scheduler (like StepLR, CosineAnnealingLR, etc.),
            # call lr_scheduler.step() after each epoch
            if self.lr_scheduler and (not self.scheduler_per_iteration):
                self.lr_scheduler.step()

            # Update best model
            if val_metrics["val/loss"] < self.best_val_loss:
                prev_best = self.best_val_loss
                self.best_val_loss = val_metrics["val/loss"]
                self.best_epoch = epoch
                if self.config.is_ref_device or (self.device == 0):
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

            # Always save "latest" checkpoint
            if self.config.is_ref_device or (self.device == 0):
                self._save_checkpoint(
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_best=False,
                )
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])
                
            print(f"gpu_id after saving checkpoint: {self.device}")
  

    def _gather_tensor_along_batch(self, local_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        """
        Gathers tensors across multiple ranks along the batch dimension (dim=0).

        :param local_tensor: Local tensor to gather from current rank.
        :param world_size: Number of participating ranks.
        :return: Concatenated tensor containing data from all ranks.
        """
        if self.config.world_size < 2:
            return local_tensor

        device = local_tensor.device
        local_size = torch.tensor([local_tensor.shape[0]], device=device, dtype=torch.long)
        sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(sizes_list, local_size)
        sizes_list = [s.item() for s in sizes_list]
        max_size = max(sizes_list)

        # Pad to max_size along dim=0
        if local_tensor.dim() == 1:
            pad = (0, max_size - local_tensor.shape[0])
            padded = torch.nn.functional.pad(local_tensor, pad, "constant", 0)
        else:
            pad_rows = max_size - local_tensor.shape[0]
            if pad_rows > 0:
                padded = torch.nn.functional.pad(local_tensor, (0, 0, 0, pad_rows))
            else:
                padded = local_tensor

        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, padded)

        cat = torch.stack(gathered, dim=0)
        out_list = []
        for rank_idx in range(world_size):
            actual_size = sizes_list[rank_idx]
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

        local_data_bytes = pickle.dumps(local_strings)
        local_size = torch.tensor([len(local_data_bytes)], dtype=torch.long, device=device)

        sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(sizes_list, local_size)
        sizes_list = [s.item() for s in sizes_list]
        max_size = max(sizes_list)

        local_buffer = torch.zeros(max_size, dtype=torch.uint8, device=device)
        local_buffer[: local_size.item()] = torch.as_tensor(
            list(local_data_bytes), dtype=torch.uint8, device=device
        )

        all_buffers = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(all_buffers, local_buffer)

        out_list = []
        for rank_idx, buf in enumerate(all_buffers):
            size = sizes_list[rank_idx]
            valid_bytes = buf[:size].cpu().numpy().tobytes()
            str_list = pickle.loads(valid_bytes)
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
        """
        assert mode in [RunMode.TRAIN, RunMode.VALIDATION]

        self.video_encoder.train(mode == "train")
        self.text_encoder.train(mode == "train")

        total_loss = 0.0
        epoch_metrics = {}

        dataloader = self.train_loader if mode == "train" else self.val_loader
        step_fn = self._train_step if mode == "train" else self._val_step

        # Store local embeddings & text for retrieval computations
        all_video_embeddings_local = []
        all_text_embeddings_local = []
        all_paths_local = []
        all_ground_truth_reports_local = []

        tqdm_desc = f"[GPU {self.device}] Running {mode} Epoch {epoch + 1}"
        if (self.config.is_ref_device or (self.device == 0)):
            data_iter = tqdm(dataloader, desc=tqdm_desc)
        else:
            data_iter = dataloader

        batch_count = 0
        for _, batch in enumerate(data_iter, start=1):
            if batch["videos"] is None or batch["encoded_texts"] is None:
                continue

            step_inputs, paths_or_sids = self._preprocess_inputs(batch)

            if not self.config.multi_video:
                # shape => [B, 1, T, H, W, C]
                step_inputs["videos"] = step_inputs["videos"].unsqueeze(1)

            batch_metrics, embeddings = step_fn(**step_inputs)

            if self.lr_scheduler and self.scheduler_per_iteration and (mode == RunMode.TRAIN):
                print(f"[DEBUG] step_fn: {step_fn}")
                self.lr_scheduler.step()

            # Store embeddings on CPU
            all_video_embeddings_local.append(embeddings["video_embeddings"].cpu())
            all_text_embeddings_local.append(embeddings["text_embeddings"].cpu())

            if self.config.multi_video:
                for sid in paths_or_sids:
                    vid_list = dataloader.dataset.get_video_paths(sid)
                    if len(vid_list) > 0:
                        all_paths_local.append(vid_list[0])
                    else:
                        all_paths_local.append(str(sid))
            else:
                all_paths_local.extend(paths_or_sids)

            all_ground_truth_reports_local.extend(dataloader.dataset.get_reports(paths_or_sids))

            # accumulate metrics
            for k, v in batch_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)

            total_loss += float(batch_metrics["loss"])
            batch_count += 1

            if (self.config.is_ref_device or (self.device == 0)) and mode == "train":
                data_iter.set_postfix(
                    {
                        f"{mode}_loss": f"{batch_metrics['loss']:.4f}",
                        "avg_loss": f"{(total_loss / batch_count):.4f}",
                    }
                )

        del batch_metrics, embeddings
        torch.cuda.empty_cache()

        # finalize epoch metrics
        if batch_count > 0:
            for k in epoch_metrics.keys():
                epoch_metrics[k] /= batch_count

        # 1) Concat local feats
        if len(all_video_embeddings_local) > 0:
            local_video_feats = torch.cat(all_video_embeddings_local, dim=0).to(self.device)
            local_text_feats = torch.cat(all_text_embeddings_local, dim=0).to(self.device)
        else:
            local_video_feats = torch.empty((0, 512), device=self.device)
            local_text_feats = torch.empty((0, 512), device=self.device)

        if (self.config.is_ref_device or (self.device == 0)):
            print(
                f"[DEBUG rank={self.device}] local_video_feats={local_video_feats.shape}, "
                f"local_text_feats={local_text_feats.shape}, mode={mode}"
            )

        # 2) gather across GPUs
        global_video_feats = self._gather_tensor_along_batch(local_video_feats, self.world_size)
        global_text_feats = self._gather_tensor_along_batch(local_text_feats, self.world_size)

        # 3) gather paths & reports
        global_paths = self._gather_strings_across_gpus(
            all_paths_local, self.world_size, device=local_video_feats.device
        )
        global_reports = self._gather_strings_across_gpus(
            all_ground_truth_reports_local, self.world_size, device=local_video_feats.device
        )

        # Optionally compute NxM retrieval metrics on rank 0
        retrieval_metrics = {}
        if mode == "val" and self.config.is_ref_device:
            print(
                f"[DEBUG rank={self.device}] Starting retrieval computation with {global_video_feats.shape[0]} videos."
            )
            
            # Step 1: Deduplicate texts and create mapping
            unique_texts = sorted(set(global_reports))
            text_to_index = {text: idx for idx, text in enumerate(unique_texts)}
            
            # Step 2: Get ground truth indices for each video
            ground_truth_indices = torch.tensor(
                [text_to_index[text] for text in global_reports],
                device=self.device
            )
            
            print(f"[DEBUG rank={self.device}] Found {len(unique_texts)} unique texts out of {len(global_reports)} total.")
            
            # Step 3: Encode unique texts
            unique_text_embeddings = []
            batch_size = 64  # Process in batches to avoid OOM
            
            self.text_encoder.eval()
            with torch.no_grad():
                for start_idx in range(0, len(unique_texts), batch_size):
                    end_idx = min(start_idx + batch_size, len(unique_texts))
                    text_batch = unique_texts[start_idx:end_idx]
                    
                    # Tokenize batch - handle DDP wrapped model
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
                    unique_text_embeddings.append(text_embs)
            
            # Concatenate all batches
            unique_text_embeddings = torch.cat(unique_text_embeddings, dim=0)
            
            # Step 4: Normalize embeddings
            global_video_feats = nn.functional.normalize(global_video_feats, dim=1)
            unique_text_embeddings = nn.functional.normalize(unique_text_embeddings, dim=1)
            
            # Step 5: Compute NxM similarity matrix
            similarity_matrix = torch.matmul(global_video_feats, unique_text_embeddings.t())
            print(
                f"[DEBUG rank={self.device}] Computed NxM sim matrix with shape={similarity_matrix.shape}"
            )

            # Log best/worst retrievals using unique texts
            log_best_worst_retrievals(
                similarity_matrix=similarity_matrix,
                all_paths=global_paths,
                unique_texts=unique_texts,  # Now using actual unique texts
                ground_truth_indices=ground_truth_indices,
                epoch=epoch,
            )

            # Save retrieval results with mapping
            save_retrieval_results(
                similarity_matrix=similarity_matrix,
                all_paths=global_paths,
                all_ground_truth_reports=global_reports,
                report_to_global_index=text_to_index,  # Pass the mapping
                epoch=epoch,
                output_dir=self.output_dir,
            )

            # Compute retrieval metrics using ground truth indices
            recall_metrics = compute_recall_at_k(
                similarity_matrix, ground_truth_indices, k_values=self.config.recall_k
            )
            mrr_score = compute_mrr(similarity_matrix, ground_truth_indices)
            map_score = compute_map(similarity_matrix, ground_truth_indices)
            median_rank_score = compute_median_rank(similarity_matrix, ground_truth_indices)
            ndcg_score = compute_ndcg_at_k(
                similarity_matrix, ground_truth_indices, k_values=self.config.ndcg_k
            )

            retrieval_metrics.update(recall_metrics)
            retrieval_metrics.update(mrr_score)
            retrieval_metrics.update(ndcg_score)
            retrieval_metrics["MAP"] = map_score
            retrieval_metrics["MedianRank_V2T"] = median_rank_score
            
            # Save unique texts and their indices
            df_texts = pd.DataFrame({
                "Index": range(len(unique_texts)),
                "Text": unique_texts
            })
            texts_csv_path = os.path.join(self.output_dir, f"val_unique_texts.csv")
            df_texts.to_csv(texts_csv_path, index=False)
            print(f"[DEBUG rank={self.device}] Saved {len(unique_texts)} unique texts to {texts_csv_path}")
            
            # Also save text embeddings for future use
            embeddings_path = os.path.join(self.output_dir, f"val_text_embeddings_epoch_{epoch}.pt")
            torch.save({
                'text_embeddings': unique_text_embeddings.cpu(),
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
        gathered_metrics = {}
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
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            return t.item()
        else:
            return val

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Saves model checkpoint (including model weights, optimizer, scheduler, and metrics).

        :param epoch: Current epoch index.
        :param metrics: Dictionary of metrics to be saved.
        :param is_best: If True, saves as 'best_epoch.pt'. Otherwise, saves as 'checkpoint.pt'.
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
        }

        save_path = os.path.join(
            checkpoint_dir, "best_epoch.pt" if is_best else "checkpoint.pt"
        )
        torch.save(checkpoint, save_path)
        print(
            f"\nSaved {'best' if is_best else 'latest'} checkpoint at epoch {epoch + 1} to {save_path}"
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
        self, videos: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[dict, dict]:
        """
        Performs a single training step, including forward, backward, gradient update,
        and metric computation.

        :param videos: Tensor of shape [B, num_clips, T, H, W, C].
        :param input_ids: Encoded text token IDs.
        :param attention_mask: Attention mask for text.
        :return: (batch_metrics, embeddings) where
          - batch_metrics is a dictionary with loss, alignment scores, etc.
          - embeddings is a dictionary with 'video_embeddings' and 'text_embeddings'.
        """
        self.optimizer.zero_grad()

        if self.scaler:
            with torch.amp.autocast("cuda"):
                video_embeddings = self.video_encoder(videos)
                text_embeddings = self.text_encoder(input_ids, attention_mask)
                loss = self.loss_fn(video_embeddings, text_embeddings, self.log_temp)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])

            # Log gradient norms
            if self.config.is_ref_device:
                log_gradient_norms(self.video_encoder, prefix="train/video_encoder/")
                log_gradient_norms(self.text_encoder, prefix="train/text_encoder/")

            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.video_encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=5.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            video_embeddings = video_embeddings.float()
            text_embeddings = text_embeddings.float()
        else:
            video_embeddings = self.video_encoder(videos)
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            loss = self.loss_fn(video_embeddings, text_embeddings, self.log_temp)

            loss.backward()

            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])

            if self.config.is_ref_device:
                log_gradient_norms(self.video_encoder, prefix="train/video_encoder/")
                log_gradient_norms(self.text_encoder, prefix="train/text_encoder/")

            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.video_encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=5.0)

            self.optimizer.step()
        
        # metrics
        with torch.no_grad():
            embedding_norms = compute_embedding_norms(video_embeddings, text_embeddings)
            alignment_score = compute_alignment_score(video_embeddings, text_embeddings)

        metrics = {
            **embedding_norms,
            "alignment_score": alignment_score,
        }

        lr_metrics = {}
        for pg in self.optimizer.param_groups:
            lr_metrics[f"lr/{pg['name']}"] = pg["lr"]

        return (
            {
                "loss": loss.detach().item(),
                **lr_metrics,
                "temperature": self.log_temp.exp().detach().item(),
                **{
                    k: v.detach().item() if torch.is_tensor(v) else v
                    for k, v in metrics.items()
                },
            },
            {
                "video_embeddings": video_embeddings,
                "text_embeddings": text_embeddings,
            },
        )

    def _val_step(
        self, videos: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
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
                loss = self.loss_fn(video_features, text_features, self.log_temp)

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
        pass

    def inference(self):
        """
        Method for a dedicated inference.
        """
        pass