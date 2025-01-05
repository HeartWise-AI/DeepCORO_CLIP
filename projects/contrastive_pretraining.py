import os
import wandb
import pickle

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from typing import Any
from models.text_encoder import TextEncoder
from models.video_encoder import VideoEncoder

from runners.video_constrative_learning import VideoContrastiveLearningRunner

from utils.ddp import DDP, dist
from utils.losses import get_loss_fn
from utils.config import HeartWiseConfig
from utils.video import cleanup_temp_video
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry, 
    ConfigRegistry,
    RunnerRegistry, 
    ProjectRegistry, 
)
from utils.logging import (
    create_logger,
    cleanup_temp_video,
    convert_video_for_wandb
)
from utils.metrics import (
    compute_mrr, 
    compute_map,
    compute_ndcg_at_k,
    compute_recall_at_k, 
    compute_median_rank,
    compute_embedding_norms, 
    compute_alignment_score, 
)
from dataloaders.stats_dataset import get_stats_dataloader
from dataloaders.video_dataset import get_distributed_video_dataloader


def save_checkpoint(model_dict, metrics_dict, output_path, is_best=False):
    """
    Save model checkpoint with metrics.

    Args:
        model_dict (dict): Dictionary containing model states
        metrics_dict (dict): Dictionary containing training metrics
        output_path (str/Path): Path to save the checkpoint
        is_best (bool): Whether this is the best model so far
    """
    # Ensure parent directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine model and metrics into one checkpoint
    checkpoint = {**model_dict, **metrics_dict}

    # Save checkpoint
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")


def validate_epoch(
    args,
    video_encoder,
    text_encoder,
    dataloader,
    device,
    wandb_run,
    rank=0,
    world_size=1,
    epoch=0,
    all_text_embeddings=None,
    all_reports=None,
    text_embedding_pickle_path="text_embeddings.pkl",
    output_dir="outputs",
    report_to_global_index=None,
    use_val_only_pool=True,
):
    """
    Validation epoch with retrieval computation and logging.

    Args:
        video_encoder, text_encoder: Models
        dataloader: Validation DataLoader
        device: torch device
        wandb_run: W&B logger
        rank: DDP rank
        world_size: DDP world size
        epoch: current epoch
        all_text_embeddings: precomputed text embeddings for the pool (val-only or global)
        all_reports: corresponding reports (val-only or global)
        report_to_global_index: mapping from report to index in `all_reports`
        use_val_only_pool (bool): If True, evaluate retrieval using only the val set embeddings,
                                  and log best/worst examples. If False, only compute metrics.

    Returns:
        avg_loss, epoch_metrics, examples_dict
    """
    video_encoder.eval()
    text_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    epoch_metrics = {
        "Recall@1_V2T": 0.0,
        "Recall@5_V2T": 0.0,
        "MRR_V2T": 0.0,
        "video_norm": 0.0,
        "text_norm": 0.0,
        "alignment_score": 0.0,
    }

    all_video_embeddings = []
    all_paths = []
    all_ground_truth_reports = []

    progress = tqdm(dataloader, desc="Validation") if rank == 0 else dataloader

    loss_fn = get_loss_fn(args.loss_name)

    with torch.no_grad():
        for _, batch in enumerate(progress):
            videos, encoded_texts, paths = batch
            batch_reports = dataloader.dataset.get_reports(paths)

            if videos is None or encoded_texts is None:
                continue

            batch_size = videos.size(0)
            if batch_size < 2:
                continue

            videos = videos.to(device, non_blocking=True).float()
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            video_features = video_encoder(videos)
            text_features = text_encoder(input_ids, attention_mask)

            # Compute validation loss on this batch
            loss = loss_fn(video_features, text_features)
            total_loss += loss.item()
            num_batches += 1

            # Store raw video embeddings (not normalized yet)
            all_video_embeddings.append(video_features.cpu())

            all_paths.extend(paths)
            all_ground_truth_reports.extend(batch_reports)

            # Clean up
            del videos, input_ids, attention_mask, video_features, text_features, loss
            torch.cuda.empty_cache()

    # Reduce loss across processes if distributed
    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = loss_tensor.tolist()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # If no batches or no embeddings, return early
    if num_batches == 0 or len(all_video_embeddings) == 0:
        if rank == 0:
            print("\nNo validation batches processed or no valid data.")
        return (
            avg_loss,
            epoch_metrics,
            {
                "best_videos": [],
                "best_reports": [],
                "best_scores": [],
                "worst_videos": [],
                "worst_reports": [],
                "worst_scores": [],
            },
        )

    # Concatenate all video embeddings
    all_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(device)
    all_text_embeddings = all_text_embeddings.to(device)

    # Compute norms before normalization
    max_len = min(all_video_embeddings.size(0), all_text_embeddings.size(0))
    truncated_video = all_video_embeddings[:max_len]
    truncated_text = all_text_embeddings[:max_len]

    # Compute norms from raw embeddings
    norm_metrics = compute_embedding_norms(truncated_video, truncated_text)
    epoch_metrics["video_norm"] = norm_metrics["video_norm"]
    epoch_metrics["text_norm"] = norm_metrics["text_norm"]

    # Now normalize embeddings for similarity calculation
    all_video_embeddings = nn.functional.normalize(all_video_embeddings, dim=1)
    all_text_embeddings = nn.functional.normalize(all_text_embeddings, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_video_embeddings, all_text_embeddings.T)

    effective_k = min(5, similarity_matrix.size(0), similarity_matrix.size(1))
    if similarity_matrix.size(0) >= 5 and similarity_matrix.size(1) >= 5:
        global_ground_truth_indices = [
            report_to_global_index[gt_report] for gt_report in all_ground_truth_reports
        ]
        global_ground_truth_indices_tensor = torch.tensor(
            global_ground_truth_indices, device=device
        )

        recall_metrics = compute_recall_at_k(
            similarity_matrix, global_ground_truth_indices_tensor, k_values=[1, effective_k]
        )
        mrr_metrics = compute_mrr(similarity_matrix, global_ground_truth_indices_tensor)

        map_metrics = compute_map(similarity_matrix, global_ground_truth_indices_tensor)
        ndcg_metrics = compute_ndcg_at_k(similarity_matrix, global_ground_truth_indices_tensor, k_values=[1, effective_k])
        median_rank_metrics = compute_median_rank(similarity_matrix, global_ground_truth_indices_tensor)

        alignment_score = compute_alignment_score(
            all_video_embeddings[:max_len],
            all_text_embeddings[:max_len],
        )

        for metric_name, value in recall_metrics.items():
            epoch_metrics[metric_name] = value
        for metric_name, value in mrr_metrics.items():
            epoch_metrics[metric_name] = value
        epoch_metrics["alignment_score"] = alignment_score

    # Only log best/worst retrieval if we are using val-only pool
    val_best_videos = []
    val_best_reports = []
    val_worst_videos = []
    val_worst_reports = []

    if use_val_only_pool:
        # For logging best and worst retrieval examples
        max_scores, _ = similarity_matrix.max(dim=1)
        k = 1  # Only pick top 1 best and worst
        best_scores, best_indices = torch.topk(max_scores, k=k)
        worst_scores, worst_indices = torch.topk(max_scores, k=k, largest=False)

        # Prepare CSV with ground truth and top 5 predictions for each sample
        val_csv_path = os.path.join(output_dir, f"val_epoch{epoch}.csv")
        import csv

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

            # For each video, get top-5 predictions and their similarity scores
            for i, path in enumerate(all_paths):
                top_5_text_indices = torch.argsort(similarity_matrix[i], descending=True)[:5]
                predicted_indices = [idx.item() for idx in top_5_text_indices]
                predicted_sims = [similarity_matrix[i, idx].item() for idx in top_5_text_indices]

                row_data = [path, i]
                for p_idx, p_sim in zip(predicted_indices, predicted_sims):
                    row_data.append(p_idx)
                    row_data.append(f"{p_sim:.4f}")

                writer.writerow(row_data)

        if best_indices.numel() > 0:
            idx = best_indices[0].item()
            score = best_scores[0].item()
            top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
            predicted_reports = [all_reports[j.item()] for j in top_5_text_indices]
            val_best_videos.append(str(all_paths[idx]))
            val_best_reports.append(
                {
                    "ground_truth": all_ground_truth_reports[idx],
                    "predicted": predicted_reports,
                    "similarity_score": score,
                }
            )

        if worst_indices.numel() > 0:
            idx = worst_indices[0].item()
            score = worst_scores[0].item()
            top_5_text_indices = torch.argsort(similarity_matrix[idx], descending=True)[:5]
            predicted_reports = [all_reports[j.item()] for j in top_5_text_indices]
            val_worst_videos.append(str(all_paths[idx]))
            val_worst_reports.append(
                {
                    "ground_truth": all_ground_truth_reports[idx],
                    "predicted": predicted_reports,
                    "similarity_score": score,
                }
            )

        # Log the best example video and top-5 texts
        if val_best_videos and val_best_reports and wandb_run is not None:
            # Convert video and get path + cleanup flag
            mp4_path, is_temp = convert_video_for_wandb(val_best_videos[0])

            wandb_run.log(
                {
                    "qualitative/good_retrieval": wandb.Video(
                        mp4_path,
                        caption=f"Sim: {val_best_reports[0]['similarity_score']:.3f}",
                        format="mp4",
                    ),
                    "epoch": epoch,
                }
            )

            predicted_html = "<br>".join(
                [f"{i+1}. {text}" for i, text in enumerate(val_best_reports[0]["predicted"])]
            )
            ground_truth_html = (
                f"<b>Ground Truth:</b> {val_best_reports[0]['ground_truth']}<br>"
                f"<b>Top 5 Predicted:</b><br>{predicted_html}"
            )
            wandb_run.log(
                {
                    f"qualitative/good_retrieval_text": wandb.Html(ground_truth_html),
                    "epoch": epoch,
                }
            )

            # Clean up temp file if needed
            if is_temp:
                cleanup_temp_video(mp4_path)

        # Similarly for worst retrieval
        if val_worst_videos and val_worst_reports and wandb_run is not None:
            # Convert video and get path + cleanup flag
            mp4_path, is_temp = convert_video_for_wandb(val_worst_videos[0])

            wandb_run.log(
                {
                    "qualitative/bad_retrieval": wandb.Video(
                        mp4_path,
                        caption=f"Sim: {val_worst_reports[0]['similarity_score']:.3f}",
                        format="mp4",
                    ),
                    "epoch": epoch,
                }
            )

            predicted_html = "<br>".join(
                [f"{i+1}. {text}" for i, text in enumerate(val_worst_reports[0]["predicted"])]
            )
            ground_truth_html = (
                f"<b>Ground Truth:</b> {val_worst_reports[0]['ground_truth']}<br>"
                f"<b>Top 5 Predicted:</b><br>{predicted_html}"
            )
            wandb_run.log(
                {f"qualitative/bad_retrieval_text": wandb.Html(ground_truth_html), "epoch": epoch}
            )

            # Clean up temp file if needed
            if is_temp:
                cleanup_temp_video(mp4_path)

    avg_text_embedding = all_text_embeddings.mean(dim=0)
    if rank == 0:
        print(f"\nAverage text embedding (first 5 dims): {avg_text_embedding[:5]}")
        print(f"Validation Loss: {avg_loss:.4f}")

        if wandb_run is not None:
            wandb_run.log({"val/avg_loss": avg_loss, "epoch": epoch})
            for metric_name, val in epoch_metrics.items():
                wandb_run.log({f"val/{metric_name}": val, "epoch": epoch})

    return (
        avg_loss,
        epoch_metrics,
        {
            "best_videos": val_best_videos,
            "best_reports": val_best_reports,
            "best_scores": [br["similarity_score"] for br in val_best_reports],
            "worst_videos": val_worst_videos,
            "worst_reports": val_worst_reports,
            "worst_scores": [wr["similarity_score"] for wr in val_worst_reports],
        },
    )


def load_train_objs(
    config: HeartWiseConfig,
)->dict:
    wandb_run = None
    if config.is_ref_device:
        wandb_run = create_logger(config=config)

        # After wandb.init(), wandb.config is available.
        # Override args with wandb.config parameters if present.
        if wandb_run is not None and len(wandb_run.config.keys()) > 0:
            for key, value in wandb_run.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"Warning: {key} in wandb.config not recognized as an arg.")    
                    
    # Calculate dataset statistics (only on rank 0)
    mean, std = None, None
    if config.is_ref_device:
        print("\n=== Calculating Dataset Statistics ===")

        stats_loader: DataLoader = get_stats_dataloader(config)

        num_stats_samples = min(100, 1000)
        print(f"Stats dataset length: {len(stats_loader)}")
        if len(stats_loader) > num_stats_samples:
            indices = torch.linspace(0, len(stats_loader) - 1, num_stats_samples).long().tolist()
            stats_loader = torch.utils.data.Subset(stats_loader, indices)

        print(f"\nUsing {num_stats_samples} samples for statistics calculation")
        print(f"Frame count per video: {config.frames}")

        mean_sum, squared_sum, pixel_count = 0.0, 0.0, 0
        for batch in tqdm(stats_loader, desc="Calculating statistics"):
            batch = batch.float()
            b, f, h, w, c = batch.shape
            batch = batch.reshape(-1, c)
            mean_sum += batch.sum(dim=0)
            squared_sum += (batch**2).sum(dim=0)
            pixel_count += batch.shape[0]

        mean: torch.Tensor = mean_sum / pixel_count
        std: torch.Tensor = torch.sqrt((squared_sum / pixel_count) - (mean**2))

        print("\nDataset Statistics:")
        print(f"Mean: {mean.tolist()}")
        print(f"Std:  {std.tolist()}")
        print(f"Calculated from {num_stats_samples} samples ({pixel_count:,} pixels)")
        print("===========================\n")                    
    
    # Broadcast stats if distributed
    if torch.distributed.is_initialized():
        if mean is not None:
            mean = mean.cuda()
            std = std.cuda()
        mean_tensor = torch.zeros(3, device="cuda")
        std_tensor = torch.zeros(3, device="cuda")
        if config.is_ref_device:
            mean_tensor.copy_(mean)
            std_tensor.copy_(std)
        torch.distributed.broadcast(mean_tensor, 0)
        torch.distributed.broadcast(std_tensor, 0)
        mean = mean_tensor.cpu()
        std = std_tensor.cpu()    
    
    print(f"Rank: {config.gpu} - mean: {mean} - std: {std}")


    train_loader: DataLoader = get_distributed_video_dataloader(
        config, 
        split="train", 
        mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
        std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
        shuffle=True,
        num_replicas=config.world_size,
        rank=config.gpu,
    )
    val_loader: DataLoader = get_distributed_video_dataloader(
        config, 
        split="val", 
        mean=(mean.tolist() if mean is not None else [0.485, 0.456, 0.406]),
        std=(std.tolist() if std is not None else [0.229, 0.224, 0.225]),
        shuffle=False,
        num_replicas=config.world_size,
        rank=config.gpu,
    )

    # Create models
    video_encoder: VideoEncoder = ModelRegistry.get(
        name="video_encoder"
    )(
        backbone=config.model_name,
        input_channels=3,
        num_frames=config.frames,
        pretrained=config.pretrained,
        output_dim=512,
    )
    video_encoder = video_encoder.to(config.gpu).float()

    text_encoder: TextEncoder = ModelRegistry.get(
        name="text_encoder"
    )()
    text_encoder = text_encoder.to(config.gpu).float()

    video_encoder = DDP(
        video_encoder, 
        device_ids=[config.gpu], 
        find_unused_parameters=True
    )
    text_encoder = DDP(
        text_encoder, 
        device_ids=[config.gpu], 
        find_unused_parameters=True
    )

    # Make temperature a trainable parameter directly on the device
    log_temperature: nn.Parameter = nn.Parameter(
        torch.log(torch.tensor([config.temperature], dtype=torch.float32, device=config.gpu))
    )

    # Include the temperature parameter in the optimizer
    optimizer_class: torch.optim.Optimizer = getattr(torch.optim, config.optimizer)
    optimizer: torch.optim.Optimizer = optimizer_class(
        [
            {"params": video_encoder.parameters()},
            {"params": text_encoder.parameters()},
            {"params": [log_temperature]},
        ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler: LRScheduler = get_scheduler(
        scheduler_name=config.scheduler_type,
        optimizer=optimizer,
        num_epochs=config.epochs,
        factor=config.factor,
        step_size=config.lr_step_period,
    )

    scaler: GradScaler = GradScaler() if config.use_amp else None

    if config.is_ref_device:
        print("\n=== Dataset Information ===")
        print(f"Training:   {len(train_loader):,} videos")
        print(f"Validation: {len(val_loader):,} videos")
        print(f"Total:      {len(train_loader) + len(val_loader):,} videos")
        print(f"\nBatch Size: {config.batch_size}")
        print(f"Training Batches: {len(train_loader) // config.batch_size:,}")
        print(
            f"Validation Batches: {len(val_loader) // config.batch_size + (1 if len(val_loader) % config.batch_size else 0):,}"
        )
        print("===========================\n")

    return {
        "video_encoder": video_encoder,
        "text_encoder": text_encoder,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": config.gpu,
        "wandb_run": wandb_run,
        "scaler": scaler,
        "log_temp": log_temperature,
    }


@ProjectRegistry.register('contrastive_pretraining')
class ContrastivePretraining:
    def __init__(
        self, 
        config: HeartWiseConfig,
    ):
        self.config: HeartWiseConfig = config
    
    def run(self):
        training_setup: dict[str, Any] = load_train_objs(
            config=self.config
        )
                
        runner: VideoContrastiveLearningRunner = RunnerRegistry.get(
            name="video_contrastive_learning"
        )(
            config=self.config,
            device=self.config.gpu,
            world_size=self.config.world_size,
            train_loader=training_setup["train_loader"],
            val_loader=training_setup["val_loader"],
            video_encoder=training_setup["video_encoder"],
            text_encoder=training_setup["text_encoder"],
            wandb_wrapper=training_setup["wandb_run"],
            optimizer=training_setup["optimizer"],
            scaler=training_setup["scaler"],
            log_temp=training_setup["log_temp"],
            lr_scheduler=training_setup["scheduler"],
            loss_fn=get_loss_fn(self.config.loss_name),
        )
        
        runner.train()        
        # is_distributed = self.world_size > 1
        # text_encoder = training_setup["text_encoder"]
        # train_loader = training_setup["train_loader"]
        # val_loader = training_setup["val_loader"]
        # device = self.device
        # log_temp = training_setup["log_temp"] if "log_temp" in training_setup else None

        # best_val_loss = float("inf")
        # best_epoch = -1

        # # Main training loop
        # for epoch in range(self.args.epochs):
        #     if self.device == 0:
        #         print(f"\nEpoch {epoch + 1}/{self.args.epochs}")

        #     train_loss, train_metrics = train_epoch(
        #         args=self.args,
        #         video_encoder=training_setup["video_encoder"],
        #         text_encoder=text_encoder,
        #         dataloader=train_loader,
        #         optimizer=training_setup["optimizer"],
        #         device=self.device,
        #         wandb_run=training_setup["wandb_run"],
        #         rank=self.args.rank,
        #         world_size=self.world_size,
        #         epoch=epoch,
        #     )

        #     # Validate on validation-only embeddings (use_val_only_pool=True)
        #     val_loss_valpool, val_metrics_valpool, _ = validate_epoch(
        #         args=self.args,
        #         video_encoder=training_setup["video_encoder"],
        #         text_encoder=text_encoder,
        #         dataloader=val_loader,
        #         device=self.device,
        #         wandb_run=training_setup["wandb_run"],
        #         rank=self.args.rank,
        #         world_size=self.world_size,
        #         epoch=epoch,
        #         all_text_embeddings=val_text_embeddings,
        #         all_reports=val_reports,
        #         text_embedding_pickle_path=os.path.join(
        #             full_output_path, "val_text_embeddings.pkl"
        #         ),
        #         output_dir=full_output_path,
        #         report_to_global_index=val_report_to_index,
        #         use_val_only_pool=True,  # <-- Val-only retrievals
        #     )

        #     # Validate on global embeddings (train+val) (use_val_only_pool=False)
        #     val_loss_global, val_metrics_global, _ = validate_epoch(
        #         args=self.args,
        #         video_encoder=training_setup["video_encoder"],
        #         text_encoder=text_encoder,
        #         dataloader=val_loader,
        #         device=self.device,
        #         wandb_run=training_setup["wandb_run"],
        #         rank=self.args.rank,
        #         world_size=self.world_size,
        #         epoch=epoch,
        #         all_text_embeddings=all_global_text_embeddings,
        #         all_reports=all_global_reports,
        #         text_embedding_pickle_path=os.path.join(
        #             full_output_path, "global_text_embeddings.pkl"
        #         ),
        #         output_dir=full_output_path,
        #         report_to_global_index=report_to_global_index,
        #         use_val_only_pool=False,  # <-- Global retrievals without top/bottom examples
        #     )

        #     # Choose one for best model comparison (typically val-only)
        #     current_val_loss = val_loss_valpool

        #     if self.device == 0 and training_setup["wandb_run"] is not None:
        #         log_data = {
        #             "epoch": epoch,
        #             "train/loss": train_loss,
        #             "train/learning_rate": training_setup["optimizer"].param_groups[0]["lr"],
        #             "val_only/loss": val_loss_valpool,
        #             **{f"val_only/{k}": v for k, v in val_metrics_valpool.items()},
        #             "val_global/loss": val_loss_global,
        #             **{f"val_global/{k}": v for k, v in val_metrics_global.items()},
        #             "best_val_loss": best_val_loss,  # Log the current best_val_loss each epoch
        #         }
        #         # Log temperature if available
        #         if log_temp is not None:
        #             current_temp = torch.exp(log_temp).item()
        #             log_data["temperature"] = current_temp
        #             log_data["log_temp"] = log_temp.item()

        #         training_setup["wandb_run"].log(log_data)

        #     if training_setup["scheduler"] is not None:
        #         training_setup["scheduler"].step()

        #     if self.device == 0:
        #         model_dict = {
        #             "video_encoder": (
        #                 training_setup["video_encoder"].module.state_dict()
        #                 if is_distributed
        #                 else training_setup["video_encoder"].state_dict()
        #             ),
        #             "text_encoder": (
        #                 text_encoder.module.state_dict()
        #                 if is_distributed
        #                 else text_encoder.state_dict()
        #             ),
        #             "optimizer": training_setup["optimizer"].state_dict(),
        #             "scheduler": (
        #                 training_setup["scheduler"].state_dict()
        #                 if training_setup["scheduler"] is not None
        #                 else None
        #             ),
        #             "epoch": epoch,
        #         }

        #         metrics_dict = {
        #             "train_loss": train_loss,
        #             "val_loss_valpool": val_loss_valpool,
        #             "val_loss_global": val_loss_global,
        #             "best_val_loss": best_val_loss,
        #             "best_epoch": best_epoch,
        #             **train_metrics,
        #             **val_metrics_valpool,  # store the val-only metrics
        #             **{f"global_{k}": v for k, v in val_metrics_global.items()},
        #         }

        #         checkpoint_dir = Path(full_output_path) / "checkpoints"
        #         checkpoint_dir.mkdir(parents=True, exist_ok=True)

        #         latest_path = checkpoint_dir / "latest.pt"
        #         save_checkpoint(model_dict, metrics_dict, latest_path)
        #         print(f"\nSaved latest checkpoint at epoch {epoch + 1}")

        #         # Update best model based on val-only performance
        #         if current_val_loss < best_val_loss:
        #             previous_best = best_val_loss
        #             best_val_loss = current_val_loss
        #             best_epoch = epoch
        #             best_path = checkpoint_dir / "best.pt"
        #             save_checkpoint(model_dict, metrics_dict, best_path, is_best=True)
        #             print(
        #                 f"\nNew best model saved! Val Loss (val-only): {current_val_loss:.4f} (previous: {previous_best:.4f})"
        #             )

        #             if training_setup["wandb_run"] is not None:
        #                 # Also log the new best_val_loss immediately when found
        #                 training_setup["wandb_run"].log(
        #                     {
        #                         "best_val_loss": best_val_loss,
        #                         "best_epoch": best_epoch,
        #                         "epoch": epoch,
        #                     }
        #                 )
        #                 training_setup["wandb_run"].save(str(best_path))      

        wandb.finish()


