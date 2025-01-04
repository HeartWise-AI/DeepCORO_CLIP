import os
import pickle

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from tqdm import tqdm

from utils.ddp import gather_loss
from utils.config import HeartWiseConfig
from utils.registry import RunnerRegistry
from utils.files_handler import generate_output_dir_name
from utils.text_encoder import (
    create_global_text_pool, 
    precompute_global_text_embeddings
)
from utils.metrics import (
    compute_mrr, 
    compute_map,
    compute_ndcg,
    compute_median_rank,
    compute_recall_at_k, 
    compute_embedding_norms, 
    compute_alignment_score, 
    compute_similarity_matrix
)

from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder


def runner_setup(
    config: HeartWiseConfig, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    wandb_wrapper
):
    # Generate output directory
    run_id = wandb_wrapper.id if wandb_wrapper is not None else "run_id_001"
    output_subdir = generate_output_dir_name(config, run_id)
    full_output_path = os.path.join(config.output_dir, output_subdir)        
    os.makedirs(full_output_path, exist_ok=True)
    print("Args: ", config)      
    
    # === Create Global Pool (train + val) ===
    all_global_reports = create_global_text_pool(train_loader, val_loader, None)
    
    # === Create Validation-Only Pool ===
    val_reports = val_loader.dataset.get_all_reports()
    val_unique_reports = list(dict.fromkeys(val_reports))  # Preserve order, remove duplicates
    
    return {
        "full_output_path": full_output_path,
        "all_global_reports": all_global_reports,
        "val_reports": val_unique_reports,
    }


@RunnerRegistry.register('video_contrastive_learning')
class VideoContrastiveLearningRunner:
    def __init__(
        self,
        config: HeartWiseConfig,
        device: int,
        world_size: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        video_encoder: VideoEncoder,
        text_encoder: TextEncoder,
        wandb_wrapper,
        optimizer: AdamW,
        scaler: GradScaler,
        lr_scheduler: LRScheduler,
        loss_fn: callable,
    ):
        self.config: HeartWiseConfig = config
        self.device: int = device
        self.world_size: int = world_size
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.video_encoder: VideoEncoder = video_encoder
        self.text_encoder: TextEncoder = text_encoder
        self.wandb_wrapper = wandb_wrapper
        self.optimizer: AdamW = optimizer
        self.scaler: GradScaler = scaler
        self.lr_scheduler: LRScheduler = lr_scheduler
        self.loss_fn: callable = loss_fn
        self.setup_dict = None
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def _compute_text_embeddings(self, reports: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
        """Compute text embeddings for a list of reports."""
        report_to_index = {r: i for i, r in enumerate(reports)}
        
        # Tokenize all reports
        encoded_texts = self.train_loader.dataset.tokenizer(
            reports,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded_texts["input_ids"].to(self.device)
        attention_mask = encoded_texts["attention_mask"].to(self.device)
        
        # Compute embeddings in batches
        batch_size = 256
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(reports), batch_size):
                batch_input_ids = input_ids[i:i + batch_size]
                batch_attention_mask = attention_mask[i:i + batch_size]
                
                embeddings = self.text_encoder(batch_input_ids, batch_attention_mask)
                all_embeddings.append(embeddings)
                
        text_embeddings = torch.cat(all_embeddings, dim=0)
        return text_embeddings, report_to_index

    def train(self):
        self.setup_dict = runner_setup(
            config=self.config,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            wandb_wrapper=self.wandb_wrapper
        )
        
        # Training loop
        for epoch in range(self.config.epochs):           
            # Training phase
            train_metrics = self._run_epoch(mode="train", epoch=epoch)
            
            if self.device == 0:
                self.wandb_wrapper.log({
                    **train_metrics
                })

            
            # # Validation phase with pool
            # val_metrics = self._run_epoch(
            #     mode="val",
            #     epoch=epoch,
            # )
            
            # # Update best model based on val-only performance
            # current_val_loss = val_metrics["val/loss"]
            # if current_val_loss < self.best_val_loss:
            #     previous_best = self.best_val_loss
            #     self.best_val_loss = current_val_loss
            #     self.best_epoch = epoch
                
            #     if self.device == 0:
            #         self._save_checkpoint(
            #             epoch=epoch,
            #             metrics={
            #                 **train_metrics,
            #                 **val_metrics
            #             },
            #             is_best=True
            #         )
            #         print(f"\nNew best model saved! Val Loss: {current_val_loss:.4f} "
            #               f"(previous: {previous_best:.4f})")

            # # Log metrics only on reference device
            # if self.device == 0 and self.wandb_wrapper:
            #     self.wandb_wrapper.log({
            #         "epoch": epoch,
            #         **train_metrics,
            #         **val_metrics,
            #         "best_val_loss": self.best_val_loss,
            #         "best_epoch": self.best_epoch
            #     })
                
            # # Save latest checkpoint
            # if self.device == 0:
            #     self._save_checkpoint(
            #         epoch=epoch,
            #         metrics={
            #             **train_metrics,
            #             **val_metrics
            #         }
            #     )
                
            # Learning rate scheduler step if it exists
            if self.lr_scheduler:
                self.lr_scheduler.step()

    def _run_epoch(
        self, 
        mode: str,
        epoch: int,
    ) -> dict[str, float]:
        assert mode in ["train", "val"]
        
        # Set model mode
        self.video_encoder.train(mode == "train")
        self.text_encoder.train(mode == "train")
        
        # Initialize metrics
        total_loss = 0.0
        epoch_metrics = {}
        
        # Get appropriate dataloader
        dataloader = self.train_loader if mode == "train" else self.val_loader
        
        # Progress bar
        tqdm_desc = f'Running {mode} Epoch {epoch + 1}'
        progress = tqdm(dataloader, desc=tqdm_desc) if self.device == 0 else dataloader

        # Collect embeddings for validation
        all_video_embeddings = []
        all_text_embeddings = []
        all_paths = []
        all_ground_truth_reports = []

        # Initialize batch_idx before the loop
        batch_idx = 0
        
        for batch_idx, batch in enumerate(progress, start=1):
            if batch["videos"] is None or batch["encoded_texts"] is None:
                continue
                
            inputs = self._preprocess_inputs(batch)
            
            if mode == "train":
                batch_metrics, embeddings = self._train_step(**inputs)
                all_video_embeddings.append(embeddings["video_embeddings"].cpu())
                all_text_embeddings.append(embeddings["text_embeddings"].cpu())
                all_paths.extend(inputs["paths"])
                all_ground_truth_reports.extend(
                    self.train_loader.dataset.get_reports(inputs["paths"])
                )
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                            
                # Update progress bar
                total_loss += batch_metrics["loss"]
                if self.device == 0:
                    progress.set_postfix({
                        f"{mode}_loss": f"{batch_metrics['loss']:.4f}",
                        "avg_loss": f"{(total_loss / batch_idx):.4f}"
                    })
                    
            else:  # Validation mode
                with torch.no_grad():
                    video_features = self.video_encoder(inputs["videos"])
                    text_features = self.text_encoder(
                        inputs["input_ids"], 
                        inputs["attention_mask"]
                    )
                                        
                    loss = self.loss_fn(video_features, text_features)
                    
                    epoch_metrics["loss"] = epoch_metrics.get("loss", 0) + loss.item()
                    
                    # Store embeddings for later computation
                    all_video_embeddings.append(video_features.cpu())
                    all_text_embeddings.append(text_features.cpu())
                    all_paths.extend(inputs["paths"])
                    all_ground_truth_reports.extend(
                        self.val_loader.dataset.get_reports(inputs["paths"])
                    )

        # Raise error if no batches were processed
        if batch_idx == 0:
            raise ValueError("No batches were processed. Check your data loader and input pipeline.")

        # At the end of the epoch, compute means and gather across processes
        if mode == "train":
            # Compute means for all metrics
            epoch_metrics = {
                k: v / batch_idx for k, v in epoch_metrics.items()
            }

            # Process embeddings for recall computation
            all_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(self.device)
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(self.device)
                        
            # Normalize embeddings
            all_video_embeddings_normalized = nn.functional.normalize(all_video_embeddings, dim=1)
            all_text_embeddings_normalized = nn.functional.normalize(all_text_embeddings, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(all_video_embeddings_normalized, all_text_embeddings_normalized.T)
            
            # Create ground truth indices (diagonal matrix since each video matches with its text)
            ground_truth_indices = torch.arange(len(all_ground_truth_reports), device=self.device)
            
            # Compute recall metrics
            recall_metrics = compute_recall_at_k(similarity_matrix, k_values=[1, 5])
            mrr_score = compute_mrr(similarity_matrix)
            map_score = compute_map(similarity_matrix, ground_truth_indices)
            median_rank_score = compute_median_rank(similarity_matrix, ground_truth_indices)
            ndcg_score = compute_ndcg(similarity_matrix, ground_truth_indices)
            
            # Update dictionary with hashmap type metrics
            epoch_metrics.update(recall_metrics)
            epoch_metrics.update(mrr_score)
            
            # Update dictionary with float type metrics
            epoch_metrics['map'] = map_score
            epoch_metrics['median_rank'] = median_rank_score
            epoch_metrics['ndcg'] = ndcg_score
            
            # Gather and average across all processes
            gathered_metrics = {
                f"{mode}/{k}": gather_loss([v], self.device) 
                for k, v in epoch_metrics.items()
            }
            
            return gathered_metrics

        # Process validation metrics
        if mode == "val":
            all_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(self.device)
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(self.device)
            print(f"all_video_embeddings: {all_video_embeddings.shape}")
            print(f"all_text_embeddings: {all_text_embeddings.shape}")
            
            # Create report to index mapping
            report_to_index = {r: i for i, r in enumerate(all_ground_truth_reports)}
            
            # Compute norms before normalization
            epoch_metrics.update(compute_embedding_norms(all_video_embeddings, all_text_embeddings))
            
            # Normalize embeddings
            all_video_embeddings_normalized = nn.functional.normalize(all_video_embeddings, dim=1)
            all_text_embeddings_normalized = nn.functional.normalize(all_text_embeddings, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(all_video_embeddings_normalized, all_text_embeddings_normalized.T)
            
            # Compute metrics
            if similarity_matrix.size(0) >= 5 and similarity_matrix.size(1) >= 5:
                global_ground_truth_indices = list(range(len(all_ground_truth_reports)))  # Each video maps to its corresponding text
                global_ground_truth_indices_tensor = torch.tensor(
                    global_ground_truth_indices, 
                    device=self.device
                )
                
                epoch_metrics.update(compute_recall_at_k(
                    similarity_matrix, 
                    global_ground_truth_indices_tensor,
                    k_values=[1, min(5, similarity_matrix.size(1))]
                ))
                epoch_metrics.update(compute_mrr(
                    similarity_matrix, 
                    global_ground_truth_indices_tensor
                ))
                
                epoch_metrics["alignment_score"] = compute_alignment_score(
                    all_video_embeddings,
                    all_text_embeddings
                )

        # Add average loss to metrics
        epoch_metrics["loss"] = epoch_metrics.get("loss", 0) / batch_idx

        return {
            **{f"{mode}/{k}": gather_loss([v], self.device) for k, v in epoch_metrics.items()}
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint_dir: str = os.path.join(self.setup_dict["full_output_path"], "checkpoints")
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
        }
        
        checkpoint = {
            **model_dict,
            **metrics,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch
        }
        
        # Save checkpoint
        save_path = os.path.join(
            checkpoint_dir, 
            "best.pt" if is_best else "latest.pt"
        )
        torch.save(checkpoint, save_path)
        print(f"\nSaved {'best' if is_best else 'latest'} checkpoint at epoch {epoch + 1}")

    def _run_batch(
        self, 
        batch: dict[str, torch.Tensor], 
        step_fn: callable
    ) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = self._preprocess_inputs(batch)               
        return step_fn(**inputs)

    def _preprocess_inputs(
        self, 
        batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "videos": batch["videos"].to(self.device).float(),
            "input_ids": batch["encoded_texts"]["input_ids"].to(self.device), 
            "attention_mask": batch["encoded_texts"]["attention_mask"].to(self.device),
            "paths": batch["paths"]
        }

    def _train_step(
        self, 
        videos: torch.Tensor,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        paths: list[str]
    ) -> dict[str, torch.Tensor]:  
        # Zero gradients at the start
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.scaler:
            with torch.amp.autocast('cuda'):
                # Get features
                video_embeddings = self.video_encoder(videos)
                text_embeddings = self.text_encoder(input_ids, attention_mask)

                # Compute loss
                loss = self.loss_fn(video_embeddings, text_embeddings)

            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Convert embeddings to float32 for metrics computation (AMP can cause issues later on)
            video_embeddings = video_embeddings.float()
            text_embeddings = text_embeddings.float()                    
            
        else:
            # Get features
            video_embeddings = self.video_encoder(videos)
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            
            # Compute loss
            loss = self.loss_fn(video_embeddings, text_embeddings)  # Use normalized features
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.video_encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=5.0)
            
            self.optimizer.step()           
        
        # Compute metrics
        with torch.no_grad():            
            # Compute metrics
            embedding_norms = compute_embedding_norms(video_embeddings, text_embeddings)
            alignment_score = compute_alignment_score(video_embeddings, text_embeddings)
            
            metrics = {
                **embedding_norms,
                "alignment_score": alignment_score
            }
            
        return {
            "loss": loss.detach(), 
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{k: v.detach() if torch.is_tensor(v) else v for k, v in metrics.items()}
        }, {
            "video_embeddings": video_embeddings,
            "text_embeddings": text_embeddings
        }

    def _val_step(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        paths: list[str]
    ) -> dict[str, torch.Tensor]:
        """Validation step function."""
        
        # get reports
        reports = self.val_loader.dataset.get_reports(paths)
                
        with torch.no_grad():
            # Get features
            video_features = self.video_encoder(videos)
            text_features = self.text_encoder(input_ids, attention_mask)
            
            # Normalize features
            normalized_video = nn.functional.normalize(video_features, dim=1)
            normalized_text = nn.functional.normalize(text_features, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(normalized_video, normalized_text.t())
            
            # Compute loss
            loss = self.loss_fn(video_features, text_features)

            # Compute metrics
            metrics = compute_recall_at_k(similarity_matrix)
            metrics.update(compute_mrr(similarity_matrix))
            metrics.update(compute_embedding_norms(video_features, text_features))
            metrics.update({'alignment_score': compute_alignment_score(video_features, text_features)})

            return {
                "loss": loss.detach(),
                **{k: v.detach() if torch.is_tensor(v) else v for k, v in metrics.items()}
            }


    def validate(self):
        pass

    def test(self):
        pass