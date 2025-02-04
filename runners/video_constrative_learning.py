import os
import wandb
import numpy as np

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
from utils.metrics import (
    compute_mrr, 
    compute_map,
    compute_ndcg_at_k,
    compute_median_rank,
    compute_recall_at_k, 
    compute_embedding_norms, 
    compute_alignment_score
)
from utils.logging import (
    log_best_worst_retrievals, 
    log_gradient_norms,
    save_retrieval_results
)

from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder


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
        optimizer: AdamW,
        scaler: GradScaler,
        log_temp: torch.Tensor,
        lr_scheduler: LRScheduler,
        loss_fn: callable,
        output_dir: str,
    ):
        # Add validation for recall_k
        if not isinstance(config.recall_k, list):
            raise ValueError("config.recall_k must be a list of integers, "
                             f"got {type(config.recall_k)}")
        if not isinstance(config.ndcg_k, list):
            raise ValueError("config.ndcg_k must be a list of integers, "
                             f"got {type(config.ndcg_k)}")
        
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
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.log_temp = log_temp
        
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
        if self.config.resume_training and not self.config.checkpoint:
            raise ValueError("Flag 'resume_training' is set, but no checkpoint provided")
        
        if self.config.resume_training and self.config.checkpoint:
            print(f"[VideoContrastiveLearningRunner] Resuming from checkpoint: {self.config.checkpoint}")
            wandb_run, start_epoch, best_val_loss, best_epoch = self._preview_checkpoint_for_resuming(
                self.config.checkpoint
            )
        
        # Training loop
        for epoch in range(self.config.epochs):           
            # Synchronize before epoch starts
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])
                
            # Training phase
            train_metrics = self._run_epoch(
                mode="train", 
                epoch=epoch
            )
            if self.config.is_ref_device:
                wandb.log(train_metrics)
            
            # Synchronize before validation
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])
                
            # Validation phase
            val_metrics = self._run_epoch(
                mode="val", 
                epoch=epoch
            )
            if self.config.is_ref_device:
                wandb.log(val_metrics)
            
            # Synchronize after validation
            if self.world_size > 1:
                torch.distributed.barrier(device_ids=[self.device])
            
            # Update best model based on validation performance
            if val_metrics["val/loss"] < self.best_val_loss:
                previous_best = self.best_val_loss
                self.best_val_loss = val_metrics["val/loss"]
                self.best_epoch = epoch
                if self.config.is_ref_device:   
                    print(f"\nNew best model! Val Loss: {val_metrics['val/loss']:.4f} "
                          f"(previous: {previous_best:.4f})")
                    self._save_checkpoint(
                        epoch=epoch,
                        metrics={
                            **train_metrics,
                            **val_metrics
                        },
                        is_best=True
                    )                    
                                
            if self.config.is_ref_device:
                self._save_checkpoint(
                    epoch=epoch,
                    metrics={
                        **train_metrics,
                        **val_metrics
                    },
                    is_best=False
                )
                
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
        
        # Get appropriate step function
        step_fn = self._train_step if mode == "train" else self._val_step      
        
        # Progress bar
        tqdm_desc = f'Running {mode} Epoch {epoch + 1}'
        progress = tqdm(dataloader, desc=tqdm_desc) if self.device == 0 else dataloader

        # Collect embeddings for validation
        all_video_embeddings = []
        all_text_embeddings = []
        all_paths = []
        all_ground_truth_reports = []
                
        # Run batches
        for batch_idx, batch in enumerate(progress, start=1):
            if batch["videos"] is None or batch["encoded_texts"] is None:
                continue
                
            step_inputs, paths_or_sids = self._preprocess_inputs(batch)

            # If not multi-video, add a singleton dimension to the video tensor 
            if not self.config.multi_video:
                step_inputs['videos'] = step_inputs['videos'].unsqueeze(1) # VideoEncoder expects (B,Nvideos,T,H,W,3)
            
            # Run step
            batch_metrics, embeddings = step_fn(**step_inputs)
            
            # Append embeddings to lists
            all_video_embeddings.append(embeddings["video_embeddings"].cpu())
            all_text_embeddings.append(embeddings["text_embeddings"].cpu())
            if self.config.multi_video:
                for sid in paths_or_sids:
                    # get the *list* of underlying video paths
                    vid_list = dataloader.dataset.get_video_paths(sid)
                    if len(vid_list) > 0:
                        all_paths.append(vid_list[0])
                    else:
                        all_paths.append(str(sid))
            else:
                all_paths.extend(paths_or_sids)
                
            all_ground_truth_reports.extend(
                dataloader.dataset.get_reports(paths_or_sids)
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

            del batch_metrics, embeddings
            torch.cuda.empty_cache()

        # Compute means for all metrics
        epoch_metrics = {
            k: v / batch_idx for k, v in epoch_metrics.items()
        }
        
        # Process embeddings for recall computation
        all_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(self.device)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(self.device)
                
        # Get texts and create mapping
        text_to_idx = {text: idx for idx, text in enumerate(all_ground_truth_reports)}
        
        # Create ground truth indices - map each video to index of its text
        ground_truth_indices = torch.tensor([
            text_to_idx[text] for text in all_ground_truth_reports
        ], device=self.device)
        
        # # Get embeddings for unique texts only
        # unique_text_embeddings = []
        # for text in all_ground_truth_reports:
        #     # Find first occurrence of this text
        #     idx = all_ground_truth_reports.index(text)
        #     unique_text_embeddings.append(all_text_embeddings[idx])
        # unique_text_embeddings = torch.stack(unique_text_embeddings)
                    
        # Normalize embeddings
        all_video_embeddings_normalized = nn.functional.normalize(all_video_embeddings, dim=1)
        all_text_embeddings_normalized = nn.functional.normalize(all_text_embeddings, dim=1)
        
        # After computing similarity matrix and before computing metrics
        if mode == "val" and self.config.is_ref_device:
            # Compute similarity matrix between videos and unique texts
            similarity_matrix = torch.matmul(
                all_video_embeddings_normalized, 
                all_text_embeddings_normalized.T
            )
            
            log_best_worst_retrievals(
                similarity_matrix=similarity_matrix,
                all_paths=all_paths,
                unique_texts=all_ground_truth_reports,
                ground_truth_indices=ground_truth_indices,
                epoch=epoch
            )
            save_retrieval_results(
                similarity_matrix=similarity_matrix,
                all_paths=all_paths,
                all_ground_truth_reports=all_ground_truth_reports,
                report_to_global_index=text_to_idx,
                epoch=epoch,
                output_dir=self.output_dir
            )

            # Compute metrics on this GPU's portion of data
            recall_metrics = compute_recall_at_k(similarity_matrix, ground_truth_indices, k_values=self.config.recall_k)
            mrr_score = compute_mrr(similarity_matrix, ground_truth_indices)
            map_score = compute_map(similarity_matrix, ground_truth_indices)
            median_rank_score = compute_median_rank(similarity_matrix, ground_truth_indices)
            ndcg_score = compute_ndcg_at_k(similarity_matrix, ground_truth_indices, k_values=self.config.ndcg_k)
            
            # Update dictionary with hashmap type metrics
            epoch_metrics.update(recall_metrics)
            epoch_metrics.update(mrr_score)
            epoch_metrics.update(ndcg_score)
            
            # Update dictionary with float type metrics
            epoch_metrics['MAP'] = map_score
            epoch_metrics['MedianRank_V2T'] = median_rank_score

        # Gather and average all metrics across GPUs
        gathered_metrics = {
            f"{mode}/{k}": gather_loss([v], self.device) 
            for k, v in epoch_metrics.items()
        }
        
        return gathered_metrics

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint_dir: str = os.path.join(self.output_dir, "checkpoints")
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
            f"best_epoch.pt" if is_best else f"checkpoint.pt"
        )
        torch.save(checkpoint, save_path)
        print(f"\nSaved {'best' if is_best else 'latest'} checkpoint at epoch {epoch + 1}")

    def _preview_checkpoint_for_resuming(self, checkpoint_path: str):
        """
        Quick "preview" function that loads minimal fields from the checkpoint 
        on CPU, to retrieve wandb_run, epoch, best_val_loss, best_epoch, etc.
        Returns:
        wandb_run (str or None)
        start_epoch (int)
        best_val_loss (float)
        best_epoch (int)
        """
        if not os.path.isfile(checkpoint_path):
            print(f"Warning: checkpoint not found at {checkpoint_path}.")
            return None, 0, float('inf'), -1

        print(f"[Preview] Loading minimal info from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        wandb_run = checkpoint.get("wandb_run", None)
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_epoch = checkpoint.get("best_epoch", -1)

        print(
            f"[Preview] Found run_id={wandb_run}, "
            f"start_epoch={start_epoch}, best_val_loss={best_val_loss}, best_epoch={best_epoch}"
        )
        return wandb_run, start_epoch, best_val_loss, best_epoch

    def _load_full_checkpoint(self, checkpoint_path: str, device, training_setup):
        """
        After the model/optimizer/etc. is initialized, call this 
        to actually load states from the checkpoint into them.
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
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        return {
            "videos": batch["videos"].to(self.device).float(),
            "input_ids": batch["encoded_texts"]["input_ids"].to(self.device), 
            "attention_mask": batch["encoded_texts"]["attention_mask"].to(self.device)
        }, batch["paths"]

    def _train_step(
        self, 
        videos: torch.Tensor,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:  
        # Zero gradients at the start
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.scaler:
            with torch.amp.autocast('cuda'):
                # Get features
                video_embeddings = self.video_encoder(videos)
                text_embeddings = self.text_encoder(input_ids, attention_mask)

                # Compute loss
                loss = self.loss_fn(video_embeddings, text_embeddings, self.log_temp)

            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Unscale once here
            self.scaler.unscale_(self.optimizer)

            # Now log gradient norms
            log_gradient_norms(
                self.video_encoder,
                prefix="train/video_encoder/",
            )
            log_gradient_norms(
                self.text_encoder,
                prefix="train/text_encoder/",
            )
            
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
            loss = self.loss_fn(video_embeddings, text_embeddings, self.log_temp)  # Use normalized features
            
            loss.backward()

            # Now log gradient norms
            log_gradient_norms(
                self.video_encoder,
                prefix="train/video_encoder/",
            )
            log_gradient_norms(
                self.text_encoder,
                prefix="train/text_encoder/",
            )
            
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
            **{f"lr/{param_group['name']}": param_group['lr'] for param_group in self.optimizer.param_groups},
            "temperature": self.log_temp.exp().detach(), # add temperature to metrics for wandb logging
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
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # Get features
                video_features = self.video_encoder(videos)
                text_features = self.text_encoder(
                    input_ids, 
                    attention_mask
                )
                                
                loss = self.loss_fn(video_features, text_features, self.log_temp)

            # Compute metrics
            embedding_norms = compute_embedding_norms(video_features, text_features)
            alignment_score = compute_alignment_score(video_features, text_features)
            
        metrics = {
            **embedding_norms,
            "alignment_score": alignment_score
        }
        
        return {
            "loss": loss.detach(),
            "temperature": self.log_temp.exp().detach(), # add temperature to metrics for wandb logging
            **{k: v.detach() if torch.is_tensor(v) else v for k, v in metrics.items()}
        }, {
            "video_embeddings": video_features.float(),
            "text_embeddings": text_features.float()
        }


    def validate(self):
        pass

    def test(self):
        pass