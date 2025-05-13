import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm, trange
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Any
from contextlib import nullcontext
import torch.nn.functional as F

from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.enums import RunMode
from utils.wandb_wrapper import WandbWrapper
from utils.loss.typing import Loss
from utils.metrics import compute_best_threshold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

@RunnerRegistry.register("DeepCORO_Multiview")
@RunnerRegistry.register("DeepCORO_Multiview_test")
class MultiviewRunner:
    """
    Runner pour train/validation multivues, utilisant directement les embeddings
    moyens par examen fournis par le DataLoader.
    """

    def __init__(
        self,
        config: Any,
        wandb_wrapper: WandbWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        scaler: GradScaler,
        output_dir: str,
        loss_fn: Loss,
        video_encoder,
        linear_probing,
    ):
        self.config = config
        self.wandb_wrapper = wandb_wrapper
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.output_dir = output_dir
        self.loss_fn = loss_fn
        self.video_encoder = video_encoder
        self.linear_probing = linear_probing

        self.run_mode = getattr(config, "run_mode", None)
        self.use_amp = getattr(config, "use_amp", False)
        self.scheduler_name = getattr(config, "scheduler_name", None)
        self.scheduler_per_iteration = self._scheduler_is_per_iteration()
        self.best_val_auc = -np.inf
        if DistributedUtils.is_initialized():
            self.world_size = DistributedUtils.dist.get_world_size()
        else:
            self.world_size = 1
            
        self.use_amp = getattr(config, "use_amp", False)
        if not torch.cuda.is_available():
            self.use_amp = False
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Ensure W&B is initialized if use_wandb is True
        if self.config.use_wandb and not self.wandb_wrapper.is_initialized():
            self.wandb_wrapper = WandbWrapper(
                config=self.config,
                initialized=True,
                is_ref_device=True
            )
            wandb.init(
                project=self.config.project_name,
                entity=self.config.entity_name,
                config=self.config.to_dict(),
                allow_val_change=True
            )
            print("[DEBUG] W&B initialized successfully.")

    def _scheduler_is_per_iteration(self):
        sched_name = getattr(self.config, "scheduler_name", "").lower()
        return "warmup" in sched_name
    def _save_predictions(self, mode, epoch, preds, targets, exam_ids):
        # Concatène toutes les prédictions et cibles
        pred_arr = torch.cat(preds, dim=0).numpy()
        tgt_arr = torch.cat(targets, dim=0).numpy()
        
        # Ensure all arrays have the same length
        if not (len(exam_ids) == len(pred_arr) == len(tgt_arr)):
            raise ValueError(f"Mismatched lengths: exam_ids({len(exam_ids)}), pred_arr({len(pred_arr)}), tgt_arr({len(tgt_arr)})")
        
        df = pd.DataFrame({
            'examen_id': exam_ids,
            'pred': pred_arr.tolist(),
            'true': tgt_arr.tolist(),
            'epoch': epoch
        })
        os.makedirs(os.path.join(self.output_dir, "predictions"), exist_ok=True)
        df.to_csv(os.path.join(self.output_dir, "predictions", f"{mode}_predictions_epoch_{epoch}.csv"), index=False)

    def train(self) -> float:
        """
        Entraînement multiview, sans recalcul des moyennes des embeddings,
        car ils sont déjà fournis par le DataLoader.
        """
        print(f"Run mode: {self.run_mode}, AMP: {self.use_amp}, Scheduler: {self.scheduler_name}")
        self.video_encoder.train()
        self.linear_probing.train()
        # === DEBUG ICI ===
        print("Début de l'entraînement")
        print(f"Nombre de batches dans le train_loader : {len(self.train_loader)}")
        for batch in self.train_loader:
            print("Batch reçu")
            break  # juste pour voir si on passe ici
    # === FIN DEBUG ===


        total_loss = 0.0
        total_batches = 0
        accumulated_preds = defaultdict(list)
        accumulated_targets = defaultdict(list)

        pbar = tqdm(
            self.train_loader,
            desc=f"[GPU {self.config.device}] Train Ép. {self.current_epoch+1}/{self.config.epochs}",
            unit="batch",
            ncols=100,
            leave=False
        )
        
        for batch in pbar:
            # Récupération des embeddings moyens et labels au niveau examen
            grouped_embs = batch['exam_embedding'].to(self.config.device)
            print("Variance embeddings (batch):", grouped_embs.var().item())  # <-- AJOUTE ICI
            grouped_embs = F.normalize(grouped_embs, dim=1)  # <-- Normalisation L2 sur la dimension des features
            grouped_labels = batch['target_label'].to(self.config.device).float()
            exam_ids = batch.get('examen_id', None)

            # Linear probing & loss
            if torch.cuda.is_available() and self.use_amp:
                autocast_ctx = torch.amp.autocast(device_type='cuda')
            else:
                autocast_ctx = nullcontext()
            with autocast_ctx:
                logits = {h: m(grouped_embs.to(next(m.parameters()).device)) for h, m in self.linear_probing.heads.items()}
                targets = {h: grouped_labels[:, i].to(next(self.linear_probing.heads[h].parameters()).device) for i, h in enumerate(self.config.head_structure.keys())}
                loss = self.loss_fn.run(outputs=logits, targets=targets)["main"]
            # === Ici, on accumule pour chaque head ===
            for i, h in enumerate(self.config.head_structure.keys()):
                logit = logits[h].detach().cpu()
                if self.config.head_structure[h] == 1:
                    pred = torch.sigmoid(logit)
                else:
                    pred = torch.softmax(logit, dim=1)
                accumulated_preds[h].append(pred)
                accumulated_targets[h].append(targets[h].cpu())
            
                print(f"[{h}] Labels (batch): {targets[h].cpu().numpy()}")
                print(f"[{h}] Prédictions (batch): {pred.numpy()}")
                print(f"[{h}] Moyenne labels: {targets[h].float().mean().item():.4f}, Moyenne préd: {pred.float().mean().item():.4f}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler_per_iteration and self.lr_scheduler:
                self.lr_scheduler.step()

            total_loss += loss.item()
            total_batches += 1
            avg_loss = total_loss / total_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            DistributedUtils.sync_process_group(world_size=self.world_size, device_ids=self.config.device)

        pbar.close()
        avg = total_loss / total_batches if total_batches > 0 else float('nan')
        # === Agrégation multi-GPU ici ===
        if self.world_size > 1:
            for head in accumulated_preds:
                local_preds = torch.cat(accumulated_preds[head], dim=0)
                local_targets = torch.cat(accumulated_targets[head], dim=0)
                gathered_preds = DistributedUtils.gather_tensor(local_preds, self.world_size)
                gathered_targets = DistributedUtils.gather_tensor(local_targets, self.world_size)
                accumulated_preds[head] = [gathered_preds]
                accumulated_targets[head] = [gathered_targets]

        print(f"[DEBUG] Train Epoch metrics: loss={avg:.4f}")
        self._log_epoch_metrics("train", self.current_epoch, accumulated_preds, accumulated_targets, avg)
        return avg

    def validation(self) -> float:
        """
        Validation multiview, utilisant les embeddings moyens fournis.
        """
        print(f"Run mode: {self.run_mode}, AMP: {self.use_amp}, Scheduler: {self.scheduler_name}")
        self.video_encoder.eval()
        self.linear_probing.eval()

        total_loss = 0.0
        total_batches = 0
        accumulated_preds = defaultdict(list)
        accumulated_targets = defaultdict(list)
        accumulated_exam_ids = defaultdict(list)

        pbar = tqdm(
            self.val_loader,
            desc=f"[GPU {self.config.device}] Val Ép. {self.current_epoch+1}/{self.config.epochs}",
            unit="batch",
            ncols=100,
            leave=False
        )

        with torch.no_grad():
            for batch in pbar:
                print("Batch keys:", batch.keys())  
                grouped_embs = batch['exam_embedding'].to(self.config.device)
                print("Variance embeddings (batch):", grouped_embs.var().item())  # <-- AJOUTE ICI
                grouped_embs = F.normalize(grouped_embs, dim=1)  # <-- Normalisation L2 sur la dimension des features
                grouped_labels = batch['target_label'].to(self.config.device).float()
                exam_ids = batch.get('examen_id', None)

                # Debugging lengths of data
                print(f"Batch exam_ids: {len(exam_ids) if exam_ids else 'None'}")
                print(f"Batch grouped_embs: {grouped_embs.shape}")
                print(f"Batch grouped_labels: {grouped_labels.shape}")

                # Ensure exam_ids is not None and matches the batch size
                if exam_ids is None or len(exam_ids) != grouped_embs.shape[0]:
                    raise ValueError(f"Mismatch in batch sizes: exam_ids({len(exam_ids) if exam_ids else 'None'}), grouped_embs({grouped_embs.shape[0]})")

                if torch.cuda.is_available() and self.use_amp:
                    autocast_ctx = torch.amp.autocast(device_type='cuda')
                else:
                    autocast_ctx = nullcontext()
                with autocast_ctx:
                    logits = {h: m(grouped_embs.to(next(m.parameters()).device)) for h, m in self.linear_probing.heads.items()}
                    targets = {h: grouped_labels[:, i].to(next(self.linear_probing.heads[h].parameters()).device) for i, h in enumerate(self.config.head_structure.keys())}
                    loss = self.loss_fn.run(outputs=logits, targets=targets)["main"]

                total_loss += loss.item()
                total_batches += 1
                avg_loss = total_loss / total_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                for i, h in enumerate(self.config.head_structure.keys()):
                    logit = logits[h].detach().cpu()
                    if self.config.head_structure[h] == 1:
                        pred = torch.sigmoid(logit)
                    else:
                        pred = torch.softmax(logit, dim=1)
                    accumulated_preds[h].append(pred)
                    accumulated_targets[h].append(targets[h].cpu())
                    if exam_ids is not None:
                        accumulated_exam_ids[h].extend(exam_ids)
                    print(f"[{h}] Labels (batch): {targets[h].cpu().numpy()}")
                    print(f"[{h}] Prédictions (batch): {pred.numpy()}")
                    print(f"[{h}] Moyenne labels: {targets[h].float().mean().item():.4f}, Moyenne préd: {pred.float().mean().item():.4f}")

                DistributedUtils.sync_process_group(world_size=self.world_size, device_ids=self.config.device)

        pbar.close()
        avg = total_loss / total_batches if total_batches > 0 else float('nan')
        
        # === Agrégation multi-GPU ici ===
        if self.world_size > 1:
            for head in accumulated_preds:
                local_preds = torch.cat(accumulated_preds[head], dim=0)
                local_targets = torch.cat(accumulated_targets[head], dim=0)
                gathered_preds = DistributedUtils.gather_tensor(local_preds, self.world_size)
                gathered_targets = DistributedUtils.gather_tensor(local_targets, self.world_size)
                accumulated_preds[head] = [gathered_preds]
                accumulated_targets[head] = [gathered_targets]

        print(f"[DEBUG] Val Epoch metrics: loss={avg:.4f}")
        val_aucs = self._log_epoch_metrics("val", self.current_epoch, accumulated_preds, accumulated_targets, avg)

        for head in accumulated_preds:
            self._save_predictions(
                "val",
                self.current_epoch,
                accumulated_preds[head],
                accumulated_targets[head],
                accumulated_exam_ids[head] if accumulated_exam_ids[head] else [None]*len(accumulated_preds[head])
            )
        return avg, val_aucs

    def run(self) -> None:
        """
        Boucle sur epochs : train + validation + scheduler + log W&B.
        """
        num_epochs = getattr(self.config, 'epochs', 1)
        # Dans run() :
        for epoch in trange(num_epochs, desc='Epochs', unit='epoch', ncols=80):
            self.current_epoch = epoch
            print(f"\n=== Époque {epoch+1}/{num_epochs} ===")
            self.train()
            val_loss, val_aucs = self.validation()  # <-- UNE SEULE FOIS
            self._save_checkpoint(epoch)
            mean_val_auc = np.mean([auc for auc in val_aucs.values() if auc is not None])
            if mean_val_auc > self.best_val_auc:
                self.best_val_auc = mean_val_auc
                self._save_checkpoint(epoch, is_best=True)
            lr = self.optimizer.param_groups[0]['lr']
            if not self.scheduler_per_iteration and self.lr_scheduler:
                self.lr_scheduler.step()
            if self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.log({
                    'epoch': epoch+1,
                    'lr': lr
                })
        print('=== Entraînement terminé ===')
    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "model": self.linear_probing.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "epoch": epoch,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        path = os.path.join(self.output_dir, "models", f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, "models", "best_model.pt"))

    def _log_epoch_metrics(self, mode, epoch, preds, targets, avg_loss):
        """
        Log metrics for the epoch, including AUC, AUPRC, and confusion matrix.
        """
        metrics = {f'{mode}/loss': avg_loss}
        aucs = {}
        for head in preds:
            pred_arr = torch.cat(preds[head], dim=0).numpy()
            tgt_arr = torch.cat(targets[head], dim=0).numpy()
            auc = None
            try:
                auc = roc_auc_score(tgt_arr, pred_arr)
                metrics[f'{mode}/{head}_auc'] = auc
            except Exception as e:
                print(f"[ERROR] {mode} - {head} AUC calculation failed: {e}")
            aucs[head] = auc

            # Calculate AUPRC
            try:
                auprc = average_precision_score(tgt_arr, pred_arr)
                metrics[f'{mode}/{head}_auprc'] = auprc
            except Exception as e:
                print(f"[ERROR] {mode} - {head} AUPRC calculation failed: {e}")

            # Calculate confusion matrix
            try:
                if self.config.head_structure[head] == 1:  # Binary classification
                    pred_labels = (pred_arr > 0.5).astype(int)
                else:  # Multi-class classification
                    pred_labels = pred_arr.argmax(axis=1)
                    tgt_arr = tgt_arr.argmax(axis=1)

                cm = confusion_matrix(tgt_arr, pred_labels)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{mode.capitalize()} Confusion Matrix – {head}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                # Log confusion matrix to W&B
                if self.wandb_wrapper.is_initialized():
                    self.wandb_wrapper.log_plot({f'confusion_matrix/{mode}/{head}': plt})
                plt.close()
            except Exception as e:
                print(f"[ERROR] {mode} - {head} Confusion Matrix calculation failed: {e}")

        # Log metrics to W&B
        if self.wandb_wrapper.is_initialized():
            self.wandb_wrapper.log({**metrics, 'epoch': epoch + 1})
        return aucs
