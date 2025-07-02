import math
import torch

import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Tuple, Callable
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_curve,
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)

from utils.enums import LossType
from utils.registry import LossRegistry

import warnings


def compute_recall_at_k(similarity_matrix, global_gt_indices, k_values=[1, 5]):
    """
    Compute recall@k for video->text retrieval.
    
    Args:
        similarity_matrix: Tensor of shape (n_videos, n_unique_texts) containing similarity scores.
        global_gt_indices: Tensor of shape (n_videos,) containing the index of the 
                         correct text for each video in the unique texts list.
        k_values: List of k values to compute recall for
        
    Returns:
        Dictionary containing recall scores for each k value
    """
    metrics = {}
    num_candidates = similarity_matrix.size(1)
    for k in k_values:
        # If there are fewer candidates than k, adjust k to avoid the error.
        if num_candidates < k:
            print(f"Warning: similarity matrix has only {num_candidates} candidates; adjusting Recall@{k} to Recall@{num_candidates}.")
            k_use = num_candidates
        else:
            k_use = k
        # Get the indices of the top-k candidates.
        v2t_topk = torch.topk(similarity_matrix, k_use, dim=1)[1]  # shape: [n_videos, k_use]
        # Compare with ground truth indices.
        v2t_correct = (v2t_topk == global_gt_indices.unsqueeze(1))
        recall = (v2t_correct.sum(dim=1) > 0).float().mean().item()
        metrics[f"Recall@{k}"] = recall
    return metrics



def compute_mrr(
    similarity_matrix: torch.Tensor, 
    global_gt_indices: torch.Tensor
) -> dict[str, float]:
    # Video to Text
    target_scores: torch.Tensor = similarity_matrix.gather(1, global_gt_indices.unsqueeze(1))
    v2t_ranks: torch.Tensor = (similarity_matrix >= target_scores).sum(1).float()
    v2t_mrr: float = (1 / v2t_ranks).mean().item()

    return {"MRR_V2T": v2t_mrr}

def compute_similarity_matrix(video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    normalized_video: torch.Tensor = nn.functional.normalize(video_features, dim=1)
    normalized_text: torch.Tensor = nn.functional.normalize(text_features, dim=1)
    return torch.matmul(normalized_video, normalized_text.T)

# Normalize embeddings
def compute_embedding_norms(video_features: torch.Tensor, text_features: torch.Tensor) -> dict:
    """Compute L2 norms of video and text embeddings."""
    video_norms: torch.Tensor = torch.norm(video_features, dim=1).mean().item()
    text_norms: torch.Tensor = torch.norm(text_features, dim=1).mean().item()
    return {"video_norm": video_norms, "text_norm": text_norms}


def compute_alignment_score(
    video_features: torch.Tensor,
    text_features: torch.Tensor,
    all_video_embeddings: torch.Tensor = None,
    all_text_embeddings: torch.Tensor = None,
    global_ground_truth_indices_tensor: torch.Tensor = None,
)-> float:
    """
    Compute average cosine similarity of positive pairs.

    Parameters:
    - video_features: torch.Tensor (batch local video embeddings)
    - text_features: torch.Tensor (batch local text embeddings)
    - all_video_embeddings: torch.Tensor of all validation video embeddings [N_videos, dim] (optional)
    - all_text_embeddings: torch.Tensor of all global text embeddings [N_texts, dim] (optional)
    - global_ground_truth_indices_tensor: torch.Tensor of global GT indices for each video (optional)

    If all_video_embeddings, all_text_embeddings, and global_ground_truth_indices_tensor
    are provided, compute global alignment using global embeddings.

    Otherwise, compute local alignment score assuming a one-to-one mapping between
    video_features[i] and text_features[i].
    """
    if (
        all_video_embeddings is not None
        and all_text_embeddings is not None
        and global_ground_truth_indices_tensor is not None
    ):
        # Global alignment scenario (for validation)
        correct_text_embeddings: torch.Tensor = all_text_embeddings[global_ground_truth_indices_tensor]
        normalized_video: torch.Tensor = nn.functional.normalize(all_video_embeddings, dim=1)
        normalized_text: torch.Tensor = nn.functional.normalize(correct_text_embeddings, dim=1)
        alignment_scores: torch.Tensor = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()
    else:
        # Local alignment scenario (for training)
        normalized_video: torch.Tensor = nn.functional.normalize(video_features, dim=1)
        normalized_text: torch.Tensor = nn.functional.normalize(text_features, dim=1)
        alignment_scores: torch.Tensor = (normalized_video * normalized_text).sum(dim=1)
        return alignment_scores.mean().item()

def compute_ndcg_at_k(
    similarity_matrix: torch.Tensor, 
    global_gt_indices: torch.Tensor, 
    k_values: List[int]
) -> dict[str, float]:
    """
    Compute NDCG@k for each query and average over all queries.
    Simplified assumption: one correct answer per query.

    Args:
        similarity_matrix (torch.Tensor): [num_queries, num_candidates]
        global_gt_indices (torch.Tensor): [num_queries], each entry is the index of the correct text
        k (int): Rank cutoff

    Returns:
        float: Average NDCG@k over all queries.
    """
    num_queries: int = similarity_matrix.size(0)
    num_candidates: int = similarity_matrix.size(1)
    if num_queries == 0:
        return 0.0

    # Sort candidates by similarity in descending order
    sorted_indices: torch.Tensor = torch.argsort(similarity_matrix, dim=1, descending=True)

    metrics: dict[str, float] = {}
    for k in k_values:
        # Adjust k if it's larger than number of candidates
        effective_k: int = min(k, num_candidates)
        
        ndcg_values: list[float] = []
        for i in range(num_queries):
            correct_idx: int = global_gt_indices[i].item()
            # Find the rank of the correct index
            ranking: torch.Tensor = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
            if ranking.numel() == 0:
                # Correct item not found (should not happen if all candidates included)
                ndcg_values.append(0.0)
                continue

            rank: int = ranking.item()
            if rank < effective_k:
                # DCG = 1 / log2(rank+2)
                dcg: float = 1.0 / math.log2(rank + 2)
            else:
                dcg: float = 0.0

            # Ideal DCG (IDCG) = 1 since there's only one relevant doc at best rank
            idcg: float = 1.0
            ndcg_values.append(dcg / idcg)

        metrics[f"NDCG@{k}_V2T"] = float(torch.tensor(ndcg_values).mean().item())

    return metrics


def compute_median_rank(
    similarity_matrix: torch.Tensor, 
    global_gt_indices: torch.Tensor
) -> int:
    """
    Compute the median rank of the correct item over all queries.
    Lower is better.
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    ranks = []
    for i in range(num_queries):
        correct_idx = global_gt_indices[i].item()
        ranking = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
        if ranking.numel() == 0:
            # Not found, assign large rank
            ranks.append(similarity_matrix.size(1))
        else:
            rank = ranking.item() + 1  # +1 because ranks are 1-based
            ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)
    median_rank = int(ranks.median().item())  # Convert to int before returning
    return median_rank


def compute_map(similarity_matrix: torch.Tensor, global_gt_indices: torch.Tensor) -> float:
    """
    Compute mean average precision (MAP).
    Assuming exactly one relevant doc per query.
    AP = 1/rank_of_correct_item
    MAP = average of AP over all queries
    """
    num_queries = similarity_matrix.size(0)
    if num_queries == 0:
        return 0.0

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    aps = []
    for i in range(num_queries):
        correct_idx = global_gt_indices[i].item()
        ranking = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
        if ranking.numel() == 0:
            # Correct not found, AP=0
            aps.append(0.0)
        else:
            rank = ranking.item() + 1
            ap = 1.0 / rank
            aps.append(ap)

    return float(torch.tensor(aps).mean().item())


def compute_best_threshold(
    df_gt_col: list[float], 
    df_pred_col: list[float]
) -> float:
    """
    Compute the best threshold that maximizes both sensitivity (true positive rate) 
    and specificity (1 - false positive rate).

    Args:
        df_gt_col (list[float]): Ground truth values for a specific column.
        df_pred_col (list[float]): Predicted values for a specific column.

    Returns:
        float: The best threshold value that maximizes sensitivity and specificity.
    """
    fpr, tpr, roc_thresholds = roc_curve(df_gt_col, df_pred_col)
    
    # Compute sensitivity (TPR) and specificity (1-FPR)
    specificity = 1 - fpr
    
    # Find threshold that maximizes both sensitivity and specificity
    # by maximizing their geometric mean
    gmeans = np.sqrt(tpr * specificity)
    best_threshold = roc_thresholds[np.argmax(gmeans)]
    
    return float(best_threshold) # convert to float instead of numpy.float32

def compute_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    mode: str,
    wandb_wrapper = None,
    is_ref_device: bool = False
) -> dict:
    """
    Compute regression metrics for the given predictions and targets,
    and log a regression plot if on the reference device and W&B is initialized.
    """
    metrics = {}

    with torch.no_grad():
        # Compute MAE
        metrics[f"{mode}/{head_name}_mae"] = LossRegistry.get(LossType.MAE)()(
            outputs=preds,
            targets=targets
        ).item()

        # Compute MSE
        metrics[f"{mode}/{head_name}_mse"] = LossRegistry.get(LossType.MSE)()(
            outputs=preds,
            targets=targets
        ).item()

        # Compute RMSE
        metrics[f"{mode}/{head_name}_rmse"] = LossRegistry.get(LossType.RMSE)()(
            outputs=preds,
            targets=targets
        ).item()

    # Convert tensors to numpy arrays
    preds_np = preds.detach().cpu().float().numpy().squeeze()
    targets_np = targets.detach().cpu().float().numpy().squeeze()

    # Calculate Pearson correlation using numpy arrays
    try:
        r, _ = pearsonr(preds_np, targets_np)
        metrics[f"{mode}/{head_name}_pearson_r"] = r
    except ValueError as e:
        print(f"Could not compute Pearson correlation for {head_name}: {e}")
        r, p_value = np.nan, np.nan # Assign NaN if calculation fails

    # Generate and log regression plot if wandb is initialized and on ref device
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized():
        try:
            plt.figure(figsize=(10, 8))
            sns.regplot(x=targets_np, y=preds_np, line_kws={'color': 'red'})
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plot_title = (
                f'{mode.capitalize()} Regression Plot - {head_name}\n'
                f'Pearson r: {r:.3f}'
            )
            plt.title(plot_title)
            plt.grid(True)

            # Log to wandb
            wandb_wrapper.log_plot({
                f"regression_plot/{mode}/{head_name}": plt
            })
            plt.close() # Close the plot to free memory

        except Exception as e:
            print(f"Error generating/logging regression plot for {head_name}: {e}")
            plt.close() # Ensure plot is closed even if error occurs

    return metrics

def compute_threshold_based_metrics(
    preds: np.ndarray, 
    targets: np.ndarray, 
    threshold: float
) -> dict[str, float]:
    """
    Compute confusion matrix based metrics (F1, PPV, NPV, etc.) using a given threshold.
    
    Args:
        preds: Array of prediction probabilities
        targets: Array of binary ground truth labels  
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dictionary containing confusion matrix metrics
    """
    # Binarize predictions using threshold
    pred_binary = (preds > threshold).astype(int)
    targets_binary = targets.astype(int)
    
    # Calculate confusion matrix elements
    tp = np.sum((pred_binary == 1) & (targets_binary == 1))
    tn = np.sum((pred_binary == 0) & (targets_binary == 0))
    fp = np.sum((pred_binary == 1) & (targets_binary == 0))
    fn = np.sum((pred_binary == 0) & (targets_binary == 1))
    
    # Calculate metrics
    metrics = {}
    
    # Precision (PPV - Positive Predictive Value)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # NPV (Negative Predictive Value) 
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Sensitivity (Recall/True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['sensitivity'] = sensitivity
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['specificity'] = specificity
    
    # F1-score
    precision = metrics['ppv']
    recall = sensitivity
    metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return metrics

def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    head_structure: dict,
    labels_map: dict = None,
    mode: str = "val",
    wandb_wrapper = None,
    is_ref_device: bool = False
) -> dict:
    """
    Compute classification metrics for the given predictions and targets.
    
    Args:
        preds: Tensor of model predictions
        targets: Tensor of ground truth labels
        head_name: Name of the model head being evaluated
        head_structure: Dictionary containing output dimensions for each head
        labels_map: Dictionary mapping class names to indices
        mode: Current mode (train or validation)
        device: Current device ID
        wandb_wrapper: WandbWrapper instance for logging
        is_ref_device: Whether this device is the reference device for logging
        
    Returns:
        Dictionary of computed metrics
    """
    
    metrics = {}
    
    # Convert to numpy for sklearn metrics
    all_preds = preds.detach().cpu().numpy()
    all_targets = targets.detach().cpu().numpy()
    
    # Compute AUC and AUPRC
    try:
        auc = roc_auc_score(all_targets.tolist(), all_preds.tolist(), average="micro")
        metrics[f"{mode}/{head_name}_auc"] = auc
    except Exception as e:
        print(f"Error computing AUC: {e}")
    
    try:
        auprc = average_precision_score(all_targets.tolist(), all_preds.tolist(), average="micro")
        metrics[f"{mode}/{head_name}_auprc"] = auprc
    except Exception as e:
        print(f"Error computing AUPRC: {e}")
    
    # Compute and log confusion matrix if wandb is initialized
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized():
        try:
            # For binary classification
            if head_structure[head_name] == 1:
                try:
                    # Compute best threshold using Youden's J statistic
                    best_threshold = compute_best_threshold(
                        all_targets.tolist(), 
                        all_preds.tolist()
                    )
                except Exception:
                    # If error, use default threshold 0.5
                    best_threshold = 0.5
                
                # Binarize predictions
                pred_labels = (all_preds > best_threshold).astype(int)
                
                # Log best threshold
                metrics[f"{mode}/{head_name}_best_threshold"] = best_threshold
            
            # For multi-class classification
            else:
                pred_labels = all_preds.argmax(axis=1)
                all_targets = all_targets.argmax(axis=1)
            
            # Create confusion matrix if labels_map is provided
            if labels_map:
                # Create labels list
                labels = [''] * len(labels_map[head_name])
                for k, v in labels_map[head_name].items():
                    labels[v] = k
                
                # Compute confusion matrix
                cm = confusion_matrix(y_true=all_targets, y_pred=pred_labels)
                
                # Create confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=labels, 
                    yticklabels=labels
                )
                plt.title(f'{mode.capitalize()} Confusion Matrix - {head_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                # Log to wandb
                wandb_wrapper.log_plot({
                    f"confusion_matrix/{mode}/{head_name}": plt
                })
                plt.close()
        
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
    
    return metrics

# Add new functions for CI computation

def bootstrap_metric(
    preds: np.ndarray, 
    targets: np.ndarray, 
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute confidence intervals for a metric using bootstrap resampling.
    
    Args:
        preds: Array of predictions
        targets: Array of targets  
        metric_fn: Function that computes metric given (preds, targets)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (metric_value, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    
    n_samples = len(preds)
    if n_samples < 10:
        warnings.warn("Sample size too small for reliable bootstrap CI")
        metric_value = metric_fn(preds, targets)
        return metric_value, metric_value, metric_value
    
    # Compute original metric
    metric_value = metric_fn(preds, targets)
    
    # Bootstrap sampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        try:
            bootstrap_metric = metric_fn(preds[indices], targets[indices])
            if not np.isnan(bootstrap_metric):
                bootstrap_metrics.append(bootstrap_metric)
        except:
            # Skip if metric computation fails for this bootstrap sample
            continue
    
    if len(bootstrap_metrics) < 10:
        warnings.warn("Too few valid bootstrap samples for reliable CI")
        return metric_value, metric_value, metric_value
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return metric_value, ci_lower, ci_upper

def compute_classification_metrics_with_ci(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    head_structure: dict,
    labels_map: dict = None,
    mode: str = "val",
    wandb_wrapper = None,
    is_ref_device: bool = False,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> dict:
    """
    Compute classification metrics with confidence intervals.
    """
    metrics = {}
    
    # Convert to numpy for sklearn metrics
    all_preds = preds.detach().cpu().numpy()
    all_targets = targets.detach().cpu().numpy()
    
    # Define metric functions for bootstrap
    def auc_fn(p, t):
        try:
            return roc_auc_score(t, p, average="micro")
        except:
            return np.nan
    
    def auprc_fn(p, t):
        try:
            return average_precision_score(t, p, average="micro")
        except:
            return np.nan
    
    # Compute AUC with CI
    try:
        auc_val, auc_ci_lower, auc_ci_upper = bootstrap_metric(
            all_preds.flatten(), 
            all_targets.flatten(), 
            auc_fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
        metrics[f"{mode}/{head_name}_auc"] = auc_val
        metrics[f"{mode}/{head_name}_auc_ci_lower"] = auc_ci_lower
        metrics[f"{mode}/{head_name}_auc_ci_upper"] = auc_ci_upper
        metrics[f"{mode}/{head_name}_auc_ci_width"] = auc_ci_upper - auc_ci_lower
    except Exception as e:
        print(f"Error computing AUC with CI: {e}")
        # Fallback to original computation
        try:
            auc = auc_fn(all_preds.flatten(), all_targets.flatten())
            metrics[f"{mode}/{head_name}_auc"] = auc
        except:
            pass
    
    # Compute AUPRC with CI
    try:
        auprc_val, auprc_ci_lower, auprc_ci_upper = bootstrap_metric(
            all_preds.flatten(), 
            all_targets.flatten(), 
            auprc_fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
        metrics[f"{mode}/{head_name}_auprc"] = auprc_val
        metrics[f"{mode}/{head_name}_auprc_ci_lower"] = auprc_ci_lower
        metrics[f"{mode}/{head_name}_auprc_ci_upper"] = auprc_ci_upper
        metrics[f"{mode}/{head_name}_auprc_ci_width"] = auprc_ci_upper - auprc_ci_lower
    except Exception as e:
        print(f"Error computing AUPRC with CI: {e}")
        # Fallback to original computation
        try:
            auprc = auprc_fn(all_preds.flatten(), all_targets.flatten())
            metrics[f"{mode}/{head_name}_auprc"] = auprc
        except:
            pass
    
    # Compute best threshold and accuracy metrics for binary classification
    if head_structure[head_name] == 1:
        def best_threshold_fn(p, t):
            try:
                return compute_best_threshold(t.tolist(), p.tolist())
            except:
                return 0.5
        
        def accuracy_at_threshold_fn(threshold):
            def acc_fn(p, t):
                pred_labels = (p > threshold).astype(int)
                return np.mean(pred_labels == t.astype(int))
            return acc_fn
        
        def f1_at_threshold_fn(threshold):
            def f1_fn(p, t):
                cm_metrics = compute_threshold_based_metrics(p, t, threshold)
                return cm_metrics['f1_score']
            return f1_fn
        
        def ppv_at_threshold_fn(threshold):
            def ppv_fn(p, t):
                cm_metrics = compute_threshold_based_metrics(p, t, threshold)
                return cm_metrics['ppv']
            return ppv_fn
        
        def npv_at_threshold_fn(threshold):
            def npv_fn(p, t):
                cm_metrics = compute_threshold_based_metrics(p, t, threshold)
                return cm_metrics['npv']
            return npv_fn
        
        def sensitivity_at_threshold_fn(threshold):
            def sens_fn(p, t):
                cm_metrics = compute_threshold_based_metrics(p, t, threshold)
                return cm_metrics['sensitivity']
            return sens_fn
        
        def specificity_at_threshold_fn(threshold):
            def spec_fn(p, t):
                cm_metrics = compute_threshold_based_metrics(p, t, threshold)
                return cm_metrics['specificity']
            return spec_fn
    
        
        # Compute best threshold with CI
        try:
            threshold_val, threshold_ci_lower, threshold_ci_upper = bootstrap_metric(
                all_preds.flatten(), 
                all_targets.flatten(), 
                best_threshold_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            metrics[f"{mode}/{head_name}_best_threshold"] = threshold_val
            metrics[f"{mode}/{head_name}_best_threshold_ci_lower"] = threshold_ci_lower
            metrics[f"{mode}/{head_name}_best_threshold_ci_upper"] = threshold_ci_upper
            
            # Compute accuracy at best threshold with CI
            acc_val, acc_ci_lower, acc_ci_upper = bootstrap_metric(
                all_preds.flatten(), 
                all_targets.flatten(), 
                accuracy_at_threshold_fn(threshold_val),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            metrics[f"{mode}/{head_name}_accuracy"] = acc_val
            metrics[f"{mode}/{head_name}_accuracy_ci_lower"] = acc_ci_lower
            metrics[f"{mode}/{head_name}_accuracy_ci_upper"] = acc_ci_upper
            metrics[f"{mode}/{head_name}_accuracy_ci_width"] = acc_ci_upper - acc_ci_lower
            
            # Compute F1 at best threshold with CI
            f1_val, f1_ci_lower, f1_ci_upper = bootstrap_metric(
                all_preds.flatten(), 
                all_targets.flatten(), 
                f1_at_threshold_fn(threshold_val),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            metrics[f"{mode}/{head_name}_f1_score"] = f1_val
            metrics[f"{mode}/{head_name}_f1_score_ci_lower"] = f1_ci_lower
            metrics[f"{mode}/{head_name}_f1_score_ci_upper"] = f1_ci_upper
            metrics[f"{mode}/{head_name}_f1_score_ci_width"] = f1_ci_upper - f1_ci_lower
            
            # Compute PPV at best threshold with CI
            ppv_val, ppv_ci_lower, ppv_ci_upper = bootstrap_metric(
                all_preds.flatten(), 
                all_targets.flatten(), 
                ppv_at_threshold_fn(threshold_val),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            metrics[f"{mode}/{head_name}_ppv"] = ppv_val
            metrics[f"{mode}/{head_name}_ppv_ci_lower"] = ppv_ci_lower
            metrics[f"{mode}/{head_name}_ppv_ci_upper"] = ppv_ci_upper
            metrics[f"{mode}/{head_name}_ppv_ci_width"] = ppv_ci_upper - ppv_ci_lower
            
            # Compute NPV at best threshold with CI
            npv_val, npv_ci_lower, npv_ci_upper = bootstrap_metric(
                all_preds.flatten(), 
                all_targets.flatten(), 
                npv_at_threshold_fn(threshold_val),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            metrics[f"{mode}/{head_name}_npv"] = npv_val
            metrics[f"{mode}/{head_name}_npv_ci_lower"] = npv_ci_lower
            metrics[f"{mode}/{head_name}_npv_ci_upper"] = npv_ci_upper
            metrics[f"{mode}/{head_name}_npv_ci_width"] = npv_ci_upper - npv_ci_lower
            
            # Compute Sensitivity at best threshold with CI
            sensitivity_val, sensitivity_ci_lower, sensitivity_ci_upper = bootstrap_metric(
                all_preds.flatten(), 
                all_targets.flatten(), 
                sensitivity_at_threshold_fn(threshold_val),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            ) 
            metrics[f"{mode}/{head_name}_sensitivity"] = sensitivity_val
            metrics[f"{mode}/{head_name}_sensitivity_ci_lower"] = sensitivity_ci_lower
            metrics[f"{mode}/{head_name}_sensitivity_ci_upper"] = sensitivity_ci_upper
            metrics[f"{mode}/{head_name}_sensitivity_ci_width"] = sensitivity_ci_upper - sensitivity_ci_lower
            
            # Compute Specificity at best threshold with CI
            specificity_val, specificity_ci_lower, specificity_ci_upper = bootstrap_metric(
                all_preds.flatten(), 
                all_targets.flatten(), 
                specificity_at_threshold_fn(threshold_val),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )   
            metrics[f"{mode}/{head_name}_specificity"] = specificity_val
            metrics[f"{mode}/{head_name}_specificity_ci_lower"] = specificity_ci_lower
            metrics[f"{mode}/{head_name}_specificity_ci_upper"] = specificity_ci_upper
            metrics[f"{mode}/{head_name}_specificity_ci_width"] = specificity_ci_upper - specificity_ci_lower
                        
        except Exception as e:
            print(f"Error computing threshold/accuracy with CI: {e}")
            # Fallback
            try:
                threshold_val = best_threshold_fn(all_preds.flatten(), all_targets.flatten())
                metrics[f"{mode}/{head_name}_best_threshold"] = threshold_val
                acc_val = accuracy_at_threshold_fn(threshold_val)(all_preds.flatten(), all_targets.flatten())
                metrics[f"{mode}/{head_name}_accuracy"] = acc_val
            except:
                pass
    
    # Confusion matrix computation (existing code)
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized():
        try:
            # For binary classification
            if head_structure[head_name] == 1:
                best_threshold = metrics.get(f"{mode}/{head_name}_best_threshold", 0.5)
                pred_labels = (all_preds > best_threshold).astype(int)
            else:
                # For multi-class classification
                pred_labels = all_preds.argmax(axis=1)
                all_targets = all_targets.argmax(axis=1)
            
            # Create confusion matrix if labels_map is provided
            if labels_map:
                # Create labels list
                labels = [''] * len(labels_map[head_name])
                for k, v in labels_map[head_name].items():
                    labels[v] = k
                
                # Compute confusion matrix
                cm = confusion_matrix(y_true=all_targets, y_pred=pred_labels)
                
                # Create confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=labels, 
                    yticklabels=labels
                )
                plt.title(f'{mode.capitalize()} Confusion Matrix - {head_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                # Log to wandb
                wandb_wrapper.log_plot({
                    f"confusion_matrix/{mode}/{head_name}": plt
                })
                plt.close()
        
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
    
    return metrics

def compute_regression_metrics_with_ci(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    mode: str,
    wandb_wrapper = None,
    is_ref_device: bool = False,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> dict:
    """
    Compute regression metrics with confidence intervals.
    """
    metrics = {}
    
    # Convert tensors to numpy arrays
    preds_np = preds.detach().cpu().float().numpy().squeeze()
    targets_np = targets.detach().cpu().float().numpy().squeeze()
    
    # Define metric functions for bootstrap
    def mae_fn(p, t):
        return np.mean(np.abs(p - t))
    
    def mse_fn(p, t):
        return np.mean((p - t) ** 2)
    
    def rmse_fn(p, t):
        return np.sqrt(np.mean((p - t) ** 2))
    
    def pearson_fn(p, t):
        try:
            r, _ = pearsonr(p, t)
            return r if not np.isnan(r) else 0.0
        except:
            return 0.0
    
    # Compute metrics with CI
    try:
        # MAE with CI
        mae_val, mae_ci_lower, mae_ci_upper = bootstrap_metric(
            preds_np, targets_np, mae_fn,
            n_bootstrap=n_bootstrap, confidence_level=confidence_level
        )
        metrics[f"{mode}/{head_name}_mae"] = mae_val
        metrics[f"{mode}/{head_name}_mae_ci_lower"] = mae_ci_lower
        metrics[f"{mode}/{head_name}_mae_ci_upper"] = mae_ci_upper
        metrics[f"{mode}/{head_name}_mae_ci_width"] = mae_ci_upper - mae_ci_lower
        
        # MSE with CI
        mse_val, mse_ci_lower, mse_ci_upper = bootstrap_metric(
            preds_np, targets_np, mse_fn,
            n_bootstrap=n_bootstrap, confidence_level=confidence_level
        )
        metrics[f"{mode}/{head_name}_mse"] = mse_val
        metrics[f"{mode}/{head_name}_mse_ci_lower"] = mse_ci_lower
        metrics[f"{mode}/{head_name}_mse_ci_upper"] = mse_ci_upper
        metrics[f"{mode}/{head_name}_mse_ci_width"] = mse_ci_upper - mse_ci_lower
        
        # RMSE with CI
        rmse_val, rmse_ci_lower, rmse_ci_upper = bootstrap_metric(
            preds_np, targets_np, rmse_fn,
            n_bootstrap=n_bootstrap, confidence_level=confidence_level
        )
        metrics[f"{mode}/{head_name}_rmse"] = rmse_val
        metrics[f"{mode}/{head_name}_rmse_ci_lower"] = rmse_ci_lower
        metrics[f"{mode}/{head_name}_rmse_ci_upper"] = rmse_ci_upper
        metrics[f"{mode}/{head_name}_rmse_ci_width"] = rmse_ci_upper - rmse_ci_lower
        
        # Pearson correlation with CI
        pearson_val, pearson_ci_lower, pearson_ci_upper = bootstrap_metric(
            preds_np, targets_np, pearson_fn,
            n_bootstrap=n_bootstrap, confidence_level=confidence_level
        )
        metrics[f"{mode}/{head_name}_pearson_r"] = pearson_val
        metrics[f"{mode}/{head_name}_pearson_r_ci_lower"] = pearson_ci_lower
        metrics[f"{mode}/{head_name}_pearson_r_ci_upper"] = pearson_ci_upper
        metrics[f"{mode}/{head_name}_pearson_r_ci_width"] = pearson_ci_upper - pearson_ci_lower
        
    except Exception as e:
        print(f"Error computing regression metrics with CI: {e}")
        # Fallback to original computation without CI
        metrics[f"{mode}/{head_name}_mae"] = mae_fn(preds_np, targets_np)
        metrics[f"{mode}/{head_name}_mse"] = mse_fn(preds_np, targets_np)
        metrics[f"{mode}/{head_name}_rmse"] = rmse_fn(preds_np, targets_np)
        metrics[f"{mode}/{head_name}_pearson_r"] = pearson_fn(preds_np, targets_np)

    # Generate and log regression plot (existing code)
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized():
        try:
            plt.figure(figsize=(10, 8))
            sns.regplot(x=targets_np, y=preds_np, line_kws={'color': 'red'})
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            
            # Add CI information to plot title if available
            if f"{mode}/{head_name}_pearson_r_ci_lower" in metrics:
                plot_title = (
                    f'{mode.capitalize()} Regression Plot - {head_name}\n'
                    f'Pearson r: {metrics[f"{mode}/{head_name}_pearson_r"]:.3f} '
                    f'[{metrics[f"{mode}/{head_name}_pearson_r_ci_lower"]:.3f}, '
                    f'{metrics[f"{mode}/{head_name}_pearson_r_ci_upper"]:.3f}]'
                )

            
            plt.title(plot_title)
            plt.grid(True)

            # Log to wandb
            wandb_wrapper.log_plot({
                f"regression_plot/{mode}/{head_name}": plt
            })
            plt.close()

        except Exception as e:
            print(f"Error generating/logging regression plot for {head_name}: {e}")
            plt.close()

    return metrics