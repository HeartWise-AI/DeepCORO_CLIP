import torch
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.wandb_wrapper import WandbWrapper

from functools import partial
from scipy.stats import pearsonr
from sklearn.metrics import (
    f1_score,
    roc_curve,
    recall_score,
    roc_auc_score, 
    accuracy_score,
    precision_score,
    confusion_matrix,
    average_precision_score, 
)
from typing import (
    Tuple, 
    Union,
    Callable, 
    Optional
)


# ================================ #
# Define metric functions for bootstrap
# ================================ #
def mae_fn(preds: np.ndarray, targets: np.ndarray):
    return np.mean(np.abs(preds - targets))

def mse_fn(preds: np.ndarray, targets: np.ndarray):
    return np.mean((preds - targets) ** 2)

def rmse_fn(preds: np.ndarray, targets: np.ndarray):
    return np.sqrt(np.mean((preds - targets) ** 2))

def pearson_fn(preds: np.ndarray, targets: np.ndarray):
    try:
        r, _ = pearsonr(preds, targets)
        return r if not np.isnan(r) else 0.0
    except:
        return 0.0

def best_threshold_fn(preds: np.ndarray, targets: np.ndarray):
    try:
        return compute_best_threshold(targets.tolist(), preds.tolist())
    except:
        return 0.5
    
def metrics_at_threshold_fn(threshold: Optional[float] = None, average: str = "macro"):
    return partial(compute_confusion_matrix_metrics, threshold=threshold, average=average)

def binary_auc_fn(preds: np.ndarray, targets: np.ndarray):
    """AUC for binary classification - expects probabilities for positive class."""
    try:
        return roc_auc_score(targets, preds)
    except:
        return np.nan

def binary_auprc_fn(preds: np.ndarray, targets: np.ndarray):
    """AUPRC for binary classification - expects probabilities for positive class."""
    try:
        return average_precision_score(targets, preds)
    except:
        return np.nan

def multiclass_auc_fn(preds: np.ndarray, targets: np.ndarray, average: str = "macro", multi_class: str = "ovr"):
    """AUC for multiclass classification - expects probabilities for positive class."""
    try:
        return roc_auc_score(targets, preds, average=average, multi_class=multi_class)
    except:
        return np.nan

def multiclass_auprc_fn(preds: np.ndarray, targets: np.ndarray, average: str = "macro"):
    """AUPRC for multiclass classification - expects probabilities for each class."""
    try:
        # Convert integer labels to one-hot if needed
        if targets.ndim == 1:
            num_classes = preds.shape[1]
            targets = np.eye(num_classes)[targets.astype(int)]
        return average_precision_score(targets, preds, average=average)
    except:
        return np.nan
    
# ================================ #

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
    compute_ci: bool = False,
    wandb_wrapper: Optional[WandbWrapper] = None,
    is_ref_device: bool = False,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> dict:
    """
    Compute regression metrics with confidence intervals.
    
    Args:
        preds: Tensor of predictions
        targets: Tensor of targets
        head_name: Name of the head
        mode: Mode of the evaluation ('train', 'val', 'test')
        compute_ci: Whether to compute confidence intervals
        wandb_wrapper: Wandb wrapper for logging
        is_ref_device: Whether the device is the reference device for logging
        confidence_level: Confidence level for the confidence intervals (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary of metrics with optional confidence intervals
    """
    metrics = {}
    
    # Convert tensors to numpy arrays
    preds_np = preds.detach().cpu().float().numpy().squeeze()
    targets_np = targets.detach().cpu().float().numpy().squeeze()
        
    # Compute metrics with CI
    if compute_ci:
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
            
    else: 
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
            else:
                plot_title = (
                    f'{mode.capitalize()} Regression Plot - {head_name}\n'
                    f'Pearson r: {metrics[f"{mode}/{head_name}_pearson_r"]:.3f}'
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

def bootstrap_metric(
    preds: np.ndarray, 
    targets: np.ndarray, 
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    metric_key: str = None,
    **metric_kwargs
) -> Union[Tuple[float, float, float], dict]:
    """
    Compute confidence intervals for a metric using bootstrap resampling.
    Handles both scalar and dictionary return types.
    
    Args:
        preds: Array of predictions
        targets: Array of targets  
        metric_fn: Function that computes metric given (preds, targets)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        metric_key: If metric_fn returns dict, extract this key for scalar return
        **metric_kwargs: Additional keyword arguments for metric_fn
        
    Returns:
        - If metric_fn returns scalar OR metric_key provided: Tuple of (metric_value, ci_lower, ci_upper)
        - If metric_fn returns dict AND no metric_key: Dict with CI bounds for all metrics
    """
    np.random.seed(random_state)
    
    n_samples = len(preds)
    if n_samples < 10:
        warnings.warn("Sample size too small for reliable bootstrap CI")
        result = metric_fn(preds, targets, **metric_kwargs)
        
        if isinstance(result, dict):
            if metric_key:
                metric_value = result[metric_key]
                return metric_value, metric_value, metric_value
            else:
                return result  # Return dict as-is for small samples
        else:
            return result, result, result
    
    # Compute original metric
    original_result = metric_fn(preds, targets, **metric_kwargs)
    is_dict_result = isinstance(original_result, dict)
    
    if is_dict_result:
        if metric_key:
            # Single metric from dict
            return _bootstrap_single_metric_from_dict(
                preds, targets, metric_fn, metric_key, 
                n_bootstrap, confidence_level, **metric_kwargs
            )
        else:
            # All metrics from dict
            return _bootstrap_all_metrics_from_dict(
                preds, targets, metric_fn, 
                n_bootstrap, confidence_level, **metric_kwargs
            )
    else:
        # Scalar result - original implementation
        return _bootstrap_scalar_metric(
            preds, targets, metric_fn, 
            n_bootstrap, confidence_level, **metric_kwargs
        )


def _bootstrap_scalar_metric(
    preds: np.ndarray, 
    targets: np.ndarray, 
    metric_fn: Callable,
    n_bootstrap: int,
    confidence_level: float,
    **metric_kwargs
) -> Tuple[float, float, float]:
    """Original scalar bootstrap implementation."""
    
    # Compute original metric
    metric_value = metric_fn(preds, targets, **metric_kwargs)
    
    # Bootstrap sampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(preds), size=len(preds), replace=True)
        try:
            bootstrap_metric = metric_fn(preds[indices], targets[indices], **metric_kwargs)
            if not np.isnan(bootstrap_metric):
                bootstrap_metrics.append(bootstrap_metric)
        except:
            continue
    
    if len(bootstrap_metrics) < 10:
        warnings.warn("Too few valid bootstrap samples for reliable CI")
        return metric_value, metric_value, metric_value
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return metric_value, ci_lower, ci_upper


def _bootstrap_single_metric_from_dict(
    preds: np.ndarray, 
    targets: np.ndarray, 
    metric_fn: Callable,
    metric_key: str,
    n_bootstrap: int,
    confidence_level: float,
    **metric_kwargs
) -> Tuple[float, float, float]:
    """Bootstrap single metric from dict-returning function."""
    
    # Compute original metric
    original_dict = metric_fn(preds, targets, **metric_kwargs)
    metric_value = original_dict[metric_key]
    
    # Bootstrap sampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(preds), size=len(preds), replace=True)
        try:
            bootstrap_dict = metric_fn(preds[indices], targets[indices], **metric_kwargs)
            bootstrap_metric = bootstrap_dict[metric_key]
            if not np.isnan(bootstrap_metric):
                bootstrap_metrics.append(bootstrap_metric)
        except:
            continue
    
    if len(bootstrap_metrics) < 10:
        warnings.warn("Too few valid bootstrap samples for reliable CI")
        return metric_value, metric_value, metric_value
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return metric_value, ci_lower, ci_upper


def _bootstrap_all_metrics_from_dict(
    preds: np.ndarray, 
    targets: np.ndarray, 
    metric_fn: Callable,
    n_bootstrap: int,
    confidence_level: float,
    **metric_kwargs
) -> dict:
    """Bootstrap all metrics from dict-returning function."""
    
    # Compute original metrics
    original_metrics = metric_fn(preds, targets, **metric_kwargs)
    
    # Initialize bootstrap storage
    bootstrap_metrics = {key: [] for key in original_metrics.keys()}
    
    # Bootstrap sampling
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(preds), size=len(preds), replace=True)
        try:
            bootstrap_dict = metric_fn(preds[indices], targets[indices], **metric_kwargs)
            for key, value in bootstrap_dict.items():
                if not np.isnan(value):
                    bootstrap_metrics[key].append(value)
        except:
            continue
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    enhanced_metrics = original_metrics.copy()
    
    for key, bootstrap_values in bootstrap_metrics.items():
        if len(bootstrap_values) >= 10:
            ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
            enhanced_metrics[f"{key}_ci_lower"] = ci_lower
            enhanced_metrics[f"{key}_ci_upper"] = ci_upper
            enhanced_metrics[f"{key}_ci_width"] = ci_upper - ci_lower
    
    return enhanced_metrics

def compute_confusion_matrix_metrics(
    preds: np.ndarray, 
    targets: np.ndarray, 
    threshold: Optional[float] = None,
    average: str = "macro"
) -> dict[str, float]:
    """
    Compute confusion matrix based metrics for binary or multi-class classification.
    
    Args:
        preds: Binary probabilities [N] or multi-class probabilities [N, C]
        targets: Integer class labels [N]  
        threshold: Threshold for binary classification (required if preds.ndim == 1)
        average: Averaging strategy for multi-class ('macro', 'micro', 'weighted')
        partial 
    Returns:
        Dictionary containing confusion matrix metrics
    """
    is_binary = preds.ndim == 1
    if is_binary:
        if threshold is None:
            raise ValueError("Threshold must be provided for binary classification")
        # Binarize predictions using threshold
        preds_classes = (preds > threshold).astype(int)
        average = "binary"
    else:
        preds_classes = np.argmax(preds, axis=1)
        
        
    targets_classes = targets.astype(int)
        
    metrics = {
        'f1_score': f1_score(targets_classes, preds_classes, average=average, zero_division=0),
        'precision': precision_score(targets_classes, preds_classes, average=average, zero_division=0),
        'recall': recall_score(targets_classes, preds_classes, average=average, zero_division=0),
        'accuracy': accuracy_score(targets_classes, preds_classes)
    }
    
    if is_binary:
        cm = confusion_matrix(targets_classes, preds_classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        else:
            # Handle edge cases where confusion matrix is not 2x2
            metrics['specificity'] = 0.0
            metrics['npv'] = 0.0
            warnings.warn("Confusion matrix is not 2x2, setting specificity and npv to 0.0")

    return metrics

def _compute_auc_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    head_name: str,
    mode: str,
    compute_ci: bool,
    metrics: dict,
    confidence_level: float,
    n_bootstrap: int,
    is_binary: bool,
    average: str = "macro",
    multi_class: str = "ovr"
) -> None:
    """
    Compute AUC metrics with confidence intervals for binary or multiclass problems.
    
    Args:
        preds: Array of predictions - shape (n,) for binary or (n, c) for multiclass
        targets: Array of targets - shape (n,) for binary, (n,) or (n, c) for multiclass
        head_name: Name of the head
        mode: Mode of the evaluation ('train', 'val', 'test')
        compute_ci: Whether to compute confidence intervals
        metrics: Dictionary to store metrics (modified in-place)
        confidence_level: Confidence level for CIs
        n_bootstrap: Number of bootstrap samples
        is_binary: Whether this is binary classification
        average: Averaging method for multiclass ('macro' or 'micro')
        multi_class: Multiclass strategy ('ovr' or 'ovo')
        
    Returns:
        None (modifies metrics dict in-place)
    """
    # Helper to add metric with optional CI
    def add_metric_with_ci(metric_name: str, metric_fn: Callable):
        """
        Add a metric to the metrics dict with optional confidence intervals.
        
        Args:
            metric_name: Name of the metric (e.g., 'auc', 'auprc')
            metric_fn: Function to compute the metric
        """
        if compute_ci:
            try:
                val, ci_lower, ci_upper = bootstrap_metric(
                    preds=preds,
                    targets=targets,
                    metric_fn=metric_fn,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level
                )
                metrics[f"{mode}/{head_name}_{metric_name}"] = val
                metrics[f"{mode}/{head_name}_{metric_name}_ci_lower"] = ci_lower
                metrics[f"{mode}/{head_name}_{metric_name}_ci_upper"] = ci_upper
                metrics[f"{mode}/{head_name}_{metric_name}_ci_width"] = ci_upper - ci_lower
            except Exception as e:
                print(f"Error computing {metric_name} with CI: {e}")
                # Fallback to point estimate
                try:
                    metrics[f"{mode}/{head_name}_{metric_name}"] = metric_fn(preds, targets)
                except Exception as e2:
                    print(f"Error computing {metric_name}: {e2}")
        else:
            try:
                print(f"Computing {metric_name} without CI: {metric_fn(preds, targets)}, average: {average}, multi_class: {multi_class}, preds shape: {preds.shape}, targets shape: {targets.shape}")
                metrics[f"{mode}/{head_name}_{metric_name}"] = metric_fn(preds, targets)
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
    
    # Compute AUC and AUPRC
    add_metric_with_ci(
        metric_name="auc" if is_binary else f"{average}_auc", 
        metric_fn=binary_auc_fn if is_binary else partial(multiclass_auc_fn, average=average, multi_class=multi_class)
    )
    add_metric_with_ci(
        metric_name="auprc" if is_binary else f"{average}_auprc", 
        metric_fn=binary_auprc_fn if is_binary else partial(multiclass_auprc_fn, average=average)
    )

def _compute_best_threshold(
    preds: np.ndarray,
    targets: np.ndarray,
    head_name: str,
    mode: str,
    compute_ci: bool,
    metrics: dict,
    confidence_level: float,
    n_bootstrap: int,
) -> float:
    """
    Compute the best threshold with optional confidence intervals.
    
    Args:
        preds: Array of predictions - shape (n,) for binary classification
        targets: Array of targets - shape (n,) for binary classification
        head_name: Name of the head
        mode: Mode of the evaluation ('train', 'val', 'test')
        compute_ci: Whether to compute confidence intervals
        metrics: Dictionary to store metrics (modified in-place)
        confidence_level: Confidence level for CIs
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        float: The computed best threshold value
    """
    threshold_val = 0.5  # Default fallback threshold
    
    if compute_ci:
        try:
            threshold_val, threshold_ci_lower, threshold_ci_upper = bootstrap_metric(
                preds=preds,
                targets=targets,
                metric_fn=best_threshold_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            metrics[f"{mode}/{head_name}_best_threshold"] = threshold_val
            metrics[f"{mode}/{head_name}_best_threshold_ci_lower"] = threshold_ci_lower
            metrics[f"{mode}/{head_name}_best_threshold_ci_upper"] = threshold_ci_upper
            metrics[f"{mode}/{head_name}_best_threshold_ci_width"] = threshold_ci_upper - threshold_ci_lower
            
        except Exception as e:
            print(f"Error computing best threshold with CI: {e}")
            # Fallback to point estimate
            threshold_val = _compute_threshold_point_estimate(preds, targets, head_name, mode, metrics)
    else:
        # Compute without CI
        threshold_val = _compute_threshold_point_estimate(preds, targets, head_name, mode, metrics)
    
    return threshold_val


def _compute_threshold_point_estimate(
    preds: np.ndarray,
    targets: np.ndarray,
    head_name: str,
    mode: str,
    metrics: dict
) -> float:
    """
    Compute best threshold as a point estimate (no confidence intervals).
    
    Returns:
        float: The computed threshold value
    """
    try:
        threshold_val = best_threshold_fn(preds, targets)
        metrics[f"{mode}/{head_name}_best_threshold"] = threshold_val
        return threshold_val
    except Exception as e:
        print(f"Error computing best threshold: {e}, using default threshold 0.5")
        metrics[f"{mode}/{head_name}_best_threshold"] = 0.5
        return 0.5


def _compute_confusion_matrix_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    head_name: str,
    mode: str,
    compute_ci: bool,
    metrics: dict,
    confidence_level: float,
    n_bootstrap: int,
    average: str = "macro",
    threshold: Optional[float] = None
) -> None:
    """
    Compute confusion matrix metrics with optional confidence intervals.
    """
    if compute_ci:
        try:
            # This works for both binary and multiclass
            all_metrics_with_ci = bootstrap_metric(
                preds=preds,
                targets=targets,
                metric_fn=metrics_at_threshold_fn(threshold=threshold, average=average),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level
            )
            
            _add_confusion_matrix_metrics_with_ci(
                all_metrics_with_ci, head_name, mode, metrics
            )
            
        except Exception as e:
            print(f"Error computing confusion matrix metrics with CI: {e}")
            _compute_confusion_matrix_point_estimates(
                preds, targets, threshold, head_name, mode, metrics, average
            )
    else:
        _compute_confusion_matrix_point_estimates(
            preds, targets, threshold, head_name, mode, metrics, average
        )


def _add_confusion_matrix_metrics_with_ci(
    all_metrics_with_ci: dict,
    head_name: str,
    mode: str,
    metrics: dict
) -> None:
    """
    Add confusion matrix metrics with confidence intervals to the metrics dictionary.
    
    Args:
        all_metrics_with_ci: Dictionary containing metrics and their CI bounds
        head_name: Name of the head
        mode: Mode of the evaluation
        metrics: Dictionary to store metrics (modified in-place)
    """
    # Define the base metrics we expect
    base_metrics = ['f1_score', 'precision', 'recall', 'specificity', 'npv', 'accuracy']
    
    for metric_name in base_metrics:
        if metric_name in all_metrics_with_ci:
            # Add the base metric
            metrics[f"{mode}/{head_name}_{metric_name}"] = all_metrics_with_ci[metric_name]
            
            # Add CI bounds if they exist
            ci_lower_key = f"{metric_name}_ci_lower"
            ci_upper_key = f"{metric_name}_ci_upper"
            ci_width_key = f"{metric_name}_ci_width"
            
            if ci_lower_key in all_metrics_with_ci:
                metrics[f"{mode}/{head_name}_{metric_name}_ci_lower"] = all_metrics_with_ci[ci_lower_key]
                metrics[f"{mode}/{head_name}_{metric_name}_ci_upper"] = all_metrics_with_ci[ci_upper_key]
                metrics[f"{mode}/{head_name}_{metric_name}_ci_width"] = all_metrics_with_ci[ci_width_key]
    
    # Add aliases for backward compatibility
    _add_metric_aliases_with_ci(all_metrics_with_ci, head_name, mode, metrics)


def _add_metric_aliases_with_ci(
    all_metrics_with_ci: dict,
    head_name: str,
    mode: str,
    metrics: dict
) -> None:
    """
    Add metric aliases (ppv, sensitivity) with confidence intervals.
    
    Args:
        all_metrics_with_ci: Dictionary containing metrics and their CI bounds
        head_name: Name of the head
        mode: Mode of the evaluation
        metrics: Dictionary to store metrics (modified in-place)
    """
    # Handle precision -> ppv alias
    if 'precision' in all_metrics_with_ci:
        metrics[f"{mode}/{head_name}_ppv"] = all_metrics_with_ci['precision']
        if 'precision_ci_lower' in all_metrics_with_ci:
            metrics[f"{mode}/{head_name}_ppv_ci_lower"] = all_metrics_with_ci['precision_ci_lower']
            metrics[f"{mode}/{head_name}_ppv_ci_upper"] = all_metrics_with_ci['precision_ci_upper']
            metrics[f"{mode}/{head_name}_ppv_ci_width"] = all_metrics_with_ci['precision_ci_width']
    
    # Handle recall -> sensitivity alias
    if 'recall' in all_metrics_with_ci:
        metrics[f"{mode}/{head_name}_sensitivity"] = all_metrics_with_ci['recall']
        if 'recall_ci_lower' in all_metrics_with_ci:
            metrics[f"{mode}/{head_name}_sensitivity_ci_lower"] = all_metrics_with_ci['recall_ci_lower']
            metrics[f"{mode}/{head_name}_sensitivity_ci_upper"] = all_metrics_with_ci['recall_ci_upper']
            metrics[f"{mode}/{head_name}_sensitivity_ci_width"] = all_metrics_with_ci['recall_ci_width']


def _compute_confusion_matrix_point_estimates(
    mode: str,
    metrics: dict,
    head_name: str,
    preds: np.ndarray,
    targets: np.ndarray,
    average: str = "macro",
    threshold: Optional[float] = None
) -> None:
    """
    Compute confusion matrix metrics as point estimates (no confidence intervals).
    
    Args:
        preds: Array of predictions
        targets: Array of targets
        threshold: Threshold value to use
        head_name: Name of the head
        mode: Mode of the evaluation
        metrics: Dictionary to store metrics (modified in-place)
    """
    try:
        # Compute all threshold-based metrics using the existing function
        threshold_metrics = metrics_at_threshold_fn(threshold=threshold, average=average)(preds, targets)
        
        # Add threshold-based metrics to main metrics dict
        for metric_name, metric_value in threshold_metrics.items():
            metrics[f"{mode}/{head_name}_{metric_name}"] = metric_value
        
        # Add aliases for backward compatibility
        if 'precision' in threshold_metrics:
            metrics[f"{mode}/{head_name}_ppv"] = threshold_metrics['precision']
        if 'recall' in threshold_metrics:
            metrics[f"{mode}/{head_name}_sensitivity"] = threshold_metrics['recall']
            
    except Exception as e:
        print(f"Error computing confusion matrix point estimates: {e}")


# Now update the main _compute_threshold_metrics function to use these helpers
def _compute_threshold_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    head_name: str,
    mode: str,
    compute_ci: bool,
    metrics: dict,
    confidence_level: float,
    n_bootstrap: int,
) -> None:
    """
    Compute threshold-based metrics with confidence intervals for binary classification.
    
    This function is now much cleaner and delegates to helper functions.
    """
    # Step 1: Compute best threshold
    threshold_val = _compute_best_threshold(
        preds=preds,
        targets=targets,
        head_name=head_name,
        mode=mode,
        compute_ci=compute_ci,
        metrics=metrics,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap
    )
    
    # Step 2: Compute confusion matrix metrics at that threshold
    _compute_confusion_matrix_metrics(
        preds=preds,
        targets=targets,
        threshold=threshold_val,
        head_name=head_name,
        mode=mode,
        compute_ci=compute_ci,
        metrics=metrics,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap
    )

def compute_binary_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    labels_map: dict = None,
    mode: str = "val",
    compute_ci: bool = False,
    wandb_wrapper: WandbWrapper = None,
    is_ref_device: bool = False,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> dict:
    """
    Compute classification metrics with confidence intervals for binary classification.
    
    Args:
        preds: Tensor of shape (batch_size,) containing probabilities [0,1] for positive class
        targets: Tensor of shape (batch_size,) containing binary labels (0 or 1)
        head_name: Name of the head
        labels_map: Dictionary mapping class names to indices (optional)
        mode: Mode of the evaluation ('train', 'val', 'test')
        compute_ci: Whether to compute confidence intervals
        wandb_wrapper: Wandb wrapper for logging
        is_ref_device: Whether the device is the reference device for logging
        confidence_level: Confidence level for the confidence intervals (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary of metrics with optional confidence intervals
    """
    
    # Validate input tensors for binary classification
    if preds.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have the same shape. "
            f"Got preds: {preds.shape}, targets: {targets.shape}"
        )
    if preds.ndim != 1:
        raise ValueError(
            f"Binary classification expects 1D prediction tensor of probabilities. "
            f"Got {preds.ndim}D tensor"
        )
    if targets.ndim != 1:
        raise ValueError(
            f"Binary classification expects 1D target tensor of labels. "
            f"Got {targets.ndim}D tensor"
        )
    if preds.min() < 0 or preds.max() > 1:
        raise ValueError(
            f"Binary probabilities should be in [0,1] range. "
            f"Got [{preds.min():.3f}, {preds.max():.3f}]"
        )        
        
    # Convert to numpy for sklearn metrics
    all_preds = preds.detach().cpu().numpy()
    all_targets = targets.detach().cpu().numpy()
        
    # Initialize metrics dictionary
    metrics = {}    
        
    # Compute AUC with CI
    _compute_auc_metrics(
        preds=all_preds,
        targets=all_targets,
        head_name=head_name,
        mode=mode,
        compute_ci=compute_ci,
        metrics=metrics,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        is_binary=True
    )
    
    _compute_threshold_metrics(
        preds=all_preds, 
        targets=all_targets, 
        head_name=head_name, 
        mode=mode, 
        compute_ci=compute_ci, 
        metrics=metrics, 
        confidence_level=confidence_level, 
        n_bootstrap=n_bootstrap
    )

            
    # Confusion matrix computation (existing code)
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized():
        try:
            # For binary classification
            best_threshold = metrics.get(f"{mode}/{head_name}_best_threshold", 0.5)
            pred_labels = (all_preds > best_threshold).astype(int)
                            
            # Create confusion matrix if labels_map is provided
            if labels_map:
                plot_confusion_matrix_wandb(
                    labels_map=labels_map, 
                    pred_labels=pred_labels, 
                    all_targets=all_targets, 
                    mode=mode, 
                    head_name=head_name, 
                    wandb_wrapper=wandb_wrapper
                )
        
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
    
    return metrics    

def compute_multiclass_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    labels_map: dict = None,
    mode: str = "val",
    compute_ci: bool = False,
    wandb_wrapper: WandbWrapper = None,
    is_ref_device: bool = False,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> dict:
    """
    Compute classification metrics with confidence intervals for multiclass classification.
    
    Args:
        preds: Tensor of shape (batch_size, num_classes) containing probabilities for each class
        targets: Tensor of shape (batch_size,) containing integer class labels
        head_name: Name of the head
        labels_map: Dictionary mapping class names to indices (optional)
        mode: Mode of the evaluation ('train', 'val', 'test')
        compute_ci: Whether to compute confidence intervals
        wandb_wrapper: Wandb wrapper for logging
        is_ref_device: Whether the device is the reference device for logging
        confidence_level: Confidence level for the confidence intervals (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary of metrics with optional confidence intervals
    """
    # Validate input tensors for multiclass classification
    if preds.ndim != 2:
        raise ValueError(
            f"Multiclass classification expects 2D prediction tensor of probabilities. "
            f"Got {preds.ndim}D tensor"
        )
    if targets.ndim != 1:
        raise ValueError(
            f"Multiclass classification expects 1D target tensor of labels. "
            f"Got {targets.ndim}D tensor"
        )
    if preds.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Predictions and targets must have the same batch size. "
            f"Got preds: {preds.shape[0]}, targets: {targets.shape[0]}"
        )        
    if preds.min() < 0 or preds.max() > 1:
        warnings.warn(
            f"Predictions outside [0,1] range: [{preds.min():.3f}, {preds.max():.3f}]. "
            "Consider applying softmax activation."
        )       
        
    # Convert to numpy for sklearn metrics
    all_preds = preds.detach().cpu().numpy()
    all_targets = targets.detach().cpu().numpy()
            
    # Initialize metrics dictionary
    metrics = {}
    
    # Compute AUC with CI
    for average in ['macro', 'micro']:
        _compute_auc_metrics(
            preds=all_preds,
            targets=all_targets,
            head_name=head_name,
            mode=mode,
            compute_ci=compute_ci,
            metrics=metrics,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            is_binary=False,
            average=average,
            multi_class='ovr'
        )
        
    # Compute macro and micro confusion matrix metrics
    for average in ['macro', 'micro']:
        # Need to update the metric keys to include average type
        temp_metrics = {}
        _compute_confusion_matrix_metrics(
            preds=all_preds,
            targets=all_targets,
            threshold=None,
            head_name=head_name,
            mode=mode,
            compute_ci=compute_ci,
            metrics=temp_metrics,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            average=average
        )
        
        # Rename keys to include average type
        for key, value in temp_metrics.items():
            # Replace the head_name part with head_name_average
            new_key = key.replace(f"{mode}/{head_name}_", f"{mode}/{head_name}_{average}_")
            metrics[new_key] = value
              
    # Plot confusion matrix
    if is_ref_device and wandb_wrapper and wandb_wrapper.is_initialized() and labels_map:
        try:
            pred_labels = np.argmax(all_preds, axis=1)
            plot_confusion_matrix_wandb(
                labels_map=labels_map,
                pred_labels=pred_labels,
                all_targets=all_targets,
                mode=mode,
                head_name=head_name,
                wandb_wrapper=wandb_wrapper
            )
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
    
    return metrics        
    
  
def plot_confusion_matrix_wandb(
    labels_map: dict, 
    pred_labels: np.ndarray, 
    all_targets: np.ndarray, 
    mode: str, 
    head_name: str, 
    wandb_wrapper: WandbWrapper
) -> None:
    try:
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
        print(f"Error creating confusion matrix plot: {e}")
        return None