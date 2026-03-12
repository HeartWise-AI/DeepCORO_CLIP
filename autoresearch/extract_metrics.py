#!/usr/bin/env python3
"""Extract metrics from DeepIFR autoresearch run.log.

Parses best val loss, per-head AUROC, and GPU memory from training output.
Primary goal: maximize mean AUROC across heads with valid data.
"""

import re
import sys


def extract_metrics(log_path: str) -> dict[str, float]:
    """Parse best val loss, per-head AUROCs, and GPU memory from run.log."""
    with open(log_path) as f:
        content = f.read()

    metrics: dict[str, float] = {}

    # Extract "New best model! Val Loss: X.XXXX"
    best_pattern = r"New best model! Val Loss:\s*([\d.]+)"
    best_matches = re.findall(best_pattern, content)
    if best_matches:
        metrics["best_val_loss"] = float(best_matches[-1])

    # Extract val/main_loss from debug output
    val_loss_pattern = r"'val/main_loss':\s*([\d.]+)"
    val_matches = re.findall(val_loss_pattern, content)
    if val_matches:
        metrics["last_val_loss"] = float(val_matches[-1])

    # Extract per-head AUROC: 'val/HEADNAME_auc': X.XXX
    auc_pattern = r"'val/(\w+)_auc':\s*([\d.]+|nan)"
    auc_matches = re.findall(auc_pattern, content)
    if auc_matches:
        # Use last occurrence of each head
        head_aucs = {}
        for head_name, auc_val in auc_matches:
            if auc_val != "nan":
                head_aucs[head_name] = float(auc_val)

        # Store individual AUROCs
        for head_name, auc_val in head_aucs.items():
            metrics[f"val/{head_name}_auc"] = auc_val

        # Compute mean AUROC across heads with valid data
        valid_aucs = [v for v in head_aucs.values() if v == v]  # filter NaN
        if valid_aucs:
            metrics["mean_val_auc"] = sum(valid_aucs) / len(valid_aucs)
            metrics["n_heads_with_auc"] = len(valid_aucs)

    # Extract per-head losses
    head_loss_pattern = r"'val/(\w+)_loss':\s*([\d.]+)"
    head_matches = re.findall(head_loss_pattern, content)
    if head_matches:
        for head_name, loss_val in head_matches:
            metrics[f"val/{head_name}_loss"] = float(loss_val)

    # Extract GPU memory
    mem_pattern = r"Final GPU memory:\s*([\d.]+)GB allocated,\s*([\d.]+)GB reserved"
    mem_match = re.search(mem_pattern, content)
    if mem_match:
        metrics["peak_memory_gb"] = float(mem_match.group(2))

    # Extract epoch info
    epoch_pattern = r"Epoch\s+(\d+)/(\d+)"
    epoch_matches = re.findall(epoch_pattern, content)
    if epoch_matches:
        metrics["last_epoch"] = float(epoch_matches[-1][0])
        metrics["total_epochs"] = float(epoch_matches[-1][1])

    return metrics


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run.log>")
        sys.exit(1)

    log_path = sys.argv[1]
    metrics = extract_metrics(log_path)

    if not metrics:
        print("ERROR: No metrics found in log file. Run may have crashed.")
        sys.exit(1)

    best_val_loss = metrics.get("best_val_loss", metrics.get("last_val_loss", None))
    mean_auc = metrics.get("mean_val_auc", "N/A")
    n_heads = metrics.get("n_heads_with_auc", 0)
    memory = metrics.get("peak_memory_gb", "N/A")

    print("\n=== EXPERIMENT RESULTS ===")
    print(f"best_val_loss: {best_val_loss}")
    print(f"mean_val_auc: {mean_auc}")
    print(f"n_heads_with_auc: {n_heads}")
    print(f"memory_gb: {memory}")
    if "last_epoch" in metrics:
        print(f"epochs: {int(metrics['last_epoch'])}/{int(metrics['total_epochs'])}")
    print("=========================\n")

    # Print per-head AUROCs
    head_aucs = {k: v for k, v in sorted(metrics.items()) if k.startswith("val/") and k.endswith("_auc")}
    if head_aucs:
        print("Per-head val AUROC:")
        for key, value in head_aucs.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
