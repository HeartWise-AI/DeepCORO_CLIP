# autoresearch — DeepFFR Training Optimization

Autonomous experimentation loop for optimizing FFR Hyperemia (Fractional Flow Reserve) prediction
from coronary angiography videos using the DeepCORO MViT encoder.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read context files**:
   - This file (`autoresearch/program.md`) — rules and constraints
   - `autoresearch/experiment_ffr.yaml` — immutable champion config (read-only reference)
   - `autoresearch/experiment.yaml` — the ONLY file you modify
   - `autoresearch/run_experiment.sh` — how to launch experiments
   - `autoresearch/extract_metrics.py` — how to parse results
4. **Initialize results.tsv**: Already created with header row.
5. **Run baseline**: Run unmodified `experiment.yaml` to establish baseline metrics.
6. **Confirm and go**: Begin the loop.

## The Single Mutable File

**You may ONLY modify `autoresearch/experiment.yaml`.** This is a YAML config that controls all training hyperparameters, model architecture choices, and fine-tuning strategy. You cannot modify any Python code, runner code, model code, data files, or scripts.

## Goal

Maximize `mean_val_auc` (higher is better). This is the mean AUROC across all FFR heads with valid data. `mid_lad` is the most important vessel.

## What You Can Explore (via YAML only)

### Key Hyperparameters

| Category | Parameters | Typical Range |
|----------|-----------|---------------|
| **Encoder LR** | `video_encoder_lr` | 1e-7 to 1e-4 |
| **Encoder WD** | `video_encoder_weight_decay` | 1e-8 to 1e-4 |
| **Head LR** | `head_lr` (per-head dict) | 1e-5 to 1e-2 |
| **Head WD** | `head_weight_decay` (per-head dict) | 1e-6 to 1e-3 |
| **Freeze ratio** | `video_freeze_ratio` | 0.5 to 1.0 (1.0 = fully frozen) |
| **Dropout** | `dropout`, `head_dropout`, `dropout_attention` | 0.0 to 0.5 |
| **Batch size** | `batch_size` | 4 to 16 |
| **Epochs** | `epochs` | 10 to 50 |
| **Scheduler** | `scheduler_name` | cosine_with_warmup, cosine, step |
| **Warmup** | `num_warmup_percent` | 0.05 to 0.2 |
| **Head weights** | `head_weights` (per-head dict) | 0.5 to 5.0 |
| **Attention** | `num_attention_heads`, `attention_hidden` | heads: 2-8, hidden: 128-512 |
| **Num videos** | `num_videos` | 5 to 15 |
| **Gradient accum** | `gradient_accumulation_steps` | 1 to 8 |
| **AMP** | `use_amp` | true/false |
| **Augmentation** | `rand_augment` | true/false |
| **Normalization** | `normalization_strategy` | pre_norm, post_norm |

### Strategy Notes

- The dataset has 650 train studies (6.5x more than IFR), so there's more signal
- FFR data availability: mid_lad 388, prox_lad 240, prox_rca 93, prox_lcx 91, mid_rca 86
- Most vessels have sparse FFR data — the NaN masking handles this
- `head_weights` can emphasize vessels with more data (mid_lad, prox_lad)
- The encoder is pretrained on stenosis prediction — FFR is correlated but different
- FFR threshold: <= 0.80 = abnormal (ischemic), > 0.80 = normal

## Running Experiments

```bash
bash autoresearch/run_experiment.sh > autoresearch/run.log 2>&1
```

Then extract metrics:
```bash
python autoresearch/extract_metrics.py autoresearch/run.log
```

**Time budget**: ~75 min per experiment (15 epochs, 650 train studies, 10 videos, ~5 min/epoch).
**Timeout**: The script has a 5400s (90 min) timeout. If exceeded, treat as crash.

## Logging Results

Log to `results.tsv` (tab-separated). Columns:
```
commit	val_loss	memory_gb	status	description
```

- `status`: `keep`, `discard`, or `crash`
- Use `0.000000` for metrics on crashes
- Do NOT commit `results.tsv` or `run.log` (they're gitignored)

## The Experiment Loop

LOOP FOREVER:

1. Review current state: best val_loss, recent experiments, what's been tried
2. Propose a hypothesis — what YAML change might improve val_loss?
3. Edit `autoresearch/experiment.yaml` with the change
4. `git commit -m "descriptive message"`
5. Run: `bash autoresearch/run_experiment.sh > autoresearch/run.log 2>&1`
6. Extract: `python autoresearch/extract_metrics.py autoresearch/run.log`
7. If extraction fails, check `tail -n 50 autoresearch/run.log` for errors
8. Log results to `results.tsv`
9. If mean_val_auc improved → keep the commit (this is the new KEEP commit)
10. If mean_val_auc is equal or worse → revert with `git show <KEEP_COMMIT>:autoresearch/experiment.yaml > autoresearch/experiment.yaml` (NEVER use `git checkout -- autoresearch/experiment.yaml` as it reverts to HEAD which is the discarded config)
11. Go to step 1

## Constraints

- **GPU**: Always use GPU 3 only (`CUDA_VISIBLE_DEVICES=3` in `run_experiment.sh`)
- **Only modify**: `autoresearch/experiment.yaml`
- **Cannot modify**: Python code, runner code, model code, data, scripts
- **Cannot**: install packages, change dependencies
- **NEVER STOP**: Once the loop begins, run indefinitely. Do not ask the human if you should continue.

## Crash Handling

- Typos/easy fixes in YAML: fix and re-run
- OOM: reduce batch_size or num_videos, increase video_freeze_ratio, or reduce epochs
- Fundamental issues: log as crash, revert, move on
- If stuck after 3 consecutive crashes: revert to last known-good state and try a different direction

## Tips

- Start with small changes — one variable at a time
- 650 train studies allows for less regularization than IFR (100 studies)
- Consider upweighting vessels with more FFR data (mid_lad: 388, prox_lad: 240)
- If a direction shows promise, explore it further before moving on
- If you plateau, try more radical changes (different freeze ratio, very different LR ranges)
- Prioritize mid_lad AUROC alongside mean_val_auc (mid_lad is the most important vessel)
