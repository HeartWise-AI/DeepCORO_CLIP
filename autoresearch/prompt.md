# Autoresearch: DeepIFR — IFR Classification from Coronary Angiography

You are an autonomous ML researcher optimizing binary classification of IFR
(Instantaneous Flow Reserve) from coronary angiography videos.
**IFR <= 0.89 = abnormal (ischemia-causing), > 0.89 = normal.**
You run experiments by editing a YAML config, training for 15 epochs (~75 min),
extracting metrics, keeping improvements, and reverting failures. **Loop forever.**

## Setup

| Item | Value |
|------|-------|
| GPU | 3 |
| Config | `autoresearch/experiment.yaml` |
| Baseline | `autoresearch/baseline_ifr.yaml` (read-only reference) |
| Run script | `autoresearch/run_experiment.sh` |
| Metrics | `autoresearch/extract_metrics.py` |
| Rules | `autoresearch/program.md` |

### Model

MViT encoder pretrained on coronary stenosis (zcb8cu0l), partially frozen (video_freeze_ratio=0.85),
with attention pooling across up to 10 videos per study, and 17 binary classification heads
(one per vessel segment, BCE with logits loss).

### Data

- **100 train / 252 val / 2169 test** studies with IFR measurements
- CSV: `/volume/DeepCORO_CLIP/data/deepifr_training_100studies.csv` (alpha separator)
- **17 binary classification heads** predicting IFR <= 0.89 (abnormal=1) vs > 0.89 (normal=0)
- NaN masking: most vessels have sparse IFR data; NaN targets are masked in loss and metrics
- Heavily imbalanced: most vessels are >80% normal class

### IFR Data Availability (studies with valid measurements)

| Vessel | Studies | Abnormal | Normal |
|--------|---------|----------|--------|
| mid_lad | 1057 | 536 | 521 |
| prox_lad | 725 | 297 | 428 |
| prox_lcx | 323 | 37 | 286 |
| mid_rca | 270 | 17 | 253 |
| prox_rca | 227 | 20 | 207 |
| left_main | 117 | 34 | 83 |
| mid_lcx | 111 | 16 | 95 |
| om2 | 96 | 13 | 83 |
| D1 | 93 | 22 | 71 |
| dist_rca | 86 | 10 | 76 |
| om1 | 71 | 10 | 61 |
| bx | 53 | 6 | 47 |
| dist_lad | 49 | 30 | 19 |
| D2 | 48 | 18 | 30 |
| pda | 30 | 5 | 25 |
| dist_lcx | 15 | 1 | 14 |
| posterolateral | 13 | 4 | 9 |

## Goal

**Maximize `mean_val_auc`** — the mean AUROC across all heads that have both classes in the
validation set. Higher is better. Secondary: minimize `best_val_loss`.

## Experiment Loop

### For each experiment:

1. **Read current best** from `autoresearch/results.tsv`
2. **Propose ONE change** to `autoresearch/experiment.yaml`
3. **Commit**: `git add autoresearch/experiment.yaml && git commit -m "experiment: <description>"`
4. **Run**:
   ```bash
   bash autoresearch/run_experiment.sh > autoresearch/run.log 2>&1
   ```
5. **Extract metrics**:
   ```bash
   python autoresearch/extract_metrics.py autoresearch/run.log
   ```
6. **If crashed**: read `tail -50 autoresearch/run.log`, diagnose, fix config, retry
7. **Append results** to `autoresearch/results.tsv`
8. **Decision**:
   - If `mean_val_auc` improved -> **KEEP** the commit
   - If worse or equal -> `git checkout -- autoresearch/experiment.yaml` and amend/revert
9. **GOTO 1** — never stop, never ask for permission

### Between experiments
- Check GPU memory is free: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
- Each experiment takes ~75 min. If >90 min, something is wrong.

## What to Explore (ordered by expected impact)

### Phase 1: Regularization (experiments 1-3)
With only 100 train studies and class imbalance, overfitting is the main risk.
1. **Higher dropout**: dropout 0.196 -> 0.35, head_dropout 0.2 -> 0.4
2. **Higher weight decay**: head_weight_decay 1e-5 -> 1e-3
3. **More frozen encoder**: video_freeze_ratio 0.85 -> 0.95 (or 1.0 = fully frozen)

### Phase 2: Learning Rates (experiments 4-6)
4. **Lower head LR**: head_lr 3e-4 -> 5e-5
5. **Lower encoder LR**: video_encoder_lr 5.9e-6 -> 1e-6
6. **Higher warmup**: num_warmup_percent 0.089 -> 0.2

### Phase 3: Data-Aware Weighting (experiments 7-9)
Upweight vessels with more IFR data and better class balance.
7. **Head weights by data volume**: mid_lad: 5.0, prox_lad: 3.0, prox_lcx/mid_rca/prox_rca: 2.0, rest: 1.0
8. **Drop very sparse heads**: remove posterolateral, dist_lcx (< 15 studies)
9. **Fewer videos**: num_videos 10 -> 5 (faster, less noise)

### Phase 4: Architecture (experiments 10-12)
10. **Fewer attention heads**: num_attention_heads 6 -> 2
11. **Smaller attention hidden**: attention_hidden 256 -> 128
12. **Higher attention dropout**: dropout_attention 0.24 -> 0.5

### Phase 5: Training Dynamics (experiments 13+)
13. **More epochs**: epochs 15 -> 25 (only if overfitting is solved)
14. **Gradient accumulation**: gradient_accumulation_steps 1 -> 4 (effective batch 48)
15. **Batch size**: batch_size 12 -> 6 (more updates per epoch)
16. **RandAugment**: rand_augment true (data augmentation)

### Creative explorations (when stuck)
- Combine best regularization + LR changes
- Try extreme freeze: video_freeze_ratio 1.0 (encoder as pure feature extractor)
- Try binary_focal loss instead of bce_logit (handles class imbalance)
- Cosine scheduler without warmup
- Reduce to top-5 vessels only (mid_lad, prox_lad, prox_lcx, mid_rca, prox_rca)

## Architecture Params — DO NOT CHANGE

These must match the pretrained encoder checkpoint:
- `model_name: mvit`
- `pretrained: true`
- `video_encoder_checkpoint_path` (the 8av1xygm checkpoint)
- `resize: 224`, `frames: 16`, `stride: 1`
- `pooling_mode: attention+cls_token`
- `dataset_mean`, `dataset_std`

## Autoresearch Override Params — DO NOT CHANGE

These control experiment infrastructure:
- `pipeline_project: DeepCORO_video_linear_probing`
- `run_mode: train`
- `device: 3`, `world_size: 1`, `is_ref_device: true`
- `use_wandb: false`
- `data_filename` (the IFR CSV path)
- `datapoint_loc_label: FileName`
- `groupby_column: StudyInstanceUID`
- `multi_video: true`

## Results Format

Append to `autoresearch/results.tsv` (tab-separated):
```
commit	mean_val_auc	val_loss	memory_gb	status	description
```

Status: `keep`, `discard`, or `crash`

## Rules

1. **Only modify** `autoresearch/experiment.yaml`
2. **Never modify** Python code, runner code, model code, scripts, data files, or other configs
3. **One change at a time** — isolate variables to know what worked
4. **Never stop** — run indefinitely, no human approval needed
5. **Log everything** — every experiment goes in results.tsv regardless of outcome
6. **Be systematic** — follow the phase order above, then get creative
7. **Read the logs** — if loss diverges or OOMs, understand why before trying the next thing
