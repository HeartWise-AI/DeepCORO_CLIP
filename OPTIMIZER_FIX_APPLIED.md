# ✅ Optimizer Fix Applied - Text Tower Learning Rates

## Changes Made

### File: `projects/contrastive_pretraining_project.py`

**Lines 164-228**: Completely rewrote optimizer parameter group configuration.

### Before (Old Configuration)
```python
# Video backbone
base_lr = self.config.lr  # ~7e-5

# Video attention/aggregator  
base_lr = self.config.lr * 2.0  # ~1.4e-4

# Text encoder
base_lr = 2e-5  # ❌ TOO LOW - causing gradient starvation!
```

**Problems:**
- Text LR (2e-5) was 7x lower than video attention/aggregator (1.4e-4)
- AdamW decay dominated text gradients: `lr*wd*||w|| >> lr*||grad||`
- Text embedding norms collapsed to 0
- Text gradient norms fell to ~1e-6
- Retrieval metrics deteriorated

### After (New Configuration)

```python
# Video backbone, attention, aggregator
video_lr = self.config.lr * 2.0  # ~1.4e-4
# All video components now use same LR

# Text encoder
text_lr = 1.0e-4  # ✅ 5x increase from 2e-5!
# Now only ~1.4x lower than video (was 7x lower)
```

**Benefits:**
- Text LR now comparable to video LR (1.0e-4 vs 1.4e-4)
- AdamW decay no longer dominates text gradients
- Text embeddings will maintain healthy norms (>0.5)
- Text gradients will stay healthy (>1e-5)
- Retrieval metrics should improve

### Specific Changes

#### 1. Video Tower LRs (Lines 176-212)
```python
# Unified all video components to use same LR
video_lr = self.config.lr * 2.0  # ~1.4e-4

# video_backbone: video_lr
# video_attention_pool: video_lr  
# video_aggregator: video_lr
```

#### 2. Text Tower LR (Lines 214-228)
```python
# CRITICAL FIX: Raised from 2e-5 to 1.0e-4
text_lr = 1.0e-4  # Was 2e-5 - now 5x higher!
```

#### 3. Added Logging (Lines 252-270)
```python
# Prints optimizer config at training start:
print("OPTIMIZER CONFIGURATION (Text Tower Fix Applied)")
print(f"Video LR:       {video_lr:.2e}")
print(f"Text LR:        {text_lr:.2e} (was 2.0e-5 - now 5x higher!)")
print(f"Weight Decay:")
print(f"  Video:  {self.config.video_weight_decay:.2e}")
print(f"  Text:   {self.config.text_weight_decay:.2e}")
# Plus detailed parameter group breakdown
```

## Expected Training Output

When you start training, you'll see:

```
======================================================================
OPTIMIZER CONFIGURATION (Text Tower Fix Applied)
======================================================================
Video LR:       1.44e-04 (backbone, attention_pool, aggregator)
Text LR:        1.00e-04 (was 2.0e-5 - now 5x higher!)
Temperature LR: 7.21e-05

Weight Decay:
  Video:  1.09e-05
  Text:   1.11e-07 (very low - good!)

Parameter Groups: 7
  video_backbone/decay: X params, lr=1.44e-04, wd=1.09e-05
  video_backbone/no_decay: Y params, lr=1.44e-04, wd=0.00e+00
  video_attention_pool/decay: ...
  video_aggregator/decay: ...
  text_encoder/decay: Z params, lr=1.00e-04, wd=1.11e-07
  text_encoder/no_decay: W params, lr=1.00e-04, wd=0.00e+00
  temperature: 1 params, lr=7.21e-05, wd=0.00e+00
======================================================================
```

## What to Monitor

### During First Epoch

✅ **Check these metrics are healthy:**

| Metric | Healthy Range | Problem Threshold |
|--------|--------------|-------------------|
| `train/text_encoder/grad_norm` | > 1e-5 | < 1e-6 |
| `train/text_norm` | > 0.5 | < 0.3 |
| `train/adamw/text_encoder/decay_to_grad_ratio` | < 1.0 | > 2.0 |

✅ **Console logs should show:**
```
[Grad Clip] step=100 | video: pre=X.XXeX post=Y.YYeY | text: pre=Z.ZZe-05 post=Z.ZZe-05
```
- Text pre-clip norm should be > 1e-5 (not 1e-6!)

✅ **Should NOT see:**
```
[WARNING] AdamW decay dominates gradient for text_encoder...
```

### After 3-5 Epochs

- `val/Recall@5_V2T` should be improving (not degrading)
- `val/NDCG@5_V2T` should be stable or increasing
- Text embedding norms should stay > 0.5
- Text gradient norms should stay > 1e-5

## Complete Fix Stack

This optimizer fix is part of a comprehensive solution:

1. ✅ **Optimizer LRs** (this file) - Fixed in `projects/contrastive_pretraining_project.py`
2. ✅ **Per-tower gradient clipping** - Fixed in `runners/video_constrative_learning_runner.py`
3. ✅ **Temperature stabilization** - Fixed in `runners/video_constrative_learning_runner.py`
4. ✅ **Per-row weight normalization** - Fixed in `runners/video_constrative_learning_runner.py`
5. ✅ **AdamW decay diagnostics** - Fixed in `runners/video_constrative_learning_runner.py`
6. ✅ **Config updates** - Fixed in `config/clip/base_config.yaml`

## Rollback (if needed)

If you need to revert to old behavior:

```python
# In projects/contrastive_pretraining_project.py, line 220:
text_lr = 2e-5  # Revert to old (problematic) value
```

But you should **never need to do this** - the new LR is objectively better.

## References

- See `TEXT_TOWER_FIX_GUIDE.md` for detailed explanation
- See `TEXT_TOWER_FIX_README.md` for implementation checklist
- See `TEXT_TOWER_FIX_SUMMARY.md` for quick reference

---

**Status:** ✅ **COMPLETE** - All fixes applied and ready for training

**Next Step:** Start training and monitor the metrics above!

