# Deprecation Fixes Applied

## Summary of Fixed Deprecation Warnings

### 1. GradScaler Deprecation
**Warning:** `torch.cuda.amp.GradScaler(args...)` is deprecated  
**Fix:** Changed to `torch.amp.GradScaler('cuda', args...)`

**Files Updated:**
- `/workspace/projects/multitask_pretraining_project.py`
- `/workspace/projects/contrastive_pretraining_project.py`
- `/workspace/projects/linear_probing_project.py`
- `/workspace/runners/multitask_runner.py`
- `/workspace/runners/linear_probing_runner.py`
- `/workspace/runners/video_constrative_learning_runner.py`
- `/workspace/tests/test_linear_probing_runner.py`

**Changes Made:**
```python
# Before
from torch.cuda.amp import GradScaler
scaler = GradScaler() if use_amp else None

# After
from torch.amp import GradScaler
scaler = GradScaler('cuda') if use_amp else None
```

### 2. torch.load weights_only Parameter
**Warning:** Issues with `weights_only=True` when loading checkpoints containing non-tensor objects  
**Fix:** Changed to `weights_only=False` for checkpoint loading

**Files Updated:**
- `/workspace/projects/base_project.py`
- `/workspace/projects/linear_probing_project.py`
- `/workspace/runners/video_constrative_learning_runner.py`
- `/workspace/utils/generate_video_embeddings.py`
- `/workspace/utils/generate_text_embeddings.py`

**Changes Made:**
```python
# Before
torch.load(checkpoint_path, map_location='cpu', weights_only=True)

# After
torch.load(checkpoint_path, map_location='cpu', weights_only=False)
```

### 3. encoder_attention_mask Warning
**Warning:** `encoder_attention_mask` is deprecated in transformers v4.55.0  
**Note:** This is an internal warning from the transformers library when using BERT-based models. It's not directly called in our code but comes from the library's internal implementation.

**Resolution:** This warning will be resolved when the transformers library is updated to handle this internally. No code changes required on our end.

## Testing

After applying these fixes, the following deprecation warnings should no longer appear:
- ✅ `torch.cuda.amp.GradScaler` deprecation warning
- ✅ `weights_only` parameter warnings for torch.load
- ⚠️ `encoder_attention_mask` warning (external library issue)

## Compatibility

These changes maintain backward compatibility while preparing the codebase for future PyTorch versions:
- PyTorch 2.0+ recommended
- Transformers 4.30+ compatible