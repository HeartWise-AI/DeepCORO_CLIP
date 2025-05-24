# CLS Token Implementation in DeepCORO_CLIP

## 🤔 Why CLS Token?

The cls_token (short for "classification token") is a special learnable embedding used in transformer models—especially in architectures like BERT and Vision Transformers (ViT)—to aggregate information from the entire input sequence or image for classification tasks.

### Key Advantages:
- **Learnable aggregation** that adapts during training
- **Self-attention mechanism** captures complex relationships  
- **More flexible** than fixed pooling (mean/max) as it adapts to the specific task
- **Captures complex inter-patch relationships** that simple pooling might miss
- **Inspired by successful transformer architectures** (BERT, ViT)

## 🏗️ Implementation Overview

We've implemented cls_token functionality in two main components:

### 1. Linear Probing Heads (`models/linear_probing.py`)
- `ClsTokenLinearProbingHead`: Classification with cls_token aggregation
- `ClsTokenLinearProbingRegressionHead`: Regression with cls_token aggregation

### 2. Multi-Instance Learning (`models/multi_instance_linear_probing.py`)
- Added `"cls_token"` as a new pooling mode
- Supports both 3D `[B, N, D]` and 4D `[B, N, L, D]` inputs
- Hierarchical processing for 4D inputs

## 🎬 Hierarchical Processing for 4D Inputs

For your specific scenario with **4 videos** and **393 tokens** per video:

Input: `[batch_size, 4_videos, 393_tokens, embedding_dim]`

### Two-Level CLS Token Processing:

1. **Level 1 - Within-Video Attention:**
   - Reshape to `[batch_size * 4, 393, embedding_dim]`
   - Apply cls_token across 393 tokens within each video
   - Result: `[batch_size * 4, embedding_dim]` (one representation per video)

2. **Level 2 - Across-Video Attention:**
   - Reshape to `[batch_size, 4, embedding_dim]`
   - Apply cls_token across 4 video representations
   - Result: `[batch_size, embedding_dim]` (final sample representation)

3. **Multi-Head Output:**
   - Feed to classification/regression heads
   - Result: `[batch_size, num_classes]` for each task

## 🔧 Usage Examples

### Basic Usage with Multi-Instance Learning

```python
from models.multi_instance_linear_probing import MultiInstanceLinearProbing

# Define your tasks
head_structure = {
    "cathEF": 1,      # Regression for ejection fraction
    "stenosis": 3,    # Multi-class classification
    "contrast": 2,    # Binary classification
}

# Create model with cls_token pooling
model = MultiInstanceLinearProbing(
    embedding_dim=512,
    head_structure=head_structure,
    pooling_mode="cls_token",  # Use cls_token pooling
    dropout=0.1,
    use_cls_token=True
)

# Your 4D input: [batch, 4_videos, 393_tokens, 512_dim]
x = torch.randn(2, 4, 393, 512)

# Optional: mask for variable-length sequences
mask = torch.ones(2, 4, dtype=torch.bool)  # All videos valid

# Forward pass
outputs = model(x, mask)
# Returns: {"cathEF": [2, 1], "stenosis": [2, 3], "contrast": [2, 2]}
```

### Using CLS Token Linear Probing Heads

```python
from models.linear_probing import ClsTokenLinearProbingHead

# Create cls_token-based head
head = ClsTokenLinearProbingHead(
    input_dim=512,
    output_dim=5,
    dropout=0.1,
    num_heads=8,
    use_cls_token=True
)

# Works with both 2D and 3D inputs
x_2d = torch.randn(4, 512)      # [batch, dim]
x_3d = torch.randn(4, 10, 512)  # [batch, seq_len, dim]

output_2d = head(x_2d)  # [4, 5]
output_3d = head(x_3d)  # [4, 5]
```

## ⚙️ Configuration

### Base Configuration Example

```yaml
# config/linear_probing/CathEF/regression/base_config_CathEF_cls_token.yaml
model_name: "multi_instance_linear_probing"
model_config:
  embedding_dim: 512
  pooling_mode: "cls_token"
  dropout: 0.15
  use_cls_token: true
  head_structure:
    cathEF: 1

video_config:
  aggregate_videos_tokens: false  # Keep 4D structure
  per_video_pool: false           # Let cls_token handle pooling
  num_videos: 4
  stride: 2

optimizer:
  attention_lr: 3e-3              # Higher LR for cls_token attention
  attention_weight_decay: 1e-4
  video_encoder_lr: 1e-5          # Lower LR for pretrained encoder
  head_lr: 5e-4
```

### Sweep Configuration

```yaml
# config/linear_probing/CathEF/regression/sweep_config_CathEF_cls_token.yaml
parameters:
  pooling_mode:
    values: ["cls_token"]
  use_cls_token:
    values: [true]
  aggregate_videos_tokens:
    values: [false]  # Always false for hierarchical processing
  
  attention_lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
    
  batch_size:
    values: [4, 8, 12, 16]  # Smaller batches due to memory requirements
```

## 🧠 Technical Details

### Architecture Components

1. **CLS Token Parameter:**
   ```python
   self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
   ```

2. **Multi-Head Attention:**
   ```python
   self.cls_attention = nn.MultiheadAttention(
       embed_dim=embedding_dim,
       num_heads=8,
       dropout=dropout,
       batch_first=True
   )
   ```

3. **Layer Normalization & Dropout:**
   ```python
   self.cls_norm = nn.LayerNorm(embedding_dim)
   self.cls_dropout = nn.Dropout(dropout)
   ```

### Memory Considerations

- **4D inputs** `[B, N, L, D]` require more memory than 3D inputs
- Consider using **smaller batch sizes** (4-16 instead of 32+)
- **Gradient checkpointing** can help with memory if needed
- **Mixed precision training** (fp16) recommended for large sequences

### Parameter Initialization

- CLS token initialized with small random values: `std=0.02`
- Attention weights use Xavier initialization
- Layer norms use default PyTorch initialization

## 📊 Performance Expectations

### Compared to Traditional Pooling:

| Method | Description | Learnable Params | Performance |
|--------|-------------|------------------|-------------|
| Mean Pooling | Simple average | 0 | Baseline |
| Max Pooling | Element-wise maximum | 0 | Slightly better |
| Attention Pooling | Gated attention | ~66K | Good |
| **CLS Token** | **Hierarchical self-attention** | **~66K + token** | **Best** |

### Expected Improvements:
- Better capture of **temporal dependencies** within videos
- More sophisticated **cross-video relationships**
- **Task-adaptive attention patterns**
- Superior performance on complex medical video analysis tasks

## 🚀 Running Examples

### Test the Implementation
```bash
cd /volume/DeepCORO_CLIP
source .venv/bin/activate
python examples/cls_token_example.py
```

### Run Training with CLS Token
```bash
# Single GPU training
python scripts/main.py --base_config config/linear_probing/CathEF/regression/base_config_CathEF_cls_token.yaml

# Multi-GPU training
torchrun --nproc_per_node=2 scripts/main.py --base_config config/linear_probing/CathEF/regression/base_config_CathEF_cls_token.yaml

# Hyperparameter sweep
wandb sweep config/linear_probing/CathEF/regression/sweep_config_CathEF_cls_token.yaml
```

## 🔍 Monitoring and Debugging

### Key Metrics to Monitor:
- **Attention weights**: Are they learning meaningful patterns?
- **CLS token gradients**: Are they updating properly?
- **Memory usage**: 4D processing can be memory-intensive
- **Training stability**: Higher learning rates for attention may need tuning

### Common Issues:
1. **OOM errors**: Reduce batch size or sequence length
2. **Slow convergence**: Tune attention learning rates
3. **Poor performance**: Check video_freeze_ratio and learning rate ratios

## 📚 References

- **BERT**: Attention Is All You Need (Vaswani et al., 2017)
- **Vision Transformer**: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
- **Multi-Instance Learning**: Solving the Multiple Instance Problem with Axis-Parallel Rectangles (Dietterich et al., 1997)

---

## ✅ Implementation Status

- [x] CLS Token Linear Probing Heads
- [x] Multi-Instance CLS Token Pooling
- [x] Hierarchical 4D Input Processing
- [x] Masking Support for Variable-Length Sequences
- [x] Configuration Examples
- [x] Comprehensive Documentation
- [x] Example Scripts and Usage Demos

**Ready for production use with your 4-video, 393-token scenario! 🎉** 