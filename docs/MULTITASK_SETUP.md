# DeepCORO-CLIP Multitask Training Setup

This document describes the implementation of a new training setup for DeepCORO-CLIP, inspired by the SigLIP 2 architecture and training regime. The goal is to move beyond global video–text contrastive learning and add:

- **LocCa-style captioning decoder** for structured angiographic report generation
- **Self-supervised masked modeling** for enhanced video understanding
- **Multitask training** with shared encoder and configurable loss weights

## Architecture Overview

The multitask setup consists of four main components:

### 1. Shared Video Encoder
- Uses existing DeepCORO-CLIP backbone (e.g., ViT+TimeSformer)
- Outputs tokenized spatial-temporal features: `[B, T×P, D]`
- Supports both aggregated features (for contrastive learning) and token-level features (for captioning and masked modeling)

### 2. Text Encoder
- PubMedBERT-based text encoder
- Processes structured angiographic reports
- Outputs text embeddings for contrastive learning

### 3. Captioning Decoder (LocCa-style)
- Transformer decoder with causal attention
- Cross-attends to video tokens
- Generates autoregressive angiographic reports
- Supports biomedical tokenizer integration

### 4. Masked Video Modeling
- Self-supervised masked patch modeling
- Randomly masks video tokens and reconstructs them
- Uses lightweight decoder for reconstruction

## Components

### Models

#### `CaptioningDecoder` (`models/captioning_decoder.py`)
```python
@ModelRegistry.register("captioning_decoder")
class CaptioningDecoder(nn.Module):
    """
    LocCa-style transformer decoder for generating structured angiographic reports.
    
    Features:
    - Causal attention for autoregressive generation
    - Cross-attention to video tokens
    - Biomedical tokenizer support
    - Configurable architecture parameters
    """
```

**Key Parameters:**
- `vocab_size`: Vocabulary size (default: 30522 for BERT)
- `hidden_size`: Hidden dimension (default: 512)
- `num_layers`: Number of decoder layers (default: 6)
- `num_heads`: Number of attention heads (default: 8)
- `use_biomed_tokenizer`: Use biomedical tokenizer (default: True)

#### `MaskedVideoModeling` (`models/masked_video_modeling.py`)
```python
@ModelRegistry.register("masked_video_modeling")
class MaskedVideoModeling(nn.Module):
    """
    Masked Video Modeling for self-supervised learning.
    
    Features:
    - Random token masking
    - Lightweight decoder for reconstruction
    - Configurable mask ratio and architecture
    """
```

**Key Parameters:**
- `mask_ratio`: Ratio of tokens to mask (default: 0.75)
- `decoder_hidden_size`: Decoder hidden dimension (default: 256)
- `decoder_layers`: Number of decoder layers (default: 2)
- `mask_token_learnable`: Learnable mask token (default: True)

### Loss Functions

#### `MultitaskLoss` (`utils/loss/multitask_loss.py`)
```python
@LossRegistry.register(LossType.MULTITASK)
class MultitaskLoss(nn.Module):
    """
    Multitask loss combining contrastive, captioning, and masked video modeling losses.
    
    Supported losses:
    - Contrastive loss (video ↔ text alignment)
    - Captioning loss (autoregressive report generation)
    - Masked video modeling loss (self-supervised learning)
    - Optional distillation loss (future task)
    """
```

**Loss Types:**
- **Contrastive**: Sigmoid (SigLIP-style) or Softmax (CLIP-style)
- **Captioning**: Cross-entropy with optional label smoothing
- **Masked Modeling**: MSE loss on masked tokens

**Configurable Weights:**
```python
loss_weights = {
    "contrastive": 1.0,
    "captioning": 1.0,
    "masked_modeling": 0.1,
    "distillation": 0.0,  # Future task
}
```

### Training Components

#### `MultitaskPretrainingProject` (`projects/multitask_pretraining_project.py`)
```python
@ProjectRegistry.register('DeepCORO_multitask')
class MultitaskPretrainingProject(BaseProject):
    """
    Multitask pretraining project combining:
    - Contrastive learning (video ↔ text)
    - Captioning (autoregressive report generation)
    - Masked video modeling (self-supervised learning)
    """
```

#### `MultitaskRunner` (`runners/multitask_runner.py`)
```python
@RunnerRegistry.register("DeepCORO_multitask")
class MultitaskRunner:
    """
    Multitask runner for DeepCORO-CLIP with captioning and masked video modeling.
    
    Handles:
    - Contrastive learning (video ↔ text)
    - Captioning (autoregressive report generation)
    - Masked video modeling (self-supervised learning)
    """
```

## Configuration

### Base Configuration (`config/clip/multitask_config.yaml`)

The configuration file includes all necessary parameters for multitask training:

```yaml
# Model parameters
pipeline_project: "DeepCORO_multitask"
model_name: "mvit"
pretrained: true

# Captioning decoder parameters
vocab_size: 30522
decoder_layers: 6
decoder_heads: 8
decoder_intermediate_size: 2048
max_position_embeddings: 512
use_biomed_tokenizer: true
max_text_length: 512
max_generation_length: 128

# Masked video modeling parameters
mvm_decoder_hidden_size: 256
mvm_decoder_layers: 2
mvm_decoder_heads: 8
mask_ratio: 0.75
mask_token_learnable: true
norm_predict_loss: true

# Learning rates for different components
text_lr: 0.00002
captioning_lr: 0.00006171328778901703
captioning_weight_decay: 0.01
mvm_lr: 0.000006171328778901703
mvm_weight_decay: 0.01

# Loss configuration
loss_name: "multitask"
contrastive_loss_type: "sigmoid"
captioning_loss_type: "cross_entropy"
masked_modeling_loss_type: "mse"
label_smoothing: 0.1
ignore_index: -100

# Loss weights
loss_weights:
  contrastive: 1.0
  captioning: 1.0
  masked_modeling: 0.1
  distillation: 0.0
```

## Usage

### 1. Testing the Setup

Run the test script to verify all components work correctly:

```bash
python test_multitask_setup.py
```

This will test:
- Model initialization
- Forward passes
- Loss computation
- Basic training step
- Caption generation

### 2. Training

To start multitask training:

```bash
python main.py --config config/clip/multitask_config.yaml
```

### 3. Configuration Options

#### Loss Weight Scheduling

You can enable dynamic loss weight scheduling:

```yaml
use_loss_weight_scheduler: true
initial_loss_weights:
  contrastive: 1.0
  captioning: 0.5
  masked_modeling: 0.1
final_loss_weights:
  contrastive: 1.0
  captioning: 1.0
  masked_modeling: 0.1
loss_warmup_steps: 1000
loss_total_steps: 10000
loss_schedule_type: "linear"  # "linear", "cosine", "step"
```

#### Captioning Generation

Configure caption generation parameters:

```yaml
max_generation_length: 128
captioning_do_sample: false
captioning_temperature: 1.0
```

#### Masked Video Modeling

Configure MVM parameters:

```yaml
mask_ratio: 0.75
mask_token_learnable: true
norm_predict_loss: true
```

## Training Process

### Forward Pass

1. **Video Encoding**: Video tokens are extracted at both token-level and aggregated levels
2. **Text Encoding**: Text reports are encoded using PubMedBERT
3. **Captioning**: Autoregressive generation with cross-attention to video tokens
4. **Masked Modeling**: Random masking and reconstruction of video tokens

### Loss Computation

The multitask loss combines:

1. **Contrastive Loss**: Aligns video and text embeddings
2. **Captioning Loss**: Cross-entropy on generated report tokens
3. **Masked Modeling Loss**: MSE on reconstructed masked tokens

### Metrics

The training tracks:

- **Contrastive Metrics**: Recall@K, NDCG@K, alignment score, median rank
- **Captioning Metrics**: BLEU, ROUGE (future implementation)
- **Masked Modeling Metrics**: Reconstruction loss

## Key Features

### 1. Shared Encoder Architecture
- Single video encoder shared across all tasks
- Efficient parameter sharing
- Consistent feature representation

### 2. Flexible Loss Weighting
- Configurable loss weights
- Optional dynamic weight scheduling
- Easy experimentation with different task balances

### 3. Biomedical Domain Integration
- PubMedBERT tokenizer for medical text
- Structured angiographic report generation
- Domain-specific vocabulary

### 4. Self-Supervised Learning
- Masked video modeling for enhanced understanding
- No additional annotations required
- Improves video representation learning

### 5. Scalable Training
- Distributed training support
- Mixed precision training
- Gradient accumulation
- Comprehensive logging and checkpointing

## Future Enhancements

### 1. Distillation Support
- Teacher-student distillation
- Knowledge transfer from larger models
- Self-distillation capabilities

### 2. Advanced Captioning Metrics
- BLEU score implementation
- ROUGE score implementation
- Medical domain-specific metrics

### 3. Enhanced Masking Strategies
- Temporal masking
- Spatial masking
- Adaptive masking ratios

### 4. Multi-Modal Integration
- Additional modalities (ECG, clinical notes)
- Cross-modal attention mechanisms
- Unified representation learning

## References

- **SigLIP 2**: [https://arxiv.org/abs/2405.09372](https://arxiv.org/abs/2405.09372)
- **LocCa**: Location-aware Captioning
- **DeepCORO-CLIP**: Baseline implementation
- **OpenCLIP**: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- **BLIP**: [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)

## Milestone

This implementation targets **Phase 2 of DeepCORO-CLIP validation for STICH3C**, supporting:
- Structured report generation
- Enhanced interpretability
- More robust spatial grounding
- Improved performance in high-risk ischemic cardiomyopathy patients