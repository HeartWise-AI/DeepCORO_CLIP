# Understanding `ground_truth_pos_probs` in Validation CSV Files

## Overview

The `ground_truth_pos_probs` column in validation epoch CSV files (e.g., `val_epoch_X.csv`) contains **predicted probabilities** for the **ground truth positive text descriptions** associated with each video. These probabilities indicate how confident the model is that the ground truth texts match the video.

## Complete Pipeline Flow

### 1. **Video and Text Encoding**
   - **File**: [models/video_encoder.py](models/video_encoder.py)
   - Videos are encoded into embeddings using the mVIT backbone
   - **File**: [models/text_encoder.py](models/text_encoder.py)
   - Text descriptions are encoded using BioMedBERT

### 2. **Similarity Computation**
   - **File**: [runners/video_constrative_learning_runner.py](runners/video_constrative_learning_runner.py)
   - **Function**: `_compose_logits()` (lines 298-345)

   **Process:**
   ```python
   # Step 1: Compute cosine similarity
   sim = torch.matmul(v, t.t())  # v: video embeddings, t: text embeddings

   # Step 2: Optional SigLIP gating (if enabled)
   if use_siglip_gating:
       sim = sim * torch.sigmoid(sim)

   # Step 3: Temperature scaling
   if apply_temperature:
       tau = torch.exp(self.log_temp)  # learnable temperature parameter
       logits = sim / tau

   # Step 4: Optional bias adjustment
   if add_bias:
       logits = logits + bias_scale * logit_bias
   ```

   **Key Point**: The similarity scores are **temperature-scaled** during retrieval (line 2331: `apply_temperature=True`)

### 3. **Retrieval and Top-K Selection**
   - **File**: [runners/video_constrative_learning_runner.py](runners/video_constrative_learning_runner.py)
   - **Location**: Lines 2324-2332 (validation epoch processing)

   For each video:
   - Compute similarity with all candidate texts
   - Extract top-K predictions using `torch.topk()`
   - Store indices and scores for further processing

### 4. **Probability Conversion for Ground Truth Texts**
   - **File**: [runners/video_constrative_learning_runner.py](runners/video_constrative_learning_runner.py)
   - **Location**: Lines 2420-2425

   ```python
   # For each video with ground truth indices
   logits_row = scores_chunk[row_offset]  # temperature-scaled similarity scores
   prob_row = torch.softmax(logits_row, dim=0)  # convert to probabilities

   # Extract probabilities for ground truth indices only
   positive_probs = [
       float(prob_row[idx].item())
       for idx in gt_indices
       if 0 <= idx < prob_row.size(0)
   ]
   local_positive_scores[global_row] = positive_probs
   ```

   **Important**: This uses **softmax** probabilities, which are normalized across all candidate texts.

### 5. **Gathering Results Across GPUs**
   - **File**: [runners/video_constrative_learning_runner.py](runners/video_constrative_learning_runner.py)
   - **Location**: Lines 2714-2979

   In distributed training:
   - Each GPU processes a subset of videos
   - Results are gathered to rank 0 using `_all_gather_objects()`
   - `local_positive_scores` → `global_positive_scores`

### 6. **CSV Export with Sigmoid Conversion**
   - **File**: [utils/wandb_logger.py](utils/wandb_logger.py)
   - **Function**: `save_retrieval_results()` (lines 1103-1377)
   - **Location**: Lines 1262-1297

   ```python
   def score_to_prob(score_val: float) -> float:
       # NOTE: score_val is already temperature-scaled from retrieval
       # This applies sigmoid to convert to a probability
       try:
           return 1.0 / (1.0 + math.exp(-score_val))
       except OverflowError:
           return 1.0 if score_val > 0 else 0.0

   # Extract ground truth probabilities
   if gt_indices:
       for idx_val in gt_indices:
           positive_texts.append(unique_texts[idx_val])
           positive_text_ids.append(resolve_text_id(idx_val))

       # Get scores from retrieval
       if positive_scores and i < len(positive_scores):
           positive_scores_for_video = [float(val) for val in positive_scores[i]]

       # If not available, extract from top-k predictions
       if len(positive_scores_for_video) != len(positive_texts):
           for idx_val in gt_indices:
               if idx_val in predicted_indices:
                   score_val = predicted_sims[predicted_indices.index(idx_val)]
                   positive_scores_for_video.append(float(score_val))
               else:
                   positive_scores_for_video.append(float("nan"))

   # Convert to probabilities using sigmoid
   positive_probs = [
       score_to_prob(score_val) if not math.isnan(score_val) else float("nan")
       for score_val in positive_scores_for_video
   ]

   # Format for CSV
   positive_probs_str = ", ".join(
       f"{prob_val:.3f}" if not math.isnan(prob_val) else "nan"
       for prob_val in positive_probs
   )
   ```

## Important Notes

### Two Different Probability Calculations

There are **two different probability values** in the codebase:

1. **Softmax Probabilities** (used for metrics, computed in runner):
   - Calculated at retrieval time (line 2421: `torch.softmax(logits_row, dim=0)`)
   - Normalized across all candidate texts
   - Sum to 1.0 across all texts
   - Stored in `local_positive_scores` → `global_positive_scores`

2. **Sigmoid Probabilities** (saved to CSV):
   - Calculated during CSV export (line 1267: `1.0 / (1.0 + math.exp(-score_val))`)
   - Independent for each text
   - Do NOT sum to 1.0
   - What you see in `ground_truth_pos_probs` column

### Why the Discrepancy?

The comment at line 1263-1265 explains:
```python
# Note: score_val is already temperature-scaled during retrieval (apply_temperature=True)
# See runners/video_constrative_learning_runner.py:2176-2184
# Do NOT divide by temperature again to avoid probability saturation
```

However, there's a **mismatch**:
- The retrieval code (line 2421) uses **softmax** probabilities
- The CSV export (line 1267) applies **sigmoid** to the temperature-scaled scores
- This means the CSV probabilities are recalculated differently from the metrics

### Interpreting the Values

**Value of 0.500**:
- Sigmoid(0) = 0.5
- This means the **temperature-scaled similarity score was exactly 0**
- The model is completely uncertain (neither positive nor negative similarity)
- Suggests poor alignment between video and text embeddings

**High values (0.8-1.0)**:
- Strong positive similarity
- Model is confident the text matches the video

**Low values (0.0-0.2)**:
- Strong negative similarity
- Model doesn't think the text matches

## Files Involved

| File | Purpose | Key Functions/Lines |
|------|---------|-------------------|
| [models/video_encoder.py](models/video_encoder.py) | Encode videos into embeddings | `forward()` |
| [models/text_encoder.py](models/text_encoder.py) | Encode text into embeddings | `forward()` |
| [runners/video_constrative_learning_runner.py](runners/video_constrative_learning_runner.py) | Main training/validation logic | Lines 298-345: `_compose_logits()`<br>Lines 2324-2425: Retrieval & probability calculation<br>Lines 2714-2979: Distributed gathering |
| [utils/wandb_logger.py](utils/wandb_logger.py) | CSV export and logging | Lines 1103-1377: `save_retrieval_results()`<br>Lines 1262-1297: Probability conversion |
| [dataloaders/video_clip_dataset.py](dataloaders/video_clip_dataset.py) | Dataset loading | Video/text pairing |
| [utils/config/clip_config.py](utils/config/clip_config.py) | Configuration | Temperature, recall@K settings |

## Configuration Parameters

- **`temperature`**: Learnable parameter (`log_temp` in runner), controls the sharpness of similarity distribution
- **`recall_k`**: List of K values for recall metrics (e.g., [1, 5, 10])
- **`retrieval.use_logit_bias_eval`**: Whether to apply bias during retrieval
- **`siglip.abnormal_margin`**: Additional margin for abnormal findings

## Debugging Poor Performance (0.500 probabilities)

If you're seeing many 0.500 probabilities:

1. **Check training progress**: Early epochs will have poor alignment
2. **Inspect similarity scores**: Look at raw cosine similarities before temperature scaling
3. **Verify embeddings**: Check that video and text embeddings are properly normalized
4. **Temperature value**: Check if temperature is too high (overly smoothed) or too low (too sharp)
5. **Data issues**: Verify that ground truth texts actually match the videos

## Example Query

To analyze probabilities in your validation CSV:

```python
import pandas as pd

df = pd.read_csv('val_epoch_X.csv')

# Parse the comma-separated probabilities
df['probs_list'] = df['ground_truth_pos_probs'].apply(
    lambda x: [float(p) for p in x.split(', ')] if pd.notna(x) else []
)

# Get mean probability per video
df['mean_prob'] = df['probs_list'].apply(lambda x: sum(x)/len(x) if x else 0)

# Distribution
print(df['mean_prob'].describe())
```
