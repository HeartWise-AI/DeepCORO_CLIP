# SigLIP Multi-Positive Pipeline Reference

This note collects the key config knobs and code paths that implement SigLIP multi-positive sampling/training so you can iterate without hunting across the repo.

## 1. Training Config

`config/clip/siglip_ddp_config.yaml` wires the canonicalized CSVs and exposes every SigLIP knob used by the runner:

```yaml
# Data bindings
data_filename: "output_dataset/siglip_generated/videos.csv"
siglip_texts_path: "output_dataset/siglip_generated/texts.csv"
siglip_max_positive_per_video: 8
siglip_negatives_per_video: 32
siglip_round_robin_sampling: true
siglip_max_segments_per_video: 15

# Severity-aware weighting + loss options
siglip_positive_severity_weights:
  normal: 1.0
  mild: 2.0
  moderate: 10.0
  severe: 10.0
  critical: 10.0
  cto: 10.0
siglip_enable_severity_weighting: true
loss_name: "INFONCE_LOSS_DDP"
siglip_positive_loss_weight: 1.0
siglip_negative_loss_weight: 1.0
```

Switching `loss_name` between `INFONCE_LOSS_DDP` and `SIGLIP_DDP` is what toggles `MultiPositiveInfoNCELoss` vs. `SiglipPairwiseLoss` inside the runner (see §5).

## 2. Canonical prompt generation

`dataset_creation/generate_dataset.py` is responsible for normalizing `texts.csv`/`videos.csv`. The CLI

```bash
python dataset_creation/generate_dataset.py \
  --canonicalize-siglip \
  --siglip-output-dir output_dataset/siglip_generated
```

funnels each prompt through `build_canonical_siglip_prompt` (`dataset_creation/generate_dataset.py:214-335`). Expanded behavior:

- **Tag parsing** – `parse_tags_field` extracts `segment`, `bin`, `medina`, etc. allowing prompts without explicit metadata columns to still resolve cleanly.
- **Severity normalization** – `_infer_severity_label` folds free-text `disease_severity` and `bin` labels into `{normal,mild,moderate,severe,critical}` so “normal segment” outputs stay consistent even when the raw prompt mentioned “stenosis”.
- **Attribute detection** – keywords in the original prompt (calcification, thrombus, stent, IFR, Medina) are converted into canonical codes and plain-English clauses. Extras are deduped so we never produce `thrombus with thrombus`.
- **Tree-aware collateral handling** – regex passes identify “receives/gives collaterals” sentences to append after the base description so collateral info survives deduplication.
- **Dedup key construction** – `(segment_code, base_code, extras_codes)` ensures identical anatomical findings collapse to a single text row even if the original wording differed.

With these safeguards, every `positive_text_ids` entry references a deduped, severity-tagged row that already reflects the cleaned phrasing (`"Distal LAD; normal segment."`, `"Proximal RCA; thrombus present."`).

## 3. SigLIP resource loader

`dataloaders/siglip_support.py` encapsulates every SigLIP-specific lookup and sampler:

```python
class SiglipSupport:
    def __init__(self, dataset: "VideoClipDataset", kwargs: Dict[str, Any]):
        self.texts_path = kwargs.pop("siglip_texts_path", None)
        self.negatives_per_video = int(kwargs.pop("siglip_negatives_per_video", 256))
        self.max_segments_per_video = int(kwargs.pop("siglip_max_segments_per_video", 15))
        self.positive_severity_weights = {...}  # overridden by config
        ...
        if self.enabled:
            self._load_resources()
```

Detailed responsibilities:

- **`_load_resources`** (`siglip_support.py:210-380`):  
  - Validates required CSV columns, normalizes `tree` hints, and records per-text dictionaries in `self.text_lookup`.  
  - Computes class-weight/logit-bias statistics via `compute_class_statistics`, allowing the sampler to balance segments/domains.  
  - Tokenizes *every* prompt once through `get_tokenizer` so later batch assembly is just an index lookup.  
  - Reads `videos.csv`, parses each `positive_text_ids` pipe list, and stores `(text_id, edge_weight)` pairs plus study-level severity votes.
- **`filter_positive_pairs`** (`siglip_support.py:510-586`): prunes contradictory or cross-tree positives, enforces `siglip_max_segments_per_video`, and prefers the most severe/specific finding per (tree,segment).
- **`_compute_positive_weight` / `compute_positive_weight`** (`siglip_support.py:598-665`): merges CSV soft weights, edge weights, and `siglip_positive_severity_weights` into a capped scalar fed to the multi-positive mask.
- **Negative assembly** – `build_negative_candidates` uses tree/segment indices to prioritize “same segment”, “same tree”, and global negatives, falling back to the full catalog if necessary.
- **Sampler wiring** – constructs `SingleHeadRetrievalSampler` (`utils/siglip/single_head_sampler.py`) which handles per-epoch phase toggles, contradiction boosts, and per-video audits when we eventually switch to the sampler-driven path.

## 4. Dataset assembly

`VideoClipDataset` injects the SigLIP positives/negatives while reading `videos.csv`:

```python
# dataloaders/video_clip_dataset.py:247-305
video_id_str = row.get(self.siglip.video_id_column)
positive_pairs = self.siglip.get_positive_pairs(video_id_str)
if not positive_pairs:
    positive_pairs = self._fallback_positive_pairs(row)
valid_pairs = [
    (text_id, self.siglip.compute_positive_weight(text_id, weight))
    for text_id, weight in positive_pairs
    if text_id in self.siglip.text_lookup
]
filtered_pairs = self.siglip.filter_positive_pairs(valid_pairs, tree_hint=tree_hint)
self.video_positive_texts.append(filtered_pairs)
self.video_negative_pool.append(
    self.siglip.build_negative_candidates({tid for tid, _ in filtered_pairs}, tree_hint)
)
```

Additional details:

- **Tree resolution** – if the CSV omits dominance info, `_resolve_tree_from_positive_pairs` inspects the surviving texts to infer “left” vs “right”, ensuring downstream filtering never mixes trees.
- **Main-structure supervision** – each video stores `self.main_structure_labels` (left/right) so the runner can train an auxiliary BCE loss if `main_structure_loss_weight > 0`.
- **Fallback safety** – `_fallback_positive_pairs` and `_fallback_negative_ids` guarantee batches still carry text supervision even if SigLIP metadata is incomplete.
- **Per-segment capping** – `_cap_positive_pairs_by_segment` enforces `siglip_max_positive_per_video` as a limit on unique segments, so we never send multiple copies of the same vessel description into the SigLIP batch.

During batch collation, `build_siglip_batch` (`video_clip_dataset.py:724-784`) performs:

1. Gather dataset indices for the sampled file paths.
2. Slice each video’s positives down to `siglip_max_positive_per_video`.
3. Pull a rolling window of negatives per video (`siglip_negatives_per_video`).
4. Deduplicate text IDs across the batch so we only encode each prompt once.
5. Use `SiglipSupport.encode_text_batch` to fetch `input_ids`/`attention_mask`.
6. Fill `positive_mask[row, col] = 1` and accumulate `positive_weights[row, col] += severity_weight`.

The resulting dict is dropped into the runner as `siglip_batch`.

## 5. Runner integration

Inside `VideoContrastiveLearningRunner` (`runners/video_constrative_learning_runner.py`):

```python
self.use_siglip_infonce = self.siglip_active and self.siglip_loss_mode == "infonce_loss_ddp"
self.multi_positive_loss = MultiPositiveInfoNCELoss() if self.use_siglip_infonce else None
...
siglip_batch = dataset.build_siglip_batch(sample_ids)
...
if self.use_siglip_infonce:
    contrastive_loss = self.multi_positive_loss(
        logits=self._scaled_logits(video_emb, text_emb),
        pos_mask=siglip_batch["positive_mask"],
        pos_weights=siglip_batch["positive_weights"] if self.siglip_severity_weighting else None,
    )
```

Expanded view:

- **SigLIP payload creation** – `_build_siglip_payload` checks that SigLIP is enabled, requests the dataset batch, and moves all tensors to the active GPU (`runners/video_constrative_learning_runner.py:946-965`).
- **`_scaled_logits`** – centralizes the cosine similarity computation, normalizes embeddings with `F.normalize`, clamps the learnable temperature (`log_temp`), and returns the scaled matrix shared by both InfoNCE directions.
- **`_compute_siglip_infonce_loss`** (`runners/video_constrative_learning_runner.py:1003-1020`):  
  1. Encodes text prompts through the CLIP text encoder (or reuses the cached embeddings if we precompute).  
  2. Calls `_scaled_logits(video_embeddings, text_embeddings)` to get `[batch, num_texts]` similarities.  
  3. Applies `MultiPositiveInfoNCELoss`, passing severity weights when `siglip_enable_severity_weighting` is set.
- **`MultiPositiveInfoNCELoss` internals** (`utils/loss/multi_positive_infonce.py`):  
  - Validates tensor shapes and multiplies `pos_mask` with any provided weights.  
  - Runs `_weighted_ce` twice (video→text and text→video), each time computing a normalized cross-entropy over the logits with multiple positives per row/column.  
  - Supports optional importance weighting (disabled in current config) to up-weight rows/cols with many positives.  
  - Returns zero when a batch accidentally has no positives, keeping gradients finite.

Together these steps ensure every clip is trained against *all* of its matching prompts, with severity-aware scaling and symmetric InfoNCE so the text tower also learns the multi-positive alignment.

---

With these references you can tweak the YAML, sampler, or loss in isolation and immediately see how multi-positive SigLIP batches are produced end-to-end.
