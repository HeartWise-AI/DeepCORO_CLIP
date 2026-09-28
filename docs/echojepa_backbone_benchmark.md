# EchoJEPA ViT-L as DeepCORO_CLIP video backbone: 4-GPU benchmark (2026-09-28)

Purpose: a GPU benchmark, not a converged model. One epoch per stage, W&B off,
`rand_augment` off. Host: 4x NVIDIA RTX PRO 5000 Blackwell (48 GB), 64 CPU cores,
torch 2.14 (NGC 26.08), data on NFS (`/media/data1`).

## What was run

| Stage | Config | Backbone | Data | Per-GPU batch |
|---|---|---|---|---|
| 1. CLIP contrastive (video/report) | `config/clip/echojepa_vitl_clip_benchmark.yaml` | EchoJEPA ViT-L/16 coronary encoder `echojepa-vitl-coro-cooldown-ep60` (RoPE, 16 f @ 224, bf16 autocast), top 87% of params trainable; PubMedBERT text | 117,120 train / 25,306 val videos, single-video mode | 24 clips (global 96) |
| 2. Linear probe, 18-target stenosis regression (multi-video MIL) | `config/linear_probing/stenosis/echojepa_vitl_stenosis_benchmark.yaml` | Stage-1 VideoEncoder checkpoint | same CSV, 10 videos per study | 2 studies = 20 clips (global 80) |

Launcher: `bash scripts/run_echojepa_benchmark.sh --selected_gpus 0,1,2,3`
(nvidia-smi sampled every 5 s; `scripts/summarize_gpu_benchmark.py` prints the tables below).

## Single-GPU backbone probe (before the run)

Fwd + bwd of the EchoJEPA ViT-L inside `VideoEncoder`, random clips, freeze ratio 0.87:

| Setting | Batch | Peak memory | Throughput |
|---|---|---|---|
| no activation checkpointing | 16 / 20 / 24 / 28 | 25.5 / 30.9 / 36.3 / 41.8 GiB | 30.1 / 30.1 / 29.3 / 28.9 clips/s |
| no activation checkpointing | 32 | OOM | |
| activation checkpointing | 32 / 48 / 64 / 80 | 8.1 / 10.5 / 12.9 / 15.3 GiB | 23.1 / 23.0 / 22.7 / 22.4 clips/s |

Batch 24 without checkpointing was chosen: largest that fits with headroom for the
text encoder and DDP buffers; checkpointing buys batch size but costs ~25% throughput.
Compute ceiling for the backbone alone: ~30 clips/s/GPU, ~120 clips/s on 4 GPUs.

## Stage 1: CLIP contrastive pretraining

Log dir: `logs/echojepa_benchmark_20260928-202134`. Wall time 61.9 min, of which
~11 min was start-up (each rank parses the 1.1 GB alpha-separated CSV with the
pandas python engine and loads the 5 GB EchoJEPA checkpoint).

| | steps | time | clips/s (4 GPUs) |
|---|---|---|---|
| train epoch 1 | 1,220 | 42.3 min | ~46 |
| val epoch 1 | 264 | 7.9 min | ~53 |

Steady state (model resident) per GPU, sampled every 5 s:

| GPU | util mean | util p50 | peak mem | mean power |
|---|---|---|---|---|
| 0 | 93.1% | 100% | 44.4 GiB | 168 W |
| 1 | 97.2% | 100% | 44.4 GiB | 178 W |
| 2 | 97.3% | 100% | 44.4 GiB | 179 W |
| 3 | 48.3% | 51% | 44.4 GiB | 171 W |

Val loss 4.32 -> 3.45 after one epoch; best checkpoint
`outputs/DeepCORO_clip/DeepCORO_Clip_EchoJEPA_benchmark/20260928-202145_no_wandb/checkpoints/best_model_epoch_0.pt`.

Reading: 46 clips/s is ~40% of the 120 clips/s the backbone can do. GPU 3's
bimodal utilisation (0% or 100%) is the signature of a data-starved rank: it waits
on its dataloader while the other three ranks spin in the NCCL all-reduce, which
nvidia-smi counts as 100% busy. The step rate swung between 1.0 and 3.3 s/step and
the CPU load average reached 84 on 64 cores. Stage 1 is bound by CPU video decoding
(OpenCV, 16 frames per .avi over NFS, 56 worker processes), not by the GPUs.

## Stage 2: stenosis linear probe

Log dir: `logs/echojepa_benchmark_20260928-213546`. Wall time 49.2 min (~9 min start-up).

| | steps | time | clips/s (4 GPUs) |
|---|---|---|---|
| train epoch 1 | 2,846 | 35.2 min | ~108 |
| val epoch 1 | 611 | 5.5 min | ~147 |

Steady state per GPU: 92.0 to 92.7% mean utilisation, p50 100%, peak 38.7 GiB,
276 to 290 W. All four ranks stayed balanced. Val main loss 45.16 after one epoch
(the run is too short for the per-segment MAE/AUROC to mean anything).

Reading: 108 clips/s is ~90% of the backbone ceiling. The multi-video loader hands
each worker 10 clips per sample, so the per-step Python and dispatch overhead is
spread over 4x more clips than in Stage 1, and the decode path keeps up.

## Fixes made along the way (all on branch `feature/echojepa-backbone`)

- `scripts/runner.sh` forced `NCCL_ALGO=Tree` and `NCCL_P2P_LEVEL=NVL`; this NCCL has
  no Tree implementation for Broadcast ("No algorithm/protocol available for function
  Broadcast"), which killed the first launch. Both are now opt-in via the environment.
- The probe config must use the same `aggregator_depth` (and `num_heads`) as the CLIP
  config, otherwise `load_state_dict` fails on `aggregator.blocks.*`.
- `yq` on this host is the Python jq wrapper; `scripts/set_yaml_key.py` replaces
  `yq eval -i` in the launchers.

## To turn this into a real training run

Raise `epochs`, set `use_wandb: true` and `rand_augment: true`, and expect Stage 1 to
be decode-bound at the current ~46 clips/s unless frames are pre-decoded (e.g. .npy
clips, which `utils/video.py` already accepts) or the loader is moved to a faster
decoder.
