#!/usr/bin/env python3
"""
Extract and compare PRE vs POST PCI study-level embeddings.

This script:
1. Loads the trained DeepCORO model
2. Extracts study-level embeddings for PRE-PCI (diagnostic) studies
3. Extracts study-level embeddings for POST-PCI studies (same patients)
4. Computes cosine similarity between matched pairs
5. Analyzes embedding changes after PCI

Usage:
    python extract_and_compare_pci_embeddings.py --config config/inference/base_config_full_4828.yaml
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

sys.path.insert(0, '/volume/DeepCORO_CLIP')

from models.video_encoder import VideoEncoder
from models.multi_instance_linear_probing import MultiInstanceLinearProbing
from projects.linear_probing_project import VideoMILWrapper
from dataloaders.video_dataset import VideoDataset
from torch.utils.data import DataLoader
from utils.config.linear_probing_config import LinearProbingConfig


def load_model(config: LinearProbingConfig, device: torch.device) -> VideoMILWrapper:
    """Load the VideoMILWrapper model from checkpoint."""

    # Build video encoder
    video_encoder = VideoEncoder(
        backbone=config.model_name,
        pretrained=config.pretrained,
        freeze_ratio=config.video_freeze_ratio,
        num_frames=config.frames,
        num_heads=config.num_heads,
        aggregator_depth=config.aggregator_depth,
        aggregate_videos_tokens=config.aggregate_videos_tokens,
        per_video_pool=config.per_video_pool,
    )

    # Get embedding dimension
    embedding_dim = video_encoder.embedding_dim
    print(f"Video encoder embedding_dim: {embedding_dim}")

    if '+' in config.pooling_mode:
        head_input_dim = 2 * embedding_dim
    else:
        head_input_dim = embedding_dim

    # Build MIL model - need to match checkpoint structure
    # The checkpoint uses separate_video_attention=True and num_attention_heads that divides embedding_dim
    # From checkpoint: cls_attention has in_proj of [1536, 512], so 512 embedding dim, likely 8 heads
    # From checkpoint: attention_V has shape [256, 512], so attention_hidden=256
    num_heads = 8  # 512 / 8 = 64
    attention_hidden = getattr(config, 'attention_hidden', 256)

    mil_model = MultiInstanceLinearProbing(
        embedding_dim=embedding_dim,
        head_structure=config.head_structure or {'dummy': 1},
        pooling_mode=config.pooling_mode,
        use_cls_token=config.use_cls_token,
        num_attention_heads=num_heads,
        separate_video_attention=True,  # Checkpoint uses separate attention
        normalization_strategy=config.normalization_strategy,
        attention_hidden=attention_hidden,
    )

    # Load full model checkpoint
    if config.inference_model_path and os.path.exists(config.inference_model_path):
        ckpt = torch.load(config.inference_model_path, map_location=device, weights_only=False)
        print(f"Loading checkpoint from {config.inference_model_path}")
        print(f"Checkpoint keys: {ckpt.keys()}")

        # This checkpoint has 'linear_probing' key with video_encoder and mil_model weights
        if 'linear_probing' in ckpt:
            state_dict = ckpt['linear_probing']

            # Extract video encoder weights (remove 'video_encoder.' prefix)
            video_encoder_dict = {}
            for k, v in state_dict.items():
                if k.startswith('video_encoder.'):
                    new_key = k[len('video_encoder.'):]
                    video_encoder_dict[new_key] = v

            if video_encoder_dict:
                video_encoder.load_state_dict(video_encoder_dict, strict=False)
                print(f"Loaded {len(video_encoder_dict)} video encoder parameters")

            # Extract MIL model weights (remove 'mil_model.module.' prefix from DDP)
            mil_model_dict = {}
            for k, v in state_dict.items():
                if k.startswith('mil_model.module.'):
                    new_key = k[len('mil_model.module.'):]
                    mil_model_dict[new_key] = v
                elif k.startswith('mil_model.'):
                    new_key = k[len('mil_model.'):]
                    mil_model_dict[new_key] = v

            if mil_model_dict:
                # Only load matching keys
                current_state = mil_model.state_dict()
                filtered_dict = {k: v for k, v in mil_model_dict.items() if k in current_state}
                mil_model.load_state_dict(filtered_dict, strict=False)
                print(f"Loaded {len(filtered_dict)} MIL model parameters")

    # Wrap
    wrapper = VideoMILWrapper(video_encoder, mil_model, num_videos=config.num_videos)
    wrapper.to(device)
    wrapper.eval()

    return wrapper, embedding_dim, head_input_dim


def custom_collate_fn(batch):
    """Custom collate function that includes video paths."""
    videos, targets, paths = zip(*batch)

    # Multi-video: videos[0] is np.ndarray [num_videos, F, H, W, C]
    if isinstance(videos[0], np.ndarray) and videos[0].ndim == 5:
        videos_tensor = torch.stack([torch.from_numpy(v) for v in videos])
    elif isinstance(videos[0], np.ndarray) and videos[0].ndim == 4:
        videos_tensor = torch.stack([torch.from_numpy(v) for v in videos])
    else:
        raise ValueError(f"Unexpected video format")

    return {
        "videos": videos_tensor,
        "targets": {},
        "video_fname": paths
    }


def create_dataloader(csv_path: str, config: LinearProbingConfig, split: str = 'inference') -> Tuple[DataLoader, 'VideoDataset', Dict[str, str]]:
    """Create a dataloader for embedding extraction with study ID mapping."""

    dataset = VideoDataset(
        data_filename=csv_path,
        split=split,
        target_label=None,  # No labels needed for embedding extraction
        datapoint_loc_label=config.datapoint_loc_label,
        num_frames=config.frames,
        backbone=config.model_name,
        normalize=True,
        mean=config.dataset_mean,
        std=config.dataset_std,
        rand_augment=False,
        stride=config.stride,
        resize=config.resize,
        multi_video=config.multi_video,
        groupby_column=config.groupby_column,
        num_videos=config.num_videos,
        shuffle_videos=False,  # Don't shuffle for reproducibility
    )

    # Build filename -> study_id mapping from the CSV
    df = pd.read_csv(csv_path, sep='α', engine='python')
    df = df[df['Split'] == split]
    filename_to_study = dict(zip(df[config.datapoint_loc_label], df[config.groupby_column]))

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    return dataloader, dataset, filename_to_study


def extract_study_embeddings(
    model: VideoMILWrapper,
    dataloader: DataLoader,
    filename_to_study: Dict[str, str],
    device: torch.device,
    embedding_dim: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract study-level embeddings by hooking into the MIL pooling layer."""

    all_embeddings = []
    all_study_ids = []

    model.eval()

    # Hook to capture pooled features before heads
    pooled_features_list = []

    def capture_pooled(module, input, output):
        # The input to heads is the pooled embedding
        if isinstance(input, tuple):
            pooled_features_list.append(input[0].detach().cpu())
        else:
            pooled_features_list.append(input.detach().cpu())

    # Register hook on the first head
    head_name = list(model.mil_model.heads.keys())[0]
    hook_handle = model.mil_model.heads[head_name].register_forward_hook(capture_pooled)

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                videos = batch['videos'].to(device)
                video_fnames = batch['video_fname']  # Tuple of lists of filenames
                batch_size = videos.shape[0]

                # Clear previous features
                pooled_features_list.clear()

                # Forward pass - triggers hook
                _ = model(videos)

                # Get captured features
                if pooled_features_list:
                    batch_embeddings = pooled_features_list[0]
                    all_embeddings.append(batch_embeddings)

                    # Get study IDs from filenames
                    for i in range(batch_size):
                        # video_fnames[i] is a list of filenames for this study
                        fnames_for_study = video_fnames[i]
                        # Find first non-PAD filename to get study ID
                        study_id = None
                        for fname in fnames_for_study:
                            if fname != "PAD" and fname in filename_to_study:
                                study_id = filename_to_study[fname]
                                break
                        if study_id is None:
                            study_id = f"unknown_{i}"
                        all_study_ids.append(study_id)

    finally:
        hook_handle.remove()

    # Concatenate all embeddings
    if all_embeddings:
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
    else:
        embeddings_tensor = torch.empty(0, embedding_dim)

    return embeddings_tensor, all_study_ids


def compare_embeddings(
    pre_embeddings: torch.Tensor,
    pre_study_ids: List[str],
    post_embeddings: torch.Tensor,
    post_study_ids: List[str],
    patient_mapping: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Compare PRE and POST PCI embeddings using cosine similarity."""

    # Create lookup dictionaries
    pre_lookup = {sid: emb for sid, emb in zip(pre_study_ids, pre_embeddings)}
    post_lookup = {sid: emb for sid, emb in zip(post_study_ids, post_embeddings)}

    results = []

    for patient_id, (pre_study, post_study) in tqdm(patient_mapping.items(), desc="Comparing embeddings"):
        if pre_study not in pre_lookup or post_study not in post_lookup:
            continue

        pre_vec = pre_lookup[pre_study]
        post_vec = post_lookup[post_study]

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            pre_vec.unsqueeze(0),
            post_vec.unsqueeze(0)
        ).item()

        # Compute L2 distance
        l2_dist = torch.norm(pre_vec - post_vec).item()

        # Compute normalized L2 (euclidean on unit vectors)
        pre_norm = F.normalize(pre_vec.unsqueeze(0), dim=1)
        post_norm = F.normalize(post_vec.unsqueeze(0), dim=1)
        normalized_l2 = torch.norm(pre_norm - post_norm).item()

        results.append({
            'patient_id_anon': patient_id,
            'pre_study_id': pre_study,
            'post_study_id': post_study,
            'cosine_similarity': cos_sim,
            'l2_distance': l2_dist,
            'normalized_l2_distance': normalized_l2,
            'embedding_change': 1 - cos_sim,
        })

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract and compare PRE vs POST PCI embeddings')
    parser.add_argument('--config', type=str,
                        default='config/inference/base_config_full_4828.yaml',
                        help='Path to PRE-PCI config YAML')
    parser.add_argument('--post_csv', type=str,
                        default='/volume/DeepCORO_CLIP_DATASET/deepcoro_POST_PCI_inference_test_set.csv',
                        help='Path to POST-PCI CSV')
    parser.add_argument('--output_dir', type=str, default='outputs/pci_comparison',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for extraction')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = LinearProbingConfig.from_yaml(args.config)
    if args.batch_size:
        config.batch_size = args.batch_size

    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    model, embedding_dim, head_input_dim = load_model(config, device)
    print(f"Embedding dim: {embedding_dim}, Head input dim: {head_input_dim}")

    # Create patient mapping between PRE and POST studies
    print(f"\n{'='*60}")
    print("Creating patient mapping...")
    print(f"{'='*60}")

    pre_df = pd.read_csv(config.data_filename, sep='α', engine='python')
    pre_df = pre_df[pre_df['Split'] == 'inference']

    post_df = pd.read_csv(args.post_csv, sep='α', engine='python')
    post_df = post_df[post_df['Split'] == 'inference']

    # Map patient -> (pre_study, post_study)
    pre_patient_study = pre_df.groupby('patient_id_anon')['StudyInstanceUID'].first().to_dict()
    post_patient_study = post_df.groupby('patient_id_anon')['StudyInstanceUID'].first().to_dict()

    patient_mapping = {}
    for patient in set(pre_patient_study.keys()) & set(post_patient_study.keys()):
        patient_mapping[patient] = (pre_patient_study[patient], post_patient_study[patient])

    print(f"Found {len(patient_mapping)} matched patient pairs")

    # Extract PRE-PCI embeddings
    print(f"\n{'='*60}")
    print("Extracting PRE-PCI embeddings...")
    print(f"{'='*60}")

    pre_dataloader, pre_dataset, pre_fname_to_study = create_dataloader(config.data_filename, config, 'inference')
    print(f"PRE dataset: {len(pre_dataset)} studies")

    pre_embeddings, pre_study_ids = extract_study_embeddings(
        model, pre_dataloader, pre_fname_to_study, device, head_input_dim
    )
    print(f"Extracted {len(pre_embeddings)} PRE embeddings")

    # Save PRE embeddings
    torch.save({
        'embeddings': pre_embeddings,
        'study_ids': pre_study_ids,
    }, os.path.join(args.output_dir, 'pre_pci_embeddings.pt'))

    # Extract POST-PCI embeddings
    print(f"\n{'='*60}")
    print("Extracting POST-PCI embeddings...")
    print(f"{'='*60}")

    # Create POST config (same model, different data)
    post_config = LinearProbingConfig.from_yaml(args.config)
    post_config.data_filename = args.post_csv
    post_config.batch_size = args.batch_size

    post_dataloader, post_dataset, post_fname_to_study = create_dataloader(args.post_csv, post_config, 'inference')
    print(f"POST dataset: {len(post_dataset)} studies")

    post_embeddings, post_study_ids = extract_study_embeddings(
        model, post_dataloader, post_fname_to_study, device, head_input_dim
    )
    print(f"Extracted {len(post_embeddings)} POST embeddings")

    # Save POST embeddings
    torch.save({
        'embeddings': post_embeddings,
        'study_ids': post_study_ids,
    }, os.path.join(args.output_dir, 'post_pci_embeddings.pt'))

    # Compare embeddings
    print(f"\n{'='*60}")
    print("Comparing PRE vs POST PCI embeddings...")
    print(f"{'='*60}")

    results_df = compare_embeddings(
        pre_embeddings, pre_study_ids,
        post_embeddings, post_study_ids,
        patient_mapping
    )

    # Add PCI status info
    pci_cols = [c for c in post_df.columns if 'pcidone' in c.lower()]
    study_pci_status = post_df.groupby('StudyInstanceUID')[pci_cols].max().fillna(0)
    study_pci_status['any_pci'] = (study_pci_status.sum(axis=1) > 0).astype(int)

    results_df['pci_performed'] = results_df['post_study_id'].map(
        study_pci_status['any_pci'].to_dict()
    ).fillna(0).astype(int)

    # Save results
    results_df.to_csv(os.path.join(args.output_dir, 'pci_embedding_comparison.csv'), index=False)

    # Print summary statistics
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total matched pairs: {len(results_df)}")
    print(f"Pairs with PCI performed: {results_df['pci_performed'].sum()}")
    print(f"Pairs without PCI: {(results_df['pci_performed'] == 0).sum()}")

    print(f"\n--- Cosine Similarity ---")
    print(f"Overall mean: {results_df['cosine_similarity'].mean():.4f} ± {results_df['cosine_similarity'].std():.4f}")

    pci_done = results_df[results_df['pci_performed'] == 1]
    no_pci = results_df[results_df['pci_performed'] == 0]

    if len(pci_done) > 0:
        print(f"WITH PCI:    {pci_done['cosine_similarity'].mean():.4f} ± {pci_done['cosine_similarity'].std():.4f} (n={len(pci_done)})")
    if len(no_pci) > 0:
        print(f"WITHOUT PCI: {no_pci['cosine_similarity'].mean():.4f} ± {no_pci['cosine_similarity'].std():.4f} (n={len(no_pci)})")

    print(f"\n--- Embedding Change (1 - cosine_sim) ---")
    print(f"Overall mean: {results_df['embedding_change'].mean():.4f} ± {results_df['embedding_change'].std():.4f}")

    if len(pci_done) > 0:
        print(f"WITH PCI:    {pci_done['embedding_change'].mean():.4f} ± {pci_done['embedding_change'].std():.4f}")
    if len(no_pci) > 0:
        print(f"WITHOUT PCI: {no_pci['embedding_change'].mean():.4f} ± {no_pci['embedding_change'].std():.4f}")

    # Statistical test
    if len(pci_done) > 10 and len(no_pci) > 10:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(pci_done['cosine_similarity'], no_pci['cosine_similarity'])
        print(f"\nT-test (PCI vs No PCI): t={t_stat:.3f}, p={p_value:.4e}")

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
