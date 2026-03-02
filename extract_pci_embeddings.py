"""
Extract study-level embeddings from pre and post PCI angiography videos.

This script:
1. Loads the trained DeepCORO model
2. Runs inference on pre-PCI and post-PCI videos
3. Extracts CLS token (study-level) embeddings
4. Saves embeddings for downstream analysis
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Add DeepCORO_CLIP to path
sys.path.insert(0, '/volume/DeepCORO_CLIP')

from models.video_encoder import VideoEncoder
from models.multi_instance_linear_probing import MultiInstanceLinearProbing
from projects.linear_probing_project import VideoMILWrapper
from dataloaders.video_dataset import get_distributed_video_dataloader
from utils.config.linear_probing_config import LinearProbingConfig


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> VideoMILWrapper:
    """Load the VideoMILWrapper model from checkpoint."""
    config = LinearProbingConfig.from_yaml(config_path)
    
    # Build video encoder
    video_encoder = VideoEncoder(
        model_name=config.model_name,
        pretrained=config.pretrained,
        freeze_ratio=config.video_freeze_ratio,
        num_frames=config.frames,
        num_heads=config.num_heads,
        aggregator_depth=config.aggregator_depth,
        aggregate_videos_tokens=config.aggregate_videos_tokens,
        per_video_pool=config.per_video_pool,
    )
    
    # Load encoder checkpoint
    if config.video_encoder_checkpoint_path:
        ckpt = torch.load(config.video_encoder_checkpoint_path, map_location=device)
        video_encoder.load_state_dict(ckpt.get('video_encoder', ckpt), strict=False)
    
    # Build MIL model
    embedding_dim = video_encoder.embedding_dim
    if '+' in config.pooling_mode:
        head_input_dim = 2 * embedding_dim
    else:
        head_input_dim = embedding_dim
        
    mil_model = MultiInstanceLinearProbing(
        embedding_dim=embedding_dim,
        head_structure=config.head_structure or {'dummy': 1},
        pooling_mode=config.pooling_mode,
        use_cls_token=config.use_cls_token,
        num_attention_heads=config.num_attention_heads,
        separate_video_attention=config.separate_video_attention,
        normalization_strategy=config.normalization_strategy,
    )
    
    # Wrap
    wrapper = VideoMILWrapper(video_encoder, mil_model, num_videos=config.num_videos)
    wrapper.to(device)
    wrapper.eval()
    
    return wrapper, config


def extract_embeddings_hook(model: VideoMILWrapper) -> Dict[str, torch.Tensor]:
    """Extract pooled embeddings by hooking into the MIL model."""
    embeddings = {}
    
    def hook_fn(module, input, output):
        # For the _pool_instances method, capture the pooled output
        if hasattr(module, '_pool_instances'):
            pass  # We'll get this from the forward pass
    
    return embeddings


def extract_study_embeddings(
    model: VideoMILWrapper,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
) -> pd.DataFrame:
    """Extract study-level embeddings for all studies in dataloader."""
    
    all_study_ids = []
    all_embeddings = []
    
    model.eval()
    
    # Hook into the MIL model to capture pooled features
    pooled_features = []
    
    def capture_pooled(module, input, output):
        # Capture the input to the heads (which is the pooled embedding)
        if isinstance(input, tuple):
            pooled_features.append(input[0].detach().cpu())
        else:
            pooled_features.append(input.detach().cpu())
    
    # Register hook on the first head to capture input
    head_name = list(model.mil_model.heads.keys())[0]
    hook_handle = model.mil_model.heads[head_name].register_forward_hook(capture_pooled)
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                videos = batch['videos'].to(device)
                study_ids = batch.get('study_ids', batch.get('filenames', []))
                
                # Clear previous features
                pooled_features.clear()
                
                # Forward pass - this triggers the hook
                _ = model(videos)
                
                # Get the captured pooled features
                if pooled_features:
                    batch_embeddings = pooled_features[0]  # [B, D] or [B, 2*D] for hybrid
                    all_embeddings.append(batch_embeddings)
                    all_study_ids.extend(study_ids)
    finally:
        hook_handle.remove()
    
    # Concatenate all embeddings
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    # Save as torch file
    torch.save({
        'study_ids': all_study_ids,
        'embeddings': embeddings_tensor,
    }, save_path)
    
    print(f"Saved {len(all_study_ids)} study embeddings to {save_path}")
    print(f"Embedding dimension: {embeddings_tensor.shape[1]}")
    
    return pd.DataFrame({
        'StudyInstanceUID': all_study_ids,
        'embedding_file': save_path,
    })


def compare_pre_post_embeddings(
    pre_embeddings_path: str,
    post_embeddings_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Compare pre and post PCI embeddings using cosine similarity."""
    
    # Load embeddings
    pre_data = torch.load(pre_embeddings_path)
    post_data = torch.load(post_embeddings_path)
    
    pre_ids = pre_data['study_ids']
    pre_emb = pre_data['embeddings']
    
    post_ids = post_data['study_ids']
    post_emb = post_data['embeddings']
    
    # Create lookup for post embeddings
    post_lookup = {sid: emb for sid, emb in zip(post_ids, post_emb)}
    
    results = []
    for i, pre_id in enumerate(pre_ids):
        # Find matching post-PCI study (need study pairing logic)
        # For now, assume same StudyInstanceUID or linked via MRN
        if pre_id in post_lookup:
            pre_vec = pre_emb[i]
            post_vec = post_lookup[pre_id]
            
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                pre_vec.unsqueeze(0), 
                post_vec.unsqueeze(0)
            ).item()
            
            # Compute L2 distance
            l2_dist = torch.norm(pre_vec - post_vec).item()
            
            results.append({
                'StudyInstanceUID': pre_id,
                'cosine_similarity': cos_sim,
                'l2_distance': l2_dist,
                'embedding_change': 1 - cos_sim,  # Higher = more change
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison results to {output_path}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pre', 'post', 'compare'], required=True)
    parser.add_argument('--config', type=str, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--pre_embeddings', type=str, help='Pre-PCI embeddings for compare mode')
    parser.add_argument('--post_embeddings', type=str, help='Post-PCI embeddings for compare mode')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.mode in ['pre', 'post']:
        model, config = load_model(args.config, args.checkpoint, device)
        
        # Create dataloader based on config
        dataloader = get_distributed_video_dataloader(
            config=config,
            split='inference',
            is_training=False,
        )
        
        extract_study_embeddings(model, dataloader, device, args.output)
        
    elif args.mode == 'compare':
        compare_pre_post_embeddings(
            args.pre_embeddings,
            args.post_embeddings,
            args.output,
        )


if __name__ == '__main__':
    main()
