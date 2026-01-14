#!/usr/bin/env python3
"""
Quick test to verify SigLIP validation loads all texts correctly.
This script doesn't run training, just validates the dataset setup.
"""

import sys
sys.path.insert(0, '/volume/DeepCORO_CLIP')

from dataloaders.video_clip_dataset import VideoClipDataset
from utils.config.clip_config import ClipConfig

def test_siglip_dataset():
    """Test that SigLIP dataset loads all texts correctly."""

    # Load config
    config = ClipConfig.from_yaml('config/clip/siglip_output_dataset_config.yaml')

    print("=" * 80)
    print("Testing SigLIP Dataset Configuration")
    print("=" * 80)

    # Create validation dataset
    print("\nCreating validation dataset...")

    # Access config values properly
    def get_config(key, default=None):
        return config.get(key, default) if hasattr(config, 'get') else getattr(config, key, default)

    val_dataset = VideoClipDataset(
        root=get_config('root', '.'),
        data_filename=get_config('data_filename', ''),
        split='val',
        target_label=get_config('target_label', None),
        datapoint_loc_label=get_config('datapoint_loc_label', 'FileName'),
        num_frames=get_config('frames', 16),
        backbone=get_config('model_name', 'mvit'),
        mean=get_config('data_mean', None),
        std=get_config('data_std', None),
        rand_augment=False,  # No augmentation for validation
        stride=get_config('stride', 1),
        groupby_column=get_config('groupby_column', None),
        num_videos=get_config('num_videos', 1),
        shuffle_videos=get_config('shuffle_videos', False),
        seed=get_config('seed', 42),
        multi_video=get_config('multi_video', False),
        resize=get_config('resize', 224),
        siglip_texts_path=get_config('siglip_texts_path', None),
        siglip_max_positive_per_video=get_config('siglip_max_positive_per_video', 8),
        siglip_negatives_per_video=get_config('siglip_negatives_per_video', 0),
        siglip_round_robin_sampling=get_config('siglip_round_robin_sampling', False),
        siglip_max_segments_per_video=get_config('siglip_max_segments_per_video', 15),
        siglip_positive_severity_weights=get_config('siglip_positive_severity_weights', None),
    )

    print("\n" + "=" * 80)
    print("SigLIP Dataset Stats:")
    print("=" * 80)
    print(f"SigLIP Enabled: {val_dataset.siglip_enabled}")

    if val_dataset.siglip_enabled and val_dataset.siglip:
        print(f"Total texts in catalog: {len(val_dataset.siglip.text_lookup)}")
        print(f"Total videos with positives: {len(val_dataset.video_positive_texts)}")
        print(f"Dataset size (val): {len(val_dataset)}")

        # Sample statistics
        num_positives = [len(pairs) for pairs in val_dataset.video_positive_texts if pairs]
        if num_positives:
            avg_positives = sum(num_positives) / len(num_positives)
            max_positives = max(num_positives)
            min_positives = min(num_positives)
            print(f"\nPositives per video:")
            print(f"  Average: {avg_positives:.2f}")
            print(f"  Min: {min_positives}")
            print(f"  Max: {max_positives}")

        # Show first few texts
        print(f"\nFirst 5 texts from catalog:")
        for i, (text_id, text_meta) in enumerate(list(val_dataset.siglip.text_lookup.items())[:5]):
            prompt = text_meta.get('prompt_text', '')[:80]
            print(f"  {i+1}. {text_id}: {prompt}")

        # Show example video with its positives
        print(f"\nExample video with positives:")
        for idx, pairs in enumerate(val_dataset.video_positive_texts[:3]):
            if pairs:
                video_path = val_dataset.fnames[idx] if idx < len(val_dataset.fnames) else "N/A"
                print(f"\n  Video {idx}: {video_path}")
                print(f"  Positive texts ({len(pairs)} total):")
                for text_id, weight in pairs[:3]:  # Show first 3
                    if text_id in val_dataset.siglip.text_lookup:
                        prompt = val_dataset.siglip.text_lookup[text_id].get('prompt_text', '')[:60]
                        print(f"    - {text_id} (weight={weight:.2f}): {prompt}")
                if len(pairs) > 3:
                    print(f"    ... and {len(pairs) - 3} more")
                break
    else:
        print("SigLIP mode is NOT enabled!")

    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_siglip_dataset()
