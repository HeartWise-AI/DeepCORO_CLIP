"""
Example script demonstrating how to use X3D_S and X3D_M models in DeepCORO_CLIP.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from utils.config.clip_config import ClipConfig
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from dataloaders.video_dataset import VideoDataset
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="X3D Models Example")
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["mvit", "r3d", "x3d_s", "x3d_m"], 
        default="x3d_s", 
        help="Model backbone type"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/", 
        help="Path to data directory"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Print model information based on the selected backbone
    print(f"Using {args.model_type} backbone")
    
    if args.model_type == "mvit":
        print("MViT settings:")
        print("- Frame count: 16")
        print("- Image size: 224x224")
    elif args.model_type == "x3d_s":
        print("X3D_S settings:")
        print("- Frame count: 13")
        print("- Image size: 182x182")
    elif args.model_type == "x3d_m":
        print("X3D_M settings:")
        print("- Frame count: 16")
        print("- Image size: 256x256")
    elif args.model_type == "r3d":
        print("R3D settings:")
        print("- Frame count: 16 (configurable)")
        print("- Image size: 224x224 (configurable)")
    
    # Initialize video encoder with specified backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create video encoder
    video_encoder = VideoEncoder(
        backbone=args.model_type,
        pretrained=True,
        output_dim=512,
        num_heads=4,
        freeze_ratio=0.8,
    ).to(device)
    
    # Create text encoder
    text_encoder = TextEncoder(
        model_name="distilbert-base-uncased",
        output_dim=512,
        freeze_backbone=True,
    ).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in video_encoder.parameters())
    trainable_params = sum(p.numel() for p in video_encoder.parameters() if p.requires_grad)
    print(f"Video encoder parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    total_params = sum(p.numel() for p in text_encoder.parameters())
    trainable_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"Text encoder parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    print("\nModel implementation complete.")
    print(f"To train with {args.model_type}, use the following in your config:")
    print(f'model_name: !!str "{args.model_type}"')
    
    print("\nTo run a hyperparameter sweep including different models:")
    print("parameters:")
    print("  model_name:")
    print('    values: ["mvit", "x3d_s", "x3d_m"]')


if __name__ == "__main__":
    main() 