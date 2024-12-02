import os
# Set tokenizer parallelism before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import os
from tqdm import tqdm
import argparse

from train_model import VideoEncoder, TextEncoder, VideoDataset, contrastive_loss, train_epoch

def parse_args():
    parser = argparse.ArgumentParser(description='Train CLIP model')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=24, help='base batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--debug', action='store_true', help='enable CUDA debug mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Enable CUDA debugging if requested
    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.backends.cudnn.enabled = False
    
    # Setup wandb
    wandb.init(project="deepcoro_clip")
    
    # Setup multi-GPU training
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    # Print CUDA memory info
    for i in range(world_size):
        print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB")
        print(f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f}GB")

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Initialize models
        video_encoder = VideoEncoder().to(device)
        text_encoder = TextEncoder().to(device)

        # Wrap models with DataParallel if using multiple GPUs
        if world_size > 1:
            print("Using DataParallel for multi-GPU training")
            video_encoder = nn.DataParallel(video_encoder)
            text_encoder = nn.DataParallel(text_encoder)

        # Create dataset
        train_dataset = VideoDataset(
            root="data/processed/reports",
            data_filename="reports_sampled_1000.csv",
            split="train",
            target_label="Report",
            datapoint_loc_label="FileName",
            mean=[107.56801, 107.56801, 107.56801],
            std=[40.988625, 40.988625, 40.988625],
        )

        # Adjust batch size based on available GPU memory
        try:
            batch_size = args.batch_size * world_size
            # Create a small test batch to check memory
            test_batch = next(iter(DataLoader(train_dataset, batch_size=batch_size)))
            del test_batch
        except RuntimeError as e:
            print("Warning: Reducing batch size due to GPU memory constraints")
            batch_size = args.batch_size  # Use base batch size without scaling
        
        print(f"Using batch size: {batch_size}")

        # Create dataloader with adjusted batch size
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(6 * world_size, os.cpu_count() or 1),  # Limit workers
            pin_memory=True,
            drop_last=True,  # Prevent issues with incomplete batches
            persistent_workers=True,  # Keep workers alive between iterations
            prefetch_factor=2  # Reduce memory pressure
        )

        # Training parameters
        num_epochs = args.epochs
        learning_rate = args.lr

        # Optimizer
        params = list(video_encoder.parameters()) + list(text_encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=learning_rate)

        # Create checkpoint directory
        checkpoint_dir = Path('models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.resume:
            if os.path.isfile(args.resume):
                print(f"Loading checkpoint '{args.resume}'")
                epoch_start, best_loss = load_checkpoint(args.resume, video_encoder, 
                                                       text_encoder, optimizer, world_size)
                print(f"Resumed from epoch {epoch_start}")
            else:
                print(f"No checkpoint found at '{args.resume}'")
                
        # Training loop
        best_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            try:
                train_loss = train_epoch(video_encoder, text_encoder, train_dataloader, 
                                       optimizer, device, rank=0)
                
                print(f"Training loss: {train_loss:.4f}")
                
                # Log to wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'gpu_memory': torch.cuda.memory_allocated(0) / 1e9  # Log GPU memory usage
                })
                
                # Save checkpoint
                is_best = train_loss < best_loss
                best_loss = min(train_loss, best_loss)
                
                checkpoint = {
                    'video_encoder': video_encoder.module.state_dict() if world_size > 1 else video_encoder.state_dict(),
                    'text_encoder': text_encoder.module.state_dict() if world_size > 1 else text_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': train_loss,
                    'best_loss': best_loss,
                    'batch_size': batch_size
                }
                
                # Save regular checkpoint
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint for epoch {epoch+1} at {checkpoint_path}")
                
                # Save best model
                if is_best:
                    best_model_path = checkpoint_dir / 'best_model.pt'
                    torch.save(checkpoint, best_model_path)
                    print(f"Saved new best model with loss: {train_loss:.4f}")
                    
            except RuntimeError as e:
                print(f"Error during training: {str(e)}")
                print("Attempting to recover...")
                torch.cuda.empty_cache()
                continue
                
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise e
    finally:
        # Cleanup
        torch.cuda.empty_cache()

def load_checkpoint(checkpoint_path, video_encoder, text_encoder, optimizer, world_size):
    checkpoint = torch.load(checkpoint_path)
    
    if world_size > 1 and not isinstance(video_encoder, nn.DataParallel):
        video_encoder = nn.DataParallel(video_encoder)
        text_encoder = nn.DataParallel(text_encoder)
    
    video_encoder.load_state_dict(checkpoint['video_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch'], checkpoint.get('best_loss', float('inf'))

if __name__ == '__main__':
    main() 