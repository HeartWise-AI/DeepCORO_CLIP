import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision.transforms import v2
import sys

# Add Orion to path
dir2 = os.path.abspath("/volume/Orion/orion")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

import orion

def get_video_path(split, fname, base_dir, external_test_location=None):
    """Determine video path based on split."""
    if split == "external_test":
        return os.path.join(external_test_location, fname)
    elif split == "clinical_test":
        return os.path.join(base_dir, "ProcessedStrainStudyA4c", fname)
    else:
        return fname

def load_video(video_path, split="train", n_frames=32, period=1, resize=224, 
               apply_mask=False, normalize=True, mean=0.0, std=1.0, 
               noise=None, pad=None, video_transforms=None, rand_augment=False):
    """
    Load and process a video with center cropping and optional augmentations.
    
    Args:
        video_path (str): Path to the video file
        split (str): Dataset split ('train', 'val', 'test', 'external_test', 'clinical_test')
        n_frames (int): Number of frames to extract
        period (int): Sampling period for frames
        resize (int): Size to resize frames to
        apply_mask (bool): Whether to apply masking
        normalize (bool): Whether to normalize the video
        mean (float): Mean for normalization
        std (float): Standard deviation for normalization
        noise (float): Fraction of pixels to black out
        pad (int): Padding size
        video_transforms (list): List of video transforms
        rand_augment (bool): Whether to apply random augmentation
    
    Returns:
        np.ndarray: Processed video array
    """
    # Load video using Orion's utility
    video = orion.utils.loadvideo(video_path).astype(np.float32)
    
    # Handle masking if required
    if apply_mask:
        path = video_path.rsplit("/", 2)
        mask_filename = f"{path[0]}/mask/{path[2]}"
        mask_filename = mask_filename.split(".avi")[0] + ".npy"
        
        if os.path.exists(mask_filename):
            mask = np.load(mask_filename).transpose(2, 0, 1)
            length = video.shape[2]
            
            # Fix mask shapes
            if mask.shape[1] < length:
                mask = np.pad(mask, [(0, 0), (length - mask.shape[1], 0), (0, 0)])
            if mask.shape[2] < length:
                mask = np.pad(mask, [(0, 0), (0, 0), (length - mask.shape[2], 0)])
            if mask.shape[1] > length:
                mask = mask[:, :length, :]
            if mask.shape[2] > length:
                mask = mask[:, :, :length]
            
            # Apply mask to each frame
            for ind in range(video.shape[0]):
                video[ind, :, :, :] = video[ind, :, :, :] * mask
    
    # Add noise if specified
    if noise is not None:
        n = video.shape[1] * video.shape[2] * video.shape[3]
        ind = np.random.choice(n, round(noise * n), replace=False)
        f = ind % video.shape[1]
        ind //= video.shape[1]
        i = ind % video.shape[2]
        ind //= video.shape[2]
        j = ind
        video[:, f, i, j] = 0
    
    # Convert to torch tensor for transforms
    video = torch.from_numpy(video)
    
    # Resize if specified
    if resize is not None:
        video = v2.Resize((resize, resize), antialias=True)(video)
    
    # Apply normalization if specified
    if normalize:
        video = v2.Normalize(mean, std)(video)
    
    # Apply video transforms if specified
    if video_transforms is not None:
        transforms = v2.RandomApply(torch.nn.ModuleList(video_transforms), p=0.5)
        scripted_transforms = torch.jit.script(transforms)
        try:
            video = scripted_transforms(video)
        except RuntimeError as e:
            print(f"Error applying transforms to video {video_path}: {str(e)}")
    
    # Apply random augmentation if specified
    if rand_augment:
        raug = [v2.RandAugment(magnitude=9, num_layers=2, prob=0.5)]
        raug_composed = v2.Compose(raug)
        video = raug_composed(video)
    
    # Convert back to numpy and handle frame ordering
    video = video.permute(1, 0, 2, 3).numpy()
    
    # Center crop/pad frames
    c, f, h, w = video.shape
    target_frames = n_frames * period
    
    if f < target_frames:
        # Pad if video is too short
        padding = target_frames - f
        pad_left = padding // 2
        pad_right = padding - pad_left
        video = np.pad(video, ((0,0), (pad_left,pad_right), (0,0), (0,0)), mode='constant')
    else:
        # Center crop
        start_frame = (f - target_frames) // 2
        video = video[:, start_frame:start_frame + target_frames:period, :, :]
    
    # Add padding if specified
    if pad is not None:
        c, l, h, w = video.shape
        temp = np.zeros((c, l, h + 2 * pad, w + 2 * pad), dtype=video.dtype)
        temp[:, :, pad:-pad, pad:-pad] = video
        i, j = np.random.randint(0, 2 * pad, 2)
        video = temp[:, :, i:(i + h), j:(j + w)]
    
    return video

# Example usage:
# video = load_video(video_path, split='train', n_frames=32, period=1, 
#                   resize=224, apply_mask=True, normalize=True)

def process_report(report_path, output_path):
    """Process and tokenize text reports."""
    with open(report_path, 'r') as f:
        report = f.read().strip()
    
    # Basic preprocessing
    report = report.lower()
    report = ' '.join(report.split())
    
    with open(output_path, 'w') as f:
        f.write(report)

def main():
    # Create necessary directories
    base_dir = Path('data')
    for dir_path in ['processed/videos', 'processed/reports']:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
if __name__ == '__main__':
    main() 