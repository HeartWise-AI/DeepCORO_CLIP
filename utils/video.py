import cv2
import torch
import numpy as np
import torchvision.transforms as v2

from pathlib import Path
from typing import Union, List


def cleanup_temp_video(video_path):
    """Delete temporary video file if it exists."""
    try:
        path = Path(video_path)
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"Warning: Failed to delete temporary video {video_path}: {str(e)}")


def load_video(
    video_path,
    split="train",
    n_frames=32,
    stride=1,
    resize=224,
    apply_mask=False,
    normalize=True,
    mean=0.0,
    std=1.0,
    noise=None,
    pad=None,
    video_transforms=None,
    rand_augment=False,
    backbone="default",
):
    """
    Load and process a video with optional resizing, normalization, and augmentations.
    Returns tensor in format [F, H, W, C].
    """
    # Force 16 frames for MViT backbone
    if backbone.lower() == "mvit":
        n_frames = 16
        stride = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {video_path}")

    video = np.stack(frames, axis=0).astype(np.float32)

    # Ensure video shape is [F, H, W, C]
    if video.ndim == 3:
        video = np.expand_dims(video, axis=-1)  # [F,H,W] -> [F,H,W,C]
    elif video.ndim != 4:
        raise ValueError(f"Invalid video shape after loading: {video.shape}")

    # Convert to torch tensor [F,H,W,C]
    video = torch.from_numpy(video)

    # Permute to [F,C,H,W]
    if video.shape[-1] in [1, 3]:
        video = video.permute(0, 3, 1, 2)

    # Resize spatial dimensions
    if resize is not None:
        video = v2.Resize((resize, resize), antialias=True)(video)

    t, c, h, w = video.shape

    # Frame sampling
    if backbone.lower() == "mvit":
        # Exactly 16 frames
        if t < 16:
            last_frame = video[-1:].repeat(16 - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > 16:
            indices = torch.linspace(0, t - 1, 16).long()
            video = video[indices]
    else:
        # Keep original frame count to n_frames
        if t < n_frames:
            last_frame = video[-1:].repeat(n_frames - t, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        elif t > n_frames:
            indices = torch.linspace(0, t - 1, n_frames).long()
            video = video[indices]

    if normalize:
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided for normalization.")
        if isinstance(mean, (int, float)):
            mean = [float(mean)] * c
        if isinstance(std, (int, float)):
            std = [float(std)] * c
        mean = mean[:c]
        std = std[:c]
        video = v2.Normalize(mean=mean, std=std)(video)

    # Optional transforms
    if video_transforms is not None:
        transforms = v2.RandomApply(torch.nn.ModuleList(video_transforms), p=0.5)
        scripted_transforms = torch.jit.script(transforms)
        try:
            video = scripted_transforms(video)
        except RuntimeError as e:
            print(f"Warning: Error applying transforms to video {video_path}: {str(e)}")

    if rand_augment:
        raug = [v2.RandAugment(magnitude=9, num_ops=2)]
        raug_composed = v2.Compose(raug)
        video = raug_composed(video)

    # Final checks
    t, c, h, w = video.shape
    expected_frames = 16 if backbone.lower() == "mvit" else n_frames
    if t != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {t}")
    if h != resize or w != resize:
        raise ValueError(f"Expected spatial dimensions {resize}x{resize}, got {h}x{w}")

    # Return [F,H,W,C]
    video = video.permute(0, 2, 3, 1).contiguous()
    return video.numpy()


def format_mean_std(input_value: Union[float, List[float], np.ndarray]) -> List[float]:
    """Format mean/std input to list of floats.

    Args:
        input_value: Input value to format

    Returns:
        List of 3 float values

    Raises:
        ValueError: If input cannot be converted to list of floats
        TypeError: If input type is not supported
    """
    if isinstance(input_value, (int, float)):
        return [float(input_value)] * 3
    elif isinstance(input_value, str):
        try:
            cleaned_input = input_value.strip("[]").split()
            formatted_value = [float(val) for val in cleaned_input]
            return formatted_value
        except ValueError as err:
            raise ValueError(
                "String input for mean/std must be space-separated numbers."
            ) from err
    elif isinstance(input_value, (list, np.ndarray)):
        try:
            formatted_value = [float(val) for val in input_value]
            return formatted_value
        except ValueError as err:
            raise ValueError("List or array input for mean/std must contain numbers.") from err
    else:
        raise TypeError("Input for mean/std must be a string, list, or numpy array.")
    
