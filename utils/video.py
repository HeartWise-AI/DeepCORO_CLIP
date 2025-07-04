import os
import cv2
import torch
import numpy as np

import torchvision.transforms as v2

from pathlib import Path
from typing import (
    Any,
    List,
    Optional,
    Union
)


def cleanup_temp_video(video_path):
    """Delete temporary video file if it exists."""
    try:
        path = Path(video_path)
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"Warning: Failed to delete temporary video {video_path}: {str(e)}")


def convert_video_for_wandb(video_path):
    """Convert video to MP4 format for wandb logging if needed.

    Args:
        video_path: Path to input video

    Returns:
        tuple: (output_path, is_temp) where is_temp indicates if the file needs cleanup
    """
    # If already MP4, return as is
    if video_path.lower().endswith(".mp4"):
        return video_path, False

    import subprocess
    import tempfile

    # Create temporary MP4 file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)

    try:
        # Convert to MP4 using ffmpeg
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-c:v", "libx264", "-preset", "fast", "-y", temp_path],
            check=True,
            capture_output=True,
        )
        return temp_path, True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to convert video {video_path}: {e.stderr.decode()}")
        os.unlink(temp_path)
        return video_path, False


def load_video(
    video_path: str,
    n_frames: int = 32,
    resize: int = 224,
    normalize: bool = True,
    mean: float = 0.0,
    std: float = 1.0,
    video_transforms: Optional[Any] = None,
    rand_augment: bool = False,
    backbone: str = "default",
    stride: int = 1,
) -> np.ndarray:
    """
    Load and process a video with optional resizing, normalization, and augmentations.
    Returns tensor in format [F, H, W, C].
    """

    # Model-specific parameters
    x3d_params = {
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": n_frames
        },
        "x3d_m": {
            "side_size": resize,
            "crop_size": resize,
            "num_frames": n_frames
        }
    }

    # --- Étape 1 : Lecture du fichier (npy ou vidéo classique) ---
    if video_path.endswith('.npy'):
        video = np.load(video_path)
        if not isinstance(video, np.ndarray):
            raise ValueError(f"Le fichier npy ne contient pas un tableau numpy: {video_path}")
        
        # Pre-validate .npy dimensions to prevent crashes
        if video.ndim < 3 or video.ndim > 4:
            raise ValueError(f"Invalid .npy file: Expected 3D or 4D array, got {video.ndim}D: {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Randomly choose stride between 1 and specified stride
        actual_stride = np.random.randint(1, stride + 1) if stride > 1 else 1

        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % actual_stride == 0:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1
        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from video: {video_path}")

        video = np.stack(frames, axis=0)
    
    video = video.astype(np.float32)
    if video.ndim == 3:
        video = np.expand_dims(video, axis=-1)  # [F,H,W] -> [F,H,W,1]
        video = np.repeat(video, 3, axis=-1)  # [F,H,W,1] -> [F,H,W,3]
    elif video.ndim == 4:
        # Handle 4D arrays with different channel configurations
        if video.shape[-1] == 1:
            # Grayscale [F,H,W,1] -> RGB [F,H,W,3]
            video = np.repeat(video, 3, axis=-1)
        elif video.shape[-1] != 3:
            raise ValueError(f"Invalid video shape after loading: Expected 1 or 3 channels, got {video.shape[-1]} channels: {video.shape}")
    else:
        raise ValueError(f"Invalid video shape after loading: {video.shape}")
        

    # --- Étape 2 : Pipeline commun pour tous les formats ---
    video = torch.from_numpy(video.astype(np.float32))

    # Permute to [F,C,H,W] si besoin
    if video.shape[-1] in [1, 3]:
        video = video.permute(0, 3, 1, 2)

    # Détermination du resize et expected_frames
    if backbone.lower() == "mvit":
        expected_frames = 16
        model_resize = resize
    elif backbone.lower() in ["x3d_s", "x3d_m"]:
        expected_frames = x3d_params[backbone.lower()]["num_frames"]
        model_resize = x3d_params[backbone.lower()]["side_size"]
    else:
        expected_frames = n_frames
        model_resize = resize

    t = video.shape[0]
    # Gestion du nombre de frames
    if t < expected_frames:
        last_frame = video[-1:].repeat(expected_frames - t, 1, 1, 1)
        video = torch.cat([video, last_frame], dim=0)
    elif t > expected_frames:
        indices = torch.linspace(0, t - 1, expected_frames).long()
        video = video[indices]

    # Resize spatial dimensions
    if model_resize is not None:
        video = v2.Resize((model_resize, model_resize), antialias=True)(video)

    # Optional transforms (assumes float input)
    if video_transforms is not None:
        transforms = v2.RandomApply(torch.nn.ModuleList(video_transforms), p=0.5)
        scripted_transforms = torch.jit.script(transforms)
        try:
            video = scripted_transforms(video)
        except RuntimeError as e:
            print(f"Warning: Error applying transforms to video {video_path}: {str(e)}")

    if rand_augment:
        # Convert video to uint8 for RandAugment
        if video.dtype != torch.uint8:
            video = video.to(torch.uint8)
        raug = [v2.RandAugment(magnitude=9, num_ops=2)]
        raug_composed = v2.Compose(raug)
        video = raug_composed(video)
        video = video.to(torch.float32) 

    if normalize:
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided for normalization.")
        t, c, h, w = video.shape
        if isinstance(mean, (int, float)):
            mean = [float(mean)] * c
        if isinstance(std, (int, float)):
            std = [float(std)] * c
        mean = mean[:c]
        std = std[:c]
        video = v2.Normalize(mean=mean, std=std)(video)

    # Final checks
    t, c, h, w = video.shape
    if t != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {t}")
    if h != model_resize or w != model_resize:
        raise ValueError(f"Expected spatial dimensions {model_resize}x{model_resize}, got {h}x{w}")

    # Return video in shape [F, H, W, C]
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
    
