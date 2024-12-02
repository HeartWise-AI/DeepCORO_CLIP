"""Data processing utilities for DeepCORO_CLIP."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision.transforms import v2

from utils.data_processing.video import Video, load_video

__all__ = ["Video", "get_mean_and_std", "format_mean_std", "load_video"]


def get_mean_and_std(
    dataset: Video, samples: Optional[int] = None, batch_size: int = 8, num_workers: int = 4
) -> Tuple[List[float], List[float]]:
    """Calculate mean and standard deviation of a video dataset.

    Args:
        dataset: Video dataset to calculate statistics for
        samples: Number of samples to use (None for all)
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (mean, std) lists with channel-wise statistics
    """
    if samples is not None:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    for batch, _, _ in dataloader:
        batch = torch.from_numpy(batch)
        batch_samples = batch.size(0)
        batch = batch.view(batch_samples, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples

    return mean.tolist(), std.tolist()


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
