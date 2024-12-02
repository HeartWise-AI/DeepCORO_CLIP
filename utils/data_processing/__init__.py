"""Utility functions for videos, plotting and computing performance metrics."""

import os
import typing

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
import zarr


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def loadvideo(filename: str) -> np.ndarray:
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Loads a video from a file
    ---------------------------------------------------------------------------------
    Args:
        * filename
            - str: filename of video
    ---------------------------------------------------------------------------------
    Returns:
        * np.ndarray: with dimensions (channels=3, frames, height, width). The
            values will be uint8's ranging from 0 to 255.
    ---------------------------------------------------------------------------------
    Sanity Checks / Bug fixes:
        * Raises
            FileNotFoundError: Could not find `filename`
            ValueError: An error occurred while reading the video
    ---------------------------------------------------------------------------------
    Called By:
        * Files
            orion.datasets.echo
            orion.datasets.echo_inf
    ---------------------------------------------------------------------------------
    TODO:
        * None
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    file_extension = os.path.splitext(filename)[1]

    if file_extension in [".mp4", ".avi"]:
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_count, frame_width, frame_height, 3), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError(f"Failed to load frame #{count} of {filename}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            v[count] = frame

        vid = v.transpose((0, 3, 1, 2))  ##

    elif file_extension == ".zarr":
        data = zarr.open(filename, mode="r", synchronizer=zarr.ThreadSynchronizer())
        vid = np.array(data.video)  # (64, 128, 128)
        vid = np.transpose(vid, (1, 2, 0))  # Change shape to (128, 128, 64)
        vid = np.stack((vid,) * 3, axis=-1)  # Change shape to (128, 128, 64, 3)
        # Reshape the video to (64, 3, 128, 128)
        vid = np.transpose(vid, (2, 3, 0, 1))  # Change shape to (64, 3, 128, 128)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return vid




def get_mean_and_std(
    dataset: torch.utils.data.Dataset,
    samples: int = 128,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Computes mean and std from samples from a Pytorch dataset
    ---------------------------------------------------------------------------------
    Args:
        * dataset
            - torch.utils.data.Dataset: A Pytorch dataset ``dataset[i][0]'' is expected
                    to be the i-th video in the dataset, which should be a ``torch.Tensor''
                    of dimensions (channels=3, frames, height, width)
        * samples
            - int or None, optional (128): Number of samples to take from dataset. If
                    ``None'', mean and standard deviation are computed over all elements
        * batch_size
            - int, optional (8): frames per second
        * num_workers
            - int, optional (4): how many subprocesses to use for data loading.
                    If 0, the data will be loaded in the main process.
    ---------------------------------------------------------------------------------
    Returns:
        * A tuple of the mean and standard deviation. Both are represented as
            np.array's of dimension (channels,)
    ---------------------------------------------------------------------------------
    Sanity Checks / Bug fixes:
        * None
    ---------------------------------------------------------------------------------
    Called By:
        * Files
            orion.utils.video_class
            orion.datasets.echo for normalization
            orion.datasets.echo_inf for normalization
    ---------------------------------------------------------------------------------
    TODO:
        * None
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.0  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.0  # sum of squares of elemenzts along channels (ends up as np.array of dimension (channels,))
    # pdb.set_trace()
    for x, *_ in tqdm.tqdm(dataloader):
        # pdb.set_trace()
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x**2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean**2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


def multi_get_mean_and_std(
    dataset: torch.utils.data.Dataset,
    samples: int = 128,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Computes mean and std by view from samples from a Pytorch dataset by column
            of filenames
    ---------------------------------------------------------------------------------
    Args:
        * dataset
            - torch.utils.data.Dataset: A Pytorch dataset ``dataset[i][0]'' is expected
                    to be the i-th video in the dataset, which should be a ``torch.Tensor''
                    of dimensions (channels=3, frames, height, width)
        * samples
            - int or None, optional (128): Number of samples to take from dataset. If
                    ``None'', mean and standard deviation are computed over all elements
        * batch_size
            - int, optional (8): frames per second
        * num_workers
            - int, optional (4): how many subprocesses to use for data loading.
                    If 0, the data will be loaded in the main process.
    ---------------------------------------------------------------------------------
    Returns:
        * A tuple of the mean and standard deviation. Both are represented as
            np.array's of dimension (channels,)
    ---------------------------------------------------------------------------------
    Sanity Checks / Bug fixes:
        * None
    ---------------------------------------------------------------------------------
    Called By:
        * Files
            orion.utils.video_class
            orion.datasets.echo for normalization
            orion.datasets.echo_inf for normalization
    ---------------------------------------------------------------------------------
    TODO:
        * None
    """
    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.0  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.0  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))

    for x, *_ in tqdm.tqdm(dataloader):
        x = torch.stack(x)
        x = x.transpose(1, 2)
        x = x.contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x**2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean**2 + 1e-8)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


__all__ = [
    'RMSELoss',
    'loadvideo',
    'get_mean_and_std',
    'multi_get_mean_and_std',
    'Video',
    'Video_Multi'
]
