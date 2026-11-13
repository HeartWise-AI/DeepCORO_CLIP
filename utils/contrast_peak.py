"""Contrast-peak frame sampling for coronary angiography cines.

Port of the ``peak_s2`` sampler used to train and evaluate DeepCORO-JEPA v9-v14
(DeepCoro ``experiments/step3/integrated_data.py``: ``dye_score`` / ``clip_indices``).
The numbers and edge handling are kept identical so the frame indices match exactly:

* each frame is read with OpenCV and converted BGR -> grayscale (uint8);
* dye score = mean black top-hat (15x15 ellipse) of the frame resized to 256x256
  (INTER_AREA), over the central field [25:231, 25:231]; dark thin structures
  (contrast-filled vessels) raise it;
* the per-frame scores are smoothed with a 5-frame box filter
  (``np.convolve(..., mode="same")``) and the peak is the argmax (first one on ties);
* the clip is 16 frames at stride 2 (span 30 frames) centred on the peak, shifted to
  stay inside the video when the peak is near the start or the end;
* videos shorter than ``stride * count`` (32) frames fall back to 16 frames spread
  uniformly over the whole video (``rint(linspace(0, n - 1, 16))``).

Only numpy and OpenCV are needed. Selected in the dataloaders with ``frame_sampling: peak``
(``peak_stride``, default 2); the same module lives in DeepCORO_Pre_Processing
(``deepcoro/video/contrast_peak.py``) with a script that writes the per-video peak to a parquet.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

COUNT = 16
STRIDE = 2
SMOOTH = 5
_RESIZE = 256
_CROP = slice(25, 231)
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def dye_score(gray: np.ndarray) -> float:
    """Opacification score of one grayscale uint8 frame (higher = more dye in the vessels)."""
    g = cv2.resize(gray, (_RESIZE, _RESIZE), interpolation=cv2.INTER_AREA)
    return float(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, _KERNEL)[_CROP, _CROP].mean())


def dye_scores(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Dye score of every frame."""
    return np.array([dye_score(f) for f in frames], dtype=np.float64)


def smooth_scores(scores: Sequence[float], width: int = SMOOTH) -> np.ndarray:
    """Box-filter smoothing with the same length as the input (zero padded at both ends)."""
    return np.convolve(np.asarray(scores, dtype=np.float64), np.ones(width) / width, mode="same")


def peak_frame(scores: Sequence[float], width: int = SMOOTH) -> int:
    """Index (in the original video) of the smoothed dye-score maximum."""
    if len(scores) < 1:
        raise ValueError("Empty video")
    return int(np.argmax(smooth_scores(scores, width)))


def uniform_indices(n: int, count: int = COUNT) -> np.ndarray:
    """``count`` frames spread over the whole video (fallback for short videos)."""
    if n < 1:
        raise ValueError("Empty video")
    return np.rint(np.linspace(0, n - 1, count)).astype(np.int64)


def window_indices(n: int, centre: float, count: int = COUNT, stride: int = STRIDE) -> np.ndarray:
    """``count`` frames at ``stride`` centred on ``centre``, kept inside [0, n - 1].

    Videos shorter than ``stride * count`` frames return :func:`uniform_indices` instead.
    """
    if n < stride * count:
        return uniform_indices(n, count)
    span = stride * (count - 1)
    start = int(np.clip(round(centre - span / 2), 0, n - 1 - span))
    return start + stride * np.arange(count, dtype=np.int64)


def peak_indices(scores: Sequence[float], count: int = COUNT, stride: int = STRIDE) -> np.ndarray:
    """Frame indices of the peak window from per-frame dye scores (``peak_s2`` with defaults)."""
    return window_indices(len(scores), peak_frame(scores), count, stride)


def clip_indices(
    frames: Sequence[np.ndarray], sampling: str = "peak_s2", count: int = COUNT
) -> np.ndarray:
    """As DeepCoro ``clip_indices``: ``uniform16``, ``peak_s2`` or ``middle_s2``."""
    n = len(frames)
    if sampling == "uniform16" or n < STRIDE * count:
        return uniform_indices(n, count)
    if sampling == "peak_s2":
        return window_indices(n, peak_frame(dye_scores(frames)), count, STRIDE)
    if sampling == "middle_s2":
        return window_indices(n, (n - 1) / 2, count, STRIDE)
    raise ValueError(f"Unknown frame sampling {sampling}")


def gray_for_dye_score(frame: np.ndarray, bgr: bool = True) -> np.ndarray:
    """Grayscale image used for the dye score: OpenCV BGR->GRAY for decoded video frames
    (as in training), RGB->GRAY for 3-channel arrays loaded from .npy, as-is for 1 channel."""
    if frame.dtype != np.uint8:
        frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return frame[..., 0]
    return frame
