"""Tests for the contrast-peak frame sampler (frame_sampling: peak) in utils/video.load_video."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from utils import contrast_peak as cp
from utils.video import load_video

VIDEO_PARQUET = Path(
    "/media/data1/datasets/DeepCoro/"
    "2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL_with_status_and_split.parquet"
)

# ---- reference: verbatim copy of DeepCoro experiments/step3/integrated_data.py (peak_s2) ----
_REF_TOPHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def _ref_dye_score(gray):
    g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    return float(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, _REF_TOPHAT)[25:231, 25:231].mean())


def _ref_peak_s2(frames, count=16):
    n = len(frames)
    if n < 2 * count:
        return np.rint(np.linspace(0, n - 1, count)).astype(np.int64)
    s = np.array([_ref_dye_score(f) for f in frames])
    centre = int(np.argmax(np.convolve(s, np.ones(5) / 5, mode="same")))
    span = 2 * (count - 1)
    start = int(np.clip(round(centre - span / 2), 0, n - 1 - span))
    return start + 2 * np.arange(count, dtype=np.int64)


def bolus_frames(n, peak, size=64, sigma=3.0):
    """Bright field, thin dark 'vessels' whose contrast peaks at frame `peak`.

    The frame index is written in pixel (0, 0)."""
    vessels = np.zeros((size, size), np.uint8)
    for k in range(3):
        cv2.line(vessels, (8 + 16 * k, 8), (size - 8 - 8 * k, size - 8), 255, 1)
    mask = vessels > 0
    frames = []
    for t in range(n):
        g = np.full((size, size), 170, np.uint8)
        g[mask] = 170 - int(round(140 * np.exp(-0.5 * ((t - peak) / sigma) ** 2)))
        g[0, 0] = t  # frame tag, too small to move the dye score peak
        frames.append(g)
    return frames


def write_video(path, frames, fps=15.0):
    h, w = frames[0].shape
    for codec in ("FFV1", "MJPG"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h), True)
        if wr.isOpened():
            for f in frames:
                wr.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
            wr.release()
            return str(path)
    pytest.skip("No usable OpenCV video codec")


def read_gray(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def _loaded_indices(path, frames, n_frames=16, **kw):
    """Map the frames returned by load_video back to source indices (no resize/normalising)."""
    size = frames[0].shape[0]
    vid = load_video(path, n_frames=n_frames, resize=size, normalize=False, **kw)
    stack = np.stack(frames).astype(np.float32)
    out = []
    for f in vid[..., 0]:
        out.append(int(np.argmin(np.abs(stack - f[None]).reshape(len(stack), -1).sum(1))))
    return out


class TestContrastPeakSampler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_indices(self):
        for n, peak in [(60, 30), (60, 4), (60, 57), (40, 20)]:
            frames = bolus_frames(n, peak)
            scores = cp.dye_scores(frames)
            self.assertEqual(cp.peak_frame(scores), peak)
            idx = cp.peak_indices(scores)
            self.assertEqual(idx[0], min(max(peak - 15, 0), n - 31))
            self.assertTrue(np.all(np.diff(idx) == 2))
            np.testing.assert_array_equal(idx, _ref_peak_s2(frames))
        short = bolus_frames(20, 10)
        np.testing.assert_array_equal(
            cp.peak_indices(cp.dye_scores(short)),
            np.rint(np.linspace(0, 19, 16)).astype(np.int64),
        )

    def test_load_video_peak_window(self):
        for n, peak in [(70, 50), (60, 57), (24, 5)]:
            frames = bolus_frames(n, peak)
            path = write_video(os.path.join(self.tmp.name, f"v{n}_{peak}.avi"), frames)
            ref = _ref_peak_s2(read_gray(path)).tolist()
            got = _loaded_indices(path, frames, frame_sampling="peak", stride=3)
            self.assertEqual(got, ref)
            # the default stays the uniform sampler
            self.assertEqual(
                _loaded_indices(path, frames), np.linspace(0, n - 1, 16).astype(int).tolist()
            )

    def test_load_video_peak_npy_and_stride(self):
        frames = bolus_frames(80, 40)
        path = os.path.join(self.tmp.name, "v.npy")
        np.save(path, np.stack(frames))
        got = _loaded_indices(path, frames, n_frames=8, frame_sampling="peak", peak_stride=3)
        # span 21, start = round(40 - 10.5) = 30 (Python rounds half to even)
        self.assertEqual(got, (30 + 3 * np.arange(8)).tolist())

    def test_rejects_unknown_mode(self):
        frames = bolus_frames(40, 20)
        path = write_video(os.path.join(self.tmp.name, "v.avi"), frames)
        with self.assertRaises(ValueError):
            load_video(path, frame_sampling="middle")

    def test_datasets_pass_frame_sampling(self):
        from dataloaders.stats_dataset import StatsDataset
        from dataloaders.video_dataset import VideoDataset

        path = write_video(os.path.join(self.tmp.name, "v.avi"), bolus_frames(40, 20))
        csv = os.path.join(self.tmp.name, "d.csv")
        pd.DataFrame({"target_video_path": [path], "Split": ["train"], "y": [1.0]}).to_csv(
            csv, sep="α", index=False
        )
        for module, ctor in (
            (
                "dataloaders.video_dataset",
                lambda **kw: VideoDataset(csv, "train", ["y"], num_frames=16, **kw),
            ),
            (
                "dataloaders.stats_dataset",
                lambda **kw: StatsDataset(csv, "train", ["y"], num_frames=16, **kw),
            ),
        ):
            with patch(
                f"{module}.load_video", return_value=np.zeros((16, 224, 224, 3), np.float32)
            ) as m:
                ctor()[0]
                self.assertEqual(m.call_args.kwargs["frame_sampling"], "uniform")
                ctor(frame_sampling="peak", peak_stride=2)[0]
                self.assertEqual(m.call_args.kwargs["frame_sampling"], "peak")
                self.assertEqual(m.call_args.kwargs["peak_stride"], 2)


def _real_videos(k=3):
    if not VIDEO_PARQUET.exists():
        pytest.skip("DeepCoro video parquet not available")
    files = pd.read_parquet(VIDEO_PARQUET, columns=["FileName"]).FileName.dropna()
    found = [f for f in files.iloc[::997][:60] if f and Path(f).is_file()][:k]
    if len(found) < k:
        pytest.skip("Videos listed in the parquet are not readable here")
    return found


@pytest.mark.real_data
def test_matches_training_sampler_on_real_videos(tmp_path):
    for i, path in enumerate(_real_videos(3)):
        frames = read_gray(path)
        ref = _ref_peak_s2(frames)
        # the reference frames, already selected, go through the same uniform pipeline (16 -> 16)
        npy = tmp_path / f"ref{i}.npy"
        np.save(npy, np.stack(frames)[ref])
        expected = load_video(str(npy), n_frames=16, normalize=False)
        got = load_video(path, n_frames=16, normalize=False, frame_sampling="peak")
        np.testing.assert_array_equal(got, expected)
