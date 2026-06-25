from types import SimpleNamespace

import pandas as pd
import torch

from runners.video_constrative_learning_runner import VideoContrastiveLearningRunner
from runners.video_constrative_learning_runner_simple import VideoContrastiveLearningRunnerSimple


class _FakeDataset:
    multi_video_mode = False


class _FakeLoader:
    def __init__(self):
        self.dataset = _FakeDataset()

    def __iter__(self):
        yield {
            "videos": torch.ones(2, 1, 1, 1, 1, 1),
            "paths": ["video_a.mp4", "video_b.mp4"],
        }


class _DictVideoEncoder:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, videos):
        assert videos.device.type == "cpu"
        assert videos.dtype == torch.float32
        return {
            "video_embeds": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        }


def _make_runner(runner_cls, tmp_path):
    embeddings_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.5, 0.5],
                ]
            )
        },
        embeddings_path,
    )
    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "numeric_col": [2.0, 4.0, 10.0],
            "string_col": ["A", "B", "C"],
        }
    ).to_parquet(metadata_path)

    runner = runner_cls.__new__(runner_cls)
    runner.config = SimpleNamespace(
        text_embeddings_path=str(embeddings_path),
        metadata_path=str(metadata_path),
        inference_results_path=str(tmp_path),
        is_ref_device=True,
        topk=2,
        groupby_column=None,
    )
    runner.device = "cpu"
    runner.world_size = 1
    runner.output_dir = str(tmp_path)
    runner.val_loader = _FakeLoader()
    runner.video_encoder = _DictVideoEncoder()
    return runner


def _assert_runner_inference(runner_cls, tmp_path):
    runner = _make_runner(runner_cls, tmp_path)

    rows = runner.inference()

    assert runner.video_encoder.eval_called
    assert rows == [
        {"video_name": "video_a.mp4", "numeric_col": 6.0, "string_col": "A"},
        {"video_name": "video_b.mp4", "numeric_col": 7.0, "string_col": "B"},
    ]
    output = pd.read_csv(tmp_path / "averaged_metadata.csv")
    assert output.to_dict("records") == rows


def test_contrastive_runner_inference_uses_shared_retrieval_path(tmp_path):
    _assert_runner_inference(VideoContrastiveLearningRunner, tmp_path)


def test_simple_contrastive_runner_inference_uses_shared_retrieval_path(tmp_path):
    _assert_runner_inference(VideoContrastiveLearningRunnerSimple, tmp_path)
