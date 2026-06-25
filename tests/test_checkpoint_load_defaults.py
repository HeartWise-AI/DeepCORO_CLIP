from unittest.mock import patch

import torch

from models.video_encoder import VideoEncoder
from runners.video_constrative_learning_runner import VideoContrastiveLearningRunner
from runners.video_constrative_learning_runner_simple import (
    VideoContrastiveLearningRunnerSimple,
)


class _StateReceiver:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state_dict, strict=None):
        self.loaded = (state_dict, strict)
        return [], []


def test_video_encoder_checkpoint_loads_with_explicit_full_checkpoint_mode(tmp_path):
    checkpoint_path = tmp_path / "encoder.pt"
    checkpoint_path.touch()
    model = _StateReceiver()
    encoder = VideoEncoder.__new__(VideoEncoder)
    encoder.encoder_path = str(checkpoint_path)
    encoder.model = model
    load_kwargs = []

    def fake_load(path, *, map_location, weights_only):
        load_kwargs.append(
            {"path": path, "map_location": map_location, "weights_only": weights_only}
        )
        return {"model_state_dict": {"module.model.backbone.weight": "video-weight"}}

    with patch("models.video_encoder.torch.load", side_effect=fake_load):
        encoder._load_encoder_checkpoint()

    assert load_kwargs == [
        {
            "path": str(checkpoint_path),
            "map_location": "cpu",
            "weights_only": False,
        }
    ]
    assert model.loaded == ({"backbone.weight": "video-weight"}, False)


def test_simple_contrastive_checkpoint_preview_uses_explicit_full_checkpoint_mode(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    load_kwargs = []

    def fake_load(path, *, map_location, weights_only):
        load_kwargs.append(
            {"path": path, "map_location": map_location, "weights_only": weights_only}
        )
        return {
            "wandb_run": "run-1",
            "epoch": 2,
            "best_val_loss": 0.25,
            "best_epoch": 1,
        }

    with patch(
        "runners.video_constrative_learning_runner_simple.torch.load",
        side_effect=fake_load,
    ):
        result = VideoContrastiveLearningRunnerSimple._preview_checkpoint_for_resuming(
            object(), str(checkpoint_path)
        )

    assert result == ("run-1", 3, 0.25, 1)
    assert load_kwargs == [
        {
            "path": str(checkpoint_path),
            "map_location": "cpu",
            "weights_only": False,
        }
    ]


def test_contrastive_full_checkpoint_loads_use_explicit_full_checkpoint_mode(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()

    for runner_cls, load_target in (
        (
            VideoContrastiveLearningRunner,
            "runners.video_constrative_learning_runner.torch.load",
        ),
        (
            VideoContrastiveLearningRunnerSimple,
            "runners.video_constrative_learning_runner_simple.torch.load",
        ),
    ):
        video_encoder = _StateReceiver()
        text_encoder = _StateReceiver()
        optimizer = _StateReceiver()
        scheduler = _StateReceiver()
        load_kwargs = []
        checkpoint = {
            "video_encoder": {"video": "video-state"},
            "text_encoder": {"text": "text-state"},
            "optimizer": {"optim": 1},
            "scheduler": {"sched": 1},
        }

        def fake_load(path, *, map_location, weights_only):
            load_kwargs.append(
                {
                    "path": path,
                    "map_location": map_location,
                    "weights_only": weights_only,
                }
            )
            return checkpoint

        with patch(load_target, side_effect=fake_load):
            runner_cls._load_full_checkpoint(
                object(),
                str(checkpoint_path),
                torch.device("cpu"),
                {
                    "video_encoder": video_encoder,
                    "text_encoder": text_encoder,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
            )

        assert load_kwargs == [
            {
                "path": str(checkpoint_path),
                "map_location": torch.device("cpu"),
                "weights_only": False,
            }
        ]
        assert video_encoder.loaded == (checkpoint["video_encoder"], False)
        assert text_encoder.loaded == (checkpoint["text_encoder"], False)
        assert optimizer.loaded == (checkpoint["optimizer"], None)
        assert scheduler.loaded == (checkpoint["scheduler"], None)
