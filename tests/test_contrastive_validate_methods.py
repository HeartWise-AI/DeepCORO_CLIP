from runners.video_constrative_learning_runner import VideoContrastiveLearningRunner
from runners.video_constrative_learning_runner_simple import VideoContrastiveLearningRunnerSimple
from utils.enums import RunMode


def _assert_validate_dispatches_to_val_epoch(runner_cls):
    runner = runner_cls.__new__(runner_cls)
    calls = []

    def fake_run_epoch(mode, epoch):
        calls.append((mode, epoch))
        return {"val/loss": 0.0, "val/num_batches": 1.0}

    runner._run_epoch = fake_run_epoch

    assert runner.validate() == {"val/loss": 0.0, "val/num_batches": 1.0}
    assert calls == [(RunMode.VALIDATE, 0)]


def test_contrastive_validate_dispatches_to_validation_epoch():
    _assert_validate_dispatches_to_val_epoch(VideoContrastiveLearningRunner)


def test_simple_contrastive_validate_dispatches_to_validation_epoch():
    _assert_validate_dispatches_to_val_epoch(VideoContrastiveLearningRunnerSimple)
