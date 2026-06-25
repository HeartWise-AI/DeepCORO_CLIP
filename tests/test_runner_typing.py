from runners.typing import Runner


class _FakeRunner:
    def train(self, start_epoch, end_epoch):
        return ("train", start_epoch, end_epoch)

    def inference(self):
        return [{"video_name": "video_a.mp4"}]

    def validate(self):
        return {"val/loss": 0.0}

    def test(self):
        return "tested"


def test_runner_wrapper_returns_underlying_results():
    runner = Runner(runner_type=_FakeRunner())

    assert runner.train(start_epoch=2, end_epoch=3) == ("train", 2, 3)
    assert runner.inference() == [{"video_name": "video_a.mp4"}]
    assert runner.validate() == {"val/loss": 0.0}
    assert runner.test() == "tested"
