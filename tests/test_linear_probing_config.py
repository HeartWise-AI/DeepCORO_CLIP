import yaml

from utils.config.heartwise_config import HeartWiseConfig
from utils.config.linear_probing_config import LinearProbingConfig  # noqa: F401


def test_linear_probing_config_preserves_resume_checkpoint_path(tmp_path):
    with open("tests/config/linear_probing_base_config.yaml", "r") as handle:
        config_data = yaml.safe_load(handle)

    config_data["base_checkpoint_path"] = "outputs"
    config_data["resume_checkpoint_path"] = "/tmp/resume_checkpoint.pt"
    config_path = tmp_path / "linear_probing_resume.yaml"
    config_path.write_text(yaml.safe_dump(config_data))

    config = HeartWiseConfig.from_yaml(str(config_path))

    assert config.resume_checkpoint_path == "/tmp/resume_checkpoint.pt"
