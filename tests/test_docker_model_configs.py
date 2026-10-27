"""Docker base configs: weights resolve to the HF download folders, and the coronary-system filter is read correctly."""
import tempfile
import unittest
from pathlib import Path

import yaml

from utils import download_pretrained_weights as downloader

try:  # external_validation imports Docker-only packages (heartwise_statplots, Orion)
    from scripts import external_validation as ev
except ImportError as error:  # pragma: no cover - depends on the environment
    ev = None
    EV_IMPORT_ERROR = str(error)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "cathef": ("config/linear_probing/CathEF/docker_base_config_cathef.yaml", ["Left Coronary"]),
    "deeprv": ("config/linear_probing/DeepRV/docker_base_config_deeprv.yaml", ["Left Coronary", "Right Coronary"]),
}


class TestDockerModelConfigs(unittest.TestCase):
    def test_inference_model_path_is_inside_downloaded_weights(self) -> None:
        for model, (path, _) in CONFIGS.items():
            with self.subTest(model=model):
                config = yaml.safe_load((ROOT / path).read_text())
                _, local_name = downloader.PRETRAINED_MODELS[model]
                self.assertTrue(config["inference_model_path"].startswith(f"weights/{local_name}/"))
                self.assertEqual(config["data_filename"], "tmp/df_preprocessed_filtered.csv")

    @unittest.skipIf(ev is None, "scripts.external_validation needs the Docker-only dependencies")
    def test_main_structures_from_config(self) -> None:
        for model, (path, expected) in CONFIGS.items():
            with self.subTest(model=model):
                self.assertEqual(ev.get_deepcoro_main_structures(ROOT / path), expected)

    @unittest.skipIf(ev is None, "scripts.external_validation needs the Docker-only dependencies")
    def test_main_structures_default_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("pipeline_project: DeepCORO_video_linear_probing\n")
            self.assertEqual(ev.get_deepcoro_main_structures(config_path), ["Left Coronary", "Right Coronary"])
            config_path.write_text("vasovision_main_structures:\n- Graft\n")
            with self.assertRaises(ValueError):
                ev.get_deepcoro_main_structures(config_path)


if __name__ == "__main__":
    unittest.main()
