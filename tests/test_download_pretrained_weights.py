import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from utils import download_pretrained_weights as downloader


class TestDownloadPretrainedWeights(unittest.TestCase):
    def test_parse_model_selection(self) -> None:
        cases = [
            ("stenosis", ["stenosis"]),
            ("MACE", ["mace"]),
            ("stenosis,mace", ["stenosis", "mace"]),
            (" mace , stenosis ", ["mace", "stenosis"]),
            ("mace,mace", ["mace"]),
        ]

        for selection, expected in cases:
            with self.subTest(selection=selection):
                self.assertEqual(downloader.parse_model_selection(selection), expected)

    def test_parse_model_selection_rejects_invalid_values(self) -> None:
        for selection in ["", "stenosis,", "unknown"]:
            with self.subTest(selection=selection):
                with self.assertRaises(ValueError):
                    downloader.parse_model_selection(selection)

    @patch.object(downloader, "download_pretrained_weights")
    def test_download_selected_models_downloads_requested_model_sets(
        self,
        mock_download,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            expected_calls = {
                "stenosis": call(
                    repo_id="heartwise/deepcoro_clip_stenosis",
                    local_dir=str(project_root / "weights" / "deepcoro_clip_generic"),
                    hugging_face_api_key="test-token",
                ),
                "mace": call(
                    repo_id="heartwise/deepcoro_clip_mace",
                    local_dir=str(project_root / "weights" / "deepcoro_clip_mace"),
                    hugging_face_api_key="test-token",
                ),
            }

            for model_names in [["stenosis"], ["mace"], ["stenosis", "mace"]]:
                with self.subTest(model_names=model_names):
                    mock_download.reset_mock()
                    downloader.download_selected_models(
                        model_names,
                        hugging_face_api_key="test-token",
                        project_root=project_root,
                    )
                    self.assertEqual(
                        mock_download.call_args_list,
                        [expected_calls[model_name] for model_name in model_names],
                    )

    @patch.object(
        downloader,
        "download_pretrained_weights",
        side_effect=PermissionError("access denied"),
    )
    def test_download_selected_models_reports_requested_inaccessible_model(
        self,
        _mock_download,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "Failed to download requested model 'mace' from "
                    "'heartwise/deepcoro_clip_mace'"
                ),
            ):
                downloader.download_selected_models(
                    ["mace"],
                    hugging_face_api_key="test-token",
                    project_root=Path(temp_dir),
                )


if __name__ == "__main__":
    unittest.main()
