import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

from utils import download_pretrained_weights as downloader


class TestDownloadPretrainedWeights(unittest.TestCase):
    @patch("utils.huggingface_wrapper.HuggingFaceWrapper")
    @patch("builtins.print")
    def test_download_pretrained_weights_uses_wrapper(
        self,
        mock_print,
        mock_wrapper_class,
    ) -> None:
        mock_wrapper_class.return_value.get_model.return_value = "/models/downloaded"

        downloader.download_pretrained_weights(
            local_dir="/models/local",
            hugging_face_api_key="test-token",
            repo_id="heartwise/test-model",
        )

        mock_wrapper_class.assert_called_once_with("test-token")
        mock_wrapper_class.return_value.get_model.assert_called_once_with(
            "heartwise/test-model",
            "/models/local",
        )
        mock_print.assert_called_once_with(
            "✓ Successfully downloaded model to: /models/downloaded"
        )

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

    @patch.object(downloader, "download_selected_models")
    @patch(
        "utils.files_handler.read_api_key",
        return_value={"HUGGING_FACE_API_KEY": "test-token"},
    )
    def test_main_reads_api_key_and_downloads_selected_models(
        self,
        mock_read_api_key,
        mock_download_selected_models,
    ) -> None:
        downloader.main(["--models", "MACE, stenosis"])

        mock_read_api_key.assert_called_once_with("api_key.json")
        mock_download_selected_models.assert_called_once_with(
            ["mace", "stenosis"],
            "test-token",
            Path(downloader.__file__).resolve().parent.parent,
        )

    def test_main_reports_invalid_model_selection(self) -> None:
        with patch("sys.stderr"), self.assertRaisesRegex(SystemExit, "2"):
            downloader.main(["--models", "unknown"])

    def test_standalone_import_fallbacks(self) -> None:
        wrapper_module = ModuleType("huggingface_wrapper")
        wrapper_class = MagicMock()
        wrapper_class.return_value.get_model.return_value = "/models/downloaded"
        wrapper_module.HuggingFaceWrapper = wrapper_class

        files_handler_module = ModuleType("files_handler")
        read_api_key = MagicMock(return_value={"HUGGING_FACE_API_KEY": "standalone-token"})
        files_handler_module.read_api_key = read_api_key

        with (
            patch.object(downloader, "__package__", ""),
            patch.dict(
                "sys.modules",
                {
                    "huggingface_wrapper": wrapper_module,
                    "files_handler": files_handler_module,
                },
            ),
            patch.object(downloader, "download_selected_models") as mock_download,
            patch("builtins.print"),
        ):
            downloader.download_pretrained_weights(
                local_dir="/models/local",
                hugging_face_api_key="standalone-token",
                repo_id="heartwise/test-model",
            )
            downloader.main(["--models", "mace"])

        wrapper_class.assert_called_once_with("standalone-token")
        wrapper_class.return_value.get_model.assert_called_once_with(
            "heartwise/test-model",
            "/models/local",
        )
        read_api_key.assert_called_once_with("api_key.json")
        mock_download.assert_called_once_with(
            ["mace"],
            "standalone-token",
            Path(downloader.__file__).resolve().parent.parent,
        )


if __name__ == "__main__":
    unittest.main()
