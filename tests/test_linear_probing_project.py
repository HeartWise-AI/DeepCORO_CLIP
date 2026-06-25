import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from projects.linear_probing_project import LinearProbingProject


class _FakeVideoEncoder(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.embedding_dim = 4

    def to(self, device):
        return self


class _FakeMIL(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def to(self, device):
        return self


class _FakeWrapper(torch.nn.Module):
    def __init__(self, video_encoder, mil_model, num_videos: int) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.mil_model = mil_model
        self.num_videos = num_videos
        self.loaded_state = None

    def load_state_dict(self, state_dict, strict: bool = True):
        self.loaded_state = state_dict

    def eval(self):
        return self


class TestLinearProbingProject(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            aggregate_videos_tokens=True,
            aggregator_depth=1,
            attention_hidden=4,
            batch_size=2,
            data_filename="unused.csv",
            datapoint_loc_label="video",
            device=torch.device("cpu"),
            dropout=0.0,
            dropout_attention=0.0,
            frames=16,
            groupby_column="StudyInstanceUID",
            head_structure={"test_head": 1},
            head_weights={"test_head": 1.0},
            inference_model_path="/tmp/unused_checkpoint.pt",
            is_ref_device=True,
            loss_structure={"test_head": "mse"},
            model_name="mvit",
            multi_video=True,
            num_heads=1,
            num_videos=3,
            num_workers=0,
            output_dir="/tmp",
            per_video_pool=False,
            pooling_mode="mean",
            pretrained=False,
            rand_augment=False,
            resize=32,
            run_mode="validate",
            shuffle_videos=False,
            stride=1,
            target_label=["test_head"],
            video_freeze_ratio=1.0,
            world_size=1,
        )

    def test_validation_setup_disables_encoder_aggregation_before_encoder_build(self):
        config = self._config()
        wandb_wrapper = Mock()
        wandb_wrapper.is_initialized.return_value = True
        encoder_flags = []

        def registry_get(name):
            if name == "video_encoder":
                def build_encoder(**kwargs):
                    encoder_flags.append(kwargs["aggregate_videos_tokens"])
                    return _FakeVideoEncoder(**kwargs)

                return build_encoder
            if name == "multi_instance_linear_probing":
                return lambda **kwargs: _FakeMIL(**kwargs)
            raise AssertionError(f"Unexpected registry lookup: {name}")

        with (
            patch(
                "projects.linear_probing_project.calculate_dataset_statistics_ddp",
                return_value=(torch.zeros(3), torch.ones(3)),
            ),
            patch(
                "projects.linear_probing_project.get_distributed_video_dataloader",
                return_value=[],
            ),
            patch("projects.linear_probing_project.ModelRegistry.get", side_effect=registry_get),
            patch("projects.linear_probing_project.VideoMILWrapper", _FakeWrapper),
            patch(
                "projects.linear_probing_project.DistributedUtils.DDP",
                side_effect=lambda module, *args, **kwargs: module,
            ),
            patch(
                "projects.linear_probing_project.LossRegistry.get",
                return_value=lambda **kwargs: object(),
            ),
            patch("projects.linear_probing_project.Loss", return_value=object()),
            patch.object(
                LinearProbingProject,
                "_load_and_fix_checkpoint",
                return_value={"linear_probing": {"sentinel": torch.tensor(1)}},
            ),
        ):
            result = LinearProbingProject(config, wandb_wrapper)._setup_validation_objects()

        self.assertEqual(encoder_flags, [False])
        self.assertFalse(config.aggregate_videos_tokens)
        wandb_wrapper.log.assert_called_once_with(
            {"config/aggregate_videos_tokens_override": True}
        )
        self.assertIsInstance(result["linear_probing"], _FakeWrapper)
        self.assertEqual(result["linear_probing"].loaded_state["sentinel"].item(), 1)


if __name__ == "__main__":
    unittest.main()
