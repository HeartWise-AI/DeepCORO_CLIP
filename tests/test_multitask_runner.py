from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from projects.multitask_pretraining_project import MultitaskPretrainingProject
from runners.multitask_runner import MultitaskRunner


class _FakeVideoEncoder(torch.nn.Module):
    token_pooling_mode = "mean"
    attention_pool = None

    def forward(self, videos: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"study_features": videos.float()}


class _Loader:
    def __init__(self, batches):
        self._batches = batches
        self.dataset = SimpleNamespace(multi_video_mode=False)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def test_multitask_inference_writes_averaged_metadata(tmp_path):
    text_embeddings_path = tmp_path / "text_embeddings.pt"
    torch.save(torch.eye(2), text_embeddings_path)

    config = SimpleNamespace(
        recall_k=[],
        ndcg_k=[],
        device="cpu",
        world_size=1,
        is_ref_device=True,
        text_embeddings_path=str(text_embeddings_path),
        metadata_path=str(tmp_path / "metadata.parquet"),
        topk=1,
        groupby_column=None,
    )
    loader = _Loader(
        [
            {
                "videos": torch.eye(2),
                "paths": ["video_a.mp4", "video_b.mp4"],
            }
        ]
    )
    runner = MultitaskRunner(
        config=config,
        val_loader=loader,
        video_encoder=_FakeVideoEncoder(),
        output_dir=str(tmp_path),
    )

    metadata = pd.DataFrame(
        {
            "numeric_col": [2.0, 4.0],
            "string_col": ["A", "B"],
        }
    )
    with patch("runners.multitask_runner.pd.read_parquet", return_value=metadata):
        rows = runner.inference()

    output = pd.read_csv(tmp_path / "averaged_metadata.csv")
    assert rows == [
        {"video_name": "video_a.mp4", "numeric_col": 2.0, "string_col": "A"},
        {"video_name": "video_b.mp4", "numeric_col": 4.0, "string_col": "B"},
    ]
    assert output.to_dict("records") == rows


def test_multitask_inference_setup_loads_video_state_without_weight_only(tmp_path):
    class _FakeInferenceEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded_state = None

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

    encoder = _FakeInferenceEncoder()
    config = SimpleNamespace(
        checkpoint=str(tmp_path / "checkpoint.pt"),
        device="cpu",
        world_size=1,
        model_name="fake",
        frames=1,
        pretrained=False,
        video_freeze_ratio=0.0,
        dropout=0.0,
        num_heads=1,
        aggregator_depth=1,
        aggregate_videos_tokens=True,
        per_video_pool=False,
        video_pooling_mode="mean",
        attention_pool_heads=1,
        attention_pool_dropout=0.0,
        use_cls_token=False,
        multi_video_cls_aggregation="mean",
        inference_results_path=str(tmp_path),
    )
    project = MultitaskPretrainingProject(config=config, wandb_wrapper=None)
    checkpoint = {
        "video_encoder": {"sentinel": torch.tensor([1.0])},
        "train/log_temp": torch.tensor([0.0]),
    }

    with (
        patch(
            "projects.multitask_pretraining_project.calculate_dataset_statistics_ddp",
            return_value=(torch.zeros(3), torch.ones(3)),
        ),
        patch(
            "projects.multitask_pretraining_project.get_distributed_video_clip_dataloader",
            return_value=object(),
        ),
        patch(
            "projects.multitask_pretraining_project.ModelRegistry.get",
            return_value=lambda **_: encoder,
        ),
        patch(
            "projects.multitask_pretraining_project.DistributedUtils.DDP",
            side_effect=lambda module, device_ids: SimpleNamespace(module=module),
        ),
        patch.object(project, "_load_checkpoint", return_value=checkpoint),
    ):
        setup = project._setup_inference_objects()

    assert encoder.loaded_state is checkpoint["video_encoder"]
    assert setup["video_encoder"].module is encoder
