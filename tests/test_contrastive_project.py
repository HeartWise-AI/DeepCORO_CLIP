from types import SimpleNamespace
from unittest.mock import patch

import torch

from projects.contrastive_pretraining_project import ContrastivePretrainingProject


def test_contrastive_inference_setup_loads_video_state_without_weight_only(tmp_path):
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
        video_pooling_mode="mean",
        attention_pool_heads=1,
        attention_pool_dropout=0.0,
        use_rope=False,
        rope_base=10000.0,
        rope_temporal_scale=1.0,
        rope_normalize_mode="separate",
        temperature=1.0,
        inference_results_path=str(tmp_path),
    )
    project = ContrastivePretrainingProject(config=config, wandb_wrapper=None)
    checkpoint = {
        "video_encoder": {"sentinel": torch.tensor([1.0])},
        "log_temp": torch.tensor([0.0]),
    }

    with (
        patch(
            "projects.contrastive_pretraining_project.calculate_dataset_statistics_ddp",
            return_value=(torch.zeros(3), torch.ones(3)),
        ),
        patch(
            "projects.contrastive_pretraining_project.get_distributed_video_clip_dataloader",
            return_value=object(),
        ),
        patch(
            "projects.contrastive_pretraining_project.ModelRegistry.get",
            return_value=lambda **_: encoder,
        ),
        patch(
            "projects.contrastive_pretraining_project.DistributedUtils.DDP",
            side_effect=lambda module, device_ids: SimpleNamespace(module=module),
        ),
        patch.object(project, "_load_checkpoint", return_value=checkpoint),
    ):
        setup = project._setup_inference_objects()

    assert encoder.loaded_state is checkpoint["video_encoder"]
    assert setup["video_encoder"].module is encoder
