"""Optional study-level tabular inputs (e.g. blood pressure, heart rate) for linear probing."""
import math

import pandas as pd
import pytest
import torch

from dataloaders.video_dataset import TABULAR_KEY, VideoDataset, custom_collate_fn
from models.multi_instance_linear_probing import MultiInstanceLinearProbing

HEADS = {"lad_ifr": 1, "lad_ifr_cat": 1}


def _mil(**kw):
    torch.manual_seed(0)
    return MultiInstanceLinearProbing(embedding_dim=8, head_structure=HEADS, pooling_mode="mean", **kw)


def test_default_is_image_only_and_unchanged():
    model = _mil()
    assert model.tabular_dim == 0
    assert "tabular_mean" not in model.state_dict()
    assert all(h.in_features == 8 for h in model.heads.values())
    x = torch.randn(3, 2, 8)
    torch.testing.assert_close(model(x)["lad_ifr"], _mil()(x)["lad_ifr"])


def test_head_width_and_missing_flags():
    with_flag = _mil(tabular_dim=4)
    without_flag = _mil(tabular_dim=4, tabular_missing_indicator=False)
    assert all(h.in_features == 8 + 8 for h in with_flag.heads.values())
    assert all(h.in_features == 8 + 4 for h in without_flag.heads.values())


def test_normalization_nan_and_clamp():
    model = _mil(tabular_dim=2)
    model.set_tabular_normalization([120.0, 70.0], [20.0, 0.0])  # std 0 -> 1
    assert model.tabular_std.tolist() == [20.0, 1.0]
    pooled = torch.zeros(3, 8)
    tab = torch.tensor([[140.0, 71.0], [float("nan"), 70.0], [1e6, float("inf")]])
    feats = model._tabular_features(tab, pooled)
    assert feats.shape == (3, 4)
    assert feats[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert feats[1].tolist() == [0.0, 0.0, 1.0, 0.0]  # NaN -> 0 with missing flag
    assert feats[2].tolist() == [10.0, 0.0, 0.0, 1.0]  # clamped, inf treated as missing
    out = model(torch.randn(3, 2, 8), tabular=tab)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_tabular_required_and_shape_checked():
    model = _mil(tabular_dim=2)
    x = torch.randn(2, 2, 8)
    with pytest.raises(ValueError, match="expects 2 tabular inputs"):
        model(x)
    with pytest.raises(ValueError, match="tabular must be"):
        model(x, tabular=torch.zeros(2, 3))


def test_statistics_saved_and_restored_with_state_dict():
    model = _mil(tabular_dim=2)
    model.set_tabular_normalization([120.0, 70.0], [20.0, 10.0])
    restored = _mil(tabular_dim=2)
    restored.load_state_dict(model.state_dict())
    x, tab = torch.randn(2, 2, 8), torch.tensor([[130.0, 80.0], [float("nan"), 60.0]])
    torch.testing.assert_close(model(x, tabular=tab)["lad_ifr"], restored(x, tabular=tab)["lad_ifr"])


def test_tabular_changes_predictions():
    model = _mil(tabular_dim=1)
    x = torch.randn(2, 2, 8)
    a = model(x, tabular=torch.tensor([[0.0], [0.0]]))["lad_ifr"]
    b = model(x, tabular=torch.tensor([[3.0], [3.0]]))["lad_ifr"]
    assert not torch.allclose(a, b)


def _write_csv(tmp_path):
    videos = []
    for i in range(4):
        p = tmp_path / f"v{i}.avi"
        p.write_bytes(b"")
        videos.append(str(p))
    df = pd.DataFrame({
        "FileName": videos,
        "StudyInstanceUID": ["s1", "s1", "s2", "s2"],
        "Split": ["train"] * 4,
        "lad_ifr": [0.85, float("nan"), 0.95, 0.95],
        "SYSTAO": [float("nan"), 140.0, 120.0, 125.0],
        "HEARTRATE": [60.0, 65.0, float("nan"), float("nan")],
    })
    path = tmp_path / "data.csv"
    path.write_text("α".join(df.columns) + "\n" + "\n".join("α".join(str(v) for v in r) for r in df.values))
    return str(path)


def test_dataset_reads_first_present_value_per_study(tmp_path):
    ds = VideoDataset(
        data_filename=_write_csv(tmp_path), split="train", target_label=["lad_ifr"],
        datapoint_loc_label="FileName", multi_video=True, groupby_column="StudyInstanceUID",
        num_videos=2, tabular_columns=["SYSTAO", "HEARTRATE"],
    )
    assert ds.tabular[0] == [140.0, 60.0]  # s1: SYSTAO from row 2, HEARTRATE from row 1
    assert ds.tabular[1][0] == 120.0 and math.isnan(ds.tabular[1][1])  # s2: HR missing
    assert ds._with_tabular({"lad_ifr": 0.85}, 0) == {"lad_ifr": 0.85, TABULAR_KEY: [140.0, 60.0]}


def test_dataset_rejects_unknown_column(tmp_path):
    with pytest.raises(ValueError, match="tabular_input_columns not found"):
        VideoDataset(
            data_filename=_write_csv(tmp_path), split="train", target_label=["lad_ifr"],
            datapoint_loc_label="FileName", multi_video=True, num_videos=2,
            tabular_columns=["MEANAO"],
        )


def test_collate_splits_tabular_from_targets():
    import numpy as np
    video = np.zeros((2, 16, 4, 4, 3), dtype=np.float32)
    batch = [
        (video, {"lad_ifr": 0.8, TABULAR_KEY: [140.0, float("nan")]}, ["a", "b"]),
        (video, {"lad_ifr": 0.9, TABULAR_KEY: [120.0, 70.0]}, ["c", "PAD"]),
    ]
    out = custom_collate_fn(batch)
    assert set(out["targets"]) == {"lad_ifr"}
    assert out["tabular"].shape == (2, 2)
    assert math.isnan(out["tabular"][0, 1].item()) and out["tabular"][1, 1].item() == 70.0


def test_collate_without_tabular_has_no_key():
    import numpy as np
    video = np.zeros((2, 16, 4, 4, 3), dtype=np.float32)
    out = custom_collate_fn([(video, {"lad_ifr": 0.8}, ["a", "b"])])
    assert "tabular" not in out


def test_wrapper_forwards_tabular():
    from projects.linear_probing_project import VideoMILWrapper

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], x.shape[1], 8)

    mil = _mil(tabular_dim=1)
    wrapper = VideoMILWrapper(Encoder(), mil, num_videos=2)
    x = torch.ones(2, 2, 16, 4, 4, 3)
    a = wrapper(x, tabular=torch.tensor([[0.0], [0.0]]))["lad_ifr"]
    b = wrapper(x, tabular=torch.tensor([[5.0], [5.0]]))["lad_ifr"]
    assert not torch.allclose(a, b)


def test_project_fits_train_statistics():
    from types import SimpleNamespace
    from projects.linear_probing_project import LinearProbingProject

    project = LinearProbingProject.__new__(LinearProbingProject)
    project.config = SimpleNamespace(tabular_input_columns=["SYSTAO", "HEARTRATE"],
                                     tabular_missing_indicator=True, is_ref_device=False)
    assert project._tabular_model_kwargs() == {"tabular_dim": 2, "tabular_missing_indicator": True}
    mil = _mil(**project._tabular_model_kwargs())
    loader = SimpleNamespace(dataset=SimpleNamespace(tabular=[[100.0, 60.0], [140.0, float("nan")]]))
    project._fit_tabular_normalization(mil, loader)
    assert mil.tabular_mean.tolist() == [120.0, 60.0]
    assert mil.tabular_std.tolist() == [20.0, 1.0]  # single value -> std 0 -> 1

    project.config.tabular_input_columns = None
    assert project._tabular_model_kwargs() == {}
