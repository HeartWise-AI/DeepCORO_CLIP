"""Regression test: ClipConfig exposes the SigLIP tuning dataclass fields.

Contract (branch autoresearch/deeprv_apr08):
    utils.config.clip_config.ClipConfig must declare the SigLIP knobs used by
    the loss factory / contrastive pretraining project.

A missing field means the config fix has not landed yet.
"""

import pytest

from utils.config.clip_config import ClipConfig

REQUIRED_SIGLIP_FIELDS = [
    "siglip_positive_loss_weight",
    "siglip_negative_loss_weight",
    "siglip_enable_severity_weighting",
    "siglip_auto_positive_loss_weight",
    "siglip_entropy_regularization",
    "siglip_entropy_weight",
    "siglip_min_entropy_threshold",
    "siglip_gather_in_ddp",
]


@pytest.mark.parametrize("field_name", REQUIRED_SIGLIP_FIELDS)
def test_clip_config_has_siglip_field(field_name):
    fields = ClipConfig.__dataclass_fields__
    assert field_name in fields, (
        f"ClipConfig missing SigLIP dataclass field '{field_name}'. "
        f"Config fix not landed yet."
    )


def test_clip_config_siglip_field_set_is_complete():
    fields = set(ClipConfig.__dataclass_fields__)
    missing = [f for f in REQUIRED_SIGLIP_FIELDS if f not in fields]
    assert not missing, f"ClipConfig missing SigLIP fields: {missing}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
