"""Regression test: CLIPLoss multi-positive pos_mask path.

Contract (branch autoresearch/deeprv_apr08):
    utils.loss.contrastive.CLIPLoss must accept a [B, B] pos_mask describing
    multiple positives (e.g. samples 0 and 1 share the same report) and
    return a finite scalar via the multi-positive InfoNCE branch.
"""

import pytest
import torch

from utils.loss.contrastive import CLIPLoss


def test_clip_loss_shared_report_multi_positive_finite():
    torch.manual_seed(0)

    embed_dim = 16
    video_features = torch.randn(3, embed_dim)
    text_features = torch.randn(3, embed_dim)
    log_temp = torch.log(torch.tensor(0.07))

    # Samples 0 and 1 share a report -> both rows/cols positive for {0,1}.
    pos_mask = torch.tensor(
        [[1.0, 1.0, 0.0],
         [1.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]
    )

    loss_fn = CLIPLoss()
    loss = loss_fn(video_features, text_features, log_temp, pos_mask=pos_mask)

    assert loss.ndim == 0, f"expected scalar, got {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.item() > 0.0, f"expected positive loss, got {loss.item()}"


def test_clip_loss_wrong_mask_shape_raises():
    video_features = torch.randn(3, 8)
    text_features = torch.randn(3, 8)
    log_temp = torch.log(torch.tensor(0.07))
    bad_mask = torch.ones(3, 4)  # mismatched vs [3, 3] logits
    loss_fn = CLIPLoss()
    with pytest.raises(ValueError):
        loss_fn(video_features, text_features, log_temp, pos_mask=bad_mask)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
