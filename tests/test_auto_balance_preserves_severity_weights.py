"""Regression test: SigLIP auto_balance preserves severity weights (item B5).

Contract (branch autoresearch/deeprv_apr08):
    With auto_balance=True AND use_severity_weights=True, the per-pair
    severity weights in pos_weights must still influence the loss (they are
    scaled by the neg/pos ratio, NOT discarded). The loss must stay finite.
"""

import pytest
import torch

from utils.loss.contrastive import SigLIPLoss


def _make_inputs():
    torch.manual_seed(0)
    embed_dim = 16
    video_features = torch.randn(2, embed_dim)
    text_features = torch.randn(3, embed_dim)
    log_temp = torch.log(torch.tensor(0.07))
    pos_mask = torch.tensor(
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]
    )
    return video_features, text_features, log_temp, pos_mask


def test_auto_balance_with_severity_weights_finite():
    video_features, text_features, log_temp, pos_mask = _make_inputs()
    # One strongly-weighted positive (severity 5.0).
    pos_weights = torch.tensor(
        [[1.0, 5.0, 1.0],
         [1.0, 1.0, 1.0]]
    )
    loss_fn = SigLIPLoss(auto_balance=True, use_severity_weights=True)
    loss = loss_fn(
        video_features, text_features, log_temp,
        pos_mask=pos_mask, pos_weights=pos_weights,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.item() > 0.0


def test_auto_balance_severity_weight_still_influences_loss():
    """A 5.0 severity weight must change the loss vs an all-ones baseline,
    proving auto_balance did not overwrite the severity-weighted positives."""
    video_features, text_features, log_temp, pos_mask = _make_inputs()

    weighted = torch.tensor(
        [[1.0, 5.0, 1.0],
         [1.0, 1.0, 5.0]]
    )
    ones = torch.ones_like(weighted)

    loss_fn = SigLIPLoss(auto_balance=True, use_severity_weights=True)
    loss_weighted = loss_fn(
        video_features, text_features, log_temp,
        pos_mask=pos_mask, pos_weights=weighted,
    )
    loss_ones = loss_fn(
        video_features, text_features, log_temp,
        pos_mask=pos_mask, pos_weights=ones,
    )

    assert torch.isfinite(loss_weighted) and torch.isfinite(loss_ones)
    assert not torch.isclose(loss_weighted, loss_ones, atol=1e-6), (
        "Severity weights had no effect under auto_balance -> the "
        "severity-weighted positive contribution was discarded (B5 regression)."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
