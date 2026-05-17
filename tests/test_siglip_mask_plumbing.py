"""Regression test: SigLIPLoss off-diagonal pos_mask + pos_weights plumbing.

Contract (branch autoresearch/deeprv_apr08):
    utils.loss.contrastive.SigLIPLoss must accept a [B, T] pos_mask and a
    [B, T] pos_weights tensor where T != B and the positives are OFF the
    diagonal, producing a finite, strictly-positive scalar loss.

If the import or call fails, the SigLIP mask-plumbing fix has not landed yet
(this is a clear signal, not a silent skip).
"""

import pytest
import torch

# A failing import here means the sibling fix is not yet on this branch.
from utils.loss.contrastive import SigLIPLoss


def test_siglip_offdiagonal_pos_mask_and_weights_finite_positive():
    torch.manual_seed(0)

    batch_size = 2
    num_texts = 3
    embed_dim = 16

    video_features = torch.randn(batch_size, embed_dim)
    text_features = torch.randn(num_texts, embed_dim)
    log_temp = torch.log(torch.tensor(0.07))

    # Off-diagonal positives: video 0 -> text 2, video 1 -> text 0.
    pos_mask = torch.tensor(
        [[0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0]]
    )
    # Per-pair severity weights (only positives are weighted internally).
    pos_weights = torch.tensor(
        [[1.0, 1.0, 3.0],
         [2.0, 1.0, 1.0]]
    )

    loss_fn = SigLIPLoss(use_severity_weights=True)
    loss = loss_fn(
        video_features,
        text_features,
        log_temp,
        pos_mask=pos_mask,
        pos_weights=pos_weights,
    )

    assert loss.ndim == 0, f"expected scalar loss, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.item() > 0.0, f"expected loss > 0, got {loss.item()}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
