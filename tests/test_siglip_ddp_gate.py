"""Regression test: SigLIPLoss DDP gather gate defaults to OFF.

Contract (branch autoresearch/deeprv_apr08):
    The gathered-DDP SigLIP path is NOT implemented (needs gathered
    text_features + block-diagonal mask), so SigLIPLoss().gather_in_ddp must
    default to False, and enabling it must raise NotImplementedError on
    forward rather than silently computing a wrong loss.
"""

import pytest
import torch

from utils.loss.contrastive import SigLIPLoss


def test_siglip_gather_in_ddp_defaults_false():
    loss_fn = SigLIPLoss()
    assert loss_fn.gather_in_ddp is False, (
        "SigLIPLoss.gather_in_ddp must default to False until the gathered "
        "path is implemented and tested."
    )


def test_siglip_gather_in_ddp_true_raises_not_implemented():
    loss_fn = SigLIPLoss(gather_in_ddp=True)
    video_features = torch.randn(2, 8)
    text_features = torch.randn(2, 8)
    log_temp = torch.log(torch.tensor(0.07))
    with pytest.raises(NotImplementedError):
        loss_fn(video_features, text_features, log_temp)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
