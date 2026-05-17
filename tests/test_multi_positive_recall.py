"""Regression test: compute_multi_positive_recall_at_k / recall_any_at_k.

Contract (branch autoresearch/deeprv_apr08, item #19 retrieval metrics):
    utils.retrieval_metrics must expose:
      - compute_multi_positive_recall_at_k(similarity, positive_mask, k_values)
            -> {f"Recall@{k}": float}
      - recall_any_at_k(similarity, positive_mask, k_values)
            -> {f"RecallAny@{k}": float}
    Both take a list of cutoffs and return a dict keyed by cutoff.

    For
        similarity     = [[0.1, 0.9, 0.2],
                           [0.8, 0.1, 0.7]]
        positive_mask  = [[0, 1, 0],
                           [0, 0, 1]]
      Row 0 top-1 = col 1 (positive)        -> hit
      Row 1 top-1 = col 0 (NOT positive; pos is col 2) -> miss
        => Recall@1 == 0.5
      Row 0 top-2 = {1, 2} contains pos col 1 -> hit
      Row 1 top-2 = {0, 2} contains pos col 2 -> hit
        => Recall@2 == 1.0

A failing import/AttributeError here means the retrieval-metrics helpers
have not been added yet (sibling task pending) -- this is an intentional
signal, not a silent skip.
"""

import importlib

import pytest
import torch

rm = importlib.import_module("utils.retrieval_metrics")

_HAS_MULTI_POS = hasattr(rm, "compute_multi_positive_recall_at_k")
_HAS_RECALL_ANY = hasattr(rm, "recall_any_at_k")

_PENDING_REASON = (
    "utils.retrieval_metrics.compute_multi_positive_recall_at_k / "
    "recall_any_at_k not implemented yet on this branch (sibling fix #19 "
    "pending)."
)


def _similarity():
    return torch.tensor(
        [[0.1, 0.9, 0.2],
         [0.8, 0.1, 0.7]]
    )


def _positive_mask():
    return torch.tensor(
        [[0, 1, 0],
         [0, 0, 1]]
    )


@pytest.mark.xfail(not _HAS_MULTI_POS, reason=_PENDING_REASON, strict=True)
def test_multi_positive_recall_at_1_and_2():
    fn = rm.compute_multi_positive_recall_at_k
    out = fn(_similarity(), _positive_mask(), [1, 2])

    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    r1 = out["Recall@1"]
    r2 = out["Recall@2"]

    assert r1 == pytest.approx(0.5), f"Recall@1 expected 0.5, got {r1}"
    assert r2 == pytest.approx(1.0), f"Recall@2 expected 1.0, got {r2}"


@pytest.mark.xfail(not _HAS_RECALL_ANY, reason=_PENDING_REASON, strict=True)
def test_recall_any_at_k_quick_check():
    fn = rm.recall_any_at_k
    out = fn(_similarity(), _positive_mask(), [1, 2])

    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    r1 = out["RecallAny@1"]
    r2 = out["RecallAny@2"]

    # "any positive in top-k": same as multi-positive recall for this case.
    assert 0.0 <= r1 <= 1.0
    assert r1 == pytest.approx(0.5), f"RecallAny@1 expected 0.5, got {r1}"
    assert r2 == pytest.approx(1.0), f"RecallAny@2 expected 1.0, got {r2}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
