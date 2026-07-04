"""Regression test: runner exposes the `num_batches` epoch-metric convention.

Contract (branch autoresearch/deeprv_apr08, item #18 runners):
    The epoch loop must record the processed batch count in its returned
    epoch metrics dict under the key ``num_batches`` (used downstream for
    DDP-correct mean reduction of losses/metrics).

A tiny end-to-end runner fixture is impractical here (needs a full model +
dataloader + distributed setup), so this is a documented SOURCE-LEVEL guard:
it asserts the runner module source contains the ``num_batches`` epoch-metric
convention. If the runner is refactored, update this guard accordingly.
"""

import re
from pathlib import Path

import pytest

# Repo root = parent of tests/
_RUNNERS_DIR = Path(__file__).resolve().parent.parent / "runners"

# Runners that own an epoch loop and must report num_batches.
_CANDIDATE_RUNNERS = [
    "multitask_runner.py",
    "video_constrative_learning_runner.py",
    "linear_probing_runner.py",
]


def test_runners_dir_exists():
    assert _RUNNERS_DIR.is_dir(), f"runners dir not found: {_RUNNERS_DIR}"


def test_at_least_one_runner_records_num_batches_epoch_metric():
    """At least one epoch-loop runner must emit `num_batches` in its
    epoch_metrics dict (source-level guard)."""
    # Matches: "num_batches": num_batches   OR   epoch_metrics["num_batches"]
    pat_dict_entry = re.compile(r'["\']num_batches["\']\s*:')
    pat_index = re.compile(r'epoch_metrics\[\s*["\']num_batches["\']\s*\]')

    found_in = []
    for name in _CANDIDATE_RUNNERS:
        path = _RUNNERS_DIR / name
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        if pat_dict_entry.search(src) or pat_index.search(src):
            found_in.append(name)

    assert found_in, (
        "No runner records the `num_batches` epoch-metric convention. "
        "Expected a '\"num_batches\": num_batches' entry (or "
        "epoch_metrics[\"num_batches\"]) in one of: "
        f"{_CANDIDATE_RUNNERS}. Runner fix (#18) not landed yet."
    )


def test_multitask_runner_specifically_reports_num_batches():
    """multitask_runner already returns num_batches today; lock it in so a
    regression there is caught immediately."""
    path = _RUNNERS_DIR / "multitask_runner.py"
    if not path.exists():
        pytest.skip("multitask_runner.py not present in this checkout")
    src = path.read_text(encoding="utf-8", errors="ignore")
    assert re.search(r'["\']num_batches["\']\s*:', src), (
        "multitask_runner.py no longer emits a '\"num_batches\":' epoch "
        "metric entry -- DDP mean-reduction guard regressed."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
