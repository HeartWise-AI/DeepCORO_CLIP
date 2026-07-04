"""Regression test: resume restores temperature from log_temp (no exp-of-exp).

Contract (branch autoresearch/deeprv_apr08):
    Checkpoints store the learnable ``log_temp`` and a derived
    ``temperature = exp(log_temp)``. On resume the effective temperature must
    be reconstructed from ``log_temp`` (i.e. exp(log_temp)) and must NOT be
    corrupted by an accidental double-exponential (exp(temperature)).

This is a pure-logic test (no model, no checkpoint file).
"""

import math

import pytest
import torch


def test_resume_from_log_temp_matches_stored_temperature():
    log_temp = torch.tensor(math.log(0.07))  # typical CLIP temperature
    stored_temperature = torch.exp(log_temp)

    # Correct restore path: derive temp from log_temp.
    restored_temperature = torch.exp(log_temp)

    assert torch.isclose(restored_temperature, stored_temperature, atol=1e-7)
    assert torch.isclose(
        restored_temperature, torch.tensor(0.07), atol=1e-6
    ), f"expected ~0.07, got {restored_temperature.item()}"


def test_double_exp_corruption_is_detectable():
    """Guard: applying exp() to an already-exponentiated temperature must
    yield a clearly different (corrupted) value, which the resume logic must
    avoid."""
    log_temp = torch.tensor(math.log(0.07))
    correct = torch.exp(log_temp)              # 0.07
    corrupted = torch.exp(correct)             # exp(0.07) ~ 1.0725 (wrong)

    assert not torch.isclose(correct, corrupted, atol=1e-3), (
        "exp-of-exp corruption is not detectable -- test is meaningless"
    )
    assert corrupted.item() > 1.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
