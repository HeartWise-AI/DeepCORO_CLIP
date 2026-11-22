"""Train-split statistics for the linear-probing heads: class weights and target z-scoring.

Both helpers read the train dataset's per-sample outcomes (a list of {head: value} dicts,
``VideoDataset.outcomes``) and ignore missing (NaN) labels.
"""
from typing import Any, Optional

import numpy as np
import torch

from utils.enums import LossType, MetricTask


def head_values(
    outcomes: list[Optional[dict]],
    head: str,
    labels_map: Optional[dict] = None,
) -> np.ndarray:
    """The finite numeric values of ``head`` over ``outcomes`` (string labels mapped via ``labels_map``)."""
    mapping = (labels_map or {}).get(head) or {}
    values = []
    for outcome in outcomes:
        if outcome is None or head not in outcome:
            continue
        value = outcome[head]
        if isinstance(value, str):
            if value not in mapping:
                raise ValueError(f"Label '{value}' not found in labels_map for column '{head}'.")
            value = mapping[value]
        values.append(float(value))
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _head_kind(head: str, head_structure: dict, loss_structure: Optional[dict]) -> str:
    """'binary' for a BCE head, 'multiclass' for a CE head; anything else cannot take class weights."""
    num_classes = head_structure[head]
    loss_name = (loss_structure or {}).get(head)
    if num_classes == 1 and loss_name in (None, LossType.BCE_LOGIT):
        return "binary"
    if num_classes > 1 and loss_name in (None, LossType.CE):
        return "multiclass"
    raise ValueError(
        f"`class_weights.{head}`: class weights need a bce_logit (1 output) or ce (>1 outputs) head, "
        f"not loss {loss_name!r} with {num_classes} output(s)."
    )


def resolve_class_weights(
    class_weights: Any,
    head_structure: dict,
    loss_structure: Optional[dict],
    train_outcomes: list[Optional[dict]],
    labels_map: Optional[dict] = None,
    cap: float = 10.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Resolve the ``class_weights`` config into (BCE ``pos_weight``s, CE per-class ``weight``s).

    ``class_weights`` is ``None`` (no weighting), ``"auto"`` (every bce_logit / ce head) or
    {head: "auto" | value}. A binary head's value is its ``pos_weight`` (a number); a multiclass
    head's is a list with one weight per class. ``auto`` is the inverse prevalence on the train
    split (binary: negatives / positives; multiclass: largest class count / class count), capped
    at ``cap`` and floored at 1 so a head with a handful of positives does not dominate.
    """
    if class_weights is None:
        return {}, {}
    if not cap >= 1:
        raise ValueError("`class_weight_cap` must be at least 1.")
    if class_weights == "auto":
        class_weights = {}
        for head in head_structure:
            try:
                _head_kind(head, head_structure, loss_structure)
            except ValueError:
                continue
            class_weights[head] = "auto"
    if not isinstance(class_weights, dict):
        raise ValueError("`class_weights` is `auto` or {head: auto | value}.")

    pos_weights: dict[str, torch.Tensor] = {}
    ce_weights: dict[str, torch.Tensor] = {}
    for head, value in class_weights.items():
        if head not in head_structure:
            raise ValueError(f"`class_weights` names head {head!r} that `head_structure` does not list.")
        kind = _head_kind(head, head_structure, loss_structure)
        expected = 1 if kind == "binary" else head_structure[head]
        if value == "auto":
            values = head_values(train_outcomes, head, labels_map)
            if values.size == 0:
                raise ValueError(f"No training sample has a value of `{head}`, so `auto` weights are undefined.")
            if kind == "binary" and not np.isin(values, (0.0, 1.0)).all():
                raise ValueError(f"`{head}`: a binary head's train targets must be 0 or 1.")
            counts = np.bincount(values.astype(int), minlength=max(2, expected))
            if kind == "binary":
                weight = [min(cap, counts[0] / counts[1]) if counts[1] else 1.0]
            else:
                weight = [min(cap, counts.max() / c) if c else 1.0 for c in counts[:expected]]
            weight = [max(1.0, float(w)) for w in weight]
        else:
            weight = [float(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
        if len(weight) != expected or not all(np.isfinite(w) and w > 0 for w in weight):
            raise ValueError(f"`class_weights.{head}` needs {expected} positive weight(s), not {value!r}.")
        tensor = torch.tensor(weight, dtype=torch.float32)
        if kind == "binary":
            pos_weights[head] = tensor
        else:
            ce_weights[head] = tensor
    return pos_weights, ce_weights


def resolve_target_normalization(
    normalize: Any,
    head_task: dict,
    train_outcomes: list[Optional[dict]],
    labels_map: Optional[dict] = None,
) -> dict[str, dict[str, float]]:
    """{head: {mean, std}} (train split, population std) for the regression heads to z-score.

    ``normalize`` is falsy (off), ``True`` (every regression head) or a list of head names.
    """
    if not normalize:
        return {}
    regression = [h for h, task in head_task.items() if task == MetricTask.REGRESSION]
    heads = regression if normalize is True else list(normalize)
    out = {}
    for head in heads:
        if head not in regression:
            raise ValueError(f"`normalize_regression_targets`: {head!r} is not a regression head.")
        values = head_values(train_outcomes, head, labels_map)
        if values.size == 0:
            raise ValueError(f"No training sample has a value of `{head}`, so it cannot be normalized.")
        mean, std = float(values.mean()), float(values.std(ddof=0))
        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
            raise ValueError(f"`{head}` normalization must be finite with a positive std, not ({mean}, {std}).")
        out[head] = {"mean": mean, "std": std}
    return out
