from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.config.clip_config import ClipConfig


def _normalize_key(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


@dataclass
class SiglipDebugSettings:
    batches_per_epoch: int
    every: int
    sample_count: int
    sync: bool
    barrier_debug: bool


@dataclass
class SiglipBagSettings:
    lambda_start: float
    lambda_end: float
    start_epoch: int
    warmup_epochs: int
    reduce: str
    topk: int
    loss_type: str
    huber_delta: float
    targets_sum: Dict[str, float] = field(default_factory=dict)
    targets_mean: Dict[str, float] = field(default_factory=dict)
    lambda_by_severity: Dict[str, float] = field(default_factory=dict)


@dataclass
class SiglipRetrievalSettings:
    fp16: bool
    use_logit_bias_eval: bool
    logit_bias_scale_eval: float
    use_textbank_cache: bool
    textbank_cache_dir: str


@dataclass
class SiglipRuntimeSettings:
    eps: float
    abnormal_margin: float
    negative_weight: float
    infonce_weight: float
    focal_infonce: bool
    focal_gamma_pos: float
    focal_gamma_neg: float
    focal_alpha_default: float
    focal_alpha_clip_min: float
    focal_alpha_clip_max: float
    focal_detach_weights: bool
    hard_neg_topk: int
    hard_neg_boost: float
    use_weighted_loss: bool
    use_logit_bias_train: bool
    logit_bias_scale_train: float
    phase_default: str
    phase_transition_epoch: Optional[int]
    debug: SiglipDebugSettings
    bag: SiglipBagSettings
    retrieval: SiglipRetrievalSettings

    @classmethod
    def from_config(cls, config: ClipConfig, output_dir: Optional[str]) -> "SiglipRuntimeSettings":
        eps = float(getattr(config, "siglip_loss_eps", 1e-6))
        abnormal_margin = float(getattr(config, "siglip_abnormal_margin", 0.0))
        negative_weight = float(getattr(config, "siglip_negative_weight", 1.0))
        infonce_weight = min(float(getattr(config, "siglip_infonce_weight", 0.25)), 0.5)
        focal_infonce = bool(getattr(config, "siglip_focal_infonce", True))
        focal_gamma_pos = float(getattr(config, "siglip_focal_gamma_pos", 2.0))
        focal_gamma_neg = float(getattr(config, "siglip_focal_gamma_neg", 0.0))
        focal_alpha_default = float(getattr(config, "siglip_focal_alpha_default", 1.0))
        focal_alpha_clip_min = float(getattr(config, "siglip_focal_alpha_clip_min", 0.5))
        focal_alpha_clip_max = float(getattr(config, "siglip_focal_alpha_clip_max", 8.0))
        focal_detach_weights = bool(getattr(config, "siglip_focal_detach_weights", True))
        if focal_alpha_clip_max < focal_alpha_clip_min:
            focal_alpha_clip_max = focal_alpha_clip_min
        hard_neg_topk = int(getattr(config, "siglip_hard_neg_topk", 0))
        hard_neg_boost = float(getattr(config, "siglip_hard_neg_boost", 0.0))
        use_weighted_loss = bool(getattr(config, "siglip_use_weighted_loss", False))
        use_logit_bias_train = bool(getattr(config, "use_logit_bias_train", False))
        logit_bias_scale_train = float(getattr(config, "logit_bias_scale_train", 0.0))

        debug = SiglipDebugSettings(
            batches_per_epoch=max(0, int(getattr(config, "siglip_debug_batch_per_epoch", 0))),
            every=max(0, int(getattr(config, "siglip_debug_every", 0))),
            sample_count=max(0, int(getattr(config, "siglip_debug_sample_count", 0))),
            sync=bool(getattr(config, "siglip_debug_sync", False)),
            barrier_debug=bool(getattr(config, "siglip_barrier_debug", False)),
        )

        default_sum_targets: Dict[str, float] = {
            "normal": 0.0,
            "mild": 0.6,
            "moderate": 1.2,
            "severe": 1.8,
            "critical": 2.0,
            "cto": 2.0,
        }
        config_sum_targets = getattr(config, "siglip_bag_targets", None)
        if isinstance(config_sum_targets, dict):
            for key, value in config_sum_targets.items():
                try:
                    default_sum_targets[_normalize_key(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        default_mean_targets: Dict[str, float] = {
            "normal": 0.02,
            "mild": 0.08,
            "moderate": 0.16,
            "severe": 0.22,
            "critical": 0.26,
            "cto": 0.30,
        }
        config_mean_targets = getattr(config, "siglip_bag_targets_mean", None)
        if isinstance(config_mean_targets, dict):
            for key, value in config_mean_targets.items():
                try:
                    default_mean_targets[_normalize_key(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        default_lambda_by_severity: Dict[str, float] = {
            "normal": 0.0,
            "mild": 0.001,
            "moderate": 0.003,
            "severe": 0.006,
            "critical": 0.008,
            "cto": 0.008,
        }
        config_lambda_map = getattr(config, "siglip_bag_lambda_by_severity", None)
        if isinstance(config_lambda_map, dict):
            for key, value in config_lambda_map.items():
                try:
                    default_lambda_by_severity[_normalize_key(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        bag = SiglipBagSettings(
            lambda_start=float(getattr(config, "siglip_bag_lambda_start", 0.0)),
            lambda_end=float(getattr(config, "siglip_bag_lambda_end", getattr(config, "siglip_bag_lambda", 0.0))),
            start_epoch=int(getattr(config, "siglip_bag_start_epoch", 0)),
            warmup_epochs=int(getattr(config, "siglip_bag_warmup_epochs", 0)),
            reduce=str(getattr(config, "siglip_bag_reduce", "sum")).lower(),
            topk=max(1, int(getattr(config, "siglip_bag_topk", 3))),
            loss_type=str(getattr(config, "siglip_bag_loss_type", "mse")).lower(),
            huber_delta=float(getattr(config, "siglip_bag_huber_delta", 0.25)),
            targets_sum=default_sum_targets,
            targets_mean=default_mean_targets,
            lambda_by_severity=default_lambda_by_severity,
        )

        use_textbank_cache = bool(getattr(config, "use_textbank_cache", True))
        cache_dir = getattr(
            config,
            "textbank_cache_dir",
            os.path.join(output_dir or ".", "textbank_cache"),
        )

        retrieval = SiglipRetrievalSettings(
            fp16=bool(getattr(config, "retrieval_fp16", False)),
            use_logit_bias_eval=bool(getattr(config, "use_logit_bias_eval", False)),
            logit_bias_scale_eval=float(getattr(config, "logit_bias_scale_eval", 0.0)),
            use_textbank_cache=use_textbank_cache,
            textbank_cache_dir=str(cache_dir),
        )

        return cls(
            eps=eps,
            abnormal_margin=abnormal_margin,
            negative_weight=negative_weight,
            infonce_weight=infonce_weight,
            focal_infonce=focal_infonce,
            focal_gamma_pos=focal_gamma_pos,
            focal_gamma_neg=focal_gamma_neg,
            focal_alpha_default=focal_alpha_default,
            focal_alpha_clip_min=focal_alpha_clip_min,
            focal_alpha_clip_max=focal_alpha_clip_max,
            focal_detach_weights=focal_detach_weights,
            hard_neg_topk=hard_neg_topk,
            hard_neg_boost=hard_neg_boost,
            use_weighted_loss=use_weighted_loss,
            use_logit_bias_train=use_logit_bias_train,
            logit_bias_scale_train=logit_bias_scale_train,
            phase_default=str(getattr(config, "siglip_phase_default", "A")).upper(),
            phase_transition_epoch=getattr(config, "siglip_phase_transition_epoch", None),
            debug=debug,
            bag=bag,
            retrieval=retrieval,
        )
