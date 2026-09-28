"""EchoJEPA (V-JEPA 2) video ViT as a DeepCORO_CLIP backbone.

EchoJEPA (HeartWise-AI/EchoJEPA) ships its encoder under a top-level ``src``
namespace package (``src.models.vision_transformer``). Rather than vendoring
the model, this module imports it from a checkout of that repository and
wraps it so the DeepCORO_CLIP ``VideoEncoder`` can treat it like any other
backbone: ``forward_features(x)`` returns the full token sequence
``[B, L, D]`` for an input clip ``[B, C, T, H, W]``.

Checkpoints written by EchoJEPA's pretraining loop hold ``encoder`` and
``target_encoder`` state dicts whose keys carry a ``backbone.`` prefix (and a
``module.`` prefix when saved from DDP). ``target_encoder`` (the EMA teacher)
is what EchoJEPA's own evals load, so it is the default here.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# EchoJEPA factory names for the architectures we expose. All EchoJEPA
# encoders released so far use RoPE, so ``use_rope`` is passed explicitly and
# the plain (non ``_rope``) factories are used.
ECHOJEPA_ARCHS: Dict[str, str] = {
    "vit_base": "vit_base",
    "vit_large": "vit_large",
    "vit_huge": "vit_huge",
    "vit_giant": "vit_giant_xformers",
    "vit_giant_xformers": "vit_giant_xformers",
}

_DEFAULT_REPO = os.environ.get("ECHOJEPA_ROOT", "/volume/EchoJEPA")


def _import_vision_transformer(repo_path: str):
    """Import ``src.models.vision_transformer`` from an EchoJEPA checkout."""
    repo_path = os.path.abspath(os.path.expanduser(repo_path))
    if not os.path.isdir(os.path.join(repo_path, "src", "models")):
        raise FileNotFoundError(
            f"EchoJEPA checkout not found at {repo_path} (expected src/models/). "
            "Set echojepa_repo in the config or the ECHOJEPA_ROOT env var."
        )
    # Append (not insert) so DeepCORO_CLIP's own top-level packages (utils,
    # models, ...) keep priority; EchoJEPA only needs its ``src`` namespace.
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    from src.models import vision_transformer as vit  # type: ignore

    return vit


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("backbone.", "")
        out[k] = v
    return out


class EchoJEPABackbone(nn.Module):
    """Thin wrapper around EchoJEPA's ``VisionTransformer``.

    Args:
        checkpoint_path: EchoJEPA pretraining checkpoint (``checkpoint.pt``) or
            ``None`` for a randomly initialised encoder (useful when the
            weights are about to be overwritten by a downstream checkpoint).
        arch: one of ``ECHOJEPA_ARCHS``.
        img_size / num_frames / patch_size / tubelet_size: must match the
            clips the dataloader produces (EchoJEPA: 224px, 16 frames, 16, 2).
        use_rope / uniform_power / use_sdpa: architecture flags; the defaults
            match every released EchoJEPA encoder (``params-pretrain.yaml``).
        use_activation_checkpointing: trade compute for memory inside blocks.
        checkpoint_key: ``target_encoder`` (EMA teacher, default) or ``encoder``.
        repo_path: EchoJEPA checkout to import the model code from.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        arch: str = "vit_large",
        img_size: int = 224,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        use_rope: bool = True,
        uniform_power: bool = True,
        use_sdpa: bool = True,
        use_activation_checkpointing: bool = False,
        checkpoint_key: str = "target_encoder",
        repo_path: str = _DEFAULT_REPO,
    ):
        super().__init__()
        if arch not in ECHOJEPA_ARCHS:
            raise ValueError(f"Unknown EchoJEPA arch '{arch}'. Choose from {sorted(ECHOJEPA_ARCHS)}")
        if num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames ({num_frames}) must be a multiple of tubelet_size ({tubelet_size})")
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be a multiple of patch_size ({patch_size})")

        vit = _import_vision_transformer(repo_path)
        factory = getattr(vit, ECHOJEPA_ARCHS[arch])
        self.encoder = factory(
            patch_size=patch_size,
            img_size=(img_size, img_size),
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_rope=use_rope,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.embed_dim: int = int(self.encoder.embed_dim)
        self.arch = arch
        self.num_frames = num_frames
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path

        if checkpoint_path:
            self.load_echojepa_checkpoint(checkpoint_path, checkpoint_key=checkpoint_key)
        else:
            print("[EchoJEPABackbone] No EchoJEPA checkpoint given; encoder is randomly initialised.")

    # ------------------------------------------------------------------
    def load_echojepa_checkpoint(self, path: str, checkpoint_key: str = "target_encoder") -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"EchoJEPA checkpoint not found: {path}")
        print(f"[EchoJEPABackbone] Loading {checkpoint_key} from {path}")
        # mmap keeps the 5-16 GB file out of RAM until the tensors are copied.
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if checkpoint_key in ckpt:
            sd = ckpt[checkpoint_key]
        elif "encoder" in ckpt:
            print(f"[EchoJEPABackbone] '{checkpoint_key}' missing, falling back to 'encoder'")
            sd = ckpt["encoder"]
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt  # bare state dict
        else:
            raise KeyError(f"No encoder weights found in {path}; keys: {list(ckpt.keys())}")
        sd = _strip_prefixes(sd)
        # The pretraining predictor / EMA bookkeeping never lands in the encoder
        # state dict, but a fixed pos_embed would if the checkpoint was trained
        # without RoPE and this model uses it (or vice-versa): drop mismatches.
        model_sd = self.encoder.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        dropped = sorted(set(sd) - set(filtered))
        missing, unexpected = self.encoder.load_state_dict(filtered, strict=False)
        print(
            f"[EchoJEPABackbone] loaded {len(filtered)}/{len(model_sd)} tensors "
            f"(epoch={ckpt.get('epoch')}); missing={len(missing)} dropped={len(dropped)}"
        )
        if missing:
            print(f"[EchoJEPABackbone]   missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if dropped:
            print(f"[EchoJEPABackbone]   dropped: {dropped[:5]}{' ...' if len(dropped) > 5 else ''}")
        del ckpt

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C, T, H, W]`` -> tokens ``[B, L, D]`` (L = T/2 * H/16 * W/16)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled clip embedding ``[B, D]``."""
        return self.forward_features(x).mean(dim=1)


def echojepa_kwargs_from_config(config: Any, num_frames: int, resize: int) -> Optional[Dict[str, Any]]:
    """Collect the ``echojepa_*`` config fields into ``EchoJEPABackbone`` kwargs.

    Returns ``None`` unless ``config.model_name == "echojepa"`` so the other
    backbones are untouched.
    """
    if getattr(config, "model_name", None) != "echojepa":
        return None
    dtype = str(getattr(config, "echojepa_dtype", "bfloat16") or "float32").lower()
    return {
        "checkpoint_path": getattr(config, "echojepa_checkpoint", None),
        "arch": getattr(config, "echojepa_arch", "vit_large"),
        "img_size": int(resize),
        "num_frames": int(num_frames),
        "patch_size": int(getattr(config, "echojepa_patch_size", 16)),
        "tubelet_size": int(getattr(config, "echojepa_tubelet_size", 2)),
        "use_rope": bool(getattr(config, "echojepa_use_rope", True)),
        "uniform_power": bool(getattr(config, "echojepa_uniform_power", True)),
        "use_activation_checkpointing": bool(getattr(config, "echojepa_activation_checkpointing", False)),
        "checkpoint_key": getattr(config, "echojepa_checkpoint_key", "target_encoder"),
        "repo_path": getattr(config, "echojepa_repo", None) or _DEFAULT_REPO,
        # consumed by VideoEncoder, not by the backbone constructor
        "autocast_dtype": dtype,
    }
