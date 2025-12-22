"""
LocCa (Localized Captions) Loss for SigLIP 2 Text Generation Tasks.

LocCa trains a decoder with cross-attention on three objectives:
1. Image/Video Captioning - Standard next-token prediction
2. Referring Expression Prediction - Generate text describing a specific region
3. Grounded Captioning - Generate captions with location tokens

Reference: https://arxiv.org/abs/2502.14786 (SigLIP 2 paper)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.enums import LossType
from utils.registry import LossRegistry


@LossRegistry.register(LossType.LOCCA_CAPTION)
class LocCaCaptioningLoss(nn.Module):
    """
    Standard captioning loss for LocCa decoder.

    This is a cross-entropy loss for next-token prediction, with support for:
    - Label smoothing
    - Ignoring padding tokens
    - Per-token weighting
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Initialize captioning loss.

        Args:
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            ignore_index: Index to ignore in loss computation (typically padding)
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute captioning loss.

        Args:
            logits: [B, L, V] predicted logits from decoder
            targets: [B, L] target token IDs (shifted right from input)
            attention_mask: [B, L] optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Scalar loss value
        """
        B, L, V = logits.shape

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        # Apply attention mask by setting masked positions to ignore_index
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            targets_flat = targets_flat.clone()
            targets_flat[mask_flat == 0] = self.ignore_index

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        return loss


@LossRegistry.register(LossType.LOCCA_REFERRING)
class LocCaReferringExpressionLoss(nn.Module):
    """
    Loss for referring expression prediction.

    Given a region (bounding box), predict the text that describes that region.
    This is similar to captioning but conditioned on a specific region.

    The region information is typically injected by:
    1. Adding region tokens to the vision features
    2. Using region embeddings as additional conditioning
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        region_weight: float = 1.5,
    ):
        """
        Initialize referring expression loss.

        Args:
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation
            region_weight: Extra weight for region-specific tokens
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.region_weight = region_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        region_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute referring expression loss.

        Args:
            logits: [B, L, V] predicted logits from decoder
            targets: [B, L] target token IDs
            attention_mask: [B, L] optional mask (1 = compute loss, 0 = ignore)
            region_token_mask: [B, L] mask indicating region-specific tokens
                              (these get extra weight)

        Returns:
            Scalar loss value
        """
        B, L, V = logits.shape

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        # Apply attention mask
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            targets_flat = targets_flat.clone()
            targets_flat[mask_flat == 0] = self.ignore_index

        # Compute per-token weights
        weights = torch.ones(B * L, device=logits.device)
        if region_token_mask is not None:
            region_mask_flat = region_token_mask.view(-1)
            weights = torch.where(
                region_mask_flat > 0,
                torch.full_like(weights, self.region_weight),
                weights,
            )

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Apply weights and compute mean over valid tokens
        valid_mask = targets_flat != self.ignore_index
        if valid_mask.sum() > 0:
            loss = (loss * weights * valid_mask).sum() / (weights * valid_mask).sum()
        else:
            loss = loss.mean()

        return loss


@LossRegistry.register(LossType.LOCCA_GROUNDED)
class LocCaGroundedCaptioningLoss(nn.Module):
    """
    Loss for grounded captioning.

    Generate captions that include location tokens (e.g., <loc_123>, <box_456>)
    indicating where objects are in the image/video.

    This loss applies extra weight to location tokens to ensure the model
    learns to ground objects properly.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        location_weight: float = 2.0,
        location_token_ids: list[int] | None = None,
    ):
        """
        Initialize grounded captioning loss.

        Args:
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation
            location_weight: Extra weight for location tokens
            location_token_ids: List of token IDs that represent locations.
                               If None, location_mask must be provided at forward.
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.location_weight = location_weight
        self.location_token_ids = set(location_token_ids) if location_token_ids else None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        location_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute grounded captioning loss.

        Args:
            logits: [B, L, V] predicted logits from decoder
            targets: [B, L] target token IDs
            attention_mask: [B, L] optional mask (1 = compute loss, 0 = ignore)
            location_mask: [B, L] mask indicating location tokens (1 = location)
                          If None and location_token_ids is set, will be computed.

        Returns:
            Scalar loss value
        """
        B, L, V = logits.shape

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        # Apply attention mask
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            targets_flat = targets_flat.clone()
            targets_flat[mask_flat == 0] = self.ignore_index

        # Compute location mask if not provided
        if location_mask is None and self.location_token_ids is not None:
            location_mask = torch.zeros_like(targets)
            for token_id in self.location_token_ids:
                location_mask = location_mask | (targets == token_id)

        # Compute per-token weights
        weights = torch.ones(B * L, device=logits.device)
        if location_mask is not None:
            location_mask_flat = location_mask.view(-1)
            weights = torch.where(
                location_mask_flat > 0,
                torch.full_like(weights, self.location_weight),
                weights,
            )

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Apply weights and compute mean over valid tokens
        valid_mask = targets_flat != self.ignore_index
        if valid_mask.sum() > 0:
            loss = (loss * weights * valid_mask).sum() / (weights * valid_mask).sum()
        else:
            loss = loss.mean()

        return loss


@LossRegistry.register(LossType.LOCCA_COMBINED)
class LocCaCombinedLoss(nn.Module):
    """
    Combined LocCa loss for multi-task training.

    This combines captioning, referring expression, and grounded captioning losses
    with configurable weights, similar to how SigLIP 2 trains LocCa.
    """

    def __init__(
        self,
        caption_weight: float = 1.0,
        referring_weight: float = 1.0,
        grounded_weight: float = 1.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Initialize combined LocCa loss.

        Args:
            caption_weight: Weight for captioning loss
            referring_weight: Weight for referring expression loss
            grounded_weight: Weight for grounded captioning loss
            label_smoothing: Label smoothing factor for all losses
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.caption_weight = caption_weight
        self.referring_weight = referring_weight
        self.grounded_weight = grounded_weight

        self.caption_loss = LocCaCaptioningLoss(label_smoothing, ignore_index)
        self.referring_loss = LocCaReferringExpressionLoss(label_smoothing, ignore_index)
        self.grounded_loss = LocCaGroundedCaptioningLoss(label_smoothing, ignore_index)

    def forward(
        self,
        caption_logits: Optional[torch.Tensor] = None,
        caption_targets: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None,
        referring_logits: Optional[torch.Tensor] = None,
        referring_targets: Optional[torch.Tensor] = None,
        referring_mask: Optional[torch.Tensor] = None,
        region_token_mask: Optional[torch.Tensor] = None,
        grounded_logits: Optional[torch.Tensor] = None,
        grounded_targets: Optional[torch.Tensor] = None,
        grounded_mask: Optional[torch.Tensor] = None,
        location_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined LocCa loss.

        Args:
            caption_logits: Logits for captioning task
            caption_targets: Targets for captioning task
            caption_mask: Attention mask for captioning
            referring_logits: Logits for referring expression task
            referring_targets: Targets for referring expression task
            referring_mask: Attention mask for referring expression
            region_token_mask: Mask for region-specific tokens
            grounded_logits: Logits for grounded captioning task
            grounded_targets: Targets for grounded captioning task
            grounded_mask: Attention mask for grounded captioning
            location_mask: Mask for location tokens

        Returns:
            Dictionary with 'total' loss and individual task losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=self._get_device(
            caption_logits, referring_logits, grounded_logits
        ))

        # Captioning loss
        if caption_logits is not None and caption_targets is not None:
            caption_loss = self.caption_loss(caption_logits, caption_targets, caption_mask)
            losses["caption"] = caption_loss
            total_loss = total_loss + self.caption_weight * caption_loss

        # Referring expression loss
        if referring_logits is not None and referring_targets is not None:
            referring_loss = self.referring_loss(
                referring_logits, referring_targets, referring_mask, region_token_mask
            )
            losses["referring"] = referring_loss
            total_loss = total_loss + self.referring_weight * referring_loss

        # Grounded captioning loss
        if grounded_logits is not None and grounded_targets is not None:
            grounded_loss = self.grounded_loss(
                grounded_logits, grounded_targets, grounded_mask, location_mask
            )
            losses["grounded"] = grounded_loss
            total_loss = total_loss + self.grounded_weight * grounded_loss

        losses["total"] = total_loss
        return losses

    @staticmethod
    def _get_device(*tensors) -> torch.device:
        """Get device from first non-None tensor."""
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")


class SigLIP2CombinedLoss(nn.Module):
    """
    Full SigLIP 2 loss combining SigLIP BCE and LocCa losses.

    SigLIP 2 = SigLIP (contrastive) + LocCa (generative)

    This provides a single module for training the complete SigLIP 2 model.
    """

    def __init__(
        self,
        siglip_weight: float = 1.0,
        locca_weight: float = 1.0,
        siglip_bias_init: float = -10.0,
        label_smoothing: float = 0.0,
        caption_weight: float = 1.0,
        referring_weight: float = 0.5,
        grounded_weight: float = 0.5,
    ):
        """
        Initialize combined SigLIP 2 loss.

        Args:
            siglip_weight: Weight for SigLIP contrastive loss
            locca_weight: Weight for LocCa generative loss
            siglip_bias_init: Initial bias for SigLIP loss
            label_smoothing: Label smoothing for all losses
            caption_weight: Weight for captioning within LocCa
            referring_weight: Weight for referring expression within LocCa
            grounded_weight: Weight for grounded captioning within LocCa
        """
        super().__init__()
        self.siglip_weight = siglip_weight
        self.locca_weight = locca_weight

        # Import here to avoid circular imports
        from utils.loss.siglip2_bce import SigLIP2BCELoss

        self.siglip_loss = SigLIP2BCELoss(
            bias_init=siglip_bias_init,
            label_smoothing=label_smoothing,
        )

        self.locca_loss = LocCaCombinedLoss(
            caption_weight=caption_weight,
            referring_weight=referring_weight,
            grounded_weight=grounded_weight,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        # SigLIP inputs
        video_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        log_temp: Optional[torch.Tensor] = None,
        # LocCa inputs
        caption_logits: Optional[torch.Tensor] = None,
        caption_targets: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None,
        referring_logits: Optional[torch.Tensor] = None,
        referring_targets: Optional[torch.Tensor] = None,
        referring_mask: Optional[torch.Tensor] = None,
        region_token_mask: Optional[torch.Tensor] = None,
        grounded_logits: Optional[torch.Tensor] = None,
        grounded_targets: Optional[torch.Tensor] = None,
        grounded_mask: Optional[torch.Tensor] = None,
        location_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined SigLIP 2 loss.

        Returns:
            Dictionary with 'total', 'siglip', and LocCa task losses
        """
        losses = {}

        # Determine device
        device = torch.device("cpu")
        for t in [video_features, text_features, caption_logits]:
            if t is not None:
                device = t.device
                break

        total_loss = torch.tensor(0.0, device=device)

        # SigLIP contrastive loss
        if video_features is not None and text_features is not None and log_temp is not None:
            siglip_loss = self.siglip_loss(video_features, text_features, log_temp)
            losses["siglip"] = siglip_loss
            total_loss = total_loss + self.siglip_weight * siglip_loss

        # LocCa generative losses
        locca_losses = self.locca_loss(
            caption_logits=caption_logits,
            caption_targets=caption_targets,
            caption_mask=caption_mask,
            referring_logits=referring_logits,
            referring_targets=referring_targets,
            referring_mask=referring_mask,
            region_token_mask=region_token_mask,
            grounded_logits=grounded_logits,
            grounded_targets=grounded_targets,
            grounded_mask=grounded_mask,
            location_mask=location_mask,
        )

        # Add LocCa losses to output
        for key, value in locca_losses.items():
            if key != "total":
                losses[f"locca_{key}"] = value

        if "total" in locca_losses and locca_losses["total"] > 0:
            total_loss = total_loss + self.locca_weight * locca_losses["total"]

        losses["total"] = total_loss
        return losses


@LossRegistry.register(LossType.SIGLIP2_COMBINED)
class SigLIP2CombinedLossRegistered(SigLIP2CombinedLoss):
    """Registered version for the loss registry."""
    pass
