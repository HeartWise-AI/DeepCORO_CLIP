"""Multi-Instance Linear Probing model.

This module implements a generic multi-head, multi-instance learning (MIL)
architecture that *aggregates* per-instance embeddings with one of four
pooling modes (mean / max / gated-attention / cls_token) and then feeds the aggregated
representation into one or more *linear heads* (one per task).

The core logic is adapted from a demo script discussed in a previous
conversation and refactored here so that it fits DeepCORO_CLIP's module /
registry system.

⸻

🤔 Why use cls_token over pooling? [IMPLEMENTED]

The cls_token (short for "classification token") is a special learnable embedding used 
in transformer models—especially in architectures like BERT and Vision Transformers (ViT)—
to aggregate information from the entire input sequence or image for classification tasks.

⸻

🔍 In Context of Transformers (e.g., BERT, ViT)
• It is a learnable vector prepended to the input (e.g., text tokens or image patches).
• After transformer layers process the input, the final hidden state corresponding to 
  cls_token is treated as a summary representation of the entire input.
• This token's output is then passed to a classification head (typically a linear layer) 
  to predict labels.

⸻

🖼️ In Vision Transformers

For images:
• The image is split into patches, each patch turned into a token.
• A cls_token is prepended as the first token.
• After passing through the transformer layers, the model uses the final cls_token 
  output to make predictions (e.g., disease presence from a coronary angiogram video).

⸻

🎯 cls_token vs Pooling Strategies (implemented in this module):

cls_token advantages [✅ IMPLEMENTED]:
• Learns how to aggregate global context during training via self-attention
• Can attend to all patches/tokens simultaneously through multi-head attention
• More flexible than fixed pooling (mean/max) as it adapts to the specific task
• Captures complex inter-patch relationships that simple pooling might miss
• Uses learnable Parameter that evolves during training for task-specific aggregation

This module's pooling alternatives:
• Mean pooling ("mean"): Simple average across instances, treats all equally
• Max pooling ("max"): Takes maximum activation per feature dimension
• Attention pooling ("attention"): Gated attention mechanism with learnable weights
  - Similar to cls_token but applied as a separate aggregation step
  - Uses tanh/sigmoid gates to learn which instances are most important
• CLS Token pooling ("cls_token") [✅ NEW]: Learnable token with self-attention
  - Prepends a learnable cls_token to the input sequence
  - Uses PyTorch MultiheadAttention for self-attention computation
  - Supports masking for variable-length sequences
  - Includes layer normalization and dropout for better training stability

The cls_token pooling mode provides the most sophisticated aggregation mechanism,
offering learnable attention patterns that can capture complex relationships
between instances in a multi-instance learning setting.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import ModelRegistry

__all__ = [
    "MultiInstanceLinearProbing",
]


@ModelRegistry.register("multi_instance_linear_probing")
class MultiInstanceLinearProbing(nn.Module):
    """Multi-Instance Linear Probing (MIL-LP).

    Args
    -----
    embedding_dim:  Size of per-instance embeddings.
    head_structure: ``dict`` mapping *head name* to *#classes*.
    pooling_mode:   One of ``"mean" | "max" | "attention" | "cls_token" | "mean+cls_token" | "attention+cls_token"``. 
                    Defaults to ``"mean"``.
    attention_hidden:  Hidden size used inside the gated-attention module when
                       ``pooling_mode == 'attention'``.
    dropout:        Optional dropout *applied only inside the attention block*.
    use_cls_token:  Whether to use learnable cls_token for aggregation. When True,
                    prepends a learnable token that aggregates information through
                    self-attention mechanism.
    num_attention_heads: Number of attention heads for cls_token processing. Defaults to 8.
    separate_video_attention: Whether to use separate attention layers for within-video 
                             and across-video attention in hierarchical cls_token processing.
    normalization_strategy: One of "post_norm" | "pre_norm". Whether to apply layer norm
                           before or after attention. Defaults to "post_norm" (like ViT).
    """

    def __init__(
        self,
        embedding_dim: int,
        head_structure: Dict[str, int],
        pooling_mode: str = "mean",
        attention_hidden: int = 128,
        dropout: float = 0.0,
        use_cls_token: bool = False,
        num_attention_heads: int = 8,
        separate_video_attention: bool = True,
        normalization_strategy: str = "post_norm",
    ) -> None:
        super().__init__()

        valid_pooling_modes = {
            "mean", "max", "attention", "cls_token", 
            "mean+cls_token", "attention+cls_token"
        }
        if pooling_mode not in valid_pooling_modes:
            raise ValueError(
                f"pooling_mode must be one of {valid_pooling_modes}"
            )
        if not head_structure:
            raise ValueError("head_structure cannot be empty")
        if normalization_strategy not in {"pre_norm", "post_norm"}:
            raise ValueError("normalization_strategy must be 'pre_norm' or 'post_norm'")

        self.embedding_dim = embedding_dim
        self.pooling_mode = pooling_mode
        self.head_structure = head_structure
        self.use_cls_token = use_cls_token or "cls_token" in pooling_mode
        self.num_attention_heads = num_attention_heads
        self.separate_video_attention = separate_video_attention
        self.normalization_strategy = normalization_strategy

        # Initialize cls_token if requested
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            
            if self.separate_video_attention:
                # Separate attention layers for within-video and across-video attention
                self.cls_attention_within = nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.cls_attention_across = nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    batch_first=True
                )
            else:
                # Shared attention layer (original implementation)
                self.cls_attention = nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    batch_first=True
                )
            
            # Normalization layers
            if self.separate_video_attention:
                self.cls_norm_within = nn.LayerNorm(embedding_dim)
                self.cls_norm_across = nn.LayerNorm(embedding_dim)
            else:
                self.cls_norm = nn.LayerNorm(embedding_dim)
                
            self.cls_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Attention parameters (if needed for attention pooling, including hybrid modes)
        if "attention" in self.pooling_mode:
            self.attention_V = nn.Linear(embedding_dim, attention_hidden)
            self.attention_U = nn.Linear(embedding_dim, attention_hidden)
            self.attention_w = nn.Linear(attention_hidden, 1)
            self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Create one linear classification / regression head per task.
        # For hybrid pooling, we need to adjust the input dimension
        head_input_dim = embedding_dim
        if "+" in self.pooling_mode:
            head_input_dim = 2 * embedding_dim  # Concatenated features
            
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(head_input_dim, n_classes)
                for name, n_classes in head_structure.items()
            }
        )

        self._reset_parameters()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args
        -----
        x:     ``[B, N, D]`` tensor where *N* is the #instances (views) per
               sample and *D == embedding_dim*.
        mask:  Optional ``[B, N]`` boolean / binary mask indicating valid
               positions. If ``None``, all instances are considered valid.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``head_name -> logits`` with shape ``[B, n_classes]`` for each
            registered head.
        """
        if x.ndim == 4:  # [B, N, L, D]
            B, N, L, D = x.shape  # noqa: N806
            if D != self.embedding_dim:
                raise ValueError(
                    f"Expected embedding_dim={self.embedding_dim} but got {D}"
                )
            # For pooling modes that use mean/max (including hybrid modes), 
            # we need to pool over patches first
            needs_patch_pooling = (
                self.pooling_mode in {"mean", "max"} or 
                ("mean" in self.pooling_mode and "+" in self.pooling_mode) or
                ("max" in self.pooling_mode and "+" in self.pooling_mode)
            )
            if needs_patch_pooling:
                import warnings
                warnings.warn(
                    f"[MultiInstanceLinearProbing] Received 4D input [B, N, L, D] with pooling_mode='{self.pooling_mode}'. "
                    "Automatically mean-pooling over patch tokens (L) to produce [B, N, D]. "
                    "If you want patch-level attention, use pooling_mode='attention' or 'cls_token'."
                )
                x = x.mean(dim=2)  # [B, N, D]
        elif x.ndim == 3:  # [B, N, D]
            B, N, D = x.shape  # noqa: N806
            if D != self.embedding_dim:
                raise ValueError(
                    f"Expected embedding_dim={self.embedding_dim} but got {D}"
                )
        else:
            raise ValueError(
                f"Unsupported input shape {x.shape}; expected 3-D or 4-D tensor."
            )

        pooled = self._pool_instances(x, mask)  # [B, D]
        return {name: head(pooled) for name, head in self.heads.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pool_instances(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Aggregate over the *instance* dimension using the configured rule.
        
        Args:
            x: Input tensor of shape [B, N, D] where N is the number of instances
            mask: Optional boolean mask of shape [B, N] indicating valid instances
            
        Returns:
            Tensor of shape [B, D] or [B, 2*D] for hybrid pooling containing aggregated features
        """
        if mask is None:
            # If no mask provided, treat all instances as valid
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        
        # Ensure mask has correct shape and type
        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match input shape {x.shape[:2]}"
            )
        if mask.dtype != torch.bool:
            mask = mask.bool()
            
        # Handle empty sequences (no valid instances)
        if not mask.any():
            output_dim = 2 * x.shape[2] if "+" in self.pooling_mode else x.shape[2]
            return torch.zeros(
                x.shape[0], output_dim, 
                dtype=x.dtype, 
                device=x.device
            )
            
        # Handle hybrid pooling modes
        if "+" in self.pooling_mode:
            return self._hybrid_pooling(x, mask)
            
        if self.pooling_mode == "cls_token":
            return self._cls_token_pooling(x, mask)
            
        elif self.pooling_mode == "mean":
            return self._mean_pooling(x, mask)
            
        elif self.pooling_mode == "max":
            return self._max_pooling(x, mask)
            
        elif self.pooling_mode == "attention":
            return self._attention_pooling(x, mask)
            
        raise RuntimeError("Invalid pooling_mode – this should never happen")

    def _hybrid_pooling(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Implement hybrid pooling by combining two different pooling methods."""
        method1, method2 = self.pooling_mode.split("+")
        
        # Get features from first method
        if method1 == "mean":
            features1 = self._mean_pooling(x, mask)
        elif method1 == "attention":
            features1 = self._attention_pooling(x, mask)
        else:
            raise ValueError(f"Unsupported hybrid pooling method: {method1}")
            
        # Get features from second method (cls_token)
        if method2 == "cls_token":
            features2 = self._cls_token_pooling(x, mask)
        else:
            raise ValueError(f"Unsupported hybrid pooling method: {method2}")
            
        # Concatenate features
        return torch.cat([features1, features2], dim=-1)

    def _cls_token_pooling(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """CLS Token pooling with hierarchical processing for 4D inputs."""
        if x.ndim == 4:  # [B, N, L, D] - hierarchical case
            return self._hierarchical_cls_token_pooling(x, mask)
        else:  # 3D input [B, N, D] - standard case
            return self._standard_cls_token_pooling(x, mask)

    def _hierarchical_cls_token_pooling(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Hierarchical cls_token pooling for 4D inputs [B, N, L, D]."""
        B, N, L, D = x.shape
        
        # Step 1: Apply cls_token within each video (across L tokens)
        x_reshaped = x.view(B * N, L, D)
        cls_tokens_video = self.cls_token.expand(B * N, -1, -1)
        x_with_cls_video = torch.cat([cls_tokens_video, x_reshaped], dim=1)
        
        # Apply within-video attention
        attention_layer = (self.cls_attention_within if self.separate_video_attention 
                          else self.cls_attention)
        norm_layer = (self.cls_norm_within if self.separate_video_attention 
                     else self.cls_norm)
        
        if self.normalization_strategy == "pre_norm":
            x_with_cls_video = norm_layer(x_with_cls_video)
            
        video_attn_out, _ = attention_layer(
            query=x_with_cls_video,
            key=x_with_cls_video,
            value=x_with_cls_video
        )
        
        if self.normalization_strategy == "post_norm":
            video_attn_out = norm_layer(video_attn_out)
            
        video_representations = video_attn_out[:, 0, :].view(B, N, D)
        video_representations = self.cls_dropout(video_representations)
        
        # Step 2: Apply cls_token across videos (across N video representations)
        cls_tokens_sample = self.cls_token.expand(B, -1, -1)
        x_with_cls_sample = torch.cat([cls_tokens_sample, video_representations], dim=1)
        
        # Handle edge case: all videos masked
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            extended_mask = torch.cat([cls_mask, mask], dim=1)
            
            # Check for samples with all videos masked
            if (~extended_mask[:, 1:]).all(dim=1).any():
                # Fallback: return zeros for samples with no valid videos
                all_masked_samples = (~extended_mask[:, 1:]).all(dim=1)
                fallback_output = torch.zeros(B, D, dtype=x.dtype, device=x.device)
                if all_masked_samples.all():
                    return fallback_output
                    
            key_padding_mask = ~extended_mask
        else:
            key_padding_mask = None
        
        # Apply across-video attention
        attention_layer = (self.cls_attention_across if self.separate_video_attention 
                          else self.cls_attention)
        norm_layer = (self.cls_norm_across if self.separate_video_attention 
                     else self.cls_norm)
        
        if self.normalization_strategy == "pre_norm":
            x_with_cls_sample = norm_layer(x_with_cls_sample)
            
        sample_attn_out, _ = attention_layer(
            query=x_with_cls_sample,
            key=x_with_cls_sample,
            value=x_with_cls_sample,
            key_padding_mask=key_padding_mask
        )
        
        if self.normalization_strategy == "post_norm":
            sample_attn_out = norm_layer(sample_attn_out)
            
        cls_output = sample_attn_out[:, 0, :]
        return self.cls_dropout(cls_output)

    def _standard_cls_token_pooling(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Standard cls_token pooling for 3D inputs [B, N, D]."""
        B, N, D = x.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)
        
        # Handle edge case: all instances masked
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        
        if (~extended_mask[:, 1:]).all(dim=1).any():
            # Fallback for samples with all instances masked
            all_masked_samples = (~extended_mask[:, 1:]).all(dim=1)
            if all_masked_samples.all():
                return torch.zeros(B, D, dtype=x.dtype, device=x.device)
                
        key_padding_mask = ~extended_mask
        
        attention_layer = self.cls_attention
        norm_layer = self.cls_norm
        
        if self.normalization_strategy == "pre_norm":
            x_with_cls = norm_layer(x_with_cls)
            
        attn_out, _ = attention_layer(
            query=x_with_cls,
            key=x_with_cls, 
            value=x_with_cls,
            key_padding_mask=key_padding_mask
        )
        
        if self.normalization_strategy == "post_norm":
            attn_out = norm_layer(attn_out)
            
        cls_output = attn_out[:, 0, :]
        return self.cls_dropout(cls_output)

    def _mean_pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling."""
        mask_f = mask.unsqueeze(-1).float()
        sum_x = (x * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1.0)
        return sum_x / count
        
    def _max_pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked max pooling."""
        x_masked = x.clone()
        x_masked[~mask] = float('-inf')
        return x_masked.max(dim=1)[0]
        
    def _attention_pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Gated attention pooling.""" 
        if x.ndim == 4:  # [B, N, L, D]
            return self._hierarchical_attention_pooling(x, mask)
            
        # Standard 3D attention pooling
        A_V = torch.tanh(self.attention_V(x))
        A_U = torch.sigmoid(self.attention_U(x))
        A = self.attention_w(A_V * A_U)
        
        A = A.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        A = F.softmax(A, dim=1)
        A = self.attn_dropout(A)
        
        return (A * x).sum(dim=1)

    def _hierarchical_attention_pooling(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Hierarchical attention pooling for 4D inputs."""
        B, N, L, D_ = x.shape
        
        # Patch-level attention within each video
        x_patch = x.view(B * N, L, D_)
        A_V_p = torch.tanh(self.attention_V(x_patch))
        A_U_p = torch.sigmoid(self.attention_U(x_patch))
        A_p = self.attention_w(A_V_p * A_U_p)
        A_p = F.softmax(A_p, dim=1)
        A_p = self.attn_dropout(A_p)
        
        video_emb = (A_p * x_patch).sum(dim=1).view(B, N, D_)
        
        # Video-level attention across videos
        A_V = torch.tanh(self.attention_V(video_emb))
        A_U = torch.sigmoid(self.attention_U(video_emb))
        A = self.attention_w(A_V * A_U)
        
        if mask is not None:
            A = A.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            
        A = F.softmax(A, dim=1)
        A = self.attn_dropout(A)
        
        return (A * video_emb).sum(dim=1)

    def _reset_parameters(self):
        """Initialize all parameters (Xavier for Linear layers)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
