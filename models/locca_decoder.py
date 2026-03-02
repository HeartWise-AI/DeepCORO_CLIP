"""
LocCa (Localized Captions) Decoder for SigLIP 2.

LocCa is a lightweight transformer decoder with cross-attention to vision encoder
features. It enables text generation capabilities for:
1. Image/Video captioning
2. Referring expression prediction
3. Grounded captioning

Reference: https://arxiv.org/abs/2502.14786 (SigLIP 2 paper)

The decoder is typically much smaller than the text encoder (fewer layers)
and uses cross-attention to the unpooled vision encoder representation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import ModelRegistry


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer decoder."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor

        Returns:
            [B, L, D] with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LocCaCrossAttention(nn.Module):
    """
    Cross-attention layer for attending to vision encoder features.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        vision_dim: int | None = None,
    ):
        """
        Args:
            d_model: Dimension of the decoder
            num_heads: Number of attention heads
            dropout: Dropout probability
            vision_dim: Dimension of vision features (if different from d_model)
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        vision_dim = vision_dim or d_model

        # Query from decoder, Key and Value from vision encoder
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(vision_dim, d_model)
        self.v_proj = nn.Linear(vision_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        vision_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L_text, D] decoder hidden states
            vision_features: [B, L_vision, D_vision] vision encoder features
            vision_mask: [B, L_vision] optional mask for vision features

        Returns:
            [B, L_text, D] attended features
        """
        B, L_text, _ = x.shape
        L_vision = vision_features.shape[1]

        # Project queries, keys, values
        q = self.q_proj(x).view(B, L_text, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(vision_features).view(B, L_vision, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(vision_features).view(B, L_vision, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply vision mask if provided
        if vision_mask is not None:
            # vision_mask: [B, L_vision] -> [B, 1, 1, L_vision]
            attn_mask = vision_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_text, self.d_model)

        return self.out_proj(attn_output)


class LocCaDecoderLayer(nn.Module):
    """
    Single decoder layer with:
    1. Masked self-attention (causal)
    2. Cross-attention to vision features
    3. Feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        vision_dim: int | None = None,
    ):
        super().__init__()

        # Masked self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention to vision features
        self.cross_attn = LocCaCrossAttention(
            d_model, num_heads, dropout, vision_dim
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        vision_features: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] decoder input
            vision_features: [B, L_v, D_v] vision encoder features
            tgt_mask: [L, L] causal mask for self-attention
            tgt_key_padding_mask: [B, L] padding mask for decoder
            vision_mask: [B, L_v] mask for vision features

        Returns:
            [B, L, D] decoder output
        """
        # Masked self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(x)

        # Cross-attention to vision features
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.cross_attn(x, vision_features, vision_mask))

        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)

        return x


@ModelRegistry.register("locca_decoder")
class LocCaDecoder(nn.Module):
    """
    LocCa Decoder for text generation with cross-attention to vision features.

    This is a lightweight transformer decoder (typically 2-4 layers) that:
    1. Takes embedded text tokens as input
    2. Cross-attends to unpooled vision encoder features
    3. Predicts next tokens for captioning/generation tasks
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        vision_dim: int | None = None,
        pad_token_id: int = 0,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Initialize LocCa Decoder.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            num_layers: Number of decoder layers (typically 2-4, less than text encoder)
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            vision_dim: Dimension of vision features (if different from d_model)
            pad_token_id: ID of padding token
            bos_token_id: ID of beginning-of-sequence token (falls back to eos_token_id for GPT-2)
            eos_token_id: ID of end-of-sequence token (defaults to vocab_size-1 if not provided)
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        # BOS/EOS token handling with GPT-2 fallback
        # GPT-2 uses EOS as BOS (no separate BOS token)
        if eos_token_id is None:
            eos_token_id = vocab_size - 1  # Default fallback
        self.eos_token_id = eos_token_id

        # For GPT-2 compatibility: use EOS as BOS if BOS not specified
        self.bos_token_id = bos_token_id if bos_token_id is not None else eos_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            LocCaDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                vision_dim=vision_dim,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        # NOTE: We do NOT tie embeddings here. Tied embeddings cause an identity
        # mapping issue at initialization where the model copies input tokens
        # instead of learning to predict next tokens. With small init weights,
        # residual connections preserve input embeddings, and tied output projection
        # maps them back to the same tokens (argmax = input token).
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask using PyTorch standard method.

        Uses nn.Transformer.generate_square_subsequent_mask() for consistency
        with PyTorch's transformer implementation. Explicitly uses float32 dtype
        to avoid precision issues with mixed-precision training.
        """
        # Use PyTorch's standard causal mask generation with explicit float32
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device, dtype=torch.float32
        )
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Args:
            input_ids: [B, L] input token IDs
            vision_features: [B, L_v, D_v] unpooled vision encoder features
            attention_mask: [B, L] attention mask for text (1 = attend, 0 = ignore)
            vision_mask: [B, L_v] attention mask for vision features

        Returns:
            [B, L, vocab_size] logits for next token prediction
        """
        B, L = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Generate causal mask
        causal_mask = self._generate_causal_mask(L, input_ids.device)

        # Convert attention mask to key_padding_mask format
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1 = attend, 0 = ignore
            # key_padding_mask: True = ignore, False = attend
            key_padding_mask = attention_mask == 0

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(
                x,
                vision_features,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=key_padding_mask,
                vision_mask=vision_mask,
            )

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        vision_features: torch.Tensor,
        start_token_id: int | None = None,
        end_token_id: int | None = None,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate captions autoregressively.

        Args:
            vision_features: [B, L_v, D_v] vision encoder features
            start_token_id: Token ID to start generation (defaults to self.bos_token_id)
            end_token_id: Token ID that signals end of generation (defaults to self.eos_token_id)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: If set, sample from top-k tokens
            top_p: If set, sample from tokens with cumulative probability < top_p
            vision_mask: [B, L_v] mask for vision features

        Returns:
            [B, L_generated] generated token IDs
        """
        B = vision_features.shape[0]
        device = vision_features.device

        # Use stored token IDs with optional overrides
        start_id = start_token_id if start_token_id is not None else self.bos_token_id
        end_id = end_token_id if end_token_id is not None else self.eos_token_id

        # Start with start token (BOS)
        generated = torch.full((B, 1), start_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Get logits for next token
            logits = self.forward(generated, vision_features, vision_mask=vision_mask)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have ended
            if (next_token == end_id).all():
                break

        return generated
