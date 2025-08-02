import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils.registry import ModelRegistry


@ModelRegistry.register("masked_video_modeling")
class MaskedVideoModeling(nn.Module):
    """
    Masked Video Modeling for self-supervised learning.
    
    This module:
    - Randomly masks video tokens
    - Reconstructs masked tokens from visible ones
    - Uses a lightweight decoder for reconstruction
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        decoder_hidden_size: int = 256,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        mask_ratio: float = 0.75,
        mask_token_learnable: bool = True,
        norm_predict_loss: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.mask_ratio = mask_ratio
        self.mask_token_learnable = mask_token_learnable
        self.norm_predict_loss = norm_predict_loss
        
        # Learnable mask token
        if mask_token_learnable:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = torch.zeros(1, 1, hidden_size)
        
        # Decoder for reconstruction
        self.decoder = MaskedVideoDecoder(
            hidden_size=hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
        )
        
        # Prediction head for masked tokens
        self.predict_head = nn.Linear(decoder_hidden_size, hidden_size)
        
        # Layer norm for prediction
        if norm_predict_loss:
            self.predict_norm = nn.LayerNorm(hidden_size)
    
    def random_masking(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on input tokens.
        
        Args:
            x: Input tokens [batch_size, num_tokens, hidden_size]
            mask_ratio: Ratio of tokens to mask
            
        Returns:
            Tuple of (masked_tokens, mask, ids_restore)
        """
        batch_size, num_tokens, hidden_size = x.shape
        device = x.device
        
        # Generate random noise for masking
        noise = torch.rand(batch_size, num_tokens, device=device)
        
        # Sort noise to get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Calculate number of tokens to mask
        len_keep = int(num_tokens * (1 - mask_ratio))
        
        # Keep first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, hidden_size))
        
        # Generate mask for the full sequence
        mask = torch.ones([batch_size, num_tokens], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder with masking.
        
        Args:
            x: Input tokens [batch_size, num_tokens, hidden_size]
            mask_ratio: Ratio of tokens to mask
            
        Returns:
            Tuple of (latent, mask, ids_restore)
        """
        # Apply random masking
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Return masked tokens as latent representation
        # In a full implementation, this would pass through an encoder
        # For now, we'll use the masked tokens directly
        latent = x_masked
        
        return latent, mask, ids_restore
    
    def forward_decoder(
        self, 
        latent: torch.Tensor, 
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the decoder to reconstruct full sequence.
        
        Args:
            latent: Latent representation [batch_size, num_visible, hidden_size]
            ids_restore: Indices to restore full sequence
            
        Returns:
            Reconstructed tokens [batch_size, num_tokens, decoder_hidden_size]
        """
        batch_size, num_visible, hidden_size = latent.shape
        num_tokens = ids_restore.shape[1]
        device = latent.device
        
        # Expand mask tokens to full sequence
        mask_tokens = self.mask_token.expand(batch_size, num_tokens - num_visible, -1)
        
        # Concatenate visible tokens and mask tokens
        x_full = torch.cat([latent, mask_tokens], dim=1)
        
        # Unshuffle to get original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_size))
        
        # Pass through decoder
        decoded = self.decoder(x_full)
        
        return decoded
    
    def forward_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for masked tokens.
        
        Args:
            pred: Predicted tokens [batch_size, num_tokens, hidden_size]
            target: Target tokens [batch_size, num_tokens, hidden_size]
            mask: Mask indicating which tokens were masked [batch_size, num_tokens]
            
        Returns:
            Reconstruction loss
        """
        # Apply prediction head
        pred = self.predict_head(pred)
        
        # Apply normalization if enabled
        if self.norm_predict_loss:
            pred = self.predict_norm(pred)
        
        # Compute loss only on masked tokens
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over hidden dimension
        
        # Apply mask to get loss only on masked tokens
        mask = mask.bool()
        loss = loss[mask].mean()
        
        return loss
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask_ratio: Optional[float] = None
    ) -> dict:
        """
        Forward pass of masked video modeling.
        
        Args:
            x: Input tokens [batch_size, num_tokens, hidden_size]
            mask_ratio: Ratio of tokens to mask (uses self.mask_ratio if None)
            
        Returns:
            dict: Contains loss and other outputs
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Forward through encoder
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        
        # Forward through decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Apply prediction head to project to hidden_size
        pred_projected = self.predict_head(pred)
        if self.norm_predict_loss:
            pred_projected = self.predict_norm(pred_projected)
        
        # Compute loss
        loss = self.forward_loss(pred, x, mask)
        
        return {
            "loss": loss,
            "pred": pred_projected,  # Return projected predictions
            "mask": mask,
            "latent": latent,
        }


class MaskedVideoDecoder(nn.Module):
    """
    Decoder for masked video modeling.
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        decoder_hidden_size: int = 256,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(hidden_size, decoder_hidden_size)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(
                hidden_size=decoder_hidden_size,
                num_heads=decoder_heads,
                dropout=dropout,
            ) for _ in range(decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(decoder_hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x: Input tokens [batch_size, num_tokens, hidden_size]
            
        Returns:
            Decoded tokens [batch_size, num_tokens, decoder_hidden_size]
        """
        # Project input
        x = self.input_proj(x)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        
        # Final projection and normalization
        x = self.output_proj(x)
        x = self.layer_norm(x)
        
        return x


class DecoderBlock(nn.Module):
    """
    Single decoder block for masked video modeling.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder block.
        
        Args:
            x: Input tokens [batch_size, num_tokens, hidden_size]
            
        Returns:
            Output tokens [batch_size, num_tokens, hidden_size]
        """
        # Self-attention with residual connection
        residual = x
        x = self.self_attention_norm(x)
        attn_out, _ = self.self_attention(x, x, x)
        x = residual + attn_out
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x