import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple

from utils.registry import ModelRegistry


@ModelRegistry.register("captioning_decoder")
class CaptioningDecoder(nn.Module):
    """
    LocCa-style transformer decoder for generating structured angiographic reports.
    
    This decoder:
    - Uses causal attention for autoregressive generation
    - Cross-attends to video tokens from the shared encoder
    - Generates structured medical reports
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,  # Default BERT vocab size
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 101,  # [CLS] token
        eos_token_id: int = 102,  # [SEP] token
        use_biomed_tokenizer: bool = True,
        biochem_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Initialize tokenizer
        if use_biomed_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(biochem_model_name)
            # Update vocab size to match tokenizer
            self.vocab_size = self.tokenizer.vocab_size
            # Update special token IDs
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token_id = self.tokenizer.cls_token_id
            self.eos_token_id = self.tokenizer.sep_token_id
        else:
            self.tokenizer = None
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Layer normalization for embeddings
        self.embedding_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Output projection to vocab
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights of the module."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.token_embeddings
    
    def set_input_embeddings(self, value):
        self.token_embeddings = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        Forward pass of the captioning decoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            video_features: Video features from encoder [batch_size, num_tokens, hidden_size]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: Cached key-value pairs for autoregressive generation
            use_cache: Whether to return cached key-value pairs
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            
        Returns:
            dict: Contains logits and optional cached key-value pairs
        """
        batch_size, seq_len = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Create causal attention mask for autoregressive generation
        causal_mask = self._get_causal_mask(seq_len, device=input_ids.device)
        
        # Create attention mask for padding and causal attention
        # For MultiheadAttention, we need either:
        # - 2D mask: [seq_len, seq_len] (same for all batch)
        # - 3D mask: [batch_size * num_heads, seq_len, seq_len]
        # We'll use 2D for simplicity since causal mask is the same for all batches
        
        # Convert attention_mask to key_padding_mask format
        # attention_mask is [batch_size, seq_len] where 1 means valid, 0 means padding
        # key_padding_mask needs True for positions to be masked
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        # causal_mask is already in the right format (2D boolean mask)
        # where True means positions to be masked
        attention_mask = causal_mask
        
        hidden_states = embeddings
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_cache = () if use_cache else None
        
        # Pass through decoder layers
        for i, decoder_layer in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                video_features=video_features,
                past_key_value=past_key_values[i] if (past_key_values is not None and len(past_key_values) > i) else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache += (layer_outputs[1],)
            
            if output_attentions:
                all_self_attentions += (layer_outputs[2],)
                all_cross_attentions += (layer_outputs[3],)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        outputs = {"logits": logits}
        
        if use_cache:
            outputs["past_key_values"] = next_cache
        
        if output_attentions:
            outputs["attentions"] = all_self_attentions
            outputs["cross_attentions"] = all_cross_attentions
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        
        return outputs
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation."""
        # Create mask where True means "should be masked"
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask
    
    def generate(
        self,
        video_features: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate captions autoregressively.
        
        Args:
            video_features: Video features [batch_size, num_tokens, hidden_size]
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            bos_token_id: Beginning-of-sequence token ID
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.bos_token_id
        
        batch_size = video_features.shape[0]
        device = video_features.device
        
        # Initialize with BOS token
        input_ids = torch.full((batch_size, 1), bos_token_id, device=device, dtype=torch.long)
        
        # Generate tokens autoregressively
        # Note: Caching is disabled since MultiheadAttention doesn't support it
        for _ in range(max_length - 1):
            outputs = self.forward(
                input_ids=input_ids,
                video_features=video_features,
                use_cache=False  # Disabled since not properly implemented
            )
            
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for j in range(input_ids.shape[1]):
                        next_token_logits[i, input_ids[i, j]] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < top_k_values[..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Ensure next_tokens is a tensor
            if not isinstance(next_tokens, torch.Tensor):
                next_tokens = torch.tensor(next_tokens, device=device, dtype=torch.long)
            if next_tokens.dim() == 0:
                next_tokens = next_tokens.unsqueeze(0)
            
            # Append new tokens
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Check if all sequences have reached EOS
            if (input_ids == eos_token_id).any(dim=-1).all():
                break
        
        return input_ids


class DecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention to video features."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Cross-attention to video features
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.feed_forward_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple:
        """
        Forward pass of the decoder layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            video_features: Video features for cross-attention [batch_size, num_tokens, hidden_size]
            past_key_value: Cached key-value pairs
            use_cache: Whether to return cached key-value pairs
            output_attentions: Whether to return attention weights
            
        Returns:
            tuple: (hidden_states, past_key_value, self_attentions, cross_attentions)
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attention_layer_norm(hidden_states)
        
        self_attention_outputs = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=output_attentions,
        )
        
        if output_attentions:
            self_attentions = self_attention_outputs[1]
        else:
            self_attentions = None
        
        hidden_states = self.dropout(self_attention_outputs[0])
        hidden_states = residual + hidden_states
        
        # Cross-attention to video features
        if video_features is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)
            
            cross_attention_outputs = self.cross_attention(
                query=hidden_states,
                key=video_features,
                value=video_features,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=output_attentions,
            )
            
            if output_attentions:
                cross_attentions = cross_attention_outputs[1]
            else:
                cross_attentions = None
            
            hidden_states = self.dropout(cross_attention_outputs[0])
            hidden_states = residual + hidden_states
        else:
            cross_attentions = None
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.feed_forward_layer_norm(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            # Note: MultiheadAttention doesn't support cache, so we return None for now
            # A proper implementation would require custom attention with cache support
            outputs += (None,)  # placeholder for key-value cache
        
        if output_attentions:
            outputs += (self_attentions, cross_attentions)
        
        return outputs