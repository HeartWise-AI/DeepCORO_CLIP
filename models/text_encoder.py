import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from utils.registry import ModelRegistry


def get_tokenizer(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
    """Get the tokenizer with proper configuration.

    Args:
        model_name (str): Name of the pretrained model

    Returns:
        tokenizer: Configured tokenizer
    """
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=1024,
        padding_side="right",
        truncation_side="right",
    )


@ModelRegistry.register("text_encoder")
class TextEncoder(nn.Module):
    """Text encoder model based on PubMedBERT."""

    def __init__(
        self,
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        output_dim=512,
        dropout=0.2,
        freeze_ratio=0.5,
    ):
        """Initialize the text encoder.

        Args:
            model_name (str): Name of the pretrained model to use
            output_dim (int): Output dimension to match video encoder
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze_ratio = freeze_ratio
        self.dropout = dropout

        self.tokenizer = get_tokenizer(model_name)
        
        # 1) BERT (or other) backbone
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        

        if hasattr(self.bert, "pooler"):
            self.bert.pooler = None

        # 2) Freeze partial layers
        self._freeze_partial_bert()

        # 3) Final projection (using GELU)
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, output_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

    def _freeze_partial_bert(self):
        """
        Freeze the bottom portion of the BERT parameters, leaving
        a fraction `freeze_ratio` trainable from the top.
        """
        if self.freeze_ratio == 0.0:  # If freeze_ratio is 0, don't freeze anything
            return
            
        all_named_params = list(self.bert.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)

        for i, (_, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False
    
    def update_freeze_ratio(self, new_freeze_ratio: float):
        """
        Dynamically update the freeze ratio during training.
        
        Args:
            new_freeze_ratio: New freeze ratio (0.0 = all trainable, 1.0 = all frozen)
        """
        self.freeze_ratio = new_freeze_ratio
        
        # First, unfreeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # Then apply the new freeze ratio
        if self.freeze_ratio == 0.0:  # If freeze_ratio is 0, don't freeze anything
            return
            
        all_named_params = list(self.bert.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)
        
        # Freeze the bottom portion, keep top `train_count` trainable
        for i, (name, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False
        
        # Count trainable parameters for logging
        trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.bert.parameters())
        trainable_percent = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        print(f"[TextEncoder] Updated freeze_ratio to {new_freeze_ratio:.2f}")
        print(f"[TextEncoder] Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_percent:.1f}%)")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the text encoder.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, sequence_length]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Output features of shape [batch_size, output_dim]
        """
        # Get BERT features and take CLS token output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token
        pooled = outputs.last_hidden_state[:, 0]
        # Project to match video encoder dimension
        features = self.proj(pooled)

        return features
