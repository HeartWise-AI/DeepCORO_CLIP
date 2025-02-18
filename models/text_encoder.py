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
        
        # 1) BERT (or other) backbone
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

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
        all_named_params = list(self.bert.named_parameters())
        total_count = len(all_named_params)
        train_count = int(self.freeze_ratio * total_count)

        for i, (_, param) in enumerate(all_named_params):
            if i < (total_count - train_count):
                param.requires_grad = False

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
        features = outputs.last_hidden_state[:, 0]

        # Project to match video encoder dimension
        return self.proj(features)