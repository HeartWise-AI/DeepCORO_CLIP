import torch.nn as nn
import torch

from typing import Dict
from utils.registry import ModelRegistry
from models.video_encoder import VideoEncoder

"""
🤔 Why use cls_token over pooling?

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

🎯 cls_token vs Pooling Strategies:

cls_token advantages:
• Learns how to aggregate global context during training
• Can attend to all patches/tokens simultaneously through self-attention
• More flexible than fixed pooling (mean/max) as it adapts to the specific task
• Captures complex inter-patch relationships that simple pooling might miss

Pooling alternatives:
• Mean pooling: Simple average, treats all patches equally
• Max pooling: Takes maximum activation, might miss global context
• Attention pooling: Weighted combination, but uses separate attention mechanism

The cls_token approach is particularly effective because the transformer's self-attention
mechanism allows it to dynamically focus on the most relevant parts of the input for
the specific classification task at hand.
"""


class BaseLinearProbingHead(nn.Module):
    """
    Base abstract class for all linear probing heads.
    
    This class defines the common interface for linear probing heads.
    All specific linear probing implementations should inherit from this class.
    
    Note: In transformer-based backbones, the input to these heads typically comes from:
    1. cls_token output: A learnable token that aggregates global context through attention
    2. Pooled representations: Mean/max pooling over spatial/temporal dimensions
    3. Attention pooling: Learned weighted combination of features
    
    The cls_token approach is often preferred as it learns task-specific aggregation patterns.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
    def forward(self, x):
        """Forward pass for the linear probing head."""
        raise NotImplementedError("Subclasses must implement forward method")


@ModelRegistry.register("simple_linear_probing_regression")
class SimpleLinearProbingRegressionHead(BaseLinearProbingHead):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout: float = 0.1
    ):
        """
        Initialize a simple linear probing head for regression tasks.

        Args:
            input_dim (int): The input dimension
            output_dim (int): The output dimension
            dropout (float): The dropout rate
        """        
        super().__init__(input_dim, output_dim, dropout)
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm = nn.LayerNorm(256)  # Add normalization
        self.linear_probe = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.tanh = nn.Tanh()  # Define the tanh activation
        
        # Initialize weights with smaller values
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.linear_probe.weight, gain=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.linear_probe.bias)
        
    def forward(self, x):
        # Handle 2D input [batch_size, input_dim]
        x = self.fc1(x)
        x = self.norm(x)  # Apply normalization
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_probe(x)
        # Scale tanh from [-1,1] to [0,100]
        x = 50 * (self.tanh(x) + 1)
        return x

@ModelRegistry.register("simple_linear_probing")
class SimpleLinearProbingHead(BaseLinearProbingHead):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout: float = 0.1
    ):
        """
        Initialize a simple linear probing head.

        Args:
            input_dim (int): The input dimension
            output_dim (int): The output dimension
            dropout (float): The dropout rate
        """        
        super().__init__(input_dim, output_dim, dropout)
        # Replace Conv3d with Linear layer since input is now [batch_size, input_dim]
        self.fc1 = nn.Linear(input_dim, 256)
        self.linear_probe = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Handle 2D input [batch_size, input_dim]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_probe(x)
        return x


@ModelRegistry.register("cls_token_linear_probing")
class ClsTokenLinearProbingHead(BaseLinearProbingHead):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout: float = 0.1,
        num_heads: int = 8,
        use_cls_token: bool = True
    ):
        """
        Initialize a cls_token-based linear probing head that uses self-attention
        to aggregate features through a learnable classification token.

        Args:
            input_dim (int): The input dimension
            output_dim (int): The output dimension  
            dropout (float): The dropout rate
            num_heads (int): Number of attention heads for cls_token self-attention
            use_cls_token (bool): Whether to use cls_token aggregation (if False, falls back to mean pooling)
        """        
        super().__init__(input_dim, output_dim, dropout)
        
        self.use_cls_token = use_cls_token
        self.num_heads = num_heads
        
        if self.use_cls_token:
            # Initialize learnable cls_token
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
            
            # Self-attention for cls_token processing
            self.cls_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Normalization and dropout layers
            self.cls_norm = nn.LayerNorm(input_dim)
            self.cls_dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm = nn.LayerNorm(256)
        self.linear_probe = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        if self.use_cls_token:
            # Initialize cls_token with small random values
            nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        """
        Forward pass with cls_token aggregation.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            
        Returns:
            Output predictions of shape [batch_size, output_dim]
        """
        # Handle different input shapes
        if x.ndim == 2:  # [batch_size, input_dim]
            # Convert to sequence format for cls_token processing
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
            
        elif x.ndim == 3:  # [batch_size, seq_len, input_dim]
            pass  # Already in correct format
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
            
        if self.use_cls_token:
            B, N, D = x.shape
            
            # Expand cls_token for batch
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            
            # Concatenate cls_token with input embeddings
            x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
            
            # Apply self-attention
            attn_out, _ = self.cls_attention(
                query=x_with_cls,
                key=x_with_cls,
                value=x_with_cls
            )  # [B, N+1, D]
            
            # Extract cls_token output (first token)
            cls_output = attn_out[:, 0, :]  # [B, D]
            
            # Apply normalization and dropout
            x = self.cls_norm(cls_output)
            x = self.cls_dropout(x)
        else:
            # Fallback to mean pooling if cls_token is disabled
            x = x.mean(dim=1)  # [B, D]
        
        # Apply classification head
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_probe(x)
        
        return x


@ModelRegistry.register("cls_token_linear_probing_regression")
class ClsTokenLinearProbingRegressionHead(ClsTokenLinearProbingHead):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout: float = 0.1,
        num_heads: int = 8,
        use_cls_token: bool = True
    ):
        """
        Initialize a cls_token-based linear probing head for regression tasks.
        Inherits from ClsTokenLinearProbingHead but adds regression-specific output scaling.

        Args:
            input_dim (int): The input dimension
            output_dim (int): The output dimension
            dropout (float): The dropout rate
            num_heads (int): Number of attention heads for cls_token self-attention
            use_cls_token (bool): Whether to use cls_token aggregation
        """        
        super().__init__(input_dim, output_dim, dropout, num_heads, use_cls_token)
        self.tanh = nn.Tanh()  # For output scaling
        
    def forward(self, x):
        """Forward pass with regression output scaling."""
        x = super().forward(x)
        # Scale tanh from [-1,1] to [0,100] for regression tasks
        x = 50 * (self.tanh(x) + 1)
        return x


class BaseLinearProbing(nn.Module):
    def __init__(self, backbone: VideoEncoder):
        """
        Initialize a base linear probing model.

        Args:
            backbone (VideoEncoder): The backbone model
        """
        super().__init__()
        self.backbone = backbone


@ModelRegistry.register("DeepCORO_video_linear_probing")
@ModelRegistry.register("DeepCORO_video_linear_probing_test")
class LinearProbing(BaseLinearProbing):
    def __init__(
        self, 
        backbone: VideoEncoder,
        head_linear_probing: Dict[str, str],
        head_structure: Dict[str, int],
        dropout: float = 0.1,
        freeze_backbone_ratio: float = 0.0
    ):
        """
        Initialize a multi-output head with different number of classes per head.

        Args:
            backbone (VideoEncoder): The backbone model
            head_linear_probing (dict[str, str]): Dictionary mapping head names to their type of linear probing head
                e.g. {"contrast_agent": "simple_linear_probing", "main_structure": "simple_linear_probing", "stent_presence": "simple_linear_probing"}
            head_structure (dict[str, int]): Dictionary mapping head names to their number of output classes
                e.g. {"contrast_agent": 1, "main_structure": 5, "stent_presence": 1}
            dropout (float): The dropout rate
            freeze_backbone_ratio (float): Ratio of backbone parameters to freeze
        """        
        super().__init__(backbone)
        
        # Sanity checks
        if not head_structure:
            raise ValueError("head_structure cannot be empty")     
        if not 0.0 <= freeze_backbone_ratio <= 1.0:
            raise ValueError("freeze_backbone_ratio must be between 0.0 and 1.0")
                    
        # Freeze a ratio of parameters
        if freeze_backbone_ratio > 0.0:
            all_params = list(self.backbone.parameters())
            num_params = len(all_params)
            num_frozen = int(freeze_backbone_ratio * num_params)
            
            for param in all_params[:num_frozen]:
                param.requires_grad = False
        
        # Initialize heads
        input_dim: int = int(backbone.aggregator.final_ln.weight.shape[0]) # Get input dimension from the final layer norm of the backbone's aggregator
        self.heads = nn.ModuleDict()
        for head_name, num_classes in head_structure.items():
            if not isinstance(num_classes, int) or num_classes < 1:
                raise ValueError(f"Invalid number of classes for head {head_name}: {num_classes}")
            head_class = ModelRegistry.get(name=head_linear_probing[head_name])
            if head_class is None:
                raise ValueError(f"Unknown linear probing head type: {head_linear_probing[head_name]}")
            self.heads[head_name] = head_class(
                input_dim=input_dim, 
                output_dim=num_classes,
                dropout=dropout
            )

    def forward(self, x):
        x = self.backbone(x)
        return {head_name: head(x) for head_name, head in self.heads.items()}


