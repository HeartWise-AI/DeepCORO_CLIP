import torch.nn as nn

from typing import Dict
from utils.registry import ModelRegistry
from models.video_encoder import VideoEncoder


class BaseLinearProbingHead(nn.Module):
    """
    Base abstract class for all linear probing heads.
    
    This class defines the common interface for linear probing heads.
    All specific linear probing implementations should inherit from this class.
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
            self.heads[head_name] = nn.Sequential(
                ModelRegistry.get(name=head_linear_probing[head_name])(
                    input_dim=input_dim, 
                    output_dim=num_classes,
                    dropout=dropout
                ), 
            )

    def forward(self, x):
        x = self.backbone(x)
        return {head_name: head(x) for head_name, head in self.heads.items()}


