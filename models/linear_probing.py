import torch.nn as nn

from utils.registry import ModelRegistry
from models.video_encoder import VideoEncoder


@ModelRegistry.register("simple_linear_probing")
class SimpleLinearProbingHead(nn.Module):
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
        super().__init__()
        self.fc1 = nn.Conv3d(input_dim, 256, bias=True, kernel_size=1, stride=1)
        self.linear_probe = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.mean([2, 3, 4])
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
class LinearProbing(BaseLinearProbing):
    def __init__(
        self, 
        backbone: VideoEncoder,
        linear_probing_head: str,
        head_structure: dict[str, int],
        dropout: float = 0.1,
        freeze_backbone_ratio: float = 0.0
    ):
        """
        Initialize a multi-output head with different number of classes per head.

        Args:
            backbone (VideoEncoder): The backbone model
            linear_probing_head (str): The type of linear probing head to use
            head_structure (dict[str, int]): Dictionary mapping head names to their number of output classes
                e.g. {"contrast_agent": 1, "main_structure": 5, "stent_presence": 1}
            dropout (float): The dropout rate
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
                ModelRegistry.get(name=linear_probing_head)(
                    input_dim=input_dim, 
                    output_dim=num_classes,
                    dropout=dropout
                ), 
            )

    def forward(self, x):
        x = self.backbone(x)
        return {head_name: head(x) for head_name, head in self.heads.items()}


