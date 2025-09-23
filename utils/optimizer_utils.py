"""Advanced optimizer utilities for two-optimizer setup with LLRD and phased training."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class PhaseConfig:
    """Configuration for training phases."""
    name: str
    epochs: int
    text_freeze_layers: Optional[int] = None  # None means all frozen, -1 means all unfrozen
    video_freeze_ratio: float = 0.0
    logit_scale_trainable: bool = False
    text_lr_multiplier: float = 1.0
    video_lr_multiplier: float = 1.0


class LayerwiseLRDecay:
    """Implements layer-wise learning rate decay for transformers."""
    
    def __init__(
        self,
        base_lr: float,
        decay_factor: float = 0.8,
        num_layers: int = 12,
    ):
        """
        Args:
            base_lr: Base learning rate for the top layer
            decay_factor: Multiplicative factor for each layer down
            num_layers: Total number of layers
        """
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.num_layers = num_layers
    
    def get_layer_lr(self, layer_idx: int) -> float:
        """Get learning rate for a specific layer."""
        # Top layer (last) gets base_lr, bottom layer (first) gets most decay
        return self.base_lr * (self.decay_factor ** (self.num_layers - 1 - layer_idx))
    
    def get_param_groups_bert(
        self,
        model: nn.Module,
        base_lr: float,
        weight_decay: float = 0.01,
        exclude_from_wd: List[str] = ["bias", "LayerNorm", "layernorm"],
    ) -> List[Dict[str, Any]]:
        """Create parameter groups for BERT-like models with LLRD.
        
        Args:
            model: The BERT model
            base_lr: Base learning rate for top layer
            weight_decay: Weight decay value
            exclude_from_wd: Parameter names to exclude from weight decay
            
        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []
        
        # Embeddings get the most decay
        embedding_params = {"params": [], "lr": self.get_layer_lr(0), "name": "embeddings"}
        embedding_params_no_wd = {"params": [], "lr": self.get_layer_lr(0), "name": "embeddings_no_wd", "weight_decay": 0.0}
        
        # Encoder layers
        layer_groups = {}
        layer_groups_no_wd = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Check if should exclude from weight decay
            exclude = any(nd in name for nd in exclude_from_wd)
            
            if "embeddings" in name:
                if exclude:
                    embedding_params_no_wd["params"].append(param)
                else:
                    embedding_params["params"].append(param)
            elif "encoder.layer" in name:
                # Extract layer number
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                lr = self.get_layer_lr(layer_num + 1)  # +1 because embeddings is layer 0
                
                if layer_num not in layer_groups:
                    layer_groups[layer_num] = {"params": [], "lr": lr, "name": f"layer_{layer_num}"}
                    layer_groups_no_wd[layer_num] = {"params": [], "lr": lr, "name": f"layer_{layer_num}_no_wd", "weight_decay": 0.0}
                
                if exclude:
                    layer_groups_no_wd[layer_num]["params"].append(param)
                else:
                    layer_groups[layer_num]["params"].append(param)
            elif "pooler" in name:
                # Pooler gets same LR as top layer
                lr = self.base_lr
                if exclude:
                    param_groups.append({"params": [param], "lr": lr, "name": f"pooler_{name}", "weight_decay": 0.0})
                else:
                    param_groups.append({"params": [param], "lr": lr, "name": f"pooler_{name}", "weight_decay": weight_decay})
        
        # Add all groups
        if embedding_params["params"]:
            embedding_params["weight_decay"] = weight_decay
            param_groups.append(embedding_params)
        if embedding_params_no_wd["params"]:
            param_groups.append(embedding_params_no_wd)
            
        for layer_num in sorted(layer_groups.keys()):
            if layer_groups[layer_num]["params"]:
                layer_groups[layer_num]["weight_decay"] = weight_decay
                param_groups.append(layer_groups[layer_num])
            if layer_groups_no_wd[layer_num]["params"]:
                param_groups.append(layer_groups_no_wd[layer_num])
        
        return param_groups
    
    def get_param_groups_mvit(
        self,
        model: nn.Module,
        base_lr: float,
        weight_decay: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Create parameter groups for MViT with LLRD.
        
        Args:
            model: The MViT model
            base_lr: Base learning rate for top layer
            weight_decay: Weight decay value
            
        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []
        
        # Count total blocks
        num_blocks = len(model.blocks) if hasattr(model, 'blocks') else 16
        
        # Patch embedding and positional encoding
        early_params = {"params": [], "lr": self.get_layer_lr(0), "name": "early_layers", "weight_decay": weight_decay}
        
        # Transformer blocks
        block_groups = {}
        
        # Final layers
        final_params = {"params": [], "lr": self.base_lr, "name": "final_layers", "weight_decay": weight_decay}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "conv_proj" in name or "pos_encoding" in name:
                early_params["params"].append(param)
            elif "blocks" in name:
                # Extract block number
                block_num = int(name.split("blocks.")[1].split(".")[0])
                # Map block number to layer index (0 to num_layers-1)
                layer_idx = int((block_num / num_blocks) * self.num_layers)
                lr = self.get_layer_lr(layer_idx)
                
                if block_num not in block_groups:
                    block_groups[block_num] = {
                        "params": [], 
                        "lr": lr, 
                        "name": f"block_{block_num}",
                        "weight_decay": weight_decay
                    }
                block_groups[block_num]["params"].append(param)
            else:
                # Final norm, head, etc.
                final_params["params"].append(param)
        
        # Add all groups
        if early_params["params"]:
            param_groups.append(early_params)
        
        for block_num in sorted(block_groups.keys()):
            if block_groups[block_num]["params"]:
                param_groups.append(block_groups[block_num])
        
        if final_params["params"]:
            param_groups.append(final_params)
        
        return param_groups


def create_two_optimizer_setup(
    video_encoder: nn.Module,
    text_encoder: nn.Module,
    video_proj: Optional[nn.Module] = None,
    text_proj: Optional[nn.Module] = None,
    logit_scale: Optional[nn.Parameter] = None,
    config: Any = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Create two separate optimizers for video and text encoders with LLRD.
    
    Args:
        video_encoder: Video encoder model (MViT)
        text_encoder: Text encoder model (PubMedBERT)
        video_proj: Video projection head (optional)
        text_proj: Text projection head (optional)
        logit_scale: Temperature parameter (optional)
        config: Configuration object
        
    Returns:
        Tuple of (video_optimizer, text_optimizer)
    """
    # Text encoder optimizer with LLRD
    text_llrd = LayerwiseLRDecay(
        base_lr=config.text_lr if hasattr(config, 'text_lr') else 1e-5,
        decay_factor=config.text_llrd_factor if hasattr(config, 'text_llrd_factor') else 0.8,
        num_layers=12,  # BERT has 12 layers
    )
    
    # Get BERT parameter groups with LLRD
    text_param_groups = text_llrd.get_param_groups_bert(
        text_encoder,
        base_lr=config.text_lr if hasattr(config, 'text_lr') else 1e-5,
        weight_decay=config.text_weight_decay if hasattr(config, 'text_weight_decay') else 0.01,
    )
    
    # Add text projection head if provided
    if text_proj is not None:
        text_param_groups.append({
            "params": text_proj.parameters(),
            "lr": config.proj_lr if hasattr(config, 'proj_lr') else 1e-3,
            "name": "text_projection",
            "weight_decay": 0.05,
        })
    
    # Create text optimizer
    text_optimizer = torch.optim.AdamW(
        text_param_groups,
        lr=1e-5,  # Default LR (will be overridden by param groups)
        betas=(0.9, 0.98),  # Different betas for text
        eps=1e-6,
    )
    
    # Video encoder optimizer with LLRD
    video_llrd = LayerwiseLRDecay(
        base_lr=config.video_lr if hasattr(config, 'video_lr') else 2e-4,
        decay_factor=config.video_llrd_factor if hasattr(config, 'video_llrd_factor') else 0.8,
        num_layers=16,  # MViT has ~16 blocks
    )
    
    # Get MViT parameter groups with LLRD
    video_param_groups = video_llrd.get_param_groups_mvit(
        video_encoder,
        base_lr=config.video_lr if hasattr(config, 'video_lr') else 2e-4,
        weight_decay=config.video_weight_decay if hasattr(config, 'video_weight_decay') else 0.05,
    )
    
    # Track which parameters have been added
    added_params = set()
    for group in video_param_groups:
        for param in group["params"]:
            added_params.add(id(param))
    
    # Add video projection head if provided
    if video_proj is not None:
        video_param_groups.append({
            "params": video_proj.parameters(),
            "lr": config.proj_lr if hasattr(config, 'proj_lr') else 1e-3,
            "name": "video_projection",
            "weight_decay": 0.05,
        })
    
    # Add aggregator with higher LR if it exists and not already added
    if hasattr(video_encoder, 'aggregator'):
        aggregator_params = []
        for param in video_encoder.aggregator.parameters():
            if id(param) not in added_params:
                aggregator_params.append(param)
                added_params.add(id(param))
        if aggregator_params:
            video_param_groups.append({
                "params": aggregator_params,
                "lr": (config.video_lr if hasattr(config, 'video_lr') else 2e-4) * 2.0,
                "name": "video_aggregator",
                "weight_decay": 0.05,
            })
    
    # Add attention pool if it exists and not already added
    if hasattr(video_encoder, 'attention_pool') and video_encoder.attention_pool is not None:
        attention_params = []
        for param in video_encoder.attention_pool.parameters():
            if id(param) not in added_params:
                attention_params.append(param)
                added_params.add(id(param))
        if attention_params:
            video_param_groups.append({
                "params": attention_params,
                "lr": (config.video_lr if hasattr(config, 'video_lr') else 2e-4) * 2.0,
                "name": "attention_pool",
                "weight_decay": 0.05,
            })
    
    # Add logit scale to video optimizer if provided
    if logit_scale is not None:
        video_param_groups.append({
            "params": [logit_scale],
            "lr": config.logit_scale_lr if hasattr(config, 'logit_scale_lr') else 0.01,
            "name": "logit_scale",
            "weight_decay": 0.0,
        })
    
    # Create video optimizer
    video_optimizer = torch.optim.AdamW(
        video_param_groups,
        lr=2e-4,  # Default LR (will be overridden by param groups)
        betas=(0.9, 0.999),  # Standard betas for video
        eps=1e-8,
    )
    
    return video_optimizer, text_optimizer


class PhasedTrainingScheduler:
    """Manages phased training with different freezing strategies."""
    
    def __init__(
        self,
        phases: List[PhaseConfig],
        total_steps: int,
        video_encoder: nn.Module,
        text_encoder: nn.Module,
        video_optimizer: torch.optim.Optimizer,
        text_optimizer: torch.optim.Optimizer,
        logit_scale: Optional[nn.Parameter] = None,
    ):
        """
        Args:
            phases: List of phase configurations
            total_steps: Total training steps
            video_encoder: Video encoder model
            text_encoder: Text encoder model
            video_optimizer: Video optimizer
            text_optimizer: Text optimizer
            logit_scale: Temperature parameter
        """
        self.phases = phases
        self.total_steps = total_steps
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.video_optimizer = video_optimizer
        self.text_optimizer = text_optimizer
        self.logit_scale = logit_scale
        
        # Calculate epoch boundaries for phases
        self.phase_epochs = []
        cumulative_epochs = 0
        for phase in phases:
            self.phase_epochs.append((cumulative_epochs, cumulative_epochs + phase.epochs, phase))
            cumulative_epochs += phase.epochs
        self.total_epochs = cumulative_epochs
    
    def get_current_phase(self, epoch: int) -> PhaseConfig:
        """Get the current training phase based on epoch."""
        for start, end, phase in self.phase_epochs:
            if start <= epoch < end:
                return phase
        return self.phases[-1]  # Return last phase if beyond
    
    def update_for_epoch(self, epoch: int) -> Dict[str, Any]:
        """Update model freezing and learning rates for current epoch.
        
        Returns:
            Dict with phase info for logging
        """
        phase = self.get_current_phase(epoch)
        
        # Update text encoder freezing
        if hasattr(self.text_encoder, 'module'):
            text_model = self.text_encoder.module
        else:
            text_model = self.text_encoder
        
        if phase.text_freeze_layers is None:
            # Freeze all text encoder
            for param in text_model.parameters():
                param.requires_grad = False
        elif phase.text_freeze_layers == -1:
            # Unfreeze all text encoder
            for param in text_model.parameters():
                param.requires_grad = True
        else:
            # Freeze bottom layers, unfreeze top layers
            if hasattr(text_model, 'bert'):
                bert = text_model.bert
                # Freeze embeddings
                for param in bert.embeddings.parameters():
                    param.requires_grad = False
                # Handle encoder layers
                num_layers = len(bert.encoder.layer)
                freeze_until = num_layers - phase.text_freeze_layers
                for i, layer in enumerate(bert.encoder.layer):
                    for param in layer.parameters():
                        param.requires_grad = (i >= freeze_until)
        
        # Update video encoder freezing
        if hasattr(self.video_encoder, 'update_freeze_ratio'):
            self.video_encoder.update_freeze_ratio(phase.video_freeze_ratio)
        
        # Update logit scale trainability
        if self.logit_scale is not None:
            self.logit_scale.requires_grad = phase.logit_scale_trainable
        
        # Update learning rates
        for param_group in self.text_optimizer.param_groups:
            param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * phase.text_lr_multiplier
        
        for param_group in self.video_optimizer.param_groups:
            param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * phase.video_lr_multiplier
        
        return {
            "phase_name": phase.name,
            "text_freeze_layers": phase.text_freeze_layers,
            "video_freeze_ratio": phase.video_freeze_ratio,
            "logit_scale_trainable": phase.logit_scale_trainable,
            "text_lr_mult": phase.text_lr_multiplier,
            "video_lr_mult": phase.video_lr_multiplier,
        }


def initialize_logit_scale(
    initial_value: float = None,
    device: torch.device = None,
) -> nn.Parameter:
    """Initialize logit scale (temperature) parameter.
    
    Args:
        initial_value: Initial temperature value (default: 1/0.07)
        device: Device to create parameter on
        
    Returns:
        Logit scale parameter
    """
    if initial_value is None:
        # ln(1/0.07) ≈ 2.659
        initial_value = math.log(1.0 / 0.07)
    
    logit_scale = nn.Parameter(
        torch.tensor([initial_value], dtype=torch.float32, device=device)
    )
    
    return logit_scale


def clamp_logit_scale(
    logit_scale: nn.Parameter,
    min_value: float = 0.0,
    max_value: float = None,
) -> None:
    """Clamp logit scale to prevent instability.
    
    Args:
        logit_scale: Logit scale parameter
        min_value: Minimum value (default: 0.0)
        max_value: Maximum value (default: ln(50) ≈ 3.912)
    """
    if max_value is None:
        max_value = math.log(50.0)  # ln(50) ≈ 3.912
    
    with torch.no_grad():
        logit_scale.clamp_(min_value, max_value)