#!/usr/bin/env python3
"""
Example demonstrating cls_token functionality in DeepCORO_CLIP models.

This script shows how to use the cls_token pooling mechanisms
in multi-instance linear probing models.
"""

import torch
import torch.nn as nn
from models.multi_instance_linear_probing import MultiInstanceLinearProbing


def example_multi_instance_cls_token():
    """Demonstrate cls_token usage in multi-instance learning."""
    print("🤔 Multi-Instance CLS Token Example")
    print("=" * 50)
    
    # Example parameters
    batch_size = 3
    num_instances = 5  # Number of videos/instances per sample
    embedding_dim = 256
    
    # Define tasks (multi-head setup)
    head_structure = {
        "contrast_agent": 2,      # Binary classification
        "main_structure": 5,      # Multi-class classification  
        "stent_presence": 1,      # Regression
    }
    
    # Create multi-instance model with cls_token pooling
    mil_cls_model = MultiInstanceLinearProbing(
        embedding_dim=embedding_dim,
        head_structure=head_structure,
        pooling_mode="cls_token",  # Use cls_token pooling
        dropout=0.1,
        use_cls_token=True
    )
    
    # Create input data [batch_size, num_instances, embedding_dim]
    x = torch.randn(batch_size, num_instances, embedding_dim)
    
    # Optional: Create mask for variable-length sequences
    mask = torch.ones(batch_size, num_instances, dtype=torch.bool)
    # Simulate some samples having fewer instances
    mask[1, 4:] = False  # Second sample only has 4 instances
    mask[2, 3:] = False  # Third sample only has 3 instances
    
    print(f"📊 Input shape: {x.shape}")
    print(f"📊 Mask shape: {mask.shape}")
    print(f"📊 Valid instances per sample: {mask.sum(dim=1).tolist()}")
    
    # Forward pass
    outputs = mil_cls_model(x, mask)
    
    print(f"\n🎯 Multi-head outputs:")
    for head_name, output in outputs.items():
        print(f"  {head_name}: shape {output.shape}, sample output: {output[0]}")
    
    # Compare with other pooling modes
    print(f"\n🔄 Comparison with other pooling modes:")
    
    pooling_modes = ["mean", "max", "attention"]
    for mode in pooling_modes:
        model = MultiInstanceLinearProbing(
            embedding_dim=embedding_dim,
            head_structure={"test_head": 2},
            pooling_mode=mode,
            attention_hidden=128,
            dropout=0.1
        )
        
        output = model(x, mask)
        print(f"  {mode:10}: {output['test_head'][0]}")
    
    # Show cls_token parameters
    print(f"\n🎯 CLS Token Model Parameters:")
    print(f"CLS token shape: {mil_cls_model.cls_token.shape}")
    print(f"CLS attention heads: {mil_cls_model.num_attention_heads}")
    print(f"Embedding dimension: {mil_cls_model.embedding_dim}")


def example_cls_token_vs_pooling_comparison():
    """Compare cls_token with traditional pooling methods."""
    print("\n\n🔍 CLS Token vs Pooling Comparison")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    num_instances = 4
    embedding_dim = 128
    
    # Create test data
    x = torch.randn(batch_size, num_instances, embedding_dim)
    
    # Test different approaches
    approaches = {
        "cls_token": {
            "model": MultiInstanceLinearProbing(
                embedding_dim=embedding_dim,
                head_structure={"output": 3},
                pooling_mode="cls_token"
            ),
            "description": "Learnable token with self-attention"
        },
        "attention": {
            "model": MultiInstanceLinearProbing(
                embedding_dim=embedding_dim,
                head_structure={"output": 3},
                pooling_mode="attention"
            ),
            "description": "Gated attention pooling"
        },
        "mean": {
            "model": MultiInstanceLinearProbing(
                embedding_dim=embedding_dim,
                head_structure={"output": 3},
                pooling_mode="mean"
            ),
            "description": "Simple mean pooling"
        }
    }
    
    print(f"📊 Input: {x.shape}")
    print(f"\n🔄 Pooling Results:")
    
    for name, config in approaches.items():
        model = config["model"]
        output = model(x)["output"]
        
        print(f"\n{name.upper():12}")
        print(f"  Description: {config['description']}")
        print(f"  Output shape: {output.shape}")
        print(f"  Sample[0]: {output[0].detach().numpy()}")
        
        # Count learnable parameters specific to pooling
        if name == "cls_token":
            cls_params = model.cls_token.numel()
            print(f"  CLS token params: {cls_params}")
            print(f"  Attention heads: {model.num_attention_heads}")
        elif name == "attention":
            attn_params = sum(p.numel() for p in model.attention_V.parameters())
            attn_params += sum(p.numel() for p in model.attention_U.parameters())
            attn_params += sum(p.numel() for p in model.attention_w.parameters())
            print(f"  Attention params: {attn_params}")


if __name__ == "__main__":
    print("🤔 DeepCORO_CLIP CLS Token Implementation Examples")
    print("=" * 60)
    print("This demonstrates the newly implemented cls_token functionality")
    print("that provides learnable aggregation through self-attention.\n")
    
    # Run examples
    example_multi_instance_cls_token()
    example_cls_token_vs_pooling_comparison()
    
    print("\n" + "=" * 60)
    print("✅ Examples completed successfully!")
    print("\n🎯 Key Benefits of CLS Token:")
    print("  • Learnable aggregation that adapts during training")
    print("  • Self-attention mechanism captures complex relationships")
    print("  • Better than fixed pooling for complex tasks")
    print("  • Inspired by successful transformer architectures")
    print("  • Hierarchical processing for 4D inputs (videos + tokens)")
    print("  • Handles variable-length sequences with masking")
    print("\n🚀 New Improvements:")
    print("  • Separate attention layers for within/across-video processing")
    print("  • Hybrid pooling modes (mean+cls_token, attention+cls_token)")
    print("  • Configurable attention heads and normalization strategies")
    print("  • Robust edge case handling for empty masks")
    print("  • Better typing and modular architecture") 