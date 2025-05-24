#!/usr/bin/env python3
"""
Example demonstrating cls_token functionality in DeepCORO_CLIP models.

This script shows how to use the newly implemented cls_token pooling mechanisms
in both the linear probing heads and multi-instance linear probing models.
"""

import torch
import torch.nn as nn
from models.linear_probing import ClsTokenLinearProbingHead, ClsTokenLinearProbingRegressionHead
from models.multi_instance_linear_probing import MultiInstanceLinearProbing


def example_cls_token_linear_probing():
    """Demonstrate cls_token usage in linear probing heads."""
    print("🤔 CLS Token Linear Probing Example")
    print("=" * 50)
    
    # Example parameters
    batch_size = 4
    input_dim = 512
    output_dim = 5
    seq_len = 10
    
    # Create cls_token-based classification head
    cls_head = ClsTokenLinearProbingHead(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=0.1,
        num_heads=8,
        use_cls_token=True
    )
    
    # Create cls_token-based regression head
    cls_regression_head = ClsTokenLinearProbingRegressionHead(
        input_dim=input_dim,
        output_dim=1,  # Single output for regression
        dropout=0.1,
        num_heads=8,
        use_cls_token=True
    )
    
    # Test with different input shapes
    
    # 1. 2D input (typical case)
    print(f"\n📊 Testing 2D input: [{batch_size}, {input_dim}]")
    x_2d = torch.randn(batch_size, input_dim)
    
    # Classification
    class_output = cls_head(x_2d)
    print(f"Classification output shape: {class_output.shape}")
    print(f"Classification output: {class_output[0]}")  # First sample
    
    # Regression
    reg_output = cls_regression_head(x_2d)
    print(f"Regression output shape: {reg_output.shape}")
    print(f"Regression output (scaled 0-100): {reg_output[0].item():.2f}")
    
    # 2. 3D input (sequence case)
    print(f"\n📊 Testing 3D input: [{batch_size}, {seq_len}, {input_dim}]")
    x_3d = torch.randn(batch_size, seq_len, input_dim)
    
    class_output_3d = cls_head(x_3d)
    print(f"Classification output shape: {class_output_3d.shape}")
    
    # Demonstrate cls_token learning
    print(f"\n🎯 CLS Token Parameters:")
    print(f"CLS token shape: {cls_head.cls_token.shape}")
    print(f"CLS token requires_grad: {cls_head.cls_token.requires_grad}")
    print(f"Number of attention heads: {cls_head.num_heads}")


def example_multi_instance_cls_token():
    """Demonstrate cls_token usage in multi-instance learning."""
    print("\n\n🤔 Multi-Instance CLS Token Example")
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
    print(f"CLS attention heads: {mil_cls_model.cls_attention.num_heads}")
    print(f"CLS attention embed_dim: {mil_cls_model.cls_attention.embed_dim}")


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
            attn_params = sum(p.numel() for p in model.cls_attention.parameters())
            print(f"  CLS token params: {cls_params}")
            print(f"  Attention params: {attn_params}")
            print(f"  Total pooling params: {cls_params + attn_params}")
        elif name == "attention":
            attn_params = sum(p.numel() for n, p in model.named_parameters() if "attention" in n)
            print(f"  Attention params: {attn_params}")
        else:
            print(f"  Pooling params: 0 (parameter-free)")


def example_hierarchical_cls_token_4d():
    """Demonstrate hierarchical cls_token usage with 4D inputs [B, N, L, D]."""
    print("\n\n🎬 Hierarchical CLS Token 4D Example (4 Videos, 393 Tokens Each)")
    print("=" * 70)
    
    # Your specific scenario: 4 videos with 393 tokens each
    batch_size = 2
    num_videos = 4        # N = 4 videos
    tokens_per_video = 393  # L = 393 tokens per video  
    embedding_dim = 512   # D = embedding dimension
    
    print(f"📊 Input scenario:")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Videos per sample: {num_videos}")
    print(f"  • Tokens per video: {tokens_per_video}")
    print(f"  • Embedding dimension: {embedding_dim}")
    print(f"  • Total input shape: [{batch_size}, {num_videos}, {tokens_per_video}, {embedding_dim}]")
    
    # Define multi-head tasks
    head_structure = {
        "cathEF": 1,           # Regression task (e.g., ejection fraction)
        "stenosis": 3,         # Multi-class classification
        "contrast": 2,         # Binary classification
    }
    
    # Create model with cls_token pooling
    model = MultiInstanceLinearProbing(
        embedding_dim=embedding_dim,
        head_structure=head_structure,
        pooling_mode="cls_token",
        dropout=0.1,
        use_cls_token=True
    )
    
    # Create 4D input data [B, N, L, D]
    x = torch.randn(batch_size, num_videos, tokens_per_video, embedding_dim)
    
    # Create video-level mask (some samples might have fewer videos)
    video_mask = torch.ones(batch_size, num_videos, dtype=torch.bool)
    video_mask[1, 3:] = False  # Second sample only has 3 videos
    
    print(f"\n📊 Data shapes:")
    print(f"  • Input: {x.shape}")
    print(f"  • Video mask: {video_mask.shape}")
    print(f"  • Valid videos per sample: {video_mask.sum(dim=1).tolist()}")
    
    # Forward pass
    print(f"\n🔄 Forward pass...")
    outputs = model(x, video_mask)
    
    print(f"\n🎯 Hierarchical CLS Token Results:")
    for task_name, output in outputs.items():
        print(f"  {task_name:>10}: shape {output.shape}, sample[0] = {output[0]}")
    
    # Show parameter counts
    cls_params = model.cls_token.numel()
    attn_params = sum(p.numel() for p in model.cls_attention.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📈 Model Parameters:")
    print(f"  • CLS token: {cls_params:,} params")
    print(f"  • Attention mechanism: {attn_params:,} params")
    print(f"  • Total model: {total_params:,} params")
    
    # Compare processing steps
    print(f"\n🔍 Processing Steps Explanation:")
    print(f"  1. Input: [{batch_size}, {num_videos}, {tokens_per_video}, {embedding_dim}]")
    print(f"  2. Step 1 - Within-video attention:")
    print(f"     • Reshape to [{batch_size * num_videos}, {tokens_per_video}, {embedding_dim}]")
    print(f"     • Apply cls_token across {tokens_per_video} tokens within each video")
    print(f"     • Result: [{batch_size * num_videos}, {embedding_dim}] (one repr per video)")
    print(f"  3. Step 2 - Across-video attention:")
    print(f"     • Reshape to [{batch_size}, {num_videos}, {embedding_dim}]")
    print(f"     • Apply cls_token across {num_videos} video representations")
    print(f"     • Result: [{batch_size}, {embedding_dim}] (final sample representation)")
    print(f"  4. Multi-head output: [{batch_size}, num_classes] for each task")
    
    return model, x, outputs


def compare_pooling_with_4d():
    """Compare different pooling methods with 4D input."""
    print("\n\n🔄 4D Pooling Methods Comparison")
    print("=" * 50)
    
    # Smaller example for comparison
    batch_size = 2
    num_videos = 4
    tokens_per_video = 50  # Smaller for faster comparison
    embedding_dim = 128
    
    x = torch.randn(batch_size, num_videos, tokens_per_video, embedding_dim)
    video_mask = torch.ones(batch_size, num_videos, dtype=torch.bool)
    
    pooling_methods = {
        "cls_token": "Hierarchical cls_token (2-level attention)",
        "attention": "Hierarchical gated attention", 
        "mean": "Mean pooling (with automatic flattening warning)",
        "max": "Max pooling (with automatic flattening warning)"
    }
    
    print(f"📊 Input: [{batch_size}, {num_videos}, {tokens_per_video}, {embedding_dim}]")
    print(f"\n🔄 Results:")
    
    for method, description in pooling_methods.items():
        model = MultiInstanceLinearProbing(
            embedding_dim=embedding_dim,
            head_structure={"output": 2},
            pooling_mode=method,
            attention_hidden=64,
            dropout=0.0
        )
        
        # Capture warnings for mean/max pooling
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = model(x, video_mask)["output"]
            warning_msg = f" (Warning: {w[0].message})" if w else ""
        
        print(f"\n{method.upper():>12}")
        print(f"  Description: {description}")
        print(f"  Output shape: {output.shape}")
        print(f"  Sample[0]: {output[0].detach().numpy()}")
        if warning_msg:
            print(f"  Note: {warning_msg}")


def example_improved_cls_token_features():
    """Demonstrate the improved cls_token features: separate attention, hybrid pooling, etc."""
    print("\n\n🚀 Improved CLS Token Features Demo")
    print("=" * 50)
    
    batch_size = 2
    num_videos = 3
    tokens_per_video = 100
    embedding_dim = 256
    
    print(f"📊 Testing improved features with shape: [{batch_size}, {num_videos}, {tokens_per_video}, {embedding_dim}]")
    
    # Test different configurations
    configs = {
        "standard_cls_token": {
            "pooling_mode": "cls_token",
            "separate_video_attention": False,
            "description": "Original cls_token (shared attention)"
        },
        "separate_attention": {
            "pooling_mode": "cls_token", 
            "separate_video_attention": True,
            "description": "Separate within/across-video attention layers"
        },
        "hybrid_mean_cls": {
            "pooling_mode": "mean+cls_token",
            "separate_video_attention": True,
            "description": "Hybrid: Mean + CLS Token (2x feature dim)"
        },
        "hybrid_attention_cls": {
            "pooling_mode": "attention+cls_token",
            "separate_video_attention": True,
            "description": "Hybrid: Attention + CLS Token (2x feature dim)"
        },
        "pre_norm": {
            "pooling_mode": "cls_token",
            "separate_video_attention": True,
            "normalization_strategy": "pre_norm",
            "description": "Pre-norm like some modern transformers"
        }
    }
    
    x = torch.randn(batch_size, num_videos, tokens_per_video, embedding_dim)
    mask = torch.ones(batch_size, num_videos, dtype=torch.bool)
    mask[1, 2] = False  # Second sample has only 2 videos
    
    print(f"\n🔍 Configuration Comparison:")
    
    for config_name, config in configs.items():
        try:
            model = MultiInstanceLinearProbing(
                embedding_dim=embedding_dim,
                head_structure={"test": 2},
                num_attention_heads=4,  # Smaller for faster testing
                **config
            )
            
            output = model(x, mask)["test"]
            expected_dim = 2 * embedding_dim if "+" in config["pooling_mode"] else embedding_dim
            
            print(f"\n{config_name.upper():>20}")
            print(f"  Description: {config['description']}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected feature dim: {expected_dim}")
            print(f"  Sample output: {output[0][:3]}...")  # First 3 values
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            cls_params = model.cls_token.numel() if hasattr(model, 'cls_token') else 0
            print(f"  Total params: {total_params:,}")
            print(f"  CLS token params: {cls_params}")
            
        except Exception as e:
            print(f"\n{config_name.upper():>20}: ERROR - {e}")


def example_edge_case_handling():
    """Demonstrate improved edge case handling."""
    print("\n\n🛡️ Edge Case Handling Demo")
    print("=" * 50)
    
    batch_size = 3
    num_instances = 4
    embedding_dim = 128
    
    # Create test cases with different edge scenarios
    x = torch.randn(batch_size, num_instances, embedding_dim)
    
    edge_cases = {
        "all_valid": torch.ones(batch_size, num_instances, dtype=torch.bool),
        "some_invalid": torch.tensor([
            [True, True, False, False],   # Sample 0: 2 valid instances
            [True, False, False, False],  # Sample 1: 1 valid instance  
            [True, True, True, True]      # Sample 2: All valid
        ], dtype=torch.bool),
        "all_invalid": torch.zeros(batch_size, num_instances, dtype=torch.bool),
        "mixed_lengths": torch.tensor([
            [True, True, True, False],    # Sample 0: 3 valid
            [True, True, False, False],   # Sample 1: 2 valid
            [True, False, False, False]   # Sample 2: 1 valid
        ], dtype=torch.bool)
    }
    
    model = MultiInstanceLinearProbing(
        embedding_dim=embedding_dim,
        head_structure={"output": 3},
        pooling_mode="cls_token",
        separate_video_attention=True,
        num_attention_heads=4
    )
    
    print(f"📊 Input shape: {x.shape}")
    print(f"\n🔍 Edge Case Results:")
    
    for case_name, mask in edge_cases.items():
        try:
            print(f"\n{case_name.upper():>15}")
            print(f"  Mask: {mask}")
            print(f"  Valid per sample: {mask.sum(dim=1).tolist()}")
            
            output = model(x, mask)["output"]
            print(f"  Output shape: {output.shape}")
            print(f"  Contains NaN: {torch.isnan(output).any().item()}")
            print(f"  Contains Inf: {torch.isinf(output).any().item()}")
            print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
        except Exception as e:
            print(f"  ERROR: {e}")


def example_attention_head_comparison():
    """Compare different numbers of attention heads."""
    print("\n\n🧠 Attention Head Comparison")
    print("=" * 50)
    
    batch_size = 2
    num_instances = 6
    embedding_dim = 192  # Divisible by multiple head counts
    
    x = torch.randn(batch_size, num_instances, embedding_dim)
    mask = torch.ones(batch_size, num_instances, dtype=torch.bool)
    
    head_counts = [1, 2, 3, 4, 6, 8, 12]
    
    print(f"📊 Input: [{batch_size}, {num_instances}, {embedding_dim}]")
    print(f"\n🔍 Attention Head Analysis:")
    
    for num_heads in head_counts:
        if embedding_dim % num_heads != 0:
            print(f"\n{num_heads:>2} heads: SKIPPED (embedding_dim not divisible)")
            continue
            
        try:
            model = MultiInstanceLinearProbing(
                embedding_dim=embedding_dim,
                head_structure={"test": 2},
                pooling_mode="cls_token",
                num_attention_heads=num_heads,
                separate_video_attention=True
            )
            
            # Time the forward pass
            import time
            start_time = time.time()
            output = model(x, mask)["test"]
            forward_time = time.time() - start_time
            
            # Calculate attention parameters
            if hasattr(model, 'cls_attention_within'):
                attn_params = sum(p.numel() for p in model.cls_attention_within.parameters())
                attn_params += sum(p.numel() for p in model.cls_attention_across.parameters())
            else:
                attn_params = sum(p.numel() for p in model.cls_attention.parameters())
                
            print(f"\n{num_heads:>2} heads:")
            print(f"  Output: {output[0][:2]}...")  
            print(f"  Attention params: {attn_params:,}")
            print(f"  Forward time: {forward_time:.4f}s")
            print(f"  Head dimension: {embedding_dim // num_heads}")
            
        except Exception as e:
            print(f"\n{num_heads:>2} heads: ERROR - {e}")


if __name__ == "__main__":
    print("🤔 DeepCORO_CLIP CLS Token Implementation Examples")
    print("=" * 60)
    print("This demonstrates the newly implemented cls_token functionality")
    print("that provides learnable aggregation through self-attention.\n")
    
    # Run examples
    example_cls_token_linear_probing()
    example_multi_instance_cls_token()
    example_cls_token_vs_pooling_comparison()
    
    # New 4D hierarchical examples
    example_hierarchical_cls_token_4d()
    compare_pooling_with_4d()
    
    # New improved features examples
    example_improved_cls_token_features()
    example_edge_case_handling()
    example_attention_head_comparison()
    
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