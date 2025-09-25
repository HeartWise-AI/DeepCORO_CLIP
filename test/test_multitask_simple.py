#!/usr/bin/env python3
"""
Simple test for multitask components without downloading pretrained models.
"""

import os
import sys

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

import torch
import torch.nn as nn

def test_captioning_decoder():
    """Test CaptioningDecoder initialization and forward pass."""
    print("Testing CaptioningDecoder...")
    
    from models.captioning_decoder import CaptioningDecoder
    
    # Create model
    model = CaptioningDecoder(
        vocab_size=30522,
        hidden_size=256,
        num_layers=2,
        num_heads=8,
        intermediate_size=512,
        max_position_embeddings=128,
        dropout=0.1,
        use_biomed_tokenizer=False,  # Don't download tokenizer
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    num_video_tokens = 16
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    video_features = torch.randn(batch_size, num_video_tokens, 256)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        video_features=video_features,
    )
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, 30522)
    
    print(f"✓ CaptioningDecoder forward pass: logits shape {outputs['logits'].shape}")
    
    # Test generation
    generated = model.generate(
        video_features=video_features,
        max_length=20,
        do_sample=False,
    )
    
    assert generated.shape[0] == batch_size
    print(f"✓ CaptioningDecoder generation: shape {generated.shape}")
    
    return True

def test_masked_video_modeling():
    """Test MaskedVideoModeling initialization and forward pass."""
    print("\nTesting MaskedVideoModeling...")
    
    from models.masked_video_modeling import MaskedVideoModeling
    
    # Create model
    model = MaskedVideoModeling(
        hidden_size=256,
        decoder_hidden_size=128,
        decoder_layers=2,
        decoder_heads=8,
        mask_ratio=0.75,
        mask_token_learnable=True,
        norm_predict_loss=True,
    )
    
    # Test forward pass
    batch_size = 2
    num_tokens = 16
    
    video_tokens = torch.randn(batch_size, num_tokens, 256)
    
    outputs = model(video_tokens)
    
    assert "pred" in outputs
    assert "mask" in outputs
    assert "target" in outputs
    assert "loss" in outputs
    
    print(f"✓ MaskedVideoModeling forward pass:")
    print(f"  - pred shape: {outputs['pred'].shape}")
    print(f"  - mask shape: {outputs['mask'].shape}")
    print(f"  - loss value: {outputs['loss'].item():.4f}")
    
    return True

def test_multitask_loss():
    """Test MultitaskLoss computation."""
    print("\nTesting MultitaskLoss...")
    
    from utils.loss.multitask_loss import MultitaskLoss
    
    # Create loss function
    loss_fn = MultitaskLoss(
        loss_weights={
            "contrastive": 1.0,
            "captioning": 1.0,
            "masked_modeling": 0.1,
            "distillation": 0.0,
        },
        contrastive_loss_type="siglip",
        captioning_loss_type="cross_entropy",
        masked_modeling_loss_type="mse",
        temperature=0.1,
        label_smoothing=0.1,
    )
    
    # Create dummy inputs
    batch_size = 2
    hidden_size = 256
    seq_len = 32
    vocab_size = 30522
    num_tokens = 16
    
    video_features = torch.randn(batch_size, hidden_size)
    text_features = torch.randn(batch_size, hidden_size)
    caption_logits = torch.randn(batch_size, seq_len, vocab_size)
    caption_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    masked_pred = torch.randn(batch_size, num_tokens, hidden_size)
    masked_target = torch.randn(batch_size, num_tokens, hidden_size)
    masked_mask = torch.randint(0, 2, (batch_size, num_tokens)).bool()
    log_temp = torch.log(torch.tensor(0.1))
    
    # Compute loss
    losses = loss_fn(
        video_features=video_features,
        text_features=text_features,
        caption_logits=caption_logits,
        caption_targets=caption_targets,
        masked_pred=masked_pred,
        masked_target=masked_target,
        masked_mask=masked_mask,
        log_temp=log_temp,
    )
    
    assert "total" in losses
    assert losses["total"].requires_grad
    
    print(f"✓ MultitaskLoss computation:")
    print(f"  - Total loss: {losses['total'].item():.4f}")
    if "contrastive" in losses:
        print(f"  - Contrastive loss: {losses['contrastive'].item():.4f}")
    if "captioning" in losses:
        print(f"  - Captioning loss: {losses['captioning'].item():.4f}")
    if "masked_modeling" in losses:
        print(f"  - Masked modeling loss: {losses['masked_modeling'].item():.4f}")
    
    return True

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    
    try:
        from models.captioning_decoder import CaptioningDecoder
        print("✓ CaptioningDecoder import successful")
    except ImportError as e:
        print(f"✗ CaptioningDecoder import failed: {e}")
        return False
    
    try:
        from models.masked_video_modeling import MaskedVideoModeling
        print("✓ MaskedVideoModeling import successful")
    except ImportError as e:
        print(f"✗ MaskedVideoModeling import failed: {e}")
        return False
    
    try:
        from utils.loss.multitask_loss import MultitaskLoss
        print("✓ MultitaskLoss import successful")
    except ImportError as e:
        print(f"✗ MultitaskLoss import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Simple Multitask Components Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n✗ Import tests failed")
    
    # Test CaptioningDecoder
    try:
        if not test_captioning_decoder():
            all_passed = False
    except Exception as e:
        print(f"✗ CaptioningDecoder test failed: {e}")
        all_passed = False
    
    # Test MaskedVideoModeling
    try:
        if not test_masked_video_modeling():
            all_passed = False
    except Exception as e:
        print(f"✗ MaskedVideoModeling test failed: {e}")
        all_passed = False
    
    # Test MultitaskLoss
    try:
        if not test_multitask_loss():
            all_passed = False
    except Exception as e:
        print(f"✗ MultitaskLoss test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("✓ Multitask components are working correctly")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the errors above")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)