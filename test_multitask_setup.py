#!/usr/bin/env python3
"""
Test script for the multitask DeepCORO-CLIP setup.

This script tests:
1. Model initialization
2. Forward passes
3. Loss computation
4. Basic training step
"""

import torch
import torch.nn as nn
from typing import Dict, Any

# Import our components
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from models.captioning_decoder import CaptioningDecoder
from models.masked_video_modeling import MaskedVideoModeling
from utils.loss.multitask_loss import MultitaskLoss
from utils.registry import ModelRegistry, LossRegistry


def test_model_initialization():
    """Test that all models can be initialized correctly."""
    print("Testing model initialization...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test video encoder
    video_encoder = VideoEncoder(
        backbone="mvit",
        input_channels=3,
        num_frames=16,
        pretrained=False,  # Don't load pretrained weights for testing
        output_dim=512,
        dropout=0.1,
        num_heads=8,
        freeze_ratio=0.0,
        aggregator_depth=2,
        aggregate_videos_tokens=False,  # We need token-level features
    ).to(device)
    
    # Test text encoder
    text_encoder = TextEncoder(
        freeze_ratio=0.0,
        dropout=0.1,
    ).to(device)
    
    # Test captioning decoder
    captioning_decoder = CaptioningDecoder(
        vocab_size=30522,
        hidden_size=512,
        num_layers=2,  # Smaller for testing
        num_heads=8,
        intermediate_size=1024,  # Smaller for testing
        max_position_embeddings=512,
        dropout=0.1,
        use_biomed_tokenizer=False,  # Don't download tokenizer for testing
    ).to(device)
    
    # Test masked video modeling
    masked_video_modeling = MaskedVideoModeling(
        hidden_size=512,
        decoder_hidden_size=256,
        decoder_layers=1,  # Smaller for testing
        decoder_heads=8,
        mask_ratio=0.75,
        mask_token_learnable=True,
        norm_predict_loss=True,
    ).to(device)
    
    print("✓ All models initialized successfully")
    return video_encoder, text_encoder, captioning_decoder, masked_video_modeling


def test_forward_passes(video_encoder, text_encoder, captioning_decoder, masked_video_modeling):
    """Test forward passes through all models."""
    print("Testing forward passes...")
    
    device = next(video_encoder.parameters()).device
    batch_size = 2
    
    # Create dummy inputs
    videos = torch.randn(batch_size, 16, 224, 224, 3).to(device)  # [B, T, H, W, C]
    input_ids = torch.randint(0, 30522, (batch_size, 64)).to(device)  # [B, seq_len]
    attention_mask = torch.ones(batch_size, 64).to(device)  # [B, seq_len]
    
    # Test video encoder
    video_tokens = video_encoder.get_tokens(videos, mode="patch")  # [B, num_tokens, hidden_size]
    video_features = video_encoder.get_tokens(videos, mode="video")  # [B, hidden_size]
    print(f"✓ Video encoder: tokens {video_tokens.shape}, features {video_features.shape}")
    
    # Test text encoder
    text_features = text_encoder(input_ids, attention_mask)  # [B, hidden_size]
    print(f"✓ Text encoder: features {text_features.shape}")
    
    # Test captioning decoder
    caption_input_ids = input_ids[:, :-1].contiguous()
    caption_attention_mask = attention_mask[:, :-1].contiguous()
    caption_outputs = captioning_decoder(
        input_ids=caption_input_ids,
        attention_mask=caption_attention_mask,
        video_features=video_tokens,
    )
    caption_logits = caption_outputs["logits"]  # [B, seq_len-1, vocab_size]
    print(f"✓ Captioning decoder: logits {caption_logits.shape}")
    
    # Test masked video modeling
    mvm_outputs = masked_video_modeling(video_tokens)
    masked_pred = mvm_outputs["pred"]  # [B, num_tokens, decoder_hidden_size]
    masked_mask = mvm_outputs["mask"]  # [B, num_tokens]
    print(f"✓ Masked video modeling: pred {masked_pred.shape}, mask {masked_mask.shape}")
    
    print("✓ All forward passes successful")
    return {
        "video_tokens": video_tokens,
        "video_features": video_features,
        "text_features": text_features,
        "caption_logits": caption_logits,
        "caption_targets": input_ids[:, 1:].contiguous(),
        "masked_pred": masked_pred,
        "masked_target": video_tokens,
        "masked_mask": masked_mask,
    }


def test_loss_computation(outputs):
    """Test multitask loss computation."""
    print("Testing loss computation...")
    
    device = next(outputs["video_features"].parameters()).device
    
    # Create multitask loss
    loss_fn = MultitaskLoss(
        loss_weights={
            "contrastive": 1.0,
            "captioning": 1.0,
            "masked_modeling": 0.1,
            "distillation": 0.0,
        },
        contrastive_loss_type="sigmoid",
        captioning_loss_type="cross_entropy",
        masked_modeling_loss_type="mse",
        temperature=0.1,
        label_smoothing=0.1,
        ignore_index=-100,
    ).to(device)
    
    # Create temperature parameter
    log_temp = torch.log(torch.tensor(0.1, device=device))
    
    # Compute loss
    loss_outputs = loss_fn(
        video_features=outputs["video_features"],
        text_features=outputs["text_features"],
        caption_logits=outputs["caption_logits"],
        caption_targets=outputs["caption_targets"],
        masked_pred=outputs["masked_pred"],
        masked_target=outputs["masked_target"],
        masked_mask=outputs["masked_mask"],
        log_temp=log_temp,
    )
    
    print(f"✓ Total loss: {loss_outputs['total'].item():.4f}")
    print(f"✓ Contrastive loss: {loss_outputs.get('contrastive', torch.tensor(0.0)).item():.4f}")
    print(f"✓ Captioning loss: {loss_outputs.get('captioning', torch.tensor(0.0)).item():.4f}")
    print(f"✓ Masked modeling loss: {loss_outputs.get('masked_modeling', torch.tensor(0.0)).item():.4f}")
    
    return loss_outputs


def test_training_step(video_encoder, text_encoder, captioning_decoder, masked_video_modeling, loss_outputs):
    """Test a complete training step."""
    print("Testing training step...")
    
    device = next(video_encoder.parameters()).device
    
    # Create optimizer
    param_groups = [
        {
            'params': video_encoder.parameters(),
            'lr': 0.0001,
            'name': 'video_encoder',
        },
        {
            'params': text_encoder.parameters(),
            'lr': 0.00002,
            'name': 'text_encoder',
        },
        {
            'params': captioning_decoder.parameters(),
            'lr': 0.0001,
            'name': 'captioning_decoder',
        },
        {
            'params': masked_video_modeling.parameters(),
            'lr': 0.00001,
            'name': 'masked_video_modeling',
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=0.0001)
    
    # Training step
    optimizer.zero_grad()
    total_loss = loss_outputs["total"]
    total_loss.backward()
    optimizer.step()
    
    print("✓ Training step completed successfully")
    
    # Check that gradients were computed
    has_gradients = False
    for param in video_encoder.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    print(f"✓ Gradients computed: {has_gradients}")


def test_caption_generation(captioning_decoder, video_tokens):
    """Test caption generation."""
    print("Testing caption generation...")
    
    # Generate captions
    generated_ids = captioning_decoder.generate(
        video_features=video_tokens,
        max_length=32,
        do_sample=False,
        temperature=1.0,
    )
    
    print(f"✓ Generated captions: {generated_ids.shape}")
    print(f"✓ Sample generated IDs: {generated_ids[0][:10]}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Multitask DeepCORO-CLIP Setup")
    print("=" * 50)
    
    try:
        # Test 1: Model initialization
        models = test_model_initialization()
        video_encoder, text_encoder, captioning_decoder, masked_video_modeling = models
        
        # Test 2: Forward passes
        outputs = test_forward_passes(video_encoder, text_encoder, captioning_decoder, masked_video_modeling)
        
        # Test 3: Loss computation
        loss_outputs = test_loss_computation(outputs)
        
        # Test 4: Training step
        test_training_step(video_encoder, text_encoder, captioning_decoder, masked_video_modeling, loss_outputs)
        
        # Test 5: Caption generation
        test_caption_generation(captioning_decoder, outputs["video_tokens"])
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("✓ Multitask setup is working correctly")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)