"""Create a test checkpoint file for mvit_rope model."""

import torch
from torchvision.models.video import mvit_v2_s

# Create an MViT v2 S model
model = mvit_v2_s(weights=None)

# Create a checkpoint with the model's state dict
checkpoint = {
    "model_state_dict": model.state_dict(),
    "epoch": 100,
    "best_loss": 0.1,
    "config": {
        "model_name": "mvit_v2_s",
        "use_rope": True,
        "task": "masked_video_modeling"
    }
}

# Save the checkpoint
checkpoint_path = "outputs_mvit_debug_binary/mvit_v2_s_16_16_3_AdamW_new_20250903-181753_xz28ymfj/mvit_v2_s_16_16_2_AdamW_new_20250922-140829_dlgtqw29/best.pt"
torch.save(checkpoint, checkpoint_path)

print(f"Created test checkpoint at: {checkpoint_path}")
print(f"Checkpoint size: {torch.load(checkpoint_path)['model_state_dict'].keys().__len__()} parameters")