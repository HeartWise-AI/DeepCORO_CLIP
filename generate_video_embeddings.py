import os
import torch

from tqdm import tqdm
from typing import Any
from torch.utils.data import DataLoader

from models.video_encoder import VideoEncoder
from dataloaders.video_dataset import VideoDataset, custom_collate_fn

cuda_idx: int = 2
multi_video: bool = False

def load_checkpoint(
    checkpoint_path: str
)->dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    print(
        f"[BaseProject] Loading checkpoint: {checkpoint_path}"
    )
    
    return torch.load(checkpoint_path, map_location='cpu', weights_only=True)

def generate_video_embeddings():
    video_dataset: VideoDataset = VideoDataset(
        data_filename='data/CathEF_MHI_UCSF_2016-to-july-2022-and-2023-08-30-post-CathEF_alpha.csv',
        split='test',
        target_label=['y_true_cat'],
        datapoint_loc_label='FileName',
        num_frames=16,
        mean=[99.54182434082031, 99.54182434082031, 99.54182434082031],
        std=[43.9721794128418, 43.9721794128418, 43.9721794128418],
    )
    
    video_dataloader: DataLoader = DataLoader(
        video_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=16,
        collate_fn=custom_collate_fn,
    )
    
    video_encoder: VideoEncoder = VideoEncoder(
        backbone='mvit',
        num_frames=16,
        pretrained=True,
        freeze_ratio=0.8,
        dropout=0.2,
        num_heads=4,
        aggregator_depth=2,
    )
    
    video_encoder = video_encoder.to(f'cuda:{cuda_idx}').float()
    checkpoint: dict[str, Any] = load_checkpoint(
        checkpoint_path='outputs/dev_deep_coro_clip_single_video/mvit_pretrained_mvit_b24_f16_AdamW_lr2.527361715636149e-05_20250325-001727_xvwwv5ar/checkpoints/best_epoch.pt'
    )
    video_encoder.load_state_dict(checkpoint["video_encoder"])
    
    video_encoder.eval()
    output_dir_embeddings: str = 'video_embeddings/xvwwv5ar'
    if not os.path.exists(output_dir_embeddings):
        os.makedirs(output_dir_embeddings)
    
    for video_data in tqdm(video_dataloader, total=len(video_dataloader)):
        if not multi_video:
            video_data['videos'] = video_data['videos'].unsqueeze(1)
        embeddings = video_encoder(video_data['videos'].to(f'cuda:{cuda_idx}'))
        
        for idx, embedding in enumerate(embeddings):
            video_fname = os.path.basename(video_data['video_fname'][idx])
            embedding_fname = os.path.join(output_dir_embeddings, video_fname.replace('.avi', '.pt'))
            torch.save(embedding.cpu(), embedding_fname)

if __name__ == "__main__":
    generate_video_embeddings()



