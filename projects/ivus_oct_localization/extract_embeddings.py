"""Extract per-video DeepCORO-CLIP embeddings (mean- and max-pooled patch tokens).

Usage: python extract_embeddings.py <videos.csv> <out_dir> <rank> <world>
Each rank processes a shard and writes <out_dir>/emb_<rank>.npz.
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/volume/DeepCORO_CLIP")
from models.video_encoder import VideoEncoder  # noqa: E402

CKPT = "/media/data1/models/DeepCoro_CLIP/8av1xygm_20250605-083820_best_single_video/checkpoints/best_model_epoch_7.pt"
MEAN, STD = 110.4954833984375, 37.805782318115234
N_FRAMES, SIZE = 16, 224


class Videos(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        frames = []
        try:
            cap = cv2.VideoCapture(p)
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (SIZE, SIZE), interpolation=cv2.INTER_AREA))
            cap.release()
        except Exception:  # noqa: BLE001
            frames = []
        if not frames:
            return torch.zeros(N_FRAMES, SIZE, SIZE, 3), i, 0, 0
        n = len(frames)
        idx = np.linspace(0, n - 1, N_FRAMES).round().astype(int)
        v = np.stack([frames[k] for k in idx]).astype(np.float32)
        v = (v - MEAN) / STD
        v = np.repeat(v[..., None], 3, axis=-1)  # [F,H,W,3]
        return torch.from_numpy(v), i, 1, n


def main():
    csv, out, rank, world = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(csv)
    paths = df.FileName.tolist()[rank::world]
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    enc = VideoEncoder(backbone="mvit", num_frames=16, pretrained=False, output_dim=512, num_heads=16,
                       aggregator_depth=1, aggregate_videos_tokens=False, per_video_pool=False, freeze_ratio=1.0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["video_encoder"]
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print("missing", missing, "unexpected", unexpected, flush=True)
    enc.eval().to(dev)
    dl = DataLoader(Videos(paths), batch_size=32, num_workers=12, pin_memory=True)
    E_mean = np.zeros((len(paths), 512), np.float16)
    E_max = np.zeros((len(paths), 512), np.float16)
    ok = np.zeros(len(paths), bool)
    nfr = np.zeros(len(paths), np.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, (x, i, good, n) in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            _, tok = enc(x, return_tokens=True, token_mode="grid")  # [B,1,L,D]
            tok = tok[:, 0].float()
            E_mean[i.numpy()] = tok.mean(1).cpu().numpy()
            E_max[i.numpy()] = tok.amax(1).cpu().numpy()
            ok[i.numpy()] = good.numpy().astype(bool)
            nfr[i.numpy()] = n.numpy()
            if b % 50 == 0:
                print(rank, b * 32, len(paths), flush=True)
    np.savez(f"{out}/emb_{rank}.npz", paths=np.array(paths), mean=E_mean, max=E_max, ok=ok, nframes=nfr)


if __name__ == "__main__":
    main()
