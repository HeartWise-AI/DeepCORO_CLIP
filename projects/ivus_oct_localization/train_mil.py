"""Attention-MIL over per-video DeepCORO-CLIP embeddings to detect IVUS / OCT studies
and localise the cine(s) acquired with the intravascular catheter.

study logit_t = sum_i a_{i,t} * c_{i,t}   (per-task gated attention a, per-instance logit c)
Per-video localisation score = a_{i,t} * sigmoid(c_{i,t}) and the instance logit c_{i,t}.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

D = sys.argv[1] if len(sys.argv) > 1 else "projects/ivus_oct_localization/data"
TASKS = ["pos_ivus", "pos_oct"]
torch.manual_seed(0)
np.random.seed(0)


def load():
    # angiogram cines only: drop IVUS/OCT recordings flagged by Stage A (recording_classifier.py)
    vids = pd.read_csv(f"{D}/all_videos_scored_stageA.csv", low_memory=False)
    vids = vids[vids.rec_class == "XA"].drop(columns=["nframes"])
    embs = [np.load(f, allow_pickle=True)
            for f in sorted(glob.glob(f"{D}/emb/emb_*.npz") + glob.glob(f"{D}/emb_extra/emb_*.npz"))]
    paths = np.concatenate([e["paths"] for e in embs])
    X = np.concatenate([np.concatenate([e["mean"], e["max"]], 1) for e in embs]).astype(np.float32)
    ok = np.concatenate([e["ok"] for e in embs])
    nfr = np.concatenate([e["nframes"] for e in embs])
    idx = pd.Series(np.arange(len(paths)), index=paths)
    idx = idx[~idx.index.duplicated()]
    vids = vids[vids.FileName.isin(idx.index)].copy()
    vids["row"] = idx[vids.FileName].values
    vids = vids[ok[vids.row]]
    vids["nframes"] = nfr[vids.row]
    # simple metadata: log length (pullback/fluoro recordings are long)
    meta = np.log1p(vids.nframes.values.astype(np.float32))[:, None] / 6.0
    feats = np.concatenate([X[vids.row.values], meta], 1)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    feats = (feats - mu) / sd
    vids["frow"] = np.arange(len(vids))
    studies = vids.groupby("StudyInstanceUID").agg(split=("split", "first"), pos_ivus=("pos_ivus", "first"),
                                                   pos_oct=("pos_oct", "first"), rows=("frow", list))
    return vids, studies, torch.tensor(feats)


class MIL(nn.Module):
    def __init__(self, d_in, d=256, n_tasks=2, p=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.GELU())
        self.V = nn.Linear(d, 128)
        self.U = nn.Linear(d, 128)
        self.w = nn.Linear(128, n_tasks)
        self.cls = nn.Linear(d, n_tasks)
        self.drop = nn.Dropout(p)

    def forward(self, x):  # x [N, d_in]
        h = self.enc(self.drop(x))
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # [N, T]
        a = torch.softmax(a, 0)
        c = self.cls(h)  # [N, T]
        return (a * c).sum(0), a, c


def run_epoch(model, studies, X, opt=None, pos_w=None):
    train = opt is not None
    model.train(train)
    order = np.random.permutation(len(studies)) if train else np.arange(len(studies))
    logits, ys, losses = [], [], []
    for k in order:
        r = studies.iloc[k]
        x = X[r.rows].cuda()
        if train and len(r.rows) > 4:  # instance dropout
            keep = torch.rand(len(r.rows)) > 0.1
            x = x[keep.cuda()] if keep.any() else x
        y = torch.tensor([float(r[t]) for t in TASKS]).cuda()
        with torch.set_grad_enabled(train):
            z, _, _ = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pos_w)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        logits.append(z.detach().cpu().numpy())
        ys.append(y.cpu().numpy())
        losses.append(loss.item())
    return np.mean(losses), np.array(logits), np.array(ys)


def metrics(z, y):
    out = {}
    for j, t in enumerate(TASKS):
        if y[:, j].min() != y[:, j].max():
            out[t] = dict(auroc=round(roc_auc_score(y[:, j], z[:, j]), 4),
                          auprc=round(average_precision_score(y[:, j], z[:, j]), 4), n_pos=int(y[:, j].sum()))
    return out


def main():
    vids, studies, X = load()
    print(studies.groupby("split")[TASKS].sum(), studies.split.value_counts())
    tr, va, te = (studies[studies.split == s] for s in ["train", "val", "test"])
    model = MIL(X.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    prev = tr[TASKS].mean().values
    pos_w = torch.tensor(np.sqrt((1 - prev) / prev), dtype=torch.float32).cuda()
    best, best_state = -1, None
    for ep in range(30):
        l, _, _ = run_epoch(model, tr, X, opt, pos_w)
        lv, zv, yv = run_epoch(model, va, X)
        m = metrics(zv, yv)
        score = np.mean([m[t]["auprc"] for t in m])
        print(ep, round(l, 4), round(lv, 4), m, flush=True)
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(dict(state=best_state, d_in=X.shape[1]), f"{D}/mil_best.pt")
    res = {}
    for name, s in [("val", va), ("test", te)]:
        _, z, y = run_epoch(model, s, X)
        res[name] = metrics(z, y)
    print("FINAL", json.dumps(res, indent=1))
    json.dump(res, open(f"{D}/mil_metrics.json", "w"), indent=1)

    # per-video scores for every study
    model.eval()
    rows = []
    with torch.no_grad():
        for sid, r in studies.iterrows():
            z, a, c = model(X[r.rows].cuda())
            a, c, z = a.cpu().numpy(), c.cpu().numpy(), torch.sigmoid(z).cpu().numpy()
            for k, fr in enumerate(r.rows):
                rows.append(dict(frow=fr, study_p_ivus=z[0], study_p_oct=z[1], att_ivus=a[k, 0], att_oct=a[k, 1],
                                 inst_ivus=c[k, 0], inst_oct=c[k, 1]))
    sc = pd.DataFrame(rows)
    out = vids.merge(sc, on="frow")
    for t in ["ivus", "oct"]:
        out[f"loc_{t}"] = out[f"att_{t}"] * (1 / (1 + np.exp(-out[f"inst_{t}"])))
        out[f"rank_{t}"] = out.groupby("StudyInstanceUID")[f"loc_{t}"].rank(ascending=False, method="first")
    out.drop(columns=["row", "frow"]).to_csv(f"{D}/video_scores.csv", index=False)


if __name__ == "__main__":
    main()
