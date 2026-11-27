"""Instance-level (per-cine) classifier for 'imaging catheter visible' from verified labels.

Positives: visually verified cines (instance_labels.csv, label=1).
Negatives: verified negatives + random angio cines from control (non-IVUS/OCT) studies.
Scores every angio cine; within each positive study, rank cines by score.
Usage: python instance_model.py <task: oct|ivus> <out_candidates.csv> <n_studies> [seed]
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = "projects/ivus_oct_localization/data"


def load():
    embs = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{D}/emb/emb_*.npz") + glob.glob(f"{D}/emb_extra/emb_*.npz"))]
    paths = np.concatenate([e["paths"] for e in embs])
    X = np.concatenate([np.concatenate([e["mean"], e["max"]], 1) for e in embs]).astype(np.float32)
    idx = pd.Series(np.arange(len(paths)), index=paths)
    idx = idx[~idx.index.duplicated()]
    v = pd.read_csv(f"{D}/all_videos_scored_stageA.csv", low_memory=False)
    v = v[(v.rec_class == "XA") & v.FileName.isin(idx.index)].copy()
    F = np.concatenate([X[idx[v.FileName].values], np.log1p(v.nframes.values)[:, None]], 1)
    return v.reset_index(drop=True), F


def main():
    task, out, n_studies = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    v, F = load()
    lab = pd.read_csv(f"{D}/instance_labels.csv")
    lab = lab[lab.task == task].drop_duplicates("FileName", keep="last")
    row = pd.Series(v.index, index=v.FileName)
    lab = lab[lab.FileName.isin(row.index)]
    neg_pool = v[~v.pos_ivus.astype(bool) & ~v.pos_oct.astype(bool)].sample(20000, random_state=0)
    Xl = np.concatenate([F[row[lab.FileName].values], F[neg_pool.index]])
    yl = np.concatenate([lab.label.values, np.zeros(len(neg_pool))])
    w = np.concatenate([np.full(len(lab), 5.0), np.ones(len(neg_pool))])
    groups = np.concatenate([lab.StudyInstanceUID.values, neg_pool.StudyInstanceUID.values])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced"))
    # quick CV on the verified labels: does the verified positive outrank the others in its study?
    if lab.label.sum() >= 5:
        oof = np.zeros(len(yl))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xl, yl, groups):
            m.fit(Xl[a], yl[a], logisticregression__sample_weight=w[a])
            oof[b] = m.predict_proba(Xl[b])[:, 1]
        from sklearn.metrics import roc_auc_score
        k = len(lab)
        print("CV AUROC verified labels only:", round(roc_auc_score(yl[:k], oof[:k]), 3),
              "| all incl. control negatives:", round(roc_auc_score(yl, oof), 3))
    m.fit(Xl, yl, logisticregression__sample_weight=w)
    v[f"inst_score_{task}"] = m.predict_proba(F)[:, 1]
    v[[f"inst_score_{task}", "FileName", "StudyInstanceUID"]].to_csv(f"{D}/instance_scores_{task}.csv", index=False)
    # candidates for next labeling round: top-1 cine of studies not yet reviewed
    pos_col = "pos_oct" if task == "oct" else "pos_ivus"
    rec = pd.read_csv(f"{D}/all_videos_scored_stageA.csv", low_memory=False)
    confirmed = rec[rec.rec_class == task.upper()].StudyInstanceUID.unique()
    s = v[v[pos_col].astype(bool)]
    s = s[~s.StudyInstanceUID.isin(lab.StudyInstanceUID)]
    s["confirmed_recording"] = s.StudyInstanceUID.isin(confirmed)
    top = s.sort_values(f"inst_score_{task}", ascending=False).groupby("StudyInstanceUID").head(1)
    top = top.sample(min(n_studies, len(top)), random_state=seed).sort_values(f"inst_score_{task}", ascending=False)
    top = top.reset_index(drop=True)
    top["cid"] = range(len(top))
    top.to_csv(out, index=False)
    print(top[[f"inst_score_{task}", "year", "view_class", "confirmed_recording"]].describe(include="all").T)


if __name__ == "__main__":
    main()
