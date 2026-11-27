"""Stage A: classify every video as angiogram (XA) / IVUS recording / OCT recording.

Pre-2024 parquets have a blank Modality, so converted IVUS/OCT recordings sit among the angio
videos.  Train a multinomial logistic regression on DeepCORO-CLIP embeddings using videos with
known Modality (2024 XA/IVUS/OCT, 2025 XA), apply to all videos.
Writes data/all_videos_scored_stageA.csv
"""
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = "projects/ivus_oct_localization/data"


def load_all():
    embs = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{D}/emb/emb_*.npz") + glob.glob(f"{D}/emb_extra/emb_*.npz"))]
    paths = np.concatenate([e["paths"] for e in embs])
    X = np.concatenate([np.concatenate([e["mean"], e["max"]], 1) for e in embs]).astype(np.float32)
    ok = np.concatenate([e["ok"] for e in embs])
    nfr = np.concatenate([e["nframes"] for e in embs])
    v = pd.concat([pd.read_csv(f"{D}/cohort_videos.csv", low_memory=False),
                   pd.read_csv(f"{D}/extra_videos.csv", low_memory=False)]).drop_duplicates("FileName")
    idx = pd.Series(np.arange(len(paths)), index=paths)
    idx = idx[~idx.index.duplicated()]
    v = v[v.FileName.isin(idx.index)].copy()
    v["row"] = idx[v.FileName].values
    v = v[ok[v.row]].copy()
    v["nframes"] = nfr[v.row]
    return v, X


def main():
    v, X = load_all()
    F = np.concatenate([X[v.row.values], np.log1p(v.nframes.values)[:, None]], 1)
    known = v.Modality.isin(["XA", "IVUS", "OCT"]).values
    y = v.Modality.where(v.Modality.isin(["IVUS", "OCT"]), "XA").values
    print("training labels", pd.Series(y[known]).value_counts().to_dict())
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
    oof = cross_val_predict(m, F[known], y[known], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    classes = np.array(sorted(set(y[known])))
    pred = classes[oof.argmax(1)]
    print(pd.crosstab(pd.Series(y[known], name="true"), pd.Series(pred, name="pred")))
    m.fit(F[known], y[known])
    P = m.predict_proba(F)
    for j, c in enumerate(m.classes_):
        v[f"p_rec_{c}"] = P[:, j]
    v["rec_class"] = m.classes_[P.argmax(1)]
    v.loc[known, "rec_class_oof"] = pred
    print(pd.crosstab(v.year, v.rec_class))
    v.drop(columns="row").to_csv(f"{D}/all_videos_scored_stageA.csv", index=False)


if __name__ == "__main__":
    main()
