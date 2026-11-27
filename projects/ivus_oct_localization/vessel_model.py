"""Predict the imaged vessel from the localised IVUS/OCT cine(s).

For each positive study and modality t, pool the embeddings of the top-k cines by loc_t
(attention x instance prob).  Train a multinomial logistic regression on studies with a
single unambiguous vessel label (patient-grouped 5-fold CV), compare against a control
that uses k random cines from the same study (tests whether localisation adds signal).
"""
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = "projects/ivus_oct_localization/data"
K = 3


def canon(label):
    s = set(x for x in str(label).split(",") if x and x != "nan")
    s.discard("GRAFT")
    if s == {"RCA"}:
        return "RCA"
    if s in ({"LAD"}, {"LM", "LAD"}):
        return "LAD"
    if s in ({"LCX"}, {"LM", "LCX"}):
        return "LCX"
    if s == {"LM"}:
        return "LM"
    return None


def load_emb():
    embs = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{D}/emb/emb_*.npz") + glob.glob(f"{D}/emb_extra/emb_*.npz"))]
    paths = np.concatenate([e["paths"] for e in embs])
    X = np.concatenate([np.concatenate([e["mean"], e["max"]], 1) for e in embs]).astype(np.float32)
    idx = pd.Series(np.arange(len(paths)), index=paths)
    return idx[~idx.index.duplicated()], X


def pooled(vs, idx, X, t, mode, rng):
    feats = {}
    for sid, g in vs.groupby("StudyInstanceUID"):
        if mode == "top":
            g = g.sort_values(f"loc_{t}", ascending=False).head(K)
            w = g[f"loc_{t}"].values
            w = w / w.sum() if w.sum() > 0 else np.ones(len(g)) / len(g)
        else:
            g = g.sample(min(K, len(g)), random_state=int(rng.integers(1e9)))
            w = np.ones(len(g)) / len(g)
        feats[sid] = (X[idx[g.FileName].values] * w[:, None]).sum(0)
    return pd.DataFrame.from_dict(feats, orient="index")


def main():
    vs = pd.read_csv(f"{D}/video_scores.csv")
    lab = pd.read_csv(f"{D}/positive_studies_vessel_labels.csv", index_col=0)
    idx, X = load_emb()
    rng = np.random.default_rng(0)
    preds = []
    report = {}
    for t in ["ivus", "oct"]:
        pos = lab[lab[f"pos_{t}"]]
        v = vs[vs.StudyInstanceUID.isin(pos.index)]
        y_all = pos[f"{t}_vessel_label"].map(canon)
        F_top = pooled(v, idx, X, t, "top", rng)
        F_rand = pooled(v, idx, X, t, "rand", rng)
        tr = y_all.dropna().index.intersection(F_top.index)
        y = y_all[tr].values
        groups = pos.loc[tr, "patient"].values
        res = {}
        for name, F in [("top", F_top), ("random", F_rand)]:
            oof = np.empty(len(tr), dtype=object)
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
            for a, b in cv.split(F.loc[tr], y, groups):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
                m.fit(F.loc[tr].values[a], y[a])
                oof[b] = m.predict(F.loc[tr].values[b])
            res[name] = round(float((oof == y).mean()), 3)
            if name == "top":
                oof_top = oof
        cm = pd.crosstab(pd.Series(y, name="label"), pd.Series(oof_top, name="pred"))
        print(t, "n_train", len(tr), pd.Series(y).value_counts().to_dict(), "CV acc", res)
        print(cm)
        report[t] = dict(n=len(tr), cv_acc=res, confusion=cm.to_dict())
        # fit on all labelled, predict every positive study
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(F_top.loc[tr].values, y)
        P = m.predict_proba(F_top.values)
        out = pd.DataFrame(P, index=F_top.index, columns=[f"p_{c}" for c in m.classes_])
        out["pred_vessel"] = m.classes_[P.argmax(1)]
        out["pred_conf"] = P.max(1)
        oofs = pd.Series(oof_top, index=tr)
        out["oof_pred_vessel"] = oofs.reindex(out.index)
        out["modality"] = t.upper()
        preds.append(out)
    pd.concat(preds).to_csv(f"{D}/vessel_predictions.csv")
    pd.Series(report).to_json(f"{D}/vessel_metrics.json", indent=1)


if __name__ == "__main__":
    main()
