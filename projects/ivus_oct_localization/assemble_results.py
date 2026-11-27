"""Final deliverable: every IVUS / OCT study, the vessel imaged, and the angiogram cine(s) showing it.

Inputs: positive_studies_vessel_labels.csv, all_videos_scored_stageA.csv, instance_scores_{oct,ivus}.csv,
        instance_labels.csv, pullbacks_dicom.csv, embeddings (for the image-based vessel model).
Outputs (data/results/):
  ivus_oct_studies.csv  one row per (study, modality)
  ivus_oct_cines.csv    top-3 candidate angio cines per (study, modality) with scores
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = "projects/ivus_oct_localization/data"
OUT = f"{D}/results"
os.makedirs(OUT, exist_ok=True)
THR = {"oct": 0.5, "ivus": 0.5}  # instance score threshold for "catheter cine found"


def canon(label):
    s = {x for x in str(label).split(",") if x and x != "nan"} - {"GRAFT"}
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


def main():
    lab = pd.read_csv(f"{D}/positive_studies_vessel_labels.csv", index_col=0, low_memory=False)
    vids = pd.read_csv(f"{D}/all_videos_scored_stageA.csv", low_memory=False)
    rec = vids[vids.rec_class != "XA"].groupby(["StudyInstanceUID", "rec_class"]).size().unstack(fill_value=0)
    il = pd.read_csv(f"{D}/instance_labels.csv")
    idx, X = load_emb()
    studies, cines = [], []
    for t in ["oct", "ivus"]:
        sc = pd.read_csv(f"{D}/instance_scores_{t}.csv")
        sc = sc.merge(vids[["FileName", "year", "view_class", "primary_angle", "secondary_angle", "nframes"]],
                      on="FileName", how="left")
        pos = lab[lab[f"pos_{t}"]].copy()
        s = sc[sc.StudyInstanceUID.isin(pos.index)].sort_values(f"inst_score_{t}", ascending=False)
        s["rank"] = s.groupby("StudyInstanceUID").cumcount() + 1
        top3 = s[s["rank"] <= 3].copy()
        top3["modality"] = t.upper()
        vlab = il[(il.task == t)].set_index("FileName")
        top3["visually_verified"] = top3.FileName.map(vlab.label)
        top3["vessel_visual"] = top3.FileName.map(vlab.vessel_visual)
        cines.append(top3.rename(columns={f"inst_score_{t}": "catheter_score"}))

        # image-based vessel model: score-weighted embedding of top-3 cines -> vessel (trained on report labels)
        feats = {}
        for sid, g in top3.groupby("StudyInstanceUID"):
            w = g[f"inst_score_{t}"].values + 1e-3
            feats[sid] = (X[idx[g.FileName].values] * (w / w.sum())[:, None]).sum(0)
        F = pd.DataFrame.from_dict(feats, orient="index")
        y_all = pos[f"{t}_vessel_label"].map(canon)
        # verified visual vessel overrides report for training when available
        vis = top3.dropna(subset=["vessel_visual"]).groupby("StudyInstanceUID").vessel_visual.first()
        y_all.loc[vis.index.intersection(y_all.index)] = vis
        tr = y_all.dropna().index.intersection(F.index)
        y = y_all[tr].values
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        oof = np.empty(len(tr), dtype=object)
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F.loc[tr], y, pos.loc[tr, "patient"].astype(str)):
            m.fit(F.loc[tr].values[a], y[a])
            oof[b] = m.predict(F.loc[tr].values[b])
        acc = (oof == y).mean()
        print(t, "image vessel model CV acc", round(acc, 3), "n", len(tr), pd.Series(y).value_counts().to_dict())
        print(pd.crosstab(pd.Series(y, name="label"), pd.Series(oof, name="pred")))
        m.fit(F.loc[tr].values, y)
        P = m.predict_proba(F.values)
        img = pd.DataFrame({"vessel_image_model": m.classes_[P.argmax(1)], "vessel_image_conf": P.max(1)}, index=F.index)

        best = s[s["rank"] == 1].set_index("StudyInstanceUID")
        out = pd.DataFrame(index=pos.index)
        out["modality"] = t.upper()
        out["year"] = pos.year
        out["patient"] = pos.patient
        out["evidence_dicom_objects"] = pos[f"n_{t}_dcm"] if f"n_{t}_dcm" in pos else 0
        out["evidence_recording_videos"] = rec.get(t.upper(), pd.Series(dtype=int)).reindex(pos.index).fillna(0).astype(int)
        out["evidence_report"] = pos[f"{t}_text"] | (pos[f"report_{t}_vessels"].fillna("") != "")
        out["vessel_report"] = pos[f"{t}_vessel_label"]
        out["vessel_report_source"] = pos[f"{t}_vessel_source"]
        out = out.join(img)
        out["vessel_visual"] = vis.reindex(out.index)
        # final vessel: visual > dicom/report segment/text line > image model (pci fallback only if image model absent)
        strong = out.vessel_report_source.isin(["dicom", "report_segment", "report_text_line"])
        out["vessel_final"] = np.where(out.vessel_visual.notna(), out.vessel_visual,
                                       np.where(strong, out.vessel_report.map(lambda v: canon(v) or v),
                                                np.where(out.vessel_image_model.notna(), out.vessel_image_model,
                                                         out.vessel_report)))
        out["vessel_final_source"] = np.where(out.vessel_visual.notna(), "visual_review",
                                              np.where(strong, out.vessel_report_source,
                                                       np.where(out.vessel_image_model.notna(), "image_model",
                                                                out.vessel_report_source)))
        out["best_cine"] = best.FileName.reindex(out.index)
        out["best_cine_score"] = best[f"inst_score_{t}"].reindex(out.index)
        out["best_cine_view"] = best.view_class.reindex(out.index)
        out["cine_found"] = out.best_cine_score >= THR[t]
        out["n_cines_above_thr"] = s[s[f"inst_score_{t}"] >= THR[t]].groupby("StudyInstanceUID").size().reindex(out.index).fillna(0).astype(int)
        studies.append(out)
    S = pd.concat(studies)
    S.index.name = "StudyInstanceUID"
    S.to_csv(f"{OUT}/ivus_oct_studies.csv")
    C = pd.concat(cines)
    C[["StudyInstanceUID", "modality", "rank", "FileName", "catheter_score", "visually_verified", "vessel_visual",
       "view_class", "primary_angle", "secondary_angle", "nframes", "year"]].to_csv(f"{OUT}/ivus_oct_cines.csv", index=False)
    print(S.groupby("modality").agg(studies=("year", "size"), cine_found=("cine_found", "sum")))
    print(pd.crosstab(S.modality, S.vessel_final_source))
    print(pd.crosstab(S.modality, S.vessel_final.fillna("unknown")))
    print(pd.crosstab([S.modality], S.year))


if __name__ == "__main__":
    main()
