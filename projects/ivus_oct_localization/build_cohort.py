"""Build study-level IVUS/OCT labels and the video list used for embedding extraction.

Label sources (study level):
  * DICOM Modality == IVUS / OCT (only 2024 archived with intravascular objects)
  * Cath report IVUS flag and per-segment *_ivusrealise / *_ivusarea columns
  * Free text (Conclusion / Recommendation) regex for OCT / IVUS
Vessel sources:
  * IVUS DICOM ImageComments ("IVUS-2;CX;Evaluation")
  * per-segment report columns
  * free-text vessel mentions near "OCT" / "IVUS"
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import pydicom

OUT = sys.argv[1] if len(sys.argv) > 1 else "projects/ivus_oct_localization/data"
os.makedirs(OUT, exist_ok=True)
SRC = "/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2025_VIDEO_LEVEL.parquet"
ANGLES = "/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL_FULL_ANGLES_CLASSIFIED.parquet"
STRUCT = "/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL_STENT_BOOST_REPORT_AND_OTHER_STRUCTURES.parquet"

SEGMENTS = ["left_main", "prox_lad", "mid_lad", "dist_lad", "D1", "D2", "D3", "S1",
            "prox_lcx", "mid_lcx", "dist_lcx", "om1", "om2", "om3", "bx", "lvp",
            "prox_rca", "mid_rca", "dist_rca", "pda", "posterolateral", "right_marginal",
            "RVG1", "RVG2", "lima_or_svg"]
SEG2VESSEL = {"left_main": "LM", "bx": "LCX"}
for s in SEGMENTS:
    if "lad" in s or s[0] in "DS" and s[1:].isdigit():
        SEG2VESSEL[s] = "LAD"
    elif "lcx" in s or s.startswith("om") or s == "lvp":
        SEG2VESSEL[s] = "LCX"
    elif "rca" in s or s in ("pda", "posterolateral", "right_marginal") or s.startswith("RVG"):
        SEG2VESSEL[s] = "RCA"
SEG2VESSEL["lima_or_svg"] = "GRAFT"

# French report vocabulary -> vessel
VESSEL_RX = [
    ("LM", r"tronc commun|\bTC\b|\bLM\b"),
    ("LAD", r"interventriculaire ant|\bIVA\b|\bLAD\b|diagonale"),
    ("LCX", r"circonflexe|\bCX\b|\bLCX\b|marginale|bissectrice|\bCirc"),
    ("RCA", r"coronaire droite|\bCD\b|\bRCA\b|\bIVP\b|\bRVG\b|r[ée]troventriculaire"),
    ("GRAFT", r"pontage|greffon|\bSVG\b|\bLIMA\b|mammaire"),
]


def text_vessels(text, key):
    """Vessels mentioned within a window around each OCT/IVUS mention."""
    found = []
    for m in re.finditer(key, text):
        win = text[max(0, m.start() - 40): m.end() + 120]
        for v, rx in VESSEL_RX:
            if re.search(rx, win, flags=re.I) and v not in found:
                found.append(v)
    return found


seg_cols = [f"{s}_ivusrealise" for s in SEGMENTS] + [f"{s}_ivusarea" for s in SEGMENTS]
base = ["StudyInstanceUID", "patient_id_anon", "year", "Modality", "FileName", "DICOMPath",
        "SOPClassUID", "Rows", "NumberOfFrames", "IVUS", "Conclusion", "Recommendation",
        "STENTS", "DILATATIONS", "SeriesDescription", "AcquisitionTime", "primary_angle",
        "secondary_angle", "dicom_id"]
df = pd.read_parquet(SRC, columns=base + seg_cols)
print("rows", len(df))

# ---- study table
g = df.groupby("StudyInstanceUID")
st = g.agg(year=("year", "first"), patient=("patient_id_anon", "first"),
           ivus_flag=("IVUS", "max"), stents=("STENTS", "max"), dil=("DILATATIONS", "max"),
           conclusion=("Conclusion", "first"), recommendation=("Recommendation", "first"))
mods = df[df.Modality.isin(["IVUS", "OCT"])].groupby(["StudyInstanceUID", "Modality"]).size().unstack(fill_value=0)
st["n_ivus_dcm"] = mods.get("IVUS", pd.Series(dtype=int)).reindex(st.index).fillna(0).astype(int)
st["n_oct_dcm"] = mods.get("OCT", pd.Series(dtype=int)).reindex(st.index).fillna(0).astype(int)
txt = (st.conclusion.fillna("") + " " + st.recommendation.fillna(""))
st["oct_text"] = txt.str.contains(r"\bOCT\b")
st["ivus_text"] = txt.str.contains(r"\bIVUS\b|[ée]chographie endo", case=False)
st["oct_text_vessels"] = [",".join(text_vessels(t, r"\bOCT\b")) for t in txt]
st["ivus_text_vessels"] = [",".join(text_vessels(t, r"\bIVUS\b")) for t in txt]

first = g.first()
real_cols = [f"{s}_ivusrealise" for s in SEGMENTS]
real = first[real_cols].astype(str).apply(lambda c: c.str.strip().str.upper())
for mod in ["IVUS", "OCT"]:
    st[f"report_{mod.lower()}_segments"] = real.apply(
        lambda r: ",".join(c.replace("_ivusrealise", "") for c in real_cols if r[c] == mod), axis=1)
area_cols = [f"{s}_ivusarea" for s in SEGMENTS]
area = first[area_cols].apply(pd.to_numeric, errors="coerce")
st["ivusarea_segments"] = (area > 0).apply(
    lambda r: ",".join(c.replace("_ivusarea", "") for c in area_cols if r[c]), axis=1)


def seg_vessels(col_list):
    return col_list.str.split(",").apply(lambda l: ",".join(sorted({SEG2VESSEL[s] for s in l if s})))


st["report_ivus_vessels"] = seg_vessels(st.report_ivus_segments + "," + st.ivusarea_segments)
st["report_oct_vessels"] = seg_vessels(st.report_oct_segments)
st["pos_ivus"] = (st.n_ivus_dcm > 0) | (st.ivus_flag == 1) | st.ivus_text | (st.report_ivus_vessels != "")
st["pos_oct"] = (st.n_oct_dcm > 0) | st.oct_text | (st.report_oct_vessels != "")
st["pci"] = (st.stents.fillna(0) > 0) | (st.dil.fillna(0) > 0)
print(st[["pos_ivus", "pos_oct", "pci"]].value_counts())
print(pd.crosstab(st.year, [st.pos_ivus, st.pos_oct]))

# ---- per-pullback DICOM table (2024) with operator vessel label
pb = df[((df.Modality == "IVUS") & (df.Rows == 512)) | (df.Modality == "OCT")].copy()
lab = []
for p in pb.DICOMPath:
    try:
        d = pydicom.dcmread(p, stop_before_pixels=True)
        lab.append((str(d.get("ImageComments", "")), str(d.get("TransducerData", "")),
                    d.get("NumberOfFrames"), d.get("IVUSPullbackRate")))
    except Exception as e:  # noqa: BLE001
        lab.append(("ERR", "", None, None))
pb[["image_comments", "transducer", "n_frames_dcm", "pullback_rate"]] = pd.DataFrame(lab, index=pb.index)


def comment_vessel(c):
    parts = [x.strip().upper() for x in c.split(";")]
    for x in parts[1:2]:
        for v, keys in [("LM", ["LM", "TC", "LMCA"]), ("LAD", ["LAD", "IVA", "DIAG", "D1", "D2"]),
                        ("LCX", ["CX", "LCX", "OM", "OM1", "OM2", "CIRC", "MARG"]),
                        ("RCA", ["RCA", "CD", "PDA", "IVP", "PLV", "RVG"]), ("GRAFT", ["SVG", "LIMA", "GREFFON"])]:
            if x in keys or any(x.startswith(k) for k in keys):
                return v
    return ""


pb["dcm_vessel"] = pb.image_comments.apply(comment_vessel)
pb = pb[["StudyInstanceUID", "patient_id_anon", "Modality", "DICOMPath", "FileName", "AcquisitionTime",
         "SeriesDescription", "image_comments", "transducer", "n_frames_dcm", "pullback_rate", "dcm_vessel"]]
pb.to_csv(f"{OUT}/pullbacks_dicom.csv", index=False)
print(pb.groupby("Modality").dcm_vessel.value_counts())
print(pb.image_comments.value_counts().head(30))

# ---- cohort: positives + year-matched PCI negatives (3:1)
pos = st[st.pos_ivus | st.pos_oct]
neg_pool = st[~st.pos_ivus & ~st.pos_oct & st.pci]
rng = np.random.default_rng(0)
negs = []
for y, n in pos.year.value_counts().items():
    pool = neg_pool[neg_pool.year == y]
    negs.append(pool.sample(min(len(pool), 3 * n), random_state=int(rng.integers(1e9))))
neg = pd.concat(negs)
st["split_role"] = ""
st.loc[pos.index, "split_role"] = "pos"
st.loc[neg.index, "split_role"] = "neg"
st.drop(columns=["conclusion", "recommendation"]).to_csv(f"{OUT}/studies_all_labels.csv")
coh = st[st.split_role != ""].copy()
# patient-level split
pats = coh.patient.unique()
rng.shuffle(pats)
n = len(pats)
split = {p: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test") for i, p in enumerate(pats)}
coh["split"] = coh.patient.map(split)
coh.drop(columns=["conclusion", "recommendation"]).to_csv(f"{OUT}/cohort_studies.csv")

# ---- videos: XA cine files of cohort studies
v = df[df.StudyInstanceUID.isin(coh.index) & df.FileName.fillna("").str.len().gt(0)]
v = v[~v.Modality.isin(["IVUS", "OCT", "SR", "OT", "US", "DOC", "CT", "PR", "HD"])]
v = v[~v.SeriesDescription.fillna("").str.contains("RAPPORT|Dose Report", case=False)]
v = v[["StudyInstanceUID", "patient_id_anon", "year", "FileName", "Modality", "SOPClassUID", "NumberOfFrames",
       "SeriesDescription", "AcquisitionTime", "primary_angle", "secondary_angle", "dicom_id"]].drop_duplicates("FileName")
v["exists"] = v.FileName.map(os.path.exists)
print("videos", len(v), "exist", v.exists.mean())
v = v[v.exists].drop(columns="exists")
# attach view class / main structure where available
ang = pd.read_parquet(ANGLES, columns=["FileName", "view_class"]).drop_duplicates("FileName")
v = v.merge(ang, on="FileName", how="left")
try:
    s2 = pd.read_parquet(STRUCT, columns=["FileName", "main_structure_class", "stent_presence_class",
                                          "contrast_agent_class"]).drop_duplicates("FileName")
    v = v.merge(s2, on="FileName", how="left")
except Exception as e:  # noqa: BLE001
    print("struct merge failed", e)
v = v.merge(coh[["split", "pos_ivus", "pos_oct"]], left_on="StudyInstanceUID", right_index=True)
v.to_csv(f"{OUT}/cohort_videos.csv", index=False)
print(v.groupby("split").StudyInstanceUID.nunique(), len(v))
