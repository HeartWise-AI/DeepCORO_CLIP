"""Study-level 'vessel imaged' labels for IVUS / OCT from reports and DICOM headers.

Priority (most to least specific):
  1. IVUS DICOM ImageComments vessel (2024 only, per pullback)
  2. Report per-segment *_ivusrealise == IVUS/OCT
  3. Vessels named in the same line/sentence as the IVUS/OCT mention
  4. Fallback: vessels with PCI done (*_pcidone) in that study
"""
import re
import sys

import pandas as pd

sys.path.insert(0, "projects/ivus_oct_localization")
D = "projects/ivus_oct_localization/data"
SRC = "/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2025_STUDY_LEVEL.parquet"

VESSEL_RX = [
    ("LM", r"tronc commun|\bTC\b|\bLM\b|left main"),
    ("LAD", r"interventriculaire ant|\bIVA\b|\bLAD\b|diagonale|\bD1\b|\bDiago"),
    ("LCX", r"circonflexe|\bCX\b|\bLCX\b|marginale|\bM[12]\b|\bOM\d?\b|bissectrice|\bCirc"),
    ("RCA", r"coronaire droite|\bCD\b|\bCD[a-z]*\b|\bRCA\b|\bIVP\b|\bRVG\b|postérolatérale|\bPL\b"),
    ("GRAFT", r"pontage|greffon\b|\bSVG\b|\bLIMA\b|mammaire|\bAMI\b|veine saph"),
]
SEGMENTS = ["left_main", "prox_lad", "mid_lad", "dist_lad", "D1", "D2", "D3", "S1",
            "prox_lcx", "mid_lcx", "dist_lcx", "om1", "om2", "om3", "bx", "lvp",
            "prox_rca", "mid_rca", "dist_rca", "pda", "posterolateral", "right_marginal",
            "RVG1", "RVG2", "lima_or_svg"]


def seg2v(s):
    if s == "left_main":
        return "LM"
    if "lad" in s or s in ("D1", "D2", "D3", "S1"):
        return "LAD"
    if "lcx" in s or s.startswith("om") or s in ("lvp", "bx"):
        return "LCX"
    if s == "lima_or_svg":
        return "GRAFT"
    return "RCA"


def line_vessels(text, key):
    out = []
    for line in re.split(r"[\n\r]+|(?<=\.)\s+", text):
        if re.search(key, line):
            for v, rx in VESSEL_RX:
                if re.search(rx, line, flags=re.I) and v not in out:
                    out.append(v)
    return out


def main():
    st = pd.read_csv(f"{D}/studies_all_labels.csv", index_col=0)
    cols = ["StudyInstanceUID", "Conclusion", "Recommendation"] + [f"{s}_pcidone" for s in SEGMENTS]
    rep = pd.read_parquet(SRC, columns=cols).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    rep = rep.reindex(st.index)
    txt = rep.Conclusion.fillna("") + "\n" + rep.Recommendation.fillna("")
    st["oct_line_vessels"] = [",".join(line_vessels(t, r"\bOCT\b")) for t in txt]
    st["ivus_line_vessels"] = [",".join(line_vessels(t, r"\bIVUS\b")) for t in txt]
    pci = rep[[f"{s}_pcidone" for s in SEGMENTS]].apply(pd.to_numeric, errors="coerce").fillna(0) > 0
    st["pci_vessels"] = pci.apply(lambda r: ",".join(sorted({seg2v(c.replace("_pcidone", "")) for c in r.index if r[c]})),
                                  axis=1)
    pb = pd.read_csv(f"{D}/pullbacks_dicom.csv")
    st["dcm_ivus_vessels"] = pb[pb.dcm_vessel.notna()].groupby("StudyInstanceUID").dcm_vessel.apply(
        lambda s: ",".join(sorted(set(s))))
    st["n_ivus_pullbacks_dcm"] = pb[pb.Modality == "IVUS"].groupby("StudyInstanceUID").size()
    st["n_oct_pullbacks_dcm"] = pb[pb.Modality == "OCT"].groupby("StudyInstanceUID").size()

    def pick(r, t):
        srcs = ([("dicom", r.dcm_ivus_vessels)] if t == "ivus" else []) + [
            ("report_segment", r[f"report_{t}_vessels"]), ("report_text_line", r[f"{t}_line_vessels"]),
            ("pci_fallback", r.pci_vessels)]
        for name, v in srcs:
            if isinstance(v, str) and v:
                return pd.Series({f"{t}_vessel_label": v, f"{t}_vessel_source": name})
        return pd.Series({f"{t}_vessel_label": "", f"{t}_vessel_source": ""})

    for t in ["ivus", "oct"]:
        lab = st.apply(lambda r: pick(r, t), axis=1)
        st[lab.columns] = lab
        p = st[st[f"pos_{t}"]]
        print(t, len(p), p[f"{t}_vessel_source"].value_counts().to_dict())
        print(p[f"{t}_vessel_label"].value_counts().head(10).to_dict())
    keep = [c for c in st.columns if c not in ("conclusion", "recommendation")]
    st[st.pos_ivus | st.pos_oct][keep].to_csv(f"{D}/positive_studies_vessel_labels.csv")


if __name__ == "__main__":
    main()
