"""Build the Labelbox dataset for IVUS/OCT device + vessel annotation (video-level classification).

Selection (>= 1000 studies):
  * every IVUS/OCT-positive study (report / DICOM / recording evidence), top-3 candidate angio cines per modality
    ranked by the per-cine catheter model (ivus_oct_cines.csv), deduplicated;
  * N_CONTROL control studies (PCI, no IVUS/OCT evidence), top-2 cines by max catheter score = hard negatives.
Annotators are blinded: no model score, report vessel or modality is uploaded; those stay in manifest.csv.

Writes to OUT:
  videos/<global_key>.mp4   512x512, 15 fps, H.264 (same format as previous DeepCORO Labelbox projects)
  manifest.csv              global_key <-> FileName / StudyInstanceUID / priors (local only, never uploaded)
  data_rows.json            Labelbox data rows (row_data = local mp4 path, global_key, study_key metadata)
"""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

D = "projects/ivus_oct_localization/data"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/media/data1/datasets/DeepCoro/labelbox_ivus_oct"
TOPK, N_CONTROL, CONTROL_K = 3, 200, 2


def key(s, n=10):
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def to_mp4(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=512:512,format=yuv420p", "-r", "15",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", "-movflags", "+faststart", dst]
    return subprocess.run(cmd).returncode == 0


def main():
    os.makedirs(f"{OUT}/videos", exist_ok=True)
    cines = pd.read_csv(f"{D}/results/ivus_oct_cines.csv")
    studies = pd.read_csv(f"{D}/results/ivus_oct_studies.csv")

    pos = cines[cines["rank"] <= TOPK].copy()
    pos = (pos.sort_values("catheter_score", ascending=False)
              .groupby("FileName").agg(StudyInstanceUID=("StudyInstanceUID", "first"),
                                       candidate_for=("modality", lambda s: ",".join(sorted(set(s)))),
                                       best_rank=("rank", "min"), max_catheter_score=("catheter_score", "max"),
                                       view_class=("view_class", "first"), primary_angle=("primary_angle", "first"),
                                       secondary_angle=("secondary_angle", "first"), nframes=("nframes", "first"),
                                       year=("year", "first")).reset_index())
    pos["cohort"] = "ivus_oct_positive"
    # study-level priors (kept local for analysis)
    pri = studies.pivot_table(index="StudyInstanceUID", columns="modality",
                              values=["vessel_final", "vessel_final_source"], aggfunc="first")
    pri.columns = [f"prior_{a}_{b.lower()}" for a, b in pri.columns]
    ev = studies.assign(e=lambda x: x.modality).groupby("StudyInstanceUID").e.agg(lambda s: ",".join(sorted(set(s))))
    pos = pos.merge(pri, left_on="StudyInstanceUID", right_index=True, how="left")
    pos["prior_modality_evidence"] = pos.StudyInstanceUID.map(ev)

    # hard-negative controls: top cines by catheter score in control studies
    so = pd.read_csv(f"{D}/instance_scores_oct.csv")
    si = pd.read_csv(f"{D}/instance_scores_ivus.csv")
    sc = so.merge(si[["FileName", "inst_score_ivus"]], on="FileName")
    vids = pd.read_csv(f"{D}/all_videos_scored_stageA.csv", low_memory=False,
                       usecols=["FileName", "pos_ivus", "pos_oct", "view_class", "primary_angle", "secondary_angle",
                                "nframes", "year"])
    sc = sc.merge(vids, on="FileName")
    ctl = sc[~sc.pos_ivus.astype(bool) & ~sc.pos_oct.astype(bool)].copy()
    ctl["max_catheter_score"] = ctl[["inst_score_oct", "inst_score_ivus"]].max(1)
    pick = ctl.StudyInstanceUID.drop_duplicates().sample(N_CONTROL, random_state=0)
    ctl = (ctl[ctl.StudyInstanceUID.isin(pick)].sort_values("max_catheter_score", ascending=False)
              .groupby("StudyInstanceUID").head(CONTROL_K))
    ctl["cohort"], ctl["candidate_for"], ctl["prior_modality_evidence"] = "control", "", ""
    ctl = ctl[[c for c in pos.columns if c in ctl.columns]]

    m = pd.concat([pos, ctl], ignore_index=True)
    m["study_key"] = "study_" + m.StudyInstanceUID.map(key)
    m["global_key"] = "ivoct_" + m.study_key.str[6:] + "_" + m.FileName.map(lambda f: key(f, 8))
    m = m.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle so annotators can't infer cohort/order
    m["mp4"] = OUT + "/videos/" + m.global_key + ".mp4"

    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(to_mp4, m.FileName, m.mp4))
    m["converted"] = ok
    print("converted", sum(ok), "/", len(m))
    m = m[m.converted]
    m.to_csv(f"{OUT}/manifest.csv", index=False)

    rows = [{"row_data": r.mp4, "global_key": r.global_key, "external_id": f"{r.global_key}.mp4",
             "metadata_fields": [{"name": "study_key", "value": r.study_key}]} for r in m.itertuples()]
    json.dump(rows, open(f"{OUT}/data_rows.json", "w"), indent=1)
    print("studies", m.StudyInstanceUID.nunique(), m.groupby("cohort").StudyInstanceUID.nunique().to_dict(),
          "videos", len(m), m.cohort.value_counts().to_dict())


if __name__ == "__main__":
    main()
