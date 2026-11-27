"""Parse a Labelbox NDJSON export of deepcoro-ivus-oct back to FileName / StudyInstanceUID.

Usage: python parse_export.py <export.ndjson> [manifest.csv] [out.csv]
Output: one row per labelled video + study-level summary (<out>_study.csv): modality (OCT/IVUS/both/none)
and vessels imaged, aggregated over the study's cines.
"""
import json
import sys

import pandas as pd

exp = sys.argv[1]
man = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "/media/data1/datasets/DeepCoro/labelbox_ivus_oct/manifest.csv")
out = sys.argv[3] if len(sys.argv) > 3 else exp.replace(".ndjson", "_labels.csv")

recs = []
for line in open(exp):
    d = json.loads(line)
    gk = d["data_row"].get("global_key")
    for proj in d.get("projects", {}).values():
        for lab in proj.get("labels", []):
            r = {"global_key": gk, "label_id": lab["id"],
                 "labeler": lab.get("label_details", {}).get("created_by", "")}
            for c in lab["annotations"].get("classifications", []):
                r[c["name"]] = c.get("radio_answer", {}).get("value")
            recs.append(r)
lab = pd.DataFrame(recs).merge(man, on="global_key", how="left")
lab.to_csv(out, index=False)

img = lab[lab.intravascular_imaging.isin(["oct", "ivus", "imaging_uncertain"])]
study = lab.groupby("StudyInstanceUID").agg(n_cines=("global_key", "size"), cohort=("cohort", "first")).join(
    img.groupby("StudyInstanceUID").agg(
        modality=("intravascular_imaging", lambda s: ",".join(sorted(set(s)))),
        vessels=("vessel", lambda s: ",".join(sorted(set(s.dropna())))),
        imaging_cines=("global_key", lambda s: ";".join(s))))
study["modality"] = study.modality.fillna("none")
study.to_csv(out.replace(".csv", "_study.csv"))
print(lab.intravascular_imaging.value_counts(dropna=False).to_dict())
print(study.modality.value_counts().to_dict())
