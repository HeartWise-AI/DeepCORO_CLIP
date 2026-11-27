"""Scan DICOM headers on disk to find IVUS/OCT objects the parquet missed (blank Modality pre-2024).

Scans every file in the study folders of: all 2023 + 2025 studies, and every positive study of other years.
Writes data/dicom_intravascular_scan.csv (one row per IVUS/OCT object).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pydicom

D = "projects/ivus_oct_localization/data"
SRC = "/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2025_VIDEO_LEVEL.parquet"
TAGS = ["Modality", "Manufacturer", "ManufacturerModelName", "SOPClassUID", "ImageComments", "SeriesDescription",
        "NumberOfFrames", "Rows", "Columns", "AcquisitionTime", "SeriesTime", "StudyInstanceUID", "TransducerData",
        "IVUSPullbackRate", "ImageType"]
IV_MODS = {"IVUS", "OCT", "OC", "IVOCT"}


def study_dirs():
    v = pd.read_parquet(SRC, columns=["StudyInstanceUID", "DICOMPath", "year"])
    v = v[v.DICOMPath.fillna("").str.len() > 0]
    pos = pd.read_csv(f"{D}/positive_studies_vessel_labels.csv", index_col=0).index
    v = v[v.year.astype(int).isin([2023, 2025]) | v.StudyInstanceUID.isin(pos)]
    v["dir"] = v.DICOMPath.map(os.path.dirname)
    # 2024/2025 layouts nest series folders under the study folder; walk the whole study folder
    nested = v.year.astype(int).isin([2024, 2025])
    v.loc[nested, "dir"] = v.loc[nested, "dir"].map(os.path.dirname)
    return v.drop_duplicates("dir")[["StudyInstanceUID", "year", "dir"]]


def scan(row):
    out = []
    for dp, _, fs in os.walk(row.dir):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                d = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=["Modality"])
                if str(d.get("Modality", "")).upper() not in IV_MODS:
                    continue
                d = pydicom.dcmread(p, stop_before_pixels=True)
                rec = {t: str(d.get(t, "")) for t in TAGS}
                rec.update(path=p, study_dir_uid=row.StudyInstanceUID, year=row.year)
                out.append(rec)
            except Exception:  # noqa: BLE001
                pass
    return out


def main():
    dirs = study_dirs()
    print("study dirs", len(dirs), dirs.year.value_counts().to_dict(), flush=True)
    rows = []
    with ThreadPoolExecutor(48) as ex:
        for k, r in enumerate(ex.map(scan, dirs.itertuples())):
            rows += r
            if k % 200 == 0:
                print(k, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{D}/dicom_intravascular_scan.csv", index=False)
    print(df.groupby(["year", "Modality"]).study_dir_uid.nunique())


if __name__ == "__main__":
    main()
