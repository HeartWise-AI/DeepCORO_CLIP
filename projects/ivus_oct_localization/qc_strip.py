"""Render frame strips for visual QC: one row per video, k evenly spaced frames."""
import sys

import cv2
import numpy as np
import pandas as pd


def strip(path, k=6, size=320, label=""):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(n - 1, 0), k).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        f = cv2.resize(f, (size, size)) if ok else np.zeros((size, size, 3), np.uint8)
        out.append(f)
    cap.release()
    row = np.hstack(out)
    cv2.putText(row, f"{label} n={n}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return row


def sheet(rows, out):
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    sheet([strip(r.FileName, label=str(r.get("label", i))) for i, r in df.iterrows()], sys.argv[2])
