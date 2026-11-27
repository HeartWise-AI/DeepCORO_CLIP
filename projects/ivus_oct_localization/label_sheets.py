"""Near-full-resolution labeling sheets: 4 cines per sheet, each a 2x2 grid of frames."""
import sys

import cv2
import numpy as np
import pandas as pd


def cine_grid(path, size=384, label=""):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for q in (0.12, 0.38, 0.62, 0.88):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(q * max(n - 1, 0)))
        ok, f = cap.read()
        fr.append(cv2.resize(f, (size, size)) if ok else np.zeros((size, size, 3), np.uint8))
    cap.release()
    g = np.vstack([np.hstack(fr[:2]), np.hstack(fr[2:])])
    cv2.rectangle(g, (0, 0), (g.shape[1] - 1, g.shape[0] - 1), (0, 0, 255), 3)
    cv2.putText(g, f"{label} n={n}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return g


def make_sheets(df, prefix):
    for i in range(0, len(df), 4):
        cells = [cine_grid(r.FileName, label=f"#{r.cid}") for r in df.iloc[i:i + 4].itertuples()]
        while len(cells) < 4:
            cells.append(np.zeros_like(cells[0]))
        cv2.imwrite(f"{prefix}_{i // 4:02d}.jpg", np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    make_sheets(pd.read_csv(sys.argv[1]), sys.argv[2])
