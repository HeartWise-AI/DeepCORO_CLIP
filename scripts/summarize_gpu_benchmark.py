"""Summarise a scripts/run_echojepa_benchmark.sh log directory.

Reads gpu_log.csv (nvidia-smi samples every 5 s) plus the stage marker files
(clip_start/clip_end/lp_start/lp_end, epoch seconds) and prints, per stage and
per GPU: mean / p50 / max SM utilisation, peak memory and mean power. Also
pulls the tqdm rate lines and the "Epoch ... time" lines out of clip.log and
lp.log when present so the throughput is visible next to the utilisation.

Usage: python scripts/summarize_gpu_benchmark.py logs/echojepa_benchmark_<ts>
"""

import os
import re
import sys
from datetime import datetime

import pandas as pd


def _read_marker(log_dir: str, name: str):
    p = os.path.join(log_dir, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return float(f.read().strip())


def _load_gpu_log(log_dir: str) -> pd.DataFrame:
    p = os.path.join(log_dir, "gpu_log.csv")
    df = pd.read_csv(p, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "utilization.gpu [%]": "util",
            "memory.used [MiB]": "mem_used",
            "memory.total [MiB]": "mem_total",
            "power.draw [W]": "power",
        }
    )
    for c in ("util", "mem_used", "mem_total", "power"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    df["t"] = pd.to_datetime(df["timestamp"].str.strip(), format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df["epoch"] = df["t"].map(lambda x: x.timestamp() if pd.notna(x) else float("nan"))
    return df.dropna(subset=["epoch"])


def _stage_table(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    s = df[(df["epoch"] >= start) & (df["epoch"] <= end)]
    if s.empty:
        return pd.DataFrame()
    g = s.groupby("index")
    out = pd.DataFrame(
        {
            "util_mean_%": g["util"].mean().round(1),
            "util_p50_%": g["util"].median().round(1),
            "util_max_%": g["util"].max(),
            "mem_peak_GiB": (g["mem_used"].max() / 1024).round(1),
            "mem_total_GiB": (g["mem_total"].max() / 1024).round(1),
            "power_mean_W": g["power"].mean().round(0),
        }
    )
    return out


def _log_highlights(path: str, max_lines: int = 12):
    if not os.path.exists(path):
        return []
    pat = re.compile(r"(it/s|s/it|Epoch .*time|epoch .*took|train/loss|val/loss|Recall@1|Saved .*checkpoint|Output directory)")
    keep = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if pat.search(line):
                # tqdm rewrites lines with \r; keep only the last segment
                keep.append(line.split("\r")[-1][:200])
    return keep[-max_lines:]


def main(log_dir: str) -> None:
    df = _load_gpu_log(log_dir)
    print(f"GPU samples: {len(df)} rows from {log_dir}/gpu_log.csv")
    for stage, log in (("clip", "clip.log"), ("lp", "lp.log")):
        start, end = _read_marker(log_dir, f"{stage}_start"), _read_marker(log_dir, f"{stage}_end")
        if start is None:
            continue
        end = end or df["epoch"].max()
        dur = end - start
        print(f"\n=== Stage {stage.upper()}  ({datetime.fromtimestamp(start):%H:%M:%S} -> "
              f"{datetime.fromtimestamp(end):%H:%M:%S}, {dur/60:.1f} min) ===")
        tab = _stage_table(df, start, end)
        print(tab.to_string() if not tab.empty else "  (no GPU samples in window)")
        hl = _log_highlights(os.path.join(log_dir, log))
        if hl:
            print("  -- log highlights --")
            for line in hl:
                print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
