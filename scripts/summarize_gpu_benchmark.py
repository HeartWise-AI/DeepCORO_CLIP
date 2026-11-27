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


def _steady_state_table(df: pd.DataFrame, start: float, end: float, mem_frac: float = 0.5) -> pd.DataFrame:
    """Same as _stage_table but only over samples where the model is resident on the
    GPU (memory > mem_frac of total), i.e. excluding CSV parsing / checkpoint loading /
    dataloader start-up at the beginning of a stage."""
    s = df[(df["epoch"] >= start) & (df["epoch"] <= end)]
    s = s[s["mem_used"] > mem_frac * s["mem_total"]]
    if s.empty:
        return pd.DataFrame()
    return _stage_table(s, start, end)


_TQDM_DONE = re.compile(
    r"\[GPU 0\] Running (?P<mode>\w+) Epoch (?P<ep>\d+): 100%\|[^|]*\|\s*(?P<steps>\d+)/\d+ \[(?P<elapsed>[\d:]+)<"
)


def _epoch_throughput(path: str, clips_per_step: int):
    """Parse the completed tqdm bars ('100%| ... N/N [MM:SS<00:00') of rank 0 and turn them
    into steps/s and clips/s over all ranks (clips_per_step = world_size * per-GPU clips)."""
    if not os.path.exists(path):
        return []
    rows = []
    seen = set()
    with open(path, errors="replace") as f:
        for raw in f:
            for line in raw.split("\r"):
                m = _TQDM_DONE.search(line)
                if not m or (m["mode"], m["ep"]) in seen:
                    continue
                seen.add((m["mode"], m["ep"]))
                parts = [int(x) for x in m["elapsed"].split(":")]
                secs = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
                steps = int(m["steps"])
                if secs > 0:
                    rows.append(
                        f"  {m['mode']:5s} epoch {m['ep']}: {steps} steps in {secs/60:.1f} min "
                        f"-> {steps/secs:.2f} steps/s, ~{steps*clips_per_step/secs:.0f} clips/s over all GPUs"
                    )
    return rows


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


def _clips_per_step(log_dir: str):
    """world_size * per-GPU clips per step, per stage, read from the benchmark configs."""
    try:
        import yaml
    except ImportError:
        return {}
    gpus_file = os.path.join(log_dir, "gpus.txt")
    world = len(open(gpus_file).read().strip().split(",")) if os.path.exists(gpus_file) else 1
    out = {}
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for stage, cfg in (("clip", "config/clip/echojepa_vitl_clip_benchmark.yaml"),
                       ("lp", "config/linear_probing/stenosis/echojepa_vitl_stenosis_benchmark.yaml")):
        cp = os.path.join(repo, cfg)
        if not os.path.exists(cp):
            continue
        with open(cp) as f:
            c = yaml.safe_load(f)
        bs = int(c.get("batch_size", 1))
        nv = int(c.get("num_videos", 1)) if c.get("multi_video") else 1
        out[stage] = world * bs * nv
    return out


def main(log_dir: str) -> None:
    clips_per_step = _clips_per_step(log_dir)
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
        print("-- whole stage (incl. CSV parsing / checkpoint load / loader start-up) --")
        print(tab.to_string() if not tab.empty else "  (no GPU samples in window)")
        ss = _steady_state_table(df, start, end)
        print("-- steady state (samples with >50% GPU memory in use, i.e. model resident) --")
        print(ss.to_string() if not ss.empty else "  (no steady-state samples)")
        cps = clips_per_step.get(stage)
        if cps:
            tp = _epoch_throughput(os.path.join(log_dir, log), cps)
            if tp:
                print(f"-- throughput from rank-0 progress bars ({cps} clips/step over all GPUs) --")
                for r in tp:
                    print(r)
        hl = _log_highlights(os.path.join(log_dir, log))
        if hl:
            print("  -- log highlights --")
            for line in hl:
                print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
