#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = {1:"blink",3:"blink_double",2:"tap",4:"tap_double",6:"fist_hold",5:"brow"}

def resolve_col(arg_val, ncols, fallback_idx):
    """Supports 1-based positive indexing and negative indexing (-1=last)."""
    if arg_val is None:
        return fallback_idx
    idx = (arg_val-1) if arg_val > 0 else (ncols + arg_val)
    if not (0 <= idx < ncols):
        raise SystemExit(f"Column {arg_val} out of bounds for file with {ncols} columns.")
    return idx

def rising_edges_for_code(code_mask: np.ndarray, debounce_ms: int, sr_hint: int, ts=None):
    """
    Rising edges for a *single* code: True when (mask goes 0->1).
    If your marker is a single-sample pulse, this still works.
    Debounce collapses accidental double presses that occur too close in time.
    """
    edges = np.flatnonzero((code_mask[1:] & ~code_mask[:-1])) + 1
    if edges.size == 0 or debounce_ms <= 0:
        return edges
    min_gap = max(1, int(round((debounce_ms/1000.0) * sr_hint)))
    keep = [int(edges[0])]
    for e in edges[1:]:
        if e - keep[-1] >= min_gap:
            keep.append(int(e))
    return np.asarray(keep, dtype=int)

def detect_sr_from_ts(ts: np.ndarray, sr_hint: int):
    dt = np.diff(ts)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return sr_hint, None
    med = float(np.median(dt))
    if med <= 0 or not np.isfinite(med):
        return sr_hint, None
    return int(round(1.0/med)), med

def build_segments_for_code(ts, mk, code, debounce_ms, sr_hint):
    mask = (mk == code)
    edges = rising_edges_for_code(mask, debounce_ms, sr_hint, ts)
    segs = []
    active_t = None
    for idx in edges:
        t = float(ts[idx]) if idx < len(ts) else float("nan")
        if active_t is None:
            active_t = t  # start
        else:
            # end
            if np.isfinite(active_t) and np.isfinite(t) and t > active_t:
                segs.append(dict(
                    label=LABELS.get(code, str(code)),
                    code=code,
                    t_start=active_t,
                    t_end=t,
                    duration_s=t-active_t
                ))
            active_t = None
    # If odd count (no closing press), close at end of recording
    if active_t is not None and len(ts):
        t_last = float(ts[-1])
        if np.isfinite(t_last) and t_last > active_t:
            segs.append(dict(
                label=LABELS.get(code, str(code)),
                code=code,
                t_start=active_t,
                t_end=t_last,
                duration_s=t_last-active_t
            ))
    return segs

def main():
    ap = argparse.ArgumentParser(description="Build start/end segments from marker toggles per code.")
    ap.add_argument("--data_dir", required=True, help="Folder with RAW CSVs")
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--codes", type=int, nargs="+", default=[1,2,3,4,5,6])
    ap.add_argument("--ts-col", type=int, default=None, help="1-based or negative (-1=last, -2=last-2). Default: last-2")
    ap.add_argument("--mk-col", type=int, default=None, help="1-based or negative (-1=last). Default: last")
    ap.add_argument("--sr-hint", type=int, default=250)
    ap.add_argument("--debounce-ms", type=int, default=400, help="Collapse accidental double presses")
    ap.add_argument("--index_time", action="store_true",
                    help="Ignore timestamp column and use sample index / sr_hint as time (0, 1/sr, 2/sr, ...).")
    ap.add_argument("--write_manifest", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted([p for p in data_dir.glob(args.glob)
                    if p.is_file() and not p.name.endswith((".markers.csv",".markers.aligned.csv",".markers.segments.csv"))])
    if not files:
        raise SystemExit(f"No input CSVs match {data_dir}/{args.glob}")

    man_rows = []
    for raw in files:
        df = pd.read_csv(raw, sep=r"\s+|\t|,", engine="python", header=None)
        ncols = df.shape[1]
        ts_col0 = resolve_col(args.ts_col, ncols, fallback_idx=ncols-2)
        mk_col0 = resolve_col(args.mk_col, ncols, fallback_idx=ncols-1)

        if args.index_time:
            sr = args.sr_hint
            ts = np.arange(len(df), dtype=float) / float(sr)
            med_dt = 1.0/float(sr)
        else:
            ts = pd.to_numeric(df.iloc[:, ts_col0], errors="coerce").astype(float).to_numpy()
            sr, med_dt = detect_sr_from_ts(ts, args.sr_hint)
            if med_dt is None:
                # Fallback: index time
                sr = args.sr_hint
                ts = np.arange(len(df), dtype=float) / float(sr)
                med_dt = 1.0/float(sr)

        mk = pd.to_numeric(df.iloc[:, mk_col0], errors="coerce").fillna(0).to_numpy()
        mk = np.rint(mk).astype(int)  # ensure integers

        all_rows = []
        for code in args.codes:
            all_rows += build_segments_for_code(ts, mk, code, args.debounce_ms, sr)

        segs = pd.DataFrame(all_rows)
        if not segs.empty:
            segs["ts_col_used_1b"] = ts_col0+1
            segs["marker_col_used_1b"] = mk_col0+1
            segs["median_dt_s"] = float(med_dt)
            segs["sr_hz_est"] = float(1.0/med_dt)

        outp = raw.with_suffix(".markers.segments.csv")
        segs.to_csv(outp, index=False)

        counts = segs["label"].value_counts().to_dict() if not segs.empty else {}
        print(f"[OK] {raw.name}: segments={len(segs)} ts_col={ts_col0+1} mk_col={mk_col0+1} per-class={counts}")

        man_rows.append(dict(file=raw.name,
                             ts_col_1b=ts_col0+1, mk_col_1b=mk_col0+1,
                             segments=int(len(segs)), sr_est=float(1.0/med_dt) if med_dt else None,
                             per_class_json=json.dumps(counts)))

    if args.write_manifest:
        pd.DataFrame(man_rows).to_csv(data_dir / "segments_manifest.csv", index=False)
        print(f"[OK] wrote manifest -> {data_dir/'segments_manifest.csv'}")

if __name__ == "__main__":
    main()
