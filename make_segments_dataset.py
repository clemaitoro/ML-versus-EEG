#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = {1:"blink",2:"blink_double",3:"tap",4:"tap_double",5:"fist_hold",6:"brow",7:"start_rest",8:"end_rest"}

def rising_edges_bool(x_bool: np.ndarray) -> np.ndarray:
    nz = x_bool.astype(np.uint8)
    return np.flatnonzero((nz[1:] == 1) & (nz[:-1] == 0)) + 1

def detect_ts_col(df: pd.DataFrame, sr_hint: int = 250):
    target = 1.0 / max(1, sr_hint)
    best = (None, float("inf"), None)
    for col in df.columns:
        ts = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
        dt = np.diff(ts); dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.size < 50: continue
        med = float(np.median(dt)); err = abs(med - target)
        if err < best[1]: best = (col, err, med)
    return best  # (col, err, med_dt) or (None, inf, None)

def detect_mk_col(df: pd.DataFrame):
    best = (None, -1, 0)
    for col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        xr = np.rint(x).astype(int)
        frac_intish = float(np.mean(np.isfinite(x) & (np.abs(x - xr) < 1e-6)))
        frac_inrange = float(np.mean((xr>=0)&(xr<=8)))
        frac_zero = float(np.mean(xr==0))
        nz = (xr!=0).astype(np.uint8)
        edges = np.flatnonzero((nz[1:]==1)&(nz[:-1]==0))+1
        if frac_intish>0.98 and frac_inrange>0.98 and frac_zero>0.90 and len(edges)>10:
            score = len(edges) + frac_zero
            if score > best[1]:
                best = (col, score, len(edges))
    return best[0] if best[0] is not None else df.columns[-1]

def build_segments_for_code(ts: np.ndarray, mk: np.ndarray, code: int):
    mask = (mk == code)
    edges = rising_edges_bool(mask)
    segs = []; active = None
    for i in edges:
        t = float(ts[i]) if i < len(ts) else float("nan")
        if active is None:
            active = t
        else:
            t0 = active; active = None
            if np.isfinite(t0) and np.isfinite(t) and t > t0:
                segs.append(dict(label=LABELS.get(code, str(code)),
                                 code=code, t_start=t0, t_end=t, duration_s=t-t0))
    if active is not None and len(ts):
        t_last = float(ts[-1])
        if np.isfinite(t_last) and t_last > active:
            segs.append(dict(label=LABELS.get(code, str(code)),
                             code=code, t_start=active, t_end=t_last, duration_s=t_last-active))
    return segs

def inspect_columns(df: pd.DataFrame, sr_hint=250):
    ncols = df.shape[1]
    rows = []
    for j in range(ncols):
        col = df.columns[j]
        x = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
        dt = np.diff(x)
        pos = dt[(dt>0)&np.isfinite(dt)]
        med_dt = float(np.median(pos)) if pos.size else None
        xr = np.rint(x).astype(int)
        intish = float(np.mean(np.isfinite(x) & (np.abs(x - xr) < 1e-6)))
        inrange = float(np.mean((xr>=0)&(xr<=8)))
        zeros = float(np.mean(xr==0))
        nz = (xr!=0).astype(np.uint8)
        edges = int(np.flatnonzero((nz[1:]==1)&(nz[:-1]==0)).size)
        rows.append(dict(col_1b=j+1, med_dt=med_dt, sr_est=(None if not med_dt else 1.0/med_dt),
                         int_0_8=inrange, frac_zero=zeros, edges=edges))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Make toggle-based segments from RAW CSVs.")
    ap.add_argument("--data_dir", required=True, help="Folder containing RAW CSVs")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern (default: *.csv)")
    ap.add_argument("--codes", type=int, nargs="+", default=[1], help="Marker codes to toggle (start/end)")
    ap.add_argument("--ts-col", type=int, default=None, help="1-based timestamp column (default: autodetect)")
    ap.add_argument("--mk-col", type=int, default=None, help="1-based marker column (default: autodetect)")
    ap.add_argument("--sr-hint", type=int, default=250)
    ap.add_argument("--write_manifest", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="Print column summary for the first file and exit")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted([p for p in data_dir.glob(args.glob)
                    if p.is_file() and not p.name.endswith((".markers.csv",".markers.aligned.csv",".markers.segments.csv"))])
    if not files:
        raise SystemExit(f"No input CSVs match {data_dir}/{args.glob}")

    if args.inspect:
        df0 = pd.read_csv(files[0], sep=r"\s+|\t|,", engine="python", header=None)
        info = inspect_columns(df0, args.sr_hint)
        print(info.to_string(index=False))
        return

    manifest_rows = []
    for raw in files:
        df = pd.read_csv(raw, sep=r"\s+|\t|,", engine="python", header=None)
        ncols = df.shape[1]

        # Timestamp column (validate or autodetect)
        if args.ts_col is not None and 1 <= args.ts_col <= ncols:
            ts_col0 = args.ts_col - 1
            ts_arr = pd.to_numeric(df.iloc[:, ts_col0], errors="coerce").astype(float).to_numpy()
            dt = np.diff(ts_arr); dt = dt[(dt>0)&np.isfinite(dt)]
            med_dt = float(np.median(dt)) if dt.size else None
        else:
            ts_col0, _, med_dt = detect_ts_col(df, args.sr_hint)
            if ts_col0 is None:
                ts_col0 = df.columns[-2]
                ts_arr = pd.to_numeric(df.iloc[:, ts_col0], errors="coerce").astype(float).to_numpy()
                dt = np.diff(ts_arr); dt = dt[(dt>0)&np.isfinite(dt)]
                med_dt = float(np.median(dt)) if dt.size else None

        ts = pd.to_numeric(df.iloc[:, ts_col0], errors="coerce").astype(float).to_numpy()

        # Marker column (validate or autodetect)
        if args.mk_col is not None and 1 <= args.mk_col <= ncols:
            mk_col0 = args.mk_col - 1
        else:
            mk_col0 = detect_mk_col(df)

        mk = pd.to_numeric(df.iloc[:, mk_col0], errors="coerce").fillna(0).to_numpy().round().astype(int)

        all_rows = []
        for code in args.codes:
            all_rows.extend(build_segments_for_code(ts, mk, code))

        segs_df = pd.DataFrame(all_rows)
        if not segs_df.empty:
            segs_df["ts_col_used_1b"] = int(ts_col0)+1
            segs_df["marker_col_used_1b"] = int(mk_col0)+1
            segs_df["median_dt_s"] = med_dt
            segs_df["sr_hz_est"] = (None if med_dt is None else float(1.0/med_dt))

        outp = raw.with_suffix(".markers.segments.csv")
        segs_df.to_csv(outp, index=False)

        counts = segs_df["label"].value_counts().to_dict() if not segs_df.empty else {}
        print(f"[OK] {raw.name}: segments={len(segs_df)} ts_col={int(ts_col0)+1} mk_col={int(mk_col0)+1} per-class={counts}")

        manifest_rows.append(dict(file=raw.name, ts_col_1b=int(ts_col0)+1, mk_col_1b=int(mk_col0)+1,
                                  segments=int(len(segs_df)), sr_est=(None if med_dt is None else float(1.0/med_dt)),
                                  per_class_json=json.dumps(counts)))

    if args.write_manifest:
        man = pd.DataFrame(manifest_rows)
        man.to_csv(data_dir / "segments_manifest.csv", index=False)
        print(f"[OK] Wrote manifest -> {data_dir/'segments_manifest.csv'}")

if __name__ == "__main__":
    main()
