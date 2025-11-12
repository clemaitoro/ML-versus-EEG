#!/usr/bin/env python3
"""
Align markers (1..6) to physiological onsets using EEG + EMG with anomaly gating.

Markers:
  1 -> blink         (EEG low band)
  2 -> blink_double  (EEG low band)
  3 -> tap           (EMG C3)
  4 -> tap_double    (EMG C3)
  5 -> fist_hold     (EMG C3, sustained)
  6 -> brow          (EEG high band)
  7 -> start_rest    (passthrough)
  8 -> end_rest      (passthrough)

Key features vs. earlier versions
---------------------------------
- Robust **timestamp auto-detection** if --ts-col not provided (searches for ~1/sr step).
- **Blink/Brow anomaly path now sets `passed=True`** when the chosen detection is an anomaly and gates clear.
- **Confidence semantics**: anomaly → `z_peak`; baseline → baseline confidence; fallback → 0.
- Safe handling of NaNs / None for np.isfinite checks.

Outputs per file: <raw>.markers.aligned.csv with columns
  t_marker, marker_val, label, t_event, offset_s, confidence, passed,
  z_peak, dur_ms, min_ms_gate, details, sr_hz, channel_used

Usage examples
--------------
# Process a folder (all CSV except marker files)
python marker_align_v2.py data/*.csv --sr 250 --fp1 1 --fp2 2 --emg 3

# Single file with auto ts detection
python marker_align_v2.py BrainFlow-RAW_Lupica_Andrei_1_blink.csv --sr 250 --fp1 1 --fp2 2 --emg -1
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import sys
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend, iirnotch

# ---------- Config ----------
EEG_BLINK_BAND = (0.5, 15.0)
EEG_BROW_BAND  = (15.0, 120.0)
EMG_BAND       = (15.0, 55.0)
NOTCH_HZ       = 50.0

EEG_PRE_S      = 0.50
EEG_POST_S     = 1.00
ANOM_PRE_S     = 0.75
ANOM_POST_S    = 0.20

ENV_BLINK_S    = 0.08
ENV_BROW_S     = 0.08
ENV_EMG_S      = 0.05

MAD_K_BLINK    = 4.0
MAD_K_BROW     = 4.5

ANOM_Z_THR     = 2.0
ANOM_MIN_MS    = 50.0
TAP_MIN_MS     = 40.0
FIST_MIN_MS    = 400.0
BLINK_MIN_MS   = 40.0
BROW_MIN_MS    = 60.0

DEFAULT_DEBOUNCE_MS = 250

MARKER_TO_LABEL = {
    1: "blink",
    2: "blink_double",
    3: "tap",
    4: "tap_double",
    5: "fist_hold",
    6: "brow",
    7: "start_rest",
    8: "end_rest",
}
VALID_MARKERS = set(MARKER_TO_LABEL)
REST_MARKERS = {7, 8}

# ---------- Utilities ----------
def butter_band(sr: float, lo: float, hi: float, order: int = 4):
    ny = 0.5 * sr
    lo = max(lo, 0.1)
    hi = min(hi, 0.45 * sr)
    if lo >= hi:
        hi = min(lo * 1.3, 0.45 * sr - 1e-3)
    b, a = butter(order, [lo / ny, hi / ny], btype="bandpass")
    return b, a


def notch(x: np.ndarray, sr: float, f0: float = NOTCH_HZ, Q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(f0 / (sr / 2.0), Q=Q)
    return filtfilt(b, a, x)


def filter_band(x: np.ndarray, sr: float, band: Tuple[float, float], notch_hz: float = NOTCH_HZ) -> np.ndarray:
    x = detrend(x, type="constant")
    if notch_hz and notch_hz > 0:
        x = notch(x, sr, f0=notch_hz)
    b, a = butter_band(sr, band[0], band[1], 4)
    return filtfilt(b, a, x).astype(np.float32)


def moving_rms(x: np.ndarray, win_samps: int) -> np.ndarray:
    w = max(1, int(round(win_samps)))
    c = np.convolve(x ** 2, np.ones(w) / w, mode="same")
    return np.sqrt(np.maximum(c, 0.0) + 1e-12)


def mad_thr(x: np.ndarray, k: float) -> Tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return med + k * mad, med


def infer_sr(ts: np.ndarray, sr_hint: int) -> int:
    dt = np.diff(ts)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return sr_hint
    med = float(np.median(dt))
    if not np.isfinite(med) or med <= 0:
        return sr_hint
    sr = 1.0 / med
    return int(round(sr)) if 100 <= sr <= 1000 else sr_hint


def autodetect_ts_col(df: pd.DataFrame, sr_hint: int) -> Tuple[Optional[int], Optional[float]]:
    """Find a column whose median dt is closest to 1/sr_hint (e.g., ≈0.004s for 250 Hz)."""
    target = 1.0 / max(1, sr_hint)
    best = (None, float("inf"), None)
    for col in df.columns:
        ts = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
        dt = np.diff(ts)
        dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.size < 50:
            continue
        med = float(np.median(dt))
        err = abs(med - target)
        if err < best[1]:
            best = (col, err, med)
    return best[0], best[2]

# ---------- EEG baseline onset ----------
def baseline_onset(env: np.ndarray, sr: float, pre_frac: float, k: float, min_ms: float, rollback_frac: float) -> Tuple[Optional[int], float]:
    L = len(env)
    if L < 10:
        return None, 0.0
    pre_end = max(1, min(L - 1, int(round(pre_frac * L))))
    base = env[:pre_end]
    thr, base_med = mad_thr(base, k=k)

    post = env[pre_end:]
    if post.size == 0:
        return None, 0.0
    p_rel = int(np.argmax(post))
    p = pre_end + p_rel

    if env[p] < thr:
        return None, 0.0

    # min contiguous duration around peak above threshold
    i0 = p
    while i0 > pre_end and env[i0 - 1] >= thr:
        i0 -= 1
    i1 = p
    while i1 + 1 < L and env[i1 + 1] >= thr:
        i1 += 1
    dur_ms = (i1 - i0 + 1) * 1000.0 / sr
    if dur_ms < min_ms:
        return None, 0.0

    # rollback toward baseline
    thr_on = base_med + rollback_frac * (env[p] - base_med)
    j = p
    while j > 0 and env[j] > thr_on:
        j -= 1

    conf = (env[p] - thr) / (np.std(env) + 1e-6)
    return j, float(conf)

# ---------- EMG robust-Z anomaly ----------
@dataclass
class Anom:
    onset_idx: int
    t_event: float
    z_peak: float
    dur_ms: float
    passed: bool
    method: str
    channel: str


def strongest_anomaly(ts: np.ndarray, x: np.ndarray, sr: float, t0: float, t1: float, t_marker: float,
                      band: Tuple[float, float], z_thr: float, min_ms_gate: float,
                      prefer_pre: bool = True, rollback_frac: float = 0.25,
                      notch_hz: float = NOTCH_HZ, channel_name: str = "") -> Optional[Anom]:
    lo = np.searchsorted(ts, t0)
    hi = np.searchsorted(ts, t1)
    if hi - lo < 10:
        return None

    xe = filter_band(x[lo:hi], sr, band, notch_hz)
    env = moving_rms(np.abs(xe), int(round(ENV_EMG_S * sr)))

    seg_ts = ts[lo:hi]
    mid = int(np.searchsorted(seg_ts, t_marker - 1e-12))
    mid = max(1, min(len(env) - 1, mid))

    base = env[:mid]
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med))) + 1e-12
    z = (env - med) / mad

    j_pre, z_pre = (int(np.argmax(z[:mid])), float(np.max(z[:mid]))) if mid > 0 else (0, -np.inf)
    j_post, z_post = (int(np.argmax(z[mid:])) + mid, float(np.max(z[mid:]))) if mid < len(z) else (mid, -np.inf)
    if prefer_pre:
        idx, zpk = (j_pre, z_pre) if z_pre >= z_post else (j_post, z_post)
    else:
        idx, zpk = (j_post, z_post) if z_post >= z_pre else (j_pre, z_pre)

    thr_abs = med + z_thr * (np.std(base) + 1e-9)
    L = len(env)
    i0 = idx
    while i0 > 0 and env[i0 - 1] >= thr_abs:
        i0 -= 1
    i1 = idx
    while i1 + 1 < L and env[i1 + 1] >= thr_abs:
        i1 += 1
    dur_ms = (i1 - i0 + 1) * 1000.0 / sr

    passed = (zpk >= z_thr) and (dur_ms >= min_ms_gate)

    thr_on = med + rollback_frac * (env[idx] - med)
    j = idx
    while j > 0 and env[j] > thr_on:
        j -= 1

    return Anom(
        onset_idx=j,
        t_event=float(ts[lo + j]),
        z_peak=float(zpk),
        dur_ms=float(dur_ms),
        passed=bool(passed),
        method="anomaly",
        channel=channel_name,
    )

# ---------- Markers ----------
def rising_edge_indices(marker_vec: np.ndarray, sr: float, debounce_ms: int) -> np.ndarray:
    nz = (marker_vec != 0).astype(np.uint8)
    edges = np.flatnonzero((nz[1:] == 1) & (nz[:-1] == 0)) + 1
    if edges.size == 0:
        return edges
    min_gap = max(1, int(round((debounce_ms / 1000.0) * sr)))
    keep = [edges[0]]
    for i in range(1, len(edges)):
        if edges[i] - keep[-1] >= min_gap:
            keep.append(edges[i])
    return np.array(keep, dtype=int)

# ---------- Core per-file processing ----------
def process_file(raw_csv: Path, sr_hint: int, ts_col: Optional[int], mk_col: Optional[int],
                 fp1_i: int, fp2_i: int, emg_i: int,
                 debounce_ms: int,
                 anom_pre_s: float, anom_post_s: float, anom_z: float, anom_min_ms: float,
                 include_eeg_for_emg: bool = False) -> Path:
    df = pd.read_csv(raw_csv, sep=r"\s+|\t|,", engine="python", header=None)

    # columns
    if ts_col is None:
        det_col, med_dt = autodetect_ts_col(df, sr_hint)
        ts_col = det_col if det_col is not None else df.columns[-2]
    if mk_col is None:
        mk_col = df.columns[-1]

    ts = pd.to_numeric(df[ts_col], errors="coerce").astype(float).to_numpy()
    mk = pd.to_numeric(df[mk_col], errors="coerce").fillna(0).astype(int).to_numpy()
    sr = infer_sr(ts, sr_hint)

    c1 = pd.to_numeric(df.iloc[:, fp1_i], errors="coerce").astype(float).to_numpy()
    c2 = pd.to_numeric(df.iloc[:, fp2_i], errors="coerce").astype(float).to_numpy()
    c3 = None if emg_i < 0 else pd.to_numeric(df.iloc[:, emg_i], errors="coerce").astype(float).to_numpy()

    mk = np.where(np.isin(mk, list(VALID_MARKERS)), mk, 0)
    edge_idx = rising_edge_indices(mk, sr, debounce_ms)

    rows = []
    for i in edge_idx:
        mval = int(mk[i])
        if mval == 0:
            continue
        label = MARKER_TO_LABEL[mval]
        t_marker = float(ts[i])

        if mval in REST_MARKERS:
            rows.append(dict(
                t_marker=t_marker, marker_val=mval, label=label,
                t_event=t_marker, offset_s=0.0, confidence=1.0, passed=True,
                z_peak=np.nan, dur_ms=np.nan, min_ms_gate=np.nan,
                details=json.dumps({"source": "marker_only_rest"}),
                sr_hz=sr, channel_used="marker"
            ))
            continue

        # windows
        if mval in (1, 2, 6):
            t0 = t_marker - EEG_PRE_S
            t1 = t_marker + EEG_POST_S
        else:
            t0 = t_marker - anom_pre_s
            t1 = t_marker + anom_post_s

        det = None
        channel_used = None
        conf = 0.0
        zpk = np.nan
        dur_ms = np.nan
        min_ms_gate = np.nan
        passed = False

        if mval in (1, 2):  # blink
            lo = np.searchsorted(ts, t0); hi = np.searchsorted(ts, t1)
            cand: List[tuple] = []
            if hi - lo >= 10:
                x1 = filter_band(c1[lo:hi], sr, EEG_BLINK_BAND, NOTCH_HZ)
                x2 = filter_band(c2[lo:hi], sr, EEG_BLINK_BAND, NOTCH_HZ)
                xb = x1 - x2
                env = moving_rms(xb, int(round(ENV_BLINK_S * sr)))
                onset_idx, bconf = baseline_onset(env, sr, pre_frac=0.5, k=MAD_K_BLINK, min_ms=BLINK_MIN_MS, rollback_frac=0.2)
                if onset_idx is not None:
                    cand.append((float(ts[lo + onset_idx]), "eeg_baseline", None, None, "c1-c2", True, bconf))
            a2 = strongest_anomaly(ts, c2, sr, t0, t1, t_marker, EEG_BLINK_BAND, ANOM_Z_THR, BLINK_MIN_MS, prefer_pre=True, channel_name="c2")
            if a2 is not None and a2.passed:
                cand.append((a2.t_event, a2.method, a2.z_peak, a2.dur_ms, a2.channel, a2.passed, None))
            if c3 is not None:
                a3 = strongest_anomaly(ts, c3, sr, t0, t1, t_marker, EMG_BAND, ANOM_Z_THR, BLINK_MIN_MS, prefer_pre=True, channel_name="c3")
                if a3 is not None and a3.passed:
                    cand.append((a3.t_event, a3.method, a3.z_peak, a3.dur_ms, a3.channel, a3.passed, None))
            if cand:
                anoms = [c for c in cand if c[1] == "anomaly"]
                if anoms:
                    det, method, zpk, dur_ms, channel_used, passed, _ = max(anoms, key=lambda t: t[2])
                    conf = float(zpk) if (zpk is not None and np.isfinite(zpk)) else 0.0
                else:
                    det, method, zpk, dur_ms, channel_used, passed, bconf = cand[0]
                    conf = float(bconf) if (bconf is not None) else 0.0

        elif mval == 6:  # brow
            lo = np.searchsorted(ts, t0); hi = np.searchsorted(ts, t1)
            cand: List[tuple] = []
            if hi - lo >= 10:
                x1 = np.abs(filter_band(c1[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                x2 = np.abs(filter_band(c2[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                env = moving_rms(np.minimum(x1, x2), int(round(ENV_BROW_S * sr)))
                onset_idx, bconf = baseline_onset(env, sr, pre_frac=0.5, k=MAD_K_BROW, min_ms=BROW_MIN_MS, rollback_frac=0.25)
                if onset_idx is not None:
                    cand.append((float(ts[lo + onset_idx]), "eeg_baseline", None, None, "min(c1,c2)", True, bconf))
            a2 = strongest_anomaly(ts, c2, sr, t0, t1, t_marker, EEG_BROW_BAND, ANOM_Z_THR, BROW_MIN_MS, prefer_pre=True, channel_name="c2")
            if a2 is not None and a2.passed:
                cand.append((a2.t_event, a2.method, a2.z_peak, a2.dur_ms, a2.channel, a2.passed, None))
            if cand:
                anoms = [c for c in cand if c[1] == "anomaly"]
                if anoms:
                    det, method, zpk, dur_ms, channel_used, passed, _ = max(anoms, key=lambda t: t[2])
                    conf = float(zpk) if (zpk is not None and np.isfinite(zpk)) else 0.0
                else:
                    det, method, zpk, dur_ms, channel_used, passed, bconf = cand[0]
                    conf = float(bconf) if (bconf is not None) else 0.0

        else:  # EMG tasks
            if c3 is not None:
                min_gate = TAP_MIN_MS if mval in (3, 4) else FIST_MIN_MS if mval == 5 else ANOM_MIN_MS
                a3 = strongest_anomaly(ts, c3, sr, t0, t1, t_marker, EMG_BAND, ANOM_Z_THR, min_gate, prefer_pre=True, channel_name="c3")
                if a3 is not None:
                    det = a3.t_event
                    zpk = a3.z_peak
                    dur_ms = a3.dur_ms
                    passed = a3.passed
                    min_ms_gate = min_gate
                    channel_used = "c3"
                    conf = float(zpk) if (zpk is not None and np.isfinite(zpk)) else 0.0
            if det is None and include_eeg_for_emg:
                lo = np.searchsorted(ts, t0); hi = np.searchsorted(ts, t1)
                if hi - lo >= 10:
                    x1 = np.abs(filter_band(c1[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                    x2 = np.abs(filter_band(c2[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                    env = moving_rms(np.maximum(x1, x2), int(round(ENV_EMG_S * sr)))
                    onset_idx, bconf = baseline_onset(env, sr, pre_frac=0.5, k=4.0, min_ms=40.0, rollback_frac=0.25)
                    if onset_idx is not None:
                        det = float(ts[lo + onset_idx])
                        channel_used = "max(c1,c2)"
                        conf = float(bconf)

        # fallback
        if det is None:
            rows.append(dict(
                t_marker=t_marker, marker_val=mval, label=label,
                t_event=t_marker, offset_s=0.0,
                confidence=float(conf), passed=False,
                z_peak=float(zpk) if (zpk is not None and np.isfinite(zpk)) else np.nan,
                dur_ms=float(dur_ms) if (dur_ms is not None and np.isfinite(dur_ms)) else np.nan,
                min_ms_gate=float(min_ms_gate) if (min_ms_gate is not None and np.isfinite(min_ms_gate)) else (
                    TAP_MIN_MS if mval in (3, 4) else FIST_MIN_MS if mval == 5 else np.nan),
                details=json.dumps({"note": "fallback_to_marker"}),
                sr_hz=sr, channel_used=channel_used or "fallback"
            ))
        else:
            rows.append(dict(
                t_marker=t_marker, marker_val=mval, label=label,
                t_event=det, offset_s=float(det - t_marker),
                confidence=float(conf), passed=bool(passed),
                z_peak=float(zpk) if (zpk is not None and np.isfinite(zpk)) else np.nan,
                dur_ms=float(dur_ms) if (dur_ms is not None and np.isfinite(dur_ms)) else np.nan,
                min_ms_gate=float(min_ms_gate) if (min_ms_gate is not None and np.isfinite(min_ms_gate)) else (
                    TAP_MIN_MS if mval in (3, 4) else FIST_MIN_MS if mval == 5 else np.nan),
                details=json.dumps({"detector": ("anomaly" if (channel_used in ("c2","c3")) else "eeg_baseline")}),
                sr_hz=sr, channel_used=channel_used or ""
            ))

    out = pd.DataFrame(rows, columns=[
        "t_marker", "marker_val", "label", "t_event", "offset_s", "confidence", "passed",
        "z_peak", "dur_ms", "min_ms_gate", "details", "sr_hz", "channel_used"
    ])
    out_path = raw_csv.with_suffix(".markers.aligned.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] {raw_csv.name}: wrote {out_path.name}  ({len(out)} events, sr≈{sr} Hz)")
    return out_path

# ---------- CLI ----------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Align markers to physiological onsets (EEG + EMG anomaly; fallback-safe).")
    ap.add_argument("files", nargs="+", help="RAW csv files or globs; directories expand to *.csv")
    ap.add_argument("--sr", type=int, default=250, help="Sample rate hint (for ts autodetect)")
    ap.add_argument("--ts-col", type=int, default=None, help="Timestamp column index (auto if omitted)")
    ap.add_argument("--marker-col", type=int, default=None, help="Marker column index (default: last-1)")
    ap.add_argument("--fp1", type=int, default=0, help="Fp1 column index")
    ap.add_argument("--fp2", type=int, default=1, help="Fp2 column index")
    ap.add_argument("--emg", type=int, default=2, help="C3 (emg) column index; set -1 if absent")
    ap.add_argument("--debounce-ms", type=int, default=DEFAULT_DEBOUNCE_MS, help="Debounce window for marker presses")
    ap.add_argument("--anom-pre-s", type=float, default=ANOM_PRE_S, help="Pre-marker window (s) for EMG anomaly")
    ap.add_argument("--anom-post-s", type=float, default=ANOM_POST_S, help="Post-marker window (s) for EMG anomaly")
    ap.add_argument("--anom-z", type=float, default=ANOM_Z_THR, help="Robust-Z threshold for EMG anomaly")
    ap.add_argument("--anom-min-ms", type=float, default=ANOM_MIN_MS, help="Default min duration (ms) for anomaly")
    ap.add_argument("--include-eeg-for-emg", action="store_true", help="If EMG fails, try EEG hi-band as fallback")
    return ap


def expand_inputs(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in patterns:
        P = Path(p)
        if P.is_dir():
            files += sorted([q for q in P.glob("*.csv") if not q.name.endswith((".markers.csv", ".markers.aligned.csv"))])
        else:
            # globs handled by shell on Unix; on Windows, expand manually
            if any(ch in p for ch in "*?["):
                files += [Path(s) for s in sorted(map(str, P.parent.glob(P.name))) if s.endswith(".csv") and not s.endswith((".markers.csv",".markers.aligned.csv"))]
            else:
                files.append(P)
    return files


def main():
    ap = build_argparser(); args = ap.parse_args()
    files = expand_inputs(args.files)
    if not files:
        print("No input files found.")
        sys.exit(1)

    for f in files:
        try:
            process_file(
                f, args.sr, args.ts_col, args.marker_col,
                args.fp1, args.fp2, args.emg, args.debounce_ms,
                args.anom_pre_s, args.anom_post_s, args.anom_z, args.anom_min_ms,
                include_eeg_for_emg=args.include_eeg_for_emg,
            )
        except Exception as e:
            print(f"[ERR] {f}: {e}")


if __name__ == "__main__":
    main()
