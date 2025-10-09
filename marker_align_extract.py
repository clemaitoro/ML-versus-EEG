#!/usr/bin/env python3
"""
Align markers to nearest physiological onset (baseline-gated), with safe fallback.

Marker map:
  1 -> blink
  2 -> blink_double
  3 -> tap
  4 -> tap_double
  5 -> fist_hold
  6 -> brow
  7 -> start_rest     (passthrough; no detection)
  8 -> end_rest       (passthrough; no detection)

What it does
- Reads RAW BrainFlow CSV (no header). Assumes last 2 columns are [timestamp, marker] unless overridden.
- Detects rising-edge button presses with debounce (default 250 ms).
- For each press, chooses the matching detector and:
    * Builds a window [t_marker - SEARCH_PRE_S, t_marker + SEARCH_POST_S].
    * Computes an envelope (RMS) in the appropriate band.
    * Estimates pre-marker baseline and requires k*MAD rise for >= min_ms.
    * Rolls back to an early onset at baseline + α*(peak-baseline).
- If nothing clears the gate, falls back to t_marker (so you always have a timestamp).
- Writes <raw>.markers.aligned.csv with: t_marker, marker_val, label, t_event, confidence, details, sr_hz.

Typical runs
------------
# Brow/ blink (EEG only)
python align_markers_baseline.py RAW_brow.csv --sr 250 --fp1 0 --fp2 1 --emg -1

# Tap / fist (EMG present)
python align_markers_baseline.py RAW_tap.csv  --sr 250 --fp1 0 --fp2 1 --emg 2
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend, iirnotch

# ---------------- Config ----------------
# Column indices (0-based). Override via CLI if needed.
FP1_COL = 0
FP2_COL = 1
EMG_COL = 2   # set --emg -1 if EMG not recorded

# Bands and DSP
EEG_BLINK_BAND = (0.5, 15.0)
EEG_BROW_BAND  = (15.0, 120.0)   # wider helps real brow EMG on forehead leads
EMG_BAND       = (15.0, 55.0)
NOTCH_HZ       = 50.0

# Search window around marker
SEARCH_PRE_S   = 0.50
SEARCH_POST_S  = 1.00

# Envelope (RMS) window lengths (in seconds)
ENV_BLINK_S = 0.08
ENV_TAP_S   = 0.05
ENV_BROW_S  = 0.08
ENV_FIST_S  = 0.12

# Baseline gate parameters
MAD_K_BLINK = 4.0
MAD_K_TAP   = 5.5
MAD_K_BROW  = 4.5
FIST_MIN_MS = 400  # sustained duration for fist gate

# Debounce between marker presses
DEFAULT_DEBOUNCE_MS = 250

# Marker mapping
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
VALID_MARKERS = set(MARKER_TO_LABEL.keys())
REST_MARKERS = {7, 8}

# ---------------- Utilities ----------------
def butter_band(sr, lo, hi, order=4):
    ny = 0.5 * sr
    lo = max(lo, 0.1)
    hi = min(hi, 0.45 * sr)
    if lo >= hi:
        hi = min(lo * 1.3, 0.45 * sr - 1e-3)
    b, a = butter(order, [lo/ny, hi/ny], btype="bandpass")
    return b, a

def notch(x, sr, f0=50.0, Q=30):
    b, a = iirnotch(f0 / (sr / 2.0), Q=Q)
    return filtfilt(b, a, x)

def filter_band(x, sr, band, notch_hz=50.0):
    x = detrend(x, type="constant")
    if notch_hz and notch_hz > 0:
        x = notch(x, sr, f0=notch_hz)
    b, a = butter_band(sr, band[0], band[1], 4)
    return filtfilt(b, a, x).astype(np.float32)

def moving_rms(x, win_samps):
    win = max(1, int(win_samps))
    c = np.convolve(x**2, np.ones(win)/win, mode="same")
    return np.sqrt(np.maximum(c, 0.0) + 1e-9)

def mad_thr(x, k):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    return med + k * mad, med

def infer_sr(ts, sr_hint):
    dt = np.diff(ts)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return sr_hint
    med = float(np.median(dt)) if np.isfinite(np.median(dt)) else 1.0 / sr_hint
    sr = 1.0 / med if med > 0 else sr_hint
    return int(round(sr)) if 100 <= sr <= 1000 else sr_hint

# ---------------- Baseline-gated onset ----------------
def baseline_onset(env, sr, pre_frac=0.5, k=4.0, min_ms=40, rollback_frac=0.2):
    """
    env: envelope over [t0..t1] window (numpy 1D)
    pre_frac: fraction of the window considered 'baseline' before marker
    k: MAD-multiple above baseline required to accept an event
    min_ms: minimum contiguous duration above threshold to accept (ms)
    rollback_frac: onset is baseline + rollback_frac*(peak-baseline)
    returns: (onset_idx or None, conf)
    """
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

    if env[p] < thr:  # not elevated enough
        return None, 0.0

    # require min contiguous samples above thr around the peak
    i0 = p
    while i0 > pre_end and env[i0 - 1] >= thr:
        i0 -= 1
    i1 = p
    while i1 + 1 < L and env[i1 + 1] >= thr:
        i1 += 1
    dur_ms = (i1 - i0 + 1) * 1000.0 / sr
    if dur_ms < min_ms:
        return None, 0.0

    # roll back to a softer onset close to baseline
    thr_on = base_med + rollback_frac * (env[p] - base_med)
    j = p
    while j > 0 and env[j] > thr_on:
        j -= 1

    conf = (env[p] - thr) / (np.std(env) + 1e-6)
    return j, float(conf)

# ---------------- Detectors ----------------
@dataclass
class Det:
    t_event: float
    conf: float
    details: dict

def detect_blink(ts, fp1, fp2, sr, t0, t1):
    lo = np.searchsorted(ts, t0); hi = np.searchsorted(ts, t1)
    if hi - lo < 10: return None
    x1 = filter_band(fp1[lo:hi], sr, EEG_BLINK_BAND, NOTCH_HZ)
    x2 = filter_band(fp2[lo:hi], sr, EEG_BLINK_BAND, NOTCH_HZ)
    xb = x1 - x2
    env = moving_rms(xb, ENV_BLINK_S * sr)
    onset_idx, conf = baseline_onset(env, sr, pre_frac=0.5, k=MAD_K_BLINK, min_ms=40, rollback_frac=0.2)
    if onset_idx is None: return None
    return Det(t_event=float(ts[lo + onset_idx]), conf=conf, details={})

def detect_tap(ts, emg, sr, t0, t1):
    if emg is None: return None
    lo = np.searchsorted(ts, t0); hi = np.searchsorted(ts, t1)
    if hi - lo < 10: return None
    xe = filter_band(emg[lo:hi], sr, EMG_BAND, NOTCH_HZ)
    env = moving_rms(np.abs(xe), ENV_TAP_S * sr)
    onset_idx, conf = baseline_onset(env, sr, pre_frac=0.5, k=MAD_K_TAP, min_ms=40, rollback_frac=0.2)
    if onset_idx is None: return None
    return Det(t_event=float(ts[lo + onset_idx]), conf=conf, details={})

def detect_fist(ts, emg, sr, t0, t1):
    if emg is None: return None
    lo = np.searchsorted(ts, t0); hi = np.searchsorted(ts, t1)
    if hi - lo < 10: return None
    xe = filter_band(emg[lo:hi], sr, EMG_BAND, NOTCH_HZ)
    env = moving_rms(np.abs(xe), ENV_FIST_S * sr)
    # Use baseline gate with sustained min duration
    onset_idx, conf = baseline_onset(env, sr, pre_frac=0.5, k=5.0, min_ms=FIST_MIN_MS, rollback_frac=0.2)
    if onset_idx is None: return None
    return Det(t_event=float(ts[lo + onset_idx]), conf=conf, details={"dur_gate_ms": FIST_MIN_MS})

def detect_brow(ts, fp1, fp2, sr, t0, t1):
    lo = np.searchsorted(ts, t0); hi = np.searchsorted(ts, t1)
    if hi - lo < 10: return None
    x1 = np.abs(filter_band(fp1[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
    x2 = np.abs(filter_band(fp2[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
    # both-eyebrow agreement by taking the minimum envelope
    env = moving_rms(np.minimum(x1, x2), ENV_BROW_S * sr)
    onset_idx, conf = baseline_onset(env, sr, pre_frac=0.5, k=MAD_K_BROW, min_ms=60, rollback_frac=0.25)
    if onset_idx is None: return None
    return Det(t_event=float(ts[lo + onset_idx]), conf=conf, details={})

# ---------------- Markers ----------------
def rising_edge_indices(marker_vec, sr, debounce_ms=DEFAULT_DEBOUNCE_MS):
    nz = (marker_vec != 0).astype(np.uint8)
    edges = np.flatnonzero((nz[1:] == 1) & (nz[:-1] == 0)) + 1
    if edges.size == 0:
        return edges
    min_gap = max(1, int(round((debounce_ms/1000.0) * sr)))
    keep = [edges[0]]
    for i in range(1, len(edges)):
        if edges[i] - keep[-1] >= min_gap:
            keep.append(edges[i])
    return np.array(keep, dtype=int)

# ---------------- Core ----------------
def process_file(raw_csv: Path, sr_hint: int, ts_col: int|None, mk_col: int|None,
                 fp1_i: int, fp2_i: int, emg_i: int, debounce_ms: int):
    df = pd.read_csv(raw_csv, sep=r"\s+|\t|,", engine="python", header=None)

    # columns
    if ts_col is None: ts_col = df.columns[-2]
    if mk_col is None: mk_col = df.columns[-1]

    ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(float)
    mk = pd.to_numeric(df[mk_col], errors="coerce").fillna(0).to_numpy(int)
    sr = infer_sr(ts, sr_hint)

    fp1 = pd.to_numeric(df.iloc[:, fp1_i], errors="coerce").to_numpy(float)
    fp2 = pd.to_numeric(df.iloc[:, fp2_i], errors="coerce").to_numpy(float)
    emg = None if emg_i < 0 else pd.to_numeric(df.iloc[:, emg_i], errors="coerce").to_numpy(float)

    # only valid markers; rising edges → one event per press
    mk = np.where(np.isin(mk, list(VALID_MARKERS)), mk, 0)
    edge_idx = rising_edge_indices(mk, sr, debounce_ms)

    rows = []
    for i in edge_idx:
        mval = int(mk[i])
        if mval == 0:
            continue
        label = MARKER_TO_LABEL[mval]
        t_marker = float(ts[i])

        # rest markers: passthrough (no detection)
        if mval in REST_MARKERS:
            rows.append(dict(
                t_marker=t_marker, marker_val=mval, label=label,
                t_event=t_marker, confidence=1.0,
                details={"source": "marker_only_rest"}, sr_hz=sr
            ))
            continue

        # search window
        t0 = t_marker - SEARCH_PRE_S
        t1 = t_marker + SEARCH_POST_S

        det = None
        if mval in (1, 2):      # blink / blink_double
            det = detect_blink(ts, fp1, fp2, sr, t0, t1)
        elif mval in (3, 4):    # tap / tap_double
            det = detect_tap(ts, emg, sr, t0, t1)
        elif mval == 5:         # fist
            det = detect_fist(ts, emg, sr, t0, t1)
        elif mval == 6:         # brow
            det = detect_brow(ts, fp1, fp2, sr, t0, t1)

        # fallback if detector fails (always produce a timestamp)
        if det is None:
            rows.append(dict(
                t_marker=t_marker, marker_val=mval, label=label,
                t_event=t_marker, confidence=0.0,
                details={"note": "fallback_to_marker"}, sr_hz=sr
            ))
        else:
            rows.append(dict(
                t_marker=t_marker, marker_val=mval, label=label,
                t_event=det.t_event, confidence=float(det.conf),
                details=det.details, sr_hz=sr
            ))

    out = pd.DataFrame(rows, columns=["t_marker","marker_val","label","t_event","confidence","details","sr_hz"])
    out_path = raw_csv.with_suffix(".markers.aligned.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] {raw_csv.name}: wrote {out_path.name}  ({len(out)} events, sr≈{sr} Hz)")
    return out_path

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Align markers to physiological onsets (baseline-gated; fallback-safe).")
    ap.add_argument("files", nargs="+", help="RAW csv files")
    ap.add_argument("--sr", type=int, default=250, help="Sample rate hint (used if timestamps are odd)")
    ap.add_argument("--ts-col", type=int, default=None, help="Timestamp column index (default: last-2)")
    ap.add_argument("--marker-col", type=int, default=None, help="Marker column index (default: last-1)")
    ap.add_argument("--fp1", type=int, default=FP1_COL, help="Fp1 column index")
    ap.add_argument("--fp2", type=int, default=FP2_COL, help="Fp2 column index")
    ap.add_argument("--emg", type=int, default=EMG_COL, help="EMG column index; set -1 if EMG not recorded")
    ap.add_argument("--debounce-ms", type=int, default=DEFAULT_DEBOUNCE_MS, help="Debounce window for marker presses")
    args = ap.parse_args()

    for f in args.files:
        process_file(Path(f), args.sr, args.ts_col, args.marker_col,
                     args.fp1, args.fp2, args.emg, args.debounce_ms)

if __name__ == "__main__":
    main()
