#!/usr/bin/env python3
#deprecated
"""
Align markers to nearest physiological onset with anomaly enforcement (EEG + EMG),
including C3 integration for tap and fist-hold based on empirical findings.

Marker map:
  1 -> blink
  2 -> blink_double
  3 -> tap
  4 -> tap_double
  5 -> fist_hold
  6 -> brow
  7 -> start_rest     (passthrough; no detection)
  8 -> end_rest       (passthrough; no detection)

What this does
--------------
- Reads RAW BrainFlow CSV (no header). Assumes last 2 columns are [timestamp, marker] unless overridden.
- Finds rising-edge button presses with debounce (default 250 ms).
- For each press:
    * Builds a window [t_marker - pre_s, t_marker + post_s] (defaults: 0.75 s / 0.20 s for EMG tasks).
    * For EEG tasks (blink/brow): bandpass, envelope (RMS), baseline-gated onset (MAD multiple), rollback.
    * For EMG tasks (tap/fist): bandpass on C3, envelope (RMS), robust-Z vs pre-baseline, choose strongest anomaly
      (prefer pre-marker), require min contiguous duration; rollback toward baseline.
    * If nothing clears the gate, falls back to t_marker (so you always get an event timestamp).
- Writes <raw>.markers.aligned.csv with: t_marker, marker_val, label, t_event, offset_s, confidence, passed,
  z_peak, dur_ms, min_ms_gate, details, sr_hz, channel_used.

Empirical defaults from your files
----------------------------------
- C3 (EMG) is primary for tap and fist-hold.
- Typical onset is ~0.48 s before the click; code prefers pre-marker anomalies automatically.

Examples
--------
# Tap (C3 on col 3 here), EEG on cols 0/1
python marker_extractor_integrated.py RAW_tap.csv  --sr 250 --fp1 0 --fp2 1 --emg 3 \
  --anom-pre-s 0.75 --anom-post-s 0.20 --anom-z 2.0 --anom-min-ms 50

# Fist-hold (C3 col 3), same windows, stricter sustain gate is auto-applied (>=400 ms)
python marker_extractor_integrated.py RAW_fist.csv --sr 250 --fp1 0 --fp2 1 --emg 3

# Multiple files at once
python marker_extractor_integrated.py data/*.csv --sr 250 --fp1 0 --fp2 1 --emg 3
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend, iirnotch

# ------------- Config -------------
# Column indices (0-based). Override via CLI.
FP1_COL_DEFAULT = 0  # c1
FP2_COL_DEFAULT = 1  # c2
EMG_COL_DEFAULT = 2  # c3 (set --emg 3 for your provided tap/fist files)

# Bands & DSP
EEG_BLINK_BAND = (0.5, 15.0)
EEG_BROW_BAND = (15.0, 120.0)
EMG_BAND = (15.0, 55.0)
NOTCH_HZ = 50.0

# Windows
EEG_PRE_S = 0.50
EEG_POST_S = 1.00
ANOM_PRE_S = 0.75  # for EMG tasks (tap/fist) and generic anomaly search
ANOM_POST_S = 0.20

# Envelope (RMS) window lengths (s)
ENV_BLINK_S = 0.08
ENV_BROW_S = 0.08
ENV_EMG_S = 0.05

# Baseline gate (EEG)
MAD_K_BLINK = 4.0
MAD_K_BROW = 4.5

# Robust-Z gate (anomaly)
ANOM_Z_THR = 2.0
ANOM_MIN_MS = 50.0
TAP_MIN_MS = 40.0
FIST_MIN_MS = 400.0
BLINK_MIN_MS = 40.0
BROW_MIN_MS = 60.0

# Debounce for marker presses (ms)
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
VALID_MARKERS = set(MARKER_TO_LABEL)
REST_MARKERS = {7, 8}


# ------------- Utilities -------------
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


def filter_band(x: np.ndarray, sr: float, band: tuple[float, float], notch_hz: float = NOTCH_HZ) -> np.ndarray:
    x = detrend(x, type="constant")
    if notch_hz and notch_hz > 0:
        x = notch(x, sr, f0=notch_hz)
    b, a = butter_band(sr, band[0], band[1], 4)
    return filtfilt(b, a, x).astype(np.float32)


def moving_rms(x: np.ndarray, win_samps: int) -> np.ndarray:
    w = max(1, int(round(win_samps)))
    c = np.convolve(x ** 2, np.ones(w) / w, mode="same")
    return np.sqrt(np.maximum(c, 0.0) + 1e-12)


def mad_thr(x: np.ndarray, k: float) -> tuple[float, float]:
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


# ------------- EEG baseline-gated onset -------------
def baseline_onset(env: np.ndarray, sr: float, pre_frac: float, k: float, min_ms: float, rollback_frac: float) -> tuple[
    int | None, float]:
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

    # require min contiguous duration around peak
    i0 = p
    while i0 > pre_end and env[i0 - 1] >= thr:
        i0 -= 1
    i1 = p
    while i1 + 1 < L and env[i1 + 1] >= thr:
        i1 += 1
    dur_ms = (i1 - i0 + 1) * 1000.0 / sr
    if dur_ms < min_ms:
        return None, 0.0

    # rollback onset: close to baseline
    thr_on = base_med + rollback_frac * (env[p] - base_med)
    j = p
    while j > 0 and env[j] > thr_on:
        j -= 1

    conf = (env[p] - thr) / (np.std(env) + 1e-6)
    return j, float(conf)


# ------------- EMG robust-Z anomaly (C3) -------------
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
                      band: tuple[float, float], z_thr: float, min_ms_gate: float,
                      prefer_pre: bool = True, rollback_frac: float = 0.25,
                      notch_hz: float = NOTCH_HZ, channel_name: str = "") -> Anom | None:
    lo = np.searchsorted(ts, t0)
    hi = np.searchsorted(ts, t1)
    if hi - lo < 10:
        return None

    xe = filter_band(x[lo:hi], sr, band, notch_hz)
    env = moving_rms(np.abs(xe), ENV_EMG_S * sr)

    # split at the marker inside the window
    seg_ts = ts[lo:hi]
    mid = int(np.searchsorted(seg_ts, t_marker - 1e-12))
    mid = max(1, min(len(env) - 1, mid))

    base = env[:mid]
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med))) + 1e-12
    z = (env - med) / mad

    # strongest anomaly (prefer pre; else post)
    j_pre, z_pre = (int(np.argmax(z[:mid])), float(np.max(z[:mid]))) if mid > 0 else (0, -np.inf)
    j_post, z_post = (int(np.argmax(z[mid:])) + mid, float(np.max(z[mid:]))) if mid < len(z) else (mid, -np.inf)
    if prefer_pre:
        idx, zpk = (j_pre, z_pre) if z_pre >= z_post else (j_post, z_post)
    else:
        idx, zpk = (j_post, z_post) if z_post >= z_pre else (j_pre, z_pre)

    # duration gate using absolute threshold derived from pre std
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

    # rollback onset toward baseline
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


# ------------- Markers -------------
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


# ------------- Core per-file processing -------------
def process_file(raw_csv: Path, sr_hint: int, ts_col: int | None, mk_col: int | None,
                 fp1_i: int, fp2_i: int, emg_i: int,
                 debounce_ms: int,
                 anom_pre_s: float, anom_post_s: float, anom_z: float, anom_min_ms: float,
                 include_eeg_for_emg: bool = False) -> Path:
    df = pd.read_csv(raw_csv, sep=r"\s+|\t|,", engine="python", header=None)

    # columns
    if ts_col is None:
        ts_col = df.columns[-2]
    if mk_col is None:
        mk_col = df.columns[-1]

    ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(float)
    mk = pd.to_numeric(df[mk_col], errors="coerce").fillna(0).to_numpy(int)
    sr = infer_sr(ts, sr_hint)

    c1 = pd.to_numeric(df.iloc[:, fp1_i], errors="coerce").to_numpy(float)
    c2 = pd.to_numeric(df.iloc[:, fp2_i], errors="coerce").to_numpy(float)
    c3 = None if emg_i < 0 else pd.to_numeric(df.iloc[:, emg_i], errors="coerce").to_numpy(float)

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
                t_event=t_marker, offset_s=0.0, confidence=1.0, passed=True,
                z_peak=np.nan, dur_ms=np.nan, min_ms_gate=np.nan,
                details=json.dumps({"source": "marker_only_rest"}),
                sr_hz=sr, channel_used="marker"
            ))
            continue

        # windows
        if mval in (1, 2, 6):  # EEG tasks
            t0 = t_marker - EEG_PRE_S
            t1 = t_marker + EEG_POST_S
        else:  # EMG tasks (tap/fist)
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
            lo = np.searchsorted(ts, t0);
            hi = np.searchsorted(ts, t1)
            cand = []
            if hi - lo >= 10:
                # baseline EEG (c1-c2)
                x1 = filter_band(c1[lo:hi], sr, EEG_BLINK_BAND, NOTCH_HZ)
                x2 = filter_band(c2[lo:hi], sr, EEG_BLINK_BAND, NOTCH_HZ)
                xb = x1 - x2
                env = moving_rms(xb, ENV_BLINK_S * sr)
                onset_idx, conf = baseline_onset(env, sr, pre_frac=0.5, k=MAD_K_BLINK, min_ms=BLINK_MIN_MS,
                                                 rollback_frac=0.2)
                if onset_idx is not None:
                    cand.append((float(ts[lo + onset_idx]), "eeg_baseline", None, None, "c1-c2"))
            # anomaly on c2 (requested)
            a2 = strongest_anomaly(ts, c2, sr, t0, t1, t_marker, EEG_BLINK_BAND, ANOM_Z_THR, BLINK_MIN_MS,
                                   prefer_pre=True, channel_name="c2")
            if a2 is not None and a2.passed:
                cand.append((a2.t_event, a2.method, a2.z_peak, a2.dur_ms, a2.channel))
            # anomaly on c3 as well (requested)
            if c3 is not None:
                a3 = strongest_anomaly(ts, c3, sr, t0, t1, t_marker, EMG_BAND, ANOM_Z_THR, BLINK_MIN_MS,
                                       prefer_pre=True, channel_name="c3")
                if a3 is not None and a3.passed:
                    cand.append((a3.t_event, a3.method, a3.z_peak, a3.dur_ms, a3.channel))
            # choose: prefer passed anomaly; if multiple, take highest z; else baseline
            if cand:
                anoms = [c for c in cand if c[1] == "anomaly"]
                if anoms:
                    det, method, zpk, dur_ms, channel_used = max(anoms, key=lambda t: t[2])
                else:
                    det, method, zpk, dur_ms, channel_used = cand[0]


        elif mval == 6:  # brow (EEG hi)
            lo = np.searchsorted(ts, t0);
            hi = np.searchsorted(ts, t1)
            cand = []
            # baseline detector on c1/c2
            if hi - lo >= 10:
                x1 = np.abs(filter_band(c1[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                x2 = np.abs(filter_band(c2[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                env = moving_rms(np.minimum(x1, x2), ENV_BROW_S * sr)
                onset_idx, conf = baseline_onset(env, sr, pre_frac=0.5, k=MAD_K_BROW, min_ms=BROW_MIN_MS,
                                                 rollback_frac=0.25)
                if onset_idx is not None:
                    cand.append((float(ts[lo + onset_idx]), "eeg_baseline", None, None, "min(c1,c2)"))
            # anomaly on c2 (requested)
            a2 = strongest_anomaly(ts, c2, sr, t0, t1, t_marker, EEG_BROW_BAND, ANOM_Z_THR, BROW_MIN_MS,
                                   prefer_pre=True, channel_name="c2")
            if a2 is not None and a2.passed:
                cand.append((a2.t_event, a2.method, a2.z_peak, a2.dur_ms, a2.channel))
            # choose: prefer passed anomaly; else baseline
            if cand:
                # pick anomaly with max z if present
                anoms = [c for c in cand if c[1] == "anomaly"]
                if anoms:
                    det, method, zpk, dur_ms, channel_used = max(anoms, key=lambda t: t[2])
                else:
                    det, method, zpk, dur_ms, channel_used = cand[0]


        else:  # EMG tasks: tap (3/4) and fist (5) — use C3 robust-Z anomaly
            if c3 is not None:
                min_gate = TAP_MIN_MS if mval in (3, 4) else FIST_MIN_MS if mval == 5 else ANOM_MIN_MS
                a3 = strongest_anomaly(ts, c3, sr, t0, t1, t_marker, EMG_BAND, ANOM_Z_THR, min_gate,
                                       prefer_pre=True, channel_name="c3")
                if a3 is not None:
                    det = a3.t_event
                    zpk = a3.z_peak
                    dur_ms = a3.dur_ms
                    passed = a3.passed
                    min_ms_gate = min_gate
                    channel_used = "c3"

            # (Optional) fall back to EEG if requested and EMG failed
            if det is None and include_eeg_for_emg:
                lo = np.searchsorted(ts, t0);
                hi = np.searchsorted(ts, t1)
                if hi - lo >= 10:
                    x1 = np.abs(filter_band(c1[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                    x2 = np.abs(filter_band(c2[lo:hi], sr, EEG_BROW_BAND, NOTCH_HZ))
                    env = moving_rms(np.maximum(x1, x2), ENV_EMG_S * sr)
                    onset_idx, conf = baseline_onset(env, sr, pre_frac=0.5, k=4.0, min_ms=40.0, rollback_frac=0.25)
                    if onset_idx is not None:
                        det = float(ts[lo + onset_idx])
                        channel_used = "max(c1,c2)"

        # fallback if detector fails (always produce a timestamp)
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
                confidence=float(conf) if (channel_used and channel_used.startswith('c')) else (
                    float(zpk) if (zpk is not None and np.isfinite(zpk)) else 0.0),
                passed=bool(passed),
                z_peak=float(zpk) if (zpk is not None and np.isfinite(zpk)) else np.nan,
                dur_ms=float(dur_ms) if (dur_ms is not None and np.isfinite(dur_ms)) else np.nan,
                min_ms_gate=float(min_ms_gate) if (min_ms_gate is not None and np.isfinite(min_ms_gate)) else (
                    TAP_MIN_MS if mval in (3, 4) else FIST_MIN_MS if mval == 5 else np.nan),
                details=json.dumps({"detector": (
                    "anomaly" if ((zpk is not None and np.isfinite(zpk)) and channel_used in ("c2", "c3")) else "eeg_baseline")}),
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


# ------------- CLI -------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Align markers to physiological onsets (EEG + EMG anomaly; fallback-safe).")
    ap.add_argument("files", nargs="+", help="RAW csv files")
    ap.add_argument("--sr", type=int, default=250, help="Sample rate hint (used if timestamps are odd)")
    ap.add_argument("--ts-col", type=int, default=None, help="Timestamp column index (default: last-2)")
    ap.add_argument("--marker-col", type=int, default=None, help="Marker column index (default: last-1)")
    ap.add_argument("--fp1", type=int, default=FP1_COL_DEFAULT, help="Fp1 (c1) column index")
    ap.add_argument("--fp2", type=int, default=FP2_COL_DEFAULT, help="Fp2 (c2) column index")
    ap.add_argument("--emg", type=int, default=EMG_COL_DEFAULT,
                    help="C3 (emg) column index; set -1 if EMG not recorded")
    ap.add_argument("--debounce-ms", type=int, default=DEFAULT_DEBOUNCE_MS,
                    help="Debounce window for marker presses")

    # Anomaly (EMG) controls
    ap.add_argument("--anom-pre-s", type=float, default=ANOM_PRE_S, help="Pre-marker window (s) for EMG anomaly")
    ap.add_argument("--anom-post-s", type=float, default=ANOM_POST_S, help="Post-marker window (s) for EMG anomaly")
    ap.add_argument("--anom-z", type=float, default=ANOM_Z_THR, help="Robust-Z threshold for EMG anomaly")
    ap.add_argument("--anom-min-ms", type=float, default=ANOM_MIN_MS, help="Default min duration (ms) for anomaly")
    ap.add_argument("--include-eeg-for-emg", action="store_true", help="If EMG fails, try EEG hi-band as fallback")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    for f in args.files:
        try:
            process_file(
                Path(f), args.sr, args.ts_col, args.marker_col,
                args.fp1, args.fp2, args.emg, args.debounce_ms,
                args.anom_pre_s, args.anom_post_s, args.anom_z, args.anom_min_ms,
                include_eeg_for_emg=args.include_eeg_for_emg,
            )
        except Exception as e:
            print(f"[ERR] {f}: {e}")


if __name__ == "__main__":
    main()
