#!/usr/bin/env python3
import argparse, json, math, pathlib as pl
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, detrend, find_peaks
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# =================== Config ===================
EEG_BAND       = (0.5, 40.0)   # features
BLINK_BAND     = (0.5, 15.0)   # detection only
BROW_BAND      = (15.0, 120.0) # detection only (will be clamped to 0.45*sr)
EMG_BAND       = (15.0, 55.0)  # features + detection
NOTCH_HZ       = 50.0
SR_DEFAULT     = 250
WIN_PRE_S      = 0.20
WIN_POST_S     = 0.80
CLASS_ORDER    = ["none","blink","blink_double","tap","tap_double","fist_hold","brow"]
POSITIVE_NAMES = set(CLASS_ORDER) - {"none"}

# defaults for detectors
BLINK_K        = 4.0     # MAD k
BLINK_ENV_WIN  = 0.08    # s
BLINK_MIN_DIST = 0.5     # s between events
BLINK_MIN_MS   = 40.0    # ms duration gate (envelope over threshold)

BROW_K         = 4.5
BROW_ENV_WIN   = 0.08
BROW_MIN_DIST  = 0.5
BROW_MIN_MS    = 60.0

EMG_Z_THR      = 2.0
EMG_MIN_DIST   = 0.5
TAP_MIN_MS     = 40.0
FIST_MIN_MS    = 400.0

NEG_GUARD_S    = 0.40    # negatives must be this far from any positive event time

# =================== DSP helpers ===================
def butter_band(sr, lo, hi, order=4):
    ny = 0.5*sr
    lo = max(lo, 0.1)
    hi = min(hi, 0.45*sr)  # keep headroom for filtfilt
    if lo >= hi:
        hi = min(lo*1.3, 0.45*sr - 1e-6)
    return butter(order, [lo/ny, hi/ny], btype="bandpass")
import numpy as np  # make sure this is present at the top

# --- robust stats helpers (used by blink/tap/fist miners) ---
def mad_thr(x, k):
    """
    Median + k * MAD threshold.
    Returns (threshold, median). Adds tiny epsilon so it's never zero.
    """
    x = np.asarray(x, dtype=np.float64)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med))) + 1e-12
    return med + k * mad, med

def moving_rms(x, win_samps):
    """
    RMS envelope with a box window of length win_samps (in samples).
    """
    w = max(1, int(round(win_samps)))
    x2 = np.square(x, dtype=np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    env = np.convolve(x2, kernel, mode="same")
    return np.sqrt(np.maximum(env, 0.0) + 1e-12).astype(np.float32)

def clean_channel(x, sr, band, notch_hz=50.0):
    x = detrend(x, type="constant")
    if notch_hz and notch_hz > 0:
        b0,a0 = iirnotch(notch_hz/(sr/2.0), Q=30.0); x = filtfilt(b0,a0,x)
    b,a = butter_band(sr, band[0], band[1], order=4)
    # short records safety
    padlen = 3*(max(len(a),len(b))-1)
    if len(x) <= padlen+1:
        return x.astype(np.float32)
    return filtfilt(b,a,x).astype(np.float32)

def moving_rms(x, win_samps):
    w = max(1, int(round(win_samps)))
    c = np.convolve(x**2, np.ones(w)/w, mode="same")
    return np.sqrt(np.maximum(c, 0.0) + 1e-12)

def mad_stats(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return med, mad

def robust_std_from_mad(mad):
    # normal equiv: sigma ≈ 1.4826 * MAD
    return 1.4826 * mad

# =================== IO ===================
def load_segments(seg_path: pl.Path):
    if not seg_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(seg_path)
    need = {"label","t_start","t_end"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["label"] = df["label"].astype(str)
    df = df[df["label"].isin(POSITIVE_NAMES)]
    for c in ("t_start","t_end"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df[np.isfinite(df["t_start"]) & np.isfinite(df["t_end"]) & (df["t_end"]>df["t_start"])]
    return df.reset_index(drop=True)

def resolve_col(arg_val, ncols, fallback_idx):
    if arg_val is None:
        return fallback_idx
    idx = (arg_val-1) if arg_val>0 else (ncols + arg_val)
    if not (0 <= idx < ncols):
        raise SystemExit(f"Column {arg_val} out of bounds ({ncols} cols).")
    return idx

def read_raw(path: pl.Path, ts_col0b: int, fp1_1b: int, fp2_1b: int, emg_1b: int, sr: int, index_time=False):
    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
    ncols = df.shape[1]
    ts_col0 = ts_col0b if ts_col0b is not None else (ncols-2)
    if index_time:
        ts = np.arange(len(df), dtype=float)/float(sr)
    else:
        ts = pd.to_numeric(df.iloc[:, ts_col0], errors="coerce").astype(float).to_numpy()
    fp1 = pd.to_numeric(df.iloc[:, fp1_1b-1], errors="coerce").astype(float).to_numpy()
    fp2 = pd.to_numeric(df.iloc[:, fp2_1b-1], errors="coerce").astype(float).to_numpy()
    if emg_1b>0 and (emg_1b-1) < ncols:
        emg = pd.to_numeric(df.iloc[:, emg_1b-1], errors="coerce").astype(float).to_numpy()
    else:
        emg = np.zeros_like(fp1, dtype=float)
    return df, ts, fp1, fp2, emg

# =================== Event mining ===================
def _find_events_from_env(env, sr, thr_abs, min_ms, min_dist_s):
    if env.size < 3:
        return np.array([], dtype=int)
    distance = max(1, int(round(min_dist_s*sr)))
    peaks, meta = find_peaks(env, height=thr_abs, distance=distance)
    if peaks.size == 0: return peaks
    # duration gate: contiguous region above thr_abs
    keep = []
    L = len(env)
    for p in peaks:
        i0 = p
        while i0 > 0 and env[i0-1] >= thr_abs: i0 -= 1
        i1 = p
        while i1+1 < L and env[i1+1] >= thr_abs: i1 += 1
        dur_ms = (i1 - i0 + 1) * 1000.0 / sr
        if dur_ms >= min_ms:
            keep.append(p)
    return np.array(keep, dtype=int)

def mine_blink_events(ts, fp1_blink, fp2_blink, sr, seg, k=BLINK_K, env_win=BLINK_ENV_WIN,
                      min_dist=BLINK_MIN_DIST, min_ms=BLINK_MIN_MS):
    # bipolar -> envelope -> MAD gate
    t0 = float(seg["t_start"]); t1 = float(seg["t_end"])
    i0 = int(np.searchsorted(ts, t0)); i1 = int(np.searchsorted(ts, t1))
    if i1 - i0 < 8: return []
    bip = (fp1_blink - fp2_blink)[i0:i1]
    env = moving_rms(np.abs(bip), env_win*sr)
    med, mad = mad_stats(env)
    thr_abs = med + k*mad
    peaks = _find_events_from_env(env, sr, thr_abs, min_ms, min_dist)
    return ts[i0 + peaks].tolist()

def mine_brow_events(ts, fp1_hi, fp2_hi, sr, seg, k=BROW_K, env_win=BROW_ENV_WIN,
                     min_dist=BROW_MIN_DIST, min_ms=BROW_MIN_MS):
    t0 = float(seg["t_start"]); t1 = float(seg["t_end"])
    i0 = int(np.searchsorted(ts, t0)); i1 = int(np.searchsorted(ts, t1))
    if i1 - i0 < 8: return []
    x1 = np.abs(fp1_hi[i0:i1]); x2 = np.abs(fp2_hi[i0:i1])
    env = moving_rms(np.minimum(x1,x2), env_win*sr)
    med, mad = mad_stats(env)
    thr_abs = med + k*mad
    peaks = _find_events_from_env(env, sr, thr_abs, min_ms, min_dist)
    return ts[i0 + peaks].tolist()

def _env_gate_regions(env, thr, sr, min_ms):
    """Return (start_idx, end_idx) contiguous regions where env >= thr and duration >= min_ms."""
    above = env >= thr
    regions = []
    i = 0
    L = len(env)
    min_len = int(round((min_ms/1000.0)*sr))
    while i < L:
        if above[i]:
            j = i
            while j+1 < L and above[j+1]:
                j += 1
            if (j - i + 1) >= min_len:
                regions.append((i, j))
            i = j + 1
        else:
            i += 1
    return regions

def mine_tap_events(ts, emgf, sr, seg, k=3.5, env_win=0.05, min_ms=40.0, min_distance_s=0.25):
    """Short EMG bursts inside [t_start,t_end]. Return list of event center times."""
    t0, t1 = float(seg["t_start"]), float(seg["t_end"])
    i0 = int(np.searchsorted(ts, t0)); i1 = int(np.searchsorted(ts, t1))
    if i1 - i0 < 8: return []
    seg_env = moving_rms(np.abs(emgf[i0:i1]), env_win*sr)
    thr, _ = mad_thr(seg_env, k)
    regs = _env_gate_regions(seg_env, thr, sr, min_ms=min_ms)
    # one center per region
    centers = []
    for a,b in regs:
        c = (a+b)//2
        centers.append(ts[i0 + c])
    # enforce min distance between centers
    if len(centers) > 1:
        out = [centers[0]]
        for v in centers[1:]:
            if (v - out[-1]) >= min_distance_s:
                out.append(v)
        centers = out
    return centers

def mine_fist_events(ts, emgf, sr, seg, k=3.0, env_win=0.05, min_ms=400.0, min_gap_s=0.6):
    """Sustained EMG holds inside [t_start,t_end]. Return list of center times (one per hold)."""
    t0, t1 = float(seg["t_start"]), float(seg["t_end"])
    i0 = int(np.searchsorted(ts, t0)); i1 = int(np.searchsorted(ts, t1))
    if i1 - i0 < 8: return []
    seg_env = moving_rms(np.abs(emgf[i0:i1]), env_win*sr)
    thr, _ = mad_thr(seg_env, k)
    regs = _env_gate_regions(seg_env, thr, sr, min_ms=min_ms)
    centers = []
    for a,b in regs:
        c = (a+b)//2
        centers.append(ts[i0 + c])
    # merge very close holds (optional)
    if len(centers) > 1:
        out = [centers[0]]
        for v in centers[1:]:
            if (v - out[-1]) >= min_gap_s:
                out.append(v)
        centers = out
    return centers

def mine_emg_events(ts, emg_env_full, sr, seg, z_thr=EMG_Z_THR, min_ms=TAP_MIN_MS, min_dist=EMG_MIN_DIST):
    t0 = float(seg["t_start"]); t1 = float(seg["t_end"])
    i0 = int(np.searchsorted(ts, t0)); i1 = int(np.searchsorted(ts, t1))
    if i1 - i0 < 8: return []
    env = emg_env_full[i0:i1]
    med, mad = mad_stats(env)
    sigma = robust_std_from_mad(mad)
    thr_abs = med + z_thr * sigma
    peaks = _find_events_from_env(env, sr, thr_abs, min_ms, min_dist)
    return ts[i0 + peaks].tolist()

# =================== Windows ===================
def window_indices(ts, t_center, sr, pre_s, post_s):
    idx = int(np.argmin(np.abs(ts - t_center)))
    lo = idx - int(round(pre_s*sr))
    hi = idx + int(round(post_s*sr))
    return lo, hi

class WindowSet(torch.utils.data.Dataset):
    def __init__(self, X, y, mean=None, std=None):
        self.X=X; self.y=y
        Xcat=np.stack(X,0).astype(np.float32)
        if mean is None or std is None:
            m=Xcat.mean(axis=(0,2)); s=Xcat.std(axis=(0,2))+1e-6
            self.mean=m[:,None].astype(np.float32); self.std=s[:,None].astype(np.float32)
        else:
            self.mean=mean.astype(np.float32); self.std=std.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        x=(self.X[i]-self.mean)/self.std
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

# =================== Model ===================
class Small1DCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=len(CLASS_ORDER)):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_ch,32,7,padding=3,bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,32,7,padding=3,groups=32,bias=False), nn.Conv1d(32,64,1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,64,5,padding=2,bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,64,5,padding=2,groups=64,bias=False), nn.Conv1d(64,96,1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96,96,3,padding=1,bias=False), nn.BatchNorm1d(96), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Flatten(), nn.Linear(96,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,n_classes))
    def forward(self,x): return self.head(self.net(x))

# =================== Train/Eval ===================
def train_epoch(model, loader, device, scaler, opt, criterion):

    model.train(); total=0.0; n=0

    for xb,yb in loader:
        xb,yb=xb.to(device,non_blocking=True), yb.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
            logits=model(xb); loss=criterion(logits,yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total+=loss.item()*xb.size(0); n+=xb.size(0)
    return total/max(1,n)

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval(); total=0.0; n=0; all_y=[]; all_p=[]
    for xb,yb in loader:
        xb,yb=xb.to(device,non_blocking=True), yb.to(device,non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
            logits=model(xb); loss=criterion(logits,yb)
        total+=loss.item()*xb.size(0); n+=xb.size(0)
        all_y.append(yb.cpu().numpy()); all_p.append(logits.argmax(1).cpu().numpy())
    y=np.concatenate(all_y) if all_y else np.array([],np.int64)
    p=np.concatenate(all_p) if all_p else np.array([],np.int64)
    return total/max(1,n), y, p

# =================== Main ===================
def main():
    ap=argparse.ArgumentParser(description="Train 1D CNN with per-segment event mining (CUDA).")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--ts_col", type=int, default=None, help="1-based or negative (-1=last, -2=last-2). Default: last-2")
    ap.add_argument("--fp1", type=int, default=1)
    ap.add_argument("--fp2", type=int, default=2)
    ap.add_argument("--emg", type=int, default=-1, help="1-based; <=0 disables EMG mining and EMG feature becomes zeros")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--weighted_loss", action="store_true")
    ap.add_argument("--index_time", action="store_true", help="use index/sr for timestamps")
    # sampling
    ap.add_argument("--stride_s", type=float, default=0.20, help="step for scanning negatives across file")
    ap.add_argument("--neg_pos_ratio", type=float, default=1.0, help="max negatives per positive")
    # detectors (override if needed)
    ap.add_argument("--blink_k", type=float, default=BLINK_K)
    ap.add_argument("--blink_win", type=float, default=BLINK_ENV_WIN)
    ap.add_argument("--blink_min_dist", type=float, default=BLINK_MIN_DIST)
    ap.add_argument("--blink_min_ms", type=float, default=BLINK_MIN_MS)
    ap.add_argument("--brow_k", type=float, default=BROW_K)
    ap.add_argument("--brow_win", type=float, default=BROW_ENV_WIN)
    ap.add_argument("--brow_min_dist", type=float, default=BROW_MIN_DIST)
    ap.add_argument("--brow_min_ms", type=float, default=BROW_MIN_MS)
    ap.add_argument("--emg_z", type=float, default=EMG_Z_THR)
    ap.add_argument("--tap_min_ms", type=float, default=TAP_MIN_MS)
    ap.add_argument("--fist_min_ms", type=float, default=FIST_MIN_MS)
    ap.add_argument("--emg_min_dist", type=float, default=EMG_MIN_DIST)
    ap.add_argument("--tap_k", type=float, default=3.5)
    ap.add_argument("--tap_win", type=float, default=0.05)
    ap.add_argument("--tap_min_dist", type=float, default=0.25)

    ap.add_argument("--fist_k", type=float, default=3.0)
    ap.add_argument("--fist_win", type=float, default=0.05)
    ap.add_argument("--fist_min_gap", type=float, default=0.6)
    args=ap.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_dir=pl.Path(args.data_dir)
    raws=sorted([p for p in data_dir.glob(args.glob)
                 if p.is_file()
                 and not p.name.endswith((".markers.csv",".markers.aligned.csv",".markers.segments.csv"))
                 and p.name!="segments_manifest.csv"])
    if not raws: raise SystemExit("No RAW CSVs found.")

    X,y,groups=[],[],[]
    per_file={}

    for i,raw in enumerate(raws):
        segp=raw.with_suffix(".markers.segments.csv")
        segs=load_segments(segp)
        if segs.empty:
            print(f"[WARN] No usable segments in {segp.name}; skipping {raw.name}.")
            continue

        # timestamp column hint
        df0=pd.read_csv(raw, sep=r"\s+|\t|,", engine="python", header=None, nrows=1)
        ncols=df0.shape[1]
        if "ts_col_used_1b" in segs.columns and pd.notna(segs["ts_col_used_1b"]).any():
            ts_col0 = int(segs["ts_col_used_1b"].iloc[0]) - 1
        else:
            ts_col0 = resolve_col(args.ts_col, ncols, fallback_idx=ncols-2)

        # read all channels
        df, ts, fp1, fp2, emg = read_raw(raw, ts_col0, args.fp1, args.fp2, args.emg, args.sr, index_time=args.index_time)

        # features filtering (across full file)
        fp1f = clean_channel(fp1, args.sr, EEG_BAND, NOTCH_HZ)
        fp2f = clean_channel(fp2, args.sr, EEG_BAND, NOTCH_HZ)
        emgf = clean_channel(emg, args.sr, EMG_BAND, NOTCH_HZ) if np.any(emg) else np.zeros_like(fp1f, dtype=np.float32)

        # detection filters (across full file)
        fp1_blink = clean_channel(fp1, args.sr, BLINK_BAND, NOTCH_HZ)
        fp2_blink = clean_channel(fp2, args.sr, BLINK_BAND, NOTCH_HZ)
        fp1_hi    = clean_channel(fp1, args.sr, BROW_BAND, NOTCH_HZ)
        fp2_hi    = clean_channel(fp2, args.sr, BROW_BAND, NOTCH_HZ)
        emg_env   = moving_rms(np.abs(emgf), 0.05*args.sr) if np.any(emgf) else np.zeros_like(emgf)

        # 1) mine events per segment
        event_times=[]; event_labels=[]
        need_emg = False
        for _, seg in segs.iterrows():
            lab = str(seg["label"])
            if lab in ("blink","blink_double"):
                ev = mine_blink_events(ts, fp1_blink, fp2_blink, args.sr, seg,
                                       k=args.blink_k, env_win=args.blink_win,
                                       min_dist=args.blink_min_dist, min_ms=args.blink_min_ms)
            elif lab == "brow":
                ev = mine_brow_events(ts, fp1_hi, fp2_hi, args.sr, seg,
                                      k=args.brow_k, env_win=args.brow_win,
                                      min_dist=args.brow_min_dist, min_ms=args.brow_min_ms)
            elif lab in ("tap","tap_double","fist_hold"):
                if not np.any(emgf):
                    need_emg = True; ev = []
                else:
                    min_ms = args.fist_min_ms if lab=="fist_hold" else args.tap_min_ms
                    ev = mine_emg_events(ts, emg_env, args.sr, seg,
                                         z_thr=args.emg_z, min_ms=min_ms, min_dist=args.emg_min_dist)
            else:
                ev = []

            if len(ev):
                event_times.extend(ev); event_labels.extend([lab]*len(ev))

        if need_emg:
            print(f"[WARN] {raw.name}: tap/dtap/fist segments present but EMG disabled -> those classes will have NO positives.")

        # 2) negatives: both inside segments (far from any event) and outside segments
        pos_times = np.array(event_times, dtype=float)
        centers_all = np.arange(float(ts[0]), float(ts[-1]), args.stride_s, dtype=float)

        # mask centers that are too close to any positive
        keep_far = np.ones_like(centers_all, dtype=bool)
        if pos_times.size:
            for te in pos_times:
                keep_far &= np.abs(centers_all - te) > NEG_GUARD_S

        # split into inside segments vs outside
        inside_any = np.zeros_like(centers_all, dtype=bool)
        for _,seg in segs.iterrows():
            inside_any |= ((centers_all >= float(seg["t_start"])) & (centers_all <= float(seg["t_end"])))

        neg_inside  = centers_all[ keep_far &  inside_any]
        neg_outside = centers_all[ keep_far & ~inside_any]
        neg_centers = np.concatenate([neg_inside, neg_outside], axis=0)

        # 3) downsample negatives
        if pos_times.size>0 and args.neg_pos_ratio>=0:
            max_neg = int(round(args.neg_pos_ratio * pos_times.size))
            if neg_centers.size > max_neg:
                rng = np.random.default_rng(42)
                neg_centers = rng.choice(neg_centers, size=max_neg, replace=False)

        # 4) materialize windows
        # positives
        for tc, lab in zip(event_times, event_labels):
            lo,hi = window_indices(ts, float(tc), args.sr, WIN_PRE_S, WIN_POST_S)
            if lo<0 or hi>len(ts): continue
            bip = (fp1f - fp2f).astype(np.float32)
            win = np.stack([fp1f[lo:hi], fp2f[lo:hi], bip[lo:hi], emgf[lo:hi]], axis=0).astype(np.float32)
            X.append(win); y.append(CLASS_ORDER.index(lab)); groups.append(i)
            per_file.setdefault(raw.name, {}).setdefault(lab, 0); per_file[raw.name][lab]=per_file[raw.name].get(lab,0)+1

        # negatives
        for tc in neg_centers.tolist():
            lo,hi = window_indices(ts, float(tc), args.sr, WIN_PRE_S, WIN_POST_S)
            if lo<0 or hi>len(ts): continue
            bip = (fp1f - fp2f).astype(np.float32)
            win = np.stack([fp1f[lo:hi], fp2f[lo:hi], bip[lo:hi], emgf[lo:hi]], axis=0).astype(np.float32)
            X.append(win); y.append(0); groups.append(i)
            per_file.setdefault(raw.name, {}).setdefault("none", 0); per_file[raw.name]["none"]=per_file[raw.name].get("none",0)+1

    if not y:
        raise SystemExit("No windows collected. Check your .markers.segments.csv files and columns.")

    y      = np.array(y, np.int64)
    groups = np.array(groups, np.int64)
    print("[INFO] Per-file counts:", per_file)

    # split
    uniq = np.unique(groups)
    if len(uniq)>=2:
        gss=GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=42)
        tr_idx, va_idx = next(gss.split(X,y,groups))
    else:
        rng=np.random.default_rng(42)
        idx_all=np.arange(len(X)); rng.shuffle(idx_all)
        n_val=max(1, int(round(args.val_frac*len(X))))
        va_idx=idx_all[:n_val]; tr_idx=idx_all[n_val:]
        if tr_idx.size==0: tr_idx=idx_all; va_idx=np.array([],dtype=int)
        print(f"[INFO] Single-file dataset: random split ({len(tr_idx)} train / {len(va_idx)} val)")

    Xtr=[X[i] for i in tr_idx]; ytr=y[tr_idx]
    Xva=[X[i] for i in va_idx]; yva=y[va_idx]

    ds_tr=WindowSet(Xtr, ytr)
    ds_va=WindowSet(Xva, yva, mean=ds_tr.mean, std=ds_tr.std)
    pin=(device.type=='cuda')
    dl_tr=DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    dl_va=DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    model=Small1DCNN(in_ch=4, n_classes=len(CLASS_ORDER)).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.weighted_loss:
        counts=np.bincount(ytr, minlength=len(CLASS_ORDER))
        w=1.0/np.clip(counts,1,None); w=w/w.mean()
        print("[INFO] Train class counts:", dict(zip(CLASS_ORDER, counts.tolist())))
        criterion=nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        criterion=nn.CrossEntropyLoss()

    scaler=torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    best=math.inf; y_true=np.array([],np.int64); y_pred=np.array([],np.int64)
    for ep in range(1, args.epochs+1):
        tr=train_epoch(model, dl_tr, device, scaler, opt, criterion)
        if len(Xva):
            va, y_true, y_pred=eval_epoch(model, dl_va, device, criterion)
            print(f"Epoch {ep:02d}  train {tr:.4f}  val {va:.4f}")
            if va<best:
                best=va
                torch.save({"model_state": model.state_dict(),
                            "meta":{"sr":args.sr,"class_order":CLASS_ORDER,
                                    "window":{"pre_s":WIN_PRE_S,"post_s":WIN_POST_S},
                                    "fp1":args.fp1,"fp2":args.fp2,"emg":args.emg,
                                    "ts_col": (args.ts_col if args.ts_col is not None else 'per-file')}},
                           "seg_cnn_events.pt")
        else:
            print(f"Epoch {ep:02d}  train {tr:.4f}  (no val set)")

    print("\n=== Validation report ===")
    if y_true.size and y_pred.size:
        present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        print(classification_report(y_true, y_pred, labels=present,
                                    target_names=[CLASS_ORDER[i] for i in present],
                                    digits=3, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=present))
    else:
        print("No validation set or no predictions.")

    stats={"mean":ds_tr.mean.squeeze().tolist(),"std":ds_tr.std.squeeze().tolist(),
           "sr":args.sr,"class_order":CLASS_ORDER,
           "fp1":args.fp1,"fp2":args.fp2,"emg":args.emg,
           "ts_col":(args.ts_col if args.ts_col is not None else 'per-file'),
           "bands":{"eeg":EEG_BAND,"blink":BLINK_BAND,"brow":BROW_BAND,"emg":EMG_BAND},
           "notch_hz":NOTCH_HZ}
    with open("seg_norm_events.json","w") as f: json.dump(stats,f,indent=2)
    print("\nSaved model to seg_cnn_events.pt")
    print("Saved normalization stats to seg_norm_events.json")

if __name__=="__main__":
    main()
