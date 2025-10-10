#!/usr/bin/env python3
"""
Train a CUDA 1-D CNN from start/end segments but avoid labeling the dead-time inside segments:
we mine blink events inside each segment using a simple envelope+MAD detector and only label
windows near those events as the segment class; everything else becomes `none`.

Currently event-mining is implemented for: blink, blink_double (same detector; windows inherit the segment label).
Other classes fall back to segment-wide labeling (optional TODO: add EMG/brow mining later).

Example:
python train_from_segments_events_cuda.py ^
  --data_dir .\Data\segment ^
  --sr 250 --ts_col 23 --fp1 1 --fp2 2 --emg -1 ^
  --epochs 30 --batch_size 128 --weighted_loss --require_cuda ^
  --stride_s 0.20 --guard_s 0.20 --neg_pos_ratio 1.0 ^
  --blink_k 4.0 --blink_win 0.08 --blink_min_dist 0.5
"""
import argparse, json, math, pathlib as pl
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, detrend, find_peaks
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Config ----------------
EEG_BAND=(0.5,40.0); BLINK_BAND=(0.5,15.0)  # BLINK_BAND kept for reference, not re-filtering segments
EMG_BAND=(15.0,55.0); NOTCH_HZ=50.0
SR_DEFAULT=250; WINDOW_PRE_S=0.20; WINDOW_POST_S=0.80  # total 1.0 s
CLASS_ORDER=["none","blink","blink_double","tap","tap_double","fist_hold","brow"]
POSITIVE_CLASSES=set(CLASS_ORDER)-{"none"}

# ---------------- DSP helpers ----------------
def butter_band(sr, lo, hi, order=4):
    ny=0.5*sr; lo=max(lo,0.1); hi=min(hi,0.45*sr)
    if lo>=hi: hi=min(lo*1.3,0.45*sr-1e-3)
    return butter(order, [lo/ny, hi/ny], btype="bandpass")

def clean_channel(x, sr, band, notch_hz=50.0):
    x=detrend(x, type="constant")
    if notch_hz and notch_hz>0:
        b,a=iirnotch(w0=notch_hz/(sr/2.0), Q=30.0); x=filtfilt(b,a,x)
    b,a=butter_band(sr, band[0], band[1], order=4); x=filtfilt(b,a,x)
    return x.astype(np.float32)

def moving_rms(x, win_samps):
    w=max(1, int(round(win_samps)))
    c=np.convolve(x**2, np.ones(w)/w, mode="same")
    return np.sqrt(np.maximum(c, 0.0) + 1e-12)

def mad_thr(x, k):
    med=float(np.median(x)); mad=float(np.median(np.abs(x-med)))+1e-12
    return med + k*mad, med

# ---------------- IO ----------------
def read_raw(path: pl.Path, ts_col0b: int):
    df=pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
    ts=pd.to_numeric(df.iloc[:, ts_col0b], errors="coerce").astype(float).to_numpy()
    return df, ts

def load_segments(seg_path: pl.Path):
    if not seg_path.exists():
        print(f"[WARN] No segments file for {seg_path.name}")
        return pd.DataFrame()
    df=pd.read_csv(seg_path)
    need={"label","t_start","t_end"}
    if not need.issubset(df.columns):
        print(f"[WARN] Bad segments file (missing {need - set(df.columns)}): {seg_path.name}")
        return pd.DataFrame()
    df=df.copy()
    df["label"]=df["label"].astype(str)
    df=df[df["label"].isin(POSITIVE_CLASSES)]
    df["t_start"]=pd.to_numeric(df["t_start"], errors="coerce").astype(float)
    df["t_end"]=pd.to_numeric(df["t_end"], errors="coerce").astype(float)
    df=df[np.isfinite(df["t_start"]) & np.isfinite(df["t_end"]) & (df["t_end"]>df["t_start"])]
    if df.empty:
        print(f"[WARN] Segments file has no usable rows: {seg_path.name}")
    else:
        print(f"[INFO] Loaded {len(df)} segment(s) from {seg_path.name} | labels: {sorted(df['label'].unique().tolist())}")
    return df.reset_index(drop=True)

# ---------------- Event mining (blink family) ----------------
def mine_blink_events(ts, fp1f, fp2f, sr, seg, k=4.0, env_win=0.08, min_distance_s=0.5):
    """
    Return list of event times (peak centers) inside [t_start,t_end] using bipolar Fp1-Fp2 envelope + MAD gate.
    IMPORTANT: no extra filtfilt here (segments can be short). We rely on the prior full-record EEG_BAND filtering.
    """
    t0=float(seg["t_start"]); t1=float(seg["t_end"])
    i0=int(np.searchsorted(ts, t0)); i1=int(np.searchsorted(ts, t1))
    if i1 - i0 < 4:  # too short for any robust detection
        return []
    bip=(fp1f-fp2f).astype(np.float32)
    seg_sig=bip[i0:i1]
    env=moving_rms(np.abs(seg_sig), env_win*sr)  # envelope over pre-filtered EEG_BAND signal
    thr,_=mad_thr(env, k)
    distance=max(1, int(round(min_distance_s*sr)))
    peaks,_=find_peaks(env, height=thr, distance=distance)
    return ts[i0+peaks].tolist()

# ---------------- Window utilities ----------------
def window_indices(ts, t_center, sr, pre_s, post_s):
    idx=int(np.argmin(np.abs(ts - t_center)))
    lo=idx - int(round(pre_s*sr))
    hi=idx + int(round(post_s*sr))
    return lo, hi

# ---------------- Dataset ----------------
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

# ---------------- Model ----------------
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

# ---------------- Train / Eval ----------------
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

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--ts_col", type=int, default=None, help="1-based; if omitted, use per-file ts_col from segments")
    ap.add_argument("--fp1", type=int, default=1)
    ap.add_argument("--fp2", type=int, default=2)
    ap.add_argument("--emg", type=int, default=16, help="1-based EMG col; set <=0 to disable and use zeros")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--weighted_loss", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # windowing/negatives
    ap.add_argument("--stride_s", type=float, default=0.20, help="for negative sampling across the file")
    ap.add_argument("--guard_s", type=float, default=0.20, help="exclude edges of segments from positives")
    ap.add_argument("--neg_pos_ratio", type=float, default=1.0, help="max negatives per positive")
    # blink event mining
    ap.add_argument("--blink_k", type=float, default=4.0, help="MAD k for blink envelope gate")
    ap.add_argument("--blink_win", type=float, default=0.08, help="envelope window (s) for blink")
    ap.add_argument("--blink_min_dist", type=float, default=0.5, help="min distance (s) between blink events")
    args=ap.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data dir: {args.data_dir} | sr={args.sr} | ts_col={args.ts_col} | fp1={args.fp1} fp2={args.fp2} emg={args.emg}")

    data_dir=pl.Path(args.data_dir)
    raws=sorted([p for p in data_dir.glob("*.csv")
                 if p.is_file()
                 and not p.name.endswith((".markers.csv",".markers.aligned.csv",".markers.segments.csv"))
                 and p.name!="segments_manifest.csv"])
    print(f"[INFO] Found {len(raws)} RAW csv(s): {[p.name for p in raws]}")
    if not raws: raise SystemExit("No RAW CSVs found.")

    X,y,groups=[],[],[]
    per_file={}

    for i,raw in enumerate(raws):
        segp=raw.with_suffix(".markers.segments.csv")
        segs=load_segments(segp)

        # timestamp column
        if args.ts_col is not None:
            ts_col0 = args.ts_col-1
        else:
            if "ts_col_used_1b" in segs.columns and pd.notna(segs["ts_col_used_1b"]).any():
                ts_col0 = int(segs["ts_col_used_1b"].iloc[0]) - 1
            else:
                ts_col0 = -2  # last-2 fallback

        df, ts = read_raw(raw, ts_col0b=ts_col0)
        if args.verbose:
            print(f"[INFO] {raw.name}: ts range {ts[0]:.3f}..{ts[-1]:.3f} (n={len(ts)})")

        fp1 = pd.to_numeric(df.iloc[:, args.fp1-1], errors="coerce").astype(float).to_numpy()
        fp2 = pd.to_numeric(df.iloc[:, args.fp2-1], errors="coerce").astype(float).to_numpy()

        if args.emg and args.emg>0 and (args.emg-1) < df.shape[1]:
            emg = pd.to_numeric(df.iloc[:, args.emg-1], errors="coerce").astype(float).to_numpy()
        else:
            emg = np.zeros_like(fp1, dtype=float)

        # filters across full file (long enough for filtfilt)
        fp1f=clean_channel(fp1, args.sr, EEG_BAND, NOTCH_HZ)
        fp2f=clean_channel(fp2, args.sr, EEG_BAND, NOTCH_HZ)
        emgf=clean_channel(emg, args.sr, EMG_BAND, NOTCH_HZ) if np.any(emg) else np.zeros_like(fp1f, dtype=np.float32)

        # 1) Mine blink events inside blink segments
        event_times=[]; event_labels=[]
        for _,seg in segs.iterrows():
            lab=str(seg["label"])
            if lab in ("blink","blink_double"):
                ev=mine_blink_events(ts, fp1f, fp2f, args.sr, seg,
                                     k=args.blink_k, env_win=args.blink_win, min_distance_s=args.blink_min_dist)
                if args.verbose:
                    print(f"[INFO]   mined {len(ev)} {lab} event(s) in [{seg['t_start']:.2f},{seg['t_end']:.2f}]")
                event_times.extend(ev); event_labels.extend([lab]*len(ev))
            else:
                # TODO: Add EMG/brow mining. For now, segment-wide positives at stride_s.
                t0=float(seg["t_start"])+args.guard_s; t1=float(seg["t_end"])-args.guard_s
                if t1>t0:
                    centers=np.arange(t0, t1, args.stride_s, dtype=float).tolist()
                    event_times.extend(centers); event_labels.extend([lab]*len(centers))

        event_times=np.array(event_times, dtype=float)
        event_labels=np.array(event_labels, dtype=object)

        # 2) Build negatives: sample across whole file at stride_s and drop those near any positive event
        centers_all=np.arange(float(ts[0]), float(ts[-1]), args.stride_s, dtype=float)
        keep_neg=np.ones_like(centers_all, dtype=bool)
        if event_times.size>0:
            radius=0.4  # exclude windows near any positive center
            for te in event_times:
                keep_neg &= np.abs(centers_all - te) > radius
        # also avoid inside-segment cores (guard)
        for _,seg in segs.iterrows():
            t0=float(seg["t_start"]); t1=float(seg["t_end"])
            inside=(centers_all >= (t0+args.guard_s)) & (centers_all <= (t1-args.guard_s))
            keep_neg &= ~inside
        neg_centers=centers_all[keep_neg]

        # 3) Downsample negatives to ratio
        if event_times.size>0 and args.neg_pos_ratio>=0:
            max_neg=int(round(args.neg_pos_ratio * event_times.size))
            if neg_centers.size > max_neg:
                rng=np.random.default_rng(42)
                neg_centers=rng.choice(neg_centers, size=max_neg, replace=False)

        # 4) Assemble windows
        # Positives
        for tc, lab in zip(event_times, event_labels):
            lo,hi=window_indices(ts, float(tc), args.sr, WINDOW_PRE_S, WINDOW_POST_S)
            if lo<0 or hi>len(ts): continue
            bip=(fp1f-fp2f).astype(np.float32)
            win=np.stack([fp1f[lo:hi], fp2f[lo:hi], bip[lo:hi], emgf[lo:hi]], axis=0).astype(np.float32)
            X.append(win); y.append(CLASS_ORDER.index(lab)); groups.append(i)
            per_file.setdefault(raw.name, {}).setdefault(lab, 0); per_file[raw.name][lab]+=1
        # Negatives
        for tc in neg_centers:
            lo,hi=window_indices(ts, float(tc), args.sr, WINDOW_PRE_S, WINDOW_POST_S)
            if lo<0 or hi>len(ts): continue
            bip=(fp1f-fp2f).astype(np.float32)
            win=np.stack([fp1f[lo:hi], fp2f[lo:hi], bip[lo:hi], emgf[lo:hi]], axis=0).astype(np.float32)
            X.append(win); y.append(0); groups.append(i)
            per_file.setdefault(raw.name, {}).setdefault("none", 0); per_file[raw.name]["none"]+=1

    if not y: raise SystemExit("No windows collected. Check segments and columns.")
    y=np.array(y, np.int64); groups=np.array(groups, np.int64)
    print("[INFO] Per-file counts:", per_file)

    # splits
    uniq=np.unique(groups)
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
                                    "window":{"pre":WINDOW_PRE_S,"post":WINDOW_POST_S},
                                    "fp1":args.fp1,"fp2":args.fp2,"emg":args.emg,
                                    "ts_col": (args.ts_col if args.ts_col is not None else 'per-file')}}, "seg_cnn_events.pt")
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
           "bands":{"eeg":EEG_BAND,"blink":BLINK_BAND,"emg":EMG_BAND},"notch_hz":NOTCH_HZ}
    with open("seg_norm_events.json","w") as f: json.dump(stats,f,indent=2)
    print("\nSaved model to seg_cnn_events.pt")
    print("Saved normalization stats to seg_norm_events.json")

if __name__=="__main__":
    main()
