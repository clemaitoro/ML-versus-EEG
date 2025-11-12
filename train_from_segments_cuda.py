#!/usr/bin/env python3
import argparse, json, math, warnings, pathlib as pl
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, detrend
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

EEG_BAND=(0.5,40.0); EMG_BAND=(15.0,55.0); NOTCH_HZ=50.0
SR_DEFAULT=250; WINDOW_PRE_S=0.20; WINDOW_POST_S=0.80
CLASS_ORDER=["blink","blink_double","tap","tap_double","fist_hold","brow"]

def butter_band(sr, lo, hi, order=4):
    ny=0.5*sr; return butter(order, [lo/ny, hi/ny], btype="bandpass")

def clean_channel(x, sr, band, notch_hz=50.0):
    x=detrend(x, type="constant")
    if notch_hz and notch_hz>0:
        b,a=iirnotch(w0=notch_hz/(sr/2.0), Q=30.0); x=filtfilt(b,a,x)
    b,a=butter_band(sr, band[0], band[1], order=4); x=filtfilt(b,a,x)
    return x.astype(np.float32)

def read_raw(path: pl.Path, ts_col0b: int):
    df=pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
    ts=pd.to_numeric(df.iloc[:, ts_col0b], errors="coerce").astype(float).to_numpy()
    return df, ts

def window_centers_from_segments(segs, stride_s=0.20, guard_s=0.20):
    centers=[]; labels=[]
    for _, r in segs.iterrows():
        lab=str(r["label"])
        if lab not in CLASS_ORDER: continue
        a=float(r["t_start"])+guard_s; b=float(r["t_end"])-guard_s
        if (b-a) < (WINDOW_PRE_S+WINDOW_POST_S): continue
        t0=a+WINDOW_PRE_S; t1=b-WINDOW_POST_S
        n=max(0, int(np.floor((t1-t0)/stride_s))+1)
        for k in range(n): centers.append(t0+k*stride_s); labels.append(lab)
    return np.array(centers, float), np.array(labels, object)

class WindowSet(Dataset):
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

class Small1DCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=6):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_ch,32,7,padding=3,bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,32,7,padding=3,groups=32,bias=False), nn.Conv1d(32,64,1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,64,5,padding=2,bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,64,5,padding=2,groups=64,bias=False), nn.Conv1d(64,96,1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96,96,3,padding=1,bias=False), nn.BatchNorm1d(96), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Flatten(), nn.Linear(96,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,n_classes))
    def forward(self,x): return self.head(self.net(x))

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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--ts_col", type=int, default=None, help="1-based; if omitted, use per-file ts_col from segments")
    ap.add_argument("--fp1", type=int, default=1)
    ap.add_argument("--fp2", type=int, default=2)
    ap.add_argument("--emg", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride_s", type=float, default=0.20)
    ap.add_argument("--guard_s", type=float, default=0.20)
    ap.add_argument("--weighted_loss", action="store_true")
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.25, help="Validation fraction when needed")
    ap.add_argument("--out", default="seg_cnn.pt")
    ap.add_argument("--stats_out", default="seg_norm.json")
    args=ap.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_dir=pl.Path(args.data_dir)
    raws=sorted([p for p in data_dir.glob("*.csv")
                 if p.is_file()
                 and not p.name.endswith((".markers.csv",".markers.aligned.csv",".markers.segments.csv"))
                 and p.name != "segments_manifest.csv"])
    if not raws: raise SystemExit("No RAW CSVs found.")

    X,y,groups=[],[],[]
    per_file={}
    for i,raw in enumerate(raws):
        segp=raw.with_suffix(".markers.segments.csv")
        if not segp.exists(): warnings.warn(f"No segments for {raw.name}, skipping."); continue
        segs=pd.read_csv(segp)
        if segs.empty: warnings.warn(f"Segments empty for {raw.name}, skipping."); continue

        # pick timestamp column: prefer per-file meta from segments unless overridden
        if args.ts_col is not None:
            ts_col0 = args.ts_col-1
        else:
            if "ts_col_used_1b" in segs.columns and pd.notna(segs["ts_col_used_1b"]).any():
                ts_col0 = int(segs["ts_col_used_1b"].iloc[0]) - 1
            else:
                ts_col0 = -2  # last-2 as a fallback

        df, ts = read_raw(raw, ts_col0b=ts_col0)
        fp1 = pd.to_numeric(df.iloc[:, args.fp1-1], errors="coerce").astype(float).to_numpy()
        fp2 = pd.to_numeric(df.iloc[:, args.fp2-1], errors="coerce").astype(float).to_numpy()
        emg = pd.to_numeric(df.iloc[:, args.emg-1], errors="coerce").astype(float).to_numpy()

        fp1f=clean_channel(fp1, args.sr, EEG_BAND, NOTCH_HZ)
        fp2f=clean_channel(fp2, args.sr, EEG_BAND, NOTCH_HZ)
        bip=(fp1f-fp2f).astype(np.float32)
        emgf=clean_channel(emg, args.sr, EMG_BAND, NOTCH_HZ)

        centers, labels = window_centers_from_segments(segs, stride_s=args.stride_s, guard_s=args.guard_s)
        for tc,lab in zip(centers, labels):
            idx=int(np.argmin(np.abs(ts - tc)))
            lo=idx - int(round(WINDOW_PRE_S*args.sr))
            hi=idx + int(round(WINDOW_POST_S*args.sr))
            if lo<0 or hi>len(ts): continue
            win=np.stack([fp1f[lo:hi], fp2f[lo:hi], bip[lo:hi], emgf[lo:hi]], axis=0).astype(np.float32)
            X.append(win); y.append(CLASS_ORDER.index(lab)); groups.append(i)
            per_file.setdefault(raw.name, {}).setdefault(lab, 0); per_file[raw.name][lab]+=1

    if not y: raise SystemExit("No windows collected. Check segments and columns.")
    y=np.array(y, np.int64); groups=np.array(groups, np.int64)
    print("[INFO] Per-file counts:", per_file)

    # --- Robust split ---
    uniq_groups = np.unique(groups)
    if len(uniq_groups) >= 2:
        gss=GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=42)
        tr_idx, va_idx = next(gss.split(X,y,groups))
    else:
        # single-file fallback: random window-level split
        rng = np.random.default_rng(42)
        idx_all = np.arange(len(X))
        rng.shuffle(idx_all)
        n_val = max(1, int(round(args.val_frac * len(X))))
        va_idx = idx_all[:n_val]
        tr_idx = idx_all[n_val:]
        if tr_idx.size == 0:  # extreme tiny set: train on all, no val
            tr_idx = idx_all
            va_idx = np.array([], dtype=int)
        print(f"[INFO] Single-file dataset: using random split ({len(tr_idx)} train / {len(va_idx)} val)")

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
        va, y_true, y_pred=eval_epoch(model, dl_va, device, criterion) if len(Xva) else (float('nan'), np.array([],np.int64), np.array([],np.int64))
        if len(Xva):
            print(f"Epoch {ep:02d}  train {tr:.4f}  val {va:.4f}")
            if va<best: best=va; torch.save({"model_state": model.state_dict(),
                                             "meta":{"sr":args.sr,"class_order":CLASS_ORDER,
                                                     "window":{"pre":WINDOW_PRE_S,"post":WINDOW_POST_S},
                                                     "fp1":args.fp1,"fp2":args.fp2,"emg":args.emg,
                                                     "ts_col": (args.ts_col if args.ts_col is not None else 'per-file')}}, args.out)
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
           "bands":{"eeg":EEG_BAND,"emg":EMG_BAND},"notch_hz":NOTCH_HZ}
    with open(args.stats_out,"w") as f: json.dump(stats,f,indent=2)
    print(f"\nSaved model to {args.out}")
    print(f"Saved normalization stats to {args.stats_out}")

if __name__=="__main__": main()
