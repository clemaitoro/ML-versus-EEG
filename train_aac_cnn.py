import argparse, pathlib as pl, json, math, random, warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, filtfilt as sfiltfilt, detrend
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# ----------------- Config defaults -----------------
EEG_BAND = (0.5, 40.0)
EMG_BAND = (15.0, 55.0)
NOTCH_HZ = 50.0
WINDOW_PRE_S  = 0.20      # 200 ms before marker
WINDOW_POST_S = 0.80      # 800 ms after marker  -> total 1.00 s
SR_DEFAULT    = 250

# Labels mapping (edit if your buttons differ)
MARK_MAP = {
    "blink": {1: "blink", 2: "blink_double"},
    "tap":   {3: "tap",   4: "tap_double"},
    "fist":  {5: "fist_hold"},
    "brow":  {6: "brow"}
}
CLASS_ORDER = ["blink","blink_double","tap","tap_double","fist_hold","brow"]

# Column indices in RAW csv
FP1_COL = 0
FP2_COL = 1
EMG_COL = 2   # set to None if you didn't record EMG

# ----------------- DSP helpers -----------------
def butter_band(sr, lo, hi, order=4):
    ny = 0.5*sr
    return butter(order, [lo/ny, hi/ny], btype="bandpass")

def apply_notch(x, sr, f0=50.0, Q=30):
    # iirnotch returns b,a for second-order notch
    b, a = iirnotch(w0=f0/(sr/2.0), Q=Q)
    return sfiltfilt(b, a, x)

def clean_channel(x, sr, band, notch_hz=50.0):
    x = detrend(x, type="constant")
    if notch_hz > 0:
        x = apply_notch(x, sr, f0=notch_hz)
    b, a = butter_band(sr, band[0], band[1], order=4)
    x = filtfilt(b, a, x)
    return x.astype(np.float32)

def infer_block_type(filename: str):
    name = filename.lower()
    if "blink" in name: return "blink"
    if "tap"   in name: return "tap"
    if "fist"  in name: return "fist"
    if "brow"  in name: return "brow"
    raise ValueError(f"Cannot infer block type from {filename}")

# ----------------- Data loading -----------------
def read_raw(path: pl.Path):
    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
    ts = df[df.columns[-2]].to_numpy(float)
    mk = df[df.columns[-1]].to_numpy(float)
    fp1 = df[FP1_COL].to_numpy(float)
    fp2 = df[FP2_COL].to_numpy(float)
    emg = df[EMG_COL].to_numpy(float) if EMG_COL is not None else None
    return ts, mk, fp1, fp2, emg

def read_markers(mfile: pl.Path):
    ev = pd.read_csv(mfile)
    assert {"t_unix","marker"}.issubset(ev.columns), f"Bad markers: {mfile}"
    return ev

def window_indices(ts, t_evt, sr, pre_s, post_s):
    idx = int(np.argmin(np.abs(ts - t_evt)))
    lo = idx - int(round(pre_s*sr))
    hi = idx + int(round(post_s*sr))
    return lo, hi

def build_examples_from_block(raw_csv: pl.Path, sr: int):
    """Return list of (tensor[C,T], label_idx, group_id) for one RAW file."""
    mfile = raw_csv.with_suffix(".markers.csv")
    assert mfile.exists(), f"Missing markers: {mfile.name}"
    block_type = infer_block_type(raw_csv.name)
    mapping = MARK_MAP[block_type]

    ts, mkcol, fp1, fp2, emg = read_raw(raw_csv)

    # preprocess channels
    fp1f = clean_channel(fp1, sr, EEG_BAND, NOTCH_HZ)
    fp2f = clean_channel(fp2, sr, EEG_BAND, NOTCH_HZ)
    bip  = (fp1f - fp2f).astype(np.float32)
    if emg is not None:
        emgf = clean_channel(emg, sr, EMG_BAND, NOTCH_HZ)
    else:
        emgf = np.zeros_like(bip, dtype=np.float32)

    ev = read_markers(mfile)

    N = int(round((WINDOW_PRE_S + WINDOW_POST_S)*sr))
    Xs, ys = [], []
    for _, r in ev.iterrows():
        mval = int(r["marker"])
        if mval not in mapping:   # ignore other block’s buttons
            continue
        lab = mapping[mval]
        lo, hi = window_indices(ts, float(r["t_unix"]), sr, WINDOW_PRE_S, WINDOW_POST_S)
        if lo < 0 or hi > len(bip):
            continue
        # stack channels: [C,T] = (Fp1, Fp2, Bipolar, EMG)
        win = np.stack([fp1f[lo:hi], fp2f[lo:hi], bip[lo:hi], emgf[lo:hi]], axis=0)
        # robust per-window normalization later (dataset class)
        Xs.append(win.astype(np.float32))
        ys.append(CLASS_ORDER.index(lab))
    return Xs, ys

# ----------------- Dataset -----------------
class WindowSet(Dataset):
    def __init__(self, X, y, per_channel_norm=True):
        self.X = X
        self.y = y
        self.per_channel_norm = per_channel_norm

        # [N, C, T]
        Xcat = np.stack(X, axis=0).astype(np.float32)
        m = Xcat.mean(axis=(0, 2)).astype(np.float32)          # (C,)
        s = Xcat.std(axis=(0, 2)).astype(np.float32) + 1e-6    # (C,)
        self.mean = m[:, None]  # -> (C, 1)
        self.std  = s[:, None]  # -> (C, 1

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)   # (C, T)
        if self.per_channel_norm:
            x = (x - self.mean) / self.std  # broadcast over T
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

# ----------------- Model (tiny, VRAM friendly) -----------------
class Small1DCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=6):
        super().__init__()
        # Depthwise separable conv blocks
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3, groups=32, bias=False),
            nn.Conv1d(32, 64, kernel_size=1), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=64, bias=False),
            nn.Conv1d(64, 96, kernel_size=1), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: [B,C,T]
        z = self.net(x)
        return self.head(z)

# ----------------- Train / Eval -----------------
def train_epoch(model, loader, device, scaler, opt, criterion):
    model.train()
    loss_sum, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * xb.size(0)
        n += xb.size(0)
    return loss_sum / max(1,n)

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    all_y, all_p = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0); n += xb.size(0)
        all_y.append(yb.cpu().numpy()); all_p.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    return loss_sum / max(1,n), y, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with RAW_*.csv + .markers.csv")
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)  # fits 4GB VRAM
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model_out", default="aac_cnn.pt")
    ap.add_argument("--stats_out", default="aac_norm_stats.json")
    args = ap.parse_args()

    data_dir = pl.Path(args.data_dir)
    raw_files = sorted([p for p in data_dir.glob("RAW_*.csv") if not p.name.endswith(".markers.csv")])
    assert raw_files, f"No RAW_*.csv in {data_dir}"

    # Build examples & group ids by file
    X, y, groups, file_names = [], [], [], []
    for i, raw in enumerate(raw_files):
        Xi, yi = build_examples_from_block(raw, sr=args.sr)
        if len(yi) == 0:
            warnings.warn(f"No samples in {raw.name}")
            continue
        X.extend(Xi); y.extend(yi); groups.extend([i]*len(yi)); file_names.append(raw.name)
        print(f"[OK] {raw.name}: {len(yi)} windows")

    y = np.array(y, dtype=np.int64); groups = np.array(groups, dtype=np.int64)
    assert len(y) > 0, "No samples collected."

    # Grouped train/val split (by file)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, va_idx = next(gss.split(X, y, groups))
    Xtr = [X[i] for i in tr_idx]; ytr = y[tr_idx]
    Xva = [X[i] for i in va_idx]; yva = y[va_idx]

    # Datasets with dataset-level normalization
    ds_tr = WindowSet(Xtr, ytr, per_channel_norm=True)
    ds_va = WindowSet(Xva, yva, per_channel_norm=True)

    # Save stats from TRAIN set (used at inference)
    stats = {
        "mean": ds_tr.mean.squeeze().tolist(),  # length 4
        "std":  ds_tr.std.squeeze().tolist(),   # length 4
        "sr": args.sr,
        "channels": ["Fp1","Fp2","Bipolar","EMG"],
        "class_order": CLASS_ORDER,
        "window": {"pre_s": WINDOW_PRE_S, "post_s": WINDOW_POST_S},
        "bands": {"eeg": EEG_BAND, "emg": EMG_BAND}, "notch_hz": NOTCH_HZ
    }
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)

    # DataLoaders
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model / CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = Small1DCNN(in_ch=4, n_classes=len(CLASS_ORDER)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_va = math.inf
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, dl_tr, device, scaler, opt, criterion)
        va_loss, y_true, y_pred = eval_epoch(model, dl_va, device, criterion)
        labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])))
        names_present = [CLASS_ORDER[i] for i in labels_present]
        print(f"Epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best_va:
            best_va = va_loss
            torch.save({"model_state": model.state_dict(), "meta": stats}, args.model_out)

    print("\n=== Validation report ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_ORDER, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    print(f"\nSaved best model → {args.model_out}")
    print(f"Saved normalization stats → {args.stats_out}")

if __name__ == "__main__":
    main()
