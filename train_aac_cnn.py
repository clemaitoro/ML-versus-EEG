#!/usr/bin/env python3
import argparse, pathlib as pl, json, math, warnings, random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, filtfilt as sfiltfilt, detrend
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# ----------------- Config defaults -----------------
EEG_BAND = (0.5, 40.0)
EMG_BAND = (15.0, 55.0)
NOTCH_HZ = 50.0
WINDOW_PRE_S  = 0.20
WINDOW_POST_S = 0.80
SR_DEFAULT    = 250
SEED          = 42

# Marker protocol (fixed)
MARKER_TO_LABEL = {
    1: "blink",
    2: "blink_double",
    3: "tap",
    4: "tap_double",
    5: "fist_hold",
    6: "brow",
    7: "start_rest",   # ignored
    8: "end_rest",     # ignored
}
CLASS_ORDER = ["blink","blink_double","tap","tap_double","fist_hold","brow"]

# ----------------- Repro -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------- DSP helpers -----------------
def butter_band(sr, lo, hi, order=4):
    ny = 0.5*sr
    return butter(order, [lo/ny, hi/ny], btype="bandpass")

def apply_notch(x, sr, f0=50.0, Q=30):
    b, a = iirnotch(w0=f0/(sr/2.0), Q=Q)
    return sfiltfilt(b, a, x)

def clean_channel(x, sr, band, notch_hz=50.0):
    x = detrend(x, type="constant")
    if notch_hz > 0:
        x = apply_notch(x, sr, f0=notch_hz)
    b, a = butter_band(sr, band[0], band[1], order=4)
    x = filtfilt(b, a, x)
    return x.astype(np.float32)

# ----------------- Timestamp autodetect -----------------
def autodetect_ts_col(df: pd.DataFrame, sr_hint: int):
    target = 1.0 / max(1, sr_hint)
    best = (None, float('inf'), None)
    for col in df.columns:
        ts = pd.to_numeric(df[col], errors='coerce').astype(float).to_numpy()
        dt = np.diff(ts)
        dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.size < 50:
            continue
        med = float(np.median(dt))
        err = abs(med - target)
        if err < best[1]:
            best = (col, err, med)
    return best[0]

# ----------------- Data loading -----------------
def read_raw(path: pl.Path, fp1_col: int, fp2_col: int, emg_col: int|None, ts_col: int|None, sr_hint: int):
    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
    if ts_col is None:
        ts_col = autodetect_ts_col(df, sr_hint) or df.columns[-2]
    mk_col = df.columns[-1]

    ts  = pd.to_numeric(df[ts_col], errors="coerce").astype(float).to_numpy()
    mk  = pd.to_numeric(df[mk_col], errors="coerce").fillna(0).astype(float).to_numpy()
    fp1 = pd.to_numeric(df.iloc[:, fp1_col], errors="coerce").astype(float).to_numpy()
    fp2 = pd.to_numeric(df.iloc[:, fp2_col], errors="coerce").astype(float).to_numpy()
    emg = (pd.to_numeric(df.iloc[:, emg_col], errors="coerce").astype(float).to_numpy()
           if (emg_col is not None and 0 <= emg_col < df.shape[1]) else None)
    return ts, mk, fp1, fp2, emg


def _parse_details_cell(cell):
    if isinstance(cell, str) and cell:
        try:
            return json.loads(cell)
        except Exception:
            return {}
    return {} if (cell is None or (isinstance(cell, float) and not np.isfinite(cell))) else cell


def read_markers_any(raw_csv: pl.Path, only_anomaly: bool = True, only_passed: bool = True):
    aligned = raw_csv.with_suffix(".markers.aligned.csv")
    simple  = raw_csv.with_suffix(".markers.csv")

    if aligned.exists():
        ev = pd.read_csv(aligned)
        if "details" in ev.columns:
            ev["_det"] = ev["details"].apply(_parse_details_cell)
            ev["detector"] = ev["_det"].apply(lambda d: d.get("detector") if isinstance(d, dict) else None)
        else:
            ev["detector"] = None

        if "label" in ev.columns and ev["label"].notna().any():
            lab = ev["label"].astype(str)
        else:
            mcol = "marker_val" if "marker_val" in ev.columns else ("marker" if "marker" in ev.columns else None)
            if mcol is None:
                raise ValueError(f"{aligned.name} missing marker value")
            lab = ev[mcol].astype(int).map(MARKER_TO_LABEL)
        ev["label"] = lab

        if "t_event" in ev.columns and ev["t_event"].notna().any():
            t = ev["t_event"].fillna(ev.get("t_marker", np.nan)).astype(float)
        elif "t_marker" in ev.columns:
            t = ev["t_marker"].astype(float)
        else:
            raise ValueError(f"{aligned.name} missing t_event/t_marker")
        ev["t_ref"] = t

        keep = ev["label"].isin(CLASS_ORDER)
        if only_passed and "passed" in ev.columns:
            keep &= ev["passed"].astype(bool)
        if only_anomaly:
            keep &= (ev["detector"] == "anomaly")

        out = ev.loc[keep, ["t_ref", "label"]].copy()
        out = out[np.isfinite(out["t_ref"]) & out["t_ref"].notna()]
        return out.reset_index(drop=True)

    if simple.exists():
        ev = pd.read_csv(simple)
        if not {"t_unix","marker"}.issubset(ev.columns):
            raise ValueError(f"Bad markers file: {simple.name}")
        out = pd.DataFrame({
            "t_ref": ev["t_unix"].astype(float),
            "label": ev["marker"].astype(int).map(MARKER_TO_LABEL)
        })
        out = out[out["label"].isin(CLASS_ORDER)]
        out = out[np.isfinite(out["t_ref"]) & out["t_ref"].notna()]
        return out.reset_index(drop=True)

    raise FileNotFoundError(f"Missing markers for {raw_csv.name}: expected {aligned.name} or {simple.name}")


def window_indices(ts, t_evt, sr, pre_s, post_s):
    idx = int(np.argmin(np.abs(ts - t_evt)))
    lo = idx - int(round(pre_s*sr))
    hi = idx + int(round(post_s*sr))
    return lo, hi


def build_examples_from_block(raw_csv: pl.Path, sr: int, fp1_col: int, fp2_col: int, emg_col: int|None, ts_col: int|None,
                              only_anomaly: bool, only_passed: bool):
    ts, _, fp1, fp2, emg = read_raw(raw_csv, fp1_col, fp2_col, emg_col, ts_col, sr)

    # preprocess channels
    fp1f = clean_channel(fp1, sr, EEG_BAND, NOTCH_HZ)
    fp2f = clean_channel(fp2, sr, EEG_BAND, NOTCH_HZ)
    bip  = (fp1f - fp2f).astype(np.float32)
    if emg is not None:
        emgf = clean_channel(emg, sr, EMG_BAND, NOTCH_HZ)
    else:
        emgf = np.zeros_like(bip, dtype=np.float32)

    ev = read_markers_any(raw_csv, only_anomaly=only_anomaly, only_passed=only_passed)

    Xs, ys = [], []
    for _, r in ev.iterrows():
        lab = str(r["label"])
        if lab not in CLASS_ORDER:
            continue
        lo, hi = window_indices(ts, float(r["t_ref"]), sr, WINDOW_PRE_S, WINDOW_POST_S)
        if lo < 0 or hi > len(bip):
            continue
        win = np.stack([fp1f[lo:hi], fp2f[lo:hi], bip[lo:hi], emgf[lo:hi]], axis=0).astype(np.float32)
        Xs.append(win)
        ys.append(CLASS_ORDER.index(lab))
    return Xs, ys

# ----------------- Dataset -----------------
class WindowSet(Dataset):
    def __init__(self, X, y, mean=None, std=None):
        self.X = X
        self.y = y
        if mean is None or std is None:
            Xcat = np.stack(X, axis=0).astype(np.float32)  # [N,C,T]
            m = Xcat.mean(axis=(0, 2)).astype(np.float32)
            s = Xcat.std(axis=(0, 2)).astype(np.float32) + 1e-6
            self.mean = m[:, None]
            self.std  = s[:, None]
        else:
            self.mean = mean.astype(np.float32)
            self.std  = std.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

# ----------------- Model -----------------
class Small1DCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=6):
        super().__init__()
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
        z = self.net(x)
        return self.head(z)

# ----------------- Train / Eval -----------------

def make_weighted_loss(y_indices, n_classes, device):
    counts = np.bincount(np.array(y_indices, dtype=np.int64), minlength=n_classes)
    # inverse freq, normalized
    w = 1.0 / np.clip(counts, 1, None)
    w = w / w.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device)), counts


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
    y = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    p = np.concatenate(all_p) if all_p else np.array([], dtype=np.int64)
    return loss_sum / max(1,n), y, p

# ----------------- Main -----------------

def main():
    set_seed(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with RAW csv + markers (.markers.aligned.csv or .markers.csv)")
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model_out", default="aac_cnn.pt")
    ap.add_argument("--stats_out", default="aac_norm_stats.json")
    # marker filters
    ap.add_argument("--only_anomaly", action="store_true", help="Use only rows whose detector=='anomaly' (aligned file)")
    ap.add_argument("--only_passed", action="store_true", help="Use only rows where 'passed'==True (aligned file)")
    # channels & ts
    ap.add_argument("--fp1", type=int, default=1)
    ap.add_argument("--fp2", type=int, default=2)
    ap.add_argument("--emg", type=int, default=3, help="Set -1 if EMG not recorded")
    ap.add_argument("--ts_col", type=int, default=None, help="Timestamp column index; autodetect if omitted")
    # training options
    ap.add_argument("--weighted_loss", action="store_true", help="Use inverse-frequency class weights")
    args = ap.parse_args()

    data_dir = pl.Path(args.data_dir)

    raw_files = sorted([
        p for p in data_dir.glob("*.csv")
        if not p.name.endswith(".markers.csv") and not p.name.endswith(".markers.aligned.csv")
    ])
    assert raw_files, f"No raw CSVs found in {data_dir}"

    # Build examples & groups by file
    X, y, groups = [], [], []
    per_file_counts = {}
    for i, raw in enumerate(raw_files):
        Xi, yi = build_examples_from_block(raw, sr=args.sr,
                                           fp1_col=args.fp1, fp2_col=args.fp2,
                                           emg_col=(None if args.emg < 0 else args.emg),
                                           ts_col=args.ts_col,
                                           only_anomaly=args.only_anomaly, only_passed=args.only_passed)
        if len(yi) == 0:
            warnings.warn(f"No samples in {raw.name} (after filtering)")
            continue
        for idx in yi:
            lab = CLASS_ORDER[idx]
            per_file_counts.setdefault(raw.name, {}).setdefault(lab, 0)
            per_file_counts[raw.name][lab] += 1
        X.extend(Xi); y.extend(yi); groups.extend([i]*len(yi))
        print(f"[OK] {raw.name}: {len(yi)} windows | per-class {per_file_counts[raw.name]}")

    if len(y) == 0:
        raise RuntimeError("No samples collected. Check your aligned markers and filters.")

    y = np.array(y, dtype=np.int64); groups = np.array(groups, dtype=np.int64)

    # Grouped train/val split (by file)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X, y, groups))
    Xtr = [X[i] for i in tr_idx]; ytr = y[tr_idx]
    Xva = [X[i] for i in va_idx]; yva = y[va_idx]

    # Datasets with TRAIN stats used for VAL
    ds_tr = WindowSet(Xtr, ytr)
    ds_va = WindowSet(Xva, yva, mean=ds_tr.mean, std=ds_tr.std)

    # Save stats
    stats = {
        "mean": ds_tr.mean.squeeze().tolist(),
        "std":  ds_tr.std.squeeze().tolist(),
        "sr": args.sr,
        "channels": ["Fp1","Fp2","Bipolar","EMG"],
        "class_order": CLASS_ORDER,
        "window": {"pre_s": WINDOW_PRE_S, "post_s": WINDOW_POST_S},
        "bands": {"eeg": EEG_BAND, "emg": EMG_BAND}, "notch_hz": NOTCH_HZ,
        "filters": {"only_anomaly": args.only_anomaly, "only_passed": args.only_passed},
        "fp1": args.fp1, "fp2": args.fp2, "emg": args.emg, "ts_col": args.ts_col,
    }
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)

    # DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # Model / Optim / Loss
    print(f"[INFO] Using device: {device}")
    model = Small1DCNN(in_ch=4, n_classes=len(CLASS_ORDER)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.weighted_loss:
        criterion, counts = make_weighted_loss(ytr, n_classes=len(CLASS_ORDER), device=device)
        print("[INFO] Class counts (train):", dict(zip(CLASS_ORDER, counts.tolist())))
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_va = math.inf
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, dl_tr, device, scaler, opt, criterion)
        va_loss, y_true, y_pred = eval_epoch(model, dl_va, device, criterion)
        print(f"Epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best_va:
            best_va = va_loss
            torch.save({"model_state": model.state_dict(), "meta": stats}, args.model_out)

    print("\n=== Validation report ===")
    if y_true.size and y_pred.size:
        labels_all = list(range(len(CLASS_ORDER)))
        print(classification_report(y_true, y_pred,
                                    labels=labels_all,
                                    target_names=CLASS_ORDER,
                                    digits=3, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=labels_all))
    else:
        print("Validation set empty or no predictions.")

    print(f"\nSaved best model → {args.model_out}")
    print(f"Saved normalization stats → {args.stats_out}")

if __name__ == "__main__":
    main()
