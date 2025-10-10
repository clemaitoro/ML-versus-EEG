#!/usr/bin/env python3
"""
Realtime/CSV inference for the segment-trained CNN.
Prints a class ONLY when detected (prob >= threshold, vote, cooldown).
Robust to older stats files (missing 'window', 'mean', 'std', etc.).

Live (Cyton on COM3):
  python infer_realtime_brainflow.py --mode live --model seg_cnn_events.pt --stats seg_norm_events.json ^
    --board-id 0 --serial-port COM3 --fp1-eeg-index 0 --fp2-eeg-index 1 ^
    --proba-thr 0.90 --vote-k 3 --cooldown-s 0.6 --stride-s 0.10

CSV sanity-check:
  python infer_realtime_brainflow.py --mode csv --csv-file .\\Data\\segment\\BrainFlow-RAW_Recordings_0.csv ^
    --ts-col 23 --fp1-col 1 --fp2-col 2 --emg-col -1 --model seg_cnn_events.pt --stats seg_norm_events.json
"""
import argparse, json, time, sys
import numpy as np
import torch, torch.nn as nn
from collections import deque
from scipy.signal import butter, filtfilt, iirnotch, detrend

# Optional BrainFlow
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    HAVE_BRAINFLOW = True
except Exception:
    HAVE_BRAINFLOW = False

# ---------- Model (must match training) ----------
class Small1DCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding=3, groups=32, bias=False), nn.Conv1d(32, 64, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2, groups=64, bias=False), nn.Conv1d(64, 96, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96, 96, 3, padding=1, bias=False), nn.BatchNorm1d(96), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))
    def forward(self, x): return self.head(self.net(x))

# ---------- DSP ----------
def butter_band(sr, lo, hi, order=4):
    from scipy.signal import butter
    ny=0.5*sr; lo=max(lo,0.1); hi=min(hi,0.45*sr)
    if lo>=hi: hi=min(lo*1.3,0.45*sr-1e-3)
    return butter(order, [lo/ny, hi/ny], btype="bandpass")

def clean_channel_window(x, sr, band, notch_hz=50.0):
    x = detrend(x, type="constant")
    if notch_hz and notch_hz > 0:
        b,a = iirnotch(w0=notch_hz/(sr/2.0), Q=30.0); x = filtfilt(b,a,x)
    b,a = butter_band(sr, band[0], band[1], order=4); x = filtfilt(b,a,x)
    return x.astype(np.float32)

# ---------- Utils ----------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def build_window(fp1, fp2, emg):
    bip = fp1 - fp2
    return np.stack([fp1, fp2, bip, emg], axis=0).astype(np.float32)

def print_detection(label, prob, t_now_s):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] DETECTED {label}  p={prob:.2f}  t={t_now_s:.2f}s", flush=True)

# ---------- Inference core ----------
def _iter_source(sample_source):
    """Accept either a generator OR a function that yields a generator."""
    try:
        # a generator/iterable has __iter__
        iter(sample_source)
        return sample_source
    except TypeError:
        # it's a function -> call it to get a generator
        return sample_source()

def run_inference_loop(sample_source, meta, device, proba_thr, vote_k, cooldown_s, stride_s, print_none,
                       win_pre_cli, win_post_cli):
    # Safe meta with fallbacks
    sr = int(meta.get("sr", 250))
    class_order = meta.get("class_order", ["none","blink","blink_double","tap","tap_double","fist_hold","brow"])

    # window (fallback to 0.20/0.80 if missing; allow CLI override)
    win = meta.get("window", {})
    win_pre = float(win_pre_cli if win_pre_cli is not None else win.get("pre", win.get("pre_s", 0.20)))
    win_post = float(win_post_cli if win_post_cli is not None else win.get("post", win.get("post_s", 0.80)))
    T = int(round((win_pre + win_post) * sr))

    bands = meta.get("bands", {})
    eeg_band = tuple(bands.get("eeg", (0.5, 40.0)))
    emg_band = tuple(bands.get("emg", (15.0, 55.0)))
    notch_hz = float(meta.get("notch_hz", 50.0))

    # normalization
    mean = np.array(meta.get("mean", [0,0,0,0]), dtype=np.float32)[:, None]
    std  = np.array(meta.get("std",  [1,1,1,1]), dtype=np.float32)[:, None]
    std[std==0] = 1.0

    # Model
    model = Small1DCNN(in_ch=4, n_classes=len(class_order)).to(device)
    ckpt = torch.load("seg_cnn_events.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Buffers
    buf_fp1 = deque(maxlen=T); buf_fp2 = deque(maxlen=T); buf_emg = deque(maxlen=T)
    last_det_time = -1e9; t_sample = 0; hop = max(1, int(round(stride_s * sr)))
    vote_labels = deque(maxlen=vote_k)

    # iterate source (generator or callable)
    for fp1_chunk, fp2_chunk, emg_chunk in _iter_source(sample_source):
        for a,b,c in zip(fp1_chunk, fp2_chunk, emg_chunk):
            buf_fp1.append(float(a)); buf_fp2.append(float(b)); buf_emg.append(float(c))
            t_sample += 1
            if len(buf_fp1) < T or (t_sample % hop) != 0:
                continue

            fp1 = np.array(buf_fp1, dtype=np.float32)
            fp2 = np.array(buf_fp2, dtype=np.float32)
            emg = np.array(buf_emg, dtype=np.float32)

            fp1f = clean_channel_window(fp1, sr, eeg_band, notch_hz)
            fp2f = clean_channel_window(fp2, sr, eeg_band, notch_hz)
            emgf = clean_channel_window(emg, sr, emg_band, notch_hz) if np.any(emg) else np.zeros_like(fp1f)

            x = build_window(fp1f, fp2f, emgf)
            x_norm = (x - mean) / std
            xb = torch.from_numpy(x_norm[None, ...]).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                logits = model(xb).detach().cpu().numpy()
            prob = softmax(logits)[0]
            top_idx = int(prob.argmax()); top_label = class_order[top_idx]; top_p = float(prob[top_idx])

            vote_labels.append(top_label)
            majority = max((vote_labels.count(l), l) for l in set(vote_labels))[1]

            now_s = t_sample / sr
            can_fire = (now_s - last_det_time) >= cooldown_s
            is_positive = (majority != "none") and (top_label == majority) and (top_p >= proba_thr)

            if is_positive and can_fire:
                print_detection(majority, top_p, now_s)
                last_det_time = now_s
            else:
                if print_none:
                    ts = time.strftime("%H:%M:%S")
                    none_idx = class_order.index("none") if "none" in class_order else 0
                    print(f"[{ts}] none p={prob[none_idx]:.2f}  top={top_label}:{top_p:.2f}", flush=True)

# ---------- Sources ----------
def make_live_source(board_id, params, sr, fp1_eeg_index, fp2_eeg_index, emg_index):
    if not HAVE_BRAINFLOW:
        sys.exit("BrainFlow not installed. Use `pip install brainflow` or run with --mode csv.")
    BoardShim.enable_dev_board_logger()
    board = BoardShim(board_id, params)
    board.prepare_session(); board.start_stream()
    time.sleep(1.0)  # warmup

    eeg_chs = BoardShim.get_eeg_channels(board_id)
    fp1_id = eeg_chs[fp1_eeg_index]; fp2_id = eeg_chs[fp2_eeg_index]
    emg_id = None
    if emg_index is not None:
        try:
            emg_chs = BoardShim.get_emg_channels(board_id)
            if emg_chs and 0 <= emg_index < len(emg_chs): emg_id = emg_chs[emg_index]
        except Exception:
            emg_id = None

    def gen():
        try:
            while True:
                data = board.get_board_data()
                if data.size == 0: time.sleep(0.02); continue
                fp1 = data[fp1_id, :].astype(np.float32)
                fp2 = data[fp2_id, :].astype(np.float32)
                emg = data[emg_id, :].astype(np.float32) if emg_id is not None else np.zeros_like(fp1, dtype=np.float32)
                yield fp1, fp2, emg
        finally:
            try:
                board.stop_stream(); board.release_session()
            except Exception:
                pass
    return gen  # return a FUNCTION

def make_csv_source(csv_path, ts_col1b, fp1_col1b, fp2_col1b, emg_col1b, sr, stride_s):
    import pandas as pd
    df = pd.read_csv(csv_path, sep=r"\s+|\t|,", engine="python", header=None)
    fp1 = df.iloc[:, fp1_col1b-1].astype(float).to_numpy()
    fp2 = df.iloc[:, fp2_col1b-1].astype(float).to_numpy()
    emg = (df.iloc[:, emg_col1b-1].astype(float).to_numpy()
           if emg_col1b is not None and emg_col1b > 0 and (emg_col1b-1) < df.shape[1]
           else np.zeros_like(fp1, dtype=np.float32))
    hop = max(1, int(round(stride_s * sr)))
    def gen():
        for i in range(0, len(fp1), hop):
            j = min(len(fp1), i+hop)
            yield fp1[i:j], fp2[i:j], emg[i:j]
    return gen  # return a FUNCTION

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live","csv"], default="live")
    ap.add_argument("--model", default="seg_cnn_events.pt")
    ap.add_argument("--stats", default="seg_norm_events.json")
    ap.add_argument("--proba-thr", type=float, default=0.90)
    ap.add_argument("--vote-k", type=int, default=3)
    ap.add_argument("--cooldown-s", type=float, default=0.6)
    ap.add_argument("--stride-s", type=float, default=0.10)
    ap.add_argument("--print-none", action="store_true")
    ap.add_argument("--window-pre", type=float, default=None, help="override window pre seconds")
    ap.add_argument("--window-post", type=float, default=None, help="override window post seconds")

    # Live (BrainFlow)
    ap.add_argument("--board-id", type=int, default=0)
    ap.add_argument("--serial-port", type=str, default=None)
    ap.add_argument("--ip-address", type=str, default=None)
    ap.add_argument("--ip-port", type=int, default=None)
    ap.add_argument("--other-info", type=str, default=None)
    ap.add_argument("--fp1-eeg-index", type=int, default=0)
    ap.add_argument("--fp2-eeg-index", type=int, default=1)
    ap.add_argument("--emg-index", type=int, default=None)

    # CSV
    ap.add_argument("--csv-file", type=str, default=None)
    ap.add_argument("--ts-col", type=int, default=23)
    ap.add_argument("--fp1-col", type=int, default=1)
    ap.add_argument("--fp2-col", type=int, default=2)
    ap.add_argument("--emg-col", type=int, default=-1)
    args = ap.parse_args()

    # Load meta/stats with fallbacks
    with open(args.stats, "r") as f:
        meta = json.load(f)
    if "mean" not in meta: meta["mean"] = [0,0,0,0]
    if "std"  not in meta: meta["std"]  = [1,1,1,1]
    if "sr"   not in meta: meta["sr"]   = 250
    if "class_order" not in meta:
        meta["class_order"] = ["none","blink","blink_double","tap","tap_double","fist_hold","brow"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "live":
        if not HAVE_BRAINFLOW:
            sys.exit("BrainFlow not installed. Use `pip install brainflow` or run in --mode csv.")
        params = BrainFlowInputParams()
        if args.serial_port: params.serial_port = args.serial_port
        if args.ip_address:  params.ip_address  = args.ip_address
        if args.ip_port:     params.ip_port     = args.ip_port
        if args.other_info:  params.other_info  = args.other_info
        src = make_live_source(args.board_id, params, int(meta["sr"]),
                               args.fp1_eeg_index, args.fp2_eeg_index, args.emg_index)
        sample_source = src  # function
    else:
        if not args.csv_file:
            sys.exit("--csv-file is required for --mode csv")
        emg_col1b = args.emg_col if (args.emg_col and args.emg_col > 0) else None
        src = make_csv_source(args.csv_file, args.ts_col, args.fp1_col, args.fp2_col, emg_col1b,
                              sr=int(meta["sr"]), stride_s=args.stride_s)
        sample_source = src  # function

    try:
        run_inference_loop(sample_source=sample_source, meta=meta, device=device,
                           proba_thr=args.proba_thr, vote_k=args.vote_k, cooldown_s=args.cooldown_s,
                           stride_s=args.stride_s, print_none=args.print_none,
                           win_pre_cli=args.window_pre, win_post_cli=args.window_post)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

if __name__ == "__main__":
    main()
