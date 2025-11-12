#!/usr/bin/env python3
# infer_segments_realtime.py
# Real-time / CSV inference for the seg_cnn_events model. Prints only when a class (≠ 'none')
# crosses probability & vote thresholds and cooldown has elapsed.

import argparse, time, json, math, sys, threading, queue
from collections import deque, Counter
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend, iirnotch
import torch
import torch.nn as nn

class BrainFlowReader(threading.Thread):
    """Direct from Cyton via BrainFlow. Pushes [Ch, Hop] chunks."""
    def __init__(self, args, out_q):
        super().__init__(daemon=True)
        self.args = args
        self.q = out_q
        self.stop_flag = threading.Event()

    def stop(self):
        self.stop_flag.set()

    def run(self):
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
        except Exception:
            print("[ERR] brainflow not installed. pip install brainflow")
            return

        # Prepare board params from CLI
        params = BrainFlowInputParams()
        params.serial_port = self.args.serial_port or "COM3"
        board_id = int(self.args.board_id)

        BoardShim.enable_dev_board_logger()
        board = BoardShim(board_id, params)
        board.prepare_session()

        # >>> YOUR OpenBCI channel configs go here <<<
        # CH1 (EEG):  x1060110X
        # CH2 (EEG):  x2060110X
        # CH3 (EMG):  x3010000X  (isolated: BIAS OFF, SRB OFF, low gain)
        if not self.args.no_openbci_config:
            try:
                board.config_board('x1060110X')  # CH1 EEG
                board.config_board('x2060110X')  # CH2 EEG
                board.config_board('x3010000X')  # CH3 EMG isolated
                print("[INFO] Applied OpenBCI config: CH1/CH2 EEG in SRB2+BIAS, CH3 EMG isolated (no BIAS/SRB).")
            except Exception as e:
                print(f"[WARN] Could not apply OpenBCI config: {e}")

        # Start the stream
        board.start_stream()  # default buffer
        sr = BoardShim.get_sampling_rate(board_id)
        hop = max(1, int(sr * self.args.hop_s))
        print(f"[INFO] BrainFlow stream started @ {sr} Hz (hop={hop} samples)")

        try:
            while not self.stop_flag.is_set():
                data = board.get_board_data()  # shape: [Ch, N]
                if data.size == 0:
                    time.sleep(0.01)
                    continue
                N = data.shape[1]
                for i in range(0, N, hop):
                    sl = data[:, i:i+hop]
                    if sl.shape[1] == 0:
                        continue
                    self.q.put(sl.astype(np.float32))
        finally:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass

# ---------------- Model (must match training) ----------------
class Small1DCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=7):
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
    def forward(self, x):  # x: [B,C,T]
        return self.head(self.net(x))

# ---------------- DSP ----------------
def _bandpass(sr, lo, hi, order=4):
    ny = 0.5*sr
    lo = max(lo, 0.1)
    hi = min(hi, 0.45*sr)
    if lo >= hi:
        hi = min(lo*1.3, 0.45*sr - 1e-3)
    b, a = butter(order, [lo/ny, hi/ny], btype="bandpass")
    return b, a

def _apply_filter(x, sr, band, notch_hz=50.0, do_filter=True):
    x = detrend(x, type="constant")
    if not do_filter:
        return x.astype(np.float32)
    if notch_hz and notch_hz > 0:
        b0, a0 = iirnotch(notch_hz/(sr/2.0), Q=30.0)
        x = filtfilt(b0, a0, x)
    b, a = _bandpass(sr, band[0], band[1], order=4)
    x = filtfilt(b, a, x)
    return x.astype(np.float32)

# ---------------- Decoder ----------------
class SlidingDecoder:
    def __init__(self, model_path, stats_path, device,
                 proba_thr=0.9, vote_k=3, cooldown_s=0.6, stride_s=0.10,
                 fp1_idx=0, fp2_idx=1, emg_idx=None,
                 no_filter=False, print_probs=False):

        self.device = device
        ckpt = torch.load(model_path, map_location=device)
        meta = ckpt.get("meta", {})

        # Load stats (preferred) else fall back to ckpt meta
        with open(stats_path, "r") as f:
            st = json.load(f)

        class_order = st.get("class_order", meta.get("class_order",
                          ["none","blink","blink_double","tap","tap_double","fist_hold","brow"]))
        self.class_order = [str(c) for c in class_order]
        n_classes = len(self.class_order)

        self.model = Small1DCNN(in_ch=4, n_classes=n_classes).to(device).eval()
        self.model.load_state_dict(ckpt["model_state"])

        self.sr = int(st.get("sr", meta.get("sr", 250)))
        bands = st.get("bands", meta.get("bands", {}))
        self.eeg_band = tuple(bands.get("eeg", (0.5, 40.0)))
        self.emg_band = tuple(bands.get("emg", (15.0, 55.0)))
        self.notch_hz = float(st.get("notch_hz", meta.get("notch_hz", 50.0)))

        win = st.get("window", meta.get("window", {}))
        # Accept either {"pre":..,"post":..} or {"pre_s":..,"post_s":..}
        self.win_pre = float(win.get("pre",  win.get("pre_s", 0.20)))
        self.win_post= float(win.get("post", win.get("post_s", 0.80)))

        self.N = int(round((self.win_pre + self.win_post) * self.sr))
        self.hop = max(1, int(round(stride_s * self.sr)))

        self.fp1_i = int(fp1_idx)
        self.fp2_i = int(fp2_idx)
        self.emg_i = (int(emg_idx) if (emg_idx is not None and emg_idx >= 0) else None)

        self.no_filter = bool(no_filter)
        self.print_probs = bool(print_probs)

        mean = np.array(st["mean"], dtype=np.float32)[:, None]  # (C,1)
        std  = np.array(st["std"],  dtype=np.float32)[:, None]
        self.mean = torch.from_numpy(mean).to(device)
        self.std  = torch.from_numpy(std).to(device)

        self.vbuf = deque(maxlen=max(1, int(vote_k)))
        self.pbuf = deque(maxlen=max(1, int(vote_k)))
        self.proba_thr = float(proba_thr)
        self.cooldown_ms = int(1000 * float(cooldown_s))
        self.last_fire_ms = -10**9

        self.buf_fp1 = deque(maxlen=self.N)
        self.buf_fp2 = deque(maxlen=self.N)
        self.buf_emg = deque(maxlen=self.N)

    def _push_samples(self, arr_ch_t):
        # arr_ch_t: [Ch, T] chunk
        ch, T = arr_ch_t.shape
        emg_on = (self.emg_i is not None) and (0 <= self.emg_i < ch)
        for i in range(T):
            s = arr_ch_t[:, i]
            self.buf_fp1.append(float(s[self.fp1_i]))
            self.buf_fp2.append(float(s[self.fp2_i]))
            self.buf_emg.append(float(s[self.emg_i]) if emg_on else 0.0)

    def _current_window(self):
        if len(self.buf_fp1) < self.N:
            return None
        fp1 = np.asarray(self.buf_fp1, dtype=np.float32)
        fp2 = np.asarray(self.buf_fp2, dtype=np.float32)
        emg = np.asarray(self.buf_emg, dtype=np.float32)

        # Filter on the *window* to approximate training preprocessing
        fp1f = _apply_filter(fp1, self.sr, self.eeg_band, self.notch_hz, do_filter=(not self.no_filter))
        fp2f = _apply_filter(fp2, self.sr, self.eeg_band, self.notch_hz, do_filter=(not self.no_filter))
        bip  = (fp1f - fp2f).astype(np.float32)
        if self.emg_i is None:
            emgf = np.zeros_like(bip, dtype=np.float32)
        else:
            emgf = _apply_filter(emg, self.sr, self.emg_band, self.notch_hz, do_filter=(not self.no_filter))

        x = np.stack([fp1f, fp2f, bip, emgf], axis=0)  # (C,T)
        return x

    @torch.no_grad()
    def step_chunk(self, arr_ch_t):
        self._push_samples(arr_ch_t)
        if len(self.buf_fp1) < self.N:
            return None

        x = self._current_window()
        t = torch.from_numpy(x).to(self.device).unsqueeze(0)  # [1,C,T]
        t = (t - self.mean) / (self.std + 1e-6)
        logits = self.model(t.float())
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        k = int(np.argmax(probs)); p = float(probs[k]); lbl = self.class_order[k]
        self.vbuf.append(lbl); self.pbuf.append(p)

        # Majority vote on last K frames
        if len(self.vbuf) == self.vbuf.maxlen:
            freq = Counter(self.vbuf)
            maj_lbl, maj_cnt = max(freq.items(), key=lambda kv: kv[1])
            # Average probability of the voted label across the vote window
            maj_p = float(np.mean([pp for ll, pp in zip(self.vbuf, self.pbuf) if ll == maj_lbl])) if any(self.vbuf) else 0.0
        else:
            maj_lbl, maj_cnt, maj_p = lbl, 1, p

        now_ms = int(time.time()*1000)
        can_fire = (maj_lbl != "none") and (maj_p >= self.proba_thr) and ((now_ms - self.last_fire_ms) >= self.cooldown_ms)

        if can_fire:
            self.last_fire_ms = now_ms
            if self.print_probs:
                print(f"{maj_lbl}  p={maj_p:.2f}", flush=True)
            else:
                print(maj_lbl, flush=True)
            # Optional: clear buffers to avoid immediate repeats
            # self.vbuf.clear(); self.pbuf.clear()

        return dict(top=lbl, p=p, voted=maj_lbl, voted_p=maj_p, fired=can_fire)

# ---------------- Sources ----------------
class CSVStreamer:
    """Yields [Ch, hop] chunks from a RAW CSV, using 1-based column indices for fp1/fp2/emg and ts."""
    def __init__(self, path, ts_col_1b, fp1_1b, fp2_1b, emg_1b, sr, stride_s=0.10, realtime=False):
        self.df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
        self.sr = int(sr)
        self.hop = max(1, int(round(stride_s * self.sr)))
        self.realtime = bool(realtime)

        n = self.df.shape[1]
        ts_col = (ts_col_1b-1) if ts_col_1b else (n-2)
        fp1_c  = fp1_1b-1
        fp2_c  = fp2_1b-1
        emg_c  = (emg_1b-1) if (emg_1b is not None and emg_1b > 0) else None

        self.fp1 = pd.to_numeric(self.df.iloc[:, fp1_c], errors="coerce").astype(float).to_numpy()
        self.fp2 = pd.to_numeric(self.df.iloc[:, fp2_c], errors="coerce").astype(float).to_numpy()
        self.emg = (pd.to_numeric(self.df.iloc[:, emg_c], errors="coerce").astype(float).to_numpy()
                    if emg_c is not None and emg_c < n else np.zeros_like(self.fp1, dtype=float))
        self.N = min(len(self.fp1), len(self.fp2), len(self.emg))
        # we don't actually need timestamps for streaming, but keep for completeness
        self.ts = pd.to_numeric(self.df.iloc[:, ts_col], errors="coerce").astype(float).to_numpy()

    def __iter__(self):
        i = 0
        hop = self.hop
        while i < self.N:
            sl = slice(i, min(i+hop, self.N))
            chunk = np.vstack([self.fp1[sl], self.fp2[sl], self.emg[sl]])  # [3, hop]
            # Decoder computes bipolar internally; we still pass 3 channels here
            yield chunk
            if self.realtime:
                time.sleep(hop / float(self.sr))
            i += hop

class BrainFlowStreamer(threading.Thread):
    """Pushes [Ch, hop] chunks (Fp1, Fp2, EMGopt) from BrainFlow into a queue."""
    def __init__(self, board_id, serial_port, fp1_eeg_idx, fp2_eeg_idx, emg_exg_idx, hop, qout):
        super().__init__(daemon=True)
        self.board_id = int(board_id)
        self.serial_port = serial_port
        self.fp1_idx = int(fp1_eeg_idx)
        self.fp2_idx = int(fp2_eeg_idx)
        self.emg_idx = (int(emg_exg_idx) if emg_exg_idx is not None and emg_exg_idx >= 0 else None)
        self.hop = int(hop)
        self.q = qout
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
        except Exception as e:
            print(f"[ERR] brainflow import failed: {e}", file=sys.stderr); return

        params = BrainFlowInputParams()
        params.serial_port = self.serial_port or "COM3"
        BoardShim.enable_dev_board_logger()
        board = BoardShim(self.board_id, params)
        try:
            board.prepare_session()
            board.start_stream()
            sr = BoardShim.get_sampling_rate(self.board_id)
            eeg_chs = BoardShim.get_eeg_channels(self.board_id)  # absolute channel ids
            exg_chs = BoardShim.get_exg_channels(self.board_id) if hasattr(BoardShim, 'get_exg_channels') else []

            # map positional EEG indices into the EEG channel list
            def take(sig, pos, ch_list):
                if pos is None: return None
                if not (0 <= pos < len(ch_list)): return None
                return sig[ch_list[pos], :]

            while not self._stop.is_set():
                data = board.get_board_data()  # shape [Ch, N]
                if data.size == 0:
                    time.sleep(0.01); continue
                N = data.shape[1]
                for i in range(0, N, self.hop):
                    sl = slice(i, min(i+self.hop, N))
                    chunk_fp1 = take(data, self.fp1_idx, eeg_chs)
                    chunk_fp2 = take(data, self.fp2_idx, eeg_chs)
                    if chunk_fp1 is None or chunk_fp2 is None: continue
                    if self.emg_idx is not None and self.emg_idx < len(exg_chs):
                        chunk_emg = data[exg_chs[self.emg_idx], sl]
                    else:
                        chunk_emg = np.zeros_like(chunk_fp1[sl])
                    arr = np.vstack([chunk_fp1[sl], chunk_fp2[sl], chunk_emg])  # [3, hop]
                    self.q.put(arr.astype(np.float32))
        finally:
            try:
                board.stop_stream()
            except Exception: pass
            try:
                board.release_session()
            except Exception: pass

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Real-time/CSV inference for seg_cnn_events model")
    ap.add_argument("--mode", choices=["csv","live"], required=True)

    # Model & stats
    ap.add_argument("--model", default="seg_cnn_events.pt")
    ap.add_argument("--stats", default="seg_norm_events.json")

    # Thresholding / logic
    ap.add_argument("--proba-thr", type=float, default=0.90)
    ap.add_argument("--vote-k", type=int, default=3)
    ap.add_argument("--cooldown-s", type=float, default=0.6)
    ap.add_argument("--stride-s", type=float, default=0.10)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--print-probs", action="store_true")

    # CSV params (1-based indices, like training CLI)
    ap.add_argument("--csv-file")
    ap.add_argument("--sr", type=int, default=None, help="If omitted, will use sr from stats")
    ap.add_argument("--ts-col", type=int, default=None, help="1-based; default last-2")
    ap.add_argument("--fp1", type=int, default=1)
    ap.add_argument("--fp2", type=int, default=2)
    ap.add_argument("--emg", type=int, default=-1, help="1-based EMG; <=0 disables")

    # BrainFlow params
    ap.add_argument("--board-id", type=int, default=0)         # e.g. 0 for CYTON_BOARD
    ap.add_argument("--serial-port", default="COM3")
    ap.add_argument("--fp1-eeg-index", type=int, default=0, help="Positional index into EEG channel list")
    ap.add_argument("--fp2-eeg-index", type=int, default=1)
    ap.add_argument("--emg-exg-index", type=int, default=-1, help="Positional index into EXG list (if available); <0 disables")

    ap.add_argument("--no-openbci-config", action="store_true",
                    help="Skip sending OpenBCI channel config commands")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load stats just to grab SR if missing
    with open(args.stats, "r") as f:
        st = json.load(f)
    sr_model = int(st.get("sr", 250))
    sr = int(args.sr or sr_model)

    dec = SlidingDecoder(
        model_path=args.model, stats_path=args.stats, device=device,
        proba_thr=args.proba_thr, vote_k=args.vote_k, cooldown_s=args.cooldown_s, stride_s=args.stride_s,
        fp1_idx=(args.fp1-1), fp2_idx=(args.fp2-1),
        emg_idx=((args.emg-1) if args.emg and args.emg>0 else None),
        no_filter=args.no_filter, print_probs=args.print_probs
    )

    if args.mode == "csv":
        if not args.csv_file:
            sys.exit("Please provide --csv-file for mode=csv")
        streamer = CSVStreamer(args.csv_file, args.ts_col, args.fp1, args.fp2, args.emg, sr, stride_s=args.stride_s, realtime=False)
        for chunk in streamer:
            # chunk: [3, hop] -> decoder builds bipolar and EMG slot internally
            # add a zero bipolar placeholder (computed inside _current_window)
            dec.step_chunk(chunk)
        return

    # live BrainFlow
    hop = max(1, int(round(args.stride_s * sr)))
    q = queue.Queue(maxsize=64)
    t = BrainFlowStreamer(
        board_id=args.board_id, serial_port=args.serial_port,
        fp1_eeg_idx=args.fp1_eeg_index, fp2_eeg_idx=args.fp2_eeg_index,
        emg_exg_idx=(args.emg_exg_index if args.emg_exg_index is not None and args.emg_exg_index >= 0 else None),
        hop=hop, qout=q
    )
    t.start()
    print("[INFO] Live inference started. Press Ctrl+C to stop.")
    try:
        while True:
            try:
                chunk = q.get(timeout=0.2)
            except queue.Empty:
                continue
            dec.step_chunk(chunk)
    except KeyboardInterrupt:
        pass
    finally:
        try: t.stop()
        except Exception: pass

if __name__ == "__main__":
    main()
