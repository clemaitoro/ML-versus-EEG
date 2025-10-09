#!/usr/bin/env python3
# Live AAC inference (CUDA) — LSL (GUI) or BrainFlow source
# Requirements: torch, numpy, scipy, pandas, pylsl (for LSL mode), brainflow (optional for direct mode)

import argparse, time, json, sys, math, threading, queue, numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, detrend, iirnotch

import torch
import torch.nn as nn

# ---------- Model definition (must match training) ----------
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
            nn.Linear(64,  n_classes)
        )
    def forward(self, x):  # x: [B,C,T]
        return self.head(self.net(x))

# ---------- DSP ----------
def bandpass(sr, lo, hi, order=4):
    ny = 0.5*sr
    b, a = butter(order, [lo/ny, hi/ny], btype="bandpass")
    return b, a

def apply_chain(x, sr, band, notch_hz=50.0):
    x = detrend(x, type="constant")
    if notch_hz and notch_hz > 0:
        b0, a0 = iirnotch(notch_hz/(sr/2.0), Q=30)
        x = filtfilt(b0, a0, x)
    b, a = bandpass(sr, band[0], band[1], order=4)
    x = filtfilt(b, a, x)
    return x.astype(np.float32)

# ---------- Source threads ----------
class LSLReader(threading.Thread):
    """Reads from OpenBCI GUI LSL EEG stream. Expects 8-ch float32 at args.sr."""
    def __init__(self, args, out_q):
        super().__init__(daemon=True); self.args=args; self.q=out_q; self.stop_flag=threading.Event()
    def stop(self): self.stop_flag.set()
    def run(self):
        try:
            from pylsl import StreamInlet, resolve_byprop
        except Exception as e:
            print("[ERR] pylsl not installed. pip install pylsl"); return
        print("[INFO] Resolving LSL EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=10)
        if not streams:
            print("[ERR] No LSL EEG stream found. Start OpenBCI GUI → Networking → LSL."); return
        inlet = StreamInlet(streams[0], max_buflen=60, processing_flags=1)
        sr = self.args.sr
        chunk_hop = max(1, int(sr * self.args.hop_s))
        buf = []
        t_last = time.time()
        while not self.stop_flag.is_set():
            samples, _ = inlet.pull_chunk(timeout=0.05)
            if not samples:
                continue
            buf.extend(samples)
            # push in hops ~hop_s
            now = time.time()
            if len(buf) >= chunk_hop:
                arr = np.asarray(buf[:chunk_hop], dtype=np.float32)  # [H, Ch]
                del buf[:chunk_hop]
                self.q.put(arr.T)  # -> [Ch, H]

class BrainFlowReader(threading.Thread):
    """Direct from Cyton via BrainFlow."""
    def __init__(self, args, out_q):
        super().__init__(daemon=True); self.args=args; self.q=out_q; self.stop_flag=threading.Event()
    def stop(self): self.stop_flag.set()
    def run(self):
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        except Exception as e:
            print("[ERR] brainflow not installed. pip install brainflow"); return
        params = BrainFlowInputParams()
        params.serial_port = self.args.port or "COM3"
        board_id = BoardIds.CYTON_BOARD
        board = BoardShim(board_id, params)
        BoardShim.enable_dev_board_logger()
        board.prepare_session(); board.start_stream()
        sr = BoardShim.get_sampling_rate(board_id)
        hop = max(1, int(sr * self.args.hop_s))
        print(f"[INFO] BrainFlow stream started @ {sr} Hz")
        while not self.stop_flag.is_set():
            data = board.get_board_data()
            if data.size==0:
                time.sleep(0.01); continue
            # data shape: [channels, samples]
            # Send out hop-sized chunks
            num = data.shape[1]
            for i in range(0, num, hop):
                sl = data[:, i:i+hop]
                if sl.shape[1] == 0: continue
                self.q.put(sl.astype(np.float32))
        try:
            board.stop_stream(); board.release_session()
        except Exception: pass

# ---------- Live decoder ----------
class LiveDecoder:
    def __init__(self, model_path, stats_json, device, sr,
                 eeg_band=(0.5,40.0), emg_band=(15.0,55.0), notch_hz=50.0,
                 win_pre=0.20, win_post=0.80, hop_s=0.05,
                 fp1_idx=0, fp2_idx=1, emg_idx=2,
                 class_order=("blink","blink_double","tap","tap_double","fist_hold","brow"),
                 min_conf=0.6, refractory_ms=350, ema=0.6):
        self.device=device; self.sr=sr
        self.eeg_band=eeg_band; self.emg_band=emg_band; self.notch=notch_hz
        self.N = int(round((win_pre+win_post)*sr))
        self.hop = max(1, int(round(hop_s*sr)))
        self.fp1_i, self.fp2_i, self.emg_i = fp1_idx, fp2_idx, emg_idx
        self.classes=list(class_order)
        self.min_conf=min_conf
        self.refractory_ms=refractory_ms
        self.last_fire_ms = -10**9
        self.ema=ema
        self.post_prob=None

        # ring buffers
        self.buf_fp1 = deque(maxlen=self.N)
        self.buf_fp2 = deque(maxlen=self.N)
        self.buf_emg = deque(maxlen=self.N)

        # Load model + stats
        ckpt = torch.load(model_path, map_location=device)
        self.model = Small1DCNN(in_ch=4, n_classes=len(self.classes)).to(device).eval()
        self.model.load_state_dict(ckpt["model_state"])

        with open(stats_json, "r") as f:
            st = json.load(f)
        mean = np.array(st["mean"], dtype=np.float32)[:,None]  # (C,1)
        std  = np.array(st["std"],  dtype=np.float32)[:,None]  # (C,1)
        self.mean = torch.from_numpy(mean).to(device)
        self.std  = torch.from_numpy(std).to(device)

    def _push_chunk(self, chunk):  # chunk: [Ch, H]
        # Some streams may include >8 channels; we only use three
        ch = chunk.shape[0]
        for i in range(chunk.shape[1]):
            s = chunk[:, i]
            # Guard against missing emg index
            emg = s[self.emg_i] if (self.emg_i is not None and self.emg_i < ch) else 0.0
            self.buf_fp1.append(float(s[self.fp1_i]))
            self.buf_fp2.append(float(s[self.fp2_i]))
            self.buf_emg.append(float(emg))

    def _current_window(self):
        if len(self.buf_fp1) < self.N: return None
        fp1 = np.asarray(self.buf_fp1, dtype=np.float32)
        fp2 = np.asarray(self.buf_fp2, dtype=np.float32)
        emg = np.asarray(self.buf_emg, dtype=np.float32)

        fp1f = apply_chain(fp1, self.sr, self.eeg_band, self.notch)
        fp2f = apply_chain(fp2, self.sr, self.eeg_band, self.notch)
        bip  = (fp1f - fp2f).astype(np.float32)
        emgf = apply_chain(emg, self.sr, self.emg_band, self.notch) if (self.emg_i is not None) else np.zeros_like(bip)

        x = np.stack([fp1f, fp2f, bip, emgf], axis=0)  # (C,T)
        return x

    @torch.no_grad()
    def step(self, chunk):
        self._push_chunk(chunk)
        if len(self.buf_fp1) < self.N:
            return None

        x = self._current_window()
        t = torch.from_numpy(x).to(self.device).unsqueeze(0)  # [1,C,T]
        # per-channel normalization
        t = (t - self.mean) / (self.std + 1e-6)
        logits = self.model(t.float())
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]  # (K,)
        # EMA smoothing
        if self.post_prob is None:
            self.post_prob = probs.copy()
        else:
            self.post_prob = self.ema*probs + (1-self.ema)*self.post_prob
        k = int(np.argmax(self.post_prob))
        p = float(self.post_prob[k])

        # refractory eventing
        now_ms = int(time.time()*1000)
        fired = False
        if p >= self.min_conf and (now_ms - self.last_fire_ms) >= self.refractory_ms:
            fired = True
            self.last_fire_ms = now_ms
        return {"top": self.classes[k], "p": p, "probs": self.post_prob.copy(), "fired": fired}

# ---------- Pretty console ----------
def bar(p, width=20):
    n = int(round(p*width))
    return "█"*n + "░"*(width-n)

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    dec = LiveDecoder(
        model_path=args.model, stats_json=args.stats, device=device, sr=args.sr,
        eeg_band=tuple(args.eeg_band), emg_band=tuple(args.emg_band), notch_hz=args.notch,
        win_pre=args.pre_s, win_post=args.post_s, hop_s=args.hop_s,
        fp1_idx=args.fp1-1, fp2_idx=args.fp2-1, emg_idx=(args.emg-1 if args.emg>0 else None),
        class_order=args.class_order.split(","),
        min_conf=args.min_conf, refractory_ms=args.refractory_ms, ema=args.ema
    )

    qin = queue.Queue(maxsize=32)
    if args.source == "lsl":
        src = LSLReader(args, qin)
    else:
        src = BrainFlowReader(args, qin)
    src.start()

    print("\n== Live decode running ==")
    print("Controls: Ctrl+C to quit")
    last_print = 0
    try:
        while True:
            try:
                chunk = qin.get(timeout=0.2)  # [Ch, H]
            except queue.Empty:
                continue

            out = dec.step(chunk)
            if not out:
                continue

            if out["fired"]:
                # Print only the chosen word, nothing else
                print(out["top"], flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            src.stop()
        except Exception:
            pass


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live AAC inference with CUDA")
    ap.add_argument("--source", choices=["lsl","brainflow"], default="lsl")
    ap.add_argument("--model", default="aac_cnn.pt")
    ap.add_argument("--stats", default="aac_norm_stats.json")

    # Sampling / preprocessing (must match training)
    ap.add_argument("--sr", type=int, default=250)
    ap.add_argument("--notch", type=float, default=50.0)
    ap.add_argument("--eeg-band", type=float, nargs=2, default=[0.5, 40.0])
    ap.add_argument("--emg-band", type=float, nargs=2, default=[15.0, 55.0])
    ap.add_argument("--pre-s", type=float, default=0.20)
    ap.add_argument("--post-s", type=float, default=0.80)
    ap.add_argument("--hop-s", type=float, default=0.05)

    # Channel indices (GUI shows 1..8; pass 0 to disable EMG)
    ap.add_argument("--fp1", type=int, default=1)
    ap.add_argument("--fp2", type=int, default=2)
    ap.add_argument("--emg", type=int, default=3)

    # Decode behavior
    ap.add_argument("--class-order", default="blink,blink_double,tap,tap_double,fist_hold,brow")
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--refractory-ms", type=int, default=350)
    ap.add_argument("--ema", type=float, default=0.6)

    # BrainFlow params (only if --source brainflow)
    ap.add_argument("--port", default="COM3")

    args = ap.parse_args()
    run(args)
