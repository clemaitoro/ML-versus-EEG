#!/usr/bin/env python3
"""
Blink + Finger Tap AAC — single-file Python app

What it does
- Streams EEG/EMG from OpenBCI Cyton via BrainFlow
- Detects BLINK (low-freq ocular artefact, bipolar Fp1–Fp2) → NEXT
- Detects FINGER TAP (forearm EMG, bipolar P/N) → SELECT
- Simple two-level UI (Categories → Phrases) using Tkinter
- Text-to-speech with pyttsx3 (offline)
- Optional MOCK mode: 'b' for blink, 't' for tap

Quick start
1) pip install brainflow pyttsx3 numpy
   # (Tkinter ships with most Python builds; on Linux: sudo apt install python3-tk)
2) Wire:
   - Ch1 & Ch2 = frontal electrodes (EEG) → use SRB2 ear reference + BIAS ear ground.
   - Ch3 = forearm EMG bipolar: P(bottom)→electrode#1, N(top)→electrode#2. Disable SRB2 for Ch3 on the board.
3) Run (hardware):
   python aac_blink_tap.py --port COM5 --blink-ch 1 2 --tap-ch 3
   Run (mock):
   python aac_blink_tap.py --mock
"""

import argparse
import sys
import time
import threading
import queue
from collections import deque

import numpy as np

try:
    import comtypes.client
    SAPI_AVAILABLE = True
except Exception:
    SAPI_AVAILABLE = False

# ----------------------- Optional deps ----------------------- #
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, DetrendOperations
    BRAINFLOW_AVAILABLE = True
except Exception:
    BRAINFLOW_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    print("Tkinter is required for the UI.", e)
    sys.exit(1)

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


# ----------------------- Phrases/UI data --------------------- #
PHRASE_TREE = [
    ("Greetings", ["Hello", "Goodbye", "Thank you", "Nice to meet you"]),
    ("Needs", ["Water please", "Help me", "Bathroom please", "I'm hungry"]),
    ("Responses", ["Yes", "No", "I'm okay", "Please wait"]),
    ("Social", ["How are you?", "All good", "I don't understand", "Can you repeat?"]),
]

# ----------------------- Defaults ---------------------------- #
DEFAULTS = dict(
    blink_band=(0.5, 8.0),           # Hz (low freq)
    tap_band=(10.0, 50.0),          # Hz (EMG)
    blink_window_sec=0.15,           # smoothing window
    tap_window_sec=1.0,
    blink_k=2.0,                     # z (MAD) threshold
    tap_k=40.0,
    blink_refractory_sec=0.5,
    tap_refractory_sec=0.35,
    veto_after_tap_ms=1000,           # suppress blink after a tap
    notch_hz=50.0,                   # 0 to disable
)

# ----------------------- Utilities --------------------------- #
def robust_mad_zpush(hist_deque: deque, value: float, maxlen: int = 200):
    """Push value, compute median & MAD zscore."""
    hist_deque.append(value)
    if len(hist_deque) > maxlen:
        hist_deque.popleft()
    arr = np.asarray(hist_deque, dtype=np.float64)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-9
    return (value - med) / mad, med, mad

def butter_bandpass_inplace(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4):
    DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(x, sr, lo, hi, order, FilterTypes.BUTTERWORTH.value, 0)

def butter_bandstop_inplace(x: np.ndarray, sr: int, center: float, bw: float = 2.0):
    DataFilter.perform_bandstop(x, sr, center - bw, center + bw, 2, FilterTypes.BUTTERWORTH.value, 0)

# ----------------------- Signal Worker ----------------------- #
class SignalWorker(threading.Thread):
    """
    Streams data, computes:
      - BlinkSignal = bandpass(0.5–8 Hz) of (ChBlink1 - ChBlink2), RMS window → MAD zscore
      - TapSignal   = bandpass(20–120 Hz) of ChTap bipolar (already P−N on hardware), RMS window → MAD zscore
    Emits events 'blink' and 'tap' with refractory & veto.
    """
    def __init__(self, args, event_q: queue.Queue):
        super().__init__(daemon=True)
        self.args = args
        self.q = event_q
        self.stop_flag = threading.Event()

        # Timing
        self.last_blink_ts = 0.0
        self.last_tap_ts = 0.0
        self.veto_until_ts = 0.0

        # State
        self.mock = bool(args.mock or (not BRAINFLOW_AVAILABLE))
        self.board = None
        self.sr = 250
        self.eeg_channels = []
        self.blink_ch_pair = None  # (ch1, ch2) indices
        self.tap_ch = None         # single channel index (expects P−N wired)
        self.buf_len = None

        # Buffers
        self.buf_b1 = deque(maxlen=1)
        self.buf_b2 = deque(maxlen=1)
        self.buf_tap = deque(maxlen=1)

        # Stats buffers
        self.blink_hist = deque(maxlen=200)
        self.tap_hist = deque(maxlen=200)

    def stop(self):
        self.stop_flag.set()

    # ---------------- MOCK ---------------- #
    def _run_mock(self):
        print("[MOCK] Press 'b' = blink (NEXT), 't' = tap (SELECT), 'q' = quit.")
        while not self.stop_flag.is_set():
            ch = sys.stdin.read(1)
            now = time.time()
            if ch.lower() == 'b':
                if now >= self.veto_until_ts and (now - self.last_blink_ts) > self.args.blink_refractory_sec:
                    self.last_blink_ts = now
                    self.q.put(("blink", now))
            elif ch.lower() == 't':
                if (now - self.last_tap_ts) > self.args.tap_refractory_sec:
                    self.last_tap_ts = now
                    self.veto_until_ts = now + (self.args.veto_after_tap_ms / 1000.0)
                    self.q.put(("tap", now))
            elif ch.lower() == 'q':
                self.q.put(("quit", now))
                break

    # --------------- HARDWARE ------------- #
    def _init_board(self):
        params = BrainFlowInputParams()
        params.serial_port = self.args.port or "COM3"
        params.ip_address = self.args.ip or ""
        params.mac_address = self.args.mac or ""
        params.other_info = self.args.other or ""
        params.ip_port = int(self.args.ip_port or 0)

        board_id = BoardIds.CYTON_BOARD if self.args.board == "cyton" else BoardIds.CYTON_DAISY_BOARD
        self.board_id = board_id
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.config_board('x1060110X')
        self.board.config_board('x2060110X')

        # CH3 (forearm EMG P/N): power ON, gain=2x, normal input, BIAS=remove, SRB2=OFF, SRB1=OFF
        self.board.config_board('x3010000X')
        self.board.start_stream()

        self.sr = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)

        # Channel selection
        if len(self.args.blink_ch) != 2:
            # default to first two EEG channels
            blks = self.eeg_channels[:2]
        else:
            blks = self.args.blink_ch
        self.blink_ch_pair = (blks[0], blks[1])

        if self.args.tap_ch is None:
            # default to last EEG channel as EMG (user wired P/N to it)
            self.tap_ch = self.eeg_channels[-1]
        else:
            self.tap_ch = int(self.args.tap_ch)

        # Buffers: ~5 s
        self.buf_len = int(self.sr * 5)
        self.buf_b1 = deque(maxlen=self.buf_len)
        self.buf_b2 = deque(maxlen=self.buf_len)
        self.buf_tap = deque(maxlen=self.buf_len)

        print(f"[INFO] SR={self.sr} Hz | EEG channels={self.eeg_channels}")
        print(f"[INFO] Blink pair: {self.blink_ch_pair} (bipolar subtract)")
        print(f"[INFO] Tap channel: {self.tap_ch} (forearm P−N)")
        if self.args.notch_hz > 0:
            print(f"[INFO] Notch around {self.args.notch_hz} Hz enabled")
        print("[NOTE] Ensure SRB2/BIAS ear clips for EEG; SRB2 DISABLED for the EMG channel on hardware.")

        # Warm-up drain
        t0 = time.time()
        while time.time() - t0 < 1.0:
            self.board.get_board_data()
            time.sleep(0.02)

    def _cleanup_board(self):
        try:
            if self.board:
                self.board.stop_stream()
                self.board.release_session()
        except Exception:
            pass

    def _run_hardware(self):
        try:
            self._init_board()  # make sure _init_board() sets: self.board_id, self.blink_ch_pair, self.tap_ch
            win_blink = max(1, int(self.sr * self.args.blink_window_sec))
            win_tap = max(1, int(self.sr * self.args.tap_window_sec))

            while not self.stop_flag.is_set():
                data = self.board.get_board_data()
                if data.size == 0:
                    time.sleep(0.01);
                    continue

                # Append raw samples into ring buffers
                b1, b2 = self.blink_ch_pair
                for v in data[b1, :]:
                    self.buf_b1.append(float(v))
                for v in data[b2, :]:
                    self.buf_b2.append(float(v))
                for v in data[self.tap_ch, :]:
                    self.buf_tap.append(float(v))

                # Need enough history
                min_need = max(win_blink * 2, win_tap * 2, self.sr)  # ~1 s
                if len(self.buf_b1) < min_need or len(self.buf_b2) < min_need or len(self.buf_tap) < min_need:
                    time.sleep(0.01);
                    continue

                # ---------- BLINK path: bipolar (Ch1 - Ch2) → 0.5–8 Hz ----------
                x1 = np.asarray(self.buf_b1, dtype=np.float64).copy()
                x2 = np.asarray(self.buf_b2, dtype=np.float64).copy()
                xb = x1 - x2
                if self.args.notch_hz > 0:
                    butter_bandstop_inplace(xb, self.sr, self.args.notch_hz)
                butter_bandpass_inplace(xb, self.sr, self.args.blink_band[0], self.args.blink_band[1])
                blink_rms = float(np.sqrt(np.mean((xb[-win_blink:]) ** 2)))
                blink_z, _, _ = robust_mad_zpush(self.blink_hist, blink_rms)

                # ---------- TAP path: convert to µV → 10–50 Hz → amplitude ----------
                xt = np.asarray(self.buf_tap, dtype=np.float64).copy()
                # Scale to microvolts for this board/channel
                try:
                    DataFilter.convert_to_uV(xt, self.board_id, self.tap_ch)
                except Exception:
                    pass
                if self.args.notch_hz > 0:
                    butter_bandstop_inplace(xt, self.sr, self.args.notch_hz)
                # Expect args.tap_band == (10.0, 50.0)
                butter_bandpass_inplace(xt, self.sr, self.args.tap_band[0], self.args.tap_band[1])

                xw = xt[-win_tap:]  # recent window
                tap_amp_uv = float(np.max(np.abs(xw)))  # peak |amplitude| in µV
                # Require it to stay above threshold for ~80 ms
                min_width_s = 0.08
                above = np.mean(np.abs(xw) > self.args.tap_uv_thresh) * (win_tap / self.sr)

                now = time.time()

                # Detect TAP first, then veto blinks briefly
                if (now - self.last_tap_ts) > self.args.tap_refractory_sec \
                        and tap_amp_uv >= self.args.tap_uv_thresh \
                        and above >= min_width_s:
                    self.last_tap_ts = now
                    self.veto_until_ts = now + (self.args.veto_after_tap_ms / 1000.0)
                    self.q.put(("tap", now))

                # BLINK (only if not vetoed by a recent tap)
                if now >= self.veto_until_ts:
                    if (now - self.last_blink_ts) > self.args.blink_refractory_sec and blink_z >= self.args.blink_k:
                        self.last_blink_ts = now
                        self.q.put(("blink", now))


            time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("[ERROR]", e)
        finally:
            self._cleanup_board()

    def run(self):
        if self.mock:
            self._run_mock()
        else:
            self._run_hardware()

class SapiTTS:
    """Very small wrapper around SAPI.SpVoice for async, purge-before-speak TTS."""
    def __init__(self, voice_name=None, rate=170, volume=1.0):
        self.voice = comtypes.client.CreateObject("SAPI.SpVoice")
        # Optional voice selection
        if voice_name:
            try:
                for v in self.voice.GetVoices():
                    if voice_name.lower() in v.GetDescription().lower():
                        self.voice.Voice = v
                        break
            except Exception:
                pass
        # Map your WPM-ish rate to SAPI’s [-10..10]
        self.voice.Rate = max(-10, min(10, int(round((rate - 170) / 10.0))))
        self.voice.Volume = int(max(0, min(100, (volume * 100.0 if volume <= 1.0 else volume))))

    def say(self, text: str, purge: bool = True):
        # 2 = SVSFPurgeBeforeSpeak, 1 = SVSFlagsAsync
        if purge:
            self.voice.Speak("", 2)
        self.voice.Speak(text, 1)

    def stop(self):
        # Purge any ongoing speech
        self.voice.Speak("", 2)

# ----------------------- UI (Tkinter) -------------------------- #
class AACApp:
    def __init__(self, root, event_q, use_tts=True, voice_name=None, rate=170, volume=1.0):
        self.root = root
        self.q = event_q
        self.level = 0
        self.cat_idx = 0
        self.item_idx = 0
        self.blink_n = 0
        self.tap_n = 0
        self.last_level_change_ts = time.time()
        self._last_say_ts = 0.0
        self._say_guard = 0.20  # 200 ms debounce
        self.current_phrase = None
        self.last_spoken = None

        # Allow mock keys inside the Tk window too (no need to focus the terminal)
        self.root.bind('<KeyPress-b>', lambda e: self.q.put(('blink', time.time())))
        self.root.bind('<KeyPress-t>', lambda e: self.q.put(('tap', time.time())))

        # Self-test: speak once after UI is up so you can confirm TTS pipeline
        self.root.after(1200, lambda: self.set_phrase("Speech path is OK"))

        self.categories = [c for c, _ in PHRASE_TREE]
        self.phrases    = [p for _, p in PHRASE_TREE]

        # ----- TTS (main-thread speak via Tk) -----
        self.use_tts = use_tts and SAPI_AVAILABLE
        self.tts = None
        if self.use_tts:
            try:
                self.tts = SapiTTS(voice_name=voice_name, rate=rate, volume=volume)
                # Warm-up so the audio device opens once
                self.tts.say("ready", purge=True)
            except Exception as e:
                print(f"[TTS] init failed: {e}")
                self.use_tts = False
                self.tts = None

        # ----- UI -----
        self._build_ui()
        self._refresh_ui()
        self.root.after(50, self._poll_events)
        self.root.after(1200, lambda: self.set_phrase("Speech path is OK"))

    def _build_ui(self):
        self.root.title("Blink + Tap AAC")
        self.root.geometry("640x480")
        self.root.configure(padx=16, pady=16)
        self.status = ttk.Label(self.root, text="Level: Categories — BLINK=next • TAP=select", font=("Segoe UI", 11))
        self.status.pack(anchor='w', pady=(0, 6))
        self.metrics = ttk.Label(self.root, text="Events — blink: 0 | tap: 0", font=("Segoe UI", 9))
        self.metrics.pack(anchor='w', pady=(0, 12))
        self.frame = ttk.Frame(self.root); self.frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(self.frame, columns=("col",), show='tree'); self.tree.pack(fill='both', expand=True)
        self.help = ttk.Label(self.root, text="Tips: BLINK = next • TAP = select  |  (Mock: 'b' / 't')", font=("Segoe UI", 9))
        self.help.pack(anchor='center', pady=(12, 0))

    def _refresh_ui(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        if self.level == 0:
            for i, c in enumerate(self.categories):
                tag = 'active' if i == self.cat_idx else ''
                self.tree.insert('', 'end', text=("→ " if tag else "   ") + c, tags=(tag,))
            self.status.config(text="Level: Categories — BLINK: next • TAP: open")
        else:
            current_cat = self.categories[self.cat_idx]; items = self.phrases[self.cat_idx]
            for i, p in enumerate(items):
                tag = 'active' if i == self.item_idx else ''
                self.tree.insert('', 'end', text=("→ " if tag else "   ") + p, tags=(tag,))
            self.status.config(text=f"Level: {current_cat} — BLINK: next • TAP: speak")

    def _poll_events(self):
        try:
            while True:
                kind, ts = self.q.get_nowait()
                if kind == "blink": self._on_blink()
                elif kind == "tap": self._on_tap()
                elif kind == "quit":
                    self.shutdown(); self.root.destroy(); return
        except queue.Empty:
            pass
        self.root.after(50, self._poll_events)

    def set_phrase(self, text: str):
        """Overwrite the current phrase and speak the latest only."""
        self.current_phrase = text
        # Speak on Tk main thread; cancel any ongoing speech
        self.root.after(0, self._speak_latest)

    def _speak_latest(self):
        if not (self.use_tts and self.tts and self.current_phrase):
            return
        if self.current_phrase == self.last_spoken:
            return
        try:
            try:
                self.tts.stop()
            except Exception:
                pass
            print(f"[SPEAK] {self.current_phrase}")
            self.tts.say(self.current_phrase, purge=True)  # async, non-blocking
            self.last_spoken = self.current_phrase
        except Exception as e:
            print(f"[TTS] speak error: {e}")

    def _on_blink(self):
        self.blink_n += 1
        if self.level == 0:
            self.cat_idx = (self.cat_idx + 1) % len(self.categories)
        else:
            items = self.phrases[self.cat_idx]
            self.item_idx = (self.item_idx + 1) % len(items)
        self.metrics.config(text=f"Events — blink: {self.blink_n} | tap: {self.tap_n}")
        self._refresh_ui()

    def _on_tap(self):
        now = time.time()
        self.tap_n += 1
        if self.level == 0:
            # enter phrases; small guard so first tap doesn't instantly speak
            self.level = 1
            self.item_idx = 0
            self.last_level_change_ts = now
        else:
            if now - self.last_level_change_ts < 0.35:
                # ignore immediate tap right after opening phrases
                self.metrics.config(text=f"Events — blink: {self.blink_n} | tap: {self.tap_n} (guard)")
                return
            phrase = self.phrases[self.cat_idx][self.item_idx]
            self.set_phrase(phrase)  # <-- overwrite & speak latest
            # (optional) return to categories after speaking:
            self.level = 0
            self.last_level_change_ts = now
        self.metrics.config(text=f"Events — blink: {self.blink_n} | tap: {self.tap_n}")
        self._refresh_ui()

        # ----- TTS helpers -----

    def say(self, text: str):
        """Debounce + queue speech onto Tk main thread."""
        now = time.time()
        if (now - self._last_say_ts) < self._say_guard:
            return
        self._last_say_ts = now
        print(f"[SPEAK] {text}")
        if self.use_tts and self.tts:
            # Run on Tk's main thread to avoid driver quirks
            self.root.after(0, self._speak_main_thread, text)

    def _speak_main_thread(self, text: str):
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            print(f"[TTS] speak error: {e}")

    def shutdown(self):
        try:
            if self.tts:
                self.tts.stop()
        except Exception:
            pass





# ----------------------- CLI / Main ---------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Blink + Finger Tap AAC")
    p.add_argument('--mock', action='store_true', help='Run without hardware; use keyboard b/t')

    # BrainFlow / Cyton params
    p.add_argument('--board', default='cyton', choices=['cyton', 'cyton-daisy'])
    p.add_argument('--port', default=None, help='Serial port (e.g., COM5, /dev/ttyUSB0)')
    p.add_argument('--ip', default=None)
    p.add_argument('--ip-port', type=int, default=0)
    p.add_argument('--mac', default=None)
    p.add_argument('--other', default=None)

    # Channels
    p.add_argument('--blink-ch', type=int, nargs=2, default=[0, 1],
                   help='Two EEG channel indices for blink (will use bipolar difference)')
    p.add_argument('--tap-ch', type=int, default=None,
                   help='One channel index wired as bipolar EMG (P−N). If not set, uses last EEG channel.')

    # Detection params
    p.add_argument('--blink-band', type=float, nargs=2, default=list(DEFAULTS['blink_band']))
    p.add_argument('--blink-window-sec', type=float, default=DEFAULTS['blink_window_sec'])
    p.add_argument('--tap-window-sec', type=float, default=DEFAULTS['tap_window_sec'])
    p.add_argument('--blink-k', type=float, default=DEFAULTS['blink_k'])
    p.add_argument('--tap-k', type=float, default=DEFAULTS['tap_k'])
    p.add_argument('--blink-refractory-sec', type=float, default=DEFAULTS['blink_refractory_sec'])
    p.add_argument('--tap-refractory-sec', type=float, default=DEFAULTS['tap_refractory_sec'])
    p.add_argument('--veto-after-tap-ms', type=int, default=DEFAULTS['veto_after_tap_ms'])
    p.add_argument('--notch-hz', type=float, default=DEFAULTS['notch_hz'],
                   help='Set 0 to disable mains notch')

    p.add_argument('--tap-uv-thresh', type=float, default=10.0,
                   help='Absolute amplitude (µV) threshold on 10–50 Hz band for tap detection')
    p.add_argument('--tap-band', type=float, nargs=2, default=[10.0, 50.0],
                   help='Band for tap detection (Hz)')

    return p.parse_args()

def main():
    args = parse_args()

    # If BrainFlow missing and not mock, auto-switch to mock to avoid crashing
    if not BRAINFLOW_AVAILABLE and not args.mock:
        print("[WARN] BrainFlow not available; starting in MOCK mode.")
        args.mock = True

    evq = queue.Queue()
    worker = SignalWorker(args, evq)
    worker.start()

    root = tk.Tk()
    app = AACApp(root, evq, use_tts=True)

    def on_close():
        worker.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
