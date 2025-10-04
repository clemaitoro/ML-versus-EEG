#!/usr/bin/env python3
"""
Blink + Tap AAC — merged detectors with crosstalk fixes

What it does
- BLINK: MAD-z on RMS of (Fp1−Fp2) band 0.5–8 Hz → Next
- TAP: EMG pulse on forearm (10–50 Hz). New **one-shot FSM** (min/max width) so holds don’t spam taps → Select
- FIST (hold): EMG RMS ≥ threshold for ≥ duration → Speak (suppresses taps while active)
- EYEBROW raise: Fp1 & Fp2 RMS (20–150 Hz) ≥ thresholds → Favorites

Key fixes
- **Tap vs Fist:** tap is detected only when the envelope crosses the threshold and returns below within a width window. While fist is active, tap FSM is suppressed & reset.
- **Brow vs Blink:** brow activity sets a **blink veto window** so eyebrow raises don’t generate multiple blinks.
- **NEW (Fist pre/during/post veto for Tap):** During fist wind-up, accumulation, and just after firing, taps are vetoed and tap state is reset to prevent stray pulses.

Mock keys inside the Tk window: b=blink, t=tap, h=fist, e=brow, q=quit.
"""

import argparse, sys, time, threading, queue
from collections import deque
import numpy as np

# ---------------- Optional deps ---------------- #
try:
    import comtypes.client
    SAPI_AVAILABLE = True
except Exception:
    SAPI_AVAILABLE = False

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
    BRAINFLOW_AVAILABLE = True
except Exception:
    BRAINFLOW_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    print("Tkinter is required for the UI.", e); sys.exit(1)

# ---------------- UI data ---------------- #
PHRASE_FAVORITES = ["Yes", "No", "Help me", "Thank you"]
PHRASE_TREE = [
    ("Greetings", ["Hello", "Goodbye", "Thank you", "Nice to meet you"]),
    ("Needs", ["Water please", "Help me", "Bathroom please", "I'm hungry"]),
    ("Responses", ["Yes", "No", "I'm okay", "Please wait"]),
    ("Social", ["How are you?", "All good", "I don't understand", "Can you repeat?"]),
]

# ---------------- Defaults ---------------- #
DEFAULTS = dict(
    double_min_gap_ms = 120,
    double_max_gap_ms = 350,
    # Optional post-event quiet time (after we finalize single or double)
    post_event_refractory_ms = 300,
    # Blink / Tap (from your original working build)
    blink_band=(0.5, 8.0),
    tap_band=(10.0, 50.0),
    blink_window_sec=0.25,
    tap_window_sec=0.15,
    tap_k=40.0,                 # shorter window helps pulse width measurement
    blink_k=0.0,                # MAD-z threshold
    tap_uv_thresh=15.0,         # µV threshold for tap pulses (raise from 10 to avoid noise)
    blink_refractory_sec=0.8,   # longer to avoid brow-induced repeats
    tap_refractory_sec=0.55,    # slightly longer between taps
    veto_after_tap_ms=100,
    notch_hz=50.0,

    # Tap pulse width gates (milliseconds)
    tap_min_on_ms=50,
    tap_max_on_ms=300,

    # Fist hold (unchanged)
    fist_min_uv=70.0,
    fist_hold_min_ms=400,
    fist_refractory_ms=900,

    # Eyebrow raise
    brow_band=(20.0, 150.0),
    brow_window_sec=0.20,
    brow_fp1_uv=85.0,             # bump a bit to avoid blink bleed
    brow_fp2_uv=100.0,
    brow_min_ms=160,
    brow_refractory_ms=700,
    brow_veto_ms=400,

    # Fist pre-veto (EMG wind-up)
    fist_pre_uv=60.0,
    fist_pre_veto_ms = 250,

    # Brow pre-veto (facial wind-up)
    brow_pre_fp1_uv = 70.0,
    brow_pre_fp2_uv = 85.0,
    brow_pre_veto_ms = 300,

    # NEW: short post-hold tap veto to block rebound taps immediately after a fist fires
    fist_post_veto_ms = 220,
)

# ---------------- Utils ---------------- #
def robust_mad_zpush(hist_deque: deque, value: float, maxlen: int = 200):
    hist_deque.append(value)
    if len(hist_deque) > maxlen: hist_deque.popleft()
    arr = np.asarray(hist_deque, dtype=np.float64)
    med = np.median(arr); mad = np.median(np.abs(arr - med)) + 1e-9
    return (value - med) / mad, med, mad

def butter_bandpass_inplace(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4):
    DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(x, sr, lo, hi, order, FilterTypes.BUTTERWORTH.value, 0)

def butter_bandstop_inplace(x: np.ndarray, sr: int, center: float, bw: float = 2.0):
    DataFilter.perform_bandstop(x, sr, center - bw, center + bw, 2, FilterTypes.BUTTERWORTH.value, 0)

def now_ms(): return int(time.time() * 1000)

class Debounce:
    def __init__(self, refractory_ms): self.refractory_ms=refractory_ms; self.last_ms=-10**9
    def ready(self, t_ms): return (t_ms - self.last_ms) >= self.refractory_ms
    def stamp(self, t_ms): self.last_ms=t_ms

class RangeHoldDetector:
    """Hold if amp ≥ lo for ≥ min_ms (optionally ≤ hi)."""
    def __init__(self, lo_uv, min_ms, hi_uv=None, refractory_ms=800):
        self.lo=lo_uv; self.min_ms=min_ms; self.hi=hi_uv
        self.db=Debounce(refractory_ms); self.on_ms=None; self.max_seen=0.0
    def update(self, amp_uv: float, t_ms: int):
        if amp_uv >= self.lo:
            if self.on_ms is None:
                self.on_ms=t_ms; self.max_seen=amp_uv
            else:
                self.max_seen=max(self.max_seen, amp_uv)
            if (t_ms - self.on_ms) >= self.min_ms and self.db.ready(t_ms):
                if self.hi is None or self.max_seen <= self.hi:
                    self.db.stamp(t_ms); self.on_ms=None; return True
        else:
            self.on_ms=None
        return False

# ---------------- Signal Worker ---------------- #
class SignalWorker(threading.Thread):
    """
    Blink: bipolar (Fp1−Fp2) 0.5–8 Hz → RMS(win) → MAD-z ≥ blink_k.
    Tap:   one-shot pulse (threshold crossing with min/max width).
    Fist:  sustained RMS ≥ fist_min for ≥ fist_hold_min_ms (suppresses taps).
    Brow:  Fp1 & Fp2 RMS 20–150 Hz ≥ thresholds for ≥ brow_min_ms (vetoes blinks).
    Emits: 'blink', 'tap', 'tap_hold', 'brow'.
    """
    def __init__(self, args, event_q: queue.Queue):
        super().__init__(daemon=True)
        self.args=args; self.q=event_q; self.stop_flag=threading.Event()
        self.mock=bool(args.mock or (not BRAINFLOW_AVAILABLE))
        self.board=None; self.board_id=None
        self.sr=250; self.eeg_channels=[]
        self.blink_ch_pair=None; self.tap_ch=None
        self.blink_pending_ms = None
        self.tap_pending_ms   = None

        # Simple rising-edge latch for blink z-crossing
        self._blink_above = False

        # Buffers
        self.buf_len=None
        self.buf_b1=deque(maxlen=1); self.buf_b2=deque(maxlen=1); self.buf_tap=deque(maxlen=1)
        self.blink_hist=deque(maxlen=200)

        # Timers/veto
        self.veto_blink_until_ms=-10**9
        self.veto_tap_until_ms  =-10**9   # NEW: tap veto window
        self.last_blink_ts=0.0; self.last_tap_ts=0.0

        # Detectors state
        self.fist_det = RangeHoldDetector(args.fist_min_uv, args.fist_hold_min_ms, None, args.fist_refractory_ms)
        self.brow_db = Debounce(args.brow_refractory_ms); self.brow_on_ms=None

        # Tap FSM
        self.tap_on_ms=None; self.tap_peak_uv=0.0

    def stop(self): self.stop_flag.set()

    # --- small helpers (NEW) --- #
    def _veto_tap_until(self, until_ms: int):
        self.veto_tap_until_ms = max(self.veto_tap_until_ms, until_ms)

    def _clear_tap_pending(self):
        self.tap_pending_ms = None
        self._reset_tap_fsm()

    # -------- MOCK -------- #
    def _run_mock(self):
        print("[MOCK] Keys: b=blink, t=tap, h=fist, e=brow, q=quit")
        while not self.stop_flag.is_set():
            ch=sys.stdin.read(1); now=time.time(); t_ms=now_ms()
            if ch=='b':
                if t_ms >= self.veto_blink_until_ms and (now - self.last_blink_ts) > self.args.blink_refractory_sec:
                    self.last_blink_ts=now; self.q.put(("blink", now))
            elif ch=='t':
                if (now - self.last_tap_ts) > self.args.tap_refractory_sec:
                    self.last_tap_ts=now; self.q.put(("tap", now))
            elif ch=='h': self.q.put(("tap_hold", now))
            elif ch=='e': self.q.put(("brow", now))
            elif ch=='q': self.q.put(("quit", now)); break

    # -------- Hardware init -------- #
    def _init_board(self):
        params=BrainFlowInputParams(); params.serial_port=self.args.port or "COM3"
        params.ip_address=self.args.ip or ""; params.mac_address=self.args.mac or ""; params.other_info=self.args.other or ""; params.ip_port=int(self.args.ip_port or 0)
        board_id=BoardIds.CYTON_BOARD if self.args.board=="cyton" else BoardIds.CYTON_DAISY_BOARD
        self.board_id=board_id; self.board=BoardShim(board_id, params)
        self.board.prepare_session()
        # Example configs (CH1/2 EEG; CH3 EMG). Ensure SRB2 OFF for EMG.
        self.board.config_board('x1060110X'); self.board.config_board('x2060110X'); self.board.config_board('x3010000X')
        self.board.start_stream()
        self.sr=BoardShim.get_sampling_rate(board_id); self.eeg_channels=BoardShim.get_eeg_channels(board_id)

        blks = self.eeg_channels[:2] if len(self.args.blink_ch)!=2 else self.args.blink_ch
        self.blink_ch_pair=(int(blks[0]), int(blks[1]))
        self.tap_ch=(self.eeg_channels[-1] if self.args.tap_ch is None else int(self.args.tap_ch))

        self.buf_len=int(self.sr*5)
        self.buf_b1=deque(maxlen=self.buf_len); self.buf_b2=deque(maxlen=self.buf_len); self.buf_tap=deque(maxlen=self.buf_len)
        print(f"[INFO] SR={self.sr} | Blink pair={self.blink_ch_pair} | Tap ch={self.tap_ch}")
        if self.args.notch_hz>0: print(f"[INFO] Notch {self.args.notch_hz} Hz enabled")
        t0=time.time();
        while time.time()-t0<1.0: self.board.get_board_data(); time.sleep(0.02)

    def _cleanup_board(self):
        try:
            if self.board: self.board.stop_stream(); self.board.release_session()
        except Exception: pass

    # -------- Hardware loop -------- #
    def _run_hardware(self):
        try:
            self._init_board()
            win_blink=max(1,int(self.sr*self.args.blink_window_sec))
            win_tap=max(1,int(self.sr*self.args.tap_window_sec))
            win_brow=max(1,int(self.sr*self.args.brow_window_sec))

            while not self.stop_flag.is_set():
                data=self.board.get_board_data()
                if data.size==0: time.sleep(0.01); continue
                b1,b2=self.blink_ch_pair
                for v in data[b1,:]: self.buf_b1.append(float(v))
                for v in data[b2,:]: self.buf_b2.append(float(v))
                for v in data[self.tap_ch,:]: self.buf_tap.append(float(v))

                min_need=max(win_blink*2, win_tap*2, self.sr)
                if len(self.buf_b1)<min_need or len(self.buf_b2)<min_need or len(self.buf_tap)<min_need:
                    time.sleep(0.005); continue

                t_ms=now_ms(); now=time.time()

                # ----- Blink: bipolar 0.5–8 Hz → RMS → MAD-z -----
                x1=np.asarray(self.buf_b1, dtype=np.float64).copy(); x2=np.asarray(self.buf_b2, dtype=np.float64).copy()
                xb=x1-x2
                if self.args.notch_hz>0: butter_bandstop_inplace(xb,self.sr,self.args.notch_hz)
                butter_bandpass_inplace(xb,self.sr,self.args.blink_band[0],self.args.blink_band[1])
                blink_rms=float(np.sqrt(np.mean((xb[-win_blink:])**2)))
                blink_z,_,_=robust_mad_zpush(self.blink_hist, blink_rms)
                # ----- Blink: rising-edge detect + double window -----
                blink_edge = (blink_z >= self.args.blink_k)
                # Track rising/falling to create discrete edges
                if blink_edge and not self._blink_above and t_ms >= self.veto_blink_until_ms:
                    # Got a blink edge
                    if self.blink_pending_ms is None:
                        # start waiting for a second blink
                        self.blink_pending_ms = t_ms
                    else:
                        gap = t_ms - self.blink_pending_ms
                        if self.args.double_min_gap_ms <= gap <= self.args.double_max_gap_ms:
                            # DOUBLE blink → Back
                            self.q.put(("blink_double", now))
                            self.blink_pending_ms = None
                            # post-quiet after finalizing
                            self.last_blink_ts = now
                            self.veto_blink_until_ms = max(self.veto_blink_until_ms,
                                                           t_ms + self.args.post_event_refractory_ms)
                        else:
                            # too far: finalize the old one as single and start new window
                            self.q.put(("blink", now))
                            self.last_blink_ts = now
                            self.veto_blink_until_ms = max(self.veto_blink_until_ms,
                                                           t_ms + self.args.post_event_refractory_ms)
                            self.blink_pending_ms = t_ms
                # finalize single if window expired
                if self.blink_pending_ms is not None and (t_ms - self.blink_pending_ms) > self.args.double_max_gap_ms:
                    self.q.put(("blink", now))
                    self.last_blink_ts = now
                    self.veto_blink_until_ms = max(self.veto_blink_until_ms,
                                                   t_ms + self.args.post_event_refractory_ms)
                    self.blink_pending_ms = None

                self._blink_above = blink_edge

                # ----- Tap/Fist: EMG 10–50 Hz -----
                xt=np.asarray(self.buf_tap, dtype=np.float64).copy()
                try: DataFilter.convert_to_uV(xt, self.board_id, self.tap_ch)
                except Exception: pass
                if self.args.notch_hz>0: butter_bandstop_inplace(xt,self.sr,self.args.notch_hz)
                butter_bandpass_inplace(xt,self.sr,self.args.tap_band[0],self.args.tap_band[1])
                xw=xt[-win_tap:]
                tap_env_uv=float(np.sqrt(np.mean(xw**2)))   # for fist & stability
                tap_amp_uv=float(np.max(np.abs(xw)))
                trigger_uv = max(tap_amp_uv, tap_env_uv)

                # --- PRE-VETO FOR FIST WIND-UP (NEW stronger behavior) ---
                if tap_env_uv >= self.args.fist_pre_uv:
                    # Veto taps during wind-up, and also reset any half-started tap windows
                    self._veto_tap_until(t_ms + self.args.fist_pre_veto_ms)
                    self._clear_tap_pending()
                    # Keep your blink veto here if desired
                    self.veto_blink_until_ms = max(self.veto_blink_until_ms, t_ms + self.args.fist_pre_veto_ms)

                # FIST detection first; if active, suppress taps
                fist_fired = self.fist_det.update(tap_env_uv, t_ms)
                fist_active = (self.fist_det.on_ms is not None)  # True while accumulating

                if fist_active:
                    # While accumulating the hold, keep taps vetoed (prevents stray pulses mid-hold)
                    self._veto_tap_until(t_ms + 50)  # small sliding window
                    self._clear_tap_pending()

                if fist_fired:
                    # when the hold actually fires: speak + veto taps briefly after release (rebound)
                    self._veto_tap_until(t_ms + self.args.fist_post_veto_ms)
                    self._clear_tap_pending()
                    self.veto_blink_until_ms = max(self.veto_blink_until_ms, t_ms + self.args.veto_after_tap_ms)
                    self.q.put(("tap_hold", now))

                # ----- TAP FSM (one-shot). Suppress if fist is active. -----
                if not fist_active:  # still suppress taps while a fist-hold is on
                    # --- EARLY GUARD: TAP VETO WINDOW (NEW) ---
                    if t_ms < self.veto_tap_until_ms:
                        # While vetoed, keep state clean so a half-pulse can’t later be finalized
                        self._clear_tap_pending()
                    else:
                        min_width_s = 0.08
                        above = np.mean(np.abs(xw) > self.args.tap_uv_thresh) * (win_tap / self.sr)

                        if ((now - self.last_tap_ts) > self.args.tap_refractory_sec
                                and tap_amp_uv >= self.args.tap_uv_thresh
                                and above >= min_width_s):
                            # We detected a TAP pulse (one-shot)
                            if self.tap_pending_ms is None:
                                # start double window
                                self.tap_pending_ms = t_ms
                            else:
                                gap = t_ms - self.tap_pending_ms
                                if self.args.double_min_gap_ms <= gap <= self.args.double_max_gap_ms:
                                    # DOUBLE TAP → Home
                                    self.q.put(("tap_double", now))
                                    self.tap_pending_ms = None
                                    # usual blink veto after taps
                                    self.veto_blink_until_ms = max(self.veto_blink_until_ms,
                                                                   t_ms + self.args.veto_after_tap_ms)
                                    # short post refractory to avoid bounce
                                    self.veto_blink_until_ms = max(self.veto_blink_until_ms,
                                                                   t_ms + self.args.post_event_refractory_ms)
                                else:
                                    # old pending too old → finalize it as single, start new
                                    self.q.put(("tap", now))
                                    self.veto_blink_until_ms = max(self.veto_blink_until_ms,
                                                                   t_ms + self.args.veto_after_tap_ms)
                                    self.tap_pending_ms = t_ms

                        # finalize single TAP if the window expires (only when fist not active)
                        if self.tap_pending_ms is not None and \
                                (t_ms - self.tap_pending_ms) > self.args.double_max_gap_ms:
                            self.q.put(("tap", now))
                            self.veto_blink_until_ms = max(self.veto_blink_until_ms,
                                                           t_ms + self.args.veto_after_tap_ms)
                            self.tap_pending_ms = None

                # ----- Brow: Fp1 & Fp2 20–150 Hz → RMS -----
                b1u=np.asarray(self.buf_b1, dtype=np.float64).copy(); b2u=np.asarray(self.buf_b2, dtype=np.float64).copy()
                try:
                    DataFilter.convert_to_uV(b1u, self.board_id, b1)
                    DataFilter.convert_to_uV(b2u, self.board_id, b2)
                except Exception: pass
                if self.args.notch_hz>0:
                    butter_bandstop_inplace(b1u,self.sr,self.args.notch_hz)
                    butter_bandstop_inplace(b2u,self.sr,self.args.notch_hz)
                butter_bandpass_inplace(b1u,self.sr,self.args.brow_band[0],self.args.brow_band[1])
                butter_bandpass_inplace(b2u,self.sr,self.args.brow_band[0],self.args.brow_band[1])
                brow_fp1_uv=float(np.sqrt(np.mean((b1u[-win_brow:])**2)))
                brow_fp2_uv=float(np.sqrt(np.mean((b2u[-win_brow:])**2)))
                if (brow_fp1_uv >= self.args.brow_pre_fp1_uv) and (brow_fp2_uv >= self.args.brow_pre_fp2_uv):
                    self.veto_blink_until_ms = max(self.veto_blink_until_ms, t_ms + self.args.brow_pre_veto_ms)

                if (brow_fp1_uv >= self.args.brow_fp1_uv) and (brow_fp2_uv >= self.args.brow_fp2_uv):
                    if self.brow_on_ms is None:
                        self.brow_on_ms=t_ms
                    elif (t_ms - self.brow_on_ms) >= self.args.brow_min_ms and self.brow_db.ready(t_ms):
                        self.brow_db.stamp(t_ms); self.brow_on_ms=None
                        # Veto subsequent blinks briefly
                        self.veto_blink_until_ms = max(self.veto_blink_until_ms, t_ms + self.args.brow_veto_ms)
                        self.q.put(("brow", now))
                else:
                    self.brow_on_ms=None

                time.sleep(0.005)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("[ERROR]", e)
        finally:
            self._cleanup_board()

    def _reset_tap_fsm(self):
        self.tap_on_ms=None; self.tap_peak_uv=0.0

    def run(self):
        if self.mock: self._run_mock()
        else: self._run_hardware()

# ---------------- TTS ---------------- #
class SapiTTS:
    def __init__(self, voice_name=None, rate=170, volume=1.0):
        self.voice = comtypes.client.CreateObject("SAPI.SpVoice")
        if voice_name:
            try:
                for v in self.voice.GetVoices():
                    if voice_name.lower() in v.GetDescription().lower():
                        self.voice.Voice=v; break
            except Exception: pass
        self.voice.Rate=max(-10,min(10,int(round((rate-170)/10.0))))
        self.voice.Volume=int(max(0,min(100,(volume*100.0 if volume<=1.0 else volume))))
    def say(self, text: str, purge: bool=True):
        if purge: self.voice.Speak("",2)
        self.voice.Speak(text,1)
    def stop(self): self.voice.Speak("",2)

# ---------------- UI ---------------- #
class AACApp:
    def __init__(self, root, event_q, use_tts=True, voice_name=None, rate=170, volume=1.0):
        self.root=root; self.q=event_q
        self.level=0; self.cat_idx=0; self.item_idx=0
        self.counts={"blink":0,"tap":0,"tap_hold":0,"brow":0}
        self.last_level_change_ts=time.time()
        self.current_phrase=None; self.last_spoken=None

        # Mock keys
        self.root.bind('<KeyPress-b>', lambda e: self.q.put(('blink', time.time())))
        self.root.bind('<KeyPress-t>', lambda e: self.q.put(('tap', time.time())))
        self.root.bind('<KeyPress-h>', lambda e: self.q.put(('tap_hold', time.time())))
        self.root.bind('<KeyPress-e>', lambda e: self.q.put(('brow', time.time())))
        self.root.bind('<KeyPress-q>', lambda e: self.q.put(('quit', time.time())))

        # Data
        self.categories=[c for c,_ in PHRASE_TREE]; self.phrases=[p for _,p in PHRASE_TREE]
        if "Favorites" not in self.categories:
            self.categories.insert(0,"Favorites"); self.phrases.insert(0,list(PHRASE_FAVORITES))
        self.fav_idx=self.categories.index("Favorites")

        # TTS
        self.use_tts=use_tts and SAPI_AVAILABLE; self.tts=None
        if self.use_tts:
            try:
                self.tts=SapiTTS(voice_name=voice_name, rate=rate, volume=volume)
                self.tts.say("ready", purge=True)
            except Exception as e:
                print(f"[TTS] init failed: {e}"); self.use_tts=False; self.tts=None

        # UI
        self._build_ui(); self._refresh_ui(); self.root.after(50,self._poll_events)
        self.root.after(1000, lambda: self.set_phrase("Speech path is OK"))

    def _build_ui(self):
        self.root.title("Blink + Tap AAC"); self.root.geometry("700x540"); self.root.configure(padx=16,pady=16)
        self.status = ttk.Label(
            self.root,
            text="Categories: Blink=Next • Tap=Open   |   Phrases: Blink=Next • Fist=Speak (Tap disabled)",
            font=("Segoe UI", 11)
        )
        self.status.pack(anchor='w', pady=(0, 6))

        self.metrics = ttk.Label(self.root, text=self._metrics_text(), font=("Segoe UI", 9))
        self.metrics.pack(anchor='w', pady=(0, 12))

        self.frame = ttk.Frame(self.root);
        self.frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(self.frame, columns=("col",), show='tree');
        self.tree.pack(fill='both', expand=True)

        self.help = ttk.Label(
            self.root,
            text="(Mock keys: b blink, t tap, h fist, e brow, q quit)",
            font=("Segoe UI", 9)
        )
        self.help.pack(anchor='center', pady=(12, 0))

    def _metrics_text(self):
        c=self.counts; return f"Events — blink:{c['blink']}  tap:{c['tap']}  fist:{c['tap_hold']}  brow:{c['brow']}"

    def back(self):
        if self.level == 1:
            self.level = 0
            self.item_idx = 0
            self._refresh_ui()
        else:
            # in categories, move to previous item for convenience
            self.prev_item()
    def _refresh_ui(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.level == 0:
            for i, cat in enumerate(self.categories):
                tag = 'active' if i == self.cat_idx else ''
                self.tree.insert('', 'end', text=("→ " if tag else "   ") + cat, tags=(tag,))
            self.status.config(text="Categories — Blink: Next • (dbl) Back • Tap: Open • (dbl) Home")

        else:
            current = self.categories[self.cat_idx]
            items = self.phrases[self.cat_idx]
            for i, p in enumerate(items):
                tag = 'active' if i == self.item_idx else ''
                self.tree.insert('', 'end', text=("→ " if tag else "   ") + p, tags=(tag,))
            self.status.config(text=f"Level: {current} — Blink: Next • Fist: Speak • (dbl Blink) Back • (dbl Tap) Home")

    def _poll_events(self):
        try:
            while True:
                kind, ts = self.q.get_nowait(); self._route_event(kind)
        except queue.Empty: pass
        self.metrics.config(text=self._metrics_text()); self.root.after(50,self._poll_events)

    def set_phrase(self, text: str):
        self.current_phrase=text; self.root.after(0,self._speak_latest)

    def _speak_latest(self):
        if not (self.use_tts and self.tts and self.current_phrase): return
        if self.current_phrase==self.last_spoken: return
        try:
            try: self.tts.stop()
            except Exception: pass
            print(f"[SPEAK] {self.current_phrase}"); self.tts.say(self.current_phrase, purge=True); self.last_spoken=self.current_phrase
        except Exception as e: print(f"[TTS] speak error: {e}")

    # Actions
    def next_item(self):
        if self.level==0: self.cat_idx=(self.cat_idx+1)%len(self.categories)
        else: items=self.phrases[self.cat_idx]; self.item_idx=(self.item_idx+1)%len(items)
        self._refresh_ui()
    def prev_item(self):
        if self.level==0: self.cat_idx=(self.cat_idx-1)%len(self.categories)
        else: items=self.phrases[self.cat_idx]; self.item_idx=(self.item_idx-1)%len(items)
        self._refresh_ui()
    def go_home(self): self.level=0; self.item_idx=0; self._refresh_ui()
    def select_item(self):
        now=time.time()
        if self.level==0:
            self.level=1; self.item_idx=0; self.last_level_change_ts=now
        else:
            if now - self.last_level_change_ts < 0.30: return
            phrase=self.phrases[self.cat_idx][self.item_idx]; self.set_phrase(phrase); self.level=0
        self._refresh_ui()
    def speak_selected(self):
        if self.level==0: return
        phrase=self.phrases[self.cat_idx][self.item_idx]; self.set_phrase(phrase); self.level=0; self._refresh_ui()
    def open_favorites(self):
        self.level=1; self.cat_idx=self.fav_idx; self.item_idx=0; self._refresh_ui()

    # Router
    def _route_event(self, kind: str):
        if kind not in self.counts and kind not in ("quit",):
            self.counts[kind] = 0
        if kind in self.counts:
            self.counts[kind] += 1

        # Always allowed
        if kind == "blink":
            self.next_item()

        # Tap: ONLY in Categories (level 0)
        elif kind == "tap":
            if self.level == 0:
                self.select_item()
            else:
                return  # ignore tap in phrases

        # Fist: ONLY in Phrases (level 1)
        elif kind == "tap_hold":
            if self.level == 1:
                self.speak_selected()
            else:
                return  # ignore fist in categories

        # Brow stays global (opens Favorites)
        elif kind == "brow":
            self.open_favorites()

        elif kind == "quit":
            self.shutdown();
            self.root.destroy()

        elif kind == "blink_double":
            self.back()
        elif kind == "tap_double":
            self.go_home()

    def shutdown(self):
        try: pass
        except Exception: pass

# ---------------- CLI / main ---------------- #
def parse_args():
    p=argparse.ArgumentParser(description="Blink + Tap AAC (crosstalk-hardened)")
    p.add_argument('--mock', action='store_true', help='Run without hardware; use b/t/h/e/q keys')

    # BrainFlow / Cyton
    p.add_argument('--board', default='cyton', choices=['cyton','cyton-daisy'])
    p.add_argument('--port', default=None); p.add_argument('--ip', default=None)
    p.add_argument('--ip-port', type=int, default=0); p.add_argument('--mac', default=None)
    p.add_argument('--other', default=None)

    # Channels
    p.add_argument('--blink-ch', type=int, nargs=2, default=[0,1], help='Fp1,Fp2 channels (bipolar for blink; per-ch for brow)')
    p.add_argument('--tap-ch', type=int, default=None, help='Forearm EMG bipolar P−N (SRB2 OFF).')

    # Blink/tap (original)
    p.add_argument('--blink-band', type=float, nargs=2, default=list(DEFAULTS['blink_band']))
    p.add_argument('--tap-band', type=float, nargs=2, default=list(DEFAULTS['tap_band']))
    p.add_argument('--blink-window-sec', type=float, default=DEFAULTS['blink_window_sec'])
    p.add_argument('--tap-window-sec', type=float, default=DEFAULTS['tap_window_sec'])
    p.add_argument('--blink-k', type=float, default=DEFAULTS['blink_k'])
    p.add_argument('--tap-uv-thresh', type=float, default=DEFAULTS['tap_uv_thresh'])
    p.add_argument('--tap-min-on-ms', type=int, default=DEFAULTS['tap_min_on_ms'])
    p.add_argument('--tap-max-on-ms', type=int, default=DEFAULTS['tap_max_on_ms'])
    p.add_argument('--blink-refractory-sec', type=float, default=DEFAULTS['blink_refractory_sec'])
    p.add_argument('--tap-refractory-sec', type=float, default=DEFAULTS['tap_refractory_sec'])
    p.add_argument('--veto-after-tap-ms', type=int, default=DEFAULTS['veto_after_tap_ms'])
    p.add_argument('--notch-hz', type=float, default=DEFAULTS['notch_hz'])

    # Fist & Brow
    p.add_argument('--fist-min-uv', type=float, default=DEFAULTS['fist_min_uv'])
    p.add_argument('--fist-hold-min-ms', type=int, default=DEFAULTS['fist_hold_min_ms'])
    p.add_argument('--fist-refractory-ms', type=int, default=DEFAULTS['fist_refractory_ms'])
    p.add_argument('--brow-band', type=float, nargs=2, default=list(DEFAULTS['brow_band']))
    p.add_argument('--brow-window-sec', type=float, default=DEFAULTS['brow_window_sec'])
    p.add_argument('--brow-fp1-uv', type=float, default=DEFAULTS['brow_fp1_uv'])
    p.add_argument('--brow-fp2-uv', type=float, default=DEFAULTS['brow_fp2_uv'])
    p.add_argument('--brow-min-ms', type=int, default=DEFAULTS['brow_min_ms'])
    p.add_argument('--brow-refractory-ms', type=int, default=DEFAULTS['brow_refractory_ms'])
    p.add_argument('--brow-veto-ms', type=int, default=DEFAULTS['brow_veto_ms'])

    p.add_argument('--fist-pre-uv', type=float, default=DEFAULTS['fist_pre_uv'])
    p.add_argument('--fist-pre-veto-ms', type=int, default=DEFAULTS['fist_pre_veto_ms'])
    p.add_argument('--brow-pre-fp1-uv', type=float, default=DEFAULTS['brow_pre_fp1_uv'])
    p.add_argument('--brow-pre-fp2-uv', type=float, default=DEFAULTS['brow_pre_fp2_uv'])
    p.add_argument('--brow-pre-veto-ms', type=int, default=DEFAULTS['brow_pre_veto_ms'])

    # NEW: post-hold veto duration for taps
    p.add_argument('--fist-post-veto-ms', type=int, default=DEFAULTS['fist_post_veto_ms'])

    return p.parse_args()

def main():
    args=parse_args()
    # ensure defaults for any missing keys (safety if edited)
    for k,v in DEFAULTS.items():
        if not hasattr(args, k.replace('-', '_')):
            setattr(args, k, v)
    if not BRAINFLOW_AVAILABLE and not args.mock:
        print("[WARN] BrainFlow not available; starting in MOCK mode."); args.mock=True
    evq=queue.Queue(); worker=SignalWorker(args, evq); worker.start()
    root=tk.Tk(); app=AACApp(root, evq, use_tts=True)
    def on_close(): worker.stop(); root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close); root.mainloop()

if __name__=="__main__":
    main()
