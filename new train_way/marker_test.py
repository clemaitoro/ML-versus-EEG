import socket
import struct
import time
import random
import sys

# ---------- LANGUAGE SWITCH (CLI) ---------- #
LANG = "en"
if "--ro" in sys.argv:
    LANG = "ro"
elif "--en" in sys.argv:
    LANG = "en"

print(f"[INFO] TTS language set to: {LANG}")

# Canonical keys for TTS lines: labels + meta prompts
TTS_TEXT = {
    "READY": {
        "en": "Ready",
        "ro": "Pregătit",
    },
    "STOP": {
        "en": "Stop",
        "ro": "Stop",
    },
    "REST": {
        "en": "Rest",
        "ro": "Relaxează-te",
    },
    "BLINK": {
        "en": "Blink",
        "ro": "Clipește",
    },
    "DOUBLE BLINK": {
        "en": "Double blink",
        "ro": "Clipește de două ori",
    },
    "FIST": {
        "en": "Fist clench",
        "ro": "Strânge pumnul",
    },
    "TAP": {
        "en": "Tap",
        "ro": "Atinge",
    },
    "DOUBLE TAP": {
        "en": "Double tap",
        "ro": "Atinge de două ori",
    },
    "BROW": {
        "en": "Eyebrow raise",
        "ro": "Ridică sprâncenele",
    },
}

def tr(key: str) -> str:
    """
    Translate a canonical TTS key or label to the selected language.
    Falls back to the key itself if not found.
    """
    entry = TTS_TEXT.get(key)
    if not entry:
        return key
    return entry.get(LANG, key)


# ---------- TTS via SAPI (same logic style as your AAC script) ---------- #
try:
    import comtypes.client
    SAPI_AVAILABLE = True
except Exception:
    SAPI_AVAILABLE = False


class SapiTTS:
    """
    Thin wrapper around SAPI.SpVoice, adapted from your AAC script.
    """
    def __init__(self, voice_name=None, rate=170, volume=1.0):
        self.voice = comtypes.client.CreateObject("SAPI.SpVoice")

        # Optional voice selection by substring
        if voice_name:
            try:
                for v in self.voice.GetVoices():
                    if voice_name.lower() in v.GetDescription().lower():
                        self.voice.Voice = v
                        break
            except Exception:
                pass

        # Map ~170 wpm to SAPI -10..10
        sapi_rate = 0
        if rate < 170:
            sapi_rate = max(-10, int((rate - 170) / 10))
        elif rate > 170:
            sapi_rate = min(10, int((rate - 170) / 10))
        self.voice.Rate = sapi_rate

        # Volume
        if volume <= 1.0:
            self.voice.Volume = int(max(0, min(100, volume * 100.0)))
        else:
            self.voice.Volume = int(max(0, min(100, volume)))

    def say(self, text: str, purge: bool = True):
        if purge:
            # SPF_PURGEBEFORESPEAK
            self.voice.Speak("", 2)
        # SPF_ASYNC (non-blocking)
        self.voice.Speak(text, 1)

    def stop(self):
        self.voice.Speak("", 2)


# =======================
# CONFIG
# =======================

# Must match Marker widget in OpenBCI GUI
MARKER_IP = "127.0.0.1"
MARKER_PORT = 12350

TRIALS_PER_CLASS = 10

# Duration (seconds) of each block
DURATIONS = {
    "REST": 10.0,
    "BLINK": 10.0,
    "DOUBLE BLINK": 10.0,
    "FIST": 10.0,
    "TAP": 10.0,
    "DOUBLE TAP": 10.0,
    "BROW": 10.0,        # eyebrow raise
}

INTER_TRIAL_PAUSE = 3.0

# Marker codes: different START/STOP codes per class
MARKERS = {
    "REST_START": 10,
    "REST_STOP": 11,
    "BLINK_START": 20,
    "BLINK_STOP": 21,
    "DBLINK_START": 22,
    "DBLINK_STOP": 23,
    "FIST_START": 30,
    "FIST_STOP": 31,
    "TAP_START": 40,
    "TAP_STOP": 41,
    "DTAP_START": 42,
    "DTAP_STOP": 43,
    "BROW_START": 50,
    "BROW_STOP": 51,
}

USE_TTS = True


# =======================
# HELPERS
# =======================

def make_trials():
    """
    Build randomized list of trials:
    (label, start_marker_key, stop_marker_key, duration)

    Labels:
      REST, BLINK, DOUBLE BLINK, FIST, TAP, DOUBLE TAP, BROW
    """
    base = [
        ("REST", "REST_START", "REST_STOP"),
        ("BLINK", "BLINK_START", "BLINK_STOP"),
        ("DOUBLE BLINK", "DBLINK_START", "DBLINK_STOP"),
        ("FIST", "FIST_START", "FIST_STOP"),
        ("TAP", "TAP_START", "TAP_STOP"),
        ("DOUBLE TAP", "DTAP_START", "DTAP_STOP"),
        ("BROW", "BROW_START", "BROW_STOP"),
    ]

    trials = []
    for label, start_key, stop_key in base:
        for _ in range(TRIALS_PER_CLASS):
            trials.append((label, start_key, stop_key, DURATIONS[label]))

    random.shuffle(trials)
    return trials


def send_marker(sock: socket.socket, value: float):
    """
    Send marker as a 4-byte big-endian float (!f).

    This matches the Marker widget's "single float" expectation and gives
    clean integer codes (10,11,20,21...) in the BrainFlow CSV marker column.
    """
    msg = struct.pack("!f", float(value))
    sock.sendto(msg, (MARKER_IP, MARKER_PORT))


def init_tts():
    if not USE_TTS:
        print("[INFO] TTS disabled (USE_TTS = False).")
        return None
    if not SAPI_AVAILABLE:
        print("[WARN] comtypes / SAPI not available, TTS disabled.")
        return None

    try:
        tts = SapiTTS(voice_name=None, rate=170, volume=1.0)
        tts.say(tr("READY"), purge=True)
        print("[INFO] SAPI TTS initialized.")
        return tts
    except Exception as e:
        print(f"[WARN] SAPI TTS init failed, disabling TTS: {e}")
        return None


def speak(tts, text_key: str):
    """
    text_key is one of the canonical keys or labels:
      READY, STOP, REST, BLINK, DOUBLE BLINK, FIST, TAP, DOUBLE TAP, BROW
    """
    text = tr(text_key)
    print(f"[VOICE-{LANG}] {text}")
    if tts is None:
        return
    try:
        tts.say(text, purge=True)
    except Exception as e:
        print(f"[WARN] TTS error, continuing without audio: {e}")


# =======================
# MAIN PROTOCOL
# =======================

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tts = init_tts()
    trials = make_trials()

    print("=== OpenBCI marker + SAPI TTS script ===")
    print(f"Will run {len(trials)} blocks ({TRIALS_PER_CLASS} per class) "
          f"on {MARKER_IP}:{MARKER_PORT}")
    print("Marker codes:")
    for k, v in MARKERS.items():
        print(f"  {k:12s} -> {v}")
    print("\nMake sure:")
    print("  1) Cyton is streaming in the OpenBCI GUI")
    print("  2) Marker widget is open")
    print("  3) Marker widget Receive IP = 127.0.0.1")
    print("  4) Marker widget Receive Port = 12350\n")

    input("Press ENTER to start the protocol...")

    for i, (label, start_key, stop_key, duration) in enumerate(trials, start=1):
        print(f"\nTrial {i}/{len(trials)}: {label}")

        start_val = MARKERS[start_key]
        send_marker(sock, start_val)
        print(f"  Sent marker {start_val} ({start_key})")

        # Voice cue in selected language
        speak(tts, label)

        print(f"  {label} for {duration} seconds...")
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted during block. Stopping protocol.")
            break

        stop_val = MARKERS[stop_key]
        send_marker(sock, stop_val)
        print(f"  Sent marker {stop_val} ({stop_key})")
        speak(tts, "STOP")

        if i < len(trials):
            print(f"  Inter-trial pause {INTER_TRIAL_PAUSE} s...")
            try:
                time.sleep(INTER_TRIAL_PAUSE)
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted during pause. Stopping protocol.")
                break

    print("\nProtocol finished (or interrupted). You can stop recording in the GUI now.")
    if tts is not None:
        try:
            tts.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
