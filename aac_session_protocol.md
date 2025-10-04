# Blink + Tap AAC — Recording Session Protocol (1 Participant)
*Version: 2025-10-04*

This protocol defines **one complete session** for collecting EEG/EMG data for **blink**, **double blink**, **tap**, **double tap**, **fist hold**, and **brow raise** gestures, plus baseline rest. It includes timing, markers, and practical instructions suitable for OpenBCI GUI with the **Marker** widget.

---

## Quick Setup
- **Board:** OpenBCI Cyton @ **250 Hz**
- **Notch:** **50 Hz**
- **EMG (forearm) viewing band:** **15–55 Hz**
- **EEG viewing bands:** blinks **0.5–8 Hz**, brows **20–150 Hz**
- **Marker mapping (GUI buttons):**
  - **1** = `cue_blink`
  - **2** = `cue_blink_double`
  - **3** = `cue_tap`
  - **4** = `cue_tap_double`
  - **5** = `cue_fist_hold`
  - **6** = `cue_brow`
  - **7** = `cue_rest_on`
  - **8** = `cue_rest_off`

> **Convention:** Press the marker **at cue onset**, then perform the action **~0.4–1.0 s** after the marker.

**Double timing rules**
- Gap between the two actions: **120–320 ms** (target ~200 ms)
- Long-hold alternative (if supported by detector):
  - **Blink long-hold ≥ 320 ms** → treat as double
  - **Tap long-press ≥ 160 ms** → treat as double

---

## Global Cadence & Breaks
- Default cue cadence: **every 6 ± 1 s** (random jitter tolerated)
- Between-block break: **1–2 minutes** (eye rest + water)
- Optional micro-rests: Press **7** (rest_on) and **8** (rest_off)

---

## Block 0 — Baseline Rest
- **Duration:** 2:00 min continuous
- **Markers:** Press **7** at start (rest_on), **8** at end (rest_off)
- **Notes:** Eyes open, neutral face, hands relaxed

---

## Block 1 — Blinks (Singles & Doubles)
- **Trials:** 12 total (≈ 2.5–3 min)
  - 6 × single blink → press **1**, blink once ~0.5 s after
  - 6 × double blink → press **2**, two blinks with ~200 ms gap
- **Spacing:** ~6 ± 1 s between cues
- **Optional:** 10–12 s micro-rest after every 4 cues

**Suggested sequence:** `1,1,2,1,2,1,2,1,1,2,2,1`

---

## Block 2 — Taps (Singles & Doubles)
- **Trials:** 18 total (≈ 4 min)
  - 10 × single tap → press **3**, quick EMG tap ~0.4–0.8 s after
  - 8 × double tap → press **4**, two taps (120–320 ms gap) or one long press ≥ 160 ms
- **Spacing:** ~6 ± 1 s; optional 15–20 s mid-rest (7 then 8)

**Suggested sequence:** `3,3,4,3,4,3,3,4,3,3,3,4,3,4,3,3,4,3`

---

## Block 3 — Fist Holds
- **Trials:** 12 total (≈ 3–4 min)
  - Press **5**, then **hold fist 0.6–0.8 s**
- **Spacing:** 8–10 s between cues

**Suggested sequence:** `5 × 12`

---

## Block 4 — Brow Raises
- **Trials:** 12 total (≈ 3–4 min)
  - Press **6**, raise brows **0.25–0.4 s**, relax
- **Spacing:** 8–10 s between cues

**Suggested sequence:** `6 × 12`

---

## Block 5 — Mixed Randomized Integration
- **Trials:** 24 total (≈ 6–7 min). Shuffle the following:
  - 4 × blink (**1**), 2 × double blink (**2**)
  - 4 × tap (**3**), 2 × double tap (**4**)
  - 6 × fist hold (**5**)
  - 6 × brow (**6**)
- **Spacing:** ~6 ± 1 s

---

## Approximate Totals
- **Blink:** 10 singles, 8 doubles  
- **Tap:** 14 singles, 10 doubles  
- **Fist holds:** 18  
- **Brow:** 18  
- **Baseline rest:** 2:00 min (+ optional micro-rests)

---

## Operator Checklist
1. Fit electrodes (Fp1, Fp2; EMG forearm), verify impedance/clean contact.
2. OpenBCI GUI: set sampling to **250 Hz**, notch **50 Hz**, add **Marker** & **Data Logger** widgets.
3. Confirm marker mapping (buttons 1–8) and test a few markers (see them spike).
4. Start **Data Logger** and proceed through blocks 0→5, following sequences above.
5. Record any anomalies (sweat, re-gel, strap changes, distractions) in a notes file.
6. After each block, allow **1–2 min** break.
7. Stop logger; move files to `dataset/sub-XX/ses-YY/raw/`.

---

## Notes on Label Timing
- **Cue-based labeling:** The marker time is **t=0**. The action should occur **+0.4 to +1.0 s**.
- **Doubles:** For two discrete actions, target **~200 ms** inter-action gap (valid window **120–320 ms**).
- **Long-hold alternative:** Blink ≥ **320 ms** or Tap ≥ **160 ms** above threshold → treat as double.

---

## Optional: Rest Windows
- Use **7** (rest_on) and **8** (rest_off) to bracket clean rest segments (eyes open, neutral face).
- Aim for total rest time ≥ total event time across the session.

