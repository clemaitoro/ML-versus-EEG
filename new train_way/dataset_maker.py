"""
build_dataset.py

Parse multiple BrainFlow CSV recordings into training datasets.

Assumptions:
- Files are BrainFlow RAW CSV (actually TSV) from the OpenBCI GUI.
- Columns:
    [sample_index, EXG0, EXG1, EXG2, ..., timestamp, marker]
  where:
    * Second-to-last column  = Unix timestamp
    * Last column           = Marker Channel
- You use:
    * EXG0 + EXG1 (columns 1,2) for blink / double-blink / brow
    * EXG2 (column 3) for tap / double-tap / fist
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ------------ CONFIG ------------ #

DATA_ROOT = r"E:\PythonProject\Licenta\new train_way\data_raw"   # TODO: set this

OUT_EEG = "dataset_eeg_blinks_brow.npz"
OUT_EMG = "dataset_emg_taps_fist.npz"

# Columns (0-based indices in the CSV)
SAMPLE_INDEX_COL = 0
BLINK_CHANNELS = [1, 2]  # EXG0, EXG1
EMG_CHANNEL = 3          # EXG2

# Fixed sampling rate for Cyton
FS_HZ = 250.0

# Global label orders (used consistently across participants/files)
EEG_LABELS = ["rest", "blink", "dblink", "brow"]
EMG_LABELS = ["rest", "tap", "dtap", "fist"]

# Marker codes
MARKER_DEFS = {
    10: ("rest", "start"),
    11: ("rest", "stop"),
    20: ("blink", "start"),
    21: ("blink", "stop"),
    22: ("dblink", "start"),
    23: ("dblink", "stop"),
    30: ("fist", "start"),
    31: ("fist", "stop"),
    40: ("tap", "start"),
    41: ("tap", "stop"),
    42: ("dtap", "start"),
    43: ("dtap", "stop"),
    50: ("brow", "start"),
    51: ("brow", "stop"),
}

# Mapping for old "weird float" markers (8-byte double read as 4-byte float)
WEIRD_MARKER_MAP = {
    0.0: 0,
    2.5625: 10,       # 10.0 as double -> float
    2.59375: 11,      # 11.0
    2.8125: 20,       # 20.0
    2.828125: 21,     # 21.0
    2.84375: 22,      # 22.0
    2.859375: 23,     # 23.0
    2.96875: 30,      # 30.0
    2.984375: 31,     # 31.0
    3.0625: 40,       # 40.0
    3.0703125: 41,    # 41.0
    3.078125: 42,     # 42.0
    3.0859375: 43,    # 43.0
    3.140625: 50,     # 50.0 (brow start)
    3.1484375: 51,    # 51.0 (brow stop)
}

# Window parameters (seconds)
WINDOW_SEC = 1.0
STEP_SEC = 0.25
MARGIN_SEC = 0.2  # ignore this much at start/end of each labeled block


# ------------ HELPERS ------------ #

def normalize_markers(marker_raw: np.ndarray) -> np.ndarray:
    """
    Normalize marker channel:
    - Fix old weird floats using WEIRD_MARKER_MAP
    - Round everything else to nearest int
    """
    marker_norm = np.zeros_like(marker_raw, dtype=int)

    # Apply weird mapping with tolerance
    assigned = np.zeros_like(marker_raw, dtype=bool)
    for weird_val, code in WEIRD_MARKER_MAP.items():
        mask = np.isclose(marker_raw, weird_val, atol=1e-6)
        marker_norm[mask] = code
        assigned[mask] = True

    # For remaining entries: round to int
    remaining_idx = ~assigned
    marker_norm[remaining_idx] = np.rint(marker_raw[remaining_idx]).astype(int)

    return marker_norm


def estimate_fs(timestamps: np.ndarray) -> float:
    """
    Estimate sampling rate from Unix timestamp column.
    BrainFlow uses Unix timestamps in seconds with microsecond precision.

    NOTE: we're not using this in the main pipeline anymore, because we want
    a fixed fs=250.0 across files. Kept for debugging if you ever want to
    check how close the timestamps are to 250 Hz.
    """
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Cannot estimate fs: timestamp diffs are zero or empty.")
    median_dt = np.median(diffs)
    return float(round(1.0 / median_dt))


def extract_segments(marker_int: np.ndarray):
    """
    From int marker array, build a dict: label -> list of (start_idx, stop_idx)
    using MARKER_DEFS.
    """
    segments = {}
    open_start = {}

    for idx, code in enumerate(marker_int):
        if code == 0:
            continue
        if code not in MARKER_DEFS:
            continue
        label, kind = MARKER_DEFS[code]
        if kind == "start":
            open_start[label] = idx
        elif kind == "stop":
            s = open_start.get(label, None)
            if s is not None and s < idx:
                segments.setdefault(label, []).append((s, idx))
                open_start[label] = None

    return segments


def build_windows_for_labels(
    data: np.ndarray,
    segments: dict,
    label_order: list,
    channel_indices: list,
    fs: float,
    window_sec: float,
    step_sec: float,
    margin_sec: float,
    participant_id: str,
    file_id: str,
):
    """
    Build sliding windows for given labels and channel indices.

    label_order: global label order (e.g. EEG_LABELS / EMG_LABELS).
    Only labels that are actually present in 'segments' will produce windows,
    but y always uses the global label-to-id mapping.
    """
    window_size = int(round(window_sec * fs))
    step_size = int(round(step_sec * fs))
    margin = int(round(margin_sec * fs))

    label_to_id = {lab: i for i, lab in enumerate(label_order)}

    X_list = []
    y_list = []
    meta_list = []

    labels_present = set(segments.keys()) & set(label_order)
    if not labels_present:
        return None, None, None

    for label in label_order:
        if label not in labels_present:
            continue
        for (start_idx, stop_idx) in segments[label]:
            s = start_idx + margin
            e = stop_idx - margin
            if e <= s + window_size:
                continue

            idx = s
            while idx + window_size <= e:
                window = data[idx : idx + window_size, channel_indices]
                X_list.append(window.astype(np.float32))
                y_list.append(label_to_id[label])

                meta_list.append(
                    {
                        "participant": participant_id,
                        "file": file_id,
                        "label": label,
                        "start_idx": int(idx),
                        "end_idx": int(idx + window_size),
                        "fs": fs,
                    }
                )
                idx += step_size

    if not X_list:
        return None, None, None

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    meta = np.array(meta_list, dtype=object)

    return X, y, meta


# ------------ MAIN PER-FILE PROCESSING ------------ #

def process_file(path: Path):
    """
    Process a single BrainFlow CSV file.
    Returns:
      - EEG windows (X_eeg, y_eeg, meta_eeg)
      - EMG windows (X_emg, y_emg, meta_emg)
    Any of them can be None if no segments/windows found.
    """
    df = pd.read_csv(path, sep="\t")

    # All channels except last two (timestamp, marker)
    data = df.iloc[:, :-2].to_numpy()
    timestamps = df.iloc[:, -2].to_numpy(dtype=float)
    marker_raw = df.iloc[:, -1].to_numpy(dtype=float)

    marker_int = normalize_markers(marker_raw)

    # Use fixed fs for Cyton recordings
    fs = FS_HZ
    # If you ever want to sanity-check:
    # estimated_fs = estimate_fs(timestamps)
    # print(path.name, "estimated fs:", estimated_fs)

    segments = extract_segments(marker_int)

    participant_id = path.parent.name
    file_id = path.name

    # EEG (blink / dblink / brow / rest) from channels 1 & 2
    X_eeg, y_eeg, meta_eeg = build_windows_for_labels(
        data=data,
        segments=segments,
        label_order=EEG_LABELS,
        channel_indices=BLINK_CHANNELS,
        fs=fs,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        margin_sec=MARGIN_SEC,
        participant_id=participant_id,
        file_id=file_id,
    )

    # EMG (tap / dtap / fist / rest) from channel 3
    X_emg, y_emg, meta_emg = build_windows_for_labels(
        data=data,
        segments=segments,
        label_order=EMG_LABELS,
        channel_indices=[EMG_CHANNEL],
        fs=fs,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        margin_sec=MARGIN_SEC,
        participant_id=participant_id,
        file_id=file_id,
    )

    return (X_eeg, y_eeg, meta_eeg), (X_emg, y_emg, meta_emg)


# ------------ GLOBAL AGGREGATION ------------ #

def main():
    root = Path(DATA_ROOT)

    eeg_X_all, eeg_y_all, eeg_meta_all = [], [], []
    emg_X_all, emg_y_all, emg_meta_all = [], [], []

    for csv_path in root.rglob("*.csv"):
        print(f"[INFO] Processing {csv_path}")
        (Xe, ye, metae), (Xm, ym, metam) = process_file(csv_path)

        if Xe is not None:
            eeg_X_all.append(Xe)
            eeg_y_all.append(ye)
            eeg_meta_all.append(metae)

        if Xm is not None:
            emg_X_all.append(Xm)
            emg_y_all.append(ym)
            emg_meta_all.append(metam)

    if eeg_X_all:
        X_eeg = np.concatenate(eeg_X_all, axis=0)
        y_eeg = np.concatenate(eeg_y_all, axis=0)
        meta_eeg = np.concatenate(eeg_meta_all, axis=0)
        np.savez(
            OUT_EEG,
            X=X_eeg,
            y=y_eeg,
            labels=np.array(EEG_LABELS, dtype=object),
            meta=meta_eeg,
        )
        print(f"[DONE] EEG dataset: {X_eeg.shape}, saved to {OUT_EEG}")
    else:
        print("[WARN] No EEG (rest/blink/dblink/brow) windows found.")

    if emg_X_all:
        X_emg = np.concatenate(emg_X_all, axis=0)
        y_emg = np.concatenate(emg_y_all, axis=0)
        meta_emg = np.concatenate(emg_meta_all, axis=0)
        np.savez(
            OUT_EMG,
            X=X_emg,
            y=y_emg,
            labels=np.array(EMG_LABELS, dtype=object),
            meta=meta_emg,
        )
        print(f"[DONE] EMG dataset: {X_emg.shape}, saved to {OUT_EMG}")
    else:
        print("[WARN] No EMG (rest/tap/dtap/fist) windows found.")


if __name__ == "__main__":
    main()
