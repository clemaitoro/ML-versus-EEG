# baseline_train.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def run_baseline(npz_path, title):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]          # (N, T, C)
    y = data["y"]          # (N,)
    labels = data["labels"]

    # flatten time x channels into one feature vector
    N, T, C = X.shape
    X_flat = X.reshape(N, T * C)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print(f"\n=== {title} ===")
    print("Labels:", labels)
    print(classification_report(y_te, y_pred, target_names=labels))

if __name__ == "__main__":
    run_baseline("dataset_eeg_blinks_brow.npz", "EEG (rest/blink/dblink/brow)")
    run_baseline("dataset_emg_taps_fist.npz", "EMG (rest/tap/dtap/fist)")
