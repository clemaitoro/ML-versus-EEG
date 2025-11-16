import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report


# ---------------- Dataset with normalization ---------------- #

class WindowDataset(Dataset):
    def __init__(self, X, y):
        """
        X: (N, T, C)
        y: (N,)
        We compute per-channel mean/std over N,T and normalize.
        """
        X = X.astype(np.float32)
        self.y = torch.from_numpy(y).long()

        # channel-wise stats: mean/std over N and T
        # shape: (1, 1, C)
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True) + 1e-6

        X_norm = (X - mean) / std
        # to (N, C, T) for Conv1d
        self.X = torch.from_numpy(X_norm).permute(0, 2, 1).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------- Simple 1D CNN ---------------- #

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(96, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        h = self.net(x)       # (B, 96, 1)
        h = h.squeeze(-1)     # (B, 96)
        return self.fc(h)     # (B, n_classes)


# ---------------- Training logic ---------------- #

def train_model(
    X,
    y,
    labels,
    title,
    epochs=100,
    batch_size=64,
    lr=5e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    out_path="model.pt",
):
    print(f"[INFO] Using device: {device}")
    dataset = WindowDataset(X, y)
    n_total = len(dataset)
    n_val = max(int(0.2 * n_total), 1)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    in_channels = X.shape[2]
    n_classes = len(labels)

    model = SimpleCNN(in_channels, n_classes).to(device)

    # optional: class weights if imbalance
    counts = Counter(y.tolist())
    class_weights = []
    for i in range(n_classes):
        class_weights.append(1.0 / counts.get(i, 1))
    class_weights = np.array(class_weights, dtype=np.float32)
    class_weights /= class_weights.mean()
    class_weights_tensor = torch.from_numpy(class_weights).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / n_train

        # validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_acc = correct / total if total > 0 else 0.0

        print(f"[{title}] Epoch {epoch:02d} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    # Final report on full dataset
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_true = []
    model.eval()
    with torch.no_grad():
        for xb, yb in all_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    print(f"\n=== {title} — CNN report on all windows ===")
    print("Labels:", labels)
    print(classification_report(all_true, all_preds, target_names=labels))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels,
        },
        out_path,
    )
    print(f"[DONE] Saved model to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        choices=["eeg", "emg"],
        required=True,
        help="Which dataset to train on",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args()

    if args.modality == "eeg":
        npz_path = "dataset_eeg_blinks_brow.npz"
        title = "EEG (rest/blink/dblink/brow)"
        out_path = "eeg_cnn.pt"
    else:
        npz_path = "dataset_emg_taps_fist.npz"
        title = "EMG (rest/tap/dtap/fist)"
        out_path = "emg_cnn.pt"

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]         # (N, T, C)
    y = data["y"]         # (N,)
    labels = data["labels"]

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    train_model(
        X,
        y,
        labels,
        title=title,
        device=device,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
