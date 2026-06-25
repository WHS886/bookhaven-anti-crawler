# PyTorch LSTM classifier — reads page-click sequences to detect rigid bot patterns.

import os
import sys

# Make sure the project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Page path -> integer ID mapping
# ============================================================
def path_to_page_id(path: str) -> int:
    """Turn a URL path into a fixed integer ID for the LSTM embedding."""
    path = path.rstrip("/") or "/"
    if path == "/":
        return 0          # home
    if path == "/about":
        return 1          # about page
    if path == "/books":
        return 2          # all books
    if path.startswith("/books/"):
        return 3          # category filter
    if path.startswith("/book/"):
        return 4          # book detail
    if path == "/cart":
        return 5          # shopping cart
    return 6              # other / unknown

VOCAB_SIZE = 7  # number of distinct page types

# ============================================================
# Dataset: loads raw_logs.csv and builds session-level sequences
# ============================================================
class PageSequenceDataset(Dataset):
    """
    Each sample is a sequence of page IDs from one session,
    padded or trimmed to `seq_len`. The label is 1 for crawler, 0 for human.
    """

    def __init__(self, csv_path: str, seq_len: int = 10, min_requests: int = 3):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # accept human, crawler, crawler-naive, crawler-evasive — any crawler variant is a bot
        df = df[df["label"].str.contains("human|crawler", na=False)].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

        self.seq_len = seq_len
        self.sequences = []
        self.labels = []

        for sid, group in df.groupby("session_id"):
            if len(group) < min_requests:
                continue

            group = group.sort_values("timestamp")
            page_ids = [path_to_page_id(str(p)) for p in group["path"].tolist()]

            # Trim or pad to seq_len
            if len(page_ids) >= seq_len:
                page_ids = page_ids[-seq_len:]  # take the most recent clicks
            else:
                page_ids = [0] * (seq_len - len(page_ids)) + page_ids  # pad with home

            label = 1 if "crawler" in str(group["label"].iloc[0]) else 0

            self.sequences.append(page_ids)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ============================================================
# LSTM Model
# ============================================================
class LSTMClassifier(nn.Module):
    """
    Standard LSTM network to check the sequence of page clicks.
    Architecture: Embedding -> LSTM -> FC -> Sigmoid
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len)
        emb = self.embedding(x)              # (batch, seq_len, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(emb)  # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]                # (batch, hidden_dim)
        out = self.fc(last_hidden)           # (batch, 1)
        return self.sigmoid(out).squeeze(-1)  # (batch,)


# ============================================================
# Training helper
# ============================================================
def train_lstm(
    csv_path: str = None,
    model_output: str = None,
    seq_len: int = 10,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.001,
    test_split: float = 0.3,
):
    """
    Train the LSTM classifier and save it to disk.

    Args:
        csv_path:     path to raw_logs.csv
        model_output: where to save the .pt file
        seq_len:      how many page clicks to look at
        epochs:       training epochs
        batch_size:   mini-batch size
        lr:           learning rate
        test_split:   fraction of data held out for validation
    """
    if csv_path is None:
        csv_path = os.path.join(_PROJECT_ROOT, "data", "raw_logs.csv")
    if model_output is None:
        model_output = os.path.join(_PROJECT_ROOT, "data", "models", "lstm_classifier.pt")

    print("=" * 55)
    print("  LSTM Sequence Classifier — Training")
    print("=" * 55)
    print(f"  CSV:       {csv_path}")
    print(f"  Output:    {model_output}")
    print(f"  Seq len:   {seq_len}")
    print(f"  Epochs:    {epochs}")
    print(f"  Batch:     {batch_size}")
    print(f"  LR:        {lr}")

    # --- Load data ---
    dataset = PageSequenceDataset(csv_path, seq_len=seq_len)
    n_total = len(dataset)
    n_human = sum(1 for _, y in dataset if y.item() == 0)
    n_crawler = sum(1 for _, y in dataset if y.item() == 1)
    print(f"\n  Total sessions: {n_total}  (human={n_human}, crawler={n_crawler})")

    if n_total < 10:
        print("\n  Not enough data — need at least 10 sessions to train.")
        return None

    # --- Train / test split ---
    n_test = max(1, int(n_total * test_split))
    n_train = n_total - n_test
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- Model, loss, optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(vocab_size=VOCAB_SIZE).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  Device:    {device}")
    print(f"\n{'='*55}")

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                predicted_labels = (preds >= 0.5).float()
                correct += (predicted_labels == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader)

        print(f"  Epoch {epoch:3d}/{epochs}  |  loss={avg_loss:.4f}  |  val_acc={val_acc:.2%}")

    # --- Final evaluation ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            all_preds.extend((preds >= 0.5).float().cpu().tolist())
            all_labels.extend(y_batch.tolist())

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
    )
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*55}")
    print(f"  Final Evaluation (Test Set)")
    print(f"{'='*55}")
    print(f"  Accuracy:  {acc:6.2%}")
    print(f"  Precision: {prec:6.2%}")
    print(f"  Recall:    {rec:6.2%}")
    print(f"  F1 Score:  {f1:6.2%}")

    # confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix (row=actual, col=predicted):")
    print(f"  ┌──────────────────────────┐")
    print(f"  │            Pred Human  Pred Crawler │")
    print(f"  │  True Human    {tn:4d}        {fp:4d}       │")
    print(f"  │  True Crawler  {fn:4d}        {tp:4d}       │")
    print(f"  └──────────────────────────┘")

    print(f"\n  Full Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["human", "crawler"],
        digits=4,
        zero_division=0,
    ))

    # --- Save ---
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    torch.save(model.state_dict(), model_output)
    file_size_kb = os.path.getsize(model_output) / 1024
    print(f"\n  Model saved: {model_output}  ({file_size_kb:.1f} KB)")
    print("=" * 55)

    return model


# ============================================================
# Quick self-test
# ============================================================
if __name__ == "__main__":
    _csv = os.path.join(_PROJECT_ROOT, "data", "raw_logs.csv")
    train_lstm(csv_path=_csv)
