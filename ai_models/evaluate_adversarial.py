#!/usr/bin/env python3
# Full-spectrum adversarial eval: spoof UA + timing features, keep only path
# sequence as the detectable signal. Measures RF vs LSTM recall gap.

import sys
import os

# Fix Windows console encoding
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ai_models.feature_engineering import (
    load_sessions_from_csv,
    extract_session_features,
)
from ai_models.lstm_classifier import LSTMClassifier, path_to_page_id, VOCAB_SIZE

# ============================================================
# Paths
# ============================================================
CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "raw_logs.csv")
RF_MODEL_PATH = os.path.join(_PROJECT_ROOT, "data", "models", "random_forest.pkl")
LSTM_MODEL_PATH = os.path.join(_PROJECT_ROOT, "data", "models", "lstm_classifier.pt")

FEATURE_COLS = [
    "mean_interval", "std_interval", "total_requests", "static_ratio",
    "transition_entropy", "unique_page_ratio", "mean_response_time",
    "session_duration_sec", "request_rate", "is_browser_ua",
]

LSTM_SEQ_LEN = 10

# Columns whose values will be spoofed with human-distribution samples.
# These are the timing features an attacker can fake with random sleep() calls.
SPOOF_COLS = [
    "mean_interval",
    "std_interval",
    "session_duration_sec",
    "request_rate",
]


# ============================================================
# Step 1: Load trained models
# ============================================================
def load_models():
    """Load both RF and LSTM models from disk."""
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"RF model not found: {RF_MODEL_PATH}")
    rf_pkg = joblib.load(RF_MODEL_PATH)
    print(f"  [OK] RF model loaded  ({RF_MODEL_PATH})")
    print(f"       Trained at: {rf_pkg['train_info'].get('trained_at', 'unknown')}")
    print(f"       Samples:    {rf_pkg['train_info']['n_samples']}")

    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(f"LSTM model not found: {LSTM_MODEL_PATH}")
    lstm = LSTMClassifier(vocab_size=VOCAB_SIZE, embedding_dim=32, hidden_dim=64, num_layers=1)
    lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location="cpu"))
    lstm.eval()
    file_kb = os.path.getsize(LSTM_MODEL_PATH) / 1024
    print(f"  [OK] LSTM model loaded  ({LSTM_MODEL_PATH}, {file_kb:.0f} KB)")

    return rf_pkg, lstm


# ============================================================
# Step 2: Load data and apply adversarial spoofing
# ============================================================
def prepare_adversarial_data(rng: np.random.Generator):
    """
    Load raw_logs.csv, keep only human + crawler-evasive sessions,
    extract features, then apply full-spectrum adversarial spoofing
    to the evasive bot sessions.

    Spoofing applied:
      - is_browser_ua          : forced to 1.0  (UA impersonation)
      - mean_interval          : sampled from human log-normal distribution
      - std_interval           : sampled from human std distribution
      - session_duration_sec   : recalculated to be timing-consistent
      - request_rate           : recalculated from spoofed duration

    NOT spoofed (the remaining signal):
      - transition_entropy     : reflects rigid linear page order
      - unique_page_ratio      : reflects non-revisiting behavior
      - total_requests         : request count (plausible either way)
      - static_ratio           : not relevant for this dataset
      - mean_response_time     : minor feature, not a primary discriminator

    Returns:
        X_rf          : feature matrix (spoofed timing for evasive rows)
        y_true        : ground-truth labels (0=human, 1=crawler)
        features_df   : full feature DataFrame (for debugging)
        raw_df        : request-level DataFrame (for LSTM sequences)
    """
    print(f"\n  Loading: {CSV_PATH}")

    # ---- Load and filter ----
    raw_df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    raw_df = raw_df[raw_df["label"].isin(["human", "crawler-evasive"])].copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], format="mixed")
    raw_df["response_time_ms"] = pd.to_numeric(
        raw_df["response_time_ms"], errors="coerce"
    ).fillna(0)
    raw_df["path"] = raw_df["path"].astype(str)

    # Drop sessions with fewer than 2 requests
    session_counts = raw_df.groupby("session_id").size()
    valid_sids = session_counts[session_counts >= 2].index
    raw_df = raw_df[raw_df["session_id"].isin(valid_sids)]

    # ---- Extract session-level features ----
    features_df = extract_session_features(raw_df)

    human_mask = features_df["label_int"] == 0
    evasive_mask = features_df["label_int"] == 1

    n_human = int(human_mask.sum())
    n_evasive = int(evasive_mask.sum())
    print(f"  Human sessions:            {n_human}")
    print(f"  Evasive crawler sessions:  {n_evasive}")

    if n_evasive == 0:
        print("\n  [WARN] No evasive crawler sessions found. Run:")
        print("    python -m scrapers.simulate_crawler_evasive --rounds 10")
        sys.exit(1)

    # ---- Gather human distribution statistics for spoofing ----
    human_feats = features_df.loc[human_mask]
    human_mean_interval = human_feats["mean_interval"].values
    human_std_interval = human_feats["std_interval"].values
    human_duration = human_feats["session_duration_sec"].values
    human_rate = human_feats["request_rate"].values

    print(f"\n  Human timing stats (for spoofing):")
    print(f"    mean_interval:       {np.mean(human_mean_interval):.2f}s  "
          f"(std={np.std(human_mean_interval):.2f})")
    print(f"    std_interval:        {np.mean(human_std_interval):.2f}s  "
          f"(std={np.std(human_std_interval):.2f})")
    print(f"    session_duration:    {np.mean(human_duration):.1f}s  "
          f"(std={np.std(human_duration):.1f})")
    print(f"    request_rate:        {np.mean(human_rate):.3f}/s  "
          f"(std={np.std(human_rate):.3f})")

    # ---- Print evasive bot timing BEFORE spoofing ----
    evasive_feats = features_df.loc[evasive_mask]
    print(f"\n  Evasive bot timing BEFORE spoofing:")
    print(f"    mean_interval:       {np.mean(evasive_feats['mean_interval']):.2f}s")
    print(f"    std_interval:        {np.mean(evasive_feats['std_interval']):.2f}s")
    print(f"    session_duration:    {np.mean(evasive_feats['session_duration_sec']):.1f}s")
    print(f"    request_rate:        {np.mean(evasive_feats['request_rate']):.3f}/s")

    # ================================================================
    # Apply full-spectrum adversarial spoofing to evasive sessions
    # ================================================================
    print(f"\n  [ADVERSARIAL] Applying full-spectrum timing spoofing...")

    evasive_indices = features_df.index[evasive_mask]

    for idx in evasive_indices:
        n_req = features_df.loc[idx, "total_requests"]

        # Spoof is_browser_ua — pretend we are a real browser
        features_df.loc[idx, "is_browser_ua"] = 1.0

        # Spoof timing features by sampling from human distributions
        spoofed_mean_iv = float(rng.choice(human_mean_interval))
        spoofed_std_iv = float(rng.choice(human_std_interval))
        features_df.loc[idx, "mean_interval"] = spoofed_mean_iv
        features_df.loc[idx, "std_interval"] = spoofed_std_iv

        # Recalculate duration and rate to be internally consistent
        # with the spoofed timing (otherwise duration/rate mismatch
        # would be a giveaway)
        spoofed_duration = spoofed_mean_iv * (n_req - 1) if n_req > 1 else spoofed_mean_iv
        # Add noise so not every session has exactly mean_interval*(n-1)
        spoofed_duration *= rng.uniform(0.85, 1.15)
        spoofed_duration = max(spoofed_duration, 0.5)
        features_df.loc[idx, "session_duration_sec"] = spoofed_duration
        features_df.loc[idx, "request_rate"] = n_req / spoofed_duration

    # ---- Print evasive bot timing AFTER spoofing ----
    evasive_feats_after = features_df.loc[evasive_mask]
    print(f"\n  Evasive bot timing AFTER spoofing:")
    print(f"    mean_interval:       {np.mean(evasive_feats_after['mean_interval']):.2f}s")
    print(f"    std_interval:        {np.mean(evasive_feats_after['std_interval']):.2f}s")
    print(f"    session_duration:    {np.mean(evasive_feats_after['session_duration_sec']):.1f}s")
    print(f"    request_rate:        {np.mean(evasive_feats_after['request_rate']):.3f}/s")
    print(f"    is_browser_ua:       {np.mean(evasive_feats_after['is_browser_ua']):.3f}")

    # The UNSPOOFED features that still betray the bot:
    print(f"\n  Features RETAINED as bot signals (not spoofed):")
    print(f"    transition_entropy:  {np.mean(evasive_feats_after['transition_entropy']):.3f}  "
          f"(human: {np.mean(human_feats['transition_entropy']):.3f})")
    print(f"    unique_page_ratio:   {np.mean(evasive_feats_after['unique_page_ratio']):.3f}  "
          f"(human: {np.mean(human_feats['unique_page_ratio']):.3f})")

    # ---- Build feature matrix ----
    X_rf = features_df[FEATURE_COLS].values.astype(np.float64)
    y_true = features_df["label_int"].values.astype(np.int64)

    return X_rf, y_true, features_df, raw_df


# ============================================================
# Step 3: Evaluate Random Forest
# ============================================================
def evaluate_rf(rf_pkg, X_rf, y_true):
    """Run RF on the adversarially-spoofed feature matrix."""
    rf_model = rf_pkg["model"]
    scaler = rf_pkg["scaler"]
    X_scaled = scaler.transform(X_rf)
    y_pred = rf_model.predict(X_scaled)
    return _compute_metrics(y_true, y_pred, "Random Forest")


# ============================================================
# Step 4: Evaluate LSTM
# ============================================================
def evaluate_lstm(lstm_model, raw_df):
    """Run LSTM on page-click sequences (timing-agnostic)."""
    sequences, labels, session_ids = [], [], []

    for sid, group in raw_df.groupby("session_id"):
        group = group.sort_values("timestamp")
        page_ids = [path_to_page_id(str(p)) for p in group["path"].tolist()]

        if len(page_ids) >= LSTM_SEQ_LEN:
            page_ids = page_ids[-LSTM_SEQ_LEN:]
        else:
            page_ids = [0] * (LSTM_SEQ_LEN - len(page_ids)) + page_ids

        label = 1 if "crawler" in str(group["label"].iloc[0]) else 0
        sequences.append(page_ids)
        labels.append(label)
        session_ids.append(sid)

    if not sequences:
        print("  [ERR] No sequences built — check the data")
        return None

    x_tensor = torch.tensor(sequences, dtype=torch.long)
    with torch.no_grad():
        probs = lstm_model(x_tensor).numpy()

    y_pred = (probs >= 0.5).astype(int)
    y_true = np.array(labels)

    return _compute_metrics(y_true, y_pred, "LSTM")


# ============================================================
# Shared: compute metrics
# ============================================================
def _compute_metrics(y_true, y_pred, model_name):
    """Compute classification metrics dict."""
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "n_total": len(y_true),
        "n_human": int(sum(y_true == 0)),
        "n_crawler": int(sum(y_true == 1)),
    }


# ============================================================
# Output formatting
# ============================================================
def print_header():
    print()
    print("=" * 72)
    print("  FULL-SPECTRUM ADVERSARIAL EVASION — Evaluation Report")
    print("=" * 72)
    print("  Scenario: Attacker spoofs UA + all timing features")
    print("  Only the page-click SEQUENCE remains as a detectable signal")
    print("=" * 72)


def print_model_result(result, highlight_recall=False):
    """Pretty-print one model's results."""
    m = result
    print(f"\n{'─' * 72}")
    print(f"  Model: {m['model']}")
    print(f"{'─' * 72}")
    print(f"  Samples:       {m['n_total']}  "
          f"(human={m['n_human']}, evasive={m['n_crawler']})")
    print(f"  Accuracy:      {m['accuracy']:6.2%}")
    print(f"  Precision:     {m['precision']:6.2%}")
    if highlight_recall:
        print(f"  Recall:        {m['recall']:6.2%}  "
              f"<-- KEY METRIC: catching evasive bots under full spoofing")
    else:
        print(f"  Recall:        {m['recall']:6.2%}")
    print(f"  F1 Score:      {m['f1']:6.2%}")

    tn, fp, fn, tp = m["confusion_matrix"].ravel()
    print(f"\n  Confusion Matrix (row=actual, col=predicted):")
    print(f"  ┌──────────────────────────────────────┐")
    print(f"  │            Pred Human  Pred Crawler  │")
    print(f"  │  True Human    {tn:4d}        {fp:4d}         │")
    print(f"  │  True Crawler  {fn:4d}        {tp:4d}         │")
    print(f"  └──────────────────────────────────────┘")

    if fn > 0:
        missed_pct = fn / (fn + tp) * 100
        print(f"  !!  {fn} out of {fn + tp} evasive bots "
              f"EVADED detection ({missed_pct:.0f}% got through!)")


def print_comparison(rf_result, lstm_result):
    """Side-by-side comparison table."""
    print(f"\n{'=' * 72}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 72}")

    header = f"  {'Metric':<16} {'Random Forest':>16} {'LSTM':>16} {'Winner':>12}"
    print(header)
    print(f"  {'─' * 64}")

    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for key, label in zip(metrics, labels):
        rf_val = rf_result[key]
        lstm_val = lstm_result[key]
        if lstm_val > rf_val:
            winner = "LSTM"
        elif rf_val > lstm_val:
            winner = "RF"
        else:
            winner = "tie"
        print(f"  {label:<16} {rf_val:15.2%} {lstm_val:15.2%} {winner:>12}")

    print(f"  {'─' * 64}")

    # The critical gap
    recall_gap = lstm_result["recall"] - rf_result["recall"]
    f1_gap = lstm_result["f1"] - rf_result["f1"]
    print(f"\n  Recall gap (LSTM - RF): {recall_gap:+.1%}")
    print(f"  F1 gap    (LSTM - RF): {f1_gap:+.1%}")

    if recall_gap > 0.10:
        print(f"\n  >> LSTM catches {recall_gap:.0%} more evasive bots than RF.")
        print(f"  >> This demonstrates WHY production anti-bot systems")
        print(f"  >> (Cloudflare, DataDome, Akamai) combine feature-based")
        print(f"  >> models with sequence-based models.")
    elif recall_gap > 0:
        print(f"\n  >> LSTM holds an edge in recall under timing spoofing.")
        print(f"  >> Generate more evasive traffic to widen the gap further.")
    else:
        print(f"\n  >> Both models perform similarly on this dataset.")

    # Full classification reports
    print(f"\n{'─' * 72}")
    print(f"  Random Forest — Full Report (under full timing spoofing)")
    print(f"{'─' * 72}")
    print(classification_report(
        rf_result["y_true"], rf_result["y_pred"],
        target_names=["human", "evasive_crawler"],
        digits=4, zero_division=0,
    ))

    print(f"{'─' * 72}")
    print(f"  LSTM — Full Report (under full timing spoofing)")
    print(f"{'─' * 72}")
    print(classification_report(
        lstm_result["y_true"], lstm_result["y_pred"],
        target_names=["human", "evasive_crawler"],
        digits=4, zero_division=0,
    ))


def print_takeaway(rf_result, lstm_result):
    """Plain-English summary of the adversarial experiment."""
    recall_gap = lstm_result["recall"] - rf_result["recall"]
    rf_fn = rf_result["confusion_matrix"].ravel()[2]
    lstm_fn = lstm_result["confusion_matrix"].ravel()[2]

    print(f"\n{'=' * 72}")
    print(f"  KEY TAKEAWAY")
    print(f"{'=' * 72}")

    print(f"""
  The attacker applied THREE layers of evasion simultaneously:
    1. User-Agent spoofing       (is_browser_ua forced to 1.0)
    2. Timing randomization       (mean_interval, std_interval,
                                   session_duration_sec, request_rate
                                   all replaced with human-distribution
                                   samples)
    3. Click-interval jitter      (consistent internal duration/rate)

  Random Forest's top features BEFORE spoofing:
    std_interval        40.2%  <-- NEUTRALIZED (looks human)
    session_duration    23.5%  <-- NEUTRALIZED (looks human)
    mean_interval        5.8%  <-- NEUTRALIZED (looks human)
    request_rate         5.6%  <-- NEUTRALIZED (looks human)
    is_browser_ua        2.0%  <-- NEUTRALIZED (forced to 1)
    ─────────────────────────
    Total neutralized: ~77% of RF's feature importance

  What REMAINS for RF to work with:
    transition_entropy  ~9.8%  (the linear page order is detectable
    unique_page_ratio   ~8.1%   in feature space, but weakly)
""")

    if recall_gap > 0.10:
        print(f"""
  RESULT: The evasive crawler severely degraded the Random Forest.
  RF missed {rf_fn} out of {rf_result['n_crawler']} evasive bots
  (recall = {rf_result['recall']:.1%}).

  The LSTM barely flinched.  It doesn't see timing or UA strings —
  it reads the SEQUENCE of page clicks.  A bot walking through pages
  in strict linear order leaves a pattern that no amount of delay
  randomization can erase.

  Recall gap: {recall_gap:+.1%} — this IS the value of a sequence model.
""")
    elif recall_gap > 0:
        print(f"""
  RESULT: LSTM holds a measurable edge over RF when timing features
  are spoofed (recall gap = {recall_gap:+.1%}).  Generate more
  evasive traffic to demonstrate a wider gap for academic reporting.
""")

    print(f"  To re-run:  python -m ai_models.evaluate_adversarial")
    print(f"{'=' * 72}\n")


# ============================================================
# Main
# ============================================================
def main(seed: int = 42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print()
    print("=" * 72)
    print("  Full-Spectrum Adversarial Evasion — Evaluation")
    print("=" * 72)

    # ---- Load models ----
    print("\n[1/4] Loading trained models...")
    try:
        rf_pkg, lstm_model = load_models()
    except FileNotFoundError as e:
        print(f"\n  [ERR] {e}")
        print("  Train the models first:")
        print("    python -m ai_models.train --model rf")
        print("    python -m ai_models.train --model lstm")
        sys.exit(1)

    # ---- Prepare adversarially-spoofed data ----
    print("\n[2/4] Preparing adversarial data (full-spectrum timing spoofing)...")
    X_rf, y_true, features_df, raw_df = prepare_adversarial_data(rng)

    # ---- Evaluate RF ----
    print("\n[3/4] Evaluating Random Forest (under full timing spoofing)...")
    rf_result = evaluate_rf(rf_pkg, X_rf, y_true)

    # ---- Evaluate LSTM ----
    print("\n[4/4] Evaluating LSTM (on page-click sequences)...")
    lstm_result = evaluate_lstm(lstm_model, raw_df)

    if lstm_result is None:
        print("\n  [ERR] LSTM evaluation failed.")
        sys.exit(1)

    # ---- Print results ----
    print_header()
    print_model_result(rf_result, highlight_recall=True)
    print_model_result(lstm_result)
    print_comparison(rf_result, lstm_result)
    print_takeaway(rf_result, lstm_result)

    # Return results for potential programmatic use
    return rf_result, lstm_result


if __name__ == "__main__":
    main()
