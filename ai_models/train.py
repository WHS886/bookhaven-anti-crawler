#!/usr/bin/env python3
# Train RF (10 hand-crafted features) or LSTM (page-click sequences) on raw_logs.csv.

import sys
import os
import argparse
import warnings

# Make sure the project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ai_models.feature_engineering import build_feature_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Default paths
# ============================================================
DEFAULT_CSV = os.path.join(_PROJECT_ROOT, "data", "raw_logs.csv")
DEFAULT_RF_OUT = os.path.join(_PROJECT_ROOT, "data", "models", "random_forest.pkl")
DEFAULT_LSTM_OUT = os.path.join(_PROJECT_ROOT, "data", "models", "lstm_classifier.pt")

# Feature column names (must match feature_engineering.py & middleware.py)
FEATURE_COLS = [
    "mean_interval",
    "std_interval",
    "total_requests",
    "static_ratio",
    "transition_entropy",
    "unique_page_ratio",
    "mean_response_time",
    "session_duration_sec",
    "request_rate",
    "is_browser_ua",
]

# Human-readable feature names for printing
FEATURE_NAMES = [
    "mean_interval",
    "std_interval",
    "total_requests",
    "static_ratio",
    "transition_entropy",
    "unique_page_ratio",
    "mean_resp_time",
    "session_duration",
    "request_rate",
    "is_browser_ua",
]


# ============================================================
# Random Forest training
# ============================================================
def train_rf(
    csv_path: str = DEFAULT_CSV,
    model_output: str = DEFAULT_RF_OUT,
    test_size: float = 0.3,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 12,
) -> dict:
    """
    Full training pipeline for the Random Forest model:
    load CSV -> extract features -> train/test split -> scale -> fit -> save.
    """

    # ===== 1. Load data & extract features =====
    print("=" * 60)
    print("  Step 1: Load data & feature engineering")
    print("=" * 60)
    print(f"  CSV: {csv_path}")

    X, y, features_df = build_feature_matrix(csv_path, feature_cols=FEATURE_COLS)

    n_human = (y == 0).sum()
    n_crawler = (y == 1).sum()
    print(f"  Samples: {len(y)}  (human={n_human}, crawler={n_crawler})")
    print(f"  Feature dim: {X.shape[1]}")
    print(f"  Class ratio: {n_crawler}/{max(n_human, 1)} = {n_crawler / max(n_human, 1):.1f}:1")

    if len(y) < 10:
        print("\n  Not enough data. Run the simulators first:")
        print("    python -m scrapers.simulate_human --users 10")
        print("    python -m scrapers.simulate_crawler_naive --rounds 20")
        print("    python -m scrapers.simulate_crawler_evasive --rounds 5")
        return {}

    if n_human < 2 or n_crawler < 2:
        print("\n  Need at least 2 samples of each class to train.")
        return {}

    # ===== 2. Train / test split =====
    print(f"\n{'='*60}")
    print(f"  Step 2: Train/test split (test_size={test_size})")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # ===== 3. Standardize =====
    print(f"\n{'='*60}")
    print(f"  Step 3: StandardScaler")
    print(f"{'='*60}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for i, (name, mean, std) in enumerate(zip(FEATURE_NAMES, scaler.mean_, scaler.scale_)):
        print(f"  {name:20s}  mean={mean:8.3f}  std={std:8.3f}")

    # ===== 4. Train Random Forest =====
    print(f"\n{'='*60}")
    print(f"  Step 4: Train Random Forest (n={n_estimators}, depth={max_depth})")
    print(f"{'='*60}")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    print("  Training done.")

    # ===== 5. Cross-validation =====
    print(f"\n{'='*60}")
    print(f"  Step 5: 5-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=min(5, min(n_human, n_crawler)), shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring="f1")
    print(f"  Per-fold F1: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean F1:     {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")

    # ===== 6. Test set evaluation =====
    print(f"\n{'='*60}")
    print(f"  Step 6: Test Set Evaluation")
    print(f"{'='*60}")

    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy:  {acc:6.2%}")
    print(f"  Precision: {prec:6.2%}")
    print(f"  Recall:    {rec:6.2%}")
    print(f"  F1 Score:  {f1:6.2%}")

    # Confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix (row=actual, col=predicted):")
    print(f"  ┌──────────────────────────┐")
    print(f"  │            Pred Human  Pred Crawler │")
    print(f"  │  True Human    {tn:4d}        {fp:4d}       │")
    print(f"  │  True Crawler  {fn:4d}        {tp:4d}       │")
    print(f"  └──────────────────────────┘")

    print(f"\n  Full Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["human", "crawler"],
        digits=4,
        zero_division=0,
    ))

    # ===== 7. Feature importance =====
    print(f"{'='*60}")
    print(f"  Step 7: Feature Importance Ranking")
    print(f"{'='*60}")

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n  {'Rank':<5} {'Feature':<20} {'Importance':<12} {'Bar'}")
    print(f"  {'-'*55}")
    for rank, idx in enumerate(indices, 1):
        bar = "#" * int(importances[idx] * 50)
        print(f"  {rank:<5} {FEATURE_NAMES[idx]:<20} {importances[idx]:.4f}       {bar}")

    # ===== 8. Save model =====
    print(f"\n{'='*60}")
    print(f"  Step 8: Save Model")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(model_output), exist_ok=True)

    model_package = {
        "model": rf,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        },
        "train_info": {
            "n_samples": len(y),
            "n_human": int(n_human),
            "n_crawler": int(n_crawler),
            "n_features": X.shape[1],
            "test_size": test_size,
            "random_state": random_state,
            "trained_at": datetime.now().isoformat(),
        },
    }

    joblib.dump(model_package, model_output)
    file_size_kb = os.path.getsize(model_output) / 1024
    print(f"  Model saved: {model_output}")
    print(f"  File size:   {file_size_kb:.1f} KB")

    # ===== 9. Summary =====
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"{'='*60}")
    print(f"""
  Model:      {model_output}
  Accuracy:   {acc:.2%}
  Precision:  {prec:.2%}
  Recall:     {rec:.2%}
  F1:         {f1:.2%}
  CV F1:      {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}
  """)

    return model_package


# ============================================================
# LSTM training (thin wrapper)
# ============================================================
def train_lstm_wrapper(
    csv_path: str = DEFAULT_CSV,
    model_output: str = DEFAULT_LSTM_OUT,
    **kwargs,
):
    """Wrapper that calls lstm_classifier.train_lstm()."""
    from ai_models.lstm_classifier import train_lstm as _train
    return _train(csv_path=csv_path, model_output=model_output, **kwargs)


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train anti-crawler models (Random Forest or LSTM)"
    )
    parser.add_argument(
        "--model", type=str, default="rf", choices=["rf", "lstm"],
        help="Which model to train: rf (Random Forest) or lstm"
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help=f"Path to raw_logs.csv (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Model output path (default depends on --model)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3,
        help="Test set fraction (default: 0.3, RF only)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Number of trees (default: 200, RF only)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=12,
        help="Max tree depth (default: 12, RF only)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=10,
        help="LSTM sequence length (default: 10, LSTM only)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="LSTM training epochs (default: 30, LSTM only)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="LSTM learning rate (default: 0.001, LSTM only)"
    )
    args = parser.parse_args()

    if args.model == "lstm":
        export_path = args.export or DEFAULT_LSTM_OUT
        train_lstm_wrapper(
            csv_path=args.csv,
            model_output=export_path,
            seq_len=args.seq_len,
            epochs=args.epochs,
            lr=args.lr,
        )
    else:
        export_path = args.export or DEFAULT_RF_OUT
        train_rf(
            csv_path=args.csv,
            model_output=export_path,
            test_size=args.test_size,
            random_state=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    main()
