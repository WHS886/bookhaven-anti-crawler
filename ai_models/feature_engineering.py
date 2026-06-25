# Session-level feature extraction: 10-dim vectors from raw request logs.
# Humans = high delay/variance/entropy + browser UA; Bots = low delay/variance + script UA.

import csv
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# Constants
# ============================================================
# File extensions that count as "static resources"
STATIC_EXTENSIONS = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot", ".map",
)

# Regex patterns to match common browser User-Agent strings
BROWSER_UA_PATTERNS = [
    r"Mozilla/5\.0.*(?:Chrome|Safari|Firefox|Edg|Edge|OPR|Opera)",
    r"Mobile.*Safari",
]


def _is_static_resource(path: str) -> bool:
    """Check if a request path is a static file (CSS, JS, image, etc.)."""
    return path.lower().endswith(STATIC_EXTENSIONS)


def _is_browser_ua(user_agent: str) -> int:
    """Return 1 if the User-Agent string looks like a real browser, 0 otherwise."""
    if not user_agent:
        return 0
    for pattern in BROWSER_UA_PATTERNS:
        if re.search(pattern, user_agent, re.IGNORECASE):
            return 1
    return 0


def _shannon_entropy(counts: list) -> float:
    """
    Shannon entropy of a list of count values.
    H = -sum(p_i * log2(p_i))
    Returns 0 if the list is empty or all zeros.
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _compute_transition_entropy(paths: list[str]) -> float:
    """
    Compute path transition entropy.

    Think of consecutive page clicks as a Markov chain.
    Count how often each (A -> B) pair happens, then compute the Shannon entropy
    of those transition counts.

    Humans: web-like browsing -> evenly spread transitions -> higher entropy
    Bots:   fixed loop pattern -> some transitions repeat a lot -> different profile
    """
    if len(paths) < 2:
        return 0.0

    transitions = Counter()
    for i in range(len(paths) - 1):
        transition = (paths[i], paths[i + 1])
        transitions[transition] += 1

    return _shannon_entropy(list(transitions.values()))


def load_sessions_from_csv(csv_path: str, min_requests: int = 2) -> pd.DataFrame:
    """
    Read raw_logs.csv and return a DataFrame grouped at the request level.

    Args:
        csv_path:     path to the CSV file
        min_requests: filter out sessions with fewer requests than this

    Returns:
        DataFrame with one row per request, including parsed timestamp and label
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Type conversions
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce").fillna(0)
    df["path"] = df["path"].astype(str)

    # Only keep rows with a known label (human, crawler, crawler-naive, crawler-evasive)
    df = df[df["label"].str.contains("human|crawler", na=False)].copy()

    # Drop sessions with too few requests
    session_counts = df.groupby("session_id").size()
    valid_sessions = session_counts[session_counts >= min_requests].index
    df = df[df["session_id"].isin(valid_sessions)]

    return df


def extract_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract session-level feature vectors from request-level data.

    Args:
        df: output of load_sessions_from_csv(), one row per request

    Returns:
        DataFrame with one row per session, columns = 10 features + label
    """
    records = []

    for session_id, group in df.groupby("session_id"):
        group = group.sort_values("timestamp")
        paths = group["path"].tolist()
        timestamps = group["timestamp"].tolist()
        n = len(paths)

        # ---- F1, F2: interval statistics ----
        if n >= 2:
            intervals = [
                (timestamps[i] - timestamps[i - 1]).total_seconds()
                for i in range(1, n)
            ]
            mean_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals, ddof=1)) if n > 2 else 0.0
        else:
            mean_interval = 0.0
            std_interval = 0.0

        # ---- F3: total requests ----
        total_requests = n

        # ---- F4: static resource ratio ----
        static_count = sum(1 for p in paths if _is_static_resource(p))
        static_ratio = static_count / n if n > 0 else 0.0

        # ---- F5: path transition entropy ----
        transition_entropy = _compute_transition_entropy(paths)

        # ---- F6: unique page ratio ----
        unique_pages = len(set(paths))
        unique_page_ratio = unique_pages / n if n > 0 else 0.0

        # ---- F7: average response time ----
        mean_response_time = float(group["response_time_ms"].mean())

        # ---- F8: session duration (seconds) ----
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        session_duration_sec = max(duration, 0.01)  # avoid division by zero

        # ---- F9: request rate (req/s) ----
        request_rate = n / session_duration_sec

        # ---- F10: browser UA flag ----
        # If any request in the session has a browser UA, count it as 1
        ua_list = group["user_agent"].dropna().unique()
        is_browser_ua = max(_is_browser_ua(str(ua)) for ua in ua_list) if len(ua_list) > 0 else 0

        # ---- Label ----
        label = group["label"].iloc[0]  # all rows in a session share the same label
        label_int = 1 if "crawler" in str(label) else 0  # 1 = crawler (any variant), 0 = human

        records.append({
            "session_id": session_id,
            "mean_interval": mean_interval,
            "std_interval": std_interval,
            "total_requests": total_requests,
            "static_ratio": static_ratio,
            "transition_entropy": transition_entropy,
            "unique_page_ratio": unique_page_ratio,
            "mean_response_time": mean_response_time,
            "session_duration_sec": session_duration_sec,
            "request_rate": request_rate,
            "is_browser_ua": is_browser_ua,
            "label": label,
            "label_int": label_int,
        })

    features_df = pd.DataFrame(records)
    return features_df


# ============================================================
# One-stop function: CSV -> feature matrix
# ============================================================
def build_feature_matrix(
    csv_path: str,
    min_requests: int = 2,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    All-in-one: CSV -> session feature matrix.

    Returns:
        X : feature matrix (numpy array)
        y : label vector (numpy array, 1=crawler, 0=human)
        df: full feature DataFrame (includes session_id and label)
    """
    df = load_sessions_from_csv(csv_path, min_requests=min_requests)
    features_df = extract_session_features(df)

    if feature_cols is None:
        feature_cols = [
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

    X = features_df[feature_cols].values.astype(np.float64)
    y = features_df["label_int"].values.astype(np.int64)

    return X, y, features_df


if __name__ == "__main__":
    # Quick self-test
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _CSV = os.path.join(_ROOT, "data", "raw_logs.csv")

    print("=" * 55)
    print("  Feature Engineering — Self Test")
    print("=" * 55)

    try:
        X, y, feats = build_feature_matrix(_CSV)
        print(f"\nTotal samples: {len(y)}")
        print(f"  human:   {(y == 0).sum()}")
        print(f"  crawler: {(y == 1).sum()}")
        print(f"  feature dim: {X.shape[1]}")
        print(f"\nFeature columns: {list(feats.columns)}")
        print(f"\nFeature stats:")
        print(feats.describe().to_string())
    except FileNotFoundError:
        print(f"\nCSV file not found: {_CSV}")
        print("Run the simulators first to generate data.")
