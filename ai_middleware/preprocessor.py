
"""
Real-time Feature Preprocessor
==============================
Unlike the "offline batch" mode in ai_models/feature_engineering.py,
this module extracts feature vectors in real time from a single session's
sliding window of the most recent N requests, so the interceptor can call
the model for online predictions.

Design notes:
  - Input: a session's most recent N request records (list[dict])
  - Output: 10-dim feature vector (aligned with training feature_cols)
  - Works even when the window has fewer than N requests (early warning)
"""

import math
import re
from collections import Counter
from typing import Optional

import numpy as np

# ---- Constants (must match the training pipeline) ----
STATIC_EXTENSIONS = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot", ".map",
)
BROWSER_UA_PATTERNS = [
    r"Mozilla/5\.0.*(?:Chrome|Safari|Firefox|Edg|Edge|OPR|Opera)",
    r"Mobile.*Safari",
]
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


# ---- Utility helpers ----
def _is_static(path: str) -> bool:
    return path.lower().endswith(STATIC_EXTENSIONS)


def _is_browser(ua: str) -> int:
    if not ua:
        return 0
    for pattern in BROWSER_UA_PATTERNS:
        if re.search(pattern, str(ua), re.IGNORECASE):
            return 1
    return 0


def _entropy(counts: list) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ============================================================
# Core: extract features from a request list
# ============================================================
def extract_realtime_features(requests: list[dict]) -> Optional[np.ndarray]:
    """
    Extract a 10-dim feature vector from a session's recent N requests.

    Args:
      requests : list[dict], each dict should have at least:
          path, timestamp (float epoch seconds), response_time_ms, user_agent

    Returns:
      (10,) numpy float64 array, or None when fewer than 2 requests
    """
    n = len(requests)
    if n < 2:
        return None

    # Sort by timestamp to make sure order is right
    sorted_reqs = sorted(requests, key=lambda r: r["timestamp"])
    paths = [r["path"] for r in sorted_reqs]
    timestamps = [r["timestamp"] for r in sorted_reqs]

    # ---- F1, F2: interval statistics ----
    intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, n)]
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals, ddof=1)) if n > 2 else 0.0

    # ---- F3: request count in window ----
    total_requests = float(n)

    # ---- F4: static resource ratio ----
    static_count = sum(1 for p in paths if _is_static(p))
    static_ratio = static_count / n

    # ---- F5: path transition entropy ----
    if n >= 2:
        transitions = Counter()
        for i in range(n - 1):
            transitions[(paths[i], paths[i + 1])] += 1
        transition_entropy = _entropy(list(transitions.values()))
    else:
        transition_entropy = 0.0

    # ---- F6: unique page ratio ----
    unique_pages = len(set(paths))
    unique_page_ratio = unique_pages / n

    # ---- F7: average response time ----
    response_times = [r.get("response_time_ms", 0) for r in sorted_reqs]
    mean_response_time = float(np.mean(response_times))

    # ---- F8: window duration ----
    duration = max(timestamps[-1] - timestamps[0], 0.001)
    session_duration_sec = float(duration)

    # ---- F9: request rate ----
    request_rate = n / session_duration_sec

    # ---- F10: browser UA flag ----
    ua_list = [r.get("user_agent", "") for r in sorted_reqs]
    is_browser_ua = float(max(_is_browser(ua) for ua in ua_list))

    # Pack into a vector with the same column order used during training
    features = np.array([
        mean_interval,
        std_interval,
        total_requests,
        static_ratio,
        transition_entropy,
        unique_page_ratio,
        mean_response_time,
        session_duration_sec,
        request_rate,
        is_browser_ua,
    ], dtype=np.float64)

    return features
