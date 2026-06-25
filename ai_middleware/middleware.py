# Flask middleware: request logging + real-time 10-dim feature extraction
# + RF/LSTM ensemble prediction + bot blocking.  Hook via init_middleware(app).

import os
import sys
import csv
import math
import re
import time
import uuid
import hashlib
import threading
from collections import Counter, deque
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import joblib
from flask import Flask, request, session, g, has_request_context, render_template, make_response

# Make sure the project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ============================================================
# Timezone & Paths
# ============================================================
TZ = timezone(timedelta(hours=8))  # Asia/Shanghai

RAW_CSV = os.path.join(_PROJECT_ROOT, "data", "raw_logs.csv")
BLOCKED_CSV = os.path.join(_PROJECT_ROOT, "data", "blocked_list.csv")
RF_MODEL_PATH = os.path.join(_PROJECT_ROOT, "data", "models", "random_forest.pkl")
LSTM_MODEL_PATH = os.path.join(_PROJECT_ROOT, "data", "models", "lstm_classifier.pt")

# ============================================================
# CSV Headers
# ============================================================
RAW_HEADER = [
    "timestamp", "ip_address", "session_id", "method", "path",
    "query_string", "status_code", "user_agent", "request_id",
    "response_time_ms", "label",
]
BLOCKED_HEADER = ["timestamp", "ip_address", "session_id", "crawler_probability", "reason"]

# ============================================================
# Tuning knobs (can be set via env vars)
# ============================================================
WINDOW_SIZE = int(os.environ.get("AI_WINDOW_SIZE", 5))
RF_THRESHOLD = float(os.environ.get("AI_THRESHOLD", 0.80))
LSTM_SEQUENCE_LENGTH = int(os.environ.get("LSTM_SEQUENCE_LENGTH", 10))

# ============================================================
# Path whitelist — skip logging/analysis for these
# ============================================================
_SKIP_EXACT = {"/api/logs", "/favicon.ico"}
_SKIP_PREFIXES = ("/static/",)

# ============================================================
# Feature extraction constants (must match training)
# ============================================================
STATIC_EXTENSIONS = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot", ".map",
)
BROWSER_UA_PATTERNS = [
    r"Mozilla/5\.0.*(?:Chrome|Safari|Firefox|Edg|Edge|OPR|Opera)",
    r"Mobile.*Safari",
]
FEATURE_COLS = [
    "mean_interval", "std_interval", "total_requests", "static_ratio",
    "transition_entropy", "unique_page_ratio", "mean_response_time",
    "session_duration_sec", "request_rate", "is_browser_ua",
]

# ============================================================
# Page-to-ID mapping for LSTM sequence input
# ============================================================
def _path_to_page_id(path: str) -> int:
    """Map a URL path to an integer page ID for the LSTM model."""
    path = path.rstrip("/") or "/"
    if path == "/":
        return 0
    if path == "/about":
        return 1
    if path == "/books":
        return 2
    if path.startswith("/books/"):
        return 3
    if path.startswith("/book/"):
        return 4
    if path == "/cart":
        return 5
    return 6  # catch-all for unknown paths

NUM_PAGE_TYPES = 7  # how many distinct page IDs we map to


# ============================================================
# Helper functions
# ============================================================
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
    """
    Shannon entropy of a list of count values.
    Returns 0.0 for empty / all-zero inputs (NaN-guarded).
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    # Guard against float rounding producing -0.0 or near-zero NaN
    if ent < 0:
        ent = 0.0
    return ent


# ============================================================
# ---- Part 1: Request Collector ----
# ============================================================
_csv_lock = threading.Lock()


def _ensure_raw_csv():
    """Create raw_logs.csv with header if it doesn't exist yet."""
    os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
    if not os.path.exists(RAW_CSV):
        with open(RAW_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(RAW_HEADER)


def _ensure_blocked_csv():
    """Create blocked_list.csv with header if it doesn't exist yet."""
    os.makedirs(os.path.dirname(BLOCKED_CSV), exist_ok=True)
    if not os.path.exists(BLOCKED_CSV):
        with open(BLOCKED_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(BLOCKED_HEADER)


def _get_or_create_session_id() -> str:
    """Get the current session ID, or create a stable anonymous one."""
    if not has_request_context():
        return "no-context"

    if hasattr(session, "sid") and session.sid:
        return session.sid

    if "anonymous_id" in session:
        return session["anonymous_id"]

    # Build a hash from IP + UA + time so each "user" is unique
    raw = f"{request.remote_addr}|{request.headers.get('User-Agent', '')}|{time.time()}"
    anonymous_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    session["anonymous_id"] = anonymous_id
    return anonymous_id


def _get_client_ip() -> str:
    """Get the real client IP, handling reverse proxies."""
    if not has_request_context():
        return "0.0.0.0"

    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "0.0.0.0"


def _write_raw_row(row: list):
    """Thread-safe CSV append for raw logs."""
    _ensure_raw_csv()
    with _csv_lock:
        try:
            with open(RAW_CSV, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
        except Exception:
            pass  # never let logging break the user's request


def _write_blocked_record(ip: str, sid: str, prob: float, reason: str):
    """Append a blocked-session record to blocked_list.csv."""
    _ensure_blocked_csv()
    with _csv_lock:
        try:
            with open(BLOCKED_CSV, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                    ip, sid, f"{prob:.4f}", reason,
                ])
                f.flush()
        except Exception:
            pass


# ============================================================
# ---- Part 2: Real-time Feature Extractor ----
# ============================================================
def extract_realtime_features(requests_list: list[dict]) -> Optional[np.ndarray]:
    """
    Extract a 10-dim feature vector from a session's recent requests.

    Args:
        requests_list: list of dicts, each with keys:
            path, timestamp (float epoch sec), response_time_ms, user_agent

    Returns:
        (10,) float64 numpy array, or None if fewer than 2 requests
    """
    n = len(requests_list)
    if n < 2:
        return None

    # Sort by timestamp to make sure order is correct
    sorted_reqs = sorted(requests_list, key=lambda r: r["timestamp"])
    paths = [r["path"] for r in sorted_reqs]
    timestamps = [r["timestamp"] for r in sorted_reqs]

    # F1, F2: interval stats
    intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, n)]
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals, ddof=1)) if n > 2 else 0.0

    # F3: total requests in window
    total_requests = float(n)

    # F4: static resource ratio
    static_count = sum(1 for p in paths if _is_static(p))
    static_ratio = static_count / n

    # F5: path transition entropy
    if n >= 2:
        transitions = Counter()
        for i in range(n - 1):
            transitions[(paths[i], paths[i + 1])] += 1
        transition_entropy = _entropy(list(transitions.values()))
    else:
        transition_entropy = 0.0

    # F6: unique page ratio
    unique_pages = len(set(paths))
    unique_page_ratio = unique_pages / n

    # F7: average response time
    response_times = [r.get("response_time_ms", 0) for r in sorted_reqs]
    mean_response_time = float(np.mean(response_times))

    # F8: session duration
    duration = max(timestamps[-1] - timestamps[0], 0.001)
    session_duration_sec = float(duration)

    # F9: request rate
    request_rate = n / session_duration_sec

    # F10: browser UA flag
    ua_list = [r.get("user_agent", "") for r in sorted_reqs]
    is_browser_ua = float(max(_is_browser(ua) for ua in ua_list))

    features = np.array([
        mean_interval, std_interval, total_requests, static_ratio,
        transition_entropy, unique_page_ratio, mean_response_time,
        session_duration_sec, request_rate, is_browser_ua,
    ], dtype=np.float64)

    # NaN guard: replace any NaN/Inf with 0.0 to prevent model prediction failures.
    # NaN can arise from degenerate inputs (all-zero intervals, single repeated path,
    # missing timestamps, etc.) — this ensures the model always receives a clean,
    # finite feature vector.
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


# ============================================================
# ---- Part 3: AI Interceptor ----
# ============================================================
# Shared in-process state
_session_buffers: dict[str, deque] = {}    # session_id -> deque of request dicts
_blocked_sessions: set[str] = set()         # session IDs already blocked
_blocked_ips: set[str] = set()              # IPs blocked
_ip_blocked_count: dict[str, int] = {}      # IP -> number of blocked sessions
IP_BLOCK_THRESHOLD = 3
_state_lock = threading.Lock()

# Models loaded lazily
_rf_model = None
_rf_scaler = None
_lstm_model = None
_lstm_vocab = None


def _load_rf_model() -> bool:
    """Load the Random Forest model from disk."""
    global _rf_model, _rf_scaler
    if _rf_model is not None:
        return True

    if not os.path.exists(RF_MODEL_PATH):
        print(f"[Middleware] RF model not found: {RF_MODEL_PATH}")
        return False

    try:
        pkg = joblib.load(RF_MODEL_PATH)
        _rf_model = pkg["model"]
        _rf_scaler = pkg["scaler"]
        print(f"[Middleware] RF model loaded, window={WINDOW_SIZE}, threshold={RF_THRESHOLD:.0%}")
        return True
    except Exception as e:
        print(f"[Middleware] Failed to load RF model: {e}")
        return False


def _load_lstm_model() -> bool:
    """Load the LSTM model from disk if PyTorch is available."""
    global _lstm_model
    if _lstm_model is not None:
        return True

    if not os.path.exists(LSTM_MODEL_PATH):
        return False  # not an error — LSTM is optional

    try:
        import torch
        from ai_models.lstm_classifier import LSTMClassifier

        _lstm_model = LSTMClassifier(
            vocab_size=NUM_PAGE_TYPES,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1,
        )
        _lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location="cpu"))
        _lstm_model.eval()
        print(f"[Middleware] LSTM model loaded from {LSTM_MODEL_PATH}")
        return True
    except ImportError:
        print("[Middleware] PyTorch not installed — LSTM unavailable")
        return False
    except Exception as e:
        print(f"[Middleware] Failed to load LSTM model: {e}")
        return False


def _lstm_predict(page_sequence: list[int]) -> Optional[float]:
    """
    Run the LSTM model on a sequence of page IDs.
    Returns a crawler probability (0–1) or None on failure.
    """
    if _lstm_model is None:
        return None

    try:
        import torch
        seq_len = min(len(page_sequence), LSTM_SEQUENCE_LENGTH)
        if seq_len < 2:
            return None

        # Take the most recent LSTM_SEQUENCE_LENGTH pages
        recent = page_sequence[-LSTM_SEQUENCE_LENGTH:]
        x = torch.tensor([recent], dtype=torch.long)  # (1, seq_len)
        with torch.no_grad():
            proba = _lstm_model(x).item()
        return float(proba)
    except Exception:
        return None


# ---- Public query helpers (used by dashboard) ----
def get_blocked_sessions() -> set[str]:
    return _blocked_sessions.copy()


def get_blocked_ips() -> set[str]:
    return _blocked_ips.copy()


def get_session_buffers() -> dict[str, list]:
    with _state_lock:
        return {sid: list(buf) for sid, buf in _session_buffers.items()}


def is_lstm_loaded() -> bool:
    return _lstm_model is not None


def is_rf_loaded() -> bool:
    return _rf_model is not None


# ---- 403 Page Renderer ----
def _render_blocked(ip: str, sid: str, reason: str):
    """Render a clean 403 blocked page."""
    try:
        return make_response(
            render_template("blocked.html", ip=ip, session_id=sid, reason=reason),
            403,
        )
    except Exception:
        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>403 Forbidden</title>
<style>
body{{font-family:"Segoe UI",sans-serif;display:flex;align-items:center;
justify-content:center;height:100vh;background:#1a1a2e;color:#eee;margin:0}}
.card{{background:#16213e;padding:3rem;border-radius:12px;text-align:center;
box-shadow:0 8px 32px rgba(0,0,0,.4);max-width:500px}}
h1{{color:#e74c3c;font-size:2.5rem;margin-bottom:0}}
h2{{color:#f39c12;margin-top:0.5rem}}
p{{color:#bdc3c7;line-height:1.8}}
code{{background:#0f3460;padding:2px 8px;border-radius:4px;font-size:0.9em}}
</style></head>
<body><div class="card">
<h1>403</h1><h2>Access Blocked</h2>
<p>The AI security system detected behavior patterns similar to an automated bot.</p>
<p>IP: <code>{ip}</code></p>
<p>Session: <code>{sid[:12]}...</code></p>
<p>Reason: <code>{reason}</code></p>
<p style="margin-top:2rem;font-size:0.85rem;color:#7f8c8d">
If you are a human, please wait and try again, or contact: security@bookhaven-demo.com</p>
</div></body></html>"""
        return make_response(html, 403)


# ============================================================
# ---- Main init: wire everything into Flask ----
# ============================================================
def init_middleware(app: Flask):
    """
    Register all middleware hooks on the Flask app.
    Call this once during app startup.

    Order matters:
      1. before_request: block check -> buffer append -> AI predict -> allow/block
      2. after_request:  fill response time -> write CSV log
    """

    # Pre-load models
    with app.app_context():
        _load_rf_model()
        _load_lstm_model()

    @app.before_request
    def _before_intercept():
        """Run before every request: check block-list, buffer, run AI, decide."""
        path = request.path

        # ---- Layer 0: whitelist ----
        if path in _SKIP_EXACT or any(path.startswith(p) for p in _SKIP_PREFIXES):
            g._skip_logging = True
            return None

        ip = _get_client_ip()
        sid = _get_or_create_session_id()

        # ---- Layer 1: fast-path — already blocked ----
        if sid in _blocked_sessions:
            g._ai_label = "crawler"
            return _render_blocked(ip, sid, "session_blocked")
        if ip in _blocked_ips:
            g._ai_label = "crawler"
            return _render_blocked(ip, sid, "ip_blocked")

        # ---- Layer 1.5: User-Agent heuristic ----
        ua = request.headers.get("User-Agent", "")
        if not g.get("_ai_label"):
            ua_lower = ua.lower()

            # Most real browsers have these tokens
            browser_signs = (
                "mozilla", "applewebkit", "chrome", "safari",
                "firefox", "edg", "opera", "opr", "gecko",
            )
            # Known bot/script UA fragments
            bot_signs = (
                "python-requests", "python-urllib", "curl/", "wget/",
                "go-http-client", "apache-httpclient", "scrapy",
                "node-fetch", "libwww-perl", "okhttp/", "spider", "crawler",
            )

            is_browser = any(s in ua_lower for s in browser_signs)
            is_bot = any(s in ua_lower for s in bot_signs)

            if is_bot and not is_browser:
                g._ai_label = "crawler"
            elif is_browser:
                g._ai_label = "human"

        # ---- Layer 2: append to session buffer ----
        req_record = {
            "path": path,
            "timestamp": time.time(),
            "response_time_ms": 0.0,
            "user_agent": ua,
        }

        with _state_lock:
            if sid not in _session_buffers:
                _session_buffers[sid] = deque(maxlen=max(WINDOW_SIZE * 2, 20))
            _session_buffers[sid].append(req_record)

        g._req_record_ref = req_record

        # ---- Layer 3: AI prediction (RF + LSTM ensemble) ----
        rf_ready = _rf_model is not None
        lstm_ready = _lstm_model is not None

        if not rf_ready and not lstm_ready:
            return None  # no models loaded, just allow

        with _state_lock:
            buffer = list(_session_buffers[sid])

        crawler_scores = []

        # --- RF prediction ---
        if rf_ready and len(buffer) >= WINDOW_SIZE:
            features = extract_realtime_features(buffer)
            if features is not None:
                try:
                    X = _rf_scaler.transform(features.reshape(1, -1))
                    proba = _rf_model.predict_proba(X)[0]
                    crawler_scores.append(float(proba[1]))
                except Exception:
                    pass

        # --- LSTM prediction ---
        if lstm_ready:
            page_ids = [_path_to_page_id(r["path"]) for r in buffer]
            lstm_prob = _lstm_predict(page_ids)
            if lstm_prob is not None:
                crawler_scores.append(lstm_prob)

        # --- Ensemble: average of available model scores ---
        if crawler_scores:
            crawler_prob = sum(crawler_scores) / len(crawler_scores)
        else:
            return None  # not enough data yet

        g._ai_crawler_prob = crawler_prob
        g._ai_label = "crawler" if crawler_prob >= 0.5 else "human"

        # ---- Layer 4: block if above threshold ----
        if crawler_prob >= RF_THRESHOLD:
            reason = f"crawler_prob={crawler_prob:.2%}>={RF_THRESHOLD:.0%}"
            with _state_lock:
                _blocked_sessions.add(sid)
                _ip_blocked_count[ip] = _ip_blocked_count.get(ip, 0) + 1
            _write_blocked_record(ip, sid, crawler_prob, reason)
            print(f"[BLOCKED] {ip} | {sid[:8]}... | prob={crawler_prob:.2%}  "
                  f"(ip_block_count={_ip_blocked_count.get(ip, 0)})")
            return _render_blocked(ip, sid, reason)

        return None

    @app.after_request
    def _after_collect(response):
        """After the response is sent: fill response time and write the CSV log."""
        # --- Fill response time into the in-memory buffer record ---
        req_ref = g.pop("_req_record_ref", None)
        if req_ref is not None:
            start = g.get("start_time")
            if start:
                req_ref["response_time_ms"] = round(
                    (time.perf_counter() - start) * 1000, 2
                )

        # --- Skip logging for whitelisted paths ---
        if g.get("_skip_logging"):
            return response

        elapsed_ms = round(
            (time.perf_counter() - g.get("start_time", time.perf_counter())) * 1000, 2
        )

        # Label priority: X-Simulator-Type header > AI prediction > "unknown"
        label = request.headers.get("X-Simulator-Type", "").strip().lower()
        if not label:
            label = g.get("_ai_label", "unknown")

        row = [
            datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            g.get("client_ip", _get_client_ip()),
            g.get("session_id", _get_or_create_session_id()),
            request.method,
            request.path,
            request.query_string.decode("utf-8") if request.query_string else "",
            response.status_code,
            request.headers.get("User-Agent", ""),
            g.get("request_id", "unknown"),
            elapsed_ms,
            label,
        ]
        _write_raw_row(row)

        response.headers["X-Request-ID"] = g.get("request_id", "unknown")
        return response

    # Record start time and request metadata at the beginning
    @app.before_request
    def _before_collect():
        """First hook: record start time and assign request metadata."""
        g.start_time = time.perf_counter()
        g.request_id = uuid.uuid4().hex[:12]
        g.session_id = _get_or_create_session_id()
        g.client_ip = _get_client_ip()
