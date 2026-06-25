"""
AI Security Middleware
======================
Request interception, behavior data collection, and real-time AI classification.

Sub-modules:
  middleware — the main module: collector + feature extractor + interceptor all in one
"""

from .middleware import init_middleware, get_blocked_sessions, get_blocked_ips, get_session_buffers
from .middleware import extract_realtime_features, is_rf_loaded, is_lstm_loaded

__all__ = [
    "init_middleware",
    "get_blocked_sessions",
    "get_blocked_ips",
    "get_session_buffers",
    "extract_realtime_features",
    "is_rf_loaded",
    "is_lstm_loaded",
]
