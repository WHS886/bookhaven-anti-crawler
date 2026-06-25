# Flask app factory — BookHaven bookstore with AI middleware and /api/logs endpoint.

import sys
import os
import json as _json

# Make sure the project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask


def _register_api_routes(app: Flask):
    """Register /api/logs so the remote dashboard can fetch log data as JSON.

    Payload decoupling strategy (avoids MB-scale transfers over WAN):
      - Compute aggregate KPIs from the FULL CSV (total_traffic, blocked_threats)
      - Only send the last 500 log rows as the "logs" slice for real-time charts.
        This keeps each response at a few KB instead of several MB.
    """

    @app.route("/api/logs")
    def api_logs():
        """Return lightweight JSON: aggregate KPIs + tail-500 log slice."""
        import pandas as pd

        result = {
            "total_traffic": 0,
            "blocked_threats": 0,
            "logs": [],
            "error": None,
        }

        raw_csv = os.path.join(_PROJECT_ROOT, "data", "raw_logs.csv")
        blocked_csv = os.path.join(_PROJECT_ROOT, "data", "blocked_list.csv")

        # --- Compute aggregate KPIs from the full CSV (cheap: just len + filter) ---
        if os.path.exists(raw_csv):
            try:
                df = pd.read_csv(raw_csv, encoding="utf-8-sig")
                result["total_traffic"] = int(len(df))

                # blocked_threats = crawler-labeled rows + blocked sessions
                crawler_mask = df["label"].str.contains("crawler", na=False)
                crawler_count = int(crawler_mask.sum())

                blocked_count = 0
                if os.path.exists(blocked_csv):
                    try:
                        bdf = pd.read_csv(blocked_csv, encoding="utf-8-sig")
                        blocked_count = int(len(bdf))
                    except Exception:
                        blocked_count = 0

                result["blocked_threats"] = crawler_count + blocked_count

                # --- Core payload optimization: only send tail-500 rows ---
                #     Full 23k-row dump = ~4 MB JSON; tail-500 = ~80 KB.
                #     This eliminates WAN bandwidth saturation at 3s refresh cycles.
                tail_df = df.tail(500)
                result["logs"] = tail_df.where(pd.notna(tail_df), None).to_dict(
                    orient="records"
                )
            except Exception as e:
                result["error"] = f"raw_logs: {e}"
        else:
            result["error"] = "raw_logs.csv not found on server"

        from flask import jsonify
        return jsonify(result)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # ---- Session security (shared key for Gunicorn multi-worker) ----
    app.config["SECRET_KEY"] = os.environ.get(
        "FLASK_SECRET_KEY",
        "sof106_secure_session_fixed_key_2026"
    )
    app.config.update(
        SESSION_COOKIE_SECURE=os.environ.get("FLASK_SECURE_COOKIE", "false").lower() == "true",
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_HTTPONLY=True,
    )

    # ---- Make sure data/ folder exists (cloud environments may lack it) ----
    _DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
    os.makedirs(_DATA_DIR, exist_ok=True)
    for _sub in ("raw", "processed", "models", "logs"):
        os.makedirs(os.path.join(_DATA_DIR, _sub), exist_ok=True)

    # ---- Register blueprints ----
    from .views.public import public_bp
    app.register_blueprint(public_bp)

    # ---- Register AI security middleware ----
    from ai_middleware.middleware import init_middleware
    init_middleware(app)

    # ---- REST API for remote dashboard ----
    _register_api_routes(app)

    return app


# ============================================================
# Direct-run entry point
# ============================================================
if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

    _bind_host = os.environ.get("FLASK_HOST", "127.0.0.1")
    _display_host = "127.0.0.1" if _bind_host == "0.0.0.0" else _bind_host

    rf_loaded = os.path.exists(os.path.join(_PROJECT_ROOT, "data", "models", "random_forest.pkl"))
    lstm_loaded = os.path.exists(os.path.join(_PROJECT_ROOT, "data", "models", "lstm_classifier.pt"))

    print(f"""
╔══════════════════════════════════════════════════════╗
║   BookHaven Online Bookstore — Secure Mode          ║
║   URL: http://{_display_host}:{port}                      ║
║   Middleware: enabled (logging + AI intercept)       ║
║   RF Model:  {'loaded' if rf_loaded else 'not found'}                            ║
║   LSTM Model:{'loaded' if lstm_loaded else 'not found'}                            ║
║   Dashboard: streamlit run dashboard/app.py          ║
║   Debug:    {'on' if debug else 'off'}                               ║
╚══════════════════════════════════════════════════════╝
""")
    app.run(host=_bind_host, port=port, debug=debug)
