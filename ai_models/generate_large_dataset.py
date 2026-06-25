#!/usr/bin/env python3
# Generate 1,000 synthetic sessions (1:1:1 human/naive-bot/evasive-bot) with
# realistic click intervals, paths, and UAs. Output: data/raw_logs.csv.

import csv
import os
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

# ============================================================
# Path Configuration
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_OUTPUT_CSV = os.path.join(_PROJECT_ROOT, "data", "raw_logs.csv")

# ============================================================
# Session Counts (1:1:1 ratio, total = 1,000)
# ============================================================
NUM_HUMAN_SESSIONS = 334
NUM_NAIVE_BOT_SESSIONS = 333
NUM_EVASIVE_BOT_SESSIONS = 333
TOTAL_SESSIONS = NUM_HUMAN_SESSIONS + NUM_NAIVE_BOT_SESSIONS + NUM_EVASIVE_BOT_SESSIONS

# ============================================================
# URL Space — matching BookHaven target website
# ============================================================
HOME = "/"
ABOUT = "/about"
BOOKS = "/books"
CATEGORIES = [
    "/books/Computer Science",
    "/books/Literature & Fiction",
    "/books/Business & Economics",
    "/books/History & Humanities",
    "/books/Science & Technology",
]
BOOK_DETAILS = [f"/book/{i}" for i in range(1, 16)]  # 15 books
CART = "/cart"

ALL_PAGE_TYPES = [HOME, ABOUT, BOOKS] + CATEGORIES + BOOK_DETAILS + [CART]

# ============================================================
# User-Agent Pools
# ============================================================
BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

BOT_UAS = [
    "python-requests/2.34.2",
    "Python-urllib/3.14",
    "curl/8.19.0",
    "Go-http-client/2.0",
    "Wget/1.24",
    "Apache-HttpClient/5.3",
    "Java/21.0.1",
    "libwww-perl/6.77",
    "axios/1.7.2",
    "node-fetch/3.3.2",
]

# ============================================================
# IP Address Pool
# ============================================================
IP_POOL = [
    "127.0.0.1",
    "192.168.1.100", "192.168.1.101", "192.168.1.102",
    "192.168.1.200", "192.168.1.201",
    "10.0.0.50", "10.0.0.51", "10.0.0.52",
    "172.16.0.10", "172.16.0.11", "172.16.0.12",
    "172.16.0.20", "172.16.0.21",
]

# Simulated date range start
BASE_DATE = datetime(2026, 6, 18, 8, 0, 0, 0)


# ============================================================
# Helper Utilities
# ============================================================
def _random_hex(rng: np.random.Generator, n: int) -> str:
    """Generate a random lowercase hex string of length n."""
    return ''.join(rng.choice(list('0123456789abcdef'), size=n))


def _session_id(rng: np.random.Generator) -> str:
    return _random_hex(rng, 16)


def _request_id(rng: np.random.Generator) -> str:
    return _random_hex(rng, 12)


def _pick(items: list, rng: np.random.Generator):
    """Pick a random item from a list using numpy RNG."""
    return items[rng.integers(0, len(items))]


def _browser_ua(rng: np.random.Generator) -> str:
    return _pick(BROWSER_UAS, rng)


def _bot_ua(rng: np.random.Generator) -> str:
    return _pick(BOT_UAS, rng)


def _ip(rng: np.random.Generator) -> str:
    return _pick(IP_POOL, rng)


# ============================================================
# Interval Distributions
# ============================================================
def human_intervals(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Log-Normal click intervals for normal human users.

    Parameters: mu = 0.75, sigma = 1.0
      - median  = exp(0.75)           ≈ 2.12 s
      - mean    = exp(0.75 + 1.0²/2) ≈ 3.49 s ≈ 3.5 s
      - std dev ≈ 3.49 * sqrt(e - 1) ≈ 4.57 s  (high jitter)

    This produces realistic human browsing: mostly 1–5 second gaps,
    with occasional 10–30 second pauses (reading, thinking).
    """
    return rng.lognormal(mean=0.75, sigma=1.0, size=n)


def naive_intervals(n: int) -> np.ndarray:
    """
    Fixed 0.1s intervals for naive bots.
    Machines don't need to read — they fire requests as fast as possible.
    """
    return np.full(n, 0.1)


def evasive_intervals(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform random delays [1.5, 4.5] seconds for evasive bots.
    Mimics human average timing but lacks the log-normal tail.
    """
    return rng.uniform(1.5, 4.5, size=n)


# ============================================================
# Path Generators
# ============================================================

def generate_human_path(rng: np.random.Generator) -> List[str]:
    """
    Generate a realistic human browsing session.

    Humans explore the site naturally:
      - Start at Home
      - Browse book listings, drill into categories
      - Open multiple book detail pages
      - Occasionally revisit a book (comparison shopping)
      - Add items to cart, check cart, continue browsing
      - Visit About page
      - Backtracking and revisits produce HIGH transition entropy

    Returns a list of URL paths for the session.
    """
    n_requests = int(rng.integers(18, 40))
    path = [HOME]
    visited_books: List[str] = []
    visited_categories: List[str] = []
    added_to_cart: List[str] = []

    for _ in range(n_requests - 1):
        roll = rng.random()
        last = path[-1]

        if roll < 0.28:
            # Visit a book detail page
            if visited_books and rng.random() < 0.35:
                # Revisit a previously seen book (backtracking → entropy signal)
                path.append(_pick(visited_books, rng))
            else:
                bk = _pick(BOOK_DETAILS, rng)
                path.append(bk)
                if bk not in visited_books:
                    visited_books.append(bk)

        elif roll < 0.45:
            # Browse a category page
            if visited_categories and rng.random() < 0.30:
                path.append(_pick(visited_categories, rng))
            else:
                cat = _pick(CATEGORIES, rng)
                path.append(cat)
                if cat not in visited_categories:
                    visited_categories.append(cat)

        elif roll < 0.58:
            # Go to all-books listing
            path.append(BOOKS)

        elif roll < 0.70:
            # Add a book to cart, then keep browsing
            if visited_books:
                bk_num = _pick(visited_books, rng).split("/")[-1]
                cart_add = f"/cart?add={bk_num}"
                path.append(cart_add)
                if bk_num not in added_to_cart:
                    added_to_cart.append(bk_num)
            else:
                path.append(CART)

        elif roll < 0.80:
            # View cart
            path.append(CART)

        elif roll < 0.90:
            # Visit About page
            path.append(ABOUT)

        else:
            # Return to Home
            path.append(HOME)

    # Ensure minimum diversity: at least 4 distinct page types visited
    unique = set(path)
    if len(unique) < 4:
        extras = [ABOUT, BOOKS, _pick(BOOK_DETAILS, rng), _pick(CATEGORIES, rng)]
        rng.shuffle(extras)
        for extra in extras[:2]:
            if extra not in unique:
                insert_pos = int(rng.integers(1, len(path)))
                path.insert(insert_pos, extra)

    return path


def generate_naive_bot_path(rng: np.random.Generator) -> List[str]:
    """
    Generate a rigid, repetitive path for a naive bot.

    Naive bots only visit book detail pages, cycling through a small
    pool of book IDs. This produces:
      - Very low unique_page_ratio (only 3–6 unique pages out of 15–30)
      - Low transition entropy (same few transitions repeat endlessly)
      - No category pages, no cart, no about → obvious pattern

    Returns a list of URL paths for the session.
    """
    # Use a small pool of book IDs (3 to 6 distinct books)
    pool_size = int(rng.integers(3, 7))
    book_pool = [f"/book/{i}" for i in range(1, pool_size + 1)]

    n_requests = int(rng.integers(15, 30))
    path = []
    for i in range(n_requests):
        path.append(book_pool[i % pool_size])

    return path


def generate_evasive_bot_path(rng: np.random.Generator) -> List[str]:
    """
    Generate a linear-sweep path for an evasive bot.

    Evasive bots crawl the site in a strict sequential order:
      Home → About → Books → Categories... → Book details... → Cart

    This produces:
      - High unique_page_ratio (almost every page is visited once)
      - Low transition entropy (predictable linear chain)
      - The LSTM detects the rigid sequential pattern despite
        human-like timing

    Returns a list of URL paths for the session.
    """
    path = [HOME]

    # Sweep through categories in shuffled order (linear, no revisits)
    sweep_cats = list(CATEGORIES)
    rng.shuffle(sweep_cats)
    num_cats = int(rng.integers(3, len(sweep_cats) + 1))
    path.extend(sweep_cats[:num_cats])

    # Linear scan through book detail pages
    start_book = int(rng.integers(1, 8))
    num_books = int(rng.integers(10, 15))
    for i in range(num_books):
        book_id = (start_book + i - 1) % 15 + 1
        path.append(f"/book/{book_id}")

    # Terminal pages in fixed order
    path.append(ABOUT)
    path.append(BOOKS)
    if rng.random() < 0.6:
        path.append(CART)

    return path


# ============================================================
# Session Builder
# ============================================================
def build_session_rows(
    session_id: str,
    ip_address: str,
    user_agent: str,
    paths: List[str],
    intervals: np.ndarray,
    response_times: np.ndarray,
    label: str,
    start_time: datetime,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Assemble a list of request-row dicts for one session.

    Each row represents one HTTP request with all 11 CSV columns.
    """
    rows = []
    ts = start_time

    for i, (p, interval, rt) in enumerate(zip(paths, intervals, response_times)):
        ts = ts + timedelta(seconds=float(interval))

        # Parse query_string from path if present (e.g., /cart?add=5)
        if "?" in p:
            clean_path, qs = p.split("?", 1)
        else:
            clean_path, qs = p, ""

        # Occasionally return 403 for naive bots (they get blocked mid-scan)
        status = 200
        if label == "crawler-naive" and i > len(paths) * 0.7:
            if rng.random() < 0.15:
                status = 403

        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "ip_address": ip_address,
            "session_id": session_id,
            "method": "GET",
            "path": clean_path,
            "query_string": qs,
            "status_code": status,
            "user_agent": user_agent,
            "request_id": _request_id(rng),
            "response_time_ms": round(float(rt), 2),
            "label": label,
        })

    return rows


# ============================================================
# Main Dataset Generator
# ============================================================
def generate_dataset(
    output_path: str = _OUTPUT_CSV,
    seed: int = 42,
    n_human: int = NUM_HUMAN_SESSIONS,
    n_naive: int = NUM_NAIVE_BOT_SESSIONS,
    n_evasive: int = NUM_EVASIVE_BOT_SESSIONS,
) -> int:
    """
    Generate the full synthetic dataset and write to CSV.

    Returns the total number of request rows generated.
    """
    total = n_human + n_naive + n_evasive
    rng = np.random.default_rng(seed)

    all_rows: List[dict] = []
    session_counter = 0

    print("=" * 65)
    print("  Synthetic Data Generator — 1,000 Sessions")
    print("=" * 65)
    print(f"  Target: {total} sessions (1:1:1 ratio)")
    print(f"  Classes:")
    print(f"    Normal Humans : {n_human}")
    print(f"    Naive Bots    : {n_naive}")
    print(f"    Evasive Bots  : {n_evasive}")
    print(f"  Seed: {seed}")
    print()

    # ================================================================
    # Phase 1 — Normal Humans
    # ================================================================
    print(f"  [1/3] Generating {n_human} Normal Human sessions ...")
    label_human_rows = 0
    for i in range(n_human):
        sid = _session_id(rng)
        ip = _ip(rng)
        ua = _browser_ua(rng)
        paths = generate_human_path(rng)
        intervals = human_intervals(len(paths), rng)
        rt_arr = rng.uniform(0.5, 50.0, size=len(paths))
        # Spread sessions across a simulated 8-hour window
        offset_seconds = float(rng.uniform(0, 28800))
        start_time = BASE_DATE + timedelta(seconds=offset_seconds)
        rows = build_session_rows(
            sid, ip, ua, paths, intervals, rt_arr, "human", start_time, rng
        )
        all_rows.extend(rows)
        label_human_rows += len(rows)
        session_counter += 1
        if (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{n_human} humans done "
                  f"({label_human_rows} requests so far)")
    print(f"    Complete: {n_human} sessions, "
          f"{label_human_rows} request rows")

    # ================================================================
    # Phase 2 — Naive Bots
    # ================================================================
    print(f"\n  [2/3] Generating {n_naive} Naive Bot sessions ...")
    label_naive_rows = 0
    for i in range(n_naive):
        sid = _session_id(rng)
        ip = _ip(rng)
        ua = _bot_ua(rng)
        paths = generate_naive_bot_path(rng)
        intervals = naive_intervals(len(paths))
        rt_arr = rng.uniform(0.3, 5.0, size=len(paths))
        offset_seconds = float(rng.uniform(0, 28800))
        start_time = BASE_DATE + timedelta(seconds=offset_seconds)
        rows = build_session_rows(
            sid, ip, ua, paths, intervals, rt_arr, "crawler-naive", start_time, rng
        )
        all_rows.extend(rows)
        label_naive_rows += len(rows)
        session_counter += 1
        if (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{n_naive} naive bots done "
                  f"({label_naive_rows} requests so far)")
    print(f"    Complete: {n_naive} sessions, "
          f"{label_naive_rows} request rows")

    # ================================================================
    # Phase 3 — Evasive Bots
    # ================================================================
    print(f"\n  [3/3] Generating {n_evasive} Evasive Bot sessions ...")
    label_evasive_rows = 0
    for i in range(n_evasive):
        sid = _session_id(rng)
        ip = _ip(rng)
        ua = _browser_ua(rng)  # Spoofed browser UA
        paths = generate_evasive_bot_path(rng)
        intervals = evasive_intervals(len(paths), rng)
        rt_arr = rng.uniform(0.5, 50.0, size=len(paths))
        offset_seconds = float(rng.uniform(0, 28800))
        start_time = BASE_DATE + timedelta(seconds=offset_seconds)
        rows = build_session_rows(
            sid, ip, ua, paths, intervals, rt_arr,
            "crawler-evasive", start_time, rng
        )
        all_rows.extend(rows)
        label_evasive_rows += len(rows)
        session_counter += 1
        if (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{n_evasive} evasive bots done "
                  f"({label_evasive_rows} requests so far)")
    print(f"    Complete: {n_evasive} sessions, "
          f"{label_evasive_rows} request rows")

    # ================================================================
    # Sort chronologically and write CSV
    # ================================================================
    all_rows.sort(key=lambda r: r["timestamp"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp", "ip_address", "session_id", "method", "path",
        "query_string", "status_code", "user_agent", "request_id",
        "response_time_ms", "label",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total_rows = len(all_rows)
    unique_sessions = len(set(r["session_id"] for r in all_rows))

    # ================================================================
    # Summary Statistics
    # ================================================================
    print(f"\n{'=' * 65}")
    print(f"  Dataset Generation Complete!")
    print(f"{'=' * 65}")
    print(f"  Output file:    {output_path}")
    print(f"  Total rows:     {total_rows:,}")
    print(f"  Unique sessions: {unique_sessions}")

    # Per-label row counts
    label_counts = Counter(r["label"] for r in all_rows)
    print(f"\n  Row-level label distribution:")
    for lbl in ["human", "crawler-naive", "crawler-evasive"]:
        cnt = label_counts.get(lbl, 0)
        bar = "#" * (cnt // 200)
        print(f"    {lbl:20s}: {cnt:6d}  {bar}")

    # Session-level class counts
    session_labels: dict = {}
    for r in all_rows:
        session_labels[r["session_id"]] = r["label"]
    session_class_counts = Counter(session_labels.values())
    print(f"\n  Session-level class distribution:")
    for lbl in ["human", "crawler-naive", "crawler-evasive"]:
        cnt = session_class_counts.get(lbl, 0)
        pct = cnt / unique_sessions * 100 if unique_sessions else 0
        print(f"    {lbl:20s}: {cnt:4d}  ({pct:.1f}%)")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  File size: {file_size_mb:.2f} MB")

    return total_rows


# ============================================================
# Verification
# ============================================================
def verify_dataset(csv_path: str) -> bool:
    """
    Quick verification that the generated dataset is compatible
    with the feature engineering pipeline (feature_engineering.py)
    and the LSTM dataset loader (lstm_classifier.py).
    """
    print(f"\n{'=' * 65}")
    print(f"  Verification — Pipeline Compatibility Check")
    print(f"{'=' * 65}")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"  Columns: {len(df.columns)} ({', '.join(df.columns)})")
        print(f"  Total rows: {len(df):,}")

        # Check required columns
        required = [
            "timestamp", "ip_address", "session_id", "method", "path",
            "query_string", "status_code", "user_agent", "request_id",
            "response_time_ms", "label",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [FAIL] Missing columns: {missing}")
            return False
        print(f"  [OK] All 11 required columns present")

        # Check labels
        valid_labels = df[df["label"].str.contains("human|crawler", na=False)]
        print(f"  Valid-label rows: {len(valid_labels):,} "
              f"({len(valid_labels) / len(df) * 100:.1f}%)")

        # Check session counts and min requests
        n_sessions = valid_labels["session_id"].nunique()
        print(f"  Unique sessions (valid labels): {n_sessions}")

        session_sizes = valid_labels.groupby("session_id").size()
        n_under2 = (session_sizes < 2).sum()
        n_under3 = (session_sizes < 3).sum()
        if n_under2 > 0:
            print(f"  [WARN]  {n_under2} sessions have < 2 requests "
                  f"(will be filtered by feature_engineering)")
        else:
            print(f"  [OK] All sessions have >= 2 requests")
        if n_under3 > 0:
            print(f"  [WARN]  {n_under3} sessions have < 3 requests "
                  f"(will be filtered by LSTM dataset)")
        else:
            print(f"  [OK] All sessions have >= 3 requests")

        # Check label distribution
        label_dist = valid_labels["label"].value_counts()
        print(f"\n  Label distribution (rows):")
        for lbl, cnt in label_dist.items():
            print(f"    {lbl}: {cnt:,}")

        # Check session-level label distribution
        session_label_dist = (
            valid_labels.groupby("session_id")["label"].first().value_counts()
        )
        print(f"\n  Session-level label distribution:")
        for lbl, cnt in session_label_dist.items():
            pct = cnt / n_sessions * 100
            print(f"    {lbl}: {cnt} ({pct:.1f}%)")

        # Quick feature engineering compatibility check
        print(f"\n  Testing feature_engineering compatibility ...")
        try:
            from ai_models.feature_engineering import build_feature_matrix
            X, y, feats = build_feature_matrix(csv_path)
            print(f"    [OK] Feature matrix built: {X.shape[0]} samples x "
                  f"{X.shape[1]} features")
            print(f"       human={int((y == 0).sum())}, "
                  f"crawler={int((y == 1).sum())}")
        except Exception as e:
            print(f"    [FAIL] Feature engineering failed: {e}")
            return False

        # Quick LSTM compatibility check
        print(f"\n  Testing LSTM dataset compatibility ...")
        try:
            from ai_models.lstm_classifier import PageSequenceDataset
            ds = PageSequenceDataset(csv_path)
            print(f"    [OK] LSTM dataset built: {len(ds)} samples")
            print(f"       human={sum(1 for _, y in ds if y.item() == 0)}, "
                  f"crawler={sum(1 for _, y in ds if y.item() == 1)}")
        except Exception as e:
            print(f"    [FAIL] LSTM dataset failed: {e}")
            return False

        print(f"\n  [OK] All verifications passed!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 1,000-session synthetic anti-crawler training dataset"
    )
    parser.add_argument(
        "--output", type=str, default=_OUTPUT_CSV,
        help=f"Output CSV path (default: {_OUTPUT_CSV})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip post-generation verification"
    )
    parser.add_argument(
        "--humans", type=int, default=NUM_HUMAN_SESSIONS,
        help=f"Number of human sessions (default: {NUM_HUMAN_SESSIONS})"
    )
    parser.add_argument(
        "--naive", type=int, default=NUM_NAIVE_BOT_SESSIONS,
        help=f"Number of naive bot sessions (default: {NUM_NAIVE_BOT_SESSIONS})"
    )
    parser.add_argument(
        "--evasive", type=int, default=NUM_EVASIVE_BOT_SESSIONS,
        help=f"Number of evasive bot sessions (default: {NUM_EVASIVE_BOT_SESSIONS})"
    )
    args = parser.parse_args()

    # Generate
    n_rows = generate_dataset(
        output_path=args.output,
        seed=args.seed,
        n_human=args.humans,
        n_naive=args.naive,
        n_evasive=args.evasive,
    )

    # Verify
    if not args.no_verify:
        ok = verify_dataset(args.output)
        if not ok:
            print("\n  [WARN]  Verification found issues — review before training.")
            sys.exit(1)

    print(f"\n  [OK] Done. {n_rows:,} request rows written to {args.output}")
