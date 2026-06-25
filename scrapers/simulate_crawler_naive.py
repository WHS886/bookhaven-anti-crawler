#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulate naive bot: fixed 0.1s delays, rigid sequential paths, script UA.

import sys
import os

# fix windows console encoding so emoji and unicode don't blow up
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# make sure we can import from the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import argparse
from datetime import datetime

import requests

# ============================================================
# Config
# ============================================================
# default target is the local dev server, override with TARGET_URL env var
BASE_URL = os.environ.get("TARGET_URL", "http://127.0.0.1:5000")

# the delay between requests — stupidly short and totally mechanical
FIXED_DELAY = 0.1

# obvious bot user-agent strings — no attempt at disguise here
NAIVE_BOT_USER_AGENTS = [
    "python-requests/2.32",
    "Python-urllib/3.14",
    "curl/8.7.1",
    "Go-http-client/2.0",
    "Wget/1.24",
    "Apache-HttpClient/5.3",
]

# all the pages on the site, visited in strict order
ALL_PATHS = [
    "/",
    "/about",
    "/books",
    "/books/Computer Science",
    "/books/Literature & Fiction",
    "/books/History & Humanities",
    "/books/Art & Design",
    "/books/Business & Economics",
    "/book/1",
    "/book/2",
    "/book/3",
    "/book/4",
    "/book/5",
    "/book/6",
    "/book/7",
    "/book/8",
    "/book/9",
    "/book/10",
    "/book/11",
    "/book/12",
    "/book/13",
    "/book/14",
    "/book/15",
    "/cart",
]


def _make_headers(user_agent: str) -> dict:
    """build headers that scream "i am a bot"."""
    return {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
        "X-Simulator-Type": "crawler-naive",
    }


# ============================================================
# Core: one round of naive crawling
# ============================================================
def simulate_one_round(round_num: int, base_url: str = BASE_URL):
    """
    blast through every page on the site in order, as fast as possible.

    args:
        round_num: which round we're on (used to rotate user-agent)
        base_url:  the target website to crawl
    """
    ua = NAIVE_BOT_USER_AGENTS[round_num % len(NAIVE_BOT_USER_AGENTS)]
    headers = _make_headers(ua)

    # fresh session each round so it gets its own session_id
    session = requests.Session()
    session.headers.update(headers)

    print(f"\n{'='*55}")
    print(f"Naive Bot Round #{round_num}")
    print(f"   UA:        {ua}")
    print(f"   Delay:     {FIXED_DELAY}s (fixed, no jitter)")
    print(f"   Pages:     {len(ALL_PATHS)}")
    print(f"{'='*55}")

    success_count = 0
    error_count = 0
    start_time = time.perf_counter()

    for i, path in enumerate(ALL_PATHS, start=1):
        full_url = base_url + path
        try:
            resp = session.get(full_url, timeout=10)
            elapsed = resp.elapsed.total_seconds()

            if resp.status_code == 200:
                status_icon = "OK"
                success_count += 1
            else:
                status_icon = "WARN"
                error_count += 1

            print(f"  [{status_icon}] [{resp.status_code}] {path:30s} "
                  f"({i}/{len(ALL_PATHS)})  {elapsed:.3f}s")

        except requests.RequestException as e:
            error_count += 1
            print(f"  [ERR] {path:30s} — {e}")

        # the naive part: always the same tiny delay, no variation at all
        if i < len(ALL_PATHS):
            time.sleep(FIXED_DELAY)

    total_time = time.perf_counter() - start_time
    avg_time = total_time / len(ALL_PATHS) * 1000

    print(f"\nRound #{round_num} stats:")
    print(f"   Success: {success_count}/{len(ALL_PATHS)}  "
          f"Errors: {error_count}  "
          f"Time: {total_time:.1f}s  "
          f"Avg: {avg_time:.0f}ms/page")


# ============================================================
# Main entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Naive bot simulator — high-speed sequential crawler, "
                    "zero subtlety at all"
    )
    parser.add_argument(
        "--rounds", "-n", type=int, default=5,
        help="Number of crawl rounds (default: 5)"
    )
    parser.add_argument(
        "--base-url", type=str, default=BASE_URL,
        help=f"Target website URL (default: {BASE_URL})"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Naive Bot Simulator (Group 2)")
    print(f"  Rounds:   {args.rounds}")
    print(f"  Delay:    {FIXED_DELAY}s (fixed)")
    print(f"  Target:   {args.base_url}")
    print(f"  Label:    crawler-naive")
    print("=" * 55)

    total_start = datetime.now()

    for r in range(1, args.rounds + 1):
        try:
            simulate_one_round(r, base_url=args.base_url)
            # tiny breather between rounds
            if r < args.rounds:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n  Round #{r} error: {e}")

    elapsed = (datetime.now() - total_start).total_seconds()
    total_requests = args.rounds * len(ALL_PATHS)
    print(f"\n{'='*55}")
    print(f"Naive crawl complete!")
    print(f"   Total requests: ~{total_requests}  "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"   Logs appended to data/raw_logs.csv (label=crawler-naive)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
