#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulate evasive bot: randomized 1.5-4.5s delays, linear sweep paths, browser UA.

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

import random
import time
import argparse
from datetime import datetime

import requests

# ============================================================
# Config
# ============================================================
# default target is the local dev server, override with TARGET_URL env var
BASE_URL = os.environ.get("TARGET_URL", "http://127.0.0.1:5000")

# delay range in seconds — picked to roughly match human hand speed
# (a real person takes about 1.5 to 4.5 seconds between clicking links)
MIN_DELAY = 1.5
MAX_DELAY = 4.5

# bot user-agent strings — yeah they're obvious, but the timing is what
# we're actually testing here, not the UA spoofing
EVASIVE_BOT_USER_AGENTS = [
    "python-requests/2.32",
    "Python-urllib/3.14",
    "curl/8.7.1",
    "Go-http-client/2.0",
    "Wget/1.24",
    "Apache-HttpClient/5.3",
]

# all the pages on the site, visited in strict order
# (this is the giveaway — a real human would jump around)
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
    """build headers that don't try too hard to look human."""
    return {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
        "X-Simulator-Type": "crawler-evasive",
    }


# ============================================================
# Core: one round of evasive crawling
# ============================================================
def simulate_one_round(round_num: int, base_url: str = BASE_URL):
    """
    walk through every page in order, but pause randomly between each one.

    the idea is that the timing looks human-ish, but the URL sequence
    is still a dead giveaway — perfect for testing LSTM defenses.

    args:
        round_num: which round we're on (used to rotate user-agent)
        base_url:  the target website to crawl
    """
    ua = EVASIVE_BOT_USER_AGENTS[round_num % len(EVASIVE_BOT_USER_AGENTS)]
    headers = _make_headers(ua)

    # fresh session each round so it gets its own session_id
    session = requests.Session()
    session.headers.update(headers)

    print(f"\n{'='*55}")
    print(f"Evasive Bot Round #{round_num}")
    print(f"   UA:        {ua}")
    print(f"   Delay:     {MIN_DELAY}s – {MAX_DELAY}s (random, uniform)")
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

        # here is the evasive part — random delay to fake human hand speed
        # the delay range (1.5–4.5s) is meant to look like a person clicking
        # around, but the sequence order is still totally linear
        if i < len(ALL_PATHS):
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            print(f"        sleep {delay:.1f}s")
            time.sleep(delay)

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
        description="Evasive bot simulator — sequential crawler with "
                    "random delays to test LSTM detection"
    )
    parser.add_argument(
        "--rounds", "-n", type=int, default=3,
        help="Number of crawl rounds (default: 3)"
    )
    parser.add_argument(
        "--base-url", type=str, default=BASE_URL,
        help=f"Target website URL (default: {BASE_URL})"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Evasive Bot Simulator (Group 3)")
    print(f"  Rounds:   {args.rounds}")
    print(f"  Delay:    {MIN_DELAY}s – {MAX_DELAY}s (random)")
    print(f"  Target:   {args.base_url}")
    print(f"  Label:    crawler-evasive")
    print("=" * 55)

    total_start = datetime.now()

    for r in range(1, args.rounds + 1):
        try:
            simulate_one_round(r, base_url=args.base_url)
            # short pause between rounds
            if r < args.rounds:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n  Round #{r} error: {e}")

    elapsed = (datetime.now() - total_start).total_seconds()
    total_requests = args.rounds * len(ALL_PATHS)
    print(f"\n{'='*55}")
    print(f"Evasive crawl complete!")
    print(f"   Total requests: ~{total_requests}  "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"   Logs appended to data/raw_logs.csv (label=crawler-evasive)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
