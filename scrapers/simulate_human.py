#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulate normal human browsing: log-normal delays, diverse paths, browser UA.

import sys
import os

# Fix Windows console encoding for emoji output
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Make sure the project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import random
import time
import argparse
from datetime import datetime
from urllib.parse import quote

import requests

# ============================================================
# Config
# ============================================================
# Default target is local dev server; override with TARGET_URL env var
BASE_URL = os.environ.get("TARGET_URL", "http://127.0.0.1:5000")

# Real browser User-Agent strings
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
]


def _make_headers(user_agent: str) -> dict:
    """Build HTTP headers that look like a real browser."""
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "X-Simulator-Type": "human",  # label for the middleware
    }


# ============================================================
# Browsing behavior
# ============================================================

# Book ID pool (matches BOOKS_DB in target_website/views/public.py)
BOOK_IDS = list(range(1, 16))

# Category paths
CATEGORY_PATHS = [
    "/books/Computer Science",
    "/books/Literature & Fiction",
    "/books/History & Humanities",
    "/books/Art & Design",
    "/books/Business & Economics",
]


def _delay(seconds: float, reason: str = ""):
    """Pause with a printed message so we can see what's happening."""
    print(f"  ... waiting {seconds:.1f}s {reason}".strip())
    time.sleep(seconds)


def _get(session: requests.Session, path: str, referer: str = "") -> requests.Response:
    """Send a GET request and print the result."""
    headers = {}
    if referer:
        # Percent-encode non-ASCII characters in the referer path
        encoded_referer = quote(referer, safe="/?=&%")
        headers["Referer"] = BASE_URL + encoded_referer

    full_url = BASE_URL + path
    try:
        resp = session.get(full_url, headers=headers, timeout=15)
        status_icon = "OK" if resp.status_code == 200 else "ERR"
        print(f"  [{status_icon}] [{resp.status_code}] {path}")
        return resp
    except requests.RequestException as e:
        print(f"  [ERR] {path} — {e}")
        raise


# ============================================================
# Core: simulate one user's browsing session
# ============================================================
def simulate_one_session(session_id: int, base_url: str = BASE_URL):
    """
    Simulate a single user's complete browsing session.

    Browsing flow (web-like, not sequential):
      Home -> Browse book list -> Click interesting books -> Go back to list
      -> Browse a category -> Read book details -> Add to cart
      -> Go home -> Check cart -> About page -> Browse some more
    """
    ua = random.choice(USER_AGENTS)
    session = requests.Session()
    session.headers.update(_make_headers(ua))

    print(f"\n{'='*55}")
    print(f"Human User #{session_id} — browsing started")
    print(f"   UA: {ua[:60]}...")
    print(f"{'='*55}")

    # ---- Phase 1: enter the site, browse the homepage ----
    print("\nPhase 1: Homepage")
    _get(session, "/")
    _delay(random.uniform(4.0, 8.0), "(reading the homepage)")

    # ---- Phase 2: browse the book list ----
    print("\nPhase 2: Book list")
    _get(session, "/books")
    _delay(random.uniform(3.0, 6.0), "(browsing the list)")

    # Click on 2-4 books to see details
    interested_books = random.sample(BOOK_IDS, random.randint(2, 4))
    for book_id in interested_books:
        book_path = f"/book/{book_id}"
        _get(session, book_path, referer="/books")
        # Spend more time on detail pages (reading the description)
        _delay(random.uniform(5.0, 10.0), f"(reading book #{book_id})")

    # ---- Phase 3: browse by category ----
    chosen_category = random.choice(CATEGORY_PATHS)
    print(f"\nPhase 3: Category page — {chosen_category}")
    _get(session, chosen_category, referer="/books")
    _delay(random.uniform(3.0, 7.0), "(browsing the category)")

    # Open 1-2 more books from this category
    extra_books = random.sample(BOOK_IDS, random.randint(1, 2))
    for book_id in extra_books:
        _get(session, f"/book/{book_id}", referer=chosen_category)
        _delay(random.uniform(4.0, 9.0), f"(reading book #{book_id})")

    # ---- Phase 4: cart actions ----
    print("\nPhase 4: Shopping cart")
    add_id = random.choice(interested_books)
    _get(session, f"/cart?add={add_id}", referer=f"/book/{add_id}")
    _delay(random.uniform(2.0, 4.0))

    _get(session, "/cart", referer=f"/cart?add={add_id}")
    _delay(random.uniform(3.0, 6.0), "(checking the cart)")

    # ---- Phase 5: random browsing (simulate wandering) ----
    print("\nPhase 5: Wandering around")
    _get(session, "/about", referer="/cart")
    _delay(random.uniform(3.0, 6.0), "(about page)")

    # Sometimes go back to the homepage
    if random.random() < 0.6:
        _get(session, "/", referer="/about")
        _delay(random.uniform(2.0, 4.0), "(back to home)")

    # Look at one or two more books
    final_books = random.sample(BOOK_IDS, random.randint(1, 2))
    for book_id in final_books:
        _get(session, f"/book/{book_id}")
        _delay(random.uniform(3.0, 7.0), f"(quick look at #{book_id})")

    print(f"\nHuman User #{session_id} — browsing finished "
          f"({datetime.now().strftime('%H:%M:%S')})")


# ============================================================
# Main entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Human user simulator — sends human-like browsing requests to BookHaven"
    )
    parser.add_argument(
        "--users", "-n", type=int, default=3,
        help="Number of users to simulate (default: 3)"
    )
    parser.add_argument(
        "--base-url", type=str, default=BASE_URL,
        help=f"Target website URL (default: {BASE_URL})"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Human User Simulator")
    print(f"  Users:    {args.users}")
    print(f"  Target:   {args.base_url}")
    print(f"  Label:    human")
    print("=" * 55)

    start_time = datetime.now()
    for i in range(1, args.users + 1):
        try:
            simulate_one_session(i, base_url=args.base_url)
            # Gap between users: 5-15 seconds
            if i < args.users:
                gap = random.uniform(5.0, 15.0)
                print(f"\n  Next user starting in {gap:.0f}s...")
                time.sleep(gap)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n  User #{i} error: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*55}")
    print(f"Simulation complete! Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Logs appended to data/raw_logs.csv (label=human)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
