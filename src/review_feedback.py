"""What people told Oikos, and the messages it couldn't handle. Private: reads data/oikos.db on this machine only.

    python src/review_feedback.py              the last 14 days, newest first
    python src/review_feedback.py --days 60    a longer stretch
    python src/review_feedback.py --cases      missed messages as test cases to paste into tests/check_messages.py

Ratings: "down" (thumbs down, with what went wrong if they said), "up", "miss" (a message the rules and the AI could not place),
"declined" (the AI turned the question away, e.g. a loan question).
"""
import argparse
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "oikos.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--cases", action="store_true")
    a = ap.parse_args()
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    try:
        rows = c.execute("select * from feedback where at > ? order by at desc", (time.time() - a.days * 86400,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if a.cases:
        seen = set()
        for r in rows:
            msg = " ".join((r["message"] or "").split())
            if r["rating"] in ("miss", "down") and msg and msg not in seen:
                seen.add(msg)
                print(f"    ({msg[:300]!r}, None, dict()),  # {r['rating']}: {(r['note'] or r['reply'] or '')[:80]}  <- fill in what should come back")
        return
    counts = {}
    for r in rows:
        counts[r["rating"]] = counts.get(r["rating"], 0) + 1
    print(f"Last {a.days} days: " + (", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing yet"))
    for r in rows:
        when = time.strftime("%d %b %H:%M", time.localtime(r["at"]))
        print(f"\n[{r['rating'].upper()}] {when}")
        if r["message"]:
            print("  asked: " + " ".join(r["message"].split())[:400])
        if r["note"]:
            print("  said:  " + " ".join(r["note"].split())[:400])
        if r["reply"]:
            print("  reply: " + " ".join(r["reply"].split())[:300])


if __name__ == "__main__":
    main()
