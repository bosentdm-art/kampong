"""Geocode every unique HDB block + street with the OneMap search API.

Reads ONEMAP_TOKEN from .env. Results are appended to
data/processed/block_coords.csv as they arrive, so an interrupted run resumes
where it stopped. Blocks with no match go to block_coords_failed.csv.

Usage:
    python src/geocode_blocks.py --limit 20   # small test run
    python src/geocode_blocks.py              # all blocks
"""
import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEARCH = "https://www.onemap.gov.sg/api/common/elastic/search"


def load_env():
    """Tiny .env reader so we do not need another dependency."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def unique_blocks(raw_dir):
    frames = [pd.read_csv(f, usecols=["block", "street_name"]) for f in glob.glob(str(raw_dir / "*.csv"))]
    return pd.concat(frames).drop_duplicates().sort_values(["street_name", "block"]).reset_index(drop=True)


def lookup(session, block, street):
    """Return (lat, lon, postal, address) for the best match, or None."""
    params = {"searchVal": f"{block} {street}", "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}
    for attempt in range(5):
        r = session.get(SEARCH, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * 2**attempt)
            continue
        if r.status_code in (401, 403):
            sys.exit(f"OneMap rejected the token (HTTP {r.status_code}). It may have expired; generate a new one.")
        r.raise_for_status()
        results = r.json().get("results", [])
        # Prefer a result whose block number matches exactly.
        for hit in results:
            if hit.get("BLK_NO", "").upper() == str(block).upper():
                return float(hit["LATITUDE"]), float(hit["LONGITUDE"]), hit.get("POSTAL"), hit.get("ADDRESS")
        return None
    sys.exit("Still rate limited after retries; wait a while and rerun (it resumes).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only process this many new blocks (for testing)")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("ONEMAP_TOKEN")
    if not token:
        sys.exit("ONEMAP_TOKEN is not set. Add it to .env (see .env.example).")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw_dir = ROOT / cfg["paths"]["raw"]
    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_path = out_dir / "block_coords.csv"
    fail_path = out_dir / "block_coords_failed.csv"

    done = set()
    for p in (ok_path, fail_path):
        if p.exists():
            prev = pd.read_csv(p, dtype=str)
            done |= set(zip(prev["block"], prev["street_name"]))

    blocks = unique_blocks(raw_dir)
    todo = [(str(b), s) for b, s in zip(blocks["block"], blocks["street_name"]) if (str(b), s) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(blocks)} unique blocks, {len(done)} already done, {len(todo)} to do")

    session = requests.Session()
    session.headers["Authorization"] = token

    ok_new = ok_path.exists()
    fail_new = fail_path.exists()
    with open(ok_path, "a", newline="", encoding="utf-8") as okf, open(fail_path, "a", newline="", encoding="utf-8") as ff:
        okw, fw = csv.writer(okf), csv.writer(ff)
        if not ok_new:
            okw.writerow(["block", "street_name", "lat", "lon", "postal", "address"])
        if not fail_new:
            fw.writerow(["block", "street_name"])
        found = missed = 0
        for i, (block, street) in enumerate(todo, 1):
            res = lookup(session, block, street)
            if res:
                okw.writerow([block, street, *res])
                found += 1
            else:
                fw.writerow([block, street])
                missed += 1
            if i % 100 == 0:
                okf.flush()
                ff.flush()
                print(f"{i}/{len(todo)}  found={found} missed={missed}")
            time.sleep(args.delay)
    print(f"done: found={found} missed={missed}")


if __name__ == "__main__":
    main()
