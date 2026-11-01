"""Geocode schools by postal code with OneMap. Reads ONEMAP_TOKEN from .env.

Output: data/processed/schools_coords.csv
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geocode_blocks import ROOT, SEARCH, load_env  # noqa: E402


def main():
    load_env()
    token = os.environ.get("ONEMAP_TOKEN")
    if not token:
        sys.exit("ONEMAP_TOKEN is not set. Add it to .env (see .env.example).")
    s = pd.read_csv(ROOT / "data/raw/amenities/schools.csv", dtype={"postal_code": str})
    s["postal_code"] = s["postal_code"].str.strip().str.zfill(6)
    sess = requests.Session()
    sess.headers["Authorization"] = token
    rows, missed = [], []
    for _, r in s.iterrows():
        resp = sess.get(SEARCH, params={"searchVal": r["postal_code"], "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}, timeout=30)
        if resp.status_code in (401, 403):
            sys.exit(f"OneMap rejected the token (HTTP {resp.status_code}).")
        resp.raise_for_status()
        res = resp.json().get("results", [])
        if res:
            rows.append({"school_name": r["school_name"].strip(), "mainlevel_code": r["mainlevel_code"],
                         "postal_code": r["postal_code"], "lat": float(res[0]["LATITUDE"]), "lon": float(res[0]["LONGITUDE"])})
        else:
            missed.append(r["school_name"])
        time.sleep(0.3)
    out = ROOT / "data/processed/schools_coords.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"matched {len(rows)} of {len(s)} schools -> {out.relative_to(ROOT)}")
    if missed:
        print("no match:", missed)


if __name__ == "__main__":
    main()
