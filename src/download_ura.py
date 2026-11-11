"""Download private home sales (condos, apartments, ECs, landed) from the URA Data Service: python src/download_ura.py

Needs URA_ACCESS_KEY in .env. URA issues a fresh token per day from that key; neither is ever printed or saved.
The service returns about five years of sales in four batches (by postal district). Raw JSON goes to
data/raw/ura/, and one tidy row per sale to data/processed/ura_transactions.parquet.

Check URA's terms of use before showing anything built on this data to the public.
"""
import json
import os
import sys
import time

import pandas as pd
import requests

from geocode_blocks import ROOT, load_env

BASES = ["https://eservice.ura.gov.sg/uraDataService", "https://www.ura.gov.sg/uraDataService"]
UA = {"User-Agent": "Mozilla/5.0"}  # URA turns away requests with no browser-like agent
RAW = ROOT / "data/raw/ura"
OUT = ROOT / "data/processed/ura_transactions.parquet"
SALE = {"1": "new sale", "2": "sub sale", "3": "resale"}


def get(url, headers, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, headers={**UA, **headers}, timeout=120)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                return r.json()
            err = f"HTTP {r.status_code}, body starts {r.text[:80]!r}"
        except requests.RequestException as e:
            err = type(e).__name__
        time.sleep(3 * (i + 1))
    raise RuntimeError(err)


def connect(key):
    """(base url, token) from whichever URA address answers."""
    last = None
    for base in BASES:
        for path in ("/insertNewToken/v1", "/insertNewToken.action"):
            try:
                j = get(base + path, {"AccessKey": key}, tries=2)
                if j.get("Status") == "Success" and j.get("Result"):
                    return base, j["Result"]
                last = f"{j.get('Status')}: {str(j.get('Message'))[:120]}"
            except RuntimeError as e:
                last = str(e)
    sys.exit(f"Could not get a URA token ({last}). Check URA_ACCESS_KEY in .env and that the key has been activated by email.")


def download():
    load_env()
    key = os.environ.get("URA_ACCESS_KEY", "").strip()
    if not key:
        sys.exit("URA_ACCESS_KEY is not set in .env")
    base, token = connect(key)
    RAW.mkdir(parents=True, exist_ok=True)
    for batch in range(1, 5):
        j = None
        for path in ("/invokeUraDS/v1", "/invokeUraDS"):
            try:
                j = get(f"{base}{path}?service=PMI_Resi_Transaction&batch={batch}", {"AccessKey": key, "Token": token})
                if j.get("Status") == "Success":
                    break
            except RuntimeError as e:
                print(f"  batch {batch} via {path}: {e}")
        if not j or j.get("Status") != "Success":
            sys.exit(f"batch {batch} failed: {str((j or {}).get('Message'))[:160]}")
        (RAW / f"batch_{batch}.json").write_text(json.dumps(j["Result"]), encoding="utf-8")
        print(f"batch {batch}: {len(j['Result']):,} projects", flush=True)
        time.sleep(1)


def tidy():
    rows = []
    for f in sorted(RAW.glob("batch_*.json")):
        for p in json.loads(f.read_text(encoding="utf-8")):
            for t in p.get("transaction", []):
                rows.append(dict(project=p.get("project"), street=p.get("street"), x=p.get("x"), y=p.get("y"), market_segment=p.get("marketSegment"), **t))
    d = pd.DataFrame(rows)
    d = d.rename(columns={"floorRange": "floor_range", "noOfUnits": "units", "contractDate": "contract", "typeOfSale": "sale_type",
                          "propertyType": "property_type", "typeOfArea": "area_type", "nettPrice": "nett_price"})
    for c in ("area", "price", "units", "x", "y", "nett_price"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["sale_date"] = pd.to_datetime("20" + d["contract"].str[2:] + "-" + d["contract"].str[:2] + "-01", errors="coerce")  # contract is MMYY
    d["sale_type"] = d["sale_type"].astype(str).map(SALE).fillna(d["sale_type"].astype(str))
    d["price_per_sqm"] = d["price"] / d["area"]
    d = d.dropna(subset=["sale_date", "price", "area"]).drop_duplicates().reset_index(drop=True)
    d.to_parquet(OUT, index=False)
    return d


if __name__ == "__main__":
    if "--tidy-only" not in sys.argv:
        download()
    d = tidy()
    print(f"\n{len(d):,} sales in {d['project'].nunique():,} projects, {d['sale_date'].min():%b %Y} to {d['sale_date'].max():%b %Y}")
    print(d.groupby(["property_type", "sale_type"]).size().unstack(fill_value=0).to_string())
    print(f"saved {OUT.relative_to(ROOT)}")
