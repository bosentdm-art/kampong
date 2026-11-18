"""HDB's property information for every block, and bus stops, for the model experiments and the 3D map.

    python src/download_hdb_property.py

data/processed/hdb_property.parquet   one row per HDB block: block, street, max_floor_lvl, year_completed, total_dwelling_units
                                      (data.gov.sg "HDB Property Information", Singapore Open Data Licence)
data/raw/amenities/osm_bus_stops.json bus stops from OpenStreetMap (ODbL: credit "© OpenStreetMap contributors")
"""
import json

import pandas as pd
import requests

from geocode_blocks import ROOT

DATASET = "d_17f5382f26140b1fdae0ba2ef6239d2f"
PAGED = "https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit=5000&offset={off}"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def hdb_property():
    rows, off = [], 0
    while True:
        r = requests.get(PAGED.format(id=DATASET, off=off), timeout=60).json()["result"]
        rows += r["records"]
        off += len(r["records"])
        if not r["records"] or off >= r["total"]:
            break
    d = pd.DataFrame(rows)
    keep = ["blk_no", "street", "max_floor_lvl", "year_completed", "residential", "total_dwelling_units", "bldg_contract_town"]
    d = d[keep].rename(columns={"blk_no": "block", "street": "street_name", "bldg_contract_town": "town_code"})
    for c in ("max_floor_lvl", "year_completed", "total_dwelling_units"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["block"], d["street_name"] = d["block"].str.upper().str.strip(), d["street_name"].str.upper().str.strip()
    out = ROOT / "data/processed/hdb_property.parquet"
    d.to_parquet(out, index=False)
    print(f"{len(d):,} blocks -> {out}")


def bus_stops():
    q = "[out:json][timeout:120];node[\"highway\"=\"bus_stop\"](1.15,103.59,1.48,104.10);out;"
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q}, timeout=180, headers={"User-Agent": "Kampong (kampong.com.de) open-data download"})
            if r.ok:
                el = r.json()["elements"]
                stops = [dict(name=e.get("tags", {}).get("name", "Bus stop"), code=e.get("tags", {}).get("asset_ref") or e.get("tags", {}).get("ref"),
                              lat=e["lat"], lon=e["lon"]) for e in el]
                out = ROOT / "data/raw/amenities/osm_bus_stops.json"
                out.write_text(json.dumps(stops), encoding="utf-8")
                print(f"{len(stops):,} bus stops -> {out}")
                return
        except requests.RequestException as e:
            print("overpass failed:", url, e)
    print("bus stops: every Overpass server failed")


if __name__ == "__main__":
    hdb_property()
    bus_stops()
