"""Match pasted HDB listings to the sales that followed, and measure what public records can't:  python src/listing_outcomes.py

For each listing kept by listings.py (asking price, Kampong's estimate at the time, what the text says), look for a
recorded resale in the same block and street, same flat type and storey range, within 2 sqm of the size, sold in
the 6 months after the listing was pasted. Several people pasting the same flat count once.

Reports (prints only, once there's enough):
  - how far asking prices sat above the final price, overall and by town      (sharper "offer between" advice)
  - how far sale prices landed from Kampong's estimate when the listing said renovated, original condition,
    an unblocked view, a corner unit or a good facing, against listings that didn't      (the condition premium)
A group needs 30 matched sales before its figure means anything; smaller groups are shown as "too few".
Nothing here changes the model yet: once a premium is clear and stable, it can become an adjustment or a feature.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MIN_N = 30


def load_listings():
    c = sqlite3.connect(ROOT / "data" / "oikos.db")
    try:
        d = pd.read_sql("select * from listings where kind = 'hdb'", c)
    except Exception:
        return pd.DataFrame()
    if d.empty:
        return d
    spec = d["spec"].map(json.loads).apply(pd.Series)
    d = pd.concat([d.drop(columns=["spec"]), spec], axis=1)
    d["listed"] = pd.to_datetime(d["at"], unit="s").dt.to_period("M").dt.to_timestamp()
    d["signals"] = d["signals"].map(json.loads)
    # the same flat pasted by several people: keep the first
    return d.sort_values("at").drop_duplicates(["block", "street_name", "flat_type", "storey_range", "floor_area_sqm"])


def match(listings, sales):
    out = []
    for r in listings.itertuples():
        s = sales[(sales["block"] == r.block) & (sales["street_name"] == r.street_name) & (sales["flat_type"] == r.flat_type)
                  & (sales["storey_range"] == r.storey_range) & ((sales["floor_area_sqm"] - r.floor_area_sqm).abs() <= 2)
                  & (sales["sale_date"] >= r.listed) & (sales["sale_date"] <= r.listed + pd.DateOffset(months=6))]
        if len(s):
            first = s.sort_values("sale_date").iloc[0]
            out.append(dict(town=r.town, asking=r.asking, fair=r.fair, sold=float(first["resale_price"]), signals=r.signals))
    return pd.DataFrame(out)


def line(name, x):
    x = pd.Series(x).dropna()
    if len(x) < MIN_N:
        return f"  {name:42} too few ({len(x)} of {MIN_N} needed)"
    return f"  {name:42} median {x.median() * 100:+.1f}%  (middle half {x.quantile(.25) * 100:+.1f}% to {x.quantile(.75) * 100:+.1f}%, n={len(x)})"


def main():
    L = load_listings()
    if L.empty:
        print("No HDB listings kept yet.")
        return
    sales = pd.read_parquet(ROOT / "data/processed/resale_clean.parquet", columns=["block", "street_name", "flat_type", "storey_range", "floor_area_sqm", "sale_date", "resale_price", "town"])
    m = match(L, sales)
    print(f"{len(L)} distinct HDB listings kept, {len(m)} matched to a later sale.\n")
    if m.empty:
        print("None matched yet: sales show up in HDB's data a month or two after they happen.")
        return
    m["ask_over_sold"] = m["asking"] / m["sold"] - 1
    m["sold_vs_estimate"] = m["sold"] / m["fair"] - 1
    print("Asking price against the final price")
    print(line("all", m["ask_over_sold"]))
    for town, g in m.groupby("town"):
        if len(g) >= MIN_N:
            print(line("  " + town.title(), g["ask_over_sold"]))
    print("\nFinal price against Kampong's estimate, by what the listing said")
    print(line("all", m["sold_vs_estimate"]))
    for key in ("renovated", "original", "well_kept", "unblocked", "corner", "facing", "keen_seller"):
        has = m["signals"].map(lambda s: key in s)
        print(line(f"says {key}", m.loc[has, "sold_vs_estimate"]))
        print(line(f"doesn't say {key}", m.loc[~has, "sold_vs_estimate"]))


if __name__ == "__main__":
    sys.exit(main())
