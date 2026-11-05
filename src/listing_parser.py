"""Turn the text of a property listing (copied and pasted by a person) into flat details.

No fetching: the user pastes the text. Extracts block, street, asking price, size, flat type and model,
and TOP year, then matches the block + street to our HDB data (listings write "Whampoa Drive", HDB data
"WHAMPOA DR"). Anything it cannot find is left as None for the user to fill in.

Usage: python src/listing_parser.py <file with pasted listing text>
"""
import functools
import re
import sys

import pandas as pd

from geocode_blocks import ROOT

SQFT_PER_SQM = 10.7639
ABBREV = {"AVE": "AVENUE", "DR": "DRIVE", "RD": "ROAD", "ST": "STREET", "CRES": "CRESCENT", "CL": "CLOSE", "PK": "PARK",
          "PL": "PLACE", "TER": "TERRACE", "CTRL": "CENTRAL", "NTH": "NORTH", "STH": "SOUTH", "UPP": "UPPER", "LOR": "LORONG",
          "BT": "BUKIT", "JLN": "JALAN", "KG": "KAMPONG", "TG": "TANJONG", "C'WEALTH": "COMMONWEALTH", "GDNS": "GARDENS",
          "HTS": "HEIGHTS", "MKT": "MARKET", "SQ": "SQUARE", "CTR": "CENTRE", "LK": "LINK", "VW": "VIEW"}
MODELS = ["IMPROVED", "NEW GENERATION", "MODEL A", "MODEL A2", "SIMPLIFIED", "STANDARD", "APARTMENT", "MAISONETTE",
          "PREMIUM APARTMENT", "PREMIUM APARTMENT LOFT", "PREMIUM MAISONETTE", "DBSS", "TYPE S1", "TYPE S2", "ADJOINED FLAT",
          "TERRACE", "MULTI GENERATION", "3GEN", "MODEL A-MAISONETTE", "IMPROVED-MAISONETTE"]


def norm_street(s):
    words = re.sub(r"[^A-Z0-9' ]", " ", s.upper()).split()
    return " ".join(ABBREV.get(w, w) for w in words)


@functools.lru_cache(maxsize=1)
def blocks():
    cols = ["block", "street_name", "town", "flat_type", "flat_model", "lease_commence_date", "floor_area_sqm"]
    d = pd.read_parquet(ROOT / "data/processed/resale_features.parquet", columns=cols)
    d["street_norm"] = d["street_name"].map(norm_street)
    return d


def parse(text):
    t = " ".join(text.split())
    out = dict(block=None, street_name=None, town=None, flat_type=None, flat_model=None, floor_area_sqm=None,
               asking=None, lease_commence_date=None, top_year=None, storey_hint=None)

    m = re.search(r"S\$\s*([\d,]{5,})", t)
    if m:
        out["asking"] = int(m.group(1).replace(",", ""))
    m = re.search(r"([\d,]{3,5})\s*sq\s*ft", t, re.I)
    if m:
        out["floor_area_sqm"] = round(int(m.group(1).replace(",", "")) / SQFT_PER_SQM, 1)
    else:
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:sqm|sq m|m2)", t, re.I)
        if m:
            out["floor_area_sqm"] = float(m.group(1))
    m = re.search(r"TOP in (\d{4})", t, re.I)
    if m:
        out["top_year"] = int(m.group(1))
    m = re.search(r"\b(\d|E)[A-Z0-9]{0,3}\s*\(([A-Za-z0-9 \-]+)\)\s*HDB", t)
    if m:
        out["flat_type"] = "EXECUTIVE" if m.group(1) == "E" else f"{m.group(1)} ROOM"
        model = m.group(2).upper().strip()
        out["flat_model"] = model if model in MODELS else None
    m = re.search(r"\b(high|mid|middle|low)\s+floor", t, re.I)
    if m:
        out["storey_hint"] = m.group(1).lower().replace("middle", "mid")

    # block + street: try each "<number><letter?> <words>" candidate against the HDB data
    d = blocks()
    for cand in re.finditer(r"\b(?:Blk\s*)?(\d{1,4}[A-Z]?)\s+([A-Z][A-Za-z']+(?:\s+[A-Z0-9][A-Za-z0-9']*){0,4})", text):
        blk, street = cand.group(1).upper(), norm_street(cand.group(2))
        hit = d[(d["block"] == blk) & (d["street_norm"].apply(lambda s: street.startswith(s) or s.startswith(street)))]
        if not hit.empty:
            out["block"], out["street_name"], out["town"] = blk, hit["street_name"].iloc[0], hit["town"].iloc[0]
            out["lease_commence_date"] = int(hit["lease_commence_date"].mode().iloc[0])
            if out["flat_type"] is None and hit["flat_type"].nunique() == 1:
                out["flat_type"] = hit["flat_type"].iloc[0]
            break
    if out["lease_commence_date"] is None and out["top_year"]:
        out["lease_commence_date"] = out["top_year"]
    return out


if __name__ == "__main__":
    print(parse(open(sys.argv[1], encoding="utf-8").read()))
