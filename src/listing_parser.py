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
    return pd.read_parquet(ROOT / "data/processed/resale_features.parquet", columns=cols)


@functools.lru_cache(maxsize=1)
def block_index():
    """block number -> [(street as people write it, street as HDB writes it, town)], built once."""
    u = blocks()[["block", "street_name", "town"]].astype(str).drop_duplicates()
    out = {}
    for blk, street, town in u.itertuples(index=False):
        out.setdefault(blk, []).append((norm_street(street), street, town))
    return out


def match_block(blk, street):
    """(HDB street name, town) for a block and a street as a person or the AI wrote it, forgiving spelling. None if no match."""
    import difflib
    cands, want = block_index().get(str(blk).upper().replace("BLK", "").strip(), []), norm_street(street)
    if not cands or not want:
        return None
    exact = [h for h in cands if h[0] == want or h[0].startswith(want + " ") or want.startswith(h[0] + " ")]
    if exact:
        return min(exact, key=lambda h: abs(len(h[0]) - len(want)))[1:]
    close = difflib.get_close_matches(want, [h[0] for h in cands], n=1, cutoff=0.75)
    return next(h[1:] for h in cands if h[0] == close[0]) if close else None


def _money(num, unit):
    v = float(num.replace(",", "").replace(" ", ""))
    unit = (unit or "").lower()
    return v * (1e6 if unit.startswith("m") else 1e3 if unit == "k" else 1)


def find_price(t):
    """An asking price from "S$420,000", "$420k", "420k", "1.2m" or "asking 420,000". None if nothing plausible."""
    pats = [r"S?\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m|mil|million)?\b", r"\b(\d{3,4}(?:\.\d+)?)\s*(k)\b", r"\b(\d(?:\.\d+)?)\s*(m|mil|million)\b",
            r"(?:asking|price|priced|listed|list|for|at|offer)\D{0,12}(\d{3}[,\s]?\d{3})\b()",
            r"\b(\d{3},\d{3}|\d,\d{3},\d{3})\b()"]
    for pat in pats:
        for m in re.finditer(pat, t, re.I):
            v = _money(m.group(1), m.group(2))
            if 100_000 <= v <= 3_000_000:
                return int(round(v))
    return None


def parse(text):
    """Works on a pasted listing page and on a typed line like "blk 85 whampoa drive, 3-room, 76 sqm, 8th floor, asking 420k"."""
    t = " ".join(text.split())
    out = dict(block=None, street_name=None, town=None, flat_type=None, flat_model=None, floor_area_sqm=None,
               asking=find_price(t), lease_commence_date=None, top_year=None, storey_hint=None, floor=None, role=None, age=None)

    m = re.search(r"([\d,]{3,5})\s*sq\s*ft", t, re.I)
    if m:
        out["floor_area_sqm"] = round(int(m.group(1).replace(",", "")) / SQFT_PER_SQM, 1)
    else:
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:sqm|sq m|m2|square met)", t, re.I)
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
    if out["flat_type"] is None:
        m = re.search(r"\b([1-5])[\s-]*(?:room|rm)\b", t, re.I)
        b = re.search(r"\b([1-4])[\s-]*(?:bed|bedroom|br)s?\b", t, re.I)
        if m:
            out["flat_type"] = f"{m.group(1)} ROOM"
        elif re.search(r"\b(executive|exec)\b", t, re.I):
            out["flat_type"] = "EXECUTIVE"
        elif b:  # an HDB "4-room" flat has 3 bedrooms
            out["flat_type"] = f"{int(b.group(1)) + 1} ROOM"
    m = re.search(r"\b(high|mid|middle|low)\s+(?:floor|storey|level)", t, re.I)
    if m:
        out["storey_hint"] = m.group(1).lower().replace("middle", "mid")
    m = (re.search(r"#(\d{1,2})-\d+", t) or re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:floor|storey|level)\b", t, re.I)
         or re.search(r"\b(?:floor|storey|level)\s*(\d{1,2})\b", t, re.I))
    if m and 1 <= int(m.group(1)) <= 60:
        out["floor"] = int(m.group(1))
    if re.search(r"\b(sell|selling|seller|my flat|my unit|my home|i own|list(?:ing)? (?:it|mine|my))\b", t, re.I):
        out["role"] = "seller"
    elif re.search(r"\b(buy|buying|buyer|offer|looking at|viewing|viewed)\b", t, re.I):
        out["role"] = "buyer"
    m = re.search(r"\b(?:i'?m|i am|aged?)\s+(\d{2})\b|\b(\d{2})\s*(?:years? old|yo)\b", t, re.I)
    if m and 21 <= int(m.group(1) or m.group(2)) <= 90:
        out["age"] = int(m.group(1) or m.group(2))

    # block + street: try every "<number><letter?> <words>" in the text against the HDB data, longest street match first
    idx, d = block_index(), blocks()
    for cand in re.finditer(r"(?=\b(\d{1,4}[A-Za-z]?)\s+([A-Za-z][A-Za-z']+(?:\s+[A-Za-z0-9][A-Za-z0-9']*){0,4}))", text):
        blk, street = cand.group(1).upper(), norm_street(cand.group(2))
        words, hits = street.split(), []
        for n in range(len(words), 0, -1):  # "85 whampoa worth" still finds Whampoa Drive: shorten until something fits
            pre = " ".join(words[:n])
            hits = [h for h in idx.get(blk, []) if h[0] == pre or (len(pre) >= 4 and h[0].startswith(pre))]
            if hits:
                break
        if not hits:
            continue
        _, name, town = min(hits, key=lambda h: len(h[0]))
        hit = d[(d["block"] == blk) & (d["street_name"] == name)]
        out["block"], out["street_name"], out["town"] = blk, name, town
        out["lease_commence_date"] = int(hit["lease_commence_date"].mode().iloc[0])
        if out["flat_type"] is None and hit["flat_type"].nunique() == 1:
            out["flat_type"] = hit["flat_type"].iloc[0]
        break
    if out["lease_commence_date"] is None and out["top_year"]:
        out["lease_commence_date"] = out["top_year"]
    return out


if __name__ == "__main__":
    print(parse(open(sys.argv[1], encoding="utf-8").read()))
