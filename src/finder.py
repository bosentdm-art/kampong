"""Find homes that fit what a person asks for: "4 room under 600k near an MRT in the east".

Oikos has no live listings (we do not scrape the portals), so the answer is where such homes have actually sold:
HDB blocks, or condo projects, with sales in the last 12 months that meet the requirements, what they went for, and
how often a sale there came in within the budget. Each result is one tap from a full check of a flat in that block.

  parse(text, asking, prev)  -> (requirements, the ones this message set). Rules only; `prev` lets "only tampines"
                                or "make it 650k" refine the last search.
  intent(text)               -> "strong" (find, recommend, shortlist...), "weak" (budget, looking for...) or None
  search(spec)               -> the matching blocks or projects, a summary by town or district, and the words for it

Requirements read: budget (or a range), flat types or bedrooms, towns or a region, distance to an MRT (or to a named
station), a primary school within 1 km, lease left, age, size, high floor, and for condos district, segment and freehold.
"""
import functools
import re

import numpy as np
import pandas as pd

import condo as cc
import condo_features as cf
import forecast as fc
import listing_parser as lp
import nearby as nb

REGIONS = {"east": ["BEDOK", "PASIR RIS", "TAMPINES"], "north-east": ["ANG MO KIO", "HOUGANG", "PUNGGOL", "SENGKANG", "SERANGOON"],
           "north": ["SEMBAWANG", "WOODLANDS", "YISHUN"], "west": ["BUKIT BATOK", "BUKIT PANJANG", "CHOA CHU KANG", "CLEMENTI", "JURONG EAST", "JURONG WEST"],
           "central": ["BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL AREA", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "QUEENSTOWN", "TOA PAYOH"]}
# postal districts by region, and the districts an HDB town roughly sits in, for condo searches
REGION_DISTRICTS = {"east": [14, 15, 16, 17, 18], "north-east": [19, 20, 28], "north": [25, 26, 27], "west": [5, 21, 22, 23, 24], "central": list(range(1, 14))}
TOWN_DISTRICTS = {"TAMPINES": [18], "PASIR RIS": [18], "BEDOK": [16], "MARINE PARADE": [15], "GEYLANG": [14], "HOUGANG": [19], "PUNGGOL": [19], "SENGKANG": [19], "SERANGOON": [19],
                  "ANG MO KIO": [20], "BISHAN": [20], "TOA PAYOH": [12], "KALLANG/WHAMPOA": [12], "QUEENSTOWN": [3], "BUKIT MERAH": [3, 4], "CLEMENTI": [5], "JURONG EAST": [22],
                  "JURONG WEST": [22], "BUKIT BATOK": [23], "CHOA CHU KANG": [23], "BUKIT PANJANG": [23], "WOODLANDS": [25], "YISHUN": [27], "SEMBAWANG": [27], "CENTRAL AREA": [1, 2, 6, 7],
                  "BUKIT TIMAH": [10, 21]}
BED_SQM = {1: (0, 58), 2: (55, 85), 3: (85, 125), 4: (120, 170), 5: (160, 1000)}  # rough: URA records give the size, not the bedrooms
NEAR_MRT, NEAR_STATION, HIGH_FLOOR = 500, 800, 10
EMPTY = dict(kind="hdb", budget=None, budget_min=None, flat_types=[], beds=None, towns=[], region=None, districts=[], segment=None, mrt_m=None, station=None, school=None,
             min_lease=None, max_age=None, min_sqm=None, high_floor=False, freehold=False, sort=None)

STRONG = re.compile(r"\b(find|recommend|suggest|shortlist|show me|list of|search for|searching for)\b", re.I)
WEAK = re.compile(r"\b(looking for|look for|budget|where can|where should|what can (?:i|we) (?:get|buy)|which (?:areas?|towns?|estates?|blocks?|flats?|condos?|projects?)|options|"
                  r"want (?:a|an|to buy|something)|need (?:a|an)|any (?:flats?|units?|condos?)|house ?hunting|hunting for)\b", re.I)
_UNIT = r"(k|m|mil|million)"


def intent(text):
    return "strong" if STRONG.search(text[:600]) else "weak" if WEAK.search(text[:600]) else None


@functools.lru_cache(maxsize=1)
def stations():
    """{station name in lower case: (lat, lon)}: the middle of its exits."""
    m = nb.tables()["mrt"]
    g = m.assign(key=m["name"].map(lambda s: nb.station_name(s).lower())).groupby("key")[["lat", "lon"]].mean()
    return {k: (float(r.lat), float(r.lon)) for k, r in g.iterrows() if len(k) >= 3}


@functools.lru_cache(maxsize=1)
def schools():
    """{school name without "primary school", lower case: (full name, lat, lon)}."""
    out = {}
    for name, lat, lon in nb.tables()["primary"][["name", "lat", "lon"]].itertuples(index=False):
        key = re.sub(r"\s+(primary\s+)?school.*$", "", str(name).lower()).strip()
        if len(key) >= 5:
            out[key] = (str(name).title(), float(lat), float(lon))
    return out


def parse(text, asking=None, prev=None):
    """(requirements, the keys this message set). Starts from `prev` (the last search) when given."""
    t = " " + " ".join(text.lower().replace("–", "-").split()) + " "
    new = {}
    condo = bool(re.search(r"\b(condos?|condominiums?|private (?:property|home|homes|apartment)|apartments?|executive condo\w*|ec)\b", t))
    if condo:
        new["kind"] = "condo"

    m = re.search(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*" + _UNIT + r"?\s*(?:-|to)\s*\$?\s?(\d[\d,]*(?:\.\d+)?)\s*" + _UNIT + r"\b", t)
    if m:
        lo, hi = lp._money(m.group(1), m.group(2) or m.group(4)), lp._money(m.group(3), m.group(4))
        if 100_000 <= lo < hi <= 30_000_000:
            new["budget_min"], new["budget"] = int(lo), int(hi)
    if "budget" not in new and asking:
        new["budget"], new["budget_min"] = int(asking), None

    types = set()
    for a, b in re.findall(r"\b([1-5])\s*(?:or|/|,|to|-|and)\s*([1-5])[\s-]*(?:rooms?|rms?|rm)\b", t):
        types |= {f"{a} ROOM", f"{b} ROOM"}
    types |= {f"{a} ROOM" for a in re.findall(r"\b([1-5])[\s-]*(?:rooms?|rms?|rm)\b", t)}
    if not condo and (re.search(r"\b(executive|exec)\b", t) or re.search(r"\b(EA|EM)\b", text)):
        types.add("EXECUTIVE")
    beds = re.search(r"\b([1-5])[\s-]*(?:bed|bedroom|bedder|br)s?\b", t)
    if beds and condo:
        new["beds"] = int(beds.group(1))
    elif beds and not types:
        types.add(f"{min(5, int(beds.group(1)) + 1)} ROOM")  # an HDB "4-room" flat has 3 bedrooms
    if types and not condo:
        new["flat_types"], new["kind"] = sorted(types), "hdb"
    elif re.search(r"\b(hdb|resale flats?|flats?)\b", t) and not condo:
        new["kind"] = "hdb"

    # distance to the MRT, and a named station ("near bishan mrt")
    for name in sorted(stations(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\s+(?:mrt|lrt|station|interchange)\b", t):
            new["station"], t = name, re.sub(rf"\b{re.escape(name)}\s+(?:mrt|lrt|station|interchange)\b", " station ", t)
            break
    walk = re.search(r"\b(\d{1,2})\s*[- ]?min(?:ute)?s?\b[^.?!]{0,25}\b(?:mrt|lrt|train|station)\b", t) or re.search(r"\b(?:mrt|lrt|train|station)\b[^.?!]{0,25}\b(\d{1,2})\s*[- ]?min(?:ute)?s?\b", t)
    dist = re.search(r"\b(\d{3,4})\s*m(?:etres?|eters?)?\b[^.?!]{0,20}\b(?:mrt|lrt|train|station)\b", t)
    if dist:
        new["mrt_m"] = int(dist.group(1))
    elif walk and 2 <= int(walk.group(1)) <= 30:
        new["mrt_m"] = int(round(int(walk.group(1)) * 80 / 1.3, -1))  # the same pace nearby.walk_minutes uses
    elif re.search(r"\b(near|close to|next to|beside|walk(?:ing)?(?: distance)? to|by|around)\b[^.?!]{0,15}\b(mrt|lrt|train|station)\b", t) or re.search(r"\b(mrt|lrt)\b[^.?!]{0,12}\b(nearby|near|close)\b", t):
        new["mrt_m"] = NEAR_STATION if new.get("station") else NEAR_MRT
    if new.get("station") and "mrt_m" not in new:
        new["mrt_m"] = NEAR_STATION

    if re.search(r"\b(school|primary|pri sch|p1)\b", t):
        for key, (full, _, _) in sorted(schools().items(), key=lambda kv: -len(kv[0])):
            if f" {key} " in t:
                new["school"], t = key, t.replace(f" {key} ", " ")
                break

    up = " " + re.sub(r"[^A-Z/ ]", " ", t.upper()) + " "
    up = " ".join(up.split()).join("  ")
    towns = [n for n in lp.places()[0] if f" {n} " in up or (n == "KALLANG/WHAMPOA" and " KALLANG WHAMPOA " in up)]
    towns += [n for a, n in lp.TOWN_ALIASES.items() if a not in ("CENTRAL", "CITY") and f" {a} " in up and n not in towns]
    rest = up
    for n in towns:
        rest = rest.replace(f" {n} ", "  ")
    reg = re.search(r"\b(NORTH ?EAST|NORTHEAST|EAST|WEST|NORTH|CENTRAL)\b", rest)
    if towns:
        new["towns"], new["region"] = towns, None
    elif reg and not re.search(r"\b(EAST|WEST|NORTH|SOUTH)\s*[- ]?FACING\b", rest):
        new["region"], new["towns"] = {"NORTH EAST": "north-east", "NORTHEAST": "north-east"}.get(reg.group(1), reg.group(1).lower()), []

    d = re.findall(r"\b(?:d|district)\s?(\d{1,2})\b", t)
    if d and all(1 <= int(x) <= 28 for x in d):
        new["districts"], new["kind"] = sorted({int(x) for x in d}), "condo"
    seg = re.search(r"\b(ccr|rcr|ocr|core central|city fringe|suburb\w*|outside central)\b", t)
    if seg:
        new["segment"] = {"core central": "CCR", "city fringe": "RCR", "outside central": "OCR"}.get(seg.group(1), "OCR" if seg.group(1).startswith("suburb") else seg.group(1).upper())

    if "lease" in t:
        m = (re.search(r"(?:at least|min(?:imum)?|more than|over|above|>)\s*(\d{2})\s*\+?\s*(?:years?|yrs?)?", t) or re.search(r"\b(\d{2})\s*\+?\s*(?:years?|yrs?)\s*(?:of |left |remaining )*lease", t)
             or re.search(r"lease\s*(?:of|>|above|at least|more than|over|left)?\s*(\d{2})\b", t))
        if m and 30 <= int(m.group(1)) <= 99:
            new["min_lease"] = int(m.group(1))
    if re.search(r"\b(new(?:er|ish)?|recent(?:ly)? (?:built|mop)|just mop|young(?:er)?)\s+(?:flat|block|hdb|unit|condo|one|ones|development)s?\b", t) or re.search(r"\bnot (?:too |so )?old\b", t):
        new["max_age"] = 12
    m = re.search(r"(?:at least|min(?:imum)?|more than|over|above|>|bigger than)\s*(\d{2,3})\s*(?:sqm|sq m|square met)", t) or re.search(r"\b(\d{2,3})\s*(?:sqm|sq m)\s*(?:and above|or more|or bigger|\+|minimum)", t)
    if m and 30 <= int(m.group(1)) <= 400:
        new["min_sqm"] = float(m.group(1))
    if re.search(r"\bhigh(?:er)?\s*(?:floor|storey|level)", t):
        new["high_floor"] = True
    if re.search(r"\bfreehold\b", t):
        new["freehold"], new["kind"] = True, "condo"
    for word, key in (("cheapest", "price"), ("lowest price", "price"), ("closest to", "mrt"), ("nearest to", "mrt"), ("longest lease", "lease"), ("newest", "lease"), ("biggest", "size"), ("largest", "size")):
        if word in t:
            new["sort"] = key

    spec = {**EMPTY, **{k: v for k, v in (prev or {}).items() if k in EMPTY}, **new}
    if spec["kind"] == "condo":
        spec["flat_types"] = []
    else:
        spec.update(beds=None, districts=[], segment=None, freehold=False)
    return spec, sorted(new)


def from_ai(a, spec):
    """Requirements the AI read from a message the rules could not, checked field by field and laid over `spec`."""
    out, a = dict(spec), a if isinstance(a, dict) else {}
    num = lambda x, lo, hi: x if isinstance(x, (int, float)) and not isinstance(x, bool) and lo <= x <= hi else None
    if a.get("condo") is True:
        out["kind"] = "condo"
    if num(a.get("budget"), 100_000, 30_000_000) and not out["budget"]:
        out["budget"] = int(a["budget"])
    types = [t for t in (a.get("flat_types") or []) if t in ("1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE")] if isinstance(a.get("flat_types"), list) else []
    if types and not out["flat_types"] and out["kind"] == "hdb":
        out["flat_types"] = sorted(set(types))
    towns = [str(t).upper() for t in (a.get("towns") or []) if str(t).upper() in lp.places()[0]] if isinstance(a.get("towns"), list) else []
    if towns and not out["towns"] and not out["region"]:
        out["towns"] = towns
    if a.get("region") in REGIONS and not out["towns"] and not out["region"]:
        out["region"] = a["region"]
    if a.get("near_mrt") is True and not out["mrt_m"]:
        out["mrt_m"] = NEAR_MRT
    if num(a.get("min_lease_years"), 30, 99) and not out["min_lease"]:
        out["min_lease"] = int(a["min_lease_years"])
    if num(a.get("min_sqm"), 30, 400) and not out["min_sqm"]:
        out["min_sqm"] = float(a["min_sqm"])
    if num(a.get("bedrooms"), 1, 5) and out["kind"] == "condo" and not out["beds"]:
        out["beds"] = int(a["bedrooms"])
    return out


def kfmt(v):
    return f"${v / 1e6:.2f}m" if v >= 1e6 else f"${round(v / 1000)}k"


def describe(spec):
    """The requirements in a sentence fragment: "4 Room flats at or under $600k, within 500 m of an MRT station, in Bedok, Pasir Ris and Tampines"."""
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    if spec["kind"] == "condo":
        what = (f"{spec['beds']}-bedroom condos (about {BED_SQM[spec['beds']][0] or 35} to {min(BED_SQM[spec['beds']][1], 250)} sqm)" if spec["beds"] else "condos") + (", freehold" if spec["freehold"] else "")
    else:
        what = (join([t.title() for t in spec["flat_types"]]) + " flats") if spec["flat_types"] else "HDB flats"
    parts = [what]
    if spec["budget"]:
        parts.append(f"between {kfmt(spec['budget_min'])} and {kfmt(spec['budget'])}" if spec["budget_min"] else f"at or under {kfmt(spec['budget'])}")
    if spec["station"]:
        parts.append(f"within {spec['mrt_m']} m of {spec['station'].title()} station")
    elif spec["mrt_m"]:
        parts.append(f"within {spec['mrt_m']} m of an MRT or LRT station (about {nb.walk_minutes(spec['mrt_m'])} min walk)")
    if spec["school"]:
        parts.append(f"within 1 km of {schools()[spec['school']][0]}")
    if spec["min_lease"]:
        parts.append(f"with at least {spec['min_lease']} years of lease left")
    if spec["max_age"]:
        parts.append(f"no more than {spec['max_age']} years old")
    if spec["min_sqm"]:
        parts.append(f"of {round(spec['min_sqm'])} sqm or more")
    if spec["high_floor"]:
        parts.append("on a high floor")
    where = (join([t.title() for t in spec["towns"]]) if spec["towns"] else f"the {spec['region']} ({join([t.title() for t in REGIONS[spec['region']]])})" if spec["region"] and spec["kind"] == "hdb"
             else f"the {spec['region']}" if spec["region"] else None)
    if spec["kind"] == "condo" and (spec["districts"] or spec["segment"]):
        where = join([f"District {d}" for d in spec["districts"]]) if spec["districts"] else {"CCR": "the core central region", "RCR": "the city fringe", "OCR": "the suburbs"}[spec["segment"]]
    head = parts[0] + (" " + parts[1] if spec["budget"] else "")
    return ", ".join([head] + parts[2 if spec["budget"] else 1:]) + (f", in {where}" if where else "")


@functools.lru_cache(maxsize=1)
def hdb_sales():
    """The last 12 months of HDB resales, each with its block's distance to the MRT and its position."""
    cols = ["town", "street_name", "block", "flat_type", "floor_area_sqm", "storey_range", "lease_commence_date", "sale_date", "resale_price"]
    d = pd.read_parquet(fc.PROC / "resale_features.parquet", columns=cols)
    d = d[d["sale_date"] > d["sale_date"].max() - pd.DateOffset(months=12)].copy()
    for c in ("town", "street_name", "block", "flat_type"):
        d[c] = d[c].astype(str)
    d["storey_low"] = d["storey_range"].str.slice(0, 2).astype(int)
    d["lease_left"] = d["lease_commence_date"] + 99 - d["sale_date"].dt.year
    d["age"] = d["sale_date"].dt.year - d["lease_commence_date"]
    b = fc.block_table()[["block", "street_name", "lat", "lon", "dist_mrt"]].astype({"block": str, "street_name": str})
    return d.merge(b, on=["block", "street_name"], how="left")


@functools.lru_cache(maxsize=1)
def condo_sales():
    """The last 12 months of private resales, each with its project's district, segment, tenure and position."""
    s = cc.sales()
    s = s[(s["sale_type"] != "new sale") & (s["sale_date"] > s["sale_date"].max() - pd.DateOffset(months=12))].copy()
    p = cc.projects().reset_index()
    ll = cf.svy21_to_latlon(p["x"].fillna(0).to_numpy(), p["y"].fillna(0).to_numpy())
    p["lat"], p["lon"] = [a for a, _ in ll], [b for _, b in ll]
    p.loc[p["x"].isna(), ["lat", "lon"]] = np.nan
    p["district"], p["freehold"] = p["district"].astype(int), p["tenure"].astype(str).str.contains("Freehold|999", case=False)
    s["project"] = s["project"].astype(str)
    s["floor_low"] = pd.to_numeric(s["floor_range"].astype(str).str.slice(0, 2), errors="coerce")
    return s.merge(p[["project", "street", "district", "market_segment", "tenure", "freehold", "dist_mrt", "lat", "lon"]].astype({"project": str}), on="project", how="inner")


def _near(d, spec):
    """Keep the sales whose block or project meets the MRT and school requirements."""
    if spec["station"] and spec["station"] in stations():
        la, lo = stations()[spec["station"]]
        d = d[nb.metres(la, lo, d["lat"].to_numpy(), d["lon"].to_numpy()) <= (spec["mrt_m"] or NEAR_STATION)]
    elif spec["mrt_m"]:
        d = d[d["dist_mrt"] <= spec["mrt_m"]]
    if spec["school"] and spec["school"] in schools():
        _, la, lo = schools()[spec["school"]]
        d = d[nb.metres(la, lo, d["lat"].to_numpy(), d["lon"].to_numpy()) <= 1000]
    return d


def _rank(g, spec, cap_key, over=False):
    """Order the blocks or projects, then keep a spread: no more than two from one street and four from one town or district."""
    order = "price" if over else spec["sort"] or ("mrt" if spec["mrt_m"] else "lease" if spec["min_lease"] or spec["max_age"] else "size" if spec["min_sqm"] else "evidence")
    if order == "mrt":
        g = g.sort_values(["dist_mrt", "n_in"], ascending=[True, False])
    elif order == "lease":
        g = g.sort_values(["lease_left", "n_in"], ascending=[False, False])
    elif order == "size":
        g = g.sort_values(["sqm", "n_in"], ascending=[False, False])
    elif order == "price":
        g = g.sort_values(["median", "n_in"], ascending=[True, False])
    else:  # where the budget most reliably buys: several sales within it, and most sales there within it
        g = g.assign(solid=g["n_in"] >= 2, share=g["n_in"] / g["n_all"]).sort_values(["solid", "share", "n_in", "latest"], ascending=False)
    seen, kept = {}, []
    for r in g.itertuples():
        a, b = ("s", r.street), ("t", getattr(r, cap_key))
        if seen.get(a, 0) >= 2 or seen.get(b, 0) >= 4:
            continue
        seen[a], seen[b] = seen.get(a, 0) + 1, seen.get(b, 0) + 1
        kept.append(r)
        if len(kept) == 8:
            break
    return kept, order


ORDERS = {"evidence": "where the budget most reliably buys", "mrt": "closest to the MRT first", "lease": "longest lease first", "size": "biggest first", "price": "cheapest first"}


def search(spec):
    """The blocks (or condo projects) where homes meeting `spec` sold in the last 12 months."""
    return _condos(spec) if spec["kind"] == "condo" else _hdb(spec)


def _within(m, spec):
    w = m if not spec["budget"] else m[m["price"] <= spec["budget"]]
    return w if not spec["budget_min"] else w[w["price"] >= spec["budget_min"]]


def _hdb(spec):
    d = hdb_sales().rename(columns={"resale_price": "price"})
    towns = spec["towns"] or (REGIONS[spec["region"]] if spec["region"] else None)
    if towns:
        d = d[d["town"].isin(towns)]
    if spec["flat_types"]:
        d = d[d["flat_type"].isin(spec["flat_types"])]
    if spec["min_sqm"]:
        d = d[d["floor_area_sqm"] >= spec["min_sqm"]]
    if spec["min_lease"]:
        d = d[d["lease_left"] >= spec["min_lease"]]
    if spec["max_age"]:
        d = d[d["age"] <= spec["max_age"]]
    if spec["high_floor"]:
        d = d[d["storey_low"] >= HIGH_FLOOR]
    m = _near(d, spec)
    w, key = _within(m, spec), ["block", "street_name", "town", "flat_type"]
    out = dict(kind="hdb", spec=spec, what=describe(spec), months=12, over=False, sales=int(len(w)), all_sales=int(len(m)), items=[], by_place=[])
    if w.empty and m.empty:
        return out
    if w.empty:  # nothing within the budget: show what came closest, and say so
        out["over"], w = True, m.nsmallest(40, "price")
    n_all = m.groupby(key).size().rename("n_all")
    g = w.groupby(key).agg(n_in=("price", "size"), median=("price", "median"), low=("price", "min"), high=("price", "max"), sqm=("floor_area_sqm", "median"), lease_left=("lease_left", "median"),
                           dist_mrt=("dist_mrt", "first"), latest=("sale_date", "max")).join(n_all).reset_index().rename(columns={"street_name": "street"})
    out["blocks"] = int(len(g))
    kept, order = _rank(g, spec, "town", out["over"])
    for r in kept:
        near = nb.around_block(r.block, r.street)
        mrt = near["mrt"][0] if near and near.get("mrt") else None
        out["items"].append(dict(name=f"Blk {r.block} {r.street.title()}", block=r.block, street_name=r.street, town=r.town, flat_type=r.flat_type, n_in=int(r.n_in), n_all=int(r.n_all),
                                 median=float(r.median), low=float(r.low), high=float(r.high), sqm=float(r.sqm), lease_left=int(round(r.lease_left)), latest=r.latest.strftime("%Y-%m"),
                                 mrt=f"{mrt['name']} {mrt['rail']}" if mrt else None, mrt_m=int(mrt["metres"]) if mrt else None, walk_min=int(mrt["walk_min"]) if mrt else None,
                                 send=f"Blk {r.block} {r.street.title()}, {r.flat_type.lower()}"))
    t = w.groupby("town").agg(sales=("price", "size"), median=("price", "median")).sort_values("sales", ascending=False).head(6)
    out["by_place"] = [dict(place=k.title(), sales=int(r.sales), median=float(r["median"])) for k, r in t.iterrows()]
    out["order"], out["order_words"] = order, ORDERS[order]
    return out


def _condos(spec):
    d = condo_sales()
    districts = spec["districts"] or (REGION_DISTRICTS[spec["region"]] if spec["region"] else sorted({x for t in spec["towns"] for x in TOWN_DISTRICTS.get(t, [])}) or None)
    if districts:
        d = d[d["district"].isin(districts)]
    if spec["segment"]:
        d = d[d["market_segment"] == spec["segment"]]
    if spec["beds"]:
        lo, hi = BED_SQM[spec["beds"]]
        d = d[(d["area"] >= lo) & (d["area"] <= hi)]
    if spec["min_sqm"]:
        d = d[d["area"] >= spec["min_sqm"]]
    if spec["freehold"]:
        d = d[d["freehold"]]
    if spec["high_floor"]:
        d = d[d["floor_low"] >= HIGH_FLOOR + 1]
    m = _near(d, spec)
    w = _within(m, spec)
    out = dict(kind="condo", spec=spec, what=describe(spec), months=12, over=False, sales=int(len(w)), all_sales=int(len(m)), items=[], by_place=[],
               districts=districts if not spec["districts"] and districts else None)
    if w.empty and m.empty:
        return out
    if w.empty:
        out["over"], w = True, m.nsmallest(40, "price")
    n_all = m.groupby("project").size().rename("n_all")
    g = w.groupby("project").agg(n_in=("price", "size"), median=("price", "median"), low=("price", "min"), high=("price", "max"), sqm=("area", "median"), dist_mrt=("dist_mrt", "first"),
                                 latest=("sale_date", "max"), street=("street", "first"), district=("district", "first"), tenure=("tenure", "first"), freehold=("freehold", "first"),
                                 lat=("lat", "first"), lon=("lon", "first")).join(n_all).reset_index()
    g["lease_left"] = np.where(g["freehold"], 999, 60)  # only for ordering: freehold first when sorting by lease
    out["blocks"] = int(len(g))
    kept, order = _rank(g, spec, "district", out["over"])
    for r in kept:
        near = nb.around(float(r.lat), float(r.lon)) if r.lat == r.lat else None
        mrt = near["mrt"][0] if near and near.get("mrt") else None
        out["items"].append(dict(name=r.project.title(), project=r.project, street=str(r.street).title(), district=int(r.district), tenure=str(r.tenure), n_in=int(r.n_in), n_all=int(r.n_all),
                                 median=float(r.median), low=float(r.low), high=float(r.high), sqm=float(r.sqm), latest=r.latest.strftime("%Y-%m"),
                                 mrt=f"{mrt['name']} {mrt['rail']}" if mrt else None, mrt_m=int(mrt["metres"]) if mrt else None, walk_min=int(mrt["walk_min"]) if mrt else None,
                                 send=f"{r.project.title()} {round(r.sqm)} sqm"))
    t = w.groupby("district").agg(sales=("price", "size"), median=("price", "median")).sort_values("sales", ascending=False).head(6)
    out["by_place"] = [dict(place=f"District {k}", sales=int(r.sales), median=float(r["median"])) for k, r in t.iterrows()]
    out["order"], out["order_words"] = order, ORDERS[order]
    return out
