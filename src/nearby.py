"""What is around a block: MRT (open and planned), primary schools, hawker centres, supermarkets, malls, parks, clinics,
and the public-transport commute to a place the person names.

Sources: MRT exits, hawker centres and schools from data.gov.sg (data/raw/amenities, data/processed/schools_coords.csv);
malls, supermarkets, parks, clinics and the position of planned stations from OpenStreetMap (download_places.py);
line and expected opening year of planned stations from reference/future_mrt.csv; commute times from OneMap routing.

Distances are straight lines. Walking minutes are an estimate (1.3 x the straight line at 80 m a minute).
School distances are a guide only: P1 registration uses SLA's own measure, on OneMap's School Query.
"""
import csv
import functools
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from geocode_blocks import ROOT, load_env

AMEN = ROOT / "data/raw/amenities"
KINDS = {"mall": "mall", "supermarket": "supermarket", "park": "park", "clinic": "clinic", "doctors": "clinic"}


def metres(lat, lon, lats, lons):
    """Straight-line distance from one point to many, in metres."""
    p = np.pi / 180
    a = np.sin((lats - lat) * p / 2) ** 2 + np.cos(lat * p) * np.cos(lats * p) * np.sin((lons - lon) * p / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def walk_minutes(m):
    return int(max(1, round(m * 1.3 / 80)))


def station_name(s):
    return re.sub(r"\s+(MRT|LRT)\s+STATION$", "", s.strip(), flags=re.I).title()


@functools.lru_cache(maxsize=1)
def tables():
    """Every list of places as (names, lats, lons) arrays, loaded once."""
    exits = json.loads((AMEN / "mrt_exits.geojson").read_text(encoding="utf-8"))["features"]
    mrt = pd.DataFrame([(station_name(f["properties"]["STATION_NA"]), "LRT" if " LRT " in f["properties"]["STATION_NA"].upper() else "MRT",
                         f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in exits], columns=["name", "rail", "lat", "lon"])
    hawk = json.loads((AMEN / "hawker_centres.geojson").read_text(encoding="utf-8"))["features"]
    hawker = pd.DataFrame([(f["properties"].get("NAME") or "Hawker centre", f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in hawk], columns=["name", "lat", "lon"])
    sch = pd.read_csv(ROOT / "data/processed/schools_coords.csv").dropna(subset=["lat", "lon"])
    primary = sch[sch["mainlevel_code"].str.upper().str.contains("PRIMARY|MIXED", na=False)].assign(name=lambda d: d["school_name"].str.title())[["name", "lat", "lon"]]
    places = {}
    f = AMEN / "osm_places.json"
    if f.exists():
        rows = [(KINDS.get(t["tags"].get("shop") or t["tags"].get("leisure") or t["tags"].get("amenity") or t["tags"].get("healthcare")), t["tags"].get("name"), t["lat"], t["lon"])
                for t in json.loads(f.read_text(encoding="utf-8"))]
        p = pd.DataFrame([r for r in rows if r[0] and r[1]], columns=["kind", "name", "lat", "lon"])
        places = {k: g.reset_index(drop=True) for k, g in p.groupby("kind")}

    # planned stations: position from OpenStreetMap (new stations) or from today's exits (a new line at an open station)
    osm = {}
    f = AMEN / "osm_future_stations.json"
    if f.exists():
        for t in json.loads(f.read_text(encoding="utf-8")):
            if t["tags"].get("name"):
                osm[t["tags"]["name"].lower()] = (t["lat"], t["lon"])
    here = mrt.groupby("name")[["lat", "lon"]].mean()
    future, missing = [], []
    with open(ROOT / "reference/future_mrt.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pos = osm.get(r["name"].lower()) if r["kind"] == "new station" else (tuple(here.loc[r["name"]]) if r["name"] in here.index else None)
            if r["kind"] == "new station" and r["name"] in here.index:
                continue  # already open in our exits file
            (future.append(dict(r, year=int(r["year"]), lat=pos[0], lon=pos[1])) if pos else missing.append(r["name"]))
    return dict(mrt=mrt, hawker=hawker, primary=primary, places=places, future=pd.DataFrame(future), future_missing=missing)


@functools.lru_cache(maxsize=1)
def block_coords():
    d = pd.read_csv(ROOT / "data/processed/block_coords.csv", dtype={"block": str}).dropna(subset=["lat", "lon"])
    return {(b, s): (la, lo) for b, s, la, lo in d[["block", "street_name", "lat", "lon"]].itertuples(index=False)}


def nearest(df, lat, lon, n=1, within=None):
    if df is None or not len(df):
        return []
    d = metres(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
    order = np.argsort(d)
    out = [dict(name=df["name"].iloc[i], metres=int(round(d[i], -1))) for i in order if within is None or d[i] <= within]
    return out[:n] if n else out


@functools.lru_cache(maxsize=512)
def around(lat, lon):
    """Everything near a point. Distances are rounded to 10 m."""
    t = tables()
    m = t["mrt"]
    d = metres(lat, lon, m["lat"].to_numpy(), m["lon"].to_numpy())
    by_station = m.assign(d=d).sort_values("d").drop_duplicates("name")  # nearest exit of each station
    pick = pd.concat([by_station.head(2), by_station[by_station["rail"] == "MRT"].head(1)]).drop_duplicates("name")  # always include the nearest MRT, not only LRT
    stations = [dict(name=r.name, rail=r.rail, metres=int(round(r.d, -1)), walk_min=walk_minutes(r.d)) for r in pick.itertuples() if r.d <= 1500 or r.Index == pick.index[0] or r.rail == "MRT"]
    future = []
    if len(t["future"]):
        f = t["future"].assign(d=metres(lat, lon, t["future"]["lat"].to_numpy(), t["future"]["lon"].to_numpy())).sort_values("d")
        future = [dict(name=r.name, line=r.line, year=int(r.year), kind=r.kind, metres=int(round(r.d, -1)), walk_min=walk_minutes(r.d)) for r in f[f["d"] <= 1200].head(3).itertuples()]
    schools = nearest(t["primary"], lat, lon, n=0, within=2000)
    out = dict(mrt=stations, future_mrt=future,
               schools_1km=[s["name"] for s in schools if s["metres"] <= 1000], schools_2km=[s["name"] for s in schools if s["metres"] > 1000],
               hawker=(nearest(t["hawker"], lat, lon) or [None])[0])
    for kind in ("supermarket", "mall", "park", "clinic"):
        df = t["places"].get(kind)
        out[kind] = (nearest(df, lat, lon) or [None])[0]
        if kind in ("supermarket", "clinic") and df is not None:
            out[kind + "s_500m"] = int((metres(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy()) <= 500).sum())
    return out


def around_block(block, street):
    pos = block_coords().get((block, street))
    return dict(around(*pos), lat=pos[0], lon=pos[1]) if pos else None


# ---- commute: public transport, arriving for a weekday morning, from OneMap
class Unavailable(Exception):
    pass


_token = dict(value=None, at=0.0)
_routes = {}


def onemap_token(force=False):
    """The OneMap token: from .env, or a fresh one if ONEMAP_EMAIL and ONEMAP_PASSWORD are set there (tokens last three days)."""
    load_env()
    if force or not _token["value"]:
        email, pw = os.environ.get("ONEMAP_EMAIL", "").strip(), os.environ.get("ONEMAP_PASSWORD", "").strip()
        if email and pw:
            r = requests.post("https://www.onemap.gov.sg/api/auth/post/getToken", json={"email": email, "password": pw}, timeout=30)
            _token.update(value=(r.json().get("access_token") if r.status_code == 200 else None), at=time.time())
        elif not force:
            _token.update(value=os.environ.get("ONEMAP_TOKEN", "").strip() or None, at=time.time())
        else:
            _token["value"] = None
    return _token["value"]


_places = {}


def _search(q):
    """OneMap's place search, signed in when we can be (anonymous use is tightly rate-limited)."""
    tok = onemap_token()
    r = requests.get("https://www.onemap.gov.sg/api/common/elastic/search", timeout=30, headers={"Authorization": tok} if tok else {},
                     params=dict(searchVal=q[:80], returnGeom="Y", getAddrDetails="Y", pageNum=1))
    if r.status_code == 401 and tok:
        tok = onemap_token(force=True)
        r = requests.get("https://www.onemap.gov.sg/api/common/elastic/search", timeout=30, headers={"Authorization": tok} if tok else {},
                         params=dict(searchVal=q[:80], returnGeom="Y", getAddrDetails="Y", pageNum=1))
    if r.status_code == 429:
        raise Unavailable("The map service is busy. Try again in a minute.")
    return (r.json().get("results") or []) if r.status_code == 200 else []


def find_place(text):
    """(label, lat, lon) for a workplace or landmark as typed, from OneMap's search. None if nothing matches."""
    q = " ".join(text.upper().split())
    if q in _places:
        return _places[q]
    res = _search(text)
    if not res:
        return None
    x = next((r for r in res if r.get("SEARCHVAL", "").upper() == q), None) or next((r for r in res if r.get("SEARCHVAL", "").upper() in (q + " MRT STATION", q + " LRT STATION")), None)
    if x is None and not any(ch.isdigit() for ch in q):  # "one-north" means the station, not the condo that borrows its name
        x = next((r for r in _search(text[:70] + " MRT STATION") if r.get("SEARCHVAL", "").upper().startswith(q)), None)
    x = x or res[0]
    label = x.get("BUILDING") if x.get("BUILDING") and x["BUILDING"] != "NIL" else x.get("ADDRESS") or text
    _places[q] = (re.sub(r"\s*\([A-Z]{2}\d+[^)]*\)", "", label.title(), flags=re.I).replace(" Mrt ", " MRT ").replace(" Lrt ", " LRT "), float(x["LATITUDE"]), float(x["LONGITUDE"]))
    return _places[q]


def next_weekday():
    t = time.gmtime(time.time() + 8 * 3600 + 86400)
    while t.tm_wday >= 5:
        t = time.gmtime(time.mktime(t) + 86400)
    return time.strftime("%m-%d-%Y", t)


def commute(lat, lon, place):
    """Public transport from a block to `place`, leaving 8am on the next weekday: minutes, transfers and the legs."""
    hit = find_place(place)
    if not hit:
        raise ValueError(f"I couldn't find \"{place}\" on the map. Try a building name, an MRT station or a postal code.")
    label, plat, plon = hit
    key = (round(lat, 5), round(lon, 5), round(plat, 5), round(plon, 5))
    if key not in _routes:
        params = dict(start=f"{lat},{lon}", end=f"{plat},{plon}", routeType="pt", date=next_weekday(), time="08:00:00", mode="TRANSIT", maxWalkDistance="1000", numItineraries="1")
        for attempt in (0, 1):
            tok = onemap_token(force=bool(attempt))
            if not tok:
                raise Unavailable("Commute times need a OneMap sign-in, and the one on file has expired.")
            r = requests.get("https://www.onemap.gov.sg/api/public/routingsvc/route", params=params, headers={"Authorization": tok}, timeout=45)
            if r.status_code != 401:
                break
        if r.status_code == 401:
            raise Unavailable("Commute times need a OneMap sign-in, and the one on file has expired.")
        its = ((r.json().get("plan") or {}).get("itineraries") or []) if r.status_code == 200 else []
        if not its:
            raise ValueError(f"I couldn't find a public transport route to {label}.")
        it = its[0]
        legs = [("Walk" if g["mode"] == "WALK" else (g.get("route") or g["mode"].title()) + (" line" if g["mode"] in ("SUBWAY", "RAIL", "TRAM") else " bus" if g["mode"] == "BUS" else ""))
                for g in it["legs"] if g["mode"] != "WALK" or g.get("duration", 0) >= 180]
        _routes[key] = dict(minutes=int(round(it["duration"] / 60)), transfers=int(it.get("transfers", 0)), legs=legs)
    return dict(place=label, **_routes[key])


if __name__ == "__main__":
    t = tables()
    print("planned stations placed:", len(t["future"]), "| not placed:", t["future_missing"])
    print(json.dumps(around_block("85", "WHAMPOA DR"), indent=1)[:1500])
