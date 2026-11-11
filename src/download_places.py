"""Download everyday places and future MRT stations from OpenStreetMap: python src/download_places.py

One request each to the public Overpass service, for Singapore only:
  places           malls, supermarkets, named parks, clinics            -> data/raw/amenities/osm_places.json
  future stations  rail stations mapped as under construction/proposed  -> data/raw/amenities/osm_future_stations.json

OpenStreetMap data is (c) OpenStreetMap contributors, under the Open Database Licence: the app must say so.
Volunteers map it, so it can be incomplete or out of date. Future stations are cross-checked against
reference/future_mrt.csv (line and expected opening year, from LTA announcements) in nearby.py.
"""
import json
import sys
import time

import requests

from geocode_blocks import ROOT

URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
AREA = 'area["ISO3166-1"="SG"][admin_level=2]->.sg;'
QUERIES = {
    "osm_places": AREA + '(nwr["shop"="mall"](area.sg);nwr["shop"="supermarket"](area.sg);nwr["leisure"="park"]["name"](area.sg);'
                         'nwr["amenity"~"^(clinic|doctors)$"](area.sg);nwr["healthcare"="clinic"](area.sg););out center tags;',
    "osm_future_stations": AREA + '(nwr["railway"~"^(construction|proposed)$"]["construction"~"station|halt"](area.sg);nwr["construction:railway"~"station|halt"](area.sg);'
                                  'nwr["proposed:railway"~"station|halt"](area.sg);nwr["railway"="construction"]["station"](area.sg);'
                                  'nwr["railway"="station"]["opening_date"](area.sg);nwr["railway"="station"]["construction"](area.sg););out center tags;',
}


def fetch(query):
    last = None
    for url in URLS * 2:
        try:
            r = requests.post(url, data={"data": "[out:json][timeout:120];" + query}, timeout=180, headers={"User-Agent": "oikos-hdb-research"})
            if r.status_code == 200:
                return r.json()["elements"]
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(8)
    sys.exit(f"Overpass did not answer ({last})")


if __name__ == "__main__":
    out = ROOT / "data/raw/amenities"
    out.mkdir(parents=True, exist_ok=True)
    for name, q in QUERIES.items():
        rows = []
        for e in fetch(q):
            lat, lon = (e.get("lat"), e.get("lon")) if "lat" in e else ((e.get("center") or {}).get("lat"), (e.get("center") or {}).get("lon"))
            if lat is not None:
                rows.append(dict(lat=lat, lon=lon, tags=e.get("tags", {})))
        (out / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
        print(f"{name}: {len(rows):,} places")
        time.sleep(3)
