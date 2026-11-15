"""More from the URA Data Service, beyond sales: python src/download_ura_more.py   (--tidy-only to rebuild the tables from saved JSON)

  developer sales   PMI_Resi_Developer_Sales, monthly, last 36 months: units in each new project, launched and sold to date
  rentals           PMI_Resi_Rental, quarterly, about five years: each private rental contract (size band, bedrooms, rent)
  rental medians    PMI_Resi_Rental_Median: median rent per sq ft by project and quarter
  pipeline          PMI_Resi_Pipeline: projects not yet completed, units and expected completion year (latest quarter only)

Needs URA_ACCESS_KEY in .env (the same key as download_ura.py). Raw JSON goes to data/raw/ura_more/, tables to data/processed/ura_*.parquet.
Months and quarters already saved are not fetched again, except the most recent few, which URA revises.
"""
import json
import os
import sys
import time

import pandas as pd

from download_ura import connect, get
from geocode_blocks import ROOT, load_env

RAW = ROOT / "data/raw/ura_more"
OUT = ROOT / "data/processed"
MONTHS, QUARTERS, REFETCH = 36, 22, 3


def call(base, h, query):
    for path in ("/invokeUraDS/v1", "/invokeUraDS"):
        try:
            j = get(f"{base}{path}?service={query}", h, tries=3)
            if j.get("Status") == "Success":
                return j.get("Result") or []
        except RuntimeError as e:
            print(f"  {query} via {path}: {e}")
    return None


def periods():
    """Months (MMYY) and quarters (YYqN) to ask for, newest first."""
    now = pd.Timestamp.today().normalize().replace(day=1)
    months = [(now - pd.DateOffset(months=i)).strftime("%m%y") for i in range(1, MONTHS + 1)]
    quarters = sorted({f"{(now - pd.DateOffset(months=3 * i)).strftime('%y')}q{(now - pd.DateOffset(months=3 * i)).quarter}" for i in range(QUARTERS)}, reverse=True)
    return months, quarters


def download():
    load_env()
    key = os.environ.get("URA_ACCESS_KEY", "").strip()
    if not key:
        sys.exit("URA_ACCESS_KEY is not set in .env")
    base, token = connect(key)
    h = {"AccessKey": key, "Token": token}
    for d in ("dev", "rental"):
        (RAW / d).mkdir(parents=True, exist_ok=True)
    months, quarters = periods()
    for kind, items, query in (("dev", months, "PMI_Resi_Developer_Sales&refPeriod={}"), ("rental", quarters, "PMI_Resi_Rental&refPeriod={}")):
        got = 0
        for i, p in enumerate(items):
            f = RAW / kind / f"{p}.json"
            if f.exists() and i >= REFETCH:
                continue
            r = call(base, h, query.format(p))
            if r is None:
                print(f"  {kind} {p}: failed, kept what was saved")
                continue
            if r or not f.exists():  # an empty answer for the newest period means URA has not published it yet
                f.write_text(json.dumps(r), encoding="utf-8")
            got += 1
            time.sleep(0.5)
        print(f"{kind}: fetched {got} periods", flush=True)
    for name, query in (("rental_median", "PMI_Resi_Rental_Median"), ("pipeline", "PMI_Resi_Pipeline")):
        r = call(base, h, query)
        if r:
            (RAW / f"{name}.json").write_text(json.dumps(r), encoding="utf-8")
        print(f"{name}: {len(r or [])} projects", flush=True)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def band_mid(s):
    """'90-100' -> 95; '>3000' or 'NA' -> NaN."""
    p = s.astype(str).str.extract(r"^(\d+)\s*-\s*(\d+)$").astype(float)
    return (p[0] + p[1]) / 2


def tidy():
    out = {}
    rows = []
    for f in sorted((RAW / "dev").glob("*.json")):
        for p in json.loads(f.read_text(encoding="utf-8")):
            for s in p.get("developerSales", []):
                rows.append(dict(project=p.get("project"), street=p.get("street"), district=p.get("district"), developer=p.get("developer"),
                                 property_type=p.get("propertyType"), market_segment=p.get("marketSegment"), **s))
    if rows:
        d = pd.DataFrame(rows).rename(columns={"refPeriod": "period", "medianPrice": "median_psf", "highestPrice": "high_psf", "lowestPrice": "low_psf",
                                                "launchedToDate": "launched_to_date", "soldInMonth": "sold_in_month", "launchedInMonth": "launched_in_month",
                                                "soldToDate": "sold_to_date", "unitsAvail": "units"})
        d["month"] = pd.to_datetime("20" + d["period"].str[2:] + "-" + d["period"].str[:2] + "-01", errors="coerce")
        for c in ("median_psf", "high_psf", "low_psf", "launched_to_date", "sold_in_month", "launched_in_month", "sold_to_date", "units"):
            d[c] = num(d[c])
        d = d.drop_duplicates(["project", "month"], keep="last").sort_values(["project", "month"]).reset_index(drop=True)
        d.to_parquet(OUT / "ura_dev_sales.parquet", index=False)
        out["developer sales"] = f"{len(d):,} project-months, {d['project'].nunique():,} projects, {d['month'].min():%b %Y} to {d['month'].max():%b %Y}"

    rows = []
    for f in sorted((RAW / "rental").glob("*.json")):
        for p in json.loads(f.read_text(encoding="utf-8")):
            for r in p.get("rental", []):
                rows.append(dict(project=p.get("project"), street=p.get("street"), x=p.get("x"), y=p.get("y"), **r))
    if rows:
        d = pd.DataFrame(rows).rename(columns={"areaSqm": "area_band", "areaSqft": "sqft_band", "leaseDate": "lease", "propertyType": "property_type",
                                                "noOfBedRoom": "bedrooms"})
        d["month"] = pd.to_datetime("20" + d["lease"].str[2:] + "-" + d["lease"].str[:2] + "-01", errors="coerce")
        d["area"] = band_mid(d["area_band"])
        d["rent"], d["x"], d["y"] = num(d["rent"]), num(d["x"]), num(d["y"])
        d["bedrooms"] = num(d["bedrooms"].replace("NA", None))
        d = d.dropna(subset=["month", "rent"]).drop_duplicates().reset_index(drop=True)
        d.to_parquet(OUT / "ura_rentals.parquet", index=False)
        out["rentals"] = f"{len(d):,} contracts in {d['project'].nunique():,} projects, {d['month'].min():%b %Y} to {d['month'].max():%b %Y}"

    f = RAW / "rental_median.json"
    if f.exists():
        rows = [dict(project=p.get("project"), street=p.get("street"), **m) for p in json.loads(f.read_text(encoding="utf-8")) for m in p.get("rentalMedian", [])]
        d = pd.DataFrame(rows).rename(columns={"refPeriod": "quarter", "median": "median_psf", "psf25": "psf25", "psf75": "psf75"})
        d.to_parquet(OUT / "ura_rental_median.parquet", index=False)
        out["rental medians"] = f"{len(d):,} project-quarters, {d['project'].nunique():,} projects, {d['quarter'].min()} to {d['quarter'].max()}"

    f = RAW / "pipeline.json"
    if f.exists():
        d = pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))).rename(columns={"expectedTOPYear": "top_year", "developerName": "developer", "totalUnits": "units",
                                                                                     "noOfCondo": "condo", "noOfApartment": "apartment", "noOfTerrace": "terrace",
                                                                                     "noOfSemiDetached": "semi_detached", "noOfDetachedHouse": "detached"})
        for c in ("units", "condo", "apartment", "terrace", "semi_detached", "detached"):
            d[c] = num(d[c])
        d.to_parquet(OUT / "ura_pipeline.parquet", index=False)
        out["pipeline"] = f"{len(d):,} projects, {int(d['units'].sum()):,} units, completing {d['top_year'].min()} to {d['top_year'].max()}"
    return out


if __name__ == "__main__":
    if "--tidy-only" not in sys.argv:
        download()
    for k, v in tidy().items():
        print(f"{k}: {v}")
