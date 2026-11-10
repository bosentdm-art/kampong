"""Oikos API: the trained models behind a small web service, plus the front end in web/.

Run:  python -m uvicorn api:app --port 8700     then open http://localhost:8700
Needs the data and models built first (python src/update.py).

  GET  /api/options   lists for the flat picker (town > street > block > flat type) and sensible defaults
  POST /api/parse     pasted listing text -> flat details
  POST /api/assess    a flat (and optionally a price) -> every fact the page shows

The page works out verdicts for any price from the returned price quantiles, so dragging the price needs no
new request. Everything returned comes from the models and the sales data; nothing is invented here.
"""
import functools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import advice as adv  # noqa: E402
import explain as ex  # noqa: E402
import forecast as fc  # noqa: E402
import listing_parser as lp  # noqa: E402

app = FastAPI(title="Oikos")


class Flat(BaseModel):
    town: str
    street_name: str
    block: str
    flat_type: str
    flat_model: str
    floor_area_sqm: float
    storey_range: str
    lease_commence_date: int
    price: float | None = None


class Listing(BaseModel):
    text: str


def clean(x):
    """Make numpy / pandas values JSON-safe (NaN becomes null)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m")
    return x


@functools.lru_cache(maxsize=1)
def sales():
    cols = ["town", "street_name", "block", "flat_type", "flat_model", "floor_area_sqm", "storey_range",
            "lease_commence_date", "sale_date", "sale_year", "price_per_sqm"]
    d = pd.read_parquet(ROOT / "data/processed/resale_features.parquet", columns=cols)
    return d[d["sale_date"] >= "2016-01-01"].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def meta():
    m = json.loads((ROOT / "models/lgbm_prod.json").read_text())
    r = json.loads((ROOT / "models/range.json").read_text())
    total = len(pd.read_parquet(ROOT / "data/processed/resale_features.parquet", columns=["sale_year"]))
    return dict(trained_through=m["trained_through"], holdout_median_error=m["holdout_median_error"],
                coverage80=r["validation"]["coverage80"], total_sales=total)


@functools.lru_cache(maxsize=1)
def recent_oos():
    """Out-of-sample estimates for the last six months of sales (from the walk-forward files)."""
    d = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/processed/oos").glob("chunk_*.parquet"))], ignore_index=True)
    r = d[d["sale_date"] > d["sale_date"].max() - pd.DateOffset(months=6)].copy()
    r["gap"] = r["resale_price"] / r["pred"] - 1
    r["town"] = r["town"].astype(str)
    return r


def block_key(b):
    m = re.match(r"\d+", b)
    return (int(m.group()) if m else 0, b)


def pick(wanted, options):
    return wanted if wanted in options else options[0]


@app.get("/api/options")
def options(town: str | None = None, street: str | None = None, block: str | None = None, flat_type: str | None = None):
    """Lists for each level of the picker. A missing or invalid choice falls back to the first option."""
    d = sales()
    d = d[d["sale_date"] >= "2017-01-01"]
    towns = sorted(d["town"].unique())
    town = pick(town, towns)
    dt = d[d["town"] == town]
    streets = sorted(dt["street_name"].unique())
    street = pick(street, streets)
    ds = dt[dt["street_name"] == street]
    blocks = sorted(ds["block"].unique(), key=block_key)
    block = pick(block, blocks)
    db = ds[ds["block"] == block]
    flat_types = sorted(db["flat_type"].unique())
    flat_type = pick(flat_type, flat_types)
    dbt = db[db["flat_type"] == flat_type]
    models = dbt["flat_model"].value_counts().index.tolist()
    return clean(dict(
        towns=towns, streets=streets, blocks=blocks, flat_types=flat_types, flat_models=models,
        storey_ranges=sorted(d["storey_range"].unique()),
        selected=dict(town=town, street=street, block=block, flat_type=flat_type),
        defaults=dict(flat_model=models[0], floor_area_sqm=float(dbt["floor_area_sqm"].median()),
                      storey_range=dbt["storey_range"].mode().iloc[0],
                      lease_commence_date=int(db["lease_commence_date"].mode().iloc[0]))))


@app.post("/api/parse")
def parse(listing: Listing):
    return clean(lp.parse(listing.text[:20000]))


def deals_nearby(town):
    k = recent_oos()
    k = k[k["town"] == town]
    low = k[(k["gap"] <= -0.08) & (k["gap"] >= -0.25)].sort_values(["sale_date", "gap"], ascending=[False, True]).head(4)
    items = [dict(month=x.sale_date, block=x.block, street_name=x.street_name, flat_type=str(x.flat_type), floor_area_sqm=x.floor_area_sqm,
                  storey=f"{int(x.storey_mid - 1):02d}-{int(x.storey_mid + 1):02d}", sold=x.resale_price, estimate=x.pred, gap=x.gap)
             for x in low.itertuples()]
    return dict(town=town, items=items, n_low=int((k["gap"] <= -0.08).sum()), n_all=int(len(k)), months=6)


def trend(town, flat_type):
    d = sales()
    t = d[(d["town"] == town) & (d["flat_type"] == flat_type)].groupby("sale_year")["price_per_sqm"].agg(["median", "size"])
    t = t[t["size"] >= 5]
    return dict(years=t.index.tolist(), values=t["median"].round().tolist(), last_year_through=meta()["trained_through"])


@app.post("/api/assess")
def assess(f: Flat):
    nums = [int(x) for x in re.findall(r"\d+", f.storey_range)]
    if len(nums) < 2:
        raise HTTPException(422, "storey_range must look like '07 TO 09'")
    spec = dict(flat_type=f.flat_type, town=f.town, flat_model=f.flat_model, block=f.block, street_name=f.street_name,
                floor_area_sqm=f.floor_area_sqm, storey_mid=(nums[0] + nums[1]) / 2, lease_commence_date=f.lease_commence_date)
    try:
        a = adv.assess(spec, asking=f.price or None)
    except ValueError as e:
        raise HTTPException(404, str(e))
    base, effects, _ = ex.drivers(spec)
    out = fc.forecast(spec, horizons=(3, 5, 10, 15, 20), today=a["today"])
    d = sales()
    blk = d[(d["block"] == f.block) & (d["street_name"] == f.street_name)]
    tops = [int(r.split(" TO ")[1]) for r in blk["storey_range"].unique() if " TO " in r]
    recent = [dict(month=r.sale_date, storey=r.storey_range.replace(" TO ", "-"), floor_area_sqm=r.floor_area_sqm, price=r.resale_price)
              for r in a["block_recent"].head(8).itertuples()]
    return clean(dict(
        flat=dict(f.model_dump(exclude={"price"}), storey=f"{nums[0]:02d}-{nums[1]:02d}", storey_low=nums[0], storey_high=nums[1],
                  top_storey=max(tops + [nums[1]])),
        price=f.price, fair=a["fair"], lease_now=a["lease_now"], quantiles=a["quantiles"],
        reliability=a["badge"], notes=a["notes"], checks=[c["message"] for c in adv.input_checks(spec)],
        average_flat=base, drivers=[[g, e] for g, e in effects.items()], what_ifs=[[k, v] for k, v in ex.what_if(spec).items()],
        recent=recent, outlook=[{k: v for k, v in r.items()} for r in out["rows"]], growth=math.expm1(out["growth_effective"]),
        deals=deals_nearby(f.town), trend=trend(f.town, f.flat_type), meta=meta()))


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
