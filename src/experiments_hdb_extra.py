"""Do three new facts make HDB prices more accurate?  python src/experiments_hdb_extra.py

  storey   the floor relative to the block's top floor, and the block's height (HDB Property Information)
  supply   new flats completed within 1 km in the five years up to the sale (a glut nearby can hold prices down)
  mrt      the nearest planned MRT station not yet open at the sale, and how many years until it opens

Same recipe as train_production.py (three seeds, recency weights). Tested on two six-month periods, each trained only on
sales before it, so a gain has to show up twice to count. Prints only.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import nearby
import train_production as tp
from geocode_blocks import ROOT


def xy(lat, lon):
    return np.c_[(np.asarray(lon) - 103.8) * 111_320 * np.cos(np.radians(1.35)), (np.asarray(lat) - 1.35) * 110_574]


def add_features(df):
    prop = pd.read_parquet(ROOT / "data/processed/hdb_property.parquet")
    prop = prop.drop_duplicates(["block", "street_name"])
    df = df.merge(prop[["block", "street_name", "max_floor_lvl"]], on=["block", "street_name"], how="left")
    df["rel_storey"] = df["storey_mid"] / df["max_floor_lvl"]

    # supply: units completed per block-year, summed within 1 km of each sale for the five years up to the sale year
    coords = pd.read_csv(ROOT / "data/processed/block_coords.csv").drop_duplicates(["block", "street_name"])
    built = prop[prop["residential"] == "Y"].merge(coords, on=["block", "street_name"])
    built = built.dropna(subset=["year_completed", "total_dwelling_units", "lat"])
    tree = cKDTree(xy(built["lat"], built["lon"]))
    keys = df[["lat", "lon"]].drop_duplicates().dropna()
    near = tree.query_ball_point(xy(keys["lat"], keys["lon"]), r=1000)
    yrs, units = built["year_completed"].to_numpy().astype(int), built["total_dwelling_units"].to_numpy()
    lo, hi = 2010, 2030
    table = np.zeros((len(keys), hi - lo + 1))
    for i, idx in enumerate(near):
        if idx:
            y = np.clip(yrs[idx], lo, hi) - lo
            np.add.at(table[i], y, units[idx])
    cum = np.cumsum(table, axis=1)
    kidx = pd.Series(range(len(keys)), index=pd.MultiIndex.from_frame(keys))
    row = kidx.reindex(pd.MultiIndex.from_frame(df[["lat", "lon"]])).to_numpy()
    ok = ~np.isnan(row)
    sy = np.clip(df["sale_year"].to_numpy(), lo + 5, hi) - lo
    supply = np.full(len(df), np.nan)
    r = row[ok].astype(int)
    supply[ok] = cum[r, sy[ok]] - cum[r, sy[ok] - 5]
    df["supply_1km_5y"] = supply

    # planned MRT: nearest listed station opening after the sale year
    fut = nearby.tables()["future"]
    fxy = xy(fut["lat"], fut["lon"])
    sxy = xy(df["lat"].fillna(1.35), df["lon"].fillna(103.8))
    d = np.sqrt(((sxy[:, None, :] - fxy[None, :, :]) ** 2).sum(-1))  # sales x stations
    open_after = fut["year"].to_numpy()[None, :] > df["sale_year"].to_numpy()[:, None]
    d = np.where(open_after, d, np.inf)
    j = d.argmin(1)
    best = d[np.arange(len(df)), j]
    df["dist_planned_mrt"] = np.where(np.isfinite(best), np.minimum(best, 5000), 5000)
    df["years_to_planned_mrt"] = np.where(best <= 2000, fut["year"].to_numpy()[j] - df["sale_year"], 15)
    return df


GROUPS = {"storey": ["rel_storey", "max_floor_lvl"], "supply": ["supply_1km_5y"],
          "mrt": ["dist_planned_mrt", "years_to_planned_mrt"]}


def run(name, feats, tr, te, ref):
    tp.FEATS = feats
    tp.PARAMS["monotone_constraints"] = [1 if c == "remaining_lease_years" else 0 for c in feats]
    pred = tp.predict(tp.fit(tr, ref), te)
    a = (pred - te["resale_price"]).abs() / te["resale_price"]
    print(f"  {name:22} median {a.median():6.2%}  mean {a.mean():6.2%}  within 5% {(a <= .05).mean():5.1%}", flush=True)


def main():
    df = pd.read_parquet(ROOT / "data/processed/resale_features.parquet")
    df = df[df["sale_date"] >= "2017-01-01"].copy()
    df["month_idx"] = (df["sale_year"] - 2017) * 12 + df["sale_month"] - 1
    df = add_features(df)
    for c in tp.CATS:
        df[c] = df[c].astype("category")
    print(f"matched to block height: {df['max_floor_lvl'].notna().mean():.1%}; supply known: {df['supply_1km_5y'].notna().mean():.1%}")
    print(df[["rel_storey", "supply_1km_5y", "dist_planned_mrt", "years_to_planned_mrt"]].describe().round(2).to_string())
    last = tp.latest_complete_month(df)
    df = df[df["sale_date"] <= last]
    base = list(tp.FEATS)
    configs = {"baseline": base, **{f"+ {k}": base + v for k, v in GROUPS.items()}, "+ all three": base + sum(GROUPS.values(), [])}
    for back in (0, 6):
        end = last - pd.DateOffset(months=back)
        first = end - pd.DateOffset(months=5)
        tr, te = df[df["sale_date"] < first], df[(df["sale_date"] >= first) & (df["sale_date"] <= end)]
        print(f"\nperiod {first:%b %Y} to {end:%b %Y}: train {len(tr):,}, test {len(te):,}", flush=True)
        for name, feats in configs.items():
            run(name, feats, tr, te, first)


if __name__ == "__main__":
    main()
