"""Add location features to the cleaned sales table.

Inputs : data/processed/resale_clean.parquet, block_coords.csv, schools_coords.csv,
         data/raw/amenities/{mrt_exits,hawker_centres}.geojson
Output : data/processed/resale_features.parquet

Per-block features (distances in metres, straight-line):
  dist_mrt, dist_hawker, dist_primary_school, dist_cbd,
  n_primary_1km, n_primary_2km, n_mrt_exits_500m
Sales whose block could not be geocoded are dropped.
"""
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geocode_blocks import ROOT

EARTH_M = 6_371_000
CBD = (1.2840, 103.8515)  # Raffles Place


def tree(lat, lon):
    return BallTree(np.radians(np.c_[lat, lon]), metric="haversine")


def nearest_m(t, pts):
    d, _ = t.query(np.radians(pts), k=1)
    return d[:, 0] * EARTH_M


def count_within(t, pts, radius_m):
    return t.query_radius(np.radians(pts), r=radius_m / EARTH_M, count_only=True)


def points(path, keep=None):
    feats = json.load(open(path, encoding="utf-8"))["features"]
    if keep:
        feats = [f for f in feats if keep(f["properties"])]
    return np.array([[f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]] for f in feats])


def main():
    proc = ROOT / "data/processed"
    sales = pd.read_parquet(proc / "resale_clean.parquet")
    blocks = pd.read_csv(proc / "block_coords.csv", dtype={"block": str})
    schools = pd.read_csv(proc / "schools_coords.csv")

    mrt = points(ROOT / "data/raw/amenities/mrt_exits.geojson")
    hawker = points(ROOT / "data/raw/amenities/hawker_centres.geojson",
                    keep=lambda p: str(p.get("STATUS", "")).startswith("Existing"))
    primary = schools[schools["mainlevel_code"].str.contains("PRIMARY", na=False)][["lat", "lon"]].to_numpy()
    print(f"{len(mrt)} MRT exits, {len(hawker)} hawker centres, {len(primary)} primary schools")

    pts = blocks[["lat", "lon"]].to_numpy()
    t_mrt, t_hawker, t_pri = tree(*mrt.T), tree(*hawker.T), tree(*primary.T)
    blocks["dist_mrt"] = nearest_m(t_mrt, pts)
    blocks["n_mrt_exits_500m"] = count_within(t_mrt, pts, 500)
    blocks["dist_hawker"] = nearest_m(t_hawker, pts)
    blocks["dist_primary_school"] = nearest_m(t_pri, pts)
    blocks["n_primary_1km"] = count_within(t_pri, pts, 1000)
    blocks["n_primary_2km"] = count_within(t_pri, pts, 2000)
    blocks["dist_cbd"] = nearest_m(tree(*np.array([CBD]).T), pts)

    keep = ["block", "street_name", "lat", "lon", "dist_mrt", "n_mrt_exits_500m", "dist_hawker",
            "dist_primary_school", "n_primary_1km", "n_primary_2km", "dist_cbd"]
    out = sales.merge(blocks[keep], on=["block", "street_name"], how="inner")
    path = proc / "resale_features.parquet"
    out.to_parquet(path, index=False)
    print(f"sales {len(sales)} -> with location {len(out)} ({len(sales) - len(out)} dropped) -> {path.relative_to(ROOT)}")
    print(out[["dist_mrt", "dist_hawker", "dist_primary_school", "dist_cbd", "n_primary_1km", "n_primary_2km"]].describe().round(0).to_string())


if __name__ == "__main__":
    main()
