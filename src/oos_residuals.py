"""Out-of-sample predictions by walking forward in time, for training the range models.

For each 4-month chunk from 2021-01 on: train the production recipe (single seed) on sales BEFORE the
chunk only, predict the chunk. The errors are therefore genuinely out of sample, as they would have been
in real use. Each chunk is saved with the features the range model needs.

Also adds data-richness features that use only EARLIER months: sales count of the same block + flat type
in the prior 24 months, and a recent-comparables estimate (mean price per sqm of the prior 12 months x area).

Output: data/processed/oos/chunk_<start>_to_<end>.parquet (a chunk is recomputed only if its end month changed)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from geocode_blocks import ROOT
from neighbor_features import window_feature
from train_model import CATS
from train_production import FEATS, HALF_LIFE_MONTHS, PARAMS, latest_complete_month

START = pd.Timestamp("2021-01-01")
CHUNK_MONTHS = 4
OUT = ROOT / "data/processed/oos"
KEEP = ["sale_date", "resale_price", "pred", "floor_area_sqm", "storey_mid", "remaining_lease_years", "lease_commence_date",
        "dist_cbd", "dist_mrt", "n_block_24m", "comps_est", "n_block_12m", "flat_type", "town", "flat_model"]


def prepared():
    df = pd.read_parquet(ROOT / "data/processed/resale_features.parquet")
    df["lp"] = np.log(df["price_per_sqm"])
    df["month_n"] = (df["sale_year"] - 1990) * 12 + df["sale_month"] - 1
    key = ["block", "street_name", "flat_type"]
    window_feature(df, key, 24, "b24")
    window_feature(df, key, 12, "b12")
    df["n_block_24m"] = df["b24_n"]
    df["n_block_12m"] = df["b12_n"]
    df["comps_est"] = np.where(df["b12_n"] >= 3, np.exp(df["b12_lp"]) * df["floor_area_sqm"], np.nan)
    df = df[df["sale_date"] >= "2017-01-01"].copy()
    df["month_idx"] = (df["sale_year"] - 2017) * 12 + df["sale_month"] - 1
    for c in CATS:
        df[c] = df[c].astype("category")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = prepared()
    last = latest_complete_month(df)
    df = df[df["sale_date"] <= last]
    s = START
    while s <= last:
        end = s + pd.DateOffset(months=CHUNK_MONTHS)
        to = min(end - pd.DateOffset(months=1), last)
        path = OUT / f"chunk_{s:%Y-%m}_to_{to:%Y-%m}.parquet"
        test = df[(df["sale_date"] >= s) & (df["sale_date"] <= to)]
        if path.exists() or test.empty:
            print(f"skip   {path.name}", flush=True)
        else:
            train = df[df["sale_date"] < s]
            w = 0.5 ** (((s - train["sale_date"]).dt.days / 30.44) / HALF_LIFE_MONTHS)
            m = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(train[FEATS], np.log(train["resale_price"]), sample_weight=w)
            out = test[[c for c in KEEP if c != "pred"]].copy()
            out["pred"] = np.exp(m.predict(test[FEATS]))
            out[KEEP].to_parquet(path, index=False)
            ape = (out["pred"] - out["resale_price"]).abs() / out["resale_price"]
            print(f"saved  {path.name}: {len(out)} sales, median error {ape.median():.2%}", flush=True)
        s = end
    # drop superseded versions of the same chunk start (keep the one reaching the latest month)
    latest = {}
    for f in sorted(OUT.glob("chunk_*.parquet")):
        latest.setdefault(f.name.split("_to_")[0], []).append(f)
    for fs in latest.values():
        for f in sorted(fs)[:-1]:
            f.unlink()


if __name__ == "__main__":
    main()
