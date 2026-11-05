"""Quick accuracy experiments on the production holdout (train < 2026-05, test 2026-05..2026-10).

Each variant uses the production recipe (recency weights, monotone lease) unless stated.
Few variants on purpose: trying many against one holdout would overfit to it.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from geocode_blocks import ROOT
from train_model import CATS, NUM
from train_production import HALF_LIFE_MONTHS, PARAMS

HOLD_START = pd.Timestamp("2026-05-01")
HOLD_END = pd.Timestamp("2026-10-01")
BASE = CATS + [c for c in NUM if c != "flat_age_years"] + ["month_idx"]
EXTRA = ["lease_commence_date", "sale_month", "storey_low", "storey_high"]


def run(name, train, test, feats, params, seeds=(0,)):
    t0 = time.time()
    w = 0.5 ** (((HOLD_START - train["sale_date"]).dt.days / 30.44) / HALF_LIFE_MONTHS)
    preds = []
    for s in seeds:
        p = dict(params, random_state=s, monotone_constraints=[1 if c == "remaining_lease_years" else 0 for c in feats])
        m = lgb.LGBMRegressor(**p).fit(train[feats], np.log(train["resale_price"]), sample_weight=w)
        preds.append(m.predict(test[feats]))
    pred = np.exp(np.mean(preds, axis=0))
    ape = (pred - test["resale_price"]).abs() / test["resale_price"]
    print(f"{name:42} median {ape.median():.3%}  mean {ape.mean():.3%}  within 5% {(ape <= .05).mean():.1%}  "
          f"within 10% {(ape <= .10).mean():.1%}  ({time.time() - t0:.0f}s)", flush=True)
    return pred


def main():
    df = pd.read_parquet(ROOT / "data/processed/resale_features.parquet")
    df = df[(df["sale_date"] >= "2017-01-01") & (df["sale_date"] <= HOLD_END)].copy()
    df["month_idx"] = (df["sale_year"] - 2017) * 12 + df["sale_month"] - 1
    for c in CATS:
        df[c] = df[c].astype("category")
    train, test = df[df["sale_date"] < HOLD_START], df[df["sale_date"] >= HOLD_START]
    print(f"train {len(train)}  test {len(test)}")

    run("A  production recipe", train, test, BASE, PARAMS)
    run("B  bigger model (255 leaves, lr .03, 2500)", train, test, BASE,
        dict(PARAMS, num_leaves=255, learning_rate=0.03, n_estimators=2500, min_child_samples=30))
    run("C  smaller leaves (63), 2500 rounds", train, test, BASE, dict(PARAMS, num_leaves=63, n_estimators=2500))
    run("D  extra features", train, test, BASE + EXTRA, PARAMS)
    run("E  5-seed average of A", train, test, BASE, PARAMS, seeds=(0, 1, 2, 3, 4))


if __name__ == "__main__":
    main()
