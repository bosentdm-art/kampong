"""Step 4: LightGBM price model, same time split as the baseline.

  fit        : 2017-01 .. 2025-06
  early stop : 2025-07 .. 2025-12   (chosen to stop training at the right point)
  test       : 2026-01 .. 2026-09   (touched once, at the end)

Target is log(resale_price). Compare against reports/baseline_test_predictions.csv.
Usage: python src/train_model.py [--no-time]   (--no-time drops the month index)
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from baseline import TEST_END, TEST_START, TRAIN_END
from geocode_blocks import ROOT

FIT_START = pd.Timestamp("2017-01-01")
VALID_START = pd.Timestamp("2025-07-01")

CATS = ["flat_type", "town", "flat_model"]
NUM = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "flat_age_years", "lat", "lon",
       "dist_mrt", "n_mrt_exits_500m", "dist_hawker", "dist_primary_school",
       "n_primary_1km", "n_primary_2km", "dist_cbd"]


def ape(actual, pred):
    return (pred - actual).abs() / actual


def summary(name, a):
    print(f"{name:34} median {a.median():6.2%}  mean {a.mean():6.2%}  within 5% {(a <= .05).mean():5.1%}  within 10% {(a <= .10).mean():5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    df = pd.read_parquet(ROOT / "data/processed/resale_features.parquet")
    df = df[df["sale_date"] >= FIT_START].copy()
    df["month_idx"] = (df["sale_year"] - 2017) * 12 + df["sale_month"] - 1
    for c in CATS:
        df[c] = df[c].astype("category")
    feats = CATS + NUM + ([] if args.no_time else ["month_idx"])

    fit = df[df["sale_date"] < VALID_START]
    valid = df[(df["sale_date"] >= VALID_START) & (df["sale_date"] <= TRAIN_END)]
    test = df[(df["sale_date"] >= TEST_START) & (df["sale_date"] <= TEST_END)]
    print(f"fit {len(fit)}  valid {len(valid)}  test {len(test)}  features {len(feats)}")

    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=5000, verbose=-1, device_type=cfg.get("device", "cpu"))
    model = lgb.LGBMRegressor(**params)
    model.fit(fit[feats], np.log(fit["resale_price"]),
              eval_set=[(valid[feats], np.log(valid["resale_price"]))],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    print(f"best iteration: {model.best_iteration_}")

    pv = np.exp(model.predict(valid[feats]))
    pt = np.exp(model.predict(test[feats]))
    print()
    summary("validation (2025-07..12)", ape(valid["resale_price"], pd.Series(pv, index=valid.index)))
    t_ape = ape(test["resale_price"], pd.Series(pt, index=test.index))
    summary("TEST  LightGBM", t_ape)

    base = pd.read_csv(ROOT / "reports/baseline_test_predictions.csv")
    summary("TEST  baseline B (same block)", ape(base["resale_price"], base["pred_b"]))

    res = test[["sale_date", "town", "flat_type", "resale_price", "remaining_lease_years", "dist_mrt"]].copy()
    res["pred"], res["ape"] = pt, t_ape.values
    print("\nLightGBM test error by flat type:")
    print(res.groupby("flat_type", observed=True)["ape"].agg(n="size", median="median", mean="mean").round(4).to_string())
    print("\nby remaining lease:")
    res["lease_band"] = pd.cut(res["remaining_lease_years"], [0, 50, 60, 70, 80, 90, 100])
    print(res.groupby("lease_band", observed=True)["ape"].agg(n="size", median="median").round(4).to_string())
    print("\nby month (bias = mean of pred/actual - 1):")
    res["bias"] = res["pred"] / res["resale_price"] - 1
    print(res.groupby(res["sale_date"].dt.strftime("%Y-%m"))[["ape", "bias"]].agg(["median" if False else "mean"]).round(4).to_string())

    imp = pd.Series(model.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\nfeature importance (share of gain):")
    print((imp / imp.sum()).round(3).head(12).to_string())

    (ROOT / "models").mkdir(exist_ok=True)
    tag = "v1_notime" if args.no_time else "v1"
    model.booster_.save_model(str(ROOT / "models" / f"lgbm_{tag}.txt"))
    res.to_csv(ROOT / "reports" / f"lgbm_{tag}_test_predictions.csv", index=False)
    print(f"\nsaved models/lgbm_{tag}.txt")


if __name__ == "__main__":
    main()
