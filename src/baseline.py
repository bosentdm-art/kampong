"""Step 3: 'similar flats nearby' baseline, no machine learning.

Split (by sale date, never random):
  train: up to 2025-12   test: 2026-01 .. 2026-09   (2026-10 is a partial month)

Predicted price = median price_per_sqm of comparable past sales x floor area.
  A) same town + flat type, sold in the last 12 months of the training period
  B) same block + flat type, last 24 months of training; falls back to A

Reports median and mean absolute percentage error (APE), and share within 5% / 10%.
"""
import numpy as np
import pandas as pd

from geocode_blocks import ROOT

TRAIN_END = pd.Timestamp("2025-12-01")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-09-01")


def split(df):
    train = df[df["sale_date"] <= TRAIN_END]
    test = df[(df["sale_date"] >= TEST_START) & (df["sale_date"] <= TEST_END)]
    return train, test


def report(name, actual, pred, mask=None):
    ape = (pred - actual).abs() / actual
    if mask is not None:
        ape = ape[mask]
    print(f"{name:28} n={len(ape):>6}  median APE {ape.median():6.2%}  mean APE {ape.mean():6.2%}  "
          f"within 5% {(ape <= .05).mean():5.1%}  within 10% {(ape <= .10).mean():5.1%}")
    return ape


def main():
    df = pd.read_parquet(ROOT / "data/processed/resale_features.parquet")
    train, test = split(df)
    print(f"train {len(train)} rows (to {TRAIN_END:%Y-%m}), test {len(test)} rows ({TEST_START:%Y-%m} to {TEST_END:%Y-%m})")

    # A: town + flat type, last 12 months of training
    rec12 = train[train["sale_date"] > TRAIN_END - pd.DateOffset(months=12)]
    a = rec12.groupby(["town", "flat_type"])["price_per_sqm"].median().rename("ppsm_a").reset_index()
    t = test.merge(a, on=["town", "flat_type"], how="left")

    # B: block + flat type, last 24 months of training
    rec24 = train[train["sale_date"] > TRAIN_END - pd.DateOffset(months=24)]
    b = rec24.groupby(["block", "street_name", "flat_type"])["price_per_sqm"].agg(["median", "size"]).reset_index()
    b = b.rename(columns={"median": "ppsm_b", "size": "n_b"})
    t = t.merge(b, on=["block", "street_name", "flat_type"], how="left")

    t["pred_a"] = t["ppsm_a"] * t["floor_area_sqm"]
    t["pred_b"] = t["ppsm_b"].fillna(t["ppsm_a"]) * t["floor_area_sqm"]
    has_b = t["ppsm_b"].notna()
    unmatched = t["ppsm_a"].isna()
    print(f"test rows with no town+type comparables: {int(unmatched.sum())}; "
          f"with block-level comparables: {has_b.mean():.1%}\n")

    ok = ~unmatched
    report("A town + type (12m)", t["resale_price"], t["pred_a"], ok)
    report("B block + type (24m), fallback A", t["resale_price"], t["pred_b"], ok)
    report("B, block comparables only", t["resale_price"], t["pred_b"], ok & has_b)

    print("\nBaseline B by flat type:")
    for ft, g in t[ok].groupby("flat_type"):
        report(f"  {ft}", g["resale_price"], g["pred_b"])

    out = ROOT / "reports" / "baseline_test_predictions.csv"
    t[["sale_date", "town", "flat_type", "block", "street_name", "floor_area_sqm", "resale_price", "pred_a", "pred_b"]].to_csv(out, index=False)
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
