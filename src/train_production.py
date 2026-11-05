"""Production model: retrain on every sale up to the latest complete month.

Settings are fixed from the tuning in train_model.py: recency weights (36-month half-life),
price may not rise as the remaining lease falls, 1,281 rounds. Re-tune now and then with
train_model.py (the number of rounds in particular) rather than every month.

Also runs a holdout check: train through six months before the latest month, test on those
six months. That gives the error figure and the error band used by forecast.py and the app.

Outputs: models/lgbm_prod.txt, models/lgbm_prod.json, reports/prod_holdout_predictions.csv
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from geocode_blocks import ROOT
from train_model import CATS, NUM

ROUNDS = 1281
HALF_LIFE_MONTHS = 36
HOLDOUT_MONTHS = 6
FEATS = CATS + [c for c in NUM if c != "flat_age_years"] + ["month_idx"]
MONO = [1 if c == "remaining_lease_years" else 0 for c in FEATS]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=ROUNDS, verbose=-1, monotone_constraints=MONO)


def latest_complete_month(df):
    """The last month, unless it has far fewer sales than the months before it (still filling in)."""
    counts = df.groupby("sale_date").size()
    if counts.iloc[-1] >= 0.7 * counts.iloc[-4:-1].median():
        return counts.index[-1]
    return counts.index[-2]


def fit(d, ref_date):
    months_back = (ref_date - d["sale_date"]).dt.days / 30.44
    w = 0.5 ** (months_back / HALF_LIFE_MONTHS)
    return lgb.LGBMRegressor(**PARAMS).fit(d[FEATS], np.log(d["resale_price"]), sample_weight=w)


def main():
    df = pd.read_parquet(ROOT / "data/processed/resale_features.parquet")
    df = df[df["sale_date"] >= "2017-01-01"].copy()
    df["month_idx"] = (df["sale_year"] - 2017) * 12 + df["sale_month"] - 1
    for c in CATS:
        df[c] = df[c].astype("category")

    last = latest_complete_month(df)
    df = df[df["sale_date"] <= last]
    print(f"latest complete month: {last:%Y-%m}; {len(df)} sales since 2017")

    # holdout check
    first_hold = last - pd.DateOffset(months=HOLDOUT_MONTHS - 1)
    train, test = df[df["sale_date"] < first_hold], df[df["sale_date"] >= first_hold]
    m = fit(train, first_hold)
    pred = np.exp(m.predict(test[FEATS]))
    ape = (pred - test["resale_price"]).abs() / test["resale_price"]
    print(f"holdout {first_hold:%Y-%m}..{last:%Y-%m} ({len(test)} sales): median error {ape.median():.2%}, "
          f"mean {ape.mean():.2%}, within 10% {(ape <= .10).mean():.1%}")
    if ape.median() > 0.05:
        print("WARNING: holdout median error is above 5%; the market may have shifted. Consider re-tuning.")
    (ROOT / "reports").mkdir(exist_ok=True)
    pd.DataFrame({"sale_date": test["sale_date"].values, "resale_price": test["resale_price"].values,
                  "pred": pred}).to_csv(ROOT / "reports/prod_holdout_predictions.csv", index=False)

    # production fit on everything
    model = fit(df, last + pd.DateOffset(months=1))
    (ROOT / "models").mkdir(exist_ok=True)
    model.booster_.save_model(str(ROOT / "models/lgbm_prod.txt"))
    meta = dict(trained_through=f"{last:%Y-%m}", rows=int(len(df)), rounds=ROUNDS,
                holdout_median_error=round(float(ape.median()), 4), holdout_mean_error=round(float(ape.mean()), 4))
    (ROOT / "models/lgbm_prod.json").write_text(json.dumps(meta, indent=2))
    print("saved models/lgbm_prod.txt", meta)


if __name__ == "__main__":
    main()
