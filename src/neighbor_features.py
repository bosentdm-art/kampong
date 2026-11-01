"""Add 'recent nearby sales' features, using only sales from EARLIER months.

For a sale in month m, every feature is computed from sales dated before m, so
a sale never sees its own month (no label leakage). Features are means of
log(price per sqm):
  town_ft_3m / town_ft_12m   : same town + flat type, prior 3 / 12 months
  block_ft_12m / block_ft_24m: same block + flat type, prior 12 / 24 months
  plus the number of sales behind each (n_*), so the model knows how reliable it is.

Input : data/processed/resale_features.parquet
Output: data/processed/resale_features_v2.parquet
"""
import numpy as np
import pandas as pd

from geocode_blocks import ROOT


def window_feature(df, key_cols, window, prefix):
    """Mean log ppsm and count over the `window` months before each row's month."""
    keys = df[key_cols].astype(str).agg("|".join, axis=1)
    k_codes, _ = pd.factorize(keys)
    m = df["month_n"].to_numpy()
    n_keys, n_months = k_codes.max() + 1, m.max() + 2

    s = np.zeros((n_keys, n_months))
    c = np.zeros((n_keys, n_months))
    np.add.at(s, (k_codes, m + 1), df["lp"].to_numpy())  # column j holds month j-1
    np.add.at(c, (k_codes, m + 1), 1.0)
    s, c = s.cumsum(axis=1), c.cumsum(axis=1)  # s[:, j] = sum over months < j

    lo = np.maximum(m - window, 0)
    ws = s[k_codes, m] - s[k_codes, lo]
    wc = c[k_codes, m] - c[k_codes, lo]
    df[f"{prefix}_lp"] = np.where(wc > 0, ws / np.maximum(wc, 1), np.nan)
    df[f"{prefix}_n"] = wc


def main():
    df = pd.read_parquet(ROOT / "data/processed/resale_features.parquet")
    df["lp"] = np.log(df["price_per_sqm"])
    df["month_n"] = (df["sale_year"] - 1990) * 12 + df["sale_month"] - 1

    window_feature(df, ["town", "flat_type"], 3, "town_ft_3m")
    window_feature(df, ["town", "flat_type"], 12, "town_ft_12m")
    window_feature(df, ["block", "street_name", "flat_type"], 12, "block_ft_12m")
    window_feature(df, ["block", "street_name", "flat_type"], 24, "block_ft_24m")

    df = df.drop(columns=["lp"])
    out = ROOT / "data/processed/resale_features_v2.parquet"
    df.to_parquet(out, index=False)
    cols = [c for c in df.columns if c.startswith(("town_ft", "block_ft"))]
    print(f"saved {out.relative_to(ROOT)}  rows {len(df)}")
    print(df.loc[df["sale_year"] >= 2017, cols].describe().round(2).T[["count", "mean", "min", "max"]].to_string())

    # leakage check: a feature must not change if we delete the row's own month
    probe = df[(df["sale_year"] == 2026) & (df["sale_month"] == 5)].iloc[0]
    same = df[(df["town"] == probe["town"]) & (df["flat_type"] == probe["flat_type"])]
    prior = same[(same["month_n"] >= probe["month_n"] - 3) & (same["month_n"] < probe["month_n"])]
    manual = np.log(prior["price_per_sqm"]).mean()
    print(f"\nleakage spot check (town_ft_3m): computed {probe['town_ft_3m_lp']:.6f}  manual {manual:.6f}")


if __name__ == "__main__":
    main()
