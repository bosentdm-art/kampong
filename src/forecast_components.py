"""Step 5, stage A: estimate the two building blocks of a resale-value forecast.

Hedonic regression on log(price per sqm) with
  - a fixed effect for every sale month           -> market index (quality-adjusted)
  - town x flat type and flat model fixed effects
  - log floor area, storey, location distances
  - a smooth spline in remaining lease            -> lease-decay curve

The month effects hold flat quality and lease constant, so the index is the pure
market. The lease spline is identified across flats of different ages in the
same month. Caveat: it is cross-sectional, so older flats' unmeasured differences
(renovation, layout, view) are mixed into it.

Outputs: data/processed/market_index.csv, data/processed/lease_curve.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from geocode_blocks import ROOT


def fit(df, offset_curve=None):
    """offset_curve: DataFrame with remaining_lease_years, log_value_vs_99. If given, that lease
    effect is subtracted from y and no lease spline is fitted."""
    y = np.log(df["price_per_sqm"].to_numpy())
    if offset_curve is not None:
        y = y - np.interp(df["remaining_lease_years"], offset_curve["remaining_lease_years"], offset_curve["log_value_vs_99"])
    cat = pd.DataFrame({
        "month": df["month_n"].to_numpy(),
        "town_ft": (df["town"] + "|" + df["flat_type"]).to_numpy(),
        "model": df["flat_model"].astype(str).to_numpy(),
    })
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    Xc = enc.fit_transform(cat)

    if offset_curve is None:
        spl = SplineTransformer(n_knots=9, degree=3, knots="uniform", extrapolation="constant", include_bias=False)
        Xl = spl.fit_transform(df[["remaining_lease_years"]])
    else:
        spl, Xl = None, np.zeros((len(df), 0))

    num = np.c_[np.log(df["floor_area_sqm"]), df["storey_mid"], df["dist_mrt"] / 1000, df["dist_cbd"] / 1000,
                df["dist_primary_school"] / 1000]
    num = (num - num.mean(0)) / num.std(0)

    X = sp.hstack([Xc, sp.csr_matrix(Xl), sp.csr_matrix(num)]).tocsr()
    model = Ridge(alpha=1e-3, solver="sparse_cg", fit_intercept=True, max_iter=2000).fit(X, y)
    resid = y - model.predict(X)
    return enc, spl, model, Xc.shape[1], Xl.shape[1], resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="1990-01-01", help="only fit on sales from this date")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--region", default="all", choices=["all", "near", "mid", "far"],
                    help="distance-from-CBD tercile of the block (cut-offs saved to region_cutoffs.json)")
    ap.add_argument("--offset-curve", default="", help="tag of a saved lease curve to apply instead of fitting one")
    args = ap.parse_args()
    proc = ROOT / "data/processed"
    df = pd.read_parquet(proc / "resale_features.parquet")
    df["month_n"] = (df["sale_year"] - 1990) * 12 + df["sale_month"] - 1
    blocks = df.drop_duplicates(["block", "street_name"])["dist_cbd"]
    c1, c2 = blocks.quantile([1 / 3, 2 / 3])
    (proc / "region_cutoffs.json").write_text(json.dumps({"near_max_m": c1, "mid_max_m": c2}))
    if args.region != "all":
        keep = {"near": df["dist_cbd"] <= c1, "mid": (df["dist_cbd"] > c1) & (df["dist_cbd"] <= c2), "far": df["dist_cbd"] > c2}
        df = df[keep[args.region]]
    df = df[(df["sale_date"] <= "2026-09-01") & (df["sale_date"] >= args.since)]  # drop the partial last month

    off = pd.read_csv(proc / f"lease_curve{args.offset_curve}.csv") if args.offset_curve else None
    enc, spl, model, n_cat, n_spl, resid = fit(df, off)
    print(f"fit on {len(df)} sales, residual std {resid.std():.3f} (log units)")

    # market index: month effects
    cats = enc.categories_
    nm = len(cats[0])
    coef_month = model.coef_[:nm]
    idx = pd.DataFrame({"month_n": cats[0].astype(int), "log_index": coef_month})
    idx["log_index"] -= idx["log_index"].iloc[0]
    idx["month"] = pd.to_datetime("1990-01-01") + pd.to_timedelta(idx["month_n"] * 30.4375, unit="D")
    idx["month"] = idx["month"].dt.to_period("M").astype(str)
    idx[["month", "month_n", "log_index"]].to_csv(proc / f"market_index{args.tag}.csv", index=False)

    # lease curve: spline part, relative to 99 years (only when one was fitted)
    if spl is not None:
        beta = model.coef_[n_cat:n_cat + n_spl]
        grid = np.arange(38, 99.01, 1.0)
        curve = spl.transform(pd.DataFrame({"remaining_lease_years": grid})) @ beta
        ref = (spl.transform(pd.DataFrame({"remaining_lease_years": [99.0]})) @ beta)[0]
        lc = pd.DataFrame({"remaining_lease_years": grid, "log_value_vs_99": curve - ref})
        lc["value_vs_99"] = np.exp(lc["log_value_vs_99"])
        lc.to_csv(proc / f"lease_curve{args.tag}.csv", index=False)

        print("\nLEASE-DECAY CURVE (value relative to a 90-year-lease flat, all else equal):")
        r90 = lc.loc[(lc.remaining_lease_years - 90).abs().idxmin(), "log_value_vs_99"]
        for L in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]:
            v = lc.loc[(lc.remaining_lease_years - L).abs().idxmin(), "log_value_vs_99"]
            print(f"  {L:>3} years left: {np.exp(v - r90):6.1%}  (vs 90 years)")
    else:
        print(f"\n(lease curve fixed from lease_curve{args.offset_curve}.csv, not refitted)")

    # market index growth
    s = idx.set_index("month_n")["log_index"]
    first = s.index.min()
    print(f"\nMARKET INDEX (price level of a constant-quality flat, {idx['month'].iloc[0]} = 100):")
    for y in [1990, 1996, 2000, 2004, 2008, 2013, 2017, 2020, 2022, 2024, 2026]:
        mn = (y - 1990) * 12 + (8 if y == 2026 else 0)
        if mn in s.index:
            print(f"  {y}: {np.exp(s.loc[mn]) * 100:6.0f}")
    print("\nANNUALISED MARKET GROWTH over rolling windows (every start month in the fit period):")
    for yrs in (3, 5, 10, 20):
        w = yrs * 12
        g = (s.shift(-w) - s).dropna() / yrs
        if len(g) < 12:
            continue
        pct = np.exp(g.quantile([.1, .25, .5, .75, .9])) - 1
        print(f"  {yrs:>2}y windows (n={len(g):3d}):  p10 {pct.iloc[0]:+5.1%}  p25 {pct.iloc[1]:+5.1%}  median {pct.iloc[2]:+5.1%}  "
              f"p75 {pct.iloc[3]:+5.1%}  p90 {pct.iloc[4]:+5.1%}")
    print(f"\nsaved data/processed/market_index{args.tag}.csv")


if __name__ == "__main__":
    main()
