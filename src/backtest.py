"""Step 5, stage C: backtest the lease-decay curve out of sample.

For each origin year T0 (windows end in September):
  1. fit the lease curve using ONLY the 6 years of sales before the origin
  2. for each block + flat type, compare the realised change in median price per sqm between
     the 12 months ending at T0 and the 12 months ending at T0 + h with the change the curve
     predicts from the lease shrinking
  3. regress realised on predicted across blocks (weighted by sales). The market's common
     movement lands in the intercept, so it cancels out; the SLOPE is the calibration:
        1.0 = curve right, <1 = curve too steep, >1 = curve too shallow.

Blocks whose lease at T0 + h falls below what the curve was fitted on are excluded
(the curve cannot be tested there) and counted separately.
The pooled curve is used everywhere; slopes by distance-from-CBD band test whether the
near / mid / far split used in forecast.py is warranted.
"""
import json

import numpy as np
import pandas as pd

from forecast_components import fit
from geocode_blocks import ROOT

PROC = ROOT / "data/processed"
ORIGINS = [2010, 2013, 2016, 2018]
HORIZONS = [5, 8, 10]
TRAIN_MONTHS = 72
MIN_SALES = 3
RNG = np.random.default_rng(0)


def month_n(y, m=9):
    return (y - 1990) * 12 + m - 1


def lease_curve(train):
    """Fit the pooled hedonic model and return (grid, non-increasing log value) as forecast.py deploys it."""
    enc, spl, model, n_cat, n_spl, _ = fit(train)
    grid = np.arange(30, 99.01, 0.5)
    c = spl.transform(pd.DataFrame({"remaining_lease_years": grid})) @ model.coef_[n_cat:n_cat + n_spl]
    c = np.minimum.accumulate(c[::-1])[::-1]
    return grid, c


def wslope(x, y, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def slope_ci(x, y, w, reps=400):
    s = [wslope(x[i], y[i], w[i]) for i in (RNG.integers(0, len(x), len(x)) for _ in range(reps))]
    return np.percentile(s, [2.5, 97.5])


def window(df, end):
    w = df[(df["month_n"] > end - 12) & (df["month_n"] <= end)]
    return w.groupby(["block", "street_name", "flat_type"]).agg(
        n=("price_per_sqm", "size"), ppsm=("price_per_sqm", "median"),
        lease=("remaining_lease_years", "median"), dist_cbd=("dist_cbd", "first")).reset_index()


def main():
    df = pd.read_parquet(PROC / "resale_features.parquet")
    df["month_n"] = (df["sale_year"] - 1990) * 12 + df["sale_month"] - 1
    df = df[df["month_n"] <= month_n(2026)]
    cut = json.loads((PROC / "region_cutoffs.json").read_text())

    rows, by_region = [], []
    for t0 in ORIGINS:
        e0 = month_n(t0)
        train = df[(df["month_n"] > e0 - TRAIN_MONTHS) & (df["month_n"] <= e0)]
        grid, c = lease_curve(train)
        lo_support = float(train["remaining_lease_years"].quantile(0.01))
        w0 = window(df, e0)
        print(f"origin {t0}: curve fitted on {len(train)} sales, lease support from {lo_support:.0f} yrs "
              f"(value at {lo_support:.0f}y vs 90y: {np.exp(np.interp(lo_support, grid, c) - np.interp(90, grid, c)):.0%})")
        for h in HORIZONS:
            if t0 + h > 2026:
                continue
            w1 = window(df, month_n(t0 + h))
            m = w0.merge(w1, on=["block", "street_name", "flat_type"], suffixes=("0", "1"))
            m = m[(m["n0"] >= MIN_SALES) & (m["n1"] >= MIN_SALES)]
            m["x"] = np.interp(m["lease1"], grid, c) - np.interp(m["lease0"], grid, c)
            m["y"] = np.log(m["ppsm1"] / m["ppsm0"])
            m["w"] = np.minimum(m["n0"], m["n1"]).astype(float)
            ins = m[m["lease1"] >= lo_support]
            out = m[m["lease1"] < lo_support]
            x, y, w = ins["x"].to_numpy(), ins["y"].to_numpy(), ins["w"].to_numpy()
            lo, hi = slope_ci(x, y, w)
            r = np.corrcoef(x, y)[0, 1]
            # does the curve beat "no lease decay"? compare spread of y around its own mean vs around the fitted line
            slope = wslope(x, y, w)
            resid = y - (np.average(y, weights=w) + slope * (x - np.average(x, weights=w)))
            rows.append(dict(origin=t0, h=h, blocks=len(ins), beyond_support=len(out),
                             pred_effect=np.average(x, weights=w), realised=np.average(y, weights=w),
                             slope=slope, ci_lo=lo, ci_hi=hi, r=r, sd_before=np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w)),
                             sd_after=np.sqrt(np.average(resid ** 2, weights=w))))
            ins = ins.assign(region=np.where(ins["dist_cbd0"] <= cut["near_max_m"], "near",
                                             np.where(ins["dist_cbd0"] <= cut["mid_max_m"], "mid", "far")))
            for reg, g in ins.groupby("region"):
                if len(g) >= 40:
                    by_region.append(dict(origin=t0, h=h, region=reg, blocks=len(g),
                                          slope=wslope(g["x"].to_numpy(), g["y"].to_numpy(), g["w"].to_numpy())))

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print("CALIBRATION OF THE LEASE CURVE (slope 1.0 = right; <1 too steep; >1 too shallow)")
    print(res.round(3).to_string(index=False))
    print()
    print("SLOPE BY DISTANCE-FROM-CBD BAND (pooled curve; near>mid>far would support the region split)")
    reg = pd.DataFrame(by_region).pivot_table(index=["origin", "h"], columns="region", values="slope").round(2)
    print(reg[["near", "mid", "far"]].to_string())
    res.to_csv(ROOT / "reports" / "backtest_lease_curve.csv", index=False)
    print()
    print("saved reports/backtest_lease_curve.csv")


if __name__ == "__main__":
    main()
