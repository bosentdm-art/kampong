"""Step 5, stage B: resale-value forecast for one HDB flat.

value(h) = V0 x exp(g x h) x lease ratio

  V0           today's price from the LightGBM refit model (models/lgbm_refit.txt)
  g            a market growth SCENARIO per year (0% stress, 2% recent regime, 4% long-run).
               Nobody can forecast 20-year market growth: 10-year windows since 1990
               ranged from about +1% to +11% a year depending on the period, so we show
               scenarios instead of pretending to a probability.
  lease ratio  value at the lease left in h years / value at the lease left now, from the
               distance-from-CBD band's lease-decay curve (fitted on 2016+ sales).
               A backtest (src/backtest.py) found the curve overstates value loss (slope ~0.2, 1.0 = right),
               so it is applied at 0%, 50% and 100% strength. No distance-from-CBD split: not validated.

The low / high band is only the model's error on today's price (10th and 90th percentile of
actual / predicted on the 2026 test months). It does not widen with the horizon.
It is a rough guide, not a guarantee: see PLAN.md for the caveats.

Run `python src/forecast.py` for a demo on real recent sales.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from geocode_blocks import ROOT

PROC = ROOT / "data/processed"
CATS = ["flat_type", "town", "flat_model"]
FEATS = CATS + ["floor_area_sqm", "storey_mid", "remaining_lease_years", "lat", "lon", "dist_mrt",
                "n_mrt_exits_500m", "dist_hawker", "dist_primary_school", "n_primary_1km", "n_primary_2km",
                "dist_cbd", "month_idx"]
SCENARIOS = {"stress 0%/yr": 0.00, "recent 2%/yr": 0.02, "long-run 4%/yr": 0.04}
LEASE_STRENGTH = {"no decay": 0.0, "half curve": 0.5, "full curve": 1.0}
EXTRAPOLATED_BELOW = 45  # years of lease: below this the data is too thin to trust the curve


def region_of(dist_cbd_m):
    cut = json.loads((PROC / "region_cutoffs.json").read_text())
    return "near" if dist_cbd_m <= cut["near_max_m"] else "mid" if dist_cbd_m <= cut["mid_max_m"] else "far"


def lease_value():
    """Return f(lease_years) -> log value vs a 99-year flat; non-increasing as the lease falls.

    Below ~50 years the fitted curve is noisy, so it is extended with the slope of the 60-to-50 segment."""
    lc = pd.read_csv(PROC / f"lease_curve_2016.csv").sort_values("remaining_lease_years")
    L, c = lc["remaining_lease_years"].to_numpy(), lc["log_value_vs_99"].to_numpy()
    c = np.minimum.accumulate(c[::-1])[::-1]  # value may not rise as the lease falls
    slope = (np.interp(60, L, c) - np.interp(50, L, c)) / 10  # log value per year of lease
    base = np.interp(50, L, c)

    def f(x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 50, np.interp(x, L, c), base - slope * (50 - x))
    return f


def model_error_band():
    """(q10, q90) of log(actual / predicted) for the model on the 2026 test months."""
    r = pd.read_csv(ROOT / "reports/lgbm_refit_test_predictions.csv")
    e = np.log(r["resale_price"] / r["pred"])
    return float(e.quantile(0.10)), float(e.quantile(0.90))


def price_today(spec, row, when):
    booster = lgb.Booster(model_file=str(ROOT / "models/lgbm_refit.txt"))
    x = pd.DataFrame([{**{k: row[k] for k in FEATS if k in row}, **spec}])
    x["remaining_lease_years"] = 99 - (when - spec["lease_commence_date"])
    x["month_idx"] = round((when - 2017) * 12)  # same convention as training: year + (month-1)/12
    x["storey_mid"] = spec["storey_mid"]
    for c in CATS:
        x[c] = x[c].astype("category")
    return float(np.exp(booster.predict(x[FEATS])[0])), float(x.loc[0, "remaining_lease_years"])


def forecast(spec, when=2026 + 9 / 12, horizons=(5, 10, 15, 20)):  # Oct 2026
    feats = pd.read_parquet(PROC / "resale_features.parquet",
                            columns=["block", "street_name", "lat", "lon", "dist_mrt", "n_mrt_exits_500m", "dist_hawker",
                                     "dist_primary_school", "n_primary_1km", "n_primary_2km", "dist_cbd"])
    hit = feats[(feats["block"] == spec["block"]) & (feats["street_name"] == spec["street_name"])]
    if hit.empty:
        raise ValueError("block not found (demolished or never geocoded)")
    v0, lease0 = price_today(spec, hit.iloc[0].to_dict(), when)
    f = lease_value()
    q10, q90 = model_error_band()

    out = []
    for h in horizons:
        L_h = lease0 - h
        if L_h <= 0:
            out.append({"years": h, "lease_left": round(L_h, 1), "note": "lease ends before this horizon"})
            continue
        lr = float(f(L_h) - f(lease0))  # log value change from the lease alone, full curve
        grid = {(g_name, l_name): v0 * np.exp(w * lr) * np.exp(g * h)
                for g_name, g in SCENARIOS.items() for l_name, w in LEASE_STRENGTH.items()}
        out.append({"years": h, "lease_left": round(L_h, 1), "lease_full": np.exp(lr) - 1, "grid": grid,
                    "note": "extrapolated lease curve" if L_h < EXTRAPOLATED_BELOW else ""})
    return {"v0": v0, "lease_now": lease0, "band": (np.exp(q10) - 1, np.exp(q90) - 1), "rows": out}


def show(res):
    lo, hi = res["band"]
    print(f"Model value today: ${res['v0']:,.0f} (typical error {lo:+.1%} to {hi:+.1%}) | lease left: {res['lease_now']:.1f} yrs")
    print("WARNING: the lease effect is NOT validated. A backtest (src/backtest.py) found the fitted curve overstates")
    print("         value loss (calibration slope about 0.2, where 1.0 = correct), so it is applied at 0%, 50%, 100%.")
    print("low = flat market + full lease curve | central = 2%/yr + half the curve | high = 4%/yr + no lease decay")
    print()
    print(f"{'years':>5} {'lease':>6} {'full-curve effect':>18} {'low':>11} {'central':>11} {'high':>11}  note")
    for r in res["rows"]:
        if "grid" not in r:
            print(f"{r['years']:>5} {r['lease_left']:>6}  {r['note']}")
            continue
        g = r["grid"]
        print(f"{r['years']:>5} {r['lease_left']:>6.1f} {r['lease_full']:>+18.0%} "
              f"{g[('stress 0%/yr', 'full curve')]:>11,.0f} {g[('recent 2%/yr', 'half curve')]:>11,.0f} "
              f"{g[('long-run 4%/yr', 'no decay')]:>11,.0f}  {r['note']}")


if __name__ == "__main__":
    df = pd.read_parquet(PROC / "resale_features.parquet")
    recent = df[(df["sale_date"] >= "2026-06-01") & (df["flat_type"] == "4 ROOM")]
    for town in ["TOA PAYOH", "TAMPINES"]:
        s = recent[recent["town"] == town].sample(1, random_state=1).iloc[0]
        spec = dict(flat_type=s["flat_type"], town=s["town"], flat_model=s["flat_model"], block=s["block"],
                    street_name=s["street_name"], floor_area_sqm=float(s["floor_area_sqm"]), storey_mid=float(s["storey_mid"]),
                    lease_commence_date=int(s["lease_commence_date"]))
        print(f"=== {s['flat_type']} {s['town']} blk {s['block']} {s['street_name']}, {s['floor_area_sqm']:.0f} sqm, "
              f"storey {s['storey_range']}, lease from {s['lease_commence_date']}; sold {s['sale_date']:%Y-%m} for ${s['resale_price']:,.0f}")
        show(forecast(spec))
        print()
