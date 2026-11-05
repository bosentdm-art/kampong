"""Step 5: resale-value forecast for one HDB flat, calibrated by backtesting.

central(h) = V0 x exp(g x h) x exp(w x lease effect) x exp(median backtest error at h)

  V0           today's price from the production model (average of 3 LightGBM models)
  g            annual log growth: 4% nudged against recent growth (g = ln1.04 + k x (trailing 5y - ln1.04),
               k = -0.25 chosen by backtest: markets mean-revert, so extrapolating momentum fails badly)
  lease effect log value change from the pooled lease-decay curve (fitted on 2016+ sales), applied at weight w
  band         10th to 90th percentile of the errors this method made on block-level prices from
               start years 1998-2016, widened until it covered ~80% of outcomes on start years left out
               (src/backtest_forecast.py -> data/processed/forecast_calibration.json), combined in
               quadrature with the model's own error on today's price.

No projection is shown where the lease would fall under 50 years: the backtests only cover flats that stay above
about that, so any number there would have no evidence behind it.

Run `python src/forecast.py` for a demo on real recent sales.
"""
import functools
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from geocode_blocks import ROOT

PROC = ROOT / "data/processed"
CATS = ["flat_type", "town", "flat_model"]
FEATS = CATS + ["floor_area_sqm", "storey_mid", "remaining_lease_years", "lat", "lon", "dist_mrt",
                "n_mrt_exits_500m", "dist_hawker", "dist_primary_school", "n_primary_1km", "n_primary_2km",
                "dist_cbd", "month_idx", "lease_commence_date", "sale_month", "storey_low", "storey_high"]
NO_PROJECTION_BELOW = 50  # years of lease: the backtests only cover flats that stay above about 50 years


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


def model_error_band(lo_q=0.10, hi_q=0.90):
    """(lo, hi) quantiles of log(actual / predicted) for today's price, from the production holdout."""
    r = pd.read_csv(ROOT / "reports/prod_holdout_predictions.csv")
    e = np.log(r["resale_price"] / r["pred"])
    return float(e.quantile(lo_q)), float(e.quantile(hi_q))


@functools.lru_cache(maxsize=1)
def load_models():
    files = sorted((ROOT / "models").glob("lgbm_prod_*.txt"))
    if not files:
        raise FileNotFoundError("no production model found; run python src/update.py")
    return [lgb.Booster(model_file=str(f)) for f in files]


def price_today(spec, row, when):
    x = pd.DataFrame([{**{k: row[k] for k in FEATS if k in row}, **spec}])
    x["remaining_lease_years"] = 99 - (when - spec["lease_commence_date"])
    x["month_idx"] = round((when - 2017) * 12)  # same convention as training: year + (month-1)/12
    x["sale_month"] = round((when % 1) * 12) + 1
    x["storey_low"], x["storey_high"] = spec["storey_mid"] - 1, spec["storey_mid"] + 1  # ranges are 3 storeys wide
    for c in CATS:
        x[c] = x[c].astype("category")
    price = float(np.exp(np.mean([b.predict(x[FEATS])[0] for b in load_models()])))
    return price, float(x.loc[0, "remaining_lease_years"])


@functools.lru_cache(maxsize=1)
def block_table():
    """One row of location features per block (about 10,000 rows)."""
    cols = ["block", "street_name", "lat", "lon", "dist_mrt", "n_mrt_exits_500m", "dist_hawker",
            "dist_primary_school", "n_primary_1km", "n_primary_2km", "dist_cbd"]
    return pd.read_parquet(PROC / "resale_features.parquet", columns=cols).drop_duplicates(["block", "street_name"])


def default_when():
    """Valuation date = the month the production model was trained through (year + (month-1)/12)."""
    meta = ROOT / "models/lgbm_prod.json"
    if meta.exists():
        y, m = json.loads(meta.read_text())["trained_through"].split("-")
        return int(y) + (int(m) - 1) / 12
    return 2026 + 9 / 12  # Oct 2026, the refit model's last month


@functools.lru_cache(maxsize=1)
def calibration():
    return json.loads((PROC / "forecast_calibration.json").read_text())


@functools.lru_cache(maxsize=1)
def monthly_prices():
    d = pd.read_parquet(PROC / "resale_features.parquet", columns=["sale_year", "sale_month", "price_per_sqm"])
    d["m"] = d["sale_year"] * 12 + d["sale_month"] - 1
    return d[["m", "price_per_sqm"]]


def growth_now(when):
    """Annual log growth to assume: ln(1+base) nudged by k x (trailing 5y growth - ln(1+base))."""
    cal = calibration()
    end = int(round(when * 12)) + 0  # month number of the valuation month, year*12 + (month-1)
    d = monthly_prices()
    p1 = d.loc[(d["m"] > end - 12) & (d["m"] <= end), "price_per_sqm"].median()
    p0 = d.loc[(d["m"] > end - 72) & (d["m"] <= end - 60), "price_per_sqm"].median()
    trailing = float(np.log(p1 / p0) / 5)
    base = np.log1p(cal["base_growth"])
    return base + (cal["damping"] or 0.0) * (trailing - base), trailing


def band_for(h):
    """Smoothed calibration for horizon h: (80% down, bias shift, 80% up, 50% down, 50% up), all in log units.

    The backtest has only 5 to 10 start years per horizon, so raw bands are erratic (a 5-year band can be
    wider than a 10-year band). Two smoothing rules: a band is never narrower than the bands of shorter
    horizons, and the bias shift is one pooled value (mean over horizons of 5+ years), not one per horizon."""
    bands = calibration()["bands"]
    down = up = down50 = up50 = 0.0
    for k in sorted(bands, key=int):
        if int(k) <= h:
            down = max(down, bands[k]["q50"] - bands[k]["q10"])
            up = max(up, bands[k]["q90"] - bands[k]["q50"])
            down50 = max(down50, bands[k]["q50"] - bands[k]["l25"])
            up50 = max(up50, bands[k]["l75"] - bands[k]["q50"])
    long = [bands[k]["q50"] for k in bands if int(k) >= 5]
    return down, float(np.mean(long)), up, down50, up50


def forecast(spec, when=None, horizons=(5, 10, 15, 20), growth=None):
    when = default_when() if when is None else when
    feats = block_table()
    hit = feats[(feats["block"] == spec["block"]) & (feats["street_name"] == spec["street_name"])]
    if hit.empty:
        raise ValueError("block not found (demolished or never geocoded)")
    v0, lease0 = price_today(spec, hit.iloc[0].to_dict(), when)
    f = lease_value()
    cal = calibration()
    g, trailing = growth_now(when)
    custom = growth is not None  # user-set annual growth rate (e.g. 0.035): a scenario, so no bias shift is applied
    if custom:
        g = float(np.log1p(growth))
    m10, m90 = model_error_band()  # log errors of today's price
    m25, m75 = model_error_band(0.25, 0.75)

    out = []
    for h in horizons:
        L_h = lease0 - h
        if L_h <= 0:
            out.append({"years": h, "lease_left": round(L_h, 1), "note": "lease ends before this horizon"})
            continue
        if L_h < NO_PROJECTION_BELOW:
            out.append({"years": h, "lease_left": round(L_h, 1),
                        "note": f"not projected: lease would be under {NO_PROJECTION_BELOW} years"})
            continue
        lease_log = float(f(L_h) - f(lease0))
        down, shift, up, down50, up50 = band_for(h)
        centre = np.log(v0) + g * h + cal["lease_weight"] * lease_log + (0.0 if custom else shift)
        lo = np.hypot(down, m10)  # the two error sources are treated as independent
        hi = np.hypot(up, m90)
        out.append({"years": h, "lease_left": round(L_h, 1), "central": float(np.exp(centre)),
                    "low": float(np.exp(centre - lo)), "high": float(np.exp(centre + hi)),
                    "likely_low": float(np.exp(centre - np.hypot(down50, m25))),
                    "likely_high": float(np.exp(centre + np.hypot(up50, m75))),
                    "note": ""})
    return {"v0": v0, "lease_now": lease0, "band": (np.exp(m10) - 1, np.exp(m90) - 1),
            "growth": g, "trailing": trailing, "custom": custom,
            # the default's central case includes a small calibrated level correction; this is the growth rate that
            # reproduces it at 5 years, so a user-set growth equal to it gives (about) the same 5-year central value
            "growth_effective": g if custom else g + band_for(5)[1] / 5,
            "rows": out}


def show(res):
    lo, hi = res["band"]
    print(f"Model value today: ${res['v0']:,.0f} (typical error {lo:+.1%} to {hi:+.1%}) | lease left: {res['lease_now']:.1f} yrs"
          f" | assumed growth {np.expm1(res['growth']):+.1%}/yr (recent 5y: {np.expm1(res['trailing']):+.1%}/yr)")
    print("Likely = 50% range, wide = 80% range; both from back-tests of this method on 1998-2016 start years.")
    print()
    print(f"{'years':>5} {'lease':>6} {'wide low':>11} {'likely low':>11} {'central':>11} {'likely high':>12} {'wide high':>11}  note")
    for r in res["rows"]:
        if "central" not in r:
            print(f"{r['years']:>5} {r['lease_left']:>6}  {r['note']}")
            continue
        print(f"{r['years']:>5} {r['lease_left']:>6.1f} {r['low']:>11,.0f} {r['likely_low']:>11,.0f} {r['central']:>11,.0f} "
              f"{r['likely_high']:>12,.0f} {r['high']:>11,.0f}  {r['note']}")


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
