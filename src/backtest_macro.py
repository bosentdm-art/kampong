"""Do macro variables improve the long-run growth rule? Honest out-of-sample test.

Starting point: the chosen rule's block-level errors from backtest_forecast.py (backtest_errors.pkl).
Adding a predictor changes the assumed annual growth by k x z (z = the predictor, standardised across
start years). That shifts every error by -k x z x h, so no refitting of lease curves is needed.

For each predictor: k is fitted on all start years EXCEPT one and scored on the one left out
(leave-one-origin-out), then compared with k = 0. A shuffle test shows how often a randomly
re-ordered predictor would have gained as much. With ~10 start years, expect most to fail.

Predictors use only information available at the origin (macro values from the previous calendar
year; HDB prices up to the origin month).
"""
import numpy as np
import pandas as pd

from backtest_forecast import ORIGINS, SCORE_HORIZONS
from geocode_blocks import ROOT

PROC = ROOT / "data/processed"
MACRO = ROOT / "data/raw/macro"
RULE, W = "4% -0.25x(trail-4%)", 0.5
KS = np.linspace(-0.03, 0.03, 25)  # change in annual log growth per 1 std of the predictor
rng = np.random.default_rng(0)


def series(name):
    d = pd.read_csv(MACRO / f"{name}.csv")
    d["date"] = d["date"].astype(str)
    return d


def annual(name):
    d = series(name)
    return pd.Series(d["value"].astype(float).values, index=d["date"].str[:4].astype(int))


def hdb_median_ppsm():
    d = pd.read_parquet(PROC / "resale_features.parquet", columns=["sale_year", "sale_month", "price_per_sqm"])
    d["m"] = (d["sale_year"] - 1990) * 12 + d["sale_month"] - 1
    out = {}
    for y in range(1990, 2027):
        e = (y - 1990) * 12 + 8  # September
        out[y] = d.loc[(d["m"] > e - 12) & (d["m"] <= e), "price_per_sqm"].median()
    return pd.Series(out)


def predictors():
    p = hdb_median_ppsm()
    infl, gdp, pop = annual("inflation"), annual("gdp_growth"), annual("population")
    lend, rreal, gdppc = annual("lending_rate"), annual("real_rate"), annual("gdp_pc_real")
    cpi = pd.Series(100.0, index=range(1985, 2026))
    for y in range(1986, 2026):
        cpi[y] = cpi[y - 1] * (1 + infl[y] / 100)
    real_p = pd.Series({y: p[y] / cpi[y - 1] for y in range(1990, 2026)})
    price_to_income = pd.Series({y: np.log(real_p[y] / gdppc[y - 1]) for y in range(1990, 2026)})
    bis = series("bis_real_property")
    bis["date"] = pd.to_datetime(bis["date"])
    bis = bis.set_index("date")["value"].astype(float)

    def gap(s, t0, log=True, lo=5, hi=10):
        hist = [s[t0 - k] for k in range(1, hi + 1) if (t0 - k) in s.index]
        return np.nan if len(hist) < lo else (np.log(s[t0]) if log else s[t0]) - np.mean(np.log(hist) if log else hist)

    rows = []
    for t0 in ORIGINS:
        bq = bis[bis.index <= pd.Timestamp(f"{t0}-01-01")]
        bhist = np.log(bq.iloc[-41:-1]) if len(bq) >= 21 else None
        rows.append(dict(
            origin=t0,
            real_price_gap=gap(real_p, t0),
            price_to_income_gap=price_to_income[t0] - np.mean([price_to_income[t0 - k] for k in range(1, 11) if (t0 - k) >= 1990])
            if t0 - 1990 >= 5 else np.nan,
            real_rate=rreal.get(t0 - 1, np.nan),
            lending_rate_change_3y=lend.get(t0 - 1, np.nan) - lend.get(t0 - 4, np.nan),
            gdp_growth_3y=np.mean([gdp[y] for y in range(t0 - 3, t0)]),
            population_growth_3y=(pop[t0 - 1] / pop[t0 - 4]) ** (1 / 3) - 1,
            bis_property_gap=float(np.log(bq.iloc[-1]) - bhist.mean()) if bhist is not None else np.nan))
    return pd.DataFrame(rows).set_index("origin")


def main():
    errors = pd.read_pickle(PROC / "backtest_errors.pkl")
    cells = {o: [(h, errors[(o, h, RULE, W)]) for h in SCORE_HORIZONS if (o, h, RULE, W) in errors] for o in ORIGINS}

    def score(o, k, z):
        return float(np.mean([np.abs(e - k * z * h).mean() for h, e in cells[o]]))

    pred = predictors()
    print("predictor values by start year (before standardising):")
    print(pred.round(3).to_string())
    print()

    def loo(z, origins):
        """Leave-one-origin-out mean score with k fitted on the others; also returns the fitted ks."""
        held, ks = [], []
        for o in origins:
            rest = [r for r in origins if r != o]
            k = KS[np.argmin([np.mean([score(r, kk, z[r]) for r in rest]) for kk in KS])]
            held.append(score(o, k, z[o]))
            ks.append(k)
        return float(np.mean(held)), ks

    print(f"{'predictor':24} {'n':>3} {'baseline':>9} {'with it':>9} {'change':>8} {'shuffle p':>10}  fitted k per fold (min..max)")
    for col in pred.columns:
        v = pred[col].dropna()
        origins = [o for o in ORIGINS if o in v.index and cells[o]]
        if len(origins) < 6:
            continue
        z = ((v - v.mean()) / v.std()).to_dict()
        base = float(np.mean([score(o, 0.0, 0.0) for o in origins]))
        with_it, ks = loo(z, origins)
        beat = 0
        for _ in range(150):
            perm = dict(zip(origins, rng.permutation([z[o] for o in origins])))
            beat += loo(perm, origins)[0] <= with_it
        print(f"{col:24} {len(origins):>3} {base:>9.4f} {with_it:>9.4f} {with_it / base - 1:>+8.1%} {beat / 150:>10.2f}  "
              f"{min(ks):+.3f}..{max(ks):+.3f}")


if __name__ == "__main__":
    main()
