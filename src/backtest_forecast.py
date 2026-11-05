"""Step 5, stage D: backtest the whole long-run forecast method and calibrate its ranges.

For each origin year (windows end in September) and horizon h:
  - base       = median price per sqm of each block + flat type in the 12 months to the origin
  - prediction = base x exp(g x h) x exp(w x lease effect), lease curve fitted ONLY on the 6 years
                 before the origin, g chosen by a growth rule, w by a lease weight
  - realised   = median price per sqm of the same block + flat type 12 months to origin + h
  - error e    = log(realised / prediction)

Rules and weights are compared on average absolute error (horizons 5 to 20 years). The chosen
combination's errors give the bias and the 10th / 90th percentile bands per horizon, and a
leave-one-origin-out check says whether those bands cover about 80% of outcomes on origins they
were not built from.

Caveats: the blocks demolished since (SERS) are not in the data, so old origins are slightly
rosy; origins overlap, so the 'independent' evidence is closer to a handful of regimes than to
thousands of blocks; and 15 / 20 year horizons exist only for the oldest origins.

Output: data/processed/forecast_calibration.json, reports/backtest_forecast_cells.csv
"""
import json

import numpy as np
import pandas as pd

from backtest import lease_curve, month_n, window
from geocode_blocks import ROOT

PROC = ROOT / "data/processed"
ORIGINS = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016]
HORIZONS = [3, 5, 8, 10, 15, 20]
SCORE_HORIZONS = [5, 8, 10, 15, 20]
MIN_SALES = 3
CONST = {"0%/yr": 0.00, "2%/yr": 0.02, "3%/yr": 0.03, "4%/yr": 0.04}
WEIGHTS = [0.0, 0.25, 0.5, 1.0]
FIXED_WEIGHT = 0.5  # all weights score within noise; 0.5 also fits the earlier cross-section test
BASE_GROWTH = 0.04
DAMPING = {"4%/yr": 0.0, "4% +0.25x(trail-4%)": 0.25, "4% -0.25x(trail-4%)": -0.25, "4% -0.50x(trail-4%)": -0.5}
TARGET_COVERAGE = 0.80


def trailing_growth(df, e0):
    """Annual log growth of the raw median price per sqm over the 5 years to the origin."""
    p1 = df.loc[(df["month_n"] > e0 - 12) & (df["month_n"] <= e0), "price_per_sqm"].median()
    p0 = df.loc[(df["month_n"] > e0 - 72) & (df["month_n"] <= e0 - 60), "price_per_sqm"].median()
    return float(np.log(p1 / p0) / 5)


def main():
    df = pd.read_parquet(PROC / "resale_features.parquet")
    df["month_n"] = (df["sale_year"] - 1990) * 12 + df["sale_month"] - 1
    df = df[df["month_n"] <= month_n(2026)]

    cells = []     # (origin, h, rule, w, abs-error mean, median error, n)
    errors = {}    # (origin, h, rule, w) -> array of e
    for t0 in ORIGINS:
        e0 = month_n(t0)
        train = df[(df["month_n"] > e0 - 72) & (df["month_n"] <= e0)]
        grid, c = lease_curve(train)
        support = float(train["remaining_lease_years"].quantile(0.01))
        rules = {f"{k}": np.log1p(v) for k, v in CONST.items()}
        tr = trailing_growth(df, e0)
        rules["trailing 5y"] = tr
        for k in (-0.5, -0.25, 0.25):  # 4% a year, nudged by how far recent growth was from 4%
            rules[f"4% {k:+.2f}x(trail-4%)"] = np.log1p(0.04) + k * (tr - np.log1p(0.04))
        w0 = window(df, e0)
        print(f"origin {t0}: {len(train)} train sales, lease support {support:.0f}y, trailing growth {rules['trailing 5y']:+.1%}/yr",
              flush=True)
        for h in HORIZONS:
            if t0 + h > 2026:
                continue
            m = w0.merge(window(df, month_n(t0 + h)), on=["block", "street_name", "flat_type"], suffixes=("0", "1"))
            m = m[(m["n0"] >= MIN_SALES) & (m["n1"] >= MIN_SALES) & (m["lease1"] >= support)]
            if len(m) < 50:
                continue
            dc = np.interp(m["lease1"], grid, c) - np.interp(m["lease0"], grid, c)
            base = np.log(m["ppsm1"] / m["ppsm0"]).to_numpy()
            for rule, g in rules.items():
                for w in WEIGHTS:
                    e = base - g * h - w * dc
                    errors[(t0, h, rule, w)] = e
                    cells.append(dict(origin=t0, h=h, rule=rule, w=w, n=len(e), mae=np.abs(e).mean(), bias=np.median(e)))

    cell = pd.DataFrame(cells)
    cell.to_csv(ROOT / "reports/backtest_forecast_cells.csv", index=False)
    pd.set_option("display.width", 200)

    sc = cell[cell["h"].isin(SCORE_HORIZONS)].groupby(["rule", "w"]).agg(mae=("mae", "mean"), bias=("bias", "mean")).reset_index()
    print()
    print("AVERAGE ABSOLUTE ERROR (log units, horizons 5-20y, each origin x horizon equal weight) and mean bias")
    print(sc.pivot(index="rule", columns="w", values="mae").round(3).to_string())
    print("bias (realised below prediction if negative):")
    print(sc.pivot(index="rule", columns="w", values="bias").round(3).to_string())
    at_w = sc[sc["w"] == FIXED_WEIGHT]
    best = at_w.loc[at_w["mae"].idxmin()]
    rule, w = best["rule"], FIXED_WEIGHT
    print(f"\nchosen: growth rule '{rule}' with lease weight {w} (mae {best['mae']:.3f}; lowest at any weight: "
          f"{sc['mae'].min():.3f})")

    band = {}
    print()
    print("CHOSEN COMBINATION by horizon (log error e = log(realised / forecast); bands widened to reach their target coverage on origins left out)")
    rng = np.random.default_rng(0)
    for h in HORIZONS:
        keys = [k for k in errors if k[1] == h and k[2] == rule and k[3] == w]
        if not keys:
            continue
        arrs = [errors[k] for k in keys]
        n = min(len(a) for a in arrs)
        samp = [rng.choice(a, n, replace=False) for a in arrs]  # equal weight per origin
        def calc(plo, phi, target):
            """Band from pooled errors, widened until leave-one-origin-out coverage reaches the target."""
            qlo, q50, qhi = np.quantile(np.concatenate(samp), [plo, 0.5, phi])

            def loo_cov(scale):
                out = []
                for i, a in enumerate(arrs):
                    olo, o50, ohi = np.quantile(np.concatenate([x for j, x in enumerate(samp) if j != i]), [plo, .5, phi])
                    out.append(((a >= o50 - scale * (o50 - olo)) & (a <= o50 + scale * (ohi - o50))).mean())
                return float(np.mean(out))

            cov0 = loo_cov(1.0)
            scale = next((x for x in np.arange(1.0, 2.001, 0.05) if loo_cov(x) >= target), 2.0)
            return dict(lo=float(q50 - scale * (q50 - qlo)), q50=float(q50), hi=float(q50 + scale * (qhi - q50)),
                        scale=float(scale), cov0=cov0, cov1=loo_cov(scale))

        wide, likely = calc(0.10, 0.90, 0.80), calc(0.25, 0.75, 0.50)
        band[h] = dict(q10=wide["lo"], q50=wide["q50"], q90=wide["hi"], scale=wide["scale"], coverage_before=wide["cov0"],
                       coverage_after=wide["cov1"], l25=likely["lo"], l75=likely["hi"], likely_scale=likely["scale"],
                       likely_coverage_before=likely["cov0"], likely_coverage_after=likely["cov1"], n_origins=len(keys))
        print(f"  {h:>2}y: origins {len(keys):>2}  median error {wide['q50']:+.3f} | 80% band x{np.exp(wide['lo']):.2f}..x{np.exp(wide['hi']):.2f}"
              f" coverage {wide['cov0']:.0%}->{wide['cov1']:.0%} | 50% band x{np.exp(likely['lo']):.2f}..x{np.exp(likely['hi']):.2f}"
              f" coverage {likely['cov0']:.0%}->{likely['cov1']:.0%}")

    (PROC / "forecast_calibration.json").write_text(json.dumps(dict(
        growth_rule=rule, base_growth=BASE_GROWTH, damping=DAMPING.get(rule), lease_weight=w,
        score_mae=float(best["mae"]), bands={str(h): v for h, v in band.items()}), indent=2))
    print("\nsaved data/processed/forecast_calibration.json")


if __name__ == "__main__":
    main()
