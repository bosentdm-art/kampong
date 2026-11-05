"""Flat-specific ranges and a bias-corrected fair value, learned from out-of-sample errors.

Target: e = log(actual / predicted) from the walk-forward chunks (oos_residuals.py). One LightGBM quantile
model per level in QUANTILES predicts the distribution of e for a flat, from its features plus data-richness
signals (sales in the block in the last 24 months, gap to recent comparables).

  fair value   = the point estimate (a bias correction from q50 was tested: validation error got WORSE)
  80% range    = fair value widened by `scale` around q50 so coverage on later months is really 80%
  percentiles  = interpolate the scaled quantiles -> where an asking price sits among comparable sales

Validation: train on chunks before VAL_START, test after. Prints coverage and error by segment against the old
single global band. Then retrains on everything and saves models/range_q*.txt and models/range.json.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from geocode_blocks import ROOT

OOS = ROOT / "data/processed/oos"
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
VAL_START = pd.Timestamp("2025-07-01")
CATS = ["flat_type", "town", "flat_model"]
RANGE_FEATS = ["pred_log", "floor_area_sqm", "storey_mid", "remaining_lease_years", "lease_commence_date", "dist_cbd",
               "dist_mrt", "n_block_24m", "block_gap"] + CATS
BADGE_HIGH, BADGE_MEDIUM = 0.12, 0.15  # 80% band width in log units (0.12 is about +-6%)
PARAMS = dict(objective="quantile", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)


def load():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OOS.glob("chunk_*.parquet"))], ignore_index=True)
    return prepare(d).assign(e=lambda x: np.log(x["resale_price"] / x["pred"]))


def prepare(d):
    """Add the derived columns the range model uses (also used for a single flat at inference time)."""
    d = d.copy()
    d["pred_log"] = np.log(d["pred"])
    d["block_gap"] = np.log(d["comps_est"] / d["pred"])
    for c in CATS:
        d[c] = d[c].astype(str).astype("category")
    return d


def fit(d):
    return {q: lgb.LGBMRegressor(alpha=q, **PARAMS).fit(d[RANGE_FEATS], d["e"]) for q in QUANTILES}


def quantile_matrix(models, d):
    m = np.column_stack([models[q].predict(d[RANGE_FEATS]) for q in QUANTILES])
    return np.sort(m, axis=1)  # undo any crossing between quantile levels


def scaled(q, scale):
    """Widen deviations from the median by `scale`."""
    med = q[:, QUANTILES.index(0.50)][:, None]
    return med + scale * (q - med)


def band(q, level):
    i = {"80": (QUANTILES.index(0.10), QUANTILES.index(0.90)), "50": (QUANTILES.index(0.25), QUANTILES.index(0.75))}[level]
    return q[:, i[0]], q[:, i[1]]


def main():
    d = load()
    train, val = d[d["sale_date"] < VAL_START], d[d["sale_date"] >= VAL_START].copy()
    print(f"out-of-sample rows: {len(d)} (train {len(train)} before {VAL_START:%Y-%m}, validate {len(val)})")

    models = fit(train)
    q = quantile_matrix(models, val)

    # point accuracy: raw vs bias-corrected
    med = q[:, QUANTILES.index(0.50)]
    ape_raw = (val["pred"] - val["resale_price"]).abs() / val["resale_price"]
    ape_adj = (val["pred"] * np.exp(med) - val["resale_price"]).abs() / val["resale_price"]
    print(f"\nPOINT ACCURACY on validation: raw median {ape_raw.median():.2%} mean {ape_raw.mean():.2%} within10 {(ape_raw <= .1).mean():.1%}"
          f"  ->  bias-corrected median {ape_adj.median():.2%} mean {ape_adj.mean():.2%} within10 {(ape_adj <= .1).mean():.1%}")

    # choose the widening so the validation 80% band covers 80%
    def cover(scale, level="80"):
        lo, hi = band(scaled(q, scale), level)
        return float(((val["e"] >= lo) & (val["e"] <= hi)).mean())
    grid = np.arange(0.6, 2.01, 0.02)
    scale = float(grid[np.argmin([abs(cover(s) - 0.80) for s in grid])])
    print(f"\nwidening factor {scale:.2f}: 80% band covers {cover(scale):.1%}, 50% band covers {cover(scale, '50'):.1%}")

    lo, hi = band(scaled(q, scale), "80")
    val["lo"], val["hi"], val["width"] = lo, hi, hi - lo
    val["inside"] = (val["e"] >= val["lo"]) & (val["e"] <= val["hi"])
    g10, g90 = train["e"].quantile([0.10, 0.90])  # the old single band, from the same training rows
    val["g_inside"] = (val["e"] >= g10) & (val["e"] <= g90)
    val["ape_adj"] = ape_adj.values
    print(f"old single band: {np.exp(g10) - 1:+.1%} to {np.exp(g90) - 1:+.1%}, covers {val['g_inside'].mean():.1%} overall")

    def seg(name, key):
        g = val.groupby(key, observed=True).agg(n=("e", "size"), old_band_cover=("g_inside", "mean"), new_band_cover=("inside", "mean"),
                                                old_width=("e", lambda x: g90 - g10), new_width=("width", "mean")).round(3)
        print(f"--- {name}"); print(g[g.n >= 150].to_string()); print()
    print()
    seg("remaining lease", pd.cut(val["remaining_lease_years"], [0, 55, 65, 75, 85, 100]))
    seg("price band", pd.qcut(val["resale_price"], 5))
    seg("flat type", val["flat_type"])

    print(f"80% band width percentiles (log units): {val['width'].quantile([.1, .25, .5, .75, .9]).round(3).to_dict()}")
    val["badge"] = np.where(val["width"] <= BADGE_HIGH, "High", np.where(val["width"] <= BADGE_MEDIUM, "Medium", "Low"))
    b = val.groupby("badge").agg(share=("e", lambda x: len(x) / len(val)), median_error=("ape_adj", "median"),
                                 within10=("ape_adj", lambda x: (x <= .1).mean()), band_cover=("inside", "mean")).round(3)
    print("\nRELIABILITY BADGE (by band width) vs realised error on validation:")
    print(b.reindex(["High", "Medium", "Low"]).to_string())

    # final models on all out-of-sample rows
    final = fit(d)
    (ROOT / "models").mkdir(exist_ok=True)
    for qq, m in final.items():
        m.booster_.save_model(str(ROOT / f"models/range_q{int(qq * 100):02d}.txt"))
    meta = dict(quantiles=QUANTILES, features=RANGE_FEATS, scale=scale, badge_high=BADGE_HIGH, badge_medium=BADGE_MEDIUM,
                trained_rows=int(len(d)), validation=dict(rows=int(len(val)), coverage80=cover(scale), coverage50=cover(scale, "50"),
                                                          median_error_raw=float(ape_raw.median()), median_error_adjusted=float(ape_adj.median())))
    (ROOT / "models/range.json").write_text(json.dumps(meta, indent=2))
    print("\nsaved models/range_q*.txt and models/range.json")


if __name__ == "__main__":
    main()
