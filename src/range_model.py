"""Flat-specific ranges, learned from out-of-sample errors and centred on the recent market.

Target: e = log(actual / predicted) from the walk-forward chunks (oos_residuals.py). One LightGBM quantile model per
level in QUANTILES predicts how widely sales scatter for a flat, from its features plus data-richness signals (sales
in the block in the last 24 months, gap to recent comparables). Where the range sits is not learned: it is centred on
the median error of the last three months for flats of the same age (see recentre.py for why).

  fair value   the point estimate (moving it by the recent centre is reported below, and left out: it changes little)
  80% range    centre + spread, widened separately below and above so that 1 sale in 10 lands under it and 1 in 10 over
  percentiles  interpolate those quantiles -> where an asking price sits among comparable sales

Validation: train on chunks before VAL_START, test after, with each month's centre taken from the months before it.
Prints how many sales land on each side of the ranges, by segment. Then retrains on everything and saves
models/range_q*.txt and models/range.json. Run weekly by update.py, so the centre follows the market.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import recentre as rc
from geocode_blocks import ROOT

OOS = ROOT / "data/processed/oos"
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
VAL_START = pd.Timestamp("2025-07-01")
CATS = ["flat_type", "town", "flat_model"]
RANGE_FEATS = ["pred_log", "floor_area_sqm", "storey_mid", "remaining_lease_years", "lease_commence_date", "dist_cbd",
               "dist_mrt", "n_block_24m", "block_gap"] + CATS
# 80% band width in log units (0.11 is about +-5.5%). Lowered from 0.12 / 0.15 in Nov 2026, when re-centring made the
# bands about 8% narrower, so that each badge still means the same realised error as before.
BADGE_HIGH, BADGE_MEDIUM = 0.11, 0.14
PARAMS = dict(objective="quantile", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)


def age_band(age):
    """The kind of flat the centre is kept for. In 2026 older flats were slipping while newer ones rose."""
    return "new" if age <= 20 else "mid" if age <= 40 else "old"


def load():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OOS.glob("chunk_*.parquet"))], ignore_index=True)
    d = prepare(d).assign(e=lambda x: np.log(x["resale_price"] / x["pred"]))
    d["m"] = d["sale_date"].dt.to_period("M")
    d["band"] = [age_band(a) for a in d["sale_date"].dt.year - d["lease_commence_date"]]
    d["e_flat"] = rc.flatten(d)
    return d


def prepare(d):
    """Add the derived columns the range model uses (also used for a single flat at inference time)."""
    d = d.copy()
    d["pred_log"] = np.log(d["pred"])
    d["block_gap"] = np.log(d["comps_est"] / d["pred"])
    for c in CATS:
        d[c] = d[c].astype(str).astype("category")
    return d


def fit(d, target="e_flat"):
    return {q: lgb.LGBMRegressor(alpha=q, **PARAMS).fit(d[RANGE_FEATS], d[target]) for q in QUANTILES}


def quantile_matrix(models, d):
    m = np.column_stack([models[q].predict(d[RANGE_FEATS]) for q in QUANTILES])
    return np.sort(m, axis=1)  # undo any crossing between quantile levels


def main():
    d = load()
    is_val = (d["sale_date"] >= VAL_START).to_numpy()
    train, val = d[~is_val], d[is_val].copy()
    print(f"out-of-sample rows: {len(d)} (train {len(train)} before {VAL_START:%Y-%m}, validate {len(val)})")

    i = QUANTILES.index
    q = quantile_matrix(fit(train), val)
    e, centre = val["e"].to_numpy(), rc.trailing(d)[is_val]  # each month's centre uses only the months before it

    ape = lambda shift: (np.exp(shift - e) - 1)
    print(f"\nPOINT ACCURACY on validation: median {np.median(np.abs(ape(0))):.2%} within10 {(np.abs(ape(0)) <= .1).mean():.1%}"
          f"  (moved by the recent centre: median {np.median(np.abs(ape(centre))):.2%})")

    s_lo, s_hi = rc.solve(e, q, centre, i(0.50), i(0.10), i(0.90))
    m = rc.spread(q, centre, i(0.50), s_lo, s_hi)
    below, above = rc.sides(e, m, i(0.10), i(0.90))
    b50, a50 = rc.sides(e, m, i(0.25), i(0.75))
    print(f"\nwidening {s_lo:.2f} below the centre, {s_hi:.2f} above: {below:.1f}% of sales under the 80% range, {above:.1f}% over it; "
          f"{b50:.1f}% under the 50% range, {a50:.1f}% over it")
    half = (val["m"] < sorted(val["m"].unique())[val["m"].nunique() // 2]).to_numpy()  # is the widening stable? choose it on the first half, test on the second
    t_lo, t_hi = rc.solve(e[half], q[half], centre[half], i(0.50), i(0.10), i(0.90))
    tb, ta = rc.sides(e[~half], rc.spread(q[~half], centre[~half], i(0.50), t_lo, t_hi), i(0.10), i(0.90))
    print(f"check: widening chosen on the first half of these months ({t_lo:.2f}, {t_hi:.2f}) leaves {tb:.1f}% under and {ta:.1f}% over on the second half")

    val["lo"], val["hi"], val["width"] = m[:, i(0.10)], m[:, i(0.90)], m[:, i(0.90)] - m[:, i(0.10)]
    val["under"], val["over"] = val["e"] < val["lo"], val["e"] > val["hi"]
    val["miss"] = np.abs(ape(0))

    def seg(name, key):
        g = val.groupby(key, observed=True).agg(n=("e", "size"), under=("under", "mean"), over=("over", "mean"), width=("width", "mean")).round(3)
        print(f"--- {name}"); print(g[g.n >= 150].to_string()); print()
    print()
    seg("age of flat", val["band"])
    seg("remaining lease", pd.cut(val["remaining_lease_years"], [0, 55, 65, 75, 85, 100]))
    seg("estimate", pd.cut(val["pred"], [0, 4e5, 6e5, 8e5, 1e6, 1e8]))
    seg("flat type", val["flat_type"])

    print(f"80% band width percentiles (log units): {val['width'].quantile([.1, .25, .5, .75, .9]).round(3).to_dict()}")
    val["badge"] = np.where(val["width"] <= BADGE_HIGH, "High", np.where(val["width"] <= BADGE_MEDIUM, "Medium", "Low"))
    b = val.groupby("badge").agg(share=("e", lambda x: len(x) / len(val)), median_error=("miss", "median"), within10=("miss", lambda x: (x <= .1).mean()),
                                 under=("under", "mean"), over=("over", "mean")).round(3)
    print("\nRELIABILITY BADGE (by band width) vs realised error on validation:")
    print(b.reindex(["High", "Medium", "Low"]).to_string())

    # final models on all out-of-sample rows, with the centre as it stands after the newest month on file
    final, now = fit(d), rc.centres(d)
    (ROOT / "models").mkdir(exist_ok=True)
    for qq, model in final.items():
        model.booster_.save_model(str(ROOT / f"models/range_q{int(qq * 100):02d}.txt"))
    last = d["m"].max()
    meta = dict(quantiles=QUANTILES, features=RANGE_FEATS, scale_low=s_lo, scale_high=s_hi, centre=now, centre_months=[str(last - rc.MONTHS + 1), str(last)],
                badge_high=BADGE_HIGH, badge_medium=BADGE_MEDIUM, trained_rows=int(len(d)),
                validation=dict(rows=int(len(val)), coverage80=(100 - below - above) / 100, under80=below / 100, over80=above / 100, coverage50=(100 - b50 - a50) / 100,
                                under50=b50 / 100, over50=a50 / 100, median_error_raw=float(np.median(np.abs(ape(0)))), median_error_moved=float(np.median(np.abs(ape(centre))))))
    (ROOT / "models/range.json").write_text(json.dumps(meta, indent=2))
    print(f"\ncentre now ({meta['centre_months'][0]} to {meta['centre_months'][1]}), in log units: {({k: round(v, 4) for k, v in now.items()})}")
    print("saved models/range_q*.txt and models/range.json")


if __name__ == "__main__":
    main()
