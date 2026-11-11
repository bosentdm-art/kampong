"""The condo model, start to finish: python src/train_condo.py   (run download_ura.py first)

Same method as the HDB model, on URA sales of condominiums, apartments and executive condominiums:

  1. Holdout check   train through six months before the latest complete month, test on those six months of resales.
  2. Walk forward    for each 4-month chunk from mid-2023, train on earlier sales only and predict the chunk, so the
                     errors are what they would have been in real use (kept in data/processed/condo_oos/).
  3. Ranges          one quantile model per level learns how far off we tend to be for a unit like this one, from
                     those errors; widened by `scale` until the 80% range really holds 80% of later sales.
                     The width of the 80% range sets the reliability badge.
  4. Production      retrain on everything (three seeds averaged) and save the models and a table of projects.

Recipe (from experiments_condo.py): LightGBM on log price per sqm, 255 leaves, 2,000 rounds at 0.03, recent sales
weighted more (24-month half-life), new sales kept in training with a flag. Bigger trees were the only change
that clearly helped; the others moved the holdout error by under 0.1 points.

Outputs: models/condo_prod_*.txt, models/condo_range_q*.txt, models/condo.json, data/processed/condo_projects.parquet
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import condo_features as cf
from geocode_blocks import ROOT

FEATS = cf.CATS + cf.NUM + cf.HIST
PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=255, min_child_samples=10, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, cat_smooth=5, reg_lambda=1.0, verbose=-1)
SEEDS = (0, 1, 2)
HALF_LIFE = 24
HOLDOUT_MONTHS = 6
OOS_START = pd.Timestamp("2023-07-01")
CHUNK_MONTHS = 4
OOS = ROOT / "data/processed/condo_oos"
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
RANGE_CATS = ["property_type", "market_segment"]
RANGE_FEATS = ["pred_lp", "area", "floor_mid", "freehold", "lease_left", "dist_cbd", "n_proj_24m", "n_proj_12m", "proj_gap"] + RANGE_CATS
RANGE_PARAMS = dict(objective="quantile", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=150,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
RANGE_VAL_MONTHS = 8
BADGE_HIGH, BADGE_MEDIUM = 0.17, 0.22  # width of the 80% range in log units (0.17 is about +-8.5%)


def fit(train, seeds=SEEDS):
    w = 0.5 ** ((train["t"].max() - train["t"]) / HALF_LIFE)
    return [lgb.LGBMRegressor(random_state=s, **PARAMS).fit(train[FEATS], train["lp"], sample_weight=w) for s in seeds]


def predict(models, d):
    return np.mean([m.predict(d[FEATS]) for m in models], axis=0)


def report(name, actual_lp, pred_lp):
    r = np.exp(pred_lp - actual_lp) - 1
    e = np.abs(r)
    print(f"{name:34} median error {e.median() * 100:.2f}%  mean {e.mean() * 100:.2f}%  within 10% {(e < .1).mean() * 100:.1f}%  bias {r.median() * 100:+.2f}%  n={len(e):,}", flush=True)
    return float(e.median()), float(e.mean())


def holdout(d, last):
    cut = last - pd.DateOffset(months=HOLDOUT_MONTHS)
    tr, te = d[d["sale_date"] <= cut], d[(d["sale_date"] > cut) & (d["is_new"] == 0)]
    p = predict(fit(tr), te)
    print(f"\nHOLDOUT: trained to {cut:%b %Y}, tested on resales to {last:%b %Y}")
    out = report("all", te["lp"], p)
    for name, g in te.assign(p=p).groupby("property_type", observed=True):
        report("  " + name, g["lp"], g["p"])
    for name, mask in [("  projects with 3+ sales in 24 months", te["n_proj_24m"] >= 3), ("  projects with fewer", te["n_proj_24m"] < 3)]:
        report(name, te["lp"][mask], p[mask.to_numpy()])
    base = te["comps_lp"].notna()
    report("  baseline: same-project sales", te["lp"][base], te["comps_lp"][base])
    return out


def walk_forward(d, last):
    """Out-of-sample predictions chunk by chunk. A chunk is recomputed only if it is missing or is the newest one."""
    OOS.mkdir(parents=True, exist_ok=True)
    starts = pd.date_range(OOS_START, last, freq=f"{CHUNK_MONTHS}MS")
    frames = []
    for i, s in enumerate(starts):
        e = min(s + pd.DateOffset(months=CHUNK_MONTHS) - pd.DateOffset(days=1), last)
        f = OOS / f"chunk_{s:%Y%m}_to_{e:%Y%m}.parquet"
        if not f.exists() or i == len(starts) - 1:
            for old in OOS.glob(f"chunk_{s:%Y%m}_to_*.parquet"):
                old.unlink()
            tr, te = d[d["sale_date"] < s], d[(d["sale_date"] >= s) & (d["sale_date"] <= e) & (d["is_new"] == 0)]
            te = te.assign(pred_lp=predict(fit(tr, seeds=(0,)), te))
            te.drop(columns=["contract", "nett_price"], errors="ignore").to_parquet(f, index=False)
            print(f"  chunk {s:%b %Y} to {e:%b %Y}: {len(te):,} resales", flush=True)
        frames.append(pd.read_parquet(f))
    o = pd.concat(frames, ignore_index=True)
    report("walk-forward, all chunks", o["lp"], o["pred_lp"])
    return o


def range_frame(o):
    o = o.copy()
    o["proj_gap"] = o["comps_lp"] - o["pred_lp"]
    o["e"] = o["lp"] - o["pred_lp"]
    for c in RANGE_CATS:
        o[c] = pd.Categorical(o[c].astype(str))
    return o


def fit_range(o):
    return {q: lgb.LGBMRegressor(alpha=q, **RANGE_PARAMS).fit(o[RANGE_FEATS], o["e"]) for q in QUANTILES}


def quantile_matrix(models, o):
    return np.sort(np.column_stack([models[q].predict(o[RANGE_FEATS]) for q in QUANTILES]), axis=1)


def scaled(q, scale):
    med = q[:, QUANTILES.index(0.50)][:, None]
    return med + scale * (q - med)


def ranges(o, last):
    o = range_frame(o)
    val_start = last - pd.DateOffset(months=RANGE_VAL_MONTHS - 1)
    tr, va = o[o["sale_date"] < val_start], o[o["sale_date"] >= val_start]
    q = quantile_matrix(fit_range(tr), va)
    e, i = va["e"].to_numpy(), QUANTILES.index
    cover = lambda m, a, b: float(((e >= m[:, i(a)]) & (e <= m[:, i(b)])).mean())
    scale = next((s for s in np.arange(1.0, 2.01, 0.02) if cover(scaled(q, s), 0.10, 0.90) >= 0.80), 2.0)
    m = scaled(q, scale)
    width = m[:, i(0.90)] - m[:, i(0.10)]
    badge = np.where(width <= BADGE_HIGH, "High", np.where(width <= BADGE_MEDIUM, "Medium", "Low"))
    err = np.abs(np.exp(-e) - 1)
    print(f"\nRANGES: fitted on {len(tr):,} walk-forward sales, checked on {len(va):,} from {val_start:%b %Y}")
    print(f"  scale {scale:.2f}: 80% range holds {cover(m, .10, .90) * 100:.1f}%, 50% range holds {cover(m, .25, .75) * 100:.1f}% (unscaled 80%: {cover(q, .10, .90) * 100:.1f}%)")
    print("  80% range width percentiles (log units):", {p: round(float(np.quantile(width, p)), 3) for p in (0.1, 0.25, 0.5, 0.75, 0.9)})
    t = pd.DataFrame(dict(badge=badge, err=err, inside=(e >= m[:, i(0.10)]) & (e <= m[:, i(0.90)])))
    print(t.groupby("badge").agg(share=("err", lambda x: len(x) / len(t)), median_error=("err", "median"), within10=("err", lambda x: (x < .1).mean()),
                                 range_holds=("inside", "mean")).reindex(["High", "Medium", "Low"]).round(3).to_string())
    return fit_range(o), dict(scale=float(scale), coverage80=cover(m, .10, .90), coverage50=cover(m, .25, .75), rows=int(len(va))), {c: list(o[c].cat.categories) for c in RANGE_CATS}


def project_table(d, last):
    """One row per project: what the model needs to price a unit there today, plus what the page shows."""
    market = cf.market_index(d)
    now_t = int(cf.month_index(pd.Series([last])).iloc[0]) + 1  # pricing "next month": every sale on file counts as history
    rows = []
    for name, g in d.groupby("project", observed=True):
        r, last_row = g[g["is_new"] == 0], g.iloc[-1]
        h = g[g["t"] >= now_t - 24]
        adj = (h["lp"] - h["t"].map(market)).to_numpy()[-12:]
        rows.append(dict(project=name, street=last_row["street"], property_type=g["property_type"].mode().iloc[0], district=last_row["district"],
                         market_segment=last_row["market_segment"], tenure=g["tenure"].mode().iloc[0], x=last_row["x"], y=last_row["y"],
                         typical_area=float(g["area"].median()), top_floor=float(g["floor_mid"].max()), n_sales=int(len(g)), n_resales=int(len(r)),
                         n_proj_24m=float(len(h)), n_proj_12m=float((g["t"] >= now_t - 12).sum()),
                         comps_lp=float(np.median(adj) + market.iloc[-1]) if len(adj) else np.nan))
    return pd.DataFrame(rows)


def main():
    d = cf.load()
    last = cf.latest_complete_month(d)
    d = cf.as_categories(d[d["sale_date"] <= last])
    print(f"{len(d):,} sales ({int((d['is_new'] == 0).sum()):,} resales) in {d['project'].nunique():,} projects, to {last:%b %Y}")
    med, mean = holdout(d, last)
    print("\nWALK FORWARD")
    range_models, val, range_levels = ranges(walk_forward(d, last), last)

    print("\nPRODUCTION: training on everything", flush=True)
    (ROOT / "models").mkdir(exist_ok=True)
    for i, m in enumerate(fit(d)):
        m.booster_.save_model(str(ROOT / f"models/condo_prod_{i}.txt"))
    for q, m in range_models.items():
        m.booster_.save_model(str(ROOT / f"models/condo_range_q{int(q * 100):02d}.txt"))
    project_table(d, last).to_parquet(ROOT / "data/processed/condo_projects.parquet", index=False)
    meta = dict(trained_through=f"{last:%Y-%m}", rows=int(len(d)), resales=int((d["is_new"] == 0).sum()), projects=int(d["project"].nunique()), models=len(SEEDS),
                holdout_median_error=round(med, 4), holdout_mean_error=round(mean, 4), features=FEATS, levels={c: list(d[c].cat.categories) for c in cf.CATS},
                quantiles=QUANTILES, range_features=RANGE_FEATS, range_levels=range_levels, scale=val["scale"], badge_high=BADGE_HIGH, badge_medium=BADGE_MEDIUM, validation=val)
    (ROOT / "models/condo.json").write_text(json.dumps(meta), encoding="utf-8")
    print("saved models/condo_prod_*.txt, models/condo_range_q*.txt, models/condo.json, data/processed/condo_projects.parquet")


if __name__ == "__main__":
    sys.exit(main())
