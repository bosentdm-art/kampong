"""The condo model, start to finish: python src/train_condo.py   (run download_ura.py first)

Same method as the HDB model, on URA sales of condominiums, apartments and executive condominiums:

  1. Holdout check   train through six months before the latest complete month, test on those six months of resales.
  2. Walk forward    for each 4-month chunk from mid-2023, train on earlier sales only and predict the chunk, so the
                     errors are what they would have been in real use (kept in data/processed/condo_oos/).
  3. Ranges          one quantile model per level learns how widely sales scatter for a unit like this one, from
                     those errors. The range is centred on the last three months' median error for the same market
                     segment and widened separately below and above, so that 1 later sale in 10 lands under the
                     80% range and 1 in 10 over it (recentre.py). Its width sets the reliability badge.
                     python src/train_condo.py --ranges-only   redoes just this step from the files on disk.
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
import recentre as rc
from geocode_blocks import ROOT

FEATS = cf.CATS + cf.NUM + cf.LOC + cf.HIST
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
    o["m"] = o["sale_date"].dt.to_period("M")
    o["band"] = o["market_segment"].astype(str)  # the centre is kept per market segment: the suburbs and the core move apart
    o["e_flat"] = rc.flatten(o)
    for c in RANGE_CATS:
        o[c] = pd.Categorical(o[c].astype(str))
    return o


def fit_range(o):
    return {q: lgb.LGBMRegressor(alpha=q, **RANGE_PARAMS).fit(o[RANGE_FEATS], o["e_flat"]) for q in QUANTILES}


def quantile_matrix(models, o):
    return np.sort(np.column_stack([models[q].predict(o[RANGE_FEATS]) for q in QUANTILES]), axis=1)


def ranges(o, last):
    o = range_frame(o)
    val_start = last - pd.DateOffset(months=RANGE_VAL_MONTHS - 1)
    is_val = (o["sale_date"] >= val_start).to_numpy()
    tr, va = o[~is_val], o[is_val]
    q = quantile_matrix(fit_range(tr), va)
    e, centre, i = va["e"].to_numpy(), rc.trailing(o)[is_val], QUANTILES.index  # each month's centre uses only the months before it
    s_lo, s_hi = rc.solve(e, q, centre, i(0.50), i(0.10), i(0.90))
    m = rc.spread(q, centre, i(0.50), s_lo, s_hi)
    (under, over), (u50, o50) = rc.sides(e, m, i(0.10), i(0.90)), rc.sides(e, m, i(0.25), i(0.75))
    width = m[:, i(0.90)] - m[:, i(0.10)]
    badge = np.where(width <= BADGE_HIGH, "High", np.where(width <= BADGE_MEDIUM, "Medium", "Low"))
    err = np.abs(np.exp(-e) - 1)
    print(f"\nRANGES: fitted on {len(tr):,} walk-forward sales, checked on {len(va):,} from {val_start:%b %Y}")
    print(f"  widening {s_lo:.2f} below the centre, {s_hi:.2f} above: {under:.1f}% of sales under the 80% range, {over:.1f}% over it; {u50:.1f}% under the 50% range, {o50:.1f}% over it")
    t = pd.DataFrame(dict(segment=va["band"].to_numpy(), badge=badge, err=err, under=e < m[:, i(0.10)], over=e > m[:, i(0.90)], width=width))
    print(t.groupby("segment").agg(n=("err", "size"), under=("under", "mean"), over=("over", "mean"), width=("width", "mean")).round(3).to_string())
    print("  80% range width percentiles (log units):", {p: round(float(np.quantile(width, p)), 3) for p in (0.1, 0.25, 0.5, 0.75, 0.9)})
    print(t.groupby("badge").agg(share=("err", lambda x: len(x) / len(t)), median_error=("err", "median"), within10=("err", lambda x: (x < .1).mean()),
                                 under=("under", "mean"), over=("over", "mean")).reindex(["High", "Medium", "Low"]).round(3).to_string())
    now, newest = rc.centres(o), o["m"].max()
    print(f"  centre now ({newest - rc.MONTHS + 1} to {newest}), in log units: {({k: round(v, 4) for k, v in now.items()})}")
    val = dict(scale_low=s_lo, scale_high=s_hi, centre=now, centre_months=[str(newest - rc.MONTHS + 1), str(newest)], coverage80=(100 - under - over) / 100,
               under80=under / 100, over80=over / 100, coverage50=(100 - u50 - o50) / 100, rows=int(len(va)))
    return fit_range(o), val, {c: list(o[c].cat.categories) for c in RANGE_CATS}


def save_ranges(range_models, val, range_levels, meta):
    for q, m in range_models.items():
        m.booster_.save_model(str(ROOT / f"models/condo_range_q{int(q * 100):02d}.txt"))
    meta.pop("scale", None)  # the single widening used before November 2026
    meta.update(quantiles=QUANTILES, range_features=RANGE_FEATS, range_levels=range_levels, scale_low=val["scale_low"], scale_high=val["scale_high"], centre=val["centre"],
                centre_months=val["centre_months"], badge_high=BADGE_HIGH, badge_medium=BADGE_MEDIUM, validation={k: v for k, v in val.items() if k not in ("centre", "centre_months")})
    (ROOT / "models/condo.json").write_text(json.dumps(meta), encoding="utf-8")


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
                         market_segment=last_row["market_segment"], tenure=g["tenure"].mode().iloc[0], x=last_row["x"], y=last_row["y"], **{c: last_row[c] for c in cf.LOC},
                         typical_area=float(g["area"].median()), top_floor=float(g["floor_mid"].max()), n_sales=int(len(g)), n_resales=int(len(r)),
                         n_proj_24m=float(len(h)), n_proj_12m=float((g["t"] >= now_t - 12).sum()),
                         comps_lp=float(np.median(adj) + market.iloc[-1]) if len(adj) else np.nan))
    return pd.DataFrame(rows)


def main():
    if "--ranges-only" in sys.argv[1:]:  # the walk-forward files are already on disk: refit the ranges from them, leave the price model alone
        o = pd.concat([pd.read_parquet(f) for f in sorted(OOS.glob("chunk_*.parquet"))], ignore_index=True)
        save_ranges(*ranges(o, o["sale_date"].max()), json.loads((ROOT / "models/condo.json").read_text(encoding="utf-8")))
        print("saved models/condo_range_q*.txt and the range settings in models/condo.json")
        return
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
    project_table(d, last).to_parquet(ROOT / "data/processed/condo_projects.parquet", index=False)
    meta = dict(trained_through=f"{last:%Y-%m}", rows=int(len(d)), resales=int((d["is_new"] == 0).sum()), projects=int(d["project"].nunique()), models=len(SEEDS),
                holdout_median_error=round(med, 4), holdout_mean_error=round(mean, 4), features=FEATS, levels={c: list(d[c].cat.categories) for c in cf.CATS})
    save_ranges(range_models, val, range_levels, meta)
    print("saved models/condo_prod_*.txt, models/condo_range_q*.txt, models/condo.json, data/processed/condo_projects.parquet")


if __name__ == "__main__":
    sys.exit(main())
