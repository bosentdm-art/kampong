"""Can the condo model do better with more from each project's own sales?  python src/experiments_condo2.py

Same holdout and recipe as train_condo.py (last six complete months of resales; 255 leaves, 2,000 rounds, 24-month
half-life), one seed per line for speed. Each line adds one group of features to the production set:

  floor comps   same-project sales in the 24 months before, on floors within 6 of this unit (their latest 12,
                moved to last month's market), and how many there were
  size comps    the same for units within 20% of this unit's size
  developer     the project's median new-sale price per sqm before this sale (market-adjusted), and whether the
                developer was still selling there in the 6 months before (resales compete with new units)
  shape         how high the unit sits in its project (floor over the highest floor ever recorded there, which is
                the building, not the future) and its size against the project's median unit size

Prints only; a group earns a place in condo_features.py only if it beats the baseline by more than seed noise.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import condo_features as cf
import train_condo as tc


def extra_history(d):
    market = cf.market_index(d)
    adj = (d["lp"] - d["t"].map(market)).to_numpy()
    t_all, f_all, a_all, new_all = d["t"].to_numpy(), d["floor_mid"].to_numpy(), d["area"].to_numpy(), d["is_new"].to_numpy()
    n = len(d)
    out = {k: np.full(n, np.nan) for k in ("comps_floor_lp", "n_floor_24m", "comps_size_lp", "n_size_24m", "dev_lp", "dev_selling", "rel_floor", "rel_size")}
    for _, idx in d.groupby("project", observed=True).indices.items():
        t, f, a, nw, ad = t_all[idx], f_all[idx], a_all[idx], new_all[idx], adj[idx]
        top = np.nanmax(f) if np.isfinite(f).any() else np.nan
        med_area = np.nanmedian(a)
        for j, i in enumerate(idx):
            lo, hi = np.searchsorted(t, t[j] - 24), np.searchsorted(t, t[j])
            mk = market.get(t[j] - 1, market.iloc[0])
            if hi > lo:
                w = slice(lo, hi)
                fl = np.abs(f[w] - f[j]) <= 6
                sz = np.abs(a[w] / a[j] - 1) <= 0.2
                out["n_floor_24m"][i], out["n_size_24m"][i] = fl.sum(), sz.sum()
                if fl.any():
                    out["comps_floor_lp"][i] = np.median(ad[w][fl][-12:]) + mk
                if sz.any():
                    out["comps_size_lp"][i] = np.median(ad[w][sz][-12:]) + mk
                dv = nw[:hi] == 1
                if dv.any():
                    out["dev_lp"][i] = np.median(ad[:hi][dv][-30:]) + mk
                out["dev_selling"][i] = float((nw[np.searchsorted(t, t[j] - 6):hi] == 1).any())
            else:
                out["n_floor_24m"][i] = out["n_size_24m"][i] = 0
                out["dev_selling"][i] = 0.0
            out["rel_floor"][i] = f[j] / top if top and top > 0 else np.nan
            out["rel_size"][i] = a[j] / med_area if med_area else np.nan
    return d.assign(**out)


GROUPS = {"floor comps": ["comps_floor_lp", "n_floor_24m"], "size comps": ["comps_size_lp", "n_size_24m"],
          "developer": ["dev_lp", "dev_selling"], "shape": ["rel_floor", "rel_size"]}


def run(name, feats, tr, te, seed=0):
    w = 0.5 ** ((tr["t"].max() - tr["t"]) / tc.HALF_LIFE)
    t0 = time.time()
    p = lgb.LGBMRegressor(random_state=seed, **tc.PARAMS).fit(tr[feats], tr["lp"], sample_weight=w).predict(te[feats])
    e = np.abs(np.exp(p - te["lp"]) - 1)
    thin = te["n_proj_24m"] < 3
    print(f"{name:34} median {e.median() * 100:.3f}%  mean {e.mean() * 100:.2f}%  within10 {(e < .1).mean() * 100:.1f}%  "
          f"thin projects {e[thin].median() * 100:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    return e


if __name__ == "__main__":
    d = cf.as_categories(extra_history(cf.load()))
    last = cf.latest_complete_month(d)
    d = d[d["sale_date"] <= last]
    cut = last - pd.DateOffset(months=tc.HOLDOUT_MONTHS)
    tr, te = d[d["sale_date"] <= cut], d[(d["sale_date"] > cut) & (d["is_new"] == 0)]
    print(f"train to {cut:%b %Y} ({len(tr):,} sales); test {len(te):,} resales to {last:%b %Y}\n", flush=True)
    base = run("baseline (production features)", tc.FEATS, tr, te)
    run("baseline, another seed (noise check)", tc.FEATS, tr, te, seed=1)
    for g, cols in GROUPS.items():
        run("+ " + g, tc.FEATS + cols, tr, te)
    allx = run("+ all four groups", tc.FEATS + sum(GROUPS.values(), []), tr, te)
    run("+ all four, another seed", tc.FEATS + sum(GROUPS.values(), []), tr, te, seed=1)
