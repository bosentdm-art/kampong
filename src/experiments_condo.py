"""Which recipe for the condo model?  python src/experiments_condo.py

Holdout: the last six complete months of resales and sub-sales. Each line is one change from the line before;
the recipe that train_condo.py uses is the best one here. Prints only.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import condo_features as cf

d = cf.as_categories(cf.load())
last = cf.latest_complete_month(d)
d = d[d["sale_date"] <= last]
cut = last - pd.DateOffset(months=6)
tr_all, te = d[d["sale_date"] <= cut], d[(d["sale_date"] > cut) & (d["is_new"] == 0)]
print(f"train to {cut:%b %Y} ({len(tr_all):,} sales, {int((tr_all['is_new'] == 0).sum()):,} resales); test {len(te):,} resales to {last:%b %Y}\n")


def run(name, feats, train, weights=None, seeds=(0,), **kw):
    params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=5, reg_lambda=1.0, verbose=-1)
    params.update(kw)
    p = np.mean([lgb.LGBMRegressor(random_state=s, **params).fit(train[feats], train["lp"], sample_weight=weights).predict(te[feats]) for s in seeds], axis=0)
    e = np.abs(np.exp(p - te["lp"]) - 1)
    thin = te["n_proj_24m"] < 3
    print(f"{name:58} median {e.median() * 100:.2f}%  mean {e.mean() * 100:.2f}%  within10 {(e < .1).mean() * 100:.1f}%  thin projects {e[thin].median() * 100:.2f}%", flush=True)
    return e.median()


base = cf.CATS + [c for c in cf.NUM if c != "is_new"]
resale = tr_all[tr_all["is_new"] == 0]
run("A  resales only, no history", base, resale)
run("B  + new sales in training (flagged)", base + ["is_new"], tr_all)
full = base + ["is_new"] + cf.HIST
run("C  + same-project history features", full, tr_all)
w = 0.5 ** ((tr_all["t"].max() - tr_all["t"]) / 24)
run("D  + recent sales weigh more (24-month half-life)", full, tr_all, weights=w)
w36 = 0.5 ** ((tr_all["t"].max() - tr_all["t"]) / 36)
run("D2 36-month half-life", full, tr_all, weights=w36)
run("E  D with 127 leaves", full, tr_all, weights=w, num_leaves=127)
run("F  D with 31 leaves, min 20 per leaf", full, tr_all, weights=w, num_leaves=31, min_child_samples=20)
run("G  D with 3000 rounds at 0.02", full, tr_all, weights=w, n_estimators=3000, learning_rate=0.02)
run("H  D with 800 rounds", full, tr_all, weights=w, n_estimators=800)
run("I  D without the project name (history only)", [c for c in full if c != "project"], tr_all, weights=w)
run("J  D, three seeds averaged", full, tr_all, weights=w, seeds=(0, 1, 2))
