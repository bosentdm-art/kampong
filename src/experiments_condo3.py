"""Is the gain from experiments_condo2.py real, or one lucky six months?  python src/experiments_condo3.py

Repeats the comparison on the two six-month periods before the usual holdout (each trained only on sales before it).
Prints only.
"""
import pandas as pd

import condo_features as cf
import experiments_condo2 as x
import train_condo as tc

d = cf.as_categories(x.extra_history(cf.load()))
last = cf.latest_complete_month(d)
d = d[d["sale_date"] <= last]
CONFIGS = {"baseline": tc.FEATS, "all four": tc.FEATS + sum(x.GROUPS.values(), []),
           "all but floor comps": tc.FEATS + x.GROUPS["size comps"] + x.GROUPS["developer"] + x.GROUPS["shape"]}
for back in (6, 12):
    end = last - pd.DateOffset(months=back)
    cut = end - pd.DateOffset(months=6)
    tr, te = d[d["sale_date"] <= cut], d[(d["sale_date"] > cut) & (d["sale_date"] <= end) & (d["is_new"] == 0)]
    print(f"\nperiod {cut + pd.DateOffset(months=1):%b %Y} to {end:%b %Y}: train {len(tr):,}, test {len(te):,} resales", flush=True)
    for name, feats in CONFIGS.items():
        x.run(name, feats, tr, te)
