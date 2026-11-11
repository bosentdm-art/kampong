"""First look at the URA data: is a condo model worth building?  python src/experiments_ura.py

Holds out the last six months of condo, apartment and EC resales and compares two estimates of each sale:
  baseline  the median price per sqm in the same project over the 12 months before, moved by the market since
  model     a quick LightGBM on log price per sqm (project, size, floor, tenure, district, month)
Nothing is saved; it prints the errors. Landed homes get the baseline only (by street), as a first reading.
"""
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from geocode_blocks import ROOT

d = pd.read_parquet(ROOT / "data/processed/ura_transactions.parquet")
d = d[d["sale_type"].isin(["resale", "sub sale"]) & (d["units"].fillna(1) == 1)].copy()
d["landed"] = d["area_type"].eq("Land")
d["t"] = (d["sale_date"].dt.year - 2021) * 12 + d["sale_date"].dt.month
d["floor_mid"] = d["floor_range"].map(lambda s: np.mean([int(x) for x in re.findall(r"\d+", s)]) * (-1 if str(s).startswith("B") else 1) if re.search(r"\d", str(s)) else np.nan)
d["freehold"] = d["tenure"].str.contains("Freehold|999|9999", case=False, na=False)
d["lease_start"] = pd.to_numeric(d["tenure"].str.extract(r"(\d{4})\s*$")[0], errors="coerce")
d["lease_left"] = np.where(d["freehold"], 999.0, 99 - (d["sale_date"].dt.year - d["lease_start"]))
d["lp"] = np.log(d["price_per_sqm"])
cut = d["sale_date"].max() - pd.DateOffset(months=6)
print(f"data to {d['sale_date'].max():%b %Y}; test = sales after {cut:%b %Y}\n")


def err(actual, pred):
    e = (np.exp(pred - actual) - 1).abs()
    return f"median error {e.median() * 100:.1f}%, within 10%: {(e < 0.10).mean() * 100:.0f}%, n={len(e):,}"


def baseline(g, key):
    """Median log price per sqm for the same key in the 12 months before each test sale, plus the market's move since then."""
    market = g.groupby("t")["lp"].median().rolling(3, min_periods=1).mean()
    test, out = g[g["sale_date"] > cut], []
    hist = {k: v[["t", "lp"]].to_numpy() for k, v in g.groupby(key)}
    for k, t in zip(test[key], test["t"]):
        h = hist[k]
        h = h[(h[:, 0] < t) & (h[:, 0] >= t - 12)]
        out.append(np.median(h[:, 1] + market.get(t - 1, market.iloc[-1]) - market.reindex(h[:, 0]).to_numpy()) if len(h) >= 2 else np.nan)
    return test, np.array(out)


c = d[~d["landed"]].copy()
test, b = baseline(c, "project")
ok = ~np.isnan(b)
print("CONDOS, APARTMENTS, ECs (resale)")
print(f"  baseline, same project : {err(test['lp'][ok], b[ok])}  (available for {ok.mean() * 100:.0f}% of sales)")

feats = ["project", "property_type", "district", "market_segment", "area", "floor_mid", "freehold", "lease_start", "lease_left", "t", "x", "y"]
for col in ("project", "property_type", "district", "market_segment"):
    c[col] = c[col].astype("category")
tr, te = c[c["sale_date"] <= cut], c[c["sale_date"] > cut]
m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      cat_smooth=5, verbose=-1).fit(tr[feats], tr["lp"])
p = m.predict(te[feats])
print(f"  quick model, all sales : {err(te['lp'], p)}")
print(f"  quick model, same sales as the baseline : {err(te['lp'][ok], p[ok])}")
print(f"  quick model, projects with no recent sales : {err(te['lp'][~ok], p[~ok])}")
for name, g in te.assign(p=p).groupby("property_type", observed=True):
    print(f"    {name:22} {err(g['lp'], g['p'])}")

land = d[d["landed"]].copy()
test, b = baseline(land, "street")
ok = ~np.isnan(b)
print("\nLANDED (resale, price per sqm of land)")
print(f"  baseline, same street  : {err(test['lp'][ok], b[ok])}  (available for {ok.mean() * 100:.0f}% of sales)")
