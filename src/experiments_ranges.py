"""Why the HDB ranges are re-centred on recent months: python src/experiments_ranges.py   (about 10 minutes)

The range models learn the spread of e = log(actual / estimate) from walk-forward errors. Until November 2026 they
learned its centre from all of 2021 onward, years in which prices mostly rose and sales landed above the estimate.
When the market went flat in 2026 the ranges stayed centred too high: on sales the models had not seen, 16% sold
below the 80% range and 4% above it (it should be 10% and 10%).

This script compares ways to centre the range, year by year from 2023, each using only what was known at the time:

  old        centre and spread both learned from all earlier errors (what we shipped before)
  shift      the old models, moved by (median error of the last 3 months - median error of the training rows)
  flat3/6    spread learned from errors with each month's median removed; centre = median error of the last 3 / 6 months
  ideal      as flat, but centred on the sale month's own median (not knowable in advance: the best a centre rule can do)

Then it checks the procedure range_model.py now uses (flat3 with separate widening below and above), choosing the
widening on the first half of the validation months and testing it on the second half.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import range_model as rm

Q = rm.QUANTILES
I = {q: i for i, q in enumerate(Q)}


def fit(t, target):
    return {q: lgb.LGBMRegressor(alpha=q, **rm.PARAMS).fit(t[rm.RANGE_FEATS], t[target]) for q in Q}


def qmat(models, t):
    return np.sort(np.column_stack([models[q].predict(t[rm.RANGE_FEATS]) for q in Q]), axis=1)


def trailing(d, k):
    """For each sale, the median error of the k months before its month (None when too few sales to trust)."""
    out = {}
    for m in sorted(d["m"].unique()):
        w = d.loc[(d["m"] >= m - k) & (d["m"] < m), "e"]
        out[m] = float(w.median()) if len(w) >= 500 else np.nan
    return d["m"].map(out)


def sides(e, centre, q, s_lo=1.16, s_hi=1.16, level=(0.10, 0.90)):
    lo = centre + s_lo * (q[:, I[level[0]]] - q[:, I[0.50]])
    hi = centre + s_hi * (q[:, I[level[1]]] - q[:, I[0.50]])
    return float((e < lo).mean() * 100), float((e > hi).mean() * 100), float((hi - lo).mean() * 100)


def solve(e, centre, q, level, target):
    """The widening that puts `target` of sales beyond the bound at quantile `level` (below it if level < 0.5)."""
    off = q[:, I[level]] - q[:, I[0.50]]
    grid = np.arange(0.6, 2.41, 0.01)
    miss = [((e < centre + s * off) if level < 0.5 else (e > centre + s * off)).mean() for s in grid]
    return float(grid[int(np.argmin(np.abs(np.array(miss) - target)))])


def main():
    d = rm.load()
    d["m"] = d["sale_date"].dt.to_period("M")
    d["e_flat"] = d["e"] - d["m"].map(d.groupby("m")["e"].median())
    d["c3"], d["c6"], d["c0"] = trailing(d, 3), trailing(d, 6), d["m"].map(d.groupby("m")["e"].median())
    print(f"{len(d):,} walk-forward sales, {d['sale_date'].min():%b %Y} to {d['sale_date'].max():%b %Y}")
    print("median error by year (%):", (d.groupby(d["sale_date"].dt.year)["e"].median() * 100).round(2).to_dict())

    rows = []
    for y in (2023, 2024, 2025, 2026):
        tr, te = d[d["sale_date"] < f"{y}-01-01"], d[d["sale_date"].dt.year == y]
        e, qr, qf = te["e"].to_numpy(), qmat(fit(tr, "e"), te), qmat(fit(tr, "e_flat"), te)
        centres = {"old": qr[:, I[0.5]], "shift": qr[:, I[0.5]] + (te["c3"].to_numpy() - tr["e"].median()), "flat3": te["c3"].to_numpy() + qf[:, I[0.5]],
                   "flat6": te["c6"].to_numpy() + qf[:, I[0.5]], "ideal": te["c0"].to_numpy() + qf[:, I[0.5]]}
        for name, c in centres.items():
            below, above, width = sides(e, c, qr if name in ("old", "shift") else qf)
            rows.append(dict(year=y, method=name, n=len(te), below_centre=round(float((e < c).mean() * 100), 1), below_range=round(below, 1), above_range=round(above, 1), width=round(width, 1)))
        print(f"  {y} done", flush=True)
    t = pd.DataFrame(rows)
    print("\nSHARE OF SALES BELOW THE CENTRE (want 50) AND OUTSIDE THE 80% RANGE (want 10 and 10), widening 1.16 throughout")
    print(t.pivot(index="year", columns="method", values="below_centre")[["old", "shift", "flat3", "flat6", "ideal"]].to_string())
    for col in ("below_range", "above_range", "width"):
        print(f"\n{col}")
        print(t.pivot(index="year", columns="method", values=col)[["old", "shift", "flat3", "flat6", "ideal"]].to_string())

    # the procedure as shipped: spread from errors before VAL_START, widening chosen on the first half of the months after it, tested on the second half
    tr, va = d[d["sale_date"] < rm.VAL_START], d[d["sale_date"] >= rm.VAL_START].copy()
    months = sorted(va["m"].unique())
    first = va["m"] < months[len(months) // 2]
    qr, qf = qmat(fit(tr, "e"), va), qmat(fit(tr, "e_flat"), va)
    e, c_old, c_new = va["e"].to_numpy(), qr[:, I[0.5]], va["c3"].to_numpy() + qf[:, I[0.5]]
    a, b = first.to_numpy(), ~first.to_numpy()
    s_old = min(np.arange(0.6, 2.41, 0.01), key=lambda s: abs(sum(sides(e[a], c_old[a], qr[a], s, s)[:2]) - 20))
    s_lo, s_hi = solve(e[a], c_new[a], qf[a], 0.10, 0.10), solve(e[a], c_new[a], qf[a], 0.90, 0.10)
    print(f"\nAS SHIPPED: widening chosen on {months[0]} to {months[len(months) // 2 - 1]} ({a.sum():,} sales), tested on {months[len(months) // 2]} to {months[-1]} ({b.sum():,} sales)")
    print(f"  old: one widening {s_old:.2f}   new: {s_lo:.2f} below, {s_hi:.2f} above")
    va["age"] = va["sale_date"].dt.year - va["lease_commence_date"]
    groups = {"all": np.ones(len(va), bool), "flats 0-20 years old": (va["age"] <= 20).to_numpy(), "21-40 years": ((va["age"] > 20) & (va["age"] <= 40)).to_numpy(),
              "over 40 years": (va["age"] > 40).to_numpy(), "estimate under $500k": (va["pred"] < 5e5).to_numpy(), "estimate $800k and over": (va["pred"] >= 8e5).to_numpy()}
    out = []
    for name, g in groups.items():
        k = g & b
        o, n = sides(e[k], c_old[k], qr[k], s_old, s_old), sides(e[k], c_new[k], qf[k], s_lo, s_hi)
        o5, n5 = sides(e[k], c_old[k], qr[k], s_old, s_old, (0.25, 0.75)), sides(e[k], c_new[k], qf[k], s_lo, s_hi, (0.25, 0.75))
        out.append(dict(group=name, n=int(k.sum()), old_below=round(o[0], 1), old_above=round(o[1], 1), new_below=round(n[0], 1), new_above=round(n[1], 1),
                        old_width=round(o[2], 1), new_width=round(n[2], 1), old_below_q25=round(o5[0], 1), old_above_q75=round(o5[1], 1), new_below_q25=round(n5[0], 1), new_above_q75=round(n5[1], 1)))
    print(pd.DataFrame(out).to_string(index=False))

    # would moving the fair value itself by the recent centre help? (not done in production: reported for the record)
    ok = d["c3"].notna() & (d["sale_date"] >= "2023-01-01")
    ape = lambda shift: (np.exp(shift - d.loc[ok, "e"]) - 1).abs()
    by = d.loc[ok, "sale_date"].dt.year
    print("\nMEDIAN ERROR OF THE FAIR VALUE (%), as is and moved by the last 3 months' median error")
    print(pd.DataFrame({"as_is": ape(0).groupby(by).median() * 100, "moved": ape(d.loc[ok, "c3"]).groupby(by).median() * 100}).round(2).to_string())


BANDS = [-1, 20, 40, 200]  # building age in years: newer, middle, older. The 2026 market split along this line.


def by_band(d, k):
    """As trailing(), but within each age band: the last k months' median error for flats of that age."""
    out = {}
    for m in sorted(d["m"].unique()):
        w = d[(d["m"] >= m - k) & (d["m"] < m)]
        for b, g in w.groupby("band", observed=True):
            out[(m, b)] = float(g["e"].median()) if len(g) >= 300 else np.nan
    return pd.Series([out.get(key, np.nan) for key in zip(d["m"], d["band"])], index=d.index)


def learned(d, months=6):
    """A small model of the recent median error, refitted for each month on the six months before it."""
    feats = ["remaining_lease_years", "pred_log", "floor_area_sqm", "storey_mid", "dist_cbd", "block_gap"]
    out = pd.Series(np.nan, index=d.index)
    for m in sorted(d["m"].unique()):
        w, te = d[(d["m"] >= m - months) & (d["m"] < m)], d[d["m"] == m]
        if len(w) < 5000 or m.year < 2023:
            continue
        model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=60, learning_rate=0.05, num_leaves=8, min_child_samples=400, lambda_l2=5.0, verbose=-1).fit(w[feats], w["e"])
        out[te.index] = model.predict(te[feats])
    return out


def segments():
    """Does the centre need to differ by kind of flat? Compares one centre for all flats, one per age band, and a learned one."""
    d = rm.load()
    d["m"] = d["sale_date"].dt.to_period("M")
    d["age"] = d["sale_date"].dt.year - d["lease_commence_date"]
    d["band"] = pd.cut(d["age"], BANDS, labels=["new", "mid", "old"]).astype(str)
    d["e_flat"] = d["e"] - d["m"].map(d.groupby("m")["e"].median())
    d["e_band"] = d["e"] - pd.Series(list(zip(d["m"], d["band"])), index=d.index).map(d.groupby(["m", "band"])["e"].median().to_dict())
    d["c3"], d["c3b"], d["cl"] = trailing(d, 3), by_band(d, 3), learned(d)
    print("median error by year and age band (%):")
    print((d.groupby([d["sale_date"].dt.year, "band"])["e"].median() * 100).round(2).unstack().to_string())
    parts = []
    for y in (2023, 2024, 2025, 2026):
        tr, te = d[d["sale_date"] < f"{y}-01-01"], d[d["sale_date"].dt.year == y].copy()
        qf, qb = qmat(fit(tr, "e_flat"), te), qmat(fit(tr, "e_band"), te)
        for name, c, q in (("one", te["c3"].to_numpy() + qf[:, I[0.5]], qf), ("band", te["c3b"].to_numpy() + qb[:, I[0.5]], qb), ("learned", te["cl"].to_numpy(), qf)):
            te[f"{name}_c"] = c
            te[f"{name}_lo"] = c + 1.16 * (q[:, I[0.10]] - q[:, I[0.50]])
            te[f"{name}_hi"] = c + 1.16 * (q[:, I[0.90]] - q[:, I[0.50]])
            te[f"{name}_l5"] = c + 1.16 * (q[:, I[0.25]] - q[:, I[0.50]])
            te[f"{name}_h5"] = c + 1.16 * (q[:, I[0.75]] - q[:, I[0.50]])
        parts.append(te)
        print(f"  {y} done", flush=True)
    t = pd.concat(parts)
    t["price"] = pd.cut(t["pred"], [0, 5e5, 8e5, 1e8], labels=["under $500k", "$500-800k", "$800k+"]).astype(str)

    def table(g):
        row = {"n": len(g)}
        for name in ("one", "band", "learned"):
            row[f"{name}: below centre"] = round(float((g["e"] < g[f"{name}_c"]).mean() * 100), 1)
            row[f"{name}: below|above 80%"] = f"{(g['e'] < g[f'{name}_lo']).mean() * 100:.1f}|{(g['e'] > g[f'{name}_hi']).mean() * 100:.1f}"
            row[f"{name}: below|above 50%"] = f"{(g['e'] < g[f'{name}_l5']).mean() * 100:.1f}|{(g['e'] > g[f'{name}_h5']).mean() * 100:.1f}"
        return pd.Series(row)

    print("\nBY YEAR (want 50 below the centre, 10|10 outside the 80% range, 25|25 outside the 50% range; widening 1.16)")
    print(t.groupby(t["sale_date"].dt.year).apply(table).to_string())
    recent = t[t["sale_date"] >= "2025-07-01"]
    for key in ("band", "price"):
        print(f"\nJUL 2025 ON, BY {key.upper()}")
        print(recent.groupby(key).apply(table).to_string())
    worst = lambda name: recent.groupby(["band", "price"]).apply(lambda g: abs((g["e"] < g[f"{name}_c"]).mean() * 100 - 50) if len(g) >= 400 else np.nan).max()
    print("\nworst cell (age band x price band, 400+ sales): points away from 50% below the centre:", {n: round(float(worst(n)), 1) for n in ("one", "band", "learned")})


if __name__ == "__main__":
    import sys
    segments() if "segments" in sys.argv[1:] else main()
