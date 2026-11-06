"""Do recent comparable sales improve the price estimate? Out-of-sample test.

Adjusted comparables: a comparable sale's price adjusted to the target flat by the model is
comp_price x pred_target / pred_comp, and its ratio to the target's own estimate is exp(e_comp), the model's
out-of-sample error on that comparable. So "what the comps say" = pred_target x exp(mean e of the comps).
Blend with shrinkage (trust the comps more when there are more of them):
    est = pred x exp( n / (n + k) x mean(e_comps) )
Levels: (B) same block + flat type, earlier W months; (C) same block + flat type + size (rounded sqm), longer
window, falling back to B when there is none. Only sales from EARLIER months are used, and every error comes
from the walk-forward files, so nothing here has seen its own answer.
Tune W and k on 2023-01..2024-12; report on 2025-01 onward (never used for tuning).
"""
import numpy as np
import pandas as pd

from geocode_blocks import ROOT

OOS = ROOT / "data/processed/oos"
TUNE = ("2023-01-01", "2024-12-01")
TEST_FROM = "2025-01-01"


def load():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OOS.glob("chunk_*.parquet"))], ignore_index=True)
    d["e"] = np.log(d["resale_price"] / d["pred"])
    d["m"] = d["sale_date"].dt.year * 12 + d["sale_date"].dt.month - 1
    d["size"] = d["floor_area_sqm"].round().astype(int)
    return d


def prior_mean(d, keys, window):
    """Mean and count of e over the `window` months BEFORE each row's month, within the same keys."""
    code, _ = pd.factorize(d[keys].astype(str).agg("|".join, axis=1))
    m = (d["m"] - d["m"].min()).to_numpy()
    nk, nm = code.max() + 1, m.max() + 2
    s, c = np.zeros((nk, nm)), np.zeros((nk, nm))
    np.add.at(s, (code, m + 1), d["e"].to_numpy())
    np.add.at(c, (code, m + 1), 1.0)
    s, c = s.cumsum(1), c.cumsum(1)
    lo = np.maximum(m - window, 0)
    ws, wc = s[code, m] - s[code, lo], c[code, m] - c[code, lo]
    return np.where(wc > 0, ws / np.maximum(wc, 1), 0.0), wc


def score(d, est):
    ape = (est - d["resale_price"]).abs() / d["resale_price"]
    return ape.median(), ape.mean(), (ape <= 0.10).mean()


def main():
    d = load()
    tune = d["sale_date"].between(*TUNE)
    test = d["sale_date"] >= TEST_FROM
    print(f"rows {len(d)}: tune {tune.sum()}, test {test.sum()}")

    best = {}
    for name, keys, windows in [("B block+type", ["block", "street_name", "flat_type"], (12, 24, 36)),
                                ("C block+type+size", ["block", "street_name", "flat_type", "size"], (24, 36, 60))]:
        rows = []
        for w in windows:
            mean, n = prior_mean(d, keys, w)
            for k in (0.5, 1, 2, 4, 8, 16):
                est = d["pred"] * np.exp(n / (n + k) * mean)
                rows.append((w, k, score(d[tune], est[tune])[0]))
        w, k, s = min(rows, key=lambda r: r[2])
        best[name] = (keys, w, k)
        print(f"{name}: best on tuning period W={w} months, k={k} (median error {s:.3%})")

    base = d["pred"]
    kb, wb, kkb = best["B block+type"]
    mb, nb = prior_mean(d, kb, wb)
    est_b = d["pred"] * np.exp(nb / (nb + kkb) * mb)
    kc, wc, kkc = best["C block+type+size"]
    mc, nc = prior_mean(d, kc, wc)
    est_c = np.where(nc > 0, d["pred"] * np.exp(nc / (nc + kkc) * mc), est_b)

    print("\nTEST PERIOD (2025-01 onward, never used for tuning):")
    print(f"{'':28} {'median':>8} {'mean':>8} {'within10':>9}")
    for label, est in [("A model only", base), ("B + block comps", est_b), ("C + same-size comps, else B", pd.Series(est_c, index=d.index))]:
        md, mn, w10 = score(d[test], est[test])
        print(f"{label:28} {md:>8.3%} {mn:>8.3%} {w10:>9.1%}")

    t = d[test].copy()
    t["ape_a"] = (base[test] - t["resale_price"]).abs() / t["resale_price"]
    t["ape_c"] = (pd.Series(est_c, index=d.index)[test] - t["resale_price"]).abs() / t["resale_price"]
    t["has_comps"] = nb[test] > 0
    print("\nby whether the block had comparable sales in the window:")
    print(t.groupby("has_comps").agg(n=("e", "size"), model_only=("ape_a", "median"), with_comps=("ape_c", "median")).round(4).to_string())
    print("\nby remaining lease:")
    print(t.groupby(pd.cut(t["remaining_lease_years"], [0, 55, 65, 75, 85, 100]), observed=True)
          .agg(n=("e", "size"), model_only=("ape_a", "median"), with_comps=("ape_c", "median")).round(4).to_string())


if __name__ == "__main__":
    main()
