"""Keep the price ranges centred on the market as it is now. Used by range_model.py (HDB) and train_condo.py (condos).

The range models learn, from walk-forward errors e = log(actual / estimate), how far off we tend to be for a home like
this one. Two things used to be learned together and are now kept apart:

  spread   how widely sales scatter around their own market that month. Learned from every year of errors, after
           taking out each month's median error for homes of the same kind (flatten).
  centre   where the market sits against our estimate right now. Not learned: it is the median error of the last
           three months of sales of the same kind (centres), so it moves when the market does.

Why: the centre used to be learned from all years, most of which had rising prices and sales landing above the
estimate. When prices went flat in 2026 the ranges stayed too high: on sales the models had not seen, 16% sold below
the 80% range and 4% above it. experiments_ranges.py has the comparison, year by year, and the reason the centre is
kept per kind of home (old flats were falling while new ones rose).

The ranges are then widened separately below and above the centre (solve), so that one sale in ten lands under the
80% range and one in ten over it on months held out for the purpose.
"""
import numpy as np
import pandas as pd

MONTHS = 3       # how far back the centre looks
MIN_SALES = 300  # a kind of home with fewer recent sales than this uses the centre of all homes


def flatten(d):
    """Each error with the median of its month and kind removed. Needs columns m (month), band (kind of home) and e."""
    key = pd.Series(list(zip(d["m"], d["band"])), index=d.index)
    return d["e"] - key.map(d.groupby(["m", "band"])["e"].median().to_dict())


def centres(d, before=None):
    """{kind: median error over the MONTHS months before `before`}, plus "all". Default: the months up to the newest on file."""
    end = d["m"].max() + 1 if before is None else before
    w = d[(d["m"] >= end - MONTHS) & (d["m"] < end)]
    out = {"all": float(w["e"].median()) if len(w) >= MIN_SALES else 0.0}
    for b, g in w.groupby("band"):
        if len(g) >= MIN_SALES:
            out[str(b)] = float(g["e"].median())
    return out


def trailing(d):
    """For every sale, the centre as it stood at the start of that sale's month: only earlier months go into it."""
    by_month = {m: centres(d, m) for m in sorted(d["m"].unique())}
    return np.array([by_month[m].get(b, by_month[m]["all"]) for m, b in zip(d["m"], d["band"])])


def spread(q, centre, i_med, s_lo, s_hi):
    """The quantiles of the error for each home: the centre, plus the learned spread widened by s_lo below and s_hi above."""
    med = q[:, [i_med]]
    return np.asarray(centre)[:, None] + med + np.where(q < med, s_lo, s_hi) * (q - med)


def solve(e, q, centre, i_med, i_lo, i_hi, target=0.10):
    """(s_lo, s_hi): the widening that leaves `target` of sales under the low bound and `target` over the high one."""
    grid = np.arange(0.60, 2.41, 0.01)
    base = np.asarray(centre) + q[:, i_med]
    under = [np.mean(e < base + s * (q[:, i_lo] - q[:, i_med])) for s in grid]
    over = [np.mean(e > base + s * (q[:, i_hi] - q[:, i_med])) for s in grid]
    return float(grid[int(np.argmin(np.abs(np.array(under) - target)))]), float(grid[int(np.argmin(np.abs(np.array(over) - target)))])


def sides(e, m, i_lo, i_hi):
    """(share of sales under the low bound, share over the high bound), in percent."""
    return float(np.mean(e < m[:, i_lo]) * 100), float(np.mean(e > m[:, i_hi]) * 100)
