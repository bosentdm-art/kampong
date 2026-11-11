"""Features for the condo model, built from data/processed/ura_transactions.parquet (see download_ura.py).

Covers condominiums, apartments and executive condominiums. One row per sale, with:
  the unit       area, floor range (URA gives a 5-floor band, not the unit), property type
  the project    name, district, market segment, tenure (freehold or years left), location (SVY21 x, y)
  the moment     month index, calendar month, type of sale (new sale / sub sale / resale)
  history        sales in the same project in the 24 months BEFORE the sale, and what those sales say this unit's
                 price per sqm should be (their median, moved to the sale month by the market index). Earlier months
                 only, so nothing leaks from the future.
"""
import re

import numpy as np
import pandas as pd

from geocode_blocks import ROOT

TYPES = ["Apartment", "Condominium", "Executive Condominium"]
CATS = ["project", "property_type", "district", "market_segment"]
NUM = ["area", "floor_low", "floor_mid", "freehold", "lease_start", "lease_left", "t", "sale_month", "x", "y", "dist_cbd", "is_new"]
HIST = ["comps_lp", "n_proj_24m", "n_proj_12m"]
CBD = (29800.0, 29700.0)  # Raffles Place, SVY21 metres (approximate)
T0 = 2021 * 12


def floor_bounds(s):
    nums = [int(x) for x in re.findall(r"\d+", str(s))]
    if not nums:
        return np.nan, np.nan
    sign = -1 if str(s).upper().startswith("B") else 1
    return sign * nums[0], sign * float(np.mean(nums))


def tenure_fields(tenure, sale_year):
    """(freehold or as good as, lease start year, years of lease left). 999-year and longer leases count as freehold."""
    t = str(tenure)
    years = re.match(r"(\d+)\s*(?:yrs|years)", t)
    start = re.search(r"(\d{4})\s*$", t)
    if "freehold" in t.lower() or (years and int(years.group(1)) >= 900):
        return 1, np.nan, 999.0
    length = int(years.group(1)) if years else 99
    return 0, (float(start.group(1)) if start else np.nan), (length - (sale_year - int(start.group(1))) if start else np.nan)


def month_index(dates):
    return dates.dt.year * 12 + dates.dt.month - T0


def market_index(d):
    """Smoothed monthly median log price per sqm of resales, with every month present from the first to the last."""
    r = d[d["is_new"] == 0].groupby("t")["lp"].median()
    r = r.reindex(range(int(d["t"].min()), int(d["t"].max()) + 1)).interpolate().bfill().ffill()
    return r.rolling(3, min_periods=1).mean()


def load():
    d = pd.read_parquet(ROOT / "data/processed/ura_transactions.parquet")
    d = d[d["property_type"].isin(TYPES) & (d["units"] == 1) & (d["area"] > 20) & (d["price"] > 100_000)].copy()
    d["is_new"] = (d["sale_type"] == "new sale").astype(int)
    d["t"] = month_index(d["sale_date"])
    d["sale_month"] = d["sale_date"].dt.month
    fb = d["floor_range"].map(floor_bounds)
    d["floor_low"], d["floor_mid"] = fb.str[0], fb.str[1]
    tf = [tenure_fields(a, b) for a, b in zip(d["tenure"], d["sale_date"].dt.year)]
    d["freehold"], d["lease_start"], d["lease_left"] = [x[0] for x in tf], [x[1] for x in tf], [x[2] for x in tf]
    d["dist_cbd"] = np.hypot(d["x"] - CBD[0], d["y"] - CBD[1]) / 1000
    d["lp"] = np.log(d["price_per_sqm"])
    return add_history(d.sort_values(["sale_date", "project"]).reset_index(drop=True))


def add_history(d):
    """Sales in the same project before each sale: how many, and the price per sqm they point to."""
    market = market_index(d)
    adj = d["lp"] - d["t"].map(market)  # each sale's price relative to the market in its month
    comps, n24, n12 = np.full(len(d), np.nan), np.zeros(len(d)), np.zeros(len(d))
    for _, idx in d.groupby("project").indices.items():
        t, a = d["t"].to_numpy()[idx], adj.to_numpy()[idx]  # already in time order
        for j, i in enumerate(idx):
            lo24, lo12, hi = np.searchsorted(t, t[j] - 24), np.searchsorted(t, t[j] - 12), np.searchsorted(t, t[j])
            n24[i], n12[i] = hi - lo24, hi - lo12
            if hi > lo24:
                comps[i] = np.median(a[max(lo24, hi - 12):hi]) + market.get(t[j] - 1, market.iloc[0])  # the latest 12 of them, at last month's market
    d["comps_lp"], d["n_proj_24m"], d["n_proj_12m"] = comps, n24, n12
    return d


def as_categories(d, levels=None):
    """Make the category columns categorical. Pass the training levels when predicting so codes line up."""
    d = d.copy()
    for c in CATS:
        d[c] = pd.Categorical(d[c].astype(str), categories=levels[c] if levels else None)
    return d


def latest_complete_month(d):
    """The last month, unless it has far fewer sales than the months before it (URA is still filling it in)."""
    counts = d[d["is_new"] == 0].groupby("sale_date").size()
    return counts.index[-1] if counts.iloc[-1] >= 0.7 * counts.iloc[-4:-1].median() else counts.index[-2]
