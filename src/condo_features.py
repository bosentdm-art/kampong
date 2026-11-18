"""Features for the condo model, built from data/processed/ura_transactions.parquet (see download_ura.py).

Covers condominiums, apartments and executive condominiums. One row per sale, with:
  the unit       area, floor range (URA gives a 5-floor band, not the unit), property type
  the project    name, district, market segment, tenure (freehold or years left), location (SVY21 x, y)
  the moment     month index, calendar month, type of sale (new sale / sub sale / resale)
  surroundings   distance to the nearest MRT exit, hawker centre and primary school, MRT exits within 500 m, primary
                 schools within 1 km (URA gives SVY21 x, y; svy21_to_latlon() converts them)
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
LOC = ["dist_mrt", "n_mrt_exits_500m", "dist_hawker", "dist_primary_school", "n_primary_1km"]
HIST = ["comps_lp", "n_proj_24m", "n_proj_12m"]
# Added 2026-11 after experiments_condo2.py and experiments_condo3.py (better on all three six-month periods tested):
#   comps_size_lp / n_size_24m   same-project sales in the 24 months before of units within 20% of this unit's size (latest 12)
#   dev_lp / dev_selling          the developer's median new-sale price there before this sale, and whether it was still selling
#                                 in the 6 months before (resales compete with new units)
#   rel_floor / rel_size          floor over the project's highest recorded floor (the building, not the future), and size
#                                 over the project's median unit
HIST2 = ["comps_size_lp", "n_size_24m", "dev_lp", "dev_selling", "rel_floor", "rel_size"]
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


def svy21_to_latlon(x, y):
    """SVY21 easting and northing (metres, as URA gives them) to latitude and longitude: the standard inverse
    transverse Mercator on WGS84 with Singapore's origin. Takes arrays, returns an (n, 2) array of (lat, lon)."""
    E, N = np.asarray(x, float), np.asarray(y, float)
    a, f = 6378137.0, 1 / 298.257223563
    o_lat, o_lon, o_n, o_e, k = np.radians(1.366666), np.radians(103.833333), 38744.572, 28001.642, 1.0
    b = a * (1 - f)
    e2 = 2 * f - f * f
    e4, e6 = e2 * e2, e2 ** 3
    A0, A2, A4, A6 = 1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256, 3 / 8 * (e2 + e4 / 4 + 15 * e6 / 128), 15 / 256 * (e4 + 3 * e6 / 4), 35 * e6 / 3072
    m0 = a * (A0 * o_lat - A2 * np.sin(2 * o_lat) + A4 * np.sin(4 * o_lat) - A6 * np.sin(6 * o_lat))
    mp = m0 + (N - o_n) / k
    n = (a - b) / (a + b)
    n2, n3, n4 = n * n, n ** 3, n ** 4
    g = a * (1 - n) * (1 - n2) * (1 + 9 * n2 / 4 + 225 * n4 / 64) * (np.pi / 180)
    sg = mp * np.pi / (180 * g)
    lp = (sg + (3 * n / 2 - 27 * n3 / 32) * np.sin(2 * sg) + (21 * n2 / 16 - 55 * n4 / 32) * np.sin(4 * sg)
          + (151 * n3 / 96) * np.sin(6 * sg) + (1097 * n4 / 512) * np.sin(8 * sg))
    s2 = np.sin(lp) ** 2
    rho, v = a * (1 - e2) / (1 - e2 * s2) ** 1.5, a / np.sqrt(1 - e2 * s2)
    psi = v / rho
    t = np.tan(lp)
    t2, t4, t6 = t * t, t ** 4, t ** 6
    ep = E - o_e
    xx = ep / (k * v)
    x3, x5, x7 = xx ** 3, xx ** 5, xx ** 7
    lf = t / (k * rho)
    lat = (lp - lf * (ep * xx / 2) + lf * (ep * x3 / 24) * (-4 * psi ** 2 + 9 * psi * (1 - t2) + 12 * t2)
           - lf * (ep * x5 / 720) * (8 * psi ** 4 * (11 - 24 * t2) - 12 * psi ** 3 * (21 - 71 * t2) + 15 * psi ** 2 * (15 - 98 * t2 + 15 * t4) + 180 * psi * (5 * t2 - 3 * t4) + 360 * t4)
           + lf * (ep * x7 / 40320) * (1385 - 3633 * t2 + 4095 * t4 + 1575 * t6))
    sec = 1 / np.cos(lp)
    lon = (o_lon + xx * sec - x3 * sec / 6 * (psi + 2 * t2)
           + x5 * sec / 120 * (-4 * psi ** 3 * (1 - 6 * t2) + psi ** 2 * (9 - 68 * t2) + 72 * psi * t2 + 24 * t4)
           - x7 * sec / 5040 * (61 + 662 * t2 + 1320 * t4 + 720 * t6))
    return np.column_stack([np.degrees(lat), np.degrees(lon)])


def surroundings(x, y):
    """The LOC features for arrays of SVY21 points (NaN where the point is missing)."""
    import nearby as nb
    t, ll = nb.tables(), svy21_to_latlon(x, y)
    out = {c: np.full(len(ll), np.nan) for c in LOC}
    for i, (lat, lon) in enumerate(ll):
        if lat != lat:
            continue
        m = nb.metres(lat, lon, t["mrt"]["lat"].to_numpy(), t["mrt"]["lon"].to_numpy())
        p = nb.metres(lat, lon, t["primary"]["lat"].to_numpy(), t["primary"]["lon"].to_numpy())
        out["dist_mrt"][i], out["n_mrt_exits_500m"][i] = m.min(), (m <= 500).sum()
        out["dist_hawker"][i] = nb.metres(lat, lon, t["hawker"]["lat"].to_numpy(), t["hawker"]["lon"].to_numpy()).min()
        out["dist_primary_school"][i], out["n_primary_1km"][i] = p.min(), (p <= 1000).sum()
    return out


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
    spots = d[["x", "y"]].drop_duplicates()
    d = d.merge(spots.assign(**surroundings(spots["x"].to_numpy(), spots["y"].to_numpy())), on=["x", "y"], how="left")
    d["lp"] = np.log(d["price_per_sqm"])
    return add_more_history(add_history(d.sort_values(["sale_date", "project"]).reset_index(drop=True)))


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


def add_more_history(d):
    """The HIST2 features for every sale, from earlier sales in the same project only (rel_floor and rel_size use the
    project's whole record, which describes the building rather than later prices)."""
    market = market_index(d)
    adj = (d["lp"] - d["t"].map(market)).to_numpy()
    t_all, a_all, f_all, new_all = d["t"].to_numpy(), d["area"].to_numpy(), d["floor_mid"].to_numpy(), d["is_new"].to_numpy()
    out = {k: np.full(len(d), np.nan) for k in HIST2}
    for _, idx in d.groupby("project", observed=True).indices.items():
        t, a, f, nw, ad = t_all[idx], a_all[idx], f_all[idx], new_all[idx], adj[idx]
        top = np.nanmax(f) if np.isfinite(f).any() else np.nan
        med_area = np.nanmedian(a)
        for j, i in enumerate(idx):
            lo, hi = np.searchsorted(t, t[j] - 24), np.searchsorted(t, t[j])
            mk = market.get(t[j] - 1, market.iloc[0])
            sz = np.abs(a[lo:hi] / a[j] - 1) <= 0.2
            out["n_size_24m"][i] = sz.sum()
            if sz.any():
                out["comps_size_lp"][i] = np.median(ad[lo:hi][sz][-12:]) + mk
            dv = nw[:hi] == 1
            if dv.any():
                out["dev_lp"][i] = np.median(ad[:hi][dv][-30:]) + mk
            out["dev_selling"][i] = float((nw[np.searchsorted(t, t[j] - 6):hi] == 1).any())
            out["rel_floor"][i] = f[j] / top if top and top > 0 else np.nan
            out["rel_size"][i] = a[j] / med_area if med_area else np.nan
    return d.assign(**out)


def query_history(recent, area, floor_mid, top_floor, typical_area, market_now):
    """HIST2 for a unit being priced today, from its project's table row (train_condo.project_table)."""
    import json
    r = np.array(json.loads(recent)) if isinstance(recent, str) and recent else np.zeros((0, 3))
    out = dict(n_size_24m=0.0, comps_size_lp=np.nan)
    if len(r):
        sz = np.abs(r[:, 1] / area - 1) <= 0.2
        out["n_size_24m"] = float(sz.sum())
        if sz.any():
            out["comps_size_lp"] = float(np.median(r[sz][-12:, 2]) + market_now)
    out["rel_floor"] = floor_mid / top_floor if top_floor and top_floor > 0 and floor_mid == floor_mid else np.nan
    out["rel_size"] = area / typical_area if typical_area else np.nan
    return out


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
