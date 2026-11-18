"""Price one condo unit with the trained condo model: the private-property twin of advice.assess().

  python src/condo.py "PARC ESTA" 70 12            fair value and ranges for a 70 sqm unit on the 12th floor
  python src/condo.py "PARC ESTA" 70 12 1600000    ... and where an asking price of $1.6m sits

assess() returns the fair value, the price at each quantile (so a page can place any price), the reliability badge
with its reasons, and the project's recent sales. Everything comes from models/condo*.{txt,json} and
data/processed/condo_projects.parquet, which train_condo.py writes.
"""
import difflib
import functools
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import condo_features as cf
from geocode_blocks import ROOT


@functools.lru_cache(maxsize=1)
def meta():
    return json.loads((ROOT / "models/condo.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def models():
    m = meta()
    return ([lgb.Booster(model_file=str(ROOT / f"models/condo_prod_{i}.txt")) for i in range(m["models"])],
            {q: lgb.Booster(model_file=str(ROOT / f"models/condo_range_q{int(q * 100):02d}.txt")) for q in m["quantiles"]})


@functools.lru_cache(maxsize=1)
def projects():
    return pd.read_parquet(ROOT / "data/processed/condo_projects.parquet").set_index("project")


@functools.lru_cache(maxsize=1)
def sales():
    cols = ["project", "sale_date", "sale_type", "area", "floor_range", "price", "property_type", "units"]
    d = pd.read_parquet(ROOT / "data/processed/ura_transactions.parquet", columns=cols)
    return d[d["units"] == 1].sort_values("sale_date", ascending=False)


def find_project(name):
    """The project as URA writes it, forgiving spelling and case. None if nothing is close."""
    names, want = projects().index, " ".join(str(name).upper().split())
    if want in names:
        return want
    starts = [n for n in names if n.startswith(want) or want.startswith(n)]
    if len(starts) == 1:
        return starts[0]
    close = difflib.get_close_matches(want, list(names), n=1, cutoff=0.8)
    return close[0] if close else None


@functools.lru_cache(maxsize=1)
def name_index():
    """Project names as people type them (letters, digits and spaces only), longest first."""
    import re
    names = {" ".join(re.sub(r"[^A-Z0-9 ]", " ", n.upper().replace("@", " AT ")).split()): n for n in projects().index}
    # one-word names that are also ordinary words would fire on normal sentences; the AI reader still finds those projects
    return sorted(((k, v) for k, v in names.items() if (" " in k and len(k) >= 6 and not k.startswith("UNIT ")) or len(k) >= 8), key=lambda kv: -len(kv[0]))


def find_in_text(text):
    """The project a message names ("parc esta 2 bed 1.6m"), or None. The whole name must appear, as whole words."""
    import re
    t = " " + " ".join(re.sub(r"[^A-Z0-9 ]", " ", text.upper().replace("@", " AT ")).split()) + " "
    return next((name for key, name in name_index() if f" {key} " in t), None)


def size_for_bedrooms(project, bedrooms):
    """A rough size for an n-bedroom unit in this project, from where such units usually sit among its sold sizes."""
    s = sales()
    areas = s[s["project"] == project]["area"]
    share = {1: 0.12, 2: 0.4, 3: 0.68, 4: 0.88, 5: 0.96}.get(int(bedrooms))
    return float(round(areas.quantile(share))) if share and len(areas) >= 8 else None


def assess(project, area, floor, asking=None):
    """Fair value and ranges for a unit of `area` sqm on storey `floor` in `project`, priced at the month after the training data ends."""
    m, name = meta(), find_project(project)
    if name is None:
        raise ValueError(f"no project called {project!r} in URA's sales of the last five years")
    p = projects().loc[name]
    point, rng = models()
    last = pd.Timestamp(m["trained_through"] + "-01")
    now = last + pd.DateOffset(months=1)
    freehold, lease_start, lease_left = cf.tenure_fields(p["tenure"], now.year)
    low = (int(floor) - 1) // 5 * 5 + 1 if floor and floor > 0 else np.nan  # URA's bands: 01-05, 06-10, ...
    row = pd.DataFrame([dict(project=name, property_type=p["property_type"], district=p["district"], market_segment=p["market_segment"], area=float(area),
                             floor_low=low, floor_mid=low + 2 if low == low else np.nan, freehold=freehold, lease_start=lease_start, lease_left=lease_left,
                             t=int(cf.month_index(pd.Series([now])).iloc[0]), sale_month=now.month, x=p["x"], y=p["y"],
                             dist_cbd=float(np.hypot(p["x"] - cf.CBD[0], p["y"] - cf.CBD[1]) / 1000), is_new=0, **{c: p[c] for c in cf.LOC},
                             comps_lp=p["comps_lp"], n_proj_24m=p["n_proj_24m"], n_proj_12m=p["n_proj_12m"],
                             dev_lp=p.get("dev_lp", np.nan), dev_selling=p.get("dev_selling", np.nan),
                             **cf.query_history(p.get("recent"), float(area), low + 2 if low == low else np.nan, p["top_floor"], p["typical_area"],
                                                p.get("market_now", np.nan)))])
    row = cf.as_categories(row, m["levels"])
    pred_lp = float(np.mean([b.predict(row[m["features"]]) for b in point]))
    fair = float(np.exp(pred_lp) * area)

    r = row.assign(pred_lp=pred_lp, proj_gap=p["comps_lp"] - pred_lp)
    for c, levels in m["range_levels"].items():
        r[c] = pd.Categorical(r[c].astype(str), categories=levels)
    q = np.sort([float(rng[x].predict(r[m["range_features"]])[0]) for x in m["quantiles"]])
    med = q[m["quantiles"].index(0.5)]
    # centred on where recent resales in this market segment landed against our estimate, then widened below and above (recentre.py)
    centre = m["centre"].get(str(p["market_segment"]), m["centre"]["all"])
    q = centre + med + np.where(q < med, m["scale_low"], m["scale_high"]) * (q - med)
    quantiles = [(lvl, fair * float(np.exp(v))) for lvl, v in zip(m["quantiles"], q)]
    width = q[m["quantiles"].index(0.9)] - q[m["quantiles"].index(0.1)]
    badge = "High" if width <= m["badge_high"] else "Medium" if width <= m["badge_medium"] else "Low"

    notes = []
    if p["n_proj_24m"] < 3:
        notes.append(f"only {int(p['n_proj_24m'])} sales in this project in the last 24 months")
    if area > 1.6 * p["typical_area"] or area < 0.6 * p["typical_area"]:
        notes.append(f"{round(area)} sqm is unusual here (typical unit {round(p['typical_area'])} sqm)")
    if p["property_type"] == "Apartment" and p["n_sales"] < 15:
        notes.append("a small development, where each unit differs more")
    if not freehold and lease_left == lease_left and lease_left < 60:
        notes.append(f"about {round(lease_left)} years of lease left, where prices vary more")

    s = sales()
    recent = s[s["project"] == name].head(8)
    out = dict(project=name, street=p["street"], property_type=p["property_type"], district=p["district"], market_segment=p["market_segment"], tenure=p["tenure"],
               freehold=bool(freehold), lease_left=None if freehold or lease_left != lease_left else float(lease_left), area=float(area), floor_band=None if low != low else f"{int(low):02d}-{int(low) + 4:02d}",
               fair=fair, fair_psm=fair / area, quantiles=quantiles, badge=badge, range_width=float(width), notes=notes, priced_as_of=f"{now:%Y-%m}",
               recent=[dict(month=f"{x.sale_date:%Y-%m}", sale_type=x.sale_type, floor=x.floor_range, area=float(x.area), price=float(x.price)) for x in recent.itertuples()])
    if asking:
        lq = np.log([v for _, v in quantiles])
        levels = [lvl for lvl, _ in quantiles]
        out["asking"] = float(asking)
        out["asking_percentile"] = float(np.interp(np.log(asking), lq, levels, left=levels[0], right=levels[-1]) * 100)
        out["asking_vs_fair"] = asking / fair - 1
    return out


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    a = assess(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else None)
    k = lambda v: f"${v / 1e6:.2f}m" if v >= 1e6 else f"${v / 1e3:.0f}k"
    q = dict(a["quantiles"])
    print(f"{a['project']} ({a['property_type']}, {a['street']}, D{a['district']}, {a['tenure']})")
    print(f"{a['area']:.0f} sqm, floors {a['floor_band']}: fair value {k(a['fair'])} (${a['fair_psm']:,.0f} per sqm), priced as of {a['priced_as_of']}")
    print(f"likely range {k(q[0.25])} to {k(q[0.75])}; wide range {k(q[0.1])} to {k(q[0.9])}; reliability {a['badge']}" + (f" ({'; '.join(a['notes'])})" if a["notes"] else ""))
    if "asking" in a:
        print(f"asking {k(a['asking'])} is {a['asking_vs_fair'] * 100:+.1f}% versus fair value; about {a['asking_percentile']:.0f}% of comparable sales would be cheaper")
    for r in a["recent"][:5]:
        print(f"  {r['month']}  {r['sale_type']:9} floors {r['floor']:6} {r['area']:6.0f} sqm  ${r['price']:,.0f}")
