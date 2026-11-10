"""Advice layer for one flat: fair value, flat-specific ranges, asking-price check, reliability badge.

assess(spec, asking=None) returns a dict. Everything comes from the trained models and the sales data:
  fair value   the point model's estimate (a learned bias correction was tested and did not help)
  ranges       80% and 50% bands from the flat-specific quantile models (range_model.py)
  asking check where the asking price sits among comparable sales (interpolated from the quantiles)
  offer guide  opening offer = 25th percentile price, target = fair value, walk-away = 75th percentile price
  reliability  High / Medium / Low from the width of the flat's 80% band (validated against realised error),
               plus plain-language notes (short lease, thin block history, unusual size, comps disagree)

Usage: python src/advice.py   (demo)
"""
import functools
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import forecast as fc
import listing_parser as lp
import range_model as rm
from geocode_blocks import ROOT


@functools.lru_cache(maxsize=1)
def range_models():
    meta = json.loads((ROOT / "models/range.json").read_text())
    models = {q: lgb.Booster(model_file=str(ROOT / f"models/range_q{int(q * 100):02d}.txt")) for q in meta["quantiles"]}
    return meta, models


@functools.lru_cache(maxsize=1)
def sales_history():
    cols = ["block", "street_name", "flat_type", "sale_date", "price_per_sqm", "floor_area_sqm", "resale_price", "storey_range"]
    d = pd.read_parquet(fc.PROC / "resale_features.parquet", columns=cols)
    return d[d["sale_date"] >= "2017-01-01"]


def money(x):
    return f"${x:,.0f}"


def assess(spec, asking=None, when=None):
    when = fc.default_when() if when is None else when
    meta, models = range_models()
    row = fc.block_table().query("block == @spec['block'] and street_name == @spec['street_name']")
    if row.empty:
        raise ValueError("block not found (demolished or never geocoded)")
    row = row.iloc[0].to_dict()
    pred, lease0 = fc.price_today(spec, row, when)

    # data-richness signals, from sales in the 24 / 12 months up to the valuation month
    last = pd.Timestamp(year=int(when), month=int(round((when % 1) * 12)) + 1, day=1)
    h = sales_history()
    b = h[(h["block"] == spec["block"]) & (h["street_name"] == spec["street_name"]) & (h["flat_type"] == spec["flat_type"])]
    w24, w12 = b[b["sale_date"] > last - pd.DateOffset(months=24)], b[b["sale_date"] > last - pd.DateOffset(months=12)]
    comps = float(np.exp(np.log(w12["price_per_sqm"]).mean()) * spec["floor_area_sqm"]) if len(w12) >= 3 else np.nan

    x = rm.prepare(pd.DataFrame([dict(
        pred=pred, floor_area_sqm=spec["floor_area_sqm"], storey_mid=spec["storey_mid"], remaining_lease_years=lease0,
        lease_commence_date=spec["lease_commence_date"], dist_cbd=row["dist_cbd"], dist_mrt=row["dist_mrt"],
        n_block_24m=len(w24), comps_est=comps, flat_type=spec["flat_type"], town=spec["town"], flat_model=spec["flat_model"])]))
    q = np.sort([models[qq].predict(x[rm.RANGE_FEATS])[0] for qq in meta["quantiles"]])
    med = q[meta["quantiles"].index(0.50)]
    qs = med + meta["scale"] * (q - med)  # widened so the 80% band really covers 80% on later months
    at = lambda level: pred * np.exp(qs[meta["quantiles"].index(level)])
    fair = pred  # a bias correction from the quantile model was tested and made validation accuracy worse
    width = float(qs[meta["quantiles"].index(0.90)] - qs[meta["quantiles"].index(0.10)])
    badge = "High" if width <= meta["badge_high"] else "Medium" if width <= meta["badge_medium"] else "Low"

    notes = []
    if lease0 < 55:
        notes.append(f"Short remaining lease ({lease0:.0f} years): prices of these flats vary more.")
    if len(w24) < 3:
        notes.append(f"Few recent sales of this flat type in this block ({len(w24)} in 24 months).")
    if len(b) >= 5:
        usual = float(b["floor_area_sqm"].median())
        if abs(np.log(spec["floor_area_sqm"] / usual)) > 0.10:
            allb = lp.blocks()
            same = allb[(allb["block"] == spec["block"]) & (allb["street_name"] == spec["street_name"]) &
                        (allb["flat_type"] == spec["flat_type"]) & ((allb["floor_area_sqm"] - spec["floor_area_sqm"]).abs() <= 1)]
            seen = f" ({len(same)} recorded sales at about this size)" if len(same) else ""
            notes.append(f"Less common size here: {spec['floor_area_sqm']:.0f} sqm{seen}, against the usual {usual:.0f} sqm, "
                         "so there are fewer direct comparisons.")
    if not np.isnan(comps) and abs(np.log(comps / fair)) > 0.08:
        notes.append(f"Recent sales in this block point to about {money(comps)}, {comps / fair - 1:+.0%} from our fair value.")

    out = dict(pred=pred, fair=fair, lease_now=lease0,
               # price at each quantile of comparable sales (widened), so a client can place any price on the scale
               quantiles=[[float(qq), float(pred * np.exp(v))] for qq, v in zip(meta["quantiles"], qs)], lo80=at(0.10), hi80=at(0.90), lo50=at(0.25), hi50=at(0.75),
               p05=at(0.05), p95=at(0.95), width=width, badge=badge, notes=notes, n_block_24m=len(w24), comps_est=comps,
               block_recent=w24.sort_values("sale_date", ascending=False),  # same flat type in this block, last 24 months
               offer=dict(opening=at(0.25), target=fair, walk_away=at(0.75)),
               # log offsets from fair value, for forecast.forecast(today=...)
               today=dict(fair=fair, m10=float(np.log(at(0.10) / fair)), m90=float(np.log(at(0.90) / fair)),
                          m25=float(np.log(at(0.25) / fair)), m75=float(np.log(at(0.75) / fair))))
    if asking:
        r = np.log(asking / pred)
        pct = float(np.interp(r, qs, meta["quantiles"]))
        beyond = "below" if r < qs[0] else "above" if r > qs[-1] else None
        out["asking"] = dict(price=asking, vs_fair=asking / fair - 1, percentile=pct, beyond=beyond, **deal_label(pct, beyond, badge))
    return out


def input_checks(spec):
    """Compare the entered size, lease start and floor with HDB's records for this block.

    Returns a list of dicts: field, message, and for size the block's usual area (`usual`) so the caller can show
    what the fair value would be at that size. Wrong inputs are the easiest way to get a wrong verdict."""
    out = []
    allb = lp.blocks()
    b = allb[(allb["block"] == spec["block"]) & (allb["street_name"] == spec["street_name"])]
    bt = b[b["flat_type"] == spec["flat_type"]]
    if bt.empty:
        out.append(dict(field="flat_type", message=f"HDB records show no sales of {spec['flat_type']} flats in this block."))
    else:
        lo, hi, usual = bt["floor_area_sqm"].min(), bt["floor_area_sqm"].max(), float(bt["floor_area_sqm"].median())
        area = spec["floor_area_sqm"]
        if area < lo - 1 or area > hi + 1:
            seen = f"{lo:.0f} sqm" if lo == hi else f"{lo:.0f} to {hi:.0f} sqm"
            out.append(dict(field="floor_area_sqm", usual=usual,
                            message=f"{spec['flat_type'].title()} flats in this block are {seen} in HDB records; you entered {area:.0f} sqm. "
                                    "Check the HDB-registered floor area: listings sometimes include extensions or round differently."))
    leases = sorted(int(x) for x in b["lease_commence_date"].unique())
    if leases and spec["lease_commence_date"] not in leases:
        out.append(dict(field="lease_commence_date",
                        message=f"HDB records give this block's lease start as {', '.join(map(str, leases))}; you entered {spec['lease_commence_date']}."))
    h = sales_history()
    hb = h[(h["block"] == spec["block"]) & (h["street_name"] == spec["street_name"])]
    if len(hb):
        top = max(int(r.split(" TO ")[1]) for r in hb["storey_range"].unique() if " TO " in r)
        if spec["storey_mid"] - 1 > top:
            out.append(dict(field="storey_mid", message=f"No sale in this block since 2017 was above storey {top}; check the floor you entered."))
    return out


def deal_label(pct, beyond, badge):
    """Big, plain verdict on the PRICE (not on whether to buy), from where it sits among comparable sales.

    pct = share of comparable sales that went for less. Low reliability adds 'LIKELY' because the range is wide."""
    if beyond == "below" or pct < 0.10:
        word, tone, line = "GREAT DEAL", "great", "Cheaper than about 9 in 10 comparable sales."
    elif pct < 0.30:
        word, tone, line = "GOOD DEAL", "good", f"Cheaper than about {1 - pct:.0%} of comparable sales."
    elif pct < 0.65:
        word, tone, line = "FAIR PRICE", "fair", "In line with what comparable flats sell for."
    elif pct < 0.90 and beyond != "above":
        word, tone, line = "PRICEY", "pricey", f"Dearer than about {pct:.0%} of comparable sales."
    else:
        word, tone, line = "OVERPRICED", "bad", "Dearer than about 9 in 10 comparable sales."
    if badge == "Low":
        word = "LIKELY " + word
    return dict(verdict=word, tone=tone, headline=line)


def describe(a):
    lines = [f"Fair value {money(a['fair'])}  |  likely {money(a['lo50'])} - {money(a['hi50'])}  |  wide {money(a['lo80'])} - {money(a['hi80'])}",
             f"Reliability: {a['badge']} (80% band is {a['hi80'] / a['lo80'] - 1:.0%} wide)"]
    lines += [f"  - {n}" for n in a["notes"]]
    if "asking" in a:
        k = a["asking"]
        lines.append(f"Asking {money(k['price'])}: {k['verdict']} - {k['headline']} ({k['vs_fair']:+.1%} vs fair value)")
        o = a["offer"]
        lines.append(f"Offer guide: open about {money(o['opening'])}, aim for {money(o['target'])}, walk away above {money(o['walk_away'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    sqm = 818 / 10.7639  # the 85 Whampoa Drive test listing
    for mid in (2, 8, 14):
        spec = dict(flat_type="3 ROOM", town="KALLANG/WHAMPOA", flat_model="IMPROVED", block="85", street_name="WHAMPOA DR",
                    floor_area_sqm=round(sqm, 1), storey_mid=float(mid), lease_commence_date=1974)
        print(f"=== 85 Whampoa Dr, 3-room Improved, {sqm:.0f} sqm, storey ~{mid}")
        print(describe(assess(spec, asking=420000)))
        print()
