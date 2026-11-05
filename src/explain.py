"""Data-grounded explanation of a price estimate: no LLM, only the trained models and the sales data.

1. Drivers: LightGBM's built-in tree attributions (pred_contrib), averaged over the 3 production models,
   grouped into location / size / floor / lease / type / market timing. In log-price space the
   contributions add up, so each group is a multiplicative effect: price = typical flat today x group effects.
2. What-ifs: re-price the flat with one input changed (higher floor, bigger, longer lease).

Usage: python src/explain.py   (demo on two real recent sales)
"""
import numpy as np
import pandas as pd

import forecast as fc

GROUPS = {
    "Location and surroundings": ["lat", "lon", "town", "dist_mrt", "n_mrt_exits_500m", "dist_hawker", "dist_primary_school",
                                  "n_primary_1km", "n_primary_2km", "dist_cbd"],
    "Size": ["floor_area_sqm"],
    "Floor level": ["storey_mid", "storey_low", "storey_high"],
    "Lease and age": ["remaining_lease_years", "lease_commence_date"],
    "Flat type and model": ["flat_type", "flat_model"],
    "Market timing": ["month_idx", "sale_month"],
}


def drivers(spec, when=None):
    when = fc.default_when() if when is None else when
    row = fc.block_table().query("block == @spec['block'] and street_name == @spec['street_name']").iloc[0].to_dict()
    x = fc.feature_frame(spec, row, when)
    contrib = np.mean([b.predict(x, pred_contrib=True)[0] for b in fc.load_models()], axis=0)  # last entry = baseline
    by_feature = dict(zip(list(x.columns) + ["_base"], contrib))
    base = float(np.exp(by_feature.pop("_base")))
    effects = {g: float(np.exp(sum(by_feature[f] for f in fs)) - 1) for g, fs in GROUPS.items()}
    # "market timing" only says prices have moved since the model's average year, so fold it into the baseline:
    # the baseline becomes a typical flat in TODAY's market, and the groups left are about this flat.
    base *= 1 + effects.pop("Market timing")
    return base, effects, float(np.exp(sum(contrib)))


def what_if(spec, when=None):
    when = fc.default_when() if when is None else when
    row = fc.block_table().query("block == @spec['block'] and street_name == @spec['street_name']").iloc[0].to_dict()
    v0 = fc.price_today(spec, row, when)[0]
    cases = {"3 storeys higher": dict(storey_mid=spec["storey_mid"] + 3), "3 storeys lower": dict(storey_mid=max(spec["storey_mid"] - 3, 2)),
             "10 sqm larger": dict(floor_area_sqm=spec["floor_area_sqm"] + 10), "10 years newer lease": dict(lease_commence_date=spec["lease_commence_date"] + 10)}
    return {k: fc.price_today({**spec, **v}, row, when)[0] / v0 - 1 for k, v in cases.items()}


if __name__ == "__main__":
    df = pd.read_parquet(fc.PROC / "resale_features.parquet")
    recent = df[(df["sale_date"] >= "2026-08-01") & (df["flat_type"] == "4 ROOM")]
    for town in ["TOA PAYOH", "BEDOK"]:
        s = recent[recent["town"] == town].sample(1, random_state=3).iloc[0]
        spec = dict(flat_type=s["flat_type"], town=s["town"], flat_model=s["flat_model"], block=s["block"], street_name=s["street_name"],
                    floor_area_sqm=float(s["floor_area_sqm"]), storey_mid=float(s["storey_mid"]), lease_commence_date=int(s["lease_commence_date"]))
        base, eff, total = drivers(spec)
        print(f"=== {s['flat_type']} {s['town']} blk {s['block']} {s['street_name']}, {s['floor_area_sqm']:.0f} sqm, storey {s['storey_range']}, "
              f"lease {s['lease_commence_date']}; sold ${s['resale_price']:,.0f}")
        print(f"typical flat in today's market: ${base:,.0f}  ->  this flat: ${total:,.0f}")
        for g, e in sorted(eff.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {g:28} {e:+7.1%}")
        print("  what if:", {k: f"{v:+.1%}" for k, v in what_if(spec).items()})
        print()
