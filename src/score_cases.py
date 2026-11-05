"""Score the model on flats you know: python src/score_cases.py [cases/known_cases.csv]

Each row is a flat, entered by hand or from a pasted listing (column listing_file = a text file with the
pasted listing; its fields fill any blanks). Columns (blank = unknown):
  name, listing_file, block, street_name, flat_type, flat_model, floor_area_sqm, storey_range (e.g. "07 TO 09"),
  lease_commence_date, asking_price, actual_price, actual_date (YYYY-MM), notes

For every flat: fair value, 50% / 80% ranges, reliability, and the deal verdict on the asking price.
Where the actual price is known: error, whether it fell inside the ranges, and whether the verdict pointed
the right way (a PRICEY / OVERPRICED call is borne out if the flat sold below its asking price).
A sale on or before the model's training month may already be in the training data: such rows are flagged,
because the score would look better than it should.

Output: printed table + summary, and reports/case_scores.csv
"""
import json
import re
import sys

import numpy as np
import pandas as pd

import advice as adv
import listing_parser as lp
from geocode_blocks import ROOT

FIELDS = ["block", "street_name", "flat_type", "flat_model", "floor_area_sqm", "lease_commence_date"]


def storey_mid(text):
    nums = [int(x) for x in re.findall(r"\d+", str(text))]
    return (nums[0] + nums[1]) / 2 if len(nums) >= 2 else None


def row_spec(r):
    """Build the flat spec from the row, filling blanks from the pasted listing if given."""
    got = {k: r.get(k) for k in FIELDS + ["asking_price"]}
    if isinstance(r.get("listing_file"), str) and r["listing_file"].strip():
        parsed = lp.parse((ROOT / r["listing_file"]).read_text(encoding="utf-8"))
        for k in FIELDS:
            if pd.isna(got.get(k)) and parsed.get(k) is not None:
                got[k] = parsed[k]
        if pd.isna(got.get("asking_price")) and parsed.get("asking"):
            got["asking_price"] = parsed["asking"]
    match = lp.blocks()[(lp.blocks()["block"] == str(got["block"]).upper()) & (lp.blocks()["street_name"] == got["street_name"])]
    if match.empty:
        raise ValueError(f"block {got['block']} {got['street_name']} not found in HDB data")
    spec = dict(flat_type=got["flat_type"], town=match["town"].iloc[0], flat_model=str(got["flat_model"]).upper(),
                block=str(got["block"]).upper(), street_name=got["street_name"], floor_area_sqm=float(got["floor_area_sqm"]),
                storey_mid=storey_mid(r.get("storey_range")), lease_commence_date=int(float(got["lease_commence_date"])))
    if spec["storey_mid"] is None:
        raise ValueError("storey_range missing (e.g. 07 TO 09)")
    asking = None if pd.isna(got.get("asking_price")) else float(got["asking_price"])
    return spec, asking


def main():
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "cases/known_cases.csv")
    cases = pd.read_csv(path, dtype=str)  # blanks come in as NaN
    trained_through = json.loads((ROOT / "models/lgbm_prod.json").read_text())["trained_through"]
    rows = []
    for _, r in cases.iterrows():
        rec = dict(name=r.get("name"))
        try:
            spec, asking = row_spec(r)
            a = adv.assess(spec, asking=asking)
        except Exception as e:  # report and keep going
            rec["problem"] = str(e)
            rows.append(rec)
            continue
        rec.update(flat=f"blk {spec['block']} {spec['street_name']}, {spec['flat_type']} {spec['flat_model']}, "
                        f"{spec['floor_area_sqm']:.0f} sqm, storey ~{spec['storey_mid']:.0f}, lease {spec['lease_commence_date']}",
                   fair=round(a["fair"]), lo50=round(a["lo50"]), hi50=round(a["hi50"]), lo80=round(a["lo80"]), hi80=round(a["hi80"]),
                   reliability=a["badge"], notes=" | ".join(a["notes"]))
        if "asking" in a:
            rec.update(asking=round(asking), verdict=a["asking"]["verdict"], vs_fair=round(a["asking"]["vs_fair"], 4),
                       walk_away=round(a["offer"]["walk_away"]))
        if not pd.isna(r.get("actual_price")):
            actual = float(r["actual_price"])
            rec.update(actual=round(actual), error=round(a["fair"] / actual - 1, 4),
                       # plain bools: numpy bools in an object column add up as logical OR, which breaks the averages
                       in50=bool(a["lo50"] <= actual <= a["hi50"]), in80=bool(a["lo80"] <= actual <= a["hi80"]),
                       maybe_in_training=bool(isinstance(r.get("actual_date"), str) and r["actual_date"][:7] <= trained_through))
            if "asking" in a:
                said_high = a["asking"]["tone"] in ("pricey", "bad")
                rec["verdict_borne_out"] = bool(actual < asking) if said_high else bool(actual >= asking * 0.97)
        rows.append(rec)

    out = pd.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    out.to_csv(ROOT / "reports/case_scores.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    show = [c for c in ["name", "fair", "lo80", "hi80", "reliability", "asking", "verdict", "actual", "error", "in80", "maybe_in_training", "problem"]
            if c in out.columns]
    print(out[show].to_string(index=False))
    if "actual" in out.columns and out["actual"].notna().any():
        k = out[out["actual"].notna()]
        clean = k[~k["maybe_in_training"].astype(bool)]
        print(f"\nflats with a known price: {len(k)} ({len(clean)} not possibly in training)")
        for label, g in [("all", k), ("not in training", clean)]:
            if len(g):
                print(f"  {label:16} median abs error {g['error'].abs().median():.1%}, inside 80% range {g['in80'].mean():.0%}, "
                      f"inside 50% range {g['in50'].mean():.0%}"
                      + (f", verdict borne out {g['verdict_borne_out'].mean():.0%}" if "verdict_borne_out" in g and g["verdict_borne_out"].notna().any() else ""))
    print("\nsaved reports/case_scores.csv")


if __name__ == "__main__":
    main()
