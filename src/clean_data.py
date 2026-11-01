"""Combine the five raw HDB resale files into one clean table.

Output: data/processed/resale_clean.parquet

What it does:
- stacks all files, dropping the remaining_lease text column (it only exists
  from 2015 and is not at the sale date for older files; we compute our own)
- standardises text (upper case, consistent flat type and model names)
- parses the sale month and the storey range
- computes remaining lease in years at the time of sale
- drops obviously invalid rows (it keeps look-alike rows, which can be real sales)
"""
import glob
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEASE_YEARS = 99


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw_dir = ROOT / cfg["paths"]["raw"]
    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in sorted(glob.glob(str(raw_dir / "*.csv"))):
        d = pd.read_csv(f).drop(columns=["remaining_lease"], errors="ignore")
        d["source_file"] = Path(f).stem
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)

    for col in ["town", "flat_type", "block", "street_name", "storey_range", "flat_model"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["flat_type"] = df["flat_type"].str.replace("-", " ", regex=False)  # MULTI-GENERATION vs MULTI GENERATION
    df["block"] = df["block"].str.replace(r"\s+", "", regex=True)

    # Sale date
    df["sale_date"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month

    # Storey range "10 TO 12" -> low, high, mid
    st = df["storey_range"].str.extract(r"(\d+)\s+TO\s+(\d+)").astype(float)
    df["storey_low"], df["storey_high"] = st[0], st[1]
    df["storey_mid"] = (st[0] + st[1]) / 2

    # Remaining lease in years at the time of sale (month precision)
    sale_year_frac = df["sale_year"] + (df["sale_month"] - 1) / 12
    df["remaining_lease_years"] = LEASE_YEARS - (sale_year_frac - df["lease_commence_date"])
    df["flat_age_years"] = sale_year_frac - df["lease_commence_date"]

    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]

    # Remove rows that cannot be right. Identical-looking rows are kept: two
    # different flats in one block can share storey range, area, month and price,
    # and the data has no unit number to tell them apart.
    bad = (
        (df["resale_price"] <= 0)
        | (df["floor_area_sqm"] <= 0)
        | df["storey_mid"].isna()
        | (df["remaining_lease_years"] <= 0)
        | (df["remaining_lease_years"] > LEASE_YEARS)
    )
    n_bad = int(bad.sum())
    df = df[~bad].reset_index(drop=True)

    out = out_dir / "resale_clean.parquet"
    df.to_parquet(out, index=False)

    print(f"raw rows: {n_raw}")
    print(f"dropped invalid: {n_bad}")
    print(f"clean rows: {len(df)} -> {out.relative_to(ROOT)}")
    print("flat types:", sorted(df["flat_type"].unique()))
    print("towns:", df["town"].nunique(), " flat models:", df["flat_model"].nunique())
    print(df[["resale_price", "floor_area_sqm", "remaining_lease_years", "storey_mid"]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
