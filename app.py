"""Local app: estimate an HDB resale flat's price today and show scenarios for later.

Run:  streamlit run app.py
Needs the data and production model built first (python src/update.py).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import forecast as fc  # noqa: E402

st.set_page_config(page_title="HDB resale estimate", page_icon="🏠", layout="centered")


@st.cache_data(show_spinner="Loading sales data...")
def load_sales():
    cols = ["town", "street_name", "block", "flat_type", "flat_model", "floor_area_sqm", "storey_range",
            "lease_commence_date", "sale_date", "resale_price"]
    d = pd.read_parquet(ROOT / "data/processed/resale_features.parquet", columns=cols)
    return d[d["sale_date"] >= "2017-01-01"].reset_index(drop=True)


def block_key(b):
    m = re.match(r"\d+", b)
    return (int(m.group()) if m else 0, b)


meta_path = ROOT / "models/lgbm_prod.json"
if not meta_path.exists():
    st.error("No trained model found. Run `python src/update.py` first.")
    st.stop()
meta = json.loads(meta_path.read_text())
d = load_sales()

st.title("HDB resale estimate")
st.caption(f"Model trained on sales through {meta['trained_through']}. On the six most recent months it was held out "
           f"from, the median error was {meta['holdout_median_error']:.1%}.")

town = st.selectbox("Town", sorted(d["town"].unique()))
dt = d[d["town"] == town]
street = st.selectbox("Street", sorted(dt["street_name"].unique()))
ds = dt[dt["street_name"] == street]
block = st.selectbox("Block", sorted(ds["block"].unique(), key=block_key))
db = ds[ds["block"] == block]
flat_type = st.selectbox("Flat type", sorted(db["flat_type"].unique()))
dbt = db[db["flat_type"] == flat_type]

k = f"{town}|{street}|{block}|{flat_type}"  # changing the flat resets the defaults below
col1, col2 = st.columns(2)
with col1:
    flat_model = st.selectbox("Flat model", dbt["flat_model"].value_counts().index.tolist(), key=f"model_{k}")
    area = st.number_input("Floor area (sqm)", 20.0, 400.0, float(dbt["floor_area_sqm"].median()), 1.0, key=f"area_{k}")
with col2:
    ranges = sorted(d["storey_range"].unique())
    storey = st.selectbox("Storey range", ranges, index=ranges.index(dbt["storey_range"].mode().iloc[0]), key=f"storey_{k}")
    lease_year = st.number_input("Lease start year", 1960, 2026, int(dbt["lease_commence_date"].mode().iloc[0]), 1,
                                 key=f"lease_{k}")

low, high = [int(x) for x in re.findall(r"\d+", storey)]
spec = dict(flat_type=flat_type, town=town, flat_model=flat_model, block=block, street_name=street,
            floor_area_sqm=float(area), storey_mid=(low + high) / 2, lease_commence_date=int(lease_year))
try:
    res = fc.forecast(spec)
except ValueError as e:
    st.error(str(e))
    st.stop()

v0 = res["v0"]
b_lo, b_hi = res["band"]
st.divider()
st.subheader("Estimated price today")
st.metric("Estimate", f"${v0:,.0f}")
usd = lambda x: f"\\${x:,.0f}"  # a bare $ pair is read as maths by Streamlit's markdown
st.write(f"Most sales (8 in 10) land between **{usd(v0 * (1 + b_lo))}** and **{usd(v0 * (1 + b_hi))}**. "
         f"Remaining lease: {res['lease_now']:.0f} years.")
st.caption("Based on size, floor range, lease, location and recent market level. It cannot see renovation, the exact "
           "floor, view or condition, so a flat that is much better or worse than its neighbours will be missed.")

recent = dbt[dbt["sale_date"] >= dbt["sale_date"].max() - pd.DateOffset(months=12)].sort_values("sale_date", ascending=False)
st.subheader("Recent sales of this flat type in the block")
if recent.empty:
    st.info("No sales of this flat type in this block in the last 12 months.")
else:
    t = recent.head(12).assign(Month=lambda x: x["sale_date"].dt.strftime("%Y-%m"))
    st.dataframe(t[["Month", "storey_range", "floor_area_sqm", "flat_model", "resale_price"]].rename(columns={
        "storey_range": "Storey", "floor_area_sqm": "Area (sqm)", "flat_model": "Model", "resale_price": "Price ($)"}),
        hide_index=True, use_container_width=True,
        column_config={"Price ($)": st.column_config.NumberColumn(format="dollar")})

with st.expander("What could it be worth later? (scenarios, not predictions)"):
    st.warning("Nobody can predict 20 years of market movement, and the lease-decay effect could not be confirmed in "
               "back-testing: the fitted curve overstated value loss, so it is applied at 0%, 50% and 100% strength.")
    rows = []
    for r in res["rows"]:
        if "grid" not in r:
            rows.append({"In years": r["years"], "Lease left": r["lease_left"], "Note": r["note"]})
            continue
        g = r["grid"]
        rows.append({"In years": r["years"], "Lease left": r["lease_left"],
                     "Lease effect, full curve": f"{r['lease_full']:+.0%}",
                     "Low ($)": f"{g[('stress 0%/yr', 'full curve')]:,.0f}",
                     "Central ($)": f"{g[('recent 2%/yr', 'half curve')]:,.0f}",
                     "High ($)": f"{g[('long-run 4%/yr', 'no decay')]:,.0f}",
                     "Note": r["note"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Low = flat market and the full lease curve. Central = 2% a year growth and half the curve. "
               "High = 4% a year growth and no lease decay. Years with under 45 years of lease left use an extrapolated curve.")

st.caption("Not a valuation for loans, sales or legal purposes. Estimates come from public HDB transaction data.")
