"""Local app: estimate an HDB resale flat's price today and show scenarios for later.

Run:  streamlit run app.py
Needs the data and production model built first (python src/update.py).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import advice as adv  # noqa: E402
import forecast as fc  # noqa: E402
import listing_parser as lp  # noqa: E402

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

listing_text = st.text_area("Paste a listing's text (optional)", height=90,
                            help="Copy the text of a listing page and paste it here; the flat's details are filled in for you.")
p = lp.parse(listing_text) if listing_text.strip() else {}
if listing_text.strip():
    if p.get("block"):
        found = ", ".join(x for x in [f"blk {p['block']} {p['street_name']}", p.get("flat_type"), p.get("flat_model"),
                                      f"{p['floor_area_sqm']:.0f} sqm" if p.get("floor_area_sqm") else None,
                                      f"asking ${p['asking']:,}" if p.get("asking") else None] if x)
        st.success("Found: " + found.replace("$", chr(92) + "$") + ". Check the storey range below: listings rarely state it.")
    else:
        st.warning("Couldn't match this listing to an HDB block. Fill in the details below.")


def pick(label, options, wanted=None, **kw):
    options = list(options)
    return st.selectbox(label, options, index=options.index(wanted) if wanted in options else 0, **kw)


town = pick("Town", sorted(d["town"].unique()), p.get("town"))
dt = d[d["town"] == town]
street = pick("Street", sorted(dt["street_name"].unique()), p.get("street_name"))
ds = dt[dt["street_name"] == street]
block = pick("Block", sorted(ds["block"].unique(), key=block_key), p.get("block"))
db = ds[ds["block"] == block]
flat_type = pick("Flat type", sorted(db["flat_type"].unique()), p.get("flat_type"))
dbt = db[db["flat_type"] == flat_type]

k = f"{town}|{street}|{block}|{flat_type}|{hash(listing_text)}"  # changing the flat or the listing resets the defaults below
col1, col2 = st.columns(2)
with col1:
    flat_model = pick("Flat model", dbt["flat_model"].value_counts().index, p.get("flat_model"), key=f"model_{k}")
    area = st.number_input("Floor area (sqm)", 20.0, 400.0, float(p.get("floor_area_sqm") or dbt["floor_area_sqm"].median()), 1.0,
                           key=f"area_{k}")
with col2:
    ranges = sorted(d["storey_range"].unique())
    storey = st.selectbox("Storey range", ranges, index=ranges.index(dbt["storey_range"].mode().iloc[0]), key=f"storey_{k}")
    lease_year = st.number_input("Lease start year", 1960, 2026, int(dbt["lease_commence_date"].mode().iloc[0]), 1,
                                 key=f"lease_{k}")

asking = st.number_input("Asking price, if you have one ($)", 0, 5_000_000, int(p.get("asking") or 0), 1000, key=f"ask_{k}",
                         help="Leave at 0 to skip the asking-price check.")

low, high = [int(x) for x in re.findall(r"\d+", storey)]
spec = dict(flat_type=flat_type, town=town, flat_model=flat_model, block=block, street_name=street,
            floor_area_sqm=float(area), storey_mid=(low + high) / 2, lease_commence_date=int(lease_year))
try:
    a = adv.assess(spec, asking=asking or None)
    res = fc.forecast(spec, horizons=(3, 5, 10, 15, 20), today=a["today"])
except (ValueError, FileNotFoundError) as e:
    st.error(f"{e}. If a model file is missing, run `python src/update.py` first.")
    st.stop()

v0 = res["v0"]
checks = adv.input_checks(spec)
usd = lambda x: f"\\${x:,.0f}"  # a bare $ pair is read as maths by Streamlit's markdown
st.divider()
for c in checks:
    msg = c["message"]
    if c["field"] == "floor_area_sqm":
        alt = adv.assess({**spec, "floor_area_sqm": c["usual"]}, asking=asking or None)
        msg += f" At {c['usual']:.0f} sqm our fair value would be ${alt['fair']:,.0f}"
        msg += (f", and this asking price would read {alt['asking']['verdict']}." if "asking" in alt else ".")
    st.warning("Check your input: " + msg.replace("$", chr(92) + "$"))
# (word colour, background, side bar): bright green, green, yellow, orange, red.
# Yellow words are hard to read on a light background, so FAIR PRICE uses a yellow bar and tint with a dark gold word.
TONES = {"great": ("#0f9d3a", "#dcfce7", "#22c55e"), "good": ("#166534", "#ecf7ee", "#15803d"),
         "fair": ("#8a6100", "#fef9c3", "#eab308"), "pricey": ("#c2410c", "#ffedd5", "#f97316"),
         "bad": ("#b42318", "#fde9e7", "#dc2626")}
if "asking" in a:
    k_ = a["asking"]
    fg, bg, bar = TONES[k_["tone"]]
    st.markdown(f"<div style='background:{bg};border-left:12px solid {bar};padding:18px 24px;border-radius:12px;margin:6px 0 14px'>"
                f"<div style='font-size:52px;font-weight:800;letter-spacing:1px;color:{fg};line-height:1.05'>{k_['verdict']}</div>"
                f"<div style='font-size:19px;margin-top:8px;color:#1f2328'>{k_['headline']} Asking &#36;{k_['price']:,.0f} is "
                f"<b>{k_['vs_fair']:+.1%}</b> versus our fair value of &#36;{a['fair']:,.0f}.</div></div>", unsafe_allow_html=True)
st.subheader("Fair value today")
col_a, col_b = st.columns([3, 2])
col_a.metric("Fair value", f"${a['fair']:,.0f}")
col_b.markdown("Reliability  \n" + {"High": ":green[**High**]", "Medium": ":orange[**Medium**]", "Low": ":red[**Low**]"}[a["badge"]])
st.write(f"Likely range (50% of sales): **{usd(a['lo50'])} - {usd(a['hi50'])}**  \n"
         f"Wide range (80% of sales): **{usd(a['lo80'])} - {usd(a['hi80'])}**  \n"
         f"Remaining lease: {a['lease_now']:.0f} years.")
for n in a["notes"]:
    st.write("- " + n.replace("$", chr(92) + "$"))
st.caption("Reliability reflects how closely estimates for similar flats matched real sales in back-tests. "
           "The estimate cannot see renovation, the exact floor, view or condition.")

if "asking" in a:
    o = a["offer"]
    st.subheader("Offer guide")
    st.write(f"Open around **{usd(o['opening'])}**, aim for **{usd(o['target'])}**, walk away above **{usd(o['walk_away'])}**.")
    st.caption("From comparable-sale prices, not advice. Renovation, view and urgency can justify a different price.")

recent = dbt[dbt["sale_date"] >= dbt["sale_date"].max() - pd.DateOffset(months=12)].sort_values("sale_date", ascending=False)
st.subheader("Recent sales of this flat type in the block")
if recent.empty:
    st.info("No sales of this flat type in this block in the last 12 months.")
else:
    t = recent.head(12).assign(Month=lambda x: x["sale_date"].dt.strftime("%Y-%m"))
    st.dataframe(t[["Month", "storey_range", "floor_area_sqm", "flat_model", "resale_price"]].rename(columns={
        "storey_range": "Storey", "floor_area_sqm": "Area (sqm)", "flat_model": "Model", "resale_price": "Price ($)"}),
        hide_index=True, width="stretch",
        column_config={"Price ($)": st.column_config.NumberColumn(format="dollar")})

def short(x):
    return f"${x / 1e6:.2f}m" if x >= 1e6 else f"${x / 1e3:,.0f}k"


st.subheader("Outlook")
default_rate = float(np.expm1(res["growth_effective"]))
own = st.checkbox("Use my own market growth assumption", value=False,
                  help="Replaces the model's assumption. Ranges keep the same spread as in back-tests, around your number.")
if own:
    rate = st.slider("Market growth per year (%)", 0.0, 8.0, float(np.clip(round(default_rate * 400) / 4, 0.0, 8.0)), 0.25) / 100
    res = fc.forecast(spec, horizons=(3, 5, 10, 15, 20), growth=rate)

by_h = {r["years"]: r for r in res["rows"]}
for col, h in zip(st.columns(2), (3, 5)):
    r = by_h[h]
    with col:
        st.markdown(f"**In {h} years**")
        if "central" not in r:
            st.write(f"Not projected: {r['note'].replace('not projected: ', '')}.")
            continue
        st.metric("Central", short(r["central"]), f"{r['central'] / v0 - 1:+.0%} vs today", delta_color="off")
        md = lambda x: short(x).replace("$", chr(92) + "$")  # a bare $ pair is read as maths by Streamlit's markdown
        st.write(f"Likely range: {md(r['likely_low'])} - {md(r['likely_high'])}")
        st.write(f"Wide range: {md(r['low'])} - {md(r['high'])}")

with st.expander("Longer horizons (10, 15 and 20 years)"):
    rows = []
    for h in (10, 15, 20):
        r = by_h[h]
        if "central" not in r:
            rows.append({"In years": h, "Lease left (yrs)": r["lease_left"], "Note": r["note"]})
        else:
            rows.append({"In years": h, "Lease left (yrs)": r["lease_left"], "Central": short(r["central"]),
                         "Likely range": f"{short(r['likely_low'])} - {short(r['likely_high'])}",
                         "Wide range": f"{short(r['low'])} - {short(r['high'])}", "Note": ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("These rest on fewer past start years, so they are the least reliable figures here.")

if own:
    st.caption(f"Your scenario: {rate:.2%} a year market growth plus lease decay. Ranges show the usual spread from back-tests "
               "around your number, not a prediction of your assumption. ")
else:
    st.caption(f"Projections, not guarantees. The model's central case works out to about {default_rate:.1%} a year market growth plus "
               "lease decay. Likely range = 50% and wide range = 80% of outcomes in back-tests on past start years.")
st.caption("No figure is shown where the lease would fall under 50 years, which the back-tests do not cover. "
           "Not a valuation for loans, sales or legal purposes. Based on public HDB transaction data.")
