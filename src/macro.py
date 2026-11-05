"""Download macro series used to test whether they improve long-run growth forecasts.

World Bank (annual, no key): GDP growth, inflation, lending rate, real interest rate,
population, real GDP per capita. FRED (quarterly, no key): BIS residential property
price index for Singapore (nominal and real).

Output: data/raw/macro/*.csv (one file per series, columns date,value). Rerunnable; skips existing files.
"""
import time

import pandas as pd
import requests

from geocode_blocks import ROOT

OUT = ROOT / "data/raw/macro"
WB = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation": "FP.CPI.TOTL.ZG",
    "lending_rate": "FR.INR.LEND",
    "real_rate": "FR.INR.RINR",
    "population": "SP.POP.TOTL",
    "gdp_pc_real": "NY.GDP.PCAP.KD",
}
FRED = {"bis_real_property": "QSGR628BIS", "bis_nominal_property": "QSGN628BIS"}


def get(url, tries=5):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"could not fetch {url}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, code in WB.items():
        path = OUT / f"{name}.csv"
        if path.exists():
            print(f"skip  {name}")
            continue
        r = get(f"https://api.worldbank.org/v2/country/SGP/indicator/{code}?format=json&per_page=100&date=1985:2025")
        rows = [(x["date"], x["value"]) for x in r.json()[1] if x["value"] is not None]
        pd.DataFrame(rows, columns=["date", "value"]).sort_values("date").to_csv(path, index=False)
        print(f"saved {name}: {len(rows)} values, {min(d for d, _ in rows)} to {max(d for d, _ in rows)}")
    for name, code in FRED.items():
        path = OUT / f"{name}.csv"
        if path.exists():
            print(f"skip  {name}")
            continue
        d = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}")
        d.columns = ["date", "value"]
        d = d[pd.to_numeric(d["value"], errors="coerce").notna()]
        d.to_csv(path, index=False)
        print(f"saved {name}: {len(d)} values, {d['date'].min()} to {d['date'].max()}")


if __name__ == "__main__":
    main()
