# realestateai

Singapore HDB resale price and future-value model. See [PLAN.md](PLAN.md).

## Setup (any machine)
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\download_data.py
```
`data/` and `models/` are not in Git. The download script rebuilds `data/raw/`,
and skips files that already exist (use `--force` to refresh).

## Data
HDB resale flat prices from data.gov.sg, split into five files (1990 to now).
All five share the same columns, except `remaining_lease`, which only exists
from 2015 onward.

## Use it
```
.venv\Scripts\python src\update.py      # refresh data, rebuild features, retrain (about 3 minutes)
.venv\Scripts\python -m streamlit run app.py    # local app: price today plus scenarios for later
```
`update.py` needs `ONEMAP_TOKEN` in `.env` only to geocode brand-new blocks; without it those
sales are left out until the token is renewed. If data.gov.sg's download link stalls, the
download step falls back to its paged query API automatically.

## How well does it work?
- Price today: median error about 2.8% on months the model had not seen (baseline: 5.1%).
- Forecasts: scenarios only. The lease-decay effect failed back-testing (see PLAN.md).
