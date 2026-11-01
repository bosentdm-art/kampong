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
