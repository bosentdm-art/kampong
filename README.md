<img src="web/icons/logo.svg" width="56" alt="Kampong logo: a grid of HDB windows with one lit">

# Kampong

**Is that flat really worth the asking price?** Kampong is a chat-first price checker for Singapore homes.
Paste a property listing (or a link to one) and it compares the flat with what similar flats actually sold for,
then gives a verdict, an offer range and what's around the home on an interactive 3D map.

A personal project, built end to end: data pipeline, machine-learning models, API, front end, AI chat and deployment.

![3D map of the area around an HDB block, with MRT, schools, bus stops and shops pinned](docs/map3d.jpg)

## Highlights

- **Accurate prices.** LightGBM models trained on every HDB resale since 2017 (about 245,000 sales) and on
  URA condo transactions. Half of HDB estimates land within **2.7%** of the price the flat later sold for, tested
  on months the model had never seen (a naive baseline misses by 5.1%). Condos: 2.9%. Price ranges are calibrated,
  and each month's estimates are checked against real sales.
- **Honest experiments.** New features are only adopted if they beat the current model on more than one held-out
  period. For example, block height, nearby new supply and planned MRT lines were tested and rejected
  (`src/experiments_hdb_extra.py`).
- **AI that can't make numbers up.** The chat uses Claude with tools that call the price engine. Every dollar figure,
  percentage and date in an answer is checked against the facts; a failed check is retried, then replaced with
  rule-based text.
- **Reads real listings.** Pasted listings are parsed into facts (block, size, floor, price, renovation, view),
  with contact details scrubbed. A pasted link is read once, with an honest user agent, respecting robots.txt
  ([how](web/bosan.html)).
- **3D area map.** MapLibre GL over OneMap tiles, with OpenStreetMap buildings, the home's own block highlighted,
  and MRT, planned stations, schools, bus stops and daily needs pinned.
- **Accessible.** Checked with GovTech's Oobee scanner (0 must-fix issues): keyboard-usable maps, screen-reader
  labels, 4.5:1 contrast, reduced-motion support.
- **Runs cheaply.** FastAPI + SQLite on one machine, exposed through a Cloudflare Tunnel, with nightly backups to
  Cloudflare R2. Optional Stripe billing and WorkOS sign-in.

**Stack:** Python, pandas, LightGBM, FastAPI, SQLite, vanilla JavaScript, MapLibre GL, Leaflet, Claude API,
Cloudflare (Tunnel, R2, Turnstile), Stripe. **Data:** data.gov.sg (HDB resale, HDB property information), URA, OneMap,
LTA/OpenStreetMap, all public.

The original modelling plan and its go/no-go decisions are in [PLAN.md](PLAN.md).

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

## Check flats you know
Add rows to `cases/known_cases.csv` (or point `listing_file` at a text file of a pasted listing), then:
```
.venv\Scripts\python src\score_cases.py
```

## Kampong web app (API + front end)

Kampong (called Oikos until November 2026) is the product; the company behind it is Oiko. Internal names such as the `oikos_session` cookie, `data/oikos.db` and `OIKOS_*` settings keep the old name on purpose.
```
.venv\Scripts\python -m uvicorn api:app --port 8700
```
Then open http://localhost:8700. `api.py` wraps the models (`/api/options`, `/api/parse`, `/api/assess`)
and serves the page in `web/`. The Streamlit app (`app.py`) still works and is now the fallback.

## Ask Kampong (the AI explanations)
Add `ANTHROPIC_API_KEY=...` to `.env`. The model (default `claude-haiku-5-5`, set `OIKOS_LLM_MODEL` to change it)
does not learn from our data: for each question it is handed that flat's facts and two tools that call the
engine, then explains them. Every dollar figure, percentage, year count and size in its answer is checked against
those facts; a failed check is retried once, then replaced by rule-based text. Without a key the tab shows the
rule-based text. Code: `src/llm.py`; endpoint: `POST /api/ask`.
