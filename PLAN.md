# Singapore HDB Resale Price and Forecast Model: Plan

## Goal
Estimate the current value of an HDB resale flat, then give a range for its
future resale value. Personal project. Keep money spent as close to S$0 as
possible, and accept slower runs to get there.

## Constraints
- Start on the laptop (i7-1270P, 16GB RAM, no NVIDIA GPU, 208GB free).
- Stay portable so the project can move to the PC with the RTX 5070 later.
- Use free, public data only. Do not scrape property portals.
- Stop and review at the go/no-go step before adding anything paid.

## Portability rules
- Use relative paths, set through one config file.
- Pin dependencies in `requirements.txt`.
- Keep code in Git. Keep `data/` and `models/` out of Git.
- Make every step rerunnable by script, so data can be rebuilt on any machine.
- Save models in a standard format (LightGBM text files).
- Set CPU or GPU in config, not in the code.

## Layout
```
realestateai/
  PLAN.md
  README.md
  requirements.txt
  config.yaml
  data/
    raw/         # downloads, never edited
    processed/   # cleaned data and features
  models/
  notebooks/
  reports/
  src/
```

## Steps
1. Data (days 1 to 3): download HDB resale data from data.gov.sg, clean it,
   and adjust prices for market movement over time.
2. Features (days 4 to 6): add OneMap coordinates, distances to MRT stations
   and schools, and remaining lease.
3. Baseline (day 7): benchmark against recent sales of similar flats nearby.
4. Price model (days 8 to 10): train LightGBM, test on the latest months only,
   and report error by town and flat type.
5. Forecast prototype (weeks 3 to 6): combine market scenarios with lease
   decay, then backtest on past years.
6. Go/no-go: decide on a web app, private property, or photos from the results.

## Progress
- Step 1 done: 990,026 clean sales, 1990-01 to 2026-10 (`src/clean_data.py`).
- Step 2 done: 98.5% of blocks geocoded; MRT, hawker and school distances added
  (`src/build_features.py`). Known issue: MRT exits are as of Dec 2025, applied
  to all years. Fix queued: add station opening years.
- Step 3 done: baseline (`src/baseline.py`), test = 2026-01 to 2026-09, trained
  to 2025-12. Same-block comparables: median error 5.06%, 80% of sales within 10%.
  This is the number the model must beat.
- Step 4 done (v1): LightGBM (`src/train_model.py`, `models/lgbm_v1.txt`), fit
  2017-01..2025-06, early stop 2025-07..12, test 2026-01..09. Median error 3.12%
  (mean 4.14%, 92.7% within 10%) vs baseline 5.06%. Error rises from ~3.6% in
  Jan to ~4.7% in Sep 2026. CORRECTION: I first blamed market drift, but median
  price per sqm was flat from 2025-01 to 2026-09. Untested hypothesis: trees
  cannot extrapolate lease decay (later sales of a block have less remaining
  lease than anything seen in training). MRT distance has only ~1% of gain, so
  station opening years may not be worth the effort yet.
- Step 4 experiment v2 (not adopted): recent-nearby-sales features
  (`src/neighbor_features.py`, `train_model.py --v2`). Test median error 3.70%
  (3.91% without the time trend), worse than v1, with +2.5% overprediction.
  v1 remains the best model.
- Lease check: lease-extrapolation hypothesis only weakly supported (error
  2.81% -> 3.49% as sales fall further below a block's training lease range).
  Clearer pattern: overprediction grows as lease shrinks (+2.6% under 55 yrs).
- Recency weighting (36-month half-life) + monotone lease constraint, tried
  together (`--recency 36 --monotone`): test median 3.10% vs 3.12%, no real
  change. Month-by-month error rise unchanged. Not a meaningful improvement.
  Trees cannot extrapolate below the lease range seen in training, so step 5
  needs an explicit lease-decay curve, not the tree model alone.
- Refit (`--recency 36 --monotone --refit`, 22s): retrained on all data to
  2025-12 with fixed rounds. Test median 2.83%, mean 3.80%, 94.0% within 10%.
  Best model so far: `models/lgbm_refit.txt`. Gain is a constant ~0.3 points
  per month; the rise in error from Jan (3.3%) to Sep (4.3%) 2026 remains
  unexplained. Whole training script runs in about a minute on the laptop, so
  no GPU is needed. Retrain regularly.

## Storage budget
| Stage | Estimate |
|---|---|
| Python environment and libraries | 2 to 3 GB (5 to 8 GB with PyTorch) |
| HDB and location data | Under 1 GB |
| Processed data and models | 1 to 2 GB |
| TimesFM weights | Under 2 GB |
| Steps 1 to 5 total | About 10 GB, keep 20 GB free |
| Optional photo stage | 20 to 50 GB more |

## Open items
- Confirm dataset coverage, API limits, and licence terms before step 1.
- Confirm the RTX 5070 machine's specs when it is time to move.
