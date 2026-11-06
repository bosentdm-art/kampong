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

- Step 5 stage A done (`src/forecast_components.py`): split a forecast into
  market level x lease decay. Pooled 1990+ lease curve was non-monotone
  (rejected). Fitted on 2016+: value vs a 90-year flat is 88% at 80 yrs, 76% at
  60, 64% at 50, 53% at 40. Long market index (1990+) uses that curve as a fixed
  offset: 10-year annualised growth p10 +1.3%, median +4.6%, p90 +11.2%.
  Caveats: only ~3 independent decades; curve assumed stable over time;
  nothing below 38 years of lease in the data.
- Region check: lease decay differs by distance-from-CBD tercile (cut-offs
  10.75km / 14.74km): at 60 yrs left value vs a 90-yr flat is 72% near, 77% mid,
  80% far; market growth differs little (10y median 5.2/4.6/4.4%). Mid-band
  curve rebounded below 45 yrs, so curves are forced non-increasing and
  extended with the 60-50 yr slope below 50 yrs; <45 yrs is flagged.
- Step 5 stage B done (`src/forecast.py`): V0 x exp(g*h) x lease ratio. First
  version used market-growth percentiles from the 1990+ index and gave absurd
  ranges (1990s boom: ~3.7x in 6 years). Replaced by explicit growth scenarios
  (0% stress, 2% recent, 4% long-run) because 10-year growth ranged +1% to +11%
  a year by period, and 20-year growth cannot honestly be forecast. The lease
  effect is shown on its own; the low/high band is only the model's error today.
- Step 5 stage C done (`src/backtest.py`, `reports/backtest_lease_curve.csv`):
  lease curve fitted on the 6 years before each origin (2010/2013/2016/2018),
  then block-level realised price change over 5/8/10 yrs regressed on the change
  the curve predicts (slope 1 = calibrated). RESULT: NOT VALIDATED. Slopes
  0.56/0.34/0.05 (2010), -0.30/-0.34/0.02 (2013), 0.02/0.12/0.06 (2016),
  0.73/0.73 (2018); mean about 0.2; correlation 0.05-0.3. The curve overstates
  value loss on average. Region split not supported either (slopes noisy, near
  often negative). Likely cause: flat age and estate maturity/location are tied
  together, so public data cannot separate lease decay from location trends.
  Consequences: today's price (2.83% model) is unaffected; the lease effect in
  `forecast.py` is an upper bound, now labelled with a warning.
- Step 5 cautious version done: one pooled lease curve (region split dropped),
  lease effect applied at 0% / 50% / 100% strength, market growth 0% / 2% / 4%.
  Output per horizon: low = flat market + full curve, central = 2% + half the
  curve, high = 4% + no lease decay. Example: 4-room Tampines (58 yrs left,
  $632k today) in 10 years: $525k / $703k / $943k.
- Status: valuation today is validated (2.83% median error); long-run forecasts
  are wide scenarios, not predictions. Open options: better lease-effect
  identification (repeat sales with neighbourhood controls), station opening
  years, or a simple web app for today's price.

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

## Latest (2026-11)
- Monthly retraining: `src/update.py` runs download, clean, geocode new blocks,
  features and `src/train_production.py` in about 160s. Production model:
  `models/lgbm_prod.txt` (+ `.json` metadata), fixed settings (recency half-life
  36 months, monotone lease, 1,281 rounds), trained through the latest complete
  month. Built-in holdout (last 6 months): median error 2.83%.
- data.gov.sg's download link stalled (status stuck at STARTING); added a
  fallback to the paged query API with a guard against partial results.
- Local app: `app.py` (Streamlit). Price today with an 80% range, recent block
  sales, and the scenario table with its warning. Tested in the browser pane.
- Re-tune (`train_model.py`, especially the number of rounds) every few months.

## Accuracy and long-run forecasts (2026-11)
- Noise floor: near-identical flats (same block, month, storey range, area) differ by a
  median 1.9% (mean 2.6%) from their group average, so no model on public data can go
  much below ~2%. Production holdout error: 2.74% median (was 2.83%).
- Accuracy experiments (`src/experiments_accuracy.py`, 6-month holdout): bigger or smaller
  trees no gain; extra features (lease start, sale month, storey bounds) 2.78%; 5-seed
  average 2.78%; both 2.745%. Production now = 3-seed average with extra features.
- Whole-forecast backtest (`src/backtest_forecast.py`, origins 1998-2016, horizons 3-20y,
  block-level medians, lease curve fitted only on the 6 years before each origin):
  * extrapolating recent growth is terrible (mean abs log error 0.76); constant 3-4%/yr
    is much better (~0.22); damping against recent growth, g = ln1.04 - 0.25 x (trailing
    5y - ln1.04), is best (0.194-0.197). Growth and lease decay are not separately
    identified (all lease weights score within 0.02), so weight 0.5 was fixed.
  * raw error bands undercovered on left-out origins (61-78% for an 80% band); widened
    until coverage was ~80% (and ~50% for the likely range). 5-10 origins per horizon,
    15y/20y only 7/5 origins: treat long horizons as least reliable.
  * bands smoothed (never narrower than shorter horizons; one pooled -5% bias shift).
  * no projection shown where lease would fall under 50 years (not covered by backtests).
- Honest range widths (80%): about x0.7 to x1.5 at 5 years, wider beyond. Survivorship:
  blocks demolished since are missing from old origins, so history is slightly rosy.
- Recalibrate (`python src/backtest_forecast.py`, ~7 min) about once a year, not monthly.

## Macro inputs (2026-11): tested, not adopted
- `src/macro.py` downloads World Bank (GDP growth, inflation, lending and real rates,
  population, real GDP per capita) and FRED/BIS Singapore property prices.
  `src/backtest_macro.py` adds each as a nudge to the assumed growth rate, fits the weight
  on all start years but one, and scores on the one left out; a shuffle test checks chance.
- Result (10 start years, baseline mean abs log error 0.190): valuation gaps (real price,
  price-to-income, BIS property) made it slightly WORSE (+2 to +3%); population growth
  +0.4%; real rate -2.4% (shuffle p 0.11); GDP growth -2.5% (p 0.07); 3-year change in the
  lending rate -4.0% (p 0.01). The last one is driven by one or two start years (most
  values are ~0 because the rate is administered), has the economically wrong sign
  (rising rates -> higher growth), is one of 7 tests, and its data ends in 2021.
  Not adopted. Ranges are not materially narrower with any macro input.
- Why: ~10 independent start years cannot support extra predictors. The wide ranges are
  real market-wide uncertainty, not a modelling gap. Realistic levers: show 3-5 year
  outlook more prominently, offer user-set growth scenarios, or accept the ranges.

## App layout and growth slider (2026-11)
- Outlook leads with 3 and 5 years (central, likely 50% range, wide 80% range); 10/15/20 years
  sit behind "Longer horizons" with a note that they rest on fewer past start years.
- Optional "Use my own market growth assumption" slider (0-8% a year, starts at the model's
  effective rate). Custom mode = scenario: no calibrated bias shift, ranges keep the back-test
  spread around the user's number, and the caption says so.
- Consistency check: the default includes a small calibrated level correction (~-5%), which
  first made a 2% user setting give a HIGHER 3-year value than the 3.5% default. Fixed by
  starting the slider at the model's effective growth (growth + shift/5 = 2.4%), so a user
  value equal to it reproduces the default 5-year central exactly; higher growth now always
  gives a higher value at every horizon.
- Streamlit reads a bare $...$ pair as maths: dollar signs in markdown text are escaped.

## Flat-specific ranges, asking-price check, reliability badge (2026-11)
- `src/oos_residuals.py`: walks forward in 4-month chunks from 2021-01, training only on earlier
  sales, so every error is genuinely out of sample (156k rows, median error ~2.6-3.1% per chunk).
- `src/range_model.py`: LightGBM quantile models (5/10/25/50/75/90/95%) of log(actual/predicted),
  plus data-richness inputs (sales in the block in 24 months, gap to recent comparables).
  Validation (train < 2025-07, test after): 80% band covers 80.2%, 50% band 51.5% after a x1.16
  widening. By segment vs the old single band: lease <=55y 64%->78%, cheapest fifth 66%->75%,
  3-room 73%->80%, executive 73%->80%. Still slightly under for the shortest leases / cheapest.
- A learned bias correction (q50) made point accuracy WORSE (2.64% -> 2.79%): the cheap/dear
  bias shifts over time. Not used; fair value = the point model's estimate.
- Reliability badge from the 80% band width (High <=0.12, Medium <=0.15 log units): High 62% of
  flats, median error 2.4%, 97% within 10%; Medium 24%, 3.3%, 92.5%; Low 14%, 4.0%, 87%.
- `src/advice.py` assess(): fair value, 50%/80% ranges, asking-price percentile and verdict,
  offer guide (open = 25th pct price, aim = fair value, walk away = 75th pct), notes (short
  lease, thin block history, unusual size, comps disagree). Barebones UI in `app.py`;
  `update.py` now also runs the walk-forward and range steps.
- Test case: 85 Whampoa Dr listing (3-room Improved, 818 sqft = 76 sqm, TOP 1973, asking
  $420,000, floor not stated). Block's recent 3-room sales are all 65 sqm ($341-350k).

## Listings, scoring, big verdicts (2026-11)
- No PropertyGuru crawler: their terms forbid automated collection. Instead `src/listing_parser.py`
  turns PASTED listing text into flat details (block/street matched to HDB data, abbreviations
  handled, lease start taken from HDB data). For automation at scale: licensed data (PropertyGuru,
  SRX, 99.co business feeds).
- `src/score_cases.py` + `cases/known_cases.csv`: score flats you know (hand-entered or a pasted
  listing file). Reports fair value, ranges, reliability, verdict; with an actual price also error,
  inside 50/80% ranges, whether the verdict was borne out; flags sales that may be in training.
  Gotcha fixed: numpy bools in an object column sum as logical OR (3 trues averaged to 33%).
- Deal verdicts (on the PRICE, not buy/don't buy), from the asking price's percentile among
  comparable sales: GREAT DEAL (<10%), GOOD DEAL (<30%), FAIR PRICE (<65%), PRICEY (<90%),
  OVERPRICED (90%+ or beyond the range); "LIKELY" prefix when reliability is Low. Big coloured
  banner in the app; paste-a-listing box fills in the flat's details.
- Test listing (85 Whampoa Dr, $420k, 76 sqm): LIKELY PRICEY at mid floor (+8.6%), LIKELY
  OVERPRICED if the HDB area is 65 sqm (+21%).

## Adjusted comparables (tested, not adopted) and price explanations (2026-11)
- `src/experiments_comps.py`: blend each flat's estimate with what its block's recent sales say
  (model-adjusted comparables = the comps' out-of-sample errors, shrunk by count). Tuned on
  2023-24, tested on 2025-26 (46,889 sales): model only 2.624% median / 96.2% within 10%;
  + block comps 2.616% / 95.9%; + same-size comps 2.639% / 95.8%. No real gain: the model
  already captures block effects. Not adopted. (Walk-forward files now keep block and street.)
- App: "In short" summary under the verdict (verdict, fair value, two biggest price drivers,
  confidence) and a "Why this price" section (driver effects vs the average HDB flat today,
  what-ifs). Size what-ifs are limited to sizes recorded for that flat type in the town.
  Known quirk: the model's size effect can be lopsided near rare sizes (e.g. 3-room 76 -> 81 sqm
  +10.5%, 76 -> 71 sqm -3.0%); it is what the model learned, shown as a pattern, not a guarantee.
