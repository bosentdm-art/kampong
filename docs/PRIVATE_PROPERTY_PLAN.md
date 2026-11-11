# Plan: adding condo and landed prices to Oikos

Written 2026-11-11. Nothing here is built yet. Items marked **(verify)** are from memory and must be checked
against the source before work starts.

## The short version

Condos are a realistic next step: about 4 to 6 weeks of evenings, most of it data work. Landed homes are a
different problem (few sales, every house different) and should ship later, with wider ranges and softer verdicts.
The chat, the verdict card, accounts, sharing and alerts can all be reused; the model and the data pipeline cannot.

## 1. Data

| What | Source | Notes |
|---|---|---|
| Private home sales (condo, apartment, EC, landed) | URA Data Service API, "private residential property transactions" | Free, needs a URA account and access key. Covers roughly the last 5 years **(verify)**. Gives project, street, area in sqm, floor range, tenure, type of sale, price, contract date, district, market segment. |
| Older history | URA REALIS | Paid subscription. Only needed if 5 years is not enough for the long-run outlook. |
| Rents (optional, later) | URA rental contracts API | Lets us show yield. Not needed for a price verdict. |
| Location | OneMap, as now | Geocode by project name and street. |

Things that are different from HDB data, and bite:

- **No unit numbers.** URA gives a floor range ("06 to 10") and an area, not the unit. Same as HDB, so the
  approach carries over.
- **Area is not comparable across years.** From June 2023 new projects stopped counting air-con ledges and
  voids in strata area **(verify the exact date and rule)**. A 100 sqm unit in a 2015 project is smaller inside
  than a 100 sqm unit in a 2025 project. The model needs the project's launch year as a feature.
- **Tenure varies.** Freehold, 999-year and 99-year, with different start dates. HDB has one lease type.
- **New sales and resales are mixed.** Developer prices are set, not negotiated. Train on resales and
  sub-sales; keep new sales only as context.
- **Licence.** The URA API has its own terms of use. Read them before building anything public: check that
  commercial use, storing the data and showing derived estimates are all allowed **(verify)**.

## 2. Models

**Condos and apartments (including ECs after they privatise).** Same recipe as HDB: LightGBM on log price per
sqm, walk-forward testing, quantile models for the ranges, reliability badge. New features: project (as a
category, with a fallback for small projects), tenure and years left, launch year, number of units, floor, area,
district, distance to MRT and schools. Expect a median error of about 4 to 6% against 2.7% for HDB, because
facing, layout and view matter more and are not in the data. The acceptance test is the same as before: beat a
"recent sales in the same project" baseline on held-out months.

**Landed.** A few thousand sales a year across terrace, semi-detached and bungalows, and the land does most of
the pricing. Use price per sqm of land, with built-up area, tenure, type and street as features. Expect 10 to 15%
median error. Rules for shipping it: always show the wide range first, never show a one-word verdict when
reliability is Low, and say plainly that a valuer is needed.

**Outlook.** The HDB forecast uses a lease-decay curve and a market growth rule fitted to HDB history. Neither
carries over. Launch without a long-run outlook for private property, then add one after a backtest.

## 3. Product changes

- **Reading the flat.** People name a condo by project ("Parc Esta, 2-bed, 70 sqm, high floor, asking 1.6m"),
  not by block and street. The parser needs a project-name index, and the AI reader's instructions need the
  condo fields. Landed homes are named by street and house number.
- **Verdict scales.** The same five colours work. The thresholds should be re-checked on private data.
- **Advice.** The adviser prompt needs private-property facts: tenure, ABSD and loan rules are out of scope
  (same stance as CPF today), en-bloc potential must not be guessed at.
- **"Will it last you?"** Only makes sense for 99-year leases. Hide it for freehold.
- **Alerts and Deals.** Work as they are once the estimates exist: "new sale in this project", "sold well under
  estimate in District 15".
- **Plans.** Private property is a natural paid feature: agents and condo buyers have more at stake per deal.

## 4. Order of work

1. Get the URA access key, read the terms of use, pull the data, and write the cleaning script (1 week).
2. Baseline and first condo model, with a held-out test and an error number (1 week).
3. Ranges, reliability badge and the facts the page needs, behind the same API shape as HDB (1 week).
4. Project-name reading in the chat, card and advice wording, alerts (1 to 2 weeks).
5. Landed model and its cautious presentation (2 weeks, after condos are live).

## 5. Risks

- **Licence** is the one that can stop the project. Check it first.
- **Accuracy.** If the condo model cannot beat "last three sales in the project", ship that simple comparison
  with the AI explaining it, and say so.
- **Legal exposure** grows with price. The open question in `legal_review.md` about giving price advice matters
  more at $2m than at $500k.
- **Cost.** Data and training stay free on the laptop; the 5070 PC is not needed for this size of data.

## Update 2026-11-11: first data pull and accuracy check

`python src/download_ura.py` pulled 123,485 sales in 2,959 projects, Nov 2021 to Nov 2026 (about 71,000 condo,
apartment and EC resales; about 9,400 landed resales). `python src/experiments_ura.py` held out the last six months:

| | Typical error | Within 10% |
|---|---|---|
| Condos, apartments, ECs: recent sales in the same project (available for 93% of sales) | 4.1% | 83% |
| Condos, apartments, ECs: quick model, no tuning | 3.2% | 90% |
| Same, for projects with no sales in the last 12 months | 5.7% | 71% |
| Landed: recent sales on the same street (available for 68% of sales) | 13.8% | 38% |

So a condo model is worth building: an untuned one already beats the same-project comparison and is close to the
HDB model's 2.7%. Landed is as hard as expected. The URA terms of use are still unread.

## Update 2026-11-11: the condo model is built

`python src/train_condo.py` (recipe chosen in `experiments_condo.py`) trains on 113,322 sales to Oct 2026 and saves
the models; `src/condo.py` prices one unit (`python src/condo.py "PARC ESTA" 70 12 1600000`).

| Held-out six months of resales | Typical error | Within 10% |
|---|---|---|
| All condos, apartments and ECs | 3.0% | 91% |
| Executive condominiums | 2.2% | 96% |
| Condominiums | 3.0% | 92% |
| Apartments | 3.3% | 88% |
| Projects with fewer than 3 sales in 24 months (5% of sales) | 5.7% | 71% |
| For comparison: recent sales in the same project | 4.2% | 83% |

Ranges: the 80% range held 80.6% of later sales and the 50% range 52.5%. Reliability: High for 81% of units
(typical error 2.7%), Medium 11% (4.7%), Low 8% (6.9%).

Not done yet: the app does not use the model (steps 4 onward above), it is not part of the weekly refresh, and
there is no long-run outlook. URA gives a 5-floor band and no unit number, facing or condition, so about 3% is
close to what this data allows.
