# UI/UX revamp plan

## Decisions so far
- **Users:** home buyers and sellers first. Property agents later, as a separate business (B2B) product.
- **Platform:** desktop web first, mobile app later.
- **Tech:** decided after the research (see "Tech decision" below).
- **Research method:** competitor review.

## Phase 1: Competitor review (research)
**Goal:** learn how existing tools present a home's value, so ours is clearer, more decisive and more
trustworthy, and find the gaps we can own.

**Who to review**
| Group | Products |
|---|---|
| Singapore, direct | SRX X-Value, 99.co property value tool, PropertyGuru (home value / price insights), EdgeProp, Ohmyhome (Homer AI), UrbanZoom |
| Singapore, official | HDB resale flat prices e-service, data.gov.sg resale data |
| Overseas benchmarks | Zillow (Zestimate), Redfin Estimate, Rightmove or Zoopla (UK), Domain (Australia) |

**What to record for each (same rubric for all)**
1. Inputs asked for, and how long it takes to get an answer.
2. The headline: a single number, a range, a verdict, or a chart.
3. How confidence or accuracy is shown (if at all).
4. Explanations: drivers, comparable sales, trends, maps.
5. Buyer features: asking-price check, offer help, affordability.
6. Seller features: suggested listing price, demand signals, "how fast will it sell".
7. Trust signals and fine print: where disclaimers sit and how heavy they are.
8. Business model: where it pushes you (agent leads, mortgages, listings).
9. Desktop layout, visual style and use of colour.
10. Anything it does badly that we can do better.

**Rules:** look only at public pages a normal visitor can see, a handful of pages per product, by hand.
No automated collection. Screenshots for our internal reference only.

**Deliverable:** `docs/research/competitor_review.md` with a comparison table, one short summary per
product, and a list of patterns to copy, avoid, and gaps to own.

## Phase 2: Product definition
- **Jobs to be done.** Buyer: "Is this listing worth the price, and what should I offer?" Seller: "What
  can I list at, and how do I price to sell?"
- **Seller flow (new).** Same models, different framing: a suggested listing range, a verdict on the
  seller's intended price ("PRICED TO SELL" / "AMBITIOUS" / "TOO HIGH"), and what the block has sold for.
- **Information architecture.** One entry screen (paste a listing, or pick a block), then one results
  page per job: verdict first, then fair value and confidence, then why, then outlook.
- **Content rules.** Big, plain verdict words; numbers rounded for reading ($387k, not $386,658);
  one short line of fine print per section.

## Phase 3: Design
- Low-fidelity wireframes of the buyer and seller results pages (desktop).
- Visual language: the verdict colour scale (bright green, green, yellow, orange, red), type scale,
  spacing, a simple brand name and logo placeholder.
- One clickable prototype of the buyer flow.

## Tech decision (after Phase 1)
Compare **Streamlit (polished)** against **a web app (React / Next.js front end + Python API)** on:
design control and polish, effort, hosting cost, accounts and saved searches, search-engine visibility,
a reusable API for the agent product later, and a path to a mobile app. Current lean: the web app plus
API, because the agent product and the mobile app both need an API anyway, but the research decides.

## Phase 4: Build and check
Build the buyer flow first, then the seller flow. Check every screen against the competitor findings
and the content rules, and re-test with the 85 Whampoa Drive listing and `cases/known_cases.csv`.

## Open questions for later
- Brand name and tone of voice.
- Accounts and saved flats: needed at launch or not.
- Legal review of verdict wording and the offer guide before public launch.

## Design direction (decided 2026-11-10)
- **Style:** technical "HUD" look from the user's reference: thin leader lines calling out a central
  object, boxed data panels with numbered headers, small monospace labels, one giant headline, a faint
  grid. Kept property-related: the central object is a wireframe of the HDB block (the flat's storeys
  highlighted) with the price drivers as callouts; the giant headline is the verdict in its colour.
- **Themes:** dark, light and blueprint, all in the prototype. Pick one as the default.
- **Rule:** decoration may be small and technical; the verdict, fair value, ranges and offer guide stay
  large and plain. Inspired by the reference, not a copy of it. Brand name: **Oikos**. Default theme: Light. Two modes: Simple (verdict, fair value, likely range, offer guide, top two drivers) and Pro (everything).
- **Prototype:** `docs/prototype/index.html` (static, desktop, real model output for 85 Whampoa Dr;
  the asking-price slider re-reads the flat's price quantiles). Preview config: `prototype`.
- **Tech consequence:** Streamlit cannot produce this layout. The real front end will be custom
  HTML/CSS/JS (or React) calling a small Python API that wraps `advice.assess`, `explain.drivers`
  and `forecast.forecast`.

## Prototype components (2026-11-10)
- Roles (buying / selling), modes (Simple / Pro), themes (Light default, Dark, Blueprint).
- New: fair-value gap chip under the verdict, range bar in the fair value panel, "Your options"
  (three rule-based options by role and verdict, filled with the flat's numbers), "Sold below our
  estimate nearby" (real completed sales, buyer view), and a Trend tab in Pro.
- **Bargains:** the panel uses completed sales, because we have no listing data. Flagging live
  listings below fair value needs a licensed listing feed; the scoring is already built.

## LLM explanations (planned)
- The models produce a facts object per flat (fair value, ranges, percentile, verdict, drivers,
  what-ifs, recent sales, reliability and its reasons, role). The LLM only rewrites those facts into
  a short explanation of the options and answers follow-up questions by calling the engine.
- Guardrails: no numbers that are not in the facts (checked automatically), no buy/sell instruction,
  no time-to-sell or future-price claims beyond the calibrated ranges, reliability always stated.
- Fallback: the rule-based "Your options" text, which is what the prototype shows today.
- Needs: a Python API around `advice.assess`, `explain.drivers`, `forecast.forecast`; an API key
  kept server-side; a cost check per explanation before launch.

## 35+ audience (2026-11)
- Ideas brainstormed: age-first entry, two verdicts (price and personal fit), a life timeline, a
  sell-and-buy view for upgraders and right-sizers, "near my parents", and a singles budget check.
- Built: "Will it last you?" in the buyer view of `web/index.html`. The buyer enters their age and a
  bar from today to age 95 shows how long the lease runs. Colours are our own rule of thumb (green if
  the lease outlasts 95, orange if it ends at 85-94, red if earlier). Pro adds lease left at 55/65/75/85.
- To verify before launch: the HDB loan and CPF rules tied to the lease covering the youngest buyer to
  age 95, singles eligibility and grants, and whether a personal "fit" verdict counts as advice.
