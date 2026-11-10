# Competitor review (Phase 1)

Desk research, November 2026: public pages, product claims, news and user complaints (forums, Q&A
boards, review sites). Vendor accuracy figures are self-reported unless stated. Complaints are
self-selected (people with bad experiences post more), so they show *what goes wrong*, not *how often*.

## Comparison
| | SRX X-Value | 99.co value tool | PropertyGuru | EdgeProp | Ohmyhome HomerAI | HDB valuation | Zillow / Redfin | Zoopla / Rightmove |
|---|---|---|---|---|---|---|---|---|
| **Inputs** | Type, address, size (suggested, "please verify"), extension area | Address or postal code; unit, size for tracking | Address | Type, address, unit details | Address | Formal request by buyer after Option to Purchase, fee, valuer visits | Address | Address |
| **Headline** | Single value | Value, rental value, appreciation chart | Value or range | "Edge Fair Value" | Indicative value report | Official valuation | Value **and a range** | Value with low / high |
| **Confidence shown** | Claims "98% accuracy"; publishes a performance page | No method or range explained | Unclear | Unclear | Claims 2-3% median error | n/a | Publish median error (on / off market) | Little |
| **Explanations** | Comparable sales, 10-year trend | Neighbouring transactions | Transaction history | Methodology endorsed by a licensed valuer | Comparisons vs rivals | Valuer's report | Comparable sales, history charts | Some |
| **Buyer help** | Comparables | - | Mortgage, affordability | Filter listings priced below fair value | - | - | Affordability | - |
| **Seller help** | Sale proceeds, tracking + alerts (3 homes) | Tracking (4 homes) | - | - | Pushes its own selling service | - | Seller tools, agent leads | "Refine estimate" |
| **Fine print** | "Indicative, not the final valuation" | "Does not consider interior condition" | "Indicative" | "Indicative" | "Indicative" | Binding for loans | "Estimate, not an appraisal" | "Estimate, not a valuation" |
| **Business model** | Agent tools, reports | Listings (powered by SRX X-Value) | Listings, agents | Listings, news | Its own agency | Government | Listings, agent leads | Listings, agent leads |

## What users complain about
1. **Big misses on individual homes.** Singapore buyers report gaps of $8k to over $50k between X-Value
   and the HDB valuer; Zestimates are described as sometimes 20-30% off; Zoopla users report a house
   overvalued by 60k next to an identical one undervalued by about 100k. A small median error hides large
   individual misses, and people judge a tool by *their* home.
2. **Tools disagree with each other.** One comparison of the same 4-room Toa Payoh flat ranged from
   about $732k (SRX) to $800k (PropertyGuru). Users don't know which to believe.
3. **Estimates jump without explanation.** An X-Value reportedly changed three times in one week; Zoopla
   and Zestimate users report sudden drops "for no reason", including right after a sale.
4. **Condition and renovation are ignored.** 99.co says outright that interior condition is not
   considered; sellers of renovated flats feel short-changed.
5. **Thin markets are unreliable.** Commenters say estimates are weak where there are few transactions.
6. **Wrong data can't be fixed.** Zoopla users couldn't correct a wrong record; Zillow homeowners
   called Zestimates hard to get changed (a 2017 lawsuit failed because it is labelled an estimate).
7. **No explanation of the range or method.** 99.co does not say how its range is computed; SRX
   advertises "98% accuracy" without saying what that means.
8. **Distrust of listings themselves.** Fake and "bait" listings priced artificially low are a
   recurring complaint about property portals (reported by the Straits Times in 2026).
9. **Tools push you somewhere.** Most are funded by listings or agent leads, and users notice.

## Patterns to copy
- Address or postal code as the first input; a **suggested size with "please verify"** (SRX).
- Comparable recent sales and a 10-year price trend next to the estimate (SRX, Zillow).
- A **range** rather than a single number (Zillow, Redfin, Zoopla).
- **Publishing accuracy openly** (SRX's performance page, Zillow and Redfin's median error), but with
  plain meaning, not a vague percentage.
- A "refine your estimate" step (Zoopla), and tracking with alerts (SRX, 99.co) once we have accounts.
- Sale-proceeds estimate for sellers (SRX).

## Patterns to avoid
- Vague accuracy claims ("98% accurate") with no definition.
- Silent changes to a home's estimate. If it changes, say why (new sales, monthly update).
- One confidence level for every home, which is what drives the "it was miles off for MY flat" anger.
- Pushing users to agents or the company's own selling service at the moment they want an answer.
- Heavy fine print that still doesn't explain anything.

## Gaps we can own
1. **A straight verdict on the price.** No Singapore tool says "GREAT DEAL" or "OVERPRICED" about a
   listing; the closest is EdgeProp's "below fair value" filter. Ours does, with colour.
2. **Honest, flat-specific confidence.** We already give each flat its own range and a High / Medium /
   Low reliability badge with reasons (short lease, few sales, unusual size). This answers complaints
   1 and 5 directly.
3. **Explain every number.** Price drivers, what-ifs, and the input checks against HDB's records
   (answers complaints 6 and 7).
4. **Stable, explained updates.** Monthly refresh with a "what changed and why" note (complaint 3).
5. **Condition as a user-set adjustment.** Let users say "recently renovated" and show its effect
   clearly as *their* adjustment, not the model's (complaint 4). Needs care: we cannot validate it.
6. **A published, plain-language track record:** "Over the last six months, half of sales landed within
   2.7% of our estimate, and 8 in 10 inside our range."
7. **Neutral by design.** No agent push in the core answer. Agent and B2B features come later and
   separately.
8. **Sellers get a verdict too:** "PRICED TO SELL / AMBITIOUS / TOO HIGH" on their intended price.

## Limits of this review
Desk research only: I looked at product pages and public discussion, not every product's live
interface, and I did not create accounts. Accuracy numbers come from the vendors. UrbanZoom's current
status could not be confirmed. Next step: a short hands-on pass of SRX, 99.co and EdgeProp with the same
flat (85 Whampoa Drive) to compare screens directly.

## Sources
- SRX X-Value product page: https://www.srx.com.sg/xvalue-pricing and HDB page: https://www.srx.com.sg/xvalue-pricing/hdb-valuation
- 99.co on X-Value accuracy: https://www.99.co/singapore/insider/x-value-agents-free-tool-track-clients-property-value/
- PropertyGuru Q&A, X-Value vs HDB valuer gap: https://www.propertyguru.com.sg/property-investment-questions/hi-everyone-i-have-a-question-in-regards-to-hdb-valuation-vs-x-value-the-hdb-flat-that-my-husband-and-i-want-to-buy-was-valuated-50k-lower-by-hdb-valuator-than-the-suggested-x-value-the-owner-was-wil-236225
- HardwareZone HDB valuation thread: https://forums.hardwarezone.com.sg/threads/hdb-valuation.6325920/post-131782065
- Ohmyhome tool comparison: https://ohmyhome.com/en-sg/blog/the-battle-of-property-valuation-tools-whos-the-most-accurate/
- 99.co property value tool help: https://intercom.help/99faq/en/articles/6654994-property-value-tool-on-99-co
- EdgeProp about page: https://www.edgeprop.sg/content/about-us
- Ohmyhome HomerAI: https://ohmyhome.com/en-sg/homer/property-value/
- Straits Times via Stomp, bait listings: https://www.stomp.sg/trending-now/too-good-be-true-bait-style-listings-being-used-some-property-agents-lure-buyers
- Zestimate criticism and lawsuit: https://www.dailyherald.com/20170512/lifestyle/zillow-under-fire-for-zestimate-system/ and https://www.geekwire.com/2019/appeals-court-sides-zillow-lawsuit-zestimate-accuracy/
- Zillow vs Redfin estimates: https://www.realestatewitch.com/redfin-vs-zillow-estimators/
- Zoopla reviews: https://ie.trustpilot.com/review/www.zoopla.co.uk?page=6 and Mumsnet: https://www.mumsnet.com/talk/property/4345372-Zoopla-valuation-wrong?page=1
