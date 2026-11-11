# Plan: showing the downsides of a location (noise and other nuisances)

Written 2026-11-11. Not built. This is item 5 from the location shortlist: tell a buyer what a seller will not.

## What we would show

On the flat card, under "Around this flat", one line per nuisance that applies, and nothing when none does:

- **Expressway** within 150 m: "PIE is 80 m away. Units facing it will hear traffic."
- **Above-ground MRT or LRT track** within 100 m (not underground lines): "The East-West Line runs above ground 60 m away."
- **Major road** (arterial, 6 lanes or more) within 50 m.
- Later, if the data is good enough: flight paths near Changi, Paya Lebar, Seletar and Tengah air bases; industrial estates; columbaria and funeral parlours; future construction next door (a BTO site or a planned MRT work site).

The AI's advice would then say what to check: which way the unit faces, visit at rush hour and at night, look for double glazing.

## Why it matters and why it is hard

- It is the thing buyers find out too late, and no portal shows it. That makes it a real difference, not a nice-to-have.
- **It depends on the unit, not the block.** A block beside the PIE has quiet units on the far side. We know the block and the storey band, not the facing. So this must be worded as "check this", never as a verdict or a price cut.
- **Height matters.** Low floors behind a noise barrier can be quieter than the 10th floor. We should not pretend to model that.

## Data

| What | Source | Notes |
|---|---|---|
| Expressways and major roads | OpenStreetMap ways tagged `highway=motorway` / `trunk` / `primary` (same Overpass download as `download_places.py`) | Free, ODbL attribution already in the app. Geometry is good in Singapore. |
| Rail track, above ground or not | OpenStreetMap `railway=subway` / `light_rail` with `tunnel` and `bridge` tags | The `tunnel` tag is what separates a noisy viaduct from a silent tunnel. Spot-check 20 known stretches before trusting it. |
| Official alternative | LTA DataMall and data.gov.sg road and rail line layers | Check whether they mark elevated sections; licence is the Singapore Open Data Licence. |
| Measured noise | NEA publishes limits, not a noise map. There is no public noise map of Singapore **(verify)**. | So we report distance, not decibels. |
| Flight paths | No clean public layer. Could approximate from runway headings. | Leave for later. |

## Method

1. Download road and rail lines, keep expressways, arterials and rail without a `tunnel` tag.
2. For every block (about 10,000) compute the shortest distance to each kind of line, and the name of the nearest one. This is a point-to-line distance, done once and stored with the block table.
3. Show a line on the card when the distance is under the threshold. Thresholds above are starting points; walk a few sites to tune them.
4. Give the same facts to the AI with the instruction to raise them under "Check first" and never to quote a price effect.

## Should it go into the price model?

Test it, but expect little. Distance to an expressway is already half-captured by the block's own sales. Add `dist_expressway` and `dist_elevated_rail` as features in `experiments_accuracy.py`; keep them only if the holdout error improves. Whatever the result, do not show a "noise discount" figure: the model cannot tell the noisy side of a block from the quiet side.

## Effort and risks

- About two evenings: one for data and distances, one for the card, the AI wording and checks.
- **Risk: false alarms.** A wrongly tagged tunnel would tell people a quiet flat is noisy, which a seller could reasonably object to. Mitigate with the spot-check, cautious wording ("is 60 m away" is a fact; "is noisy" is not), and a "tell us if this is wrong" link.
- **Risk: sellers.** In seller mode, show the same facts as "buyers will ask about this" with how to answer it, not as a warning.
