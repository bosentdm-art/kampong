# From web page to application (research for step 3)

Mostly design analysis of our own product against the competitor review, with a little outside reading.
It is a proposal to react to, not established fact.

## Why Oikos feels like a web page today
| Symptom | What a person experiences |
|---|---|
| One long scroll | Everything is laid out to be read top to bottom, like an article |
| A form at the top | It looks like a search page: fill in, press a button, get a page back |
| Nothing is kept | Close the tab and the flat, the price and the answer are gone |
| One flat at a time | No way to hold two or three flats side by side, which is how people actually decide |
| Nothing to do afterwards | The page ends. There is no next step, nothing to take away, nothing to come back for |
| No sense of "me" | It asks about the flat, never about the person, so it cannot feel like a service for them |
| Every answer is shown at once | An app shows what you asked for and lets you go deeper; a page shows everything |

Competitors mostly have the same problem: they are calculators attached to listing sites. That is the opening.

## What makes something feel like an application
1. **A workspace that stays put.** A fixed frame (navigation on the left, bar on top) with the content
   changing inside it, instead of a page that scrolls away.
2. **Things you own.** Saved flats with a status (watching, viewed, offered), your notes, and the prices
   you tested. The product remembers your work.
3. **A flow with a beginning and an end.** Who are you, which flat, what price, verdict, what next.
4. **Actions that produce something.** A one-page "offer brief" or "pricing brief" to save, print or send.
5. **Comparison.** Two or three flats side by side on the same measures.
6. **It works while you are away.** Saved flats are re-checked each month and the app says what changed
   and why ("fair value up 1.2%: three new sales in the block").
7. **It answers back.** A docked "Ask Oikos" panel that explains the options in plain words (the LLM).
8. **It responds instantly.** No full-page waits; panels update in place (the price slider already does).

## What makes it a specialised service, not a generic tool
- It is **about a decision**, not about data: "should I offer, and how much" or "what do I list at".
- It **knows the person**: role, age, budget, and what stage they are at.
- It **hands over a deliverable**: the brief, with the evidence behind each number.
- It **stays with them** until the decision is made: shortlist, monitoring, alerts.

## Proposed structure
```
+---------+------------------------------------------------------+
| OIKOS   |  [ search any block or paste a listing ]   Simple/Pro |
|         +------------------------------------------------------+
| Check   |                                                      |
| My flats|   the current view, fitted to the screen:            |
| Compare |   verdict + key panels on top, tabs for the rest     |
| Deals   |                                                      |
| Market  |                                              [Ask    |
|         |                                               Oikos] |
| Buying/ |                                                      |
| Selling |                                                      |
+---------+------------------------------------------------------+
```
- **Check:** the guided flow. Start screen asks role (and age, for buyers), then the flat, then the price.
  The result fits one screen: verdict, fair value, guide and the block on top; drivers, sales, lease,
  outlook and trend behind tabs. Buttons: Save, Compare, Get the brief.
- **My flats:** the shortlist as cards with verdict colour, status, notes and "changed since you saved it".
- **Compare:** up to three saved flats in columns: price, verdict, fair value, lease-to-age, drivers.
- **Deals:** the below-estimate sales, by town (live listings once we have a licensed feed).
- **Market:** town and flat-type trends.
- **Ask Oikos:** docked panel, later the LLM; until then the rule-based options.

## Build order
| Step | What | Needs |
|---|---|---|
| 1 | App shell with navigation; results fitted to one screen with tabs; guided start | Front end only |
| 2 | Save flats, statuses and notes; My flats | Browser storage (no accounts yet) |
| 3 | Compare view | Step 2 |
| 4 | The brief (print-ready one-pager) | Front end only |
| 5 | Speed (under one second per check), loading states, errors | API work |
| 6 | Accounts, so saved flats follow the person across devices | Hosting, PDPA review |
| 7 | Monthly re-check of saved flats with "what changed" | Step 6 |
| 8 | Ask Oikos with the LLM | API key, guardrails |

Steps 1 to 5 need no accounts and no new data, and would change how the product feels the most.

## Open questions
- Does the shortlist need accounts from day one, or is "saved on this device" acceptable to start?
- Is the brief free, or is it the first thing worth charging for? (Charging raises the legal questions
  in `docs/legal_review.md`.)
- Mobile: the shell collapses to a bottom tab bar. Design it now or after the desktop version?
