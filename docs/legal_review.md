# Legal brief for Oikos (research, not legal advice)

Prepared November 2026 from public sources, to brief a Singapore lawyer. I am not a lawyer. Each finding is
marked **Confirmed** (official source read), **Reported** (secondary source or search summary), or
**Unverified** (could not read the primary text). The two statutes on sso.agc.gov.sg refused automated
access, so their definitions below are unverified and must be read by the lawyer.

## 1. Data licence
| Question | Finding | Status |
|---|---|---|
| Can we use data.gov.sg data commercially? | Yes. The Singapore Open Data Licence v1.0 is worldwide, perpetual, royalty-free; dataset pages say "free for personal or commercial use". | Reported (agency copies of the licence) |
| What must we do? | Show a conspicuous notice naming the source and linking to the licence. Do not imply official status or government endorsement. Data is "as is". | Reported |
| OneMap (block coordinates) | The OneMap website terms grant a licence "for any usage" and no attribution rule was found, but they are revocable, a charge may be introduced, and data must be destroyed on termination. API use is under a separate **Developer Agreement that I have not read**. | Confirmed for website terms; API terms unverified |

**Done in the product:** an attribution and no-affiliation notice is now in the page footer.
**Ask the lawyer:** read the OneMap Developer Agreement (stored geocodes, commercial use, attribution).

## 2. Is Oikos doing regulated work?
| Regime | Finding | Risk for Oikos | Status |
|---|---|---|---|
| **Appraisers Act 1906** (IRAS) | Still in force. IRAS: "Anyone who appraises lands and buildings in Singapore must apply for an Appraiser's Licence." The exact definition of "appraiser" (I believe it turns on doing it for a fee or reward) was not read. | **Highest.** We estimate what a specific flat is worth. Free vs paid may matter. | Licence requirement confirmed; definition unverified |
| Precedent: SRX X-Value | The valuers' institute (SISV) said in 2016 it "does not recognise computer-generated valuations". After mediation both sides agreed tools like X-Value help "pricing" and that "the opinion of a licensed appraiser ... remains an important part" where a formal valuation is required. X-Value is described as giving an "indicative value". | Shows the accepted framing: a **pricing tool / indicative estimate, not a valuation**. Competitors operate this way. | Reported (news article on the joint statement) |
| **Estate Agents Act 2010** (CEA) | Agencies need a licence and agents registration for "estate agency work". CEA has prosecuted people who marketed properties for clients. Whether information or advice alone counts was not found; I believe the definition centres on acting for a client to introduce parties or negotiate. | Low to medium today (we introduce no one, hold no listings, take no commission). **Rises** with an agent product, lead referral, or a listing feed. | Licensing confirmed; definition unverified |
| **Financial Advisers Act** (MAS) | A joint MAS-CEA reply (Dec 2025) says advice on property transactions "falls outside the scope" of the FAA, which covers investment products. | Low. Keep away from loans, CPF planning and "investment return" language, which move toward regulated ground. | Confirmed (MAS letter via search summary) |
| **Consumer Protection (Fair Trading) Act** (CCS) | Unfair to say or omit anything by which a consumer "might reasonably be deceived or misled", or to make a false claim. Tested on the overall impression. Using small print to hide a material fact is itself unfair. Claims should be substantiated, with assumptions stated. Applies to consumer transactions, so exposure grows once we charge. | **Medium.** Big verdict words and accuracy claims are the exposure; small disclaimers will not cure a misleading headline. | Reported (law-firm summaries, CCS guide) |
| **PDPA** | Not researched. Today the app stores age and settings only in the visitor's browser and keeps no accounts. | Low now; real once we add accounts, saved flats or analytics. | Not researched |

## 3. What this means for the product
1. **Never call the number a valuation or appraisal.** Use "estimate" or "fair value estimate", say it is
   indicative, and say a licensed valuer is needed for loans and sales. The page already says "Not a
   valuation"; keep it visible in both modes, not only in fine print.
2. **Verdicts must be claims we can substantiate.** "PRICEY: dearer than about 83% of comparable sales" is
   backed by back-tests; keep that evidence (`reports/`, `PLAN.md`). The word "LIKELY" and the reliability
   badge are part of what keeps the overall impression fair, so do not remove them to look more confident.
   "OVERPRICED" about a specific live listing may also upset the agent who listed it: ask about defamation
   or malicious-falsehood risk once we show verdicts on real listings.
3. **The offer guide and pricing guide are the most advice-like features.** They tell a person what to do
   with money. Ask the lawyer whether they need rewording ("comparable sales suggest...") or a stronger notice.
4. **"Will it last you?"** The age-95 rule is real: since May 2019, CPF use and HDB loans are pro-rated if
   the lease does not cover the youngest buyer to 95 (official 2019 releases; a secondary source adds a
   20-year minimum lease). Not checked for changes since, so the page only says rules "can depend on" it.
5. **Do not imply government backing.** We use HDB data but are not HDB. Stated in the footer.
6. **Before charging money:** terms of use, a limitation of liability, and a refund position.

## 4. Questions for the lawyer (in priority order)
1. Does giving a free, automated price estimate for a specific flat require an Appraiser's Licence? Does
   charging for it change the answer? Is the "indicative estimate, not a valuation" framing sufficient?
2. Do the offer guide, pricing guide, verdict words or the lease panel amount to estate agency work or any
   other regulated advice? What changes if we add agent referrals or a listing feed?
3. What must the disclaimers say, and where must they appear, to hold up under the CPFTA?
4. Is there defamation or similar exposure in labelling a third party's live listing "OVERPRICED"?
5. OneMap Developer Agreement: may we store and commercially use geocoded block locations?
6. What do we need under the PDPA before adding accounts, saved flats and analytics?
7. What terms of use and liability limits should the site carry?

## Sources
- Open Data Licence: https://data.gov.sg/open-data-licence ; agency copy with attribution terms: https://sg-mdh.mpa.gov.sg/node/162
- OneMap terms of use: https://www.onemap.gov.sg/legal/termsofuse.html
- IRAS appraiser's licence: https://iras.gov.sg/taxes/property-tax/property-professionals/appraisers-valuer/about-appraiser's-licence
- Appraisers Act 1906 (not read, access refused): https://sso.agc.gov.sg/Act/AA1906
- Estate Agents Act 2010 (not read, access refused): https://sso.agc.gov.sg/Act/EAA2010
- StreetSine and SISV on automated valuation models: https://www.lushhomemedia.com/?p=71377
- MAS-CEA joint response on property agents and the FAA: https://www.mas.gov.sg/news/letters-to-editor/2025/joint-response-to-regulating-property-agents-under-faa
- CPFTA commentary: https://www.rajahtannasia.com/viewpoints/consumer-protection-in-singapore-five-key-observations-for-businesses-in-a-heightened-enforcement-climate/ ; CCS guide on claims: https://www.rahmatlim.com/sg/publication/articles/31353/ccs-issues-guide-to-assist-businesses-in-ensuring-quality-related-claims-are-accurate-and-not-misleading
- CPF and HDB loan lease rule (2019): https://www.mom.gov.sg/newsroom/press-releases/2019/0509-more-flexibility-to-buy-a-home-for-life
