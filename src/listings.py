"""What a pasted listing says beyond the basics, and a store of listings (without contact details) to learn from.

signals(text)     the things public sale records can't see but listings often state: renovation (and the year),
                  original condition, an unblocked view, a corner unit, the facing, how keen the seller sounds.
                  Read with plain word patterns: no AI call, nothing guessed. Kampong shows them beside the verdict.
scrub(text)       the listing without phone numbers, emails, CEA registration numbers or agent names.
save(...)         keeps the scrubbed listing with its asking price and Kampong's estimate at the time. Over months
                  this becomes data no one else has: how far asking prices sit above what flats sell for, and what
                  renovation or a view is worth. src/listing_outcomes.py matches them to later sales.

People never have to give anything extra: it's what they paste anyway. The privacy page must say listings are kept.
"""
import json
import re
import time

import accounts as acc

YEAR = r"(20[0-3]\d|19[89]\d)"
PATTERNS = {
    "renovated": r"\b(?:newly |recently |fully |nicely |tastefully |extensively |completely )?(?:renovated|renovation done|reno(?:vated)?\b(?! needed)|refurbished|upgraded)",
    "original": r"\b(?:original (?:condition|state|unit)|unrenovated|un-renovated|needs? (?:some )?(?:renovation|reno|work|tlc)|as[- ]is)\b",
    "well_kept": r"\b(?:well[- ]kept|well[- ]maintained|move[- ]in (?:ready|condition)|ready to move in|good condition|tip[- ]top)\b",
    "unblocked": r"\b(?:unblocked|unobstructed|panoramic|open|full|sea|park|city|greenery|pool) view\b|\bunblock(?:ed)?\b",
    "corner": r"\bcorner (?:unit|flat|apartment)\b|\bcorner\b(?= ?unit)",
    "facing": r"\b(?:north[- ]?south|n[ /-]?s) facing\b|\bno (?:west|afternoon) sun\b|\bwindy\b|\bbreezy\b",
    "high_floor": r"\b(?:high|top|penthouse) floor\b|\bhigh[- ]rise\b",
    "keen_seller": r"\b(?:motivated|urgent(?:ly)?|must sell|keen) (?:seller|sale|to sell)\b|\bpriced to sell\b|\bbelow (?:valuation|market)\b|\bmust go\b",
    "negotiable": r"\b(?:nego(?:tiable)?|neg|ono|or nearest offer|open to offers?)\b",
}
LABELS = {"renovated": "renovated", "original": "original condition", "well_kept": "well kept", "unblocked": "an open or unblocked view",
          "corner": "a corner unit", "facing": "a good facing (no west sun or north-south)", "high_floor": "a high floor",
          "keen_seller": "a keen seller", "negotiable": "price negotiable"}


def signals(text):
    """{"found": [keys], "labels": [plain words], "renovated_year": int or None} from a listing's text."""
    t = " ".join(str(text).lower().split())[:20000]
    found = [k for k, p in PATTERNS.items() if re.search(p, t)]
    if "original" in found and "renovated" in found and re.search(r"\b(?:needs?|to be) (?:some )?(?:renovation|reno)", t):
        found.remove("renovated")  # "needs renovation" mentions renovation without being renovated
    year = None
    if "renovated" in found:
        m = re.search(r"(?:renovated|reno(?:vation)?|refurbished|upgraded)\D{0,12}" + YEAR, t) or re.search(YEAR + r"\D{0,6}(?:renovation|reno)\b", t)
        year = int(m.group(1)) if m else None
    return {"found": found, "labels": [LABELS[k] + (f" ({year})" if k == "renovated" and year else "") for k in found], "renovated_year": year}


PHONE = re.compile(r"(?:\+?65[\s-]?)?\b[689]\d{3}[\s-]?\d{4}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CEA = re.compile(r"\b[RL]\d{6,7}[A-Z]\b", re.I)
AGENT = re.compile(r"\b(?:agent|call|contact|whatsapp|wa|sms|text|pm|dm|msg|ask for)\b[:\s-]*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", re.I)
NRIC = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I)


def scrub(text):
    t = str(text)[:20000]
    for rx, sub in ((EMAIL, "[email]"), (NRIC, "[id]"), (CEA, "[cea no.]"), (PHONE, "[phone]"), (AGENT, "[contact]")):
        t = rx.sub(sub, t)
    return t


def save(kind, spec, asking, estimate, text, source="paste"):
    """Keep one pasted listing. estimate: Kampong's fair value and likely range at the time ({"fair", "q25", "q75"})."""
    sig = signals(text)
    with acc.db() as c:
        c.execute("create table if not exists listings (id integer primary key, at real not null, kind text not null, spec text not null, "
                  "asking real, fair real, q25 real, q75 real, signals text not null, text text not null, source text not null)")
        c.execute("insert into listings (at, kind, spec, asking, fair, q25, q75, signals, text, source) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (time.time(), kind, json.dumps(spec, sort_keys=True), asking, estimate.get("fair"), estimate.get("q25"), estimate.get("q75"),
                   json.dumps(sig["found"] + ([f"renovated_{sig['renovated_year']}"] if sig["renovated_year"] else [])), scrub(text), source))
    return sig
