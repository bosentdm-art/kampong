"""Oikos API: the trained models behind a small web service, plus the front end in web/.

Run:  python -m uvicorn api:app --port 8700     then open http://localhost:8700
Needs the data and models built first (python src/update.py).

  GET  /api/options   lists for the flat picker (town > street > block > flat type) and sensible defaults
  POST /api/parse     pasted listing text -> flat details
  POST /api/understand  a chat message -> the flat it describes, a change to the flat on screen, or an AI reply
  POST /api/ask       the AI's advice on a flat, every figure checked against the facts
  POST /api/share     make a link to a flat's verdict and advice; GET /api/share/{id} reads it; /s/{id} is the page
  GET/POST /api/alerts  watch a block for new sales, or a town for sales well under our estimate
  POST /api/signup, /api/login, /api/logout, GET /api/me, GET/POST /api/flats   accounts (see src/accounts.py)
  POST /api/assess    a flat (and optionally a price) -> every fact the page shows
  GET  /api/deals     below-estimate sales in a town       GET /api/trend   yearly price per sqm for a town and flat type

The page works out verdicts for any price from the returned price quantiles, so dragging the price needs no
new request. Everything returned comes from the models and the sales data; nothing is invented here.
"""
import functools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import advice as adv  # noqa: E402
import explain as ex  # noqa: E402
import forecast as fc  # noqa: E402
import listing_parser as lp  # noqa: E402
import llm  # noqa: E402
import accounts as acc  # noqa: E402
import nearby as nb  # noqa: E402

app = FastAPI(title="Oikos")


# ---- picking up a data refresh without a restart. src/update.py finishes by writing models/range.json; when that
# file changes, every cached table and model in this process is dropped and reloaded on the next request.
STAMP = ROOT / "models/range.json"
_fresh = dict(mtime=STAMP.stat().st_mtime if STAMP.exists() else 0, checked=0.0)


def refresh_if_new(force_check=False):
    import time
    now = time.time()
    if not force_check and now - _fresh["checked"] < 60:
        return False
    _fresh["checked"] = now
    mtime = STAMP.stat().st_mtime if STAMP.exists() else 0
    if mtime == _fresh["mtime"]:
        return False
    _fresh["mtime"] = mtime
    for mod in (sys.modules[__name__], adv, ex, fc, lp, nb):
        for obj in vars(mod).values():
            if callable(getattr(obj, "cache_clear", None)):
                obj.cache_clear()
    return True


@app.middleware("http")
async def fresh_data(request, call_next):
    if request.url.path.startswith("/api/"):
        refresh_if_new()
    return await call_next(request)


class Flat(BaseModel):
    town: str
    street_name: str
    block: str
    flat_type: str
    flat_model: str
    floor_area_sqm: float
    storey_range: str
    lease_commence_date: int
    price: float | None = None


class Listing(BaseModel):
    text: str


def clean(x):
    """Make numpy / pandas values JSON-safe (NaN becomes null)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m")
    return x


@functools.lru_cache(maxsize=1)
def sales():
    cols = ["town", "street_name", "block", "flat_type", "flat_model", "floor_area_sqm", "storey_range",
            "lease_commence_date", "sale_date", "sale_year", "price_per_sqm"]
    d = pd.read_parquet(ROOT / "data/processed/resale_features.parquet", columns=cols)
    return d[d["sale_date"] >= "2016-01-01"].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def meta():
    m = json.loads((ROOT / "models/lgbm_prod.json").read_text())
    r = json.loads((ROOT / "models/range.json").read_text())
    total = len(pd.read_parquet(ROOT / "data/processed/resale_features.parquet", columns=["sale_year"]))
    return dict(trained_through=m["trained_through"], holdout_median_error=m["holdout_median_error"],
                coverage80=r["validation"]["coverage80"], total_sales=total)


@functools.lru_cache(maxsize=1)
def recent_oos():
    """Out-of-sample estimates for the last six months of sales (from the walk-forward files)."""
    d = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/processed/oos").glob("chunk_*.parquet"))], ignore_index=True)
    r = d[d["sale_date"] > d["sale_date"].max() - pd.DateOffset(months=6)].copy()
    r["gap"] = r["resale_price"] / r["pred"] - 1
    r["town"] = r["town"].astype(str)
    return r


def block_key(b):
    m = re.match(r"\d+", b)
    return (int(m.group()) if m else 0, b)


def pick(wanted, options):
    return wanted if wanted in options else options[0]


@app.get("/api/options")
def options(town: str | None = None, street: str | None = None, block: str | None = None, flat_type: str | None = None):
    """Lists for each level of the picker. A missing or invalid choice falls back to the first option."""
    d = sales()
    d = d[d["sale_date"] >= "2017-01-01"]
    towns = sorted(d["town"].unique())
    town = pick(town, towns)
    dt = d[d["town"] == town]
    streets = sorted(dt["street_name"].unique())
    street = pick(street, streets)
    ds = dt[dt["street_name"] == street]
    blocks = sorted(ds["block"].unique(), key=block_key)
    block = pick(block, blocks)
    db = ds[ds["block"] == block]
    flat_types = sorted(db["flat_type"].unique())
    flat_type = pick(flat_type, flat_types)
    dbt = db[db["flat_type"] == flat_type]
    models = dbt["flat_model"].value_counts().index.tolist()
    return clean(dict(
        towns=towns, streets=streets, blocks=blocks, flat_types=flat_types, flat_models=models,
        storey_ranges=sorted(d["storey_range"].unique()),
        selected=dict(town=town, street=street, block=block, flat_type=flat_type),
        defaults=dict(flat_model=models[0], floor_area_sqm=float(dbt["floor_area_sqm"].median()),
                      storey_range=dbt["storey_range"].mode().iloc[0],
                      lease_commence_date=int(db["lease_commence_date"].mode().iloc[0]))))


@app.post("/api/parse")
def parse(listing: Listing):
    return clean(lp.parse(listing.text[:20000]))


class Message(BaseModel):
    text: str
    current: dict | None = None  # the flat already on screen, if any
    history: list[dict] | None = None  # the last few chat turns, for the AI's reply
    context: dict | None = None  # what earlier short messages already told us: town, street, flat type, price, floor, size


class Login(BaseModel):
    email: str
    password: str


class Saved(BaseModel):
    flats: list[dict]


# ---- accounts: a session cookie for people who sign in, an anonymous id for everyone else (so usage can be counted)
def visitor(request, response):
    """('u:<id>' or 'a:<anonymous id>', the signed-in user or None)."""
    user = acc.user_for(request.cookies.get("oikos_session"))
    if user:
        return f"u:{user['id']}", user
    anon = request.cookies.get("oikos_anon")
    if not anon or not re.fullmatch(r"[\w-]{16,64}", anon):
        anon = acc.secrets.token_urlsafe(18)
        response.set_cookie("oikos_anon", anon, max_age=365 * 86400, httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return f"a:{anon}", None


def me_view(who, user):
    plan = acc.plan_of(who)
    return dict(user=dict(email=user["email"]) if user else None, plan=plan, plan_name=acc.PLANS[plan]["name"], used_today=acc.used_today(who),
                limit=acc.limit_for(who), plans=[dict(id=k, **v) for k, v in acc.PLANS.items() if k != "visitor"])


class Want(BaseModel):
    plan: str


@app.post("/api/interest")
def interest(body: Want, request: Request, response: Response):
    """Paid plans cannot be bought yet. This records that a signed-in person asked for one."""
    user = visitor(request, response)[1]
    if not user:
        raise HTTPException(401, "Create a free account first.")
    try:
        acc.want_plan(user["id"], body.plan)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return dict(ok=True)


def start_session(user_id, request, response):
    response.set_cookie("oikos_session", acc.new_session(user_id), max_age=acc.SESSION_DAYS * 86400, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")
    return me_view(f"u:{user_id}", acc.user_for_id(user_id))


@app.get("/api/me")
def me(request: Request, response: Response):
    return me_view(*visitor(request, response))


@app.post("/api/signup")
def signup(body: Login, request: Request, response: Response):
    try:
        return start_session(acc.signup(body.email, body.password), request, response)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    try:
        uid = acc.login(body.email, body.password)
    except ValueError as e:
        raise HTTPException(429, str(e))
    if uid is None:
        raise HTTPException(401, "Wrong email or password.")
    return start_session(uid, request, response)


@app.post("/api/logout")
def logout(request: Request, response: Response):
    acc.end_session(request.cookies.get("oikos_session"))
    response.delete_cookie("oikos_session")
    return dict(ok=True)


@app.get("/api/flats")
def flats_get(request: Request, response: Response):
    user = visitor(request, response)[1]
    if not user:
        raise HTTPException(401, "Sign in to keep your flats.")
    return dict(flats=acc.get_flats(user["id"]))


@app.post("/api/flats")
def flats_put(body: Saved, request: Request, response: Response):
    user = visitor(request, response)[1]
    if not user:
        raise HTTPException(401, "Sign in to keep your flats.")
    try:
        acc.put_flats(user["id"], body.flats[:200])
    except ValueError as e:
        raise HTTPException(413, str(e))
    return dict(ok=True)


WORK = re.compile(r"\b(?:i work|i'm working|my (?:office|work|workplace|job)(?: is)?|office is|work(?:ing)? (?:is )?(?:at|in|near)|commute|travel(?:ling)? to work|get to work)\b[^.?!]*?\b(?:at|in|to|near|is)\s+([A-Za-z0-9][A-Za-z0-9 '&-]{2,50})", re.I)
WATCH = re.compile(r"\b(alert me|notify me|let me know when|tell me when|watch this|keep an eye)\b", re.I)
QUESTION = re.compile(r"\?|\b(what if|what about|should|could|can i|would|how|why|worth it|is it|is this|explain|compare)\b", re.I)
SPEC = ("town", "street_name", "block", "flat_type", "flat_model", "floor_area_sqm", "storey_range", "lease_commence_date")


def band(floor):
    lo = (floor - 1) // 3 * 3 + 1
    return f"{lo:02d} TO {lo + 2:02d}"


OTHER_FLAT = re.compile(r"\b(blk|block|another|other flat|different flat|instead|avenue|ave|street|road|drive|lorong|jalan)\b", re.I)


def new_flat(p, base):
    """The reply for a message that names a flat: its details, with sensible defaults for anything not said."""
    o = options(town=p["town"], street=p["street_name"], block=p["block"], flat_type=p["flat_type"])
    sel = o["selected"]
    if sel["block"] != p["block"] or sel["street"] != p["street_name"]:
        return {**base, "note": f"I found blk {p['block']} {p['street_name'].title()}, but it has no resale records since 2017, so I can't price it."}
    d = sales()
    blk = d[(d["block"] == sel["block"]) & (d["street_name"] == sel["street"])]
    bands, df, assumed = sorted(blk["storey_range"].unique()), o["defaults"], []
    if p["flat_type"] != sel["flat_type"]:
        others = [t.title() for t in o["flat_types"] if t != sel["flat_type"]]
        if p["flat_type"]:  # they named a type this block does not have: say so plainly, the verdict depends on it
            assumed.append(f"a {sel['flat_type'].title()} flat, because this block has no {p['flat_type'].title()} flats on record" + (f" (it has {', '.join([sel['flat_type'].title()] + others)})" if others else ""))
        else:
            assumed.append(f"a {sel['flat_type'].title()} flat" + (f" (this block also has {', '.join(others)})" if others else ""))
    area = p["floor_area_sqm"] or df["floor_area_sqm"]
    if not p["floor_area_sqm"]:
        assumed.append(f"{round(area)} sqm, the usual size here")
    if p["floor"]:
        storey = band(p["floor"])
    elif p["storey_hint"] in ("high", "low") and bands:
        storey = bands[-1] if p["storey_hint"] == "high" else bands[0]
        assumed.append(f"storey {storey.replace(' TO ', ' to ')} for a {p['storey_hint']} floor")
    else:
        storey = df["storey_range"]
        assumed.append(f"storey {storey.replace(' TO ', ' to ')}, the most common here")
    flat = dict(town=sel["town"], street_name=sel["street"], block=sel["block"], flat_type=sel["flat_type"],
                flat_model=p["flat_model"] if p["flat_model"] in o["flat_models"] else df["flat_model"],
                floor_area_sqm=float(area), storey_range=storey, lease_commence_date=df["lease_commence_date"])
    return clean({**base, "found": True, "flat": flat, "price": p["asking"], "assumed": assumed})


def kfmt(v):
    return f"${v / 1e6:.2f}m" if v >= 1e6 else f"${round(v / 1000)}k"


def area_stats(town, street=None, flat_type=None, price=None):
    """What flats sold for in a town (or on one street) lately: the answer when someone has not named an exact flat."""
    d = sales()
    d = d[d["town"] == town]
    if street:
        d = d[d["street_name"] == street]
    d = d.assign(price=d["price_per_sqm"] * d["floor_area_sqm"])
    end = d["sale_date"].max()
    recent = d[d["sale_date"] > end - pd.DateOffset(months=12)]
    by_type = [dict(flat_type=t, median=float(g["price"].median()), low=float(g["price"].quantile(.25)), high=float(g["price"].quantile(.75)), n=int(len(g)))
               for t, g in recent.groupby("flat_type") if len(g) >= 3]
    out = dict(town=town, street_name=street, flat_type=flat_type, months=12, by_type=by_type, n=int(len(recent)))
    if flat_type:
        g, months = recent[recent["flat_type"] == flat_type], 12
        if len(g) < 8:  # thin: look back further
            g, months = d[(d["flat_type"] == flat_type) & (d["sale_date"] > end - pd.DateOffset(months=36))], 36
        if len(g) < 3:
            return dict(out, flat_type=None, missing_type=flat_type)
        before = d[(d["flat_type"] == flat_type) & (d["sale_date"] <= end - pd.DateOffset(months=12)) & (d["sale_date"] > end - pd.DateOffset(months=24))]
        out.update(months=months, n=int(len(g)), median=float(g["price"].median()), low=float(g["price"].quantile(.25)), high=float(g["price"].quantile(.75)),
                   typical_sqm=float(g["floor_area_sqm"].median()),
                   change=float(g["price_per_sqm"].median() / before["price_per_sqm"].median() - 1) if months == 12 and len(before) >= 10 and len(g) >= 10 else None)
        if price and len(g) >= 10:
            out.update(price=float(price), price_pct=float((g["price"] < price).mean() * 100))
        src = g
    else:
        src = recent
    key = "block" if street else "street_name"
    out["top"] = [str(x) for x in src[key].value_counts().head(6).index]
    return out


def area_answer(a, text, who):
    """The reply for a town or street: the numbers for the card, a short comment, and choices that narrow it down."""
    where = (f"on {a['street_name'].title()}" if a["street_name"] else f"in {a['town'].title()}")
    if a.get("flat_type"):
        ask = "Which block is it?" if a["street_name"] else "Which block and street is it?"
        view = dict(area=where, flat_type=a["flat_type"].title(), sales_counted=a["n"], months=a["months"], typical_price=round(a["median"], -3),
                    half_sold_between=[round(a["low"], -3), round(a["high"], -3)], typical_size_sqm=round(a["typical_sqm"]), ask_next=ask)
        if a.get("change") is not None:
            view["price_per_sqm_change_on_year_before_pct"] = round(a["change"] * 100, 1)
        if a.get("price_pct") is not None:
            view.update(their_price=round(a["price"]), share_of_these_sales_cheaper_than_their_price_pct=round(a["price_pct"]),
                        share_of_these_sales_dearer_than_their_price_pct=100 - round(a["price_pct"]))
        rule = (f"{a['flat_type'].title()} flats {where} sold for about {kfmt(a['median'])} over the last {a['months']} months; half went for between {kfmt(a['low'])} and {kfmt(a['high'])}. "
                + ((f"{kfmt(a['price'])} is cheaper than about {100 - round(a['price_pct'])}% of those sales" if a["price_pct"] < 50 else f"{kfmt(a['price'])} is dearer than about {round(a['price_pct'])}% of those sales")
                   + ", as a rough guide. " if a.get("price_pct") is not None else "") + ask + " Then I can check that exact flat.")
        choices = ([dict(label=f"Blk {b}", send=f"Blk {b} {a['street_name'].title()}") for b in a["top"]] if a["street_name"]
                   else [dict(label=x.title(), send=x.title()) for x in a["top"]])
    else:
        ask = "Which flat type are you looking at?"
        view = dict(area=where, months=a["months"], typical_price_by_flat_type={t["flat_type"].title(): round(t["median"], -3) for t in a["by_type"]}, ask_next=ask)
        rule = (f"I don't have enough recent {a['missing_type'].title()} sales {where}. " if a.get("missing_type") else "") + f"Here is what flats {where} sold for over the last 12 months. " + ask
        choices = [dict(label=t["flat_type"].title(), send=t["flat_type"].title()) for t in a["by_type"]]
    if who and acc.over_limit(who):
        reply, made = rule, False
    else:
        reply, cost, made = llm.area_reply(view, text, rule)
        if who:
            acc.record(who, "area", cost, made)
    return dict(area=a, reply=reply, ai=made, choices=choices)


def num(x, lo, hi, kind=float):
    try:
        v = kind(x)
        return v if lo <= v <= hi else None
    except (TypeError, ValueError):
        return None


@app.post("/api/understand")
def understand(m: Message, request: Request = None, response: Response = None):
    """Read a chat message. Returns the flat it describes (with sensible defaults for anything not said), or
    changes to the flat on screen, or found=False: then either `reply` holds the AI's answer, or the message is a
    question about the flat on screen. Rules read the message first; what they miss goes to the AI."""
    p = lp.parse(m.text[:20000])
    cur = {k: m.current[k] for k in SPEC} if m.current and all(k in m.current for k in SPEC) else None
    base = dict(found=False, role=p["role"], age=p["age"], price=None, assumed=[], changed=[], reply=None, watch=bool(WATCH.search(m.text[:600])))
    w = WORK.search(m.text[:600])
    base["work"] = re.sub(r"\s+(?:and|but|so|then|every|daily|now)\b.*$", "", w.group(1).strip(), flags=re.I) if w else None
    if base["work"] and not p["block"]:  # "I work at one-north": the page works out the commute
        return base
    if base["watch"] and not p["block"]:  # "alert me when this block has a new sale": the page sets the alert up
        return base
    asks = bool(QUESTION.search(m.text[:600]))
    same = cur and p["block"] == cur["block"] and p["street_name"] == cur["street_name"]

    # Short messages build up: "bishan", then "4 room", then "blk 123". What earlier ones said arrives in m.context.
    ctx = {k: v for k, v in (m.context or {}).items() if k in ("town", "street_name", "block", "flat_type", "price", "floor", "sqm") and v} if not cur else {}
    if ctx and not p["block"]:
        p["flat_type"] = p["flat_type"] or (ctx.get("flat_type") if ctx.get("flat_type") in ("1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE") else None)
        p["asking"] = p["asking"] or num(ctx.get("price"), 100_000, 3_000_000, int)
        p["floor"] = p["floor"] or num(ctx.get("floor"), 1, 60, int)
        p["floor_area_sqm"] = p["floor_area_sqm"] or num(ctx.get("sqm"), 20, 400)
    elif ctx:  # a full address this time: still use the type, price, floor and size given earlier
        p["flat_type"] = p["flat_type"] or ctx.get("flat_type")
        p["asking"] = p["asking"] or num(ctx.get("price"), 100_000, 3_000_000, int)
        p["floor"] = p["floor"] or num(ctx.get("floor"), 1, 60, int)
        p["floor_area_sqm"] = p["floor_area_sqm"] or num(ctx.get("sqm"), 20, 400)

    if p["block"] and not same:  # a new flat
        return new_flat(p, base)

    who = visitor(request, response)[0] if request is not None else None
    towns, streets = lp.places()
    known_street = {v[0]: v[1] for v in streets.values()}
    street = p["street_hint"] or (ctx.get("street_name") if ctx.get("street_name") in known_street and not p["town_hint"] else None)
    town = known_street[street] if street else (p["town_hint"] or (ctx.get("town") if ctx.get("town") in towns else None))
    block = p["block_hint"] or (str(ctx.get("block"))[:6].upper() if ctx.get("block") and not (p["street_hint"] or p["town_hint"]) else None)
    new_place = (p["street_hint"] or p["town_hint"] or p["block_hint"]) and not (cur and (asks or (town == cur["town"] and not p["block_hint"])))
    if not cur or new_place:
        base["ctx"] = dict(town=town, street_name=street, block=block, flat_type=p["flat_type"], price=p["asking"], floor=p["floor"], sqm=p["floor_area_sqm"])
        if block:  # a block number without a full address: find where it could be
            cands = [h for h in lp.block_index().get(block, []) if (not street or h[1] == street) and (not town or h[2] == town)]
            if len(cands) == 1:
                return new_flat(dict(p, block=block, street_name=cands[0][1], town=cands[0][2]), {k: v for k, v in base.items() if k != "ctx"})
            if cands:
                base["ctx"].update(street_name=None)
                return clean({**base, "reply": f"There is a Blk {block} on {len(cands)} streets{' in ' + town.title() if town else ''}. Which one is it?",
                              "choices": [dict(label=f"{h[1].title()}" + ("" if town else f" ({h[2].title()})"), send=f"Blk {block} {h[1].title()}") for h in sorted(cands, key=lambda h: h[1])[:10]]})
            base["ctx"].update(block=None)
            if not (street or town):
                return clean({**base, "note": f"I can't find a Blk {block} in HDB resale records. Check the number, or tell me the street."})
        if not town and re.search(r"\bjurong\b", m.text, re.I):
            return clean({**base, "reply": "Jurong East or Jurong West?", "choices": [dict(label="Jurong East", send="Jurong East"), dict(label="Jurong West", send="Jurong West")]})
        if not town and (p["flat_type"] or p["asking"]) and not p["street_hint"] and len(m.text) < 40 and not asks:  # "4 room", "600k": we need a place next
            return clean({**base, "reply": "Which town is it in? Or give me the block and street if you have them.",
                          "choices": [dict(label=t.title(), send=t.title()) for t in ("ANG MO KIO", "BEDOK", "BISHAN", "JURONG WEST", "PUNGGOL", "SENGKANG", "TAMPINES", "WOODLANDS", "YISHUN")]})
        if town:
            return clean({**base, **area_answer(area_stats(town, street, p["flat_type"], p["asking"]), m.text, who)})

    if cur and not asks:  # a correction to the flat on screen: "it's on the 11th floor", "asking is 400k", "it's 68 sqm"
        flat, changed = dict(cur), []
        if p["floor"]:
            flat["storey_range"] = band(p["floor"])
            changed.append("storey " + flat["storey_range"].replace(" TO ", " to "))
        elif p["storey_hint"] in ("high", "low"):
            d = sales()
            bands = sorted(d[(d["block"] == cur["block"]) & (d["street_name"] == cur["street_name"])]["storey_range"].unique())
            if bands and (bands[-1] if p["storey_hint"] == "high" else bands[0]) != cur["storey_range"]:
                flat["storey_range"] = bands[-1] if p["storey_hint"] == "high" else bands[0]
                changed.append("storey " + flat["storey_range"].replace(" TO ", " to "))
        if p["floor_area_sqm"]:
            flat["floor_area_sqm"] = float(p["floor_area_sqm"])
            changed.append(f"{round(p['floor_area_sqm'])} sqm")
        if p["flat_type"] and p["flat_type"] != cur["flat_type"]:
            o = options(town=cur["town"], street=cur["street_name"], block=cur["block"], flat_type=p["flat_type"])
            if o["selected"]["flat_type"] == p["flat_type"]:
                flat.update(flat_type=p["flat_type"], flat_model=o["defaults"]["flat_model"])
                if not p["floor_area_sqm"]:
                    flat["floor_area_sqm"] = o["defaults"]["floor_area_sqm"]
                changed.append(p["flat_type"].title())
        if p["asking"]:
            changed.append(f"price ${p['asking']:,}")
        if changed:
            return clean({**base, "found": True, "flat": flat, "price": p["asking"], "changed": changed})

    # The rules found no flat. With no flat on screen, or when the message seems to name another one, let the AI read it.
    if (not cur or (not same and OTHER_FLAT.search(m.text[:600]))) and not (who and acc.over_limit(who)):
        r, cost = llm.route_message(m.text, m.history)
        if who:
            acc.record(who, "route", cost, bool(r))
        if r:
            hit = lp.match_block(r["block"], r["street"]) if r.get("block") and r.get("street") else None
            if hit and not (cur and str(r["block"]).upper() == cur["block"] and hit[0] == cur["street_name"]):
                ft = r.get("flat_type") if r.get("flat_type") in ("1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE") else None
                q = dict(p, block=str(r["block"]).upper(), street_name=hit[0], town=hit[1], flat_type=p["flat_type"] or ft,
                         floor=p["floor"] or num(r.get("floor"), 1, 60, int), floor_area_sqm=p["floor_area_sqm"] or num(r.get("sqm"), 20, 400),
                         asking=p["asking"] or num(r.get("price"), 100_000, 3_000_000, int))
                return new_flat(q, {**base, "role": p["role"] or (r.get("role") if r.get("role") in ("buyer", "seller") else None),
                                    "age": p["age"] or num(r.get("age"), 21, 90, int)})
            reply = r.get("reply") if isinstance(r.get("reply"), str) else ""
            if r.get("block") and r.get("street") and not hit:
                return {**base, "note": f"I couldn't find blk {str(r['block'])[:8]} {str(r['street'])[:60].title()} in HDB resale records. Check the block number and street, or pick the flat from the list."}
            if not cur and reply.strip() and not re.search(r"[$%]|\d{3,}", reply):  # a chat reply may not carry figures: nothing here to check them against
                return {**base, "reply": reply.strip()[:600]}
    return clean(base)


def deals_nearby(town):
    k = recent_oos()
    k = k[k["town"] == town]
    low = k[(k["gap"] <= -0.08) & (k["gap"] >= -0.25)].sort_values(["sale_date", "gap"], ascending=[False, True]).head(4)
    items = [dict(month=x.sale_date, block=x.block, street_name=x.street_name, flat_type=str(x.flat_type), floor_area_sqm=x.floor_area_sqm,
                  storey=f"{int(x.storey_mid - 1):02d}-{int(x.storey_mid + 1):02d}", sold=x.resale_price, estimate=x.pred, gap=x.gap)
             for x in low.itertuples()]
    return dict(town=town, items=items, n_low=int((k["gap"] <= -0.08).sum()), n_all=int(len(k)), months=6)


def trend(town, flat_type):
    d = sales()
    t = d[(d["town"] == town) & (d["flat_type"] == flat_type)].groupby("sale_year")["price_per_sqm"].agg(["median", "size"])
    t = t[t["size"] >= 5]
    return dict(years=t.index.tolist(), values=t["median"].round().tolist(), last_year_through=meta()["trained_through"])


@functools.lru_cache(maxsize=64)
def facts_for(town, street_name, block, flat_type, flat_model, floor_area_sqm, storey_range, lease_commence_date, price):
    """All the facts for one flat and price. Cached, so follow-up questions to Ask Oikos do not recompute them."""
    f = Flat(town=town, street_name=street_name, block=block, flat_type=flat_type, flat_model=flat_model, floor_area_sqm=floor_area_sqm,
             storey_range=storey_range, lease_commence_date=lease_commence_date, price=price)
    nums = [int(x) for x in re.findall(r"\d+", f.storey_range)]
    if len(nums) < 2:
        raise HTTPException(422, "storey_range must look like '07 TO 09'")
    spec = dict(flat_type=f.flat_type, town=f.town, flat_model=f.flat_model, block=f.block, street_name=f.street_name,
                floor_area_sqm=f.floor_area_sqm, storey_mid=(nums[0] + nums[1]) / 2, lease_commence_date=f.lease_commence_date)
    try:
        a = adv.assess(spec, asking=f.price or None)
    except ValueError as e:
        raise HTTPException(404, str(e))
    base, effects, _ = ex.drivers(spec)
    out = fc.forecast(spec, horizons=(3, 5, 10, 15, 20), today=a["today"])
    d = sales()
    blk = d[(d["block"] == f.block) & (d["street_name"] == f.street_name)]
    tops = [int(r.split(" TO ")[1]) for r in blk["storey_range"].unique() if " TO " in r]
    recent = [dict(month=r.sale_date, storey=r.storey_range.replace(" TO ", "-"), floor_area_sqm=r.floor_area_sqm, price=r.resale_price)
              for r in a["block_recent"].head(8).itertuples()]
    return clean(dict(
        flat=dict(f.model_dump(exclude={"price"}), storey=f"{nums[0]:02d}-{nums[1]:02d}", storey_low=nums[0], storey_high=nums[1],
                  top_storey=max(tops + [nums[1]])),
        price=f.price, fair=a["fair"], lease_now=a["lease_now"], quantiles=a["quantiles"],
        reliability=a["badge"], notes=a["notes"], checks=[c["message"] for c in adv.input_checks(spec)],
        average_flat=base, drivers=[[g, e] for g, e in effects.items()], what_ifs=[[k, v] for k, v in ex.what_if(spec).items()],
        recent=recent, outlook=[{k: v for k, v in r.items()} for r in out["rows"]], growth=math.expm1(out["growth_effective"]),
        deals=deals_nearby(f.town), trend=trend(f.town, f.flat_type), meta=meta(), nearby=nb.around_block(f.block, f.street_name)))



@app.post("/api/assess")
def assess(f: Flat):
    return facts_for(*flat_key(f))


def flat_key(f):
    return (f.town, f.street_name, f.block, f.flat_type, f.flat_model, float(f.floor_area_sqm), f.storey_range, int(f.lease_commence_date), f.price or None)


class Ask(Flat):
    role: str = "buyer"
    age: float = 35
    question: str | None = None
    history: list[dict] | None = None
    assumed: list[str] | None = None  # details the person did not give and we filled in


@app.post("/api/ask")
def ask(a: Ask, request: Request, response: Response):
    """Ask Oikos: a short plain-English explanation of this flat's options, written by a model from the facts."""
    if a.role not in ("buyer", "seller"):
        raise HTTPException(422, "role must be buyer or seller")
    who, user = visitor(request, response)
    if acc.over_limit(who):
        cap = acc.limit_for(who)
        more = ("A free account gives you more, or come back tomorrow." if not user else
                "You're on the top plan; they reset at midnight." if acc.plan_of(who) == "pro" else "A bigger plan gives you more, or they reset at midnight.")
        raise HTTPException(429, f"You've used today's {cap} AI answers. {more}")
    facts = facts_for(*flat_key(a))
    price, question = a.price or facts["fair"], a.question
    if not a.price:  # nothing to judge yet: say what the flat is worth instead of giving a verdict on a made-up price
        question = ("(No price has been given yet, so price_being_checked is only our fair value. Do not give a verdict on it. Say what this flat is worth, what to "
                    + ("offer" if a.role == "buyer" else "list it at") + ", and ask for the " + ("asking" if a.role == "buyer" else "listing") + " price.) " + (question or "What is this flat worth?"))
    if a.assumed:
        question = ("(The person did not say these, so we assumed: " + "; ".join(str(x)[:120] for x in a.assumed[:4])
                    + ". Mention that in one short sentence and invite a correction.) " + (question or "Explain my options."))
    out = llm.explain(facts, a.role, a.age, price, question, a.history)
    acc.record(who, "ask", out.get("cost_cents"), out.get("source") == "model")
    if not a.question and out.get("source") == "model":  # the opening advice, kept so a shared link can show it
        acc.put_advice(advice_key(a, a.role), out["text"])
    return clean(out)


def advice_key(f, role):
    return json.dumps([*flat_key(f), role])


class Commute(BaseModel):
    block: str
    street_name: str
    place: str


@app.post("/api/commute")
def commute(c: Commute):
    """Public transport time from a block to a workplace or landmark, on a weekday morning."""
    pos = nb.block_coords().get((c.block, c.street_name))
    if not pos:
        raise HTTPException(404, "I don't have a map position for that block.")
    try:
        return nb.commute(pos[0], pos[1], c.place[:80])
    except nb.Unavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception:
        raise HTTPException(502, "The map service did not answer. Try again in a moment.")


class Share(Flat):
    role: str = "buyer"


@app.post("/api/share")
def share(sh: Share, request: Request, response: Response):
    """A short link to this flat, price and role, with the advice Oikos gave for it (or the rule-based text if none is on file)."""
    if sh.role not in ("buyer", "seller"):
        raise HTTPException(422, "role must be buyer or seller")
    who = visitor(request, response)[0]
    if acc.recent_count(who, "share") >= 30:
        raise HTTPException(429, "That's a lot of links for one day. Try again tomorrow.")
    facts = facts_for(*flat_key(sh))  # also checks that the flat is real
    advice = acc.get_advice(advice_key(sh, sh.role)) or llm.fallback_text(facts, sh.role, 35, sh.price or facts["fair"])
    sid = acc.new_share(dict(flat=sh.model_dump(exclude={"price", "role"}), price=sh.price or None, role=sh.role, advice=advice))
    acc.record(who, "share", 0, False)
    return dict(id=sid)


@app.get("/api/share/{sid}")
def share_get(sid: str):
    out = acc.get_share(sid) if re.fullmatch(r"[\w-]{6,16}", sid) else None
    if not out:
        raise HTTPException(404, "This link has expired or never existed.")
    return out


@app.get("/s/{sid}")
def share_page(sid: str):
    return FileResponse(ROOT / "web/index.html")


# ---- alerts
class NewAlert(BaseModel):
    kind: str
    town: str
    flat_type: str | None = None
    street_name: str | None = None
    block: str | None = None


class Seen(BaseModel):
    keys: list[str]


def alert_matches(kind, spec):
    """Sales that match an alert, newest first. Each has a key, so we can tell which ones a person has seen."""
    if kind == "block":
        d = sales()
        m = d[(d["block"] == spec["block"]) & (d["street_name"] == spec["street_name"])]
        if spec.get("flat_type"):
            m = m[m["flat_type"] == spec["flat_type"]]
        m = m.sort_values("sale_date", ascending=False).head(60)
        rows = [dict(month=x.sale_date.strftime("%Y-%m"), flat_type=str(x.flat_type), storey=x.storey_range.replace(" TO ", "-"), floor_area_sqm=float(x.floor_area_sqm),
                     sold=float(round(x.price_per_sqm * x.floor_area_sqm))) for x in m.itertuples()]
    else:
        k = recent_oos()
        m = k[(k["town"] == spec["town"]) & (k["gap"] <= -0.08) & (k["gap"] >= -0.25)]
        if spec.get("flat_type"):
            m = m[m["flat_type"].astype(str) == spec["flat_type"]]
        m = m.sort_values("sale_date", ascending=False).head(200)
        rows = [dict(month=x.sale_date.strftime("%Y-%m"), block=str(x.block), street_name=str(x.street_name), flat_type=str(x.flat_type),
                     storey=f"{int(x.storey_mid - 1):02d}-{int(x.storey_mid + 1):02d}", floor_area_sqm=float(x.floor_area_sqm), sold=float(x.resale_price),
                     estimate=float(x.pred), gap=float(x.gap)) for x in m.itertuples()]
    for r in rows:
        r["key"] = "|".join(str(r.get(c, "")) for c in ("month", "block", "street_name", "flat_type", "storey", "floor_area_sqm", "sold"))
    return rows


def data_through(kind="block"):
    """The newest sale month on file: bargain alerts use the estimates file, which can be a month behind the sales."""
    return (recent_oos() if kind == "bargain" else sales())["sale_date"].max().strftime("%Y-%m")


def alert_label(kind, spec):
    kind_of = (spec["flat_type"].title() + " flats") if spec.get("flat_type") else "flats of any type"
    return (f"New sales at Blk {spec['block']} {spec['street_name'].title()}" + (f" ({spec['flat_type'].title()})" if spec.get("flat_type") else "") if kind == "block"
            else f"Bargains in {spec['town'].title()}: {kind_of} sold 8% or more under our estimate")


def signed_in(request, response):
    user = visitor(request, response)[1]
    if not user:
        raise HTTPException(401, "Sign in to use alerts.")
    return user


@app.get("/api/alerts")
def alerts_get(request: Request, response: Response):
    user = signed_in(request, response)
    out = []
    for a in acc.alerts_for(user["id"]):
        rows = alert_matches(a["kind"], a["spec"])
        new = [r for r in rows if r["month"] >= a["baseline"] and r["key"] not in a["seen"]]
        out.append(dict(id=a["id"], kind=a["kind"], label=alert_label(a["kind"], a["spec"]), created=a["created"], new=new, recent=rows[:3]))
    return clean(dict(alerts=out, limit=acc.PLANS[acc.plan_of(f"u:{user['id']}")]["alerts"], data_through=data_through()))


@app.post("/api/alerts")
def alerts_add(body: NewAlert, request: Request, response: Response):
    user = signed_in(request, response)
    d = sales()
    if body.kind == "block":
        if not ((d["block"] == body.block) & (d["street_name"] == body.street_name)).any():
            raise HTTPException(404, "I can't find that block.")
        spec = dict(town=body.town, street_name=body.street_name, block=body.block, flat_type=body.flat_type or None)
    elif body.kind == "bargain":
        if body.town not in set(d["town"].unique()):
            raise HTTPException(404, "I can't find that town.")
        spec = dict(town=body.town, flat_type=body.flat_type or None)
    else:
        raise HTTPException(422, "kind must be block or bargain")
    base = data_through(body.kind)  # sales already on file for the newest month count as seen; later ones are new
    seen = [r["key"] for r in alert_matches(body.kind, spec) if r["month"] >= base]
    try:
        aid = acc.add_alert(user["id"], body.kind, spec, base, seen)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return dict(id=aid, label=alert_label(body.kind, spec))


@app.post("/api/alerts/{aid}/seen")
def alerts_seen(aid: int, body: Seen, request: Request, response: Response):
    acc.mark_seen(signed_in(request, response)["id"], aid, [k[:200] for k in body.keys[:500]])
    return dict(ok=True)


@app.post("/api/alerts/{aid}/delete")
def alerts_delete(aid: int, request: Request, response: Response):
    acc.drop_alert(signed_in(request, response)["id"], aid)
    return dict(ok=True)


@app.get("/api/deals")
def deals(town: str):
    """Recent sales in a town that closed 8% or more under our estimate at the time."""
    return clean(deals_nearby(town))


@app.get("/api/trend")
def trend_api(town: str, flat_type: str):
    return clean(trend(town, flat_type))


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
