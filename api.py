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
  POST /api/compare   two homes side by side: the AI's comparison, every figure checked
  POST /api/find      blocks or condo projects where homes meeting a set of requirements sold in the last 12 months
  GET  /api/deals     below-estimate sales in a town       GET /api/trend   yearly price per sqm for a town and flat type

The page works out verdicts for any price from the returned price quantiles, so dragging the price needs no
new request. Everything returned comes from the models and the sales data; nothing is invented here.
"""
import functools
import json
import math
import os
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
import condo as cc  # noqa: E402
import condo_features as cf  # noqa: E402
import finder as fd  # noqa: E402

app = FastAPI(title="Oikos")


# ---- picking up a data refresh without a restart. src/update.py finishes by writing models/range.json; when that
# file changes, every cached table and model in this process is dropped and reloaded on the next request.
STAMPS = [ROOT / "models/range.json", ROOT / "models/condo.json"]
stamp = lambda: max((f.stat().st_mtime for f in STAMPS if f.exists()), default=0)
_fresh = dict(mtime=stamp(), checked=0.0)


def refresh_if_new(force_check=False):
    import time
    now = time.time()
    if not force_check and now - _fresh["checked"] < 60:
        return False
    _fresh["checked"] = now
    mtime = stamp()
    if mtime == _fresh["mtime"]:
        return False
    _fresh["mtime"] = mtime
    for mod in (sys.modules[__name__], adv, ex, fc, lp, nb, cc, fd):
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
    detail: str = "short"  # "long" when the Research option is on


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


def client_ip(request):
    """The visitor's network address as a daily hash (None on this machine itself), for the per-network allowance."""
    if request is None:
        return None
    ip = request.client.host if request.client else None
    if os.environ.get("OIKOS_TRUST_PROXY", "").strip() == "1":
        ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or ip
    return acc.ip_key(ip)


def gate(request, response):
    """(who, network address) for an AI answer, or a 429 that says which allowance ran out."""
    who, user = visitor(request, response)
    ip = client_ip(request)
    hit = acc.which_limit(who, ip)
    if hit == "network":
        raise HTTPException(429, "Visitors on this network have used today's free AI allowance. A free account gives you your own.")
    if hit:
        when = acc.windows(who)[hit]["resets_at"]
        at = sg_time(when, hit == "week")
        more = ("A free account gives you more." if not user else "" if acc.plan_of(who) == "pro" else "A bigger plan gives you more.")
        what = "this 5-hour session's AI allowance" if hit == "session" else "this week's AI allowance"
        raise HTTPException(429, f"You've used {what}. It resets {at}. {more} The price card and the numbers still work without it.".replace("  ", " "))
    return who, ip


def sg_time(epoch, with_day=False):
    """"at 3:20 pm" or "on Tue 18 Nov at 3:20 pm", Singapore time."""
    import time as _t
    if not epoch:
        return "soon"
    t = _t.gmtime(epoch + 8 * 3600)
    clock = _t.strftime("%I:%M %p", t).lstrip("0").lower()
    return f"on {_t.strftime('%a %d %b', t)} at {clock}" if with_day else f"at {clock}"


def answer(who, ip, facts, a, price, question):
    """The AI's answer, recorded. An answer that only turns the question away does not use up one of the person's answers."""
    if acc.budget_spent():
        out = dict(text=llm.fallback_text(facts, a.role, a.age, price), source="rules", reason="daily_budget", cost_cents=0.0)
    else:
        out = llm.explain(facts, a.role, a.age, price, question, a.history, profile=a.profile, assumed=a.assumed, detail=a.detail)
    out["counted"] = out.get("source") == "model" and not out.get("declined")
    acc.record(who, "ask", out.get("cost_cents"), out.get("source") == "model", ip, charge=out["counted"])
    return out


def me_view(who, user):
    plan = acc.plan_of(who)
    free = acc.PLANS["free"]
    plans = [dict(id=k, name=v["name"], price=v["price"], alerts=v["alerts"], blurb=v["blurb"], times_free=round(v["week"] / free["week"], 1),
                  session_answers=int(v["session"] / acc.TYPICAL_ANSWER), week_answers=int(v["week"] / acc.TYPICAL_ANSWER)) for k, v in acc.PLANS.items() if k != "visitor"]
    w = acc.windows(who)
    use = {k: dict(pct=round(v["pct"]), resets=sg_time(v["resets_at"], k == "week") if v["resets_at"] else None) for k, v in w.items()}
    return dict(user=dict(email=user["email"]) if user else None, plan=plan, plan_name=acc.PLANS[plan]["name"], usage=use, plans=plans)


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
    ip = client_ip(request)
    if acc.signups_today(ip) >= 3:
        raise HTTPException(429, "Too many new accounts from this network today. Try again tomorrow.")
    try:
        uid = acc.signup(body.email, body.password)
        acc.record(f"u:{uid}", "signup", 0, False, ip)
        return start_session(uid, request, response)
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
COMPARE = re.compile(r"\b(compare|compared|comparing|comparison|versus|vs|better|worse|which one|which is|which should|which would|rather|instead of|cheaper than|dearer than)\b", re.I)
SWITCH = re.compile(r"\b(what about|how about|what of|look at|check|instead|switch|another|as well|also)\b", re.I)
BEDS = re.compile(r"\b[1-5][\s-]*(?:bed|bedroom|bedder|br)s?\b", re.I)
TIMING = re.compile(r"\b(good time|right time|bad time|best time|should (?:i|we) wait|wait (?:for|till|until|a bit|a while|a year|and see)|buy now|sell now|hold off|"
                    r"prices? (?:are |is |will |going to |to |keep )?(?:\w+ )?(?:drop|dropping|fall|falling|fell|crash|rise|rising|going up|going down|go up|go down|increase|decrease|peak|peaked|soften|cool)|"
                    r"market (?:now|doing|trend|outlook|crash|cooling|heading|conditions?)|bubble|how is the market|hows the market|how's the market)\b", re.I)


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
        if not street:  # the cheapest and dearest streets for this flat type, for the comment under the card
            st = g.groupby("street_name")["price"].agg(["median", "size"])
            st = st[st["size"] >= 5].sort_values("median")
            if len(st) >= 3:
                out["streets"] = [[str(st.index[0]), float(st["median"].iloc[0])], [str(st.index[-1]), float(st["median"].iloc[-1])]]
        src = g
    else:
        src = recent
    key = "block" if street else "street_name"
    out["top"] = [str(x) for x in src[key].value_counts().head(6).index]
    return out


def area_answer(a, text, who, ip=None):
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
        if a.get("streets"):
            view["cheapest_and_dearest_streets_typical_price"] = {x[0].title(): round(x[1], -3) for x in a["streets"]}
        # the card already shows the typical price and the range, so the words only add what it does not
        rule = ((f"{kfmt(a['price'])} is cheaper than about {100 - round(a['price_pct'])}% of those sales" if a["price_pct"] < 50 else f"{kfmt(a['price'])} is dearer than about {round(a['price_pct'])}% of those sales")
                + ", a rough guide until I know the exact flat. " if a.get("price_pct") is not None else f"That is what {a['flat_type'].title()} flats {where} have sold for lately. ") + ask
        choices = ([dict(label=f"Blk {b}", send=f"Blk {b} {a['street_name'].title()}") for b in a["top"]] if a["street_name"]
                   else [dict(label=x.title(), send=x.title()) for x in a["top"]])
    else:
        ask = "Which flat type are you looking at?"
        view = dict(area=where, months=a["months"], typical_price_by_flat_type={t["flat_type"].title(): round(t["median"], -3) for t in a["by_type"]}, ask_next=ask)
        rule = (f"I don't have enough recent {a['missing_type'].title()} sales {where}. " if a.get("missing_type") else "") + f"Here is what flats {where} sold for over the last 12 months. " + ask
        choices = [dict(label=t["flat_type"].title(), send=t["flat_type"].title()) for t in a["by_type"]]
    if who and (acc.over_limit(who, ip) or acc.budget_spent()):
        reply, made = rule, False
    else:
        reply, cost, made = llm.area_reply(view, text, rule)
        if who:
            acc.record(who, "area", cost, made, ip)
    return dict(area=a, reply=reply, ai=made, choices=choices)


# ---- "is now a good time?": what prices and sales have done lately. Never a forecast.
def like_for_like(now, then):
    """Change in price per sqm between two sets of sales, comparing the same town, flat type and age group, weighted by sales. None if too thin."""
    key = ["town", "flat_type", "older"]
    a = now.groupby(key, observed=True)["price_per_sqm"].agg(med="median", n="size")
    b = then.groupby(key, observed=True)["price_per_sqm"].agg(med="median", n="size")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["n_a"] >= 5) & (j["n_b"] >= 5)]
    w = j[["n_a", "n_b"]].min(axis=1)
    if w.sum() < 40:
        return None
    return float(np.expm1((w * np.log(j["med_a"] / j["med_b"])).sum() / w.sum()))


@functools.lru_cache(maxsize=64)
def market_stats(town=None):
    """The last three complete months against the same months a year before (and the three months before), like for like."""
    d, end = sales(), pd.Timestamp(meta()["trained_through"] + "-01")
    d = d[d["sale_date"] <= end]
    if town:
        d = d[d["town"] == town]
    d = d.assign(older=(d["sale_year"] - d["lease_commence_date"]) > 20)
    win = lambda a, b: d[(d["sale_date"] > end - pd.DateOffset(months=a)) & (d["sale_date"] <= end - pd.DateOffset(months=b))]
    now, prev, yago = win(3, 0), win(6, 3), win(15, 12)
    rows = []
    for t in ("3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"):
        f = lambda x: x[x["flat_type"] == t]
        ch = like_for_like(f(now), f(yago))
        if ch is not None:
            rows.append(dict(flat_type=t, change_year=ch, change_quarter=like_for_like(f(now), f(prev)), sales=int(len(f(now))), sales_year_ago=int(len(f(yago)))))
    return dict(town=town, months=f"{end - pd.DateOffset(months=2):%b} to {end:%b %Y}", all=like_for_like(now, yago), sales=int(len(now)), sales_year_ago=int(len(yago)), by_type=rows,
                newer=like_for_like(now[~now["older"]], yago[~yago["older"]]), older=like_for_like(now[now["older"]], yago[yago["older"]]))


def pct1(x):
    return None if x is None else round(x * 100, 1)


def market_for_flat(town, flat_type, lease_commence_date):
    """The market facts the model may use when asked about timing, for one flat's town, type and age group."""
    isl, twn = market_stats(None), market_stats(town)
    row = lambda m: next((dict(price_per_sqm_change_pct=pct1(r["change_year"]), sales=r["sales"], sales_a_year_before=r["sales_year_ago"]) for r in m["by_type"] if r["flat_type"] == flat_type), None)
    older = int(meta()["trained_through"][:4]) - int(lease_commence_date) > 20
    return {"what_this_is": f"what resale prices did over {isl['months']} against the same months a year before, like for like. Not a forecast.",
            f"{flat_type.title()} flats in {town.title()}": row(twn), f"{flat_type.title()} flats across Singapore": row(isl),
            ("flats over 20 years old" if older else "flats up to 20 years old") + " across Singapore, price_per_sqm_change_pct": pct1(isl["older" if older else "newer"])}


def market_answer(town, text, who, ip=None):
    m = market_stats(town)
    if m["all"] is None:
        m = market_stats(None)
    where = f"in {m['town'].title()}" if m["town"] else "across Singapore"
    ask = "Which flat are you looking at?"
    view = dict(area=where, period=f"{m['months']} against the same months a year before, like for like", all_flats_price_per_sqm_change_pct=pct1(m["all"]), sales=m["sales"], sales_a_year_before=m["sales_year_ago"],
                by_flat_type={r["flat_type"].title(): dict(price_per_sqm_change_pct=pct1(r["change_year"]), sales=r["sales"], sales_a_year_before=r["sales_year_ago"]) for r in m["by_type"]},
                flats_up_to_20_years_old_change_pct=pct1(m["newer"]), flats_over_20_years_old_change_pct=pct1(m["older"]), ask_next=ask)
    rule = (f"I can't tell you where prices go next, and nobody can reliably. What I can show is what they have done: over {m['months']}, resale prices {where} were {m['all'] * 100:+.1f}% on the same months a year before, "
            f"like for like, with {m['sales']:,} sales against {m['sales_year_ago']:,}. Give me a block and street and I'll tell you whether that flat is fairly priced today. " + ask)
    if who and (acc.over_limit(who, ip) or acc.budget_spent()):
        reply, made = rule, False
    else:
        reply, cost, made = llm.market_reply(view, text, rule)
        if who:
            acc.record(who, "area", cost, made, ip)
    return dict(market=m, reply=reply, ai=made)


# ---- "find me a 4 room under 600k near an MRT": where such homes have sold (finder.py). Sold records, not live listings.
FIND_BEYOND = ("budget", "mrt_m", "school", "min_lease", "max_age", "min_sqm", "region", "districts", "segment", "high_floor", "beds", "freehold")


def find_view(r, ask):
    return dict(requirements=r["what"], months=12, nothing_within_budget=r["over"], sales_that_fit=r["sales"], sales_of_this_kind_at_any_price=r["all_sales"], blocks_or_projects_that_fit=r.get("blocks"),
                where_the_matches_are=[dict(place=x["place"], sales=x["sales"], typical_price=round(x["median"], -3)) for x in r["by_place"]], listed_in=r.get("order_words"),
                matches_on_the_card=[{k: (round(v, -3) if k in ("median", "low", "high") else round(v) if k in ("sqm", "psm") else v) for k, v in x.items()
                                      if k in ("name", "town", "district", "flat_type", "tenure", "n_in", "n_all", "median", "low", "high", "sqm", "psm", "lease_left", "mrt", "mrt_m", "walk_min")} for x in r["items"][:6]],
                ask_next=ask)


def find_both(spec, text, who, ip=None, comment=True, detail="short"):
    """"HDBs and condos under 2m": one card for each, and one comment that covers both."""
    rh, rc = fd.search(fd.narrow(spec, "hdb")), fd.search(fd.narrow(spec, "condo"))
    ask = "What is the most you would pay?" if not spec["budget"] else "Tap a block or project and I'll check a home there, or tell me what to change."
    part = lambda r, place: (f"{r['sales']:,} HDB resales fit in {r.get('blocks', 0):,} {place}s" if place == "block" else f"{r['sales']:,} condo resales in {r.get('blocks', 0):,} {place}s") + (
        ", most in " + ", ".join(f"{x['place']} (around {fd.kfmt(x['median'])})" for x in r["by_place"][:2]) if r["by_place"] and not r["over"] else "")
    rule = (f"In the last 12 months, {part(rh, 'block')}; {part(rc, 'project')}. The cards list them, {rh.get('order_words') or 'in order'}. "
            "These are past sales, not homes on sale today: look for a unit on sale in one of these places, then send it to me to check the price. " + ask)
    view = dict(hdb=find_view(rh, ask), condos=find_view(rc, ask), ask_next=ask)
    if not comment or (who and (acc.over_limit(who, ip) or acc.budget_spent())):
        reply, made = rule, False
    else:
        reply, cost, made = llm.find_reply(view, text, rule, detail=detail)
        if who:
            acc.record(who, "ask" if detail == "long" else "area", cost, made, ip)
    return dict(find=rh, find2=rc, reply=reply, ai=made, ctx=dict(find=spec))


def find_answer(spec, text, who, ip=None, comment=True, detail="short"):
    if spec["kind"] == "both":
        return find_both(spec, text, who, ip, comment, detail)
    r = fd.search(spec)
    place = "project" if r["kind"] == "condo" else "block"
    ask = ("What is the most you would pay?" if not spec["budget"] else "Which part of Singapore do you have in mind?" if r["kind"] == "hdb" and not (spec["towns"] or spec["region"] or spec["station"] or spec["school"]) and r.get("blocks", 0) > 40
           else f"Tap a {place} and I'll check a home there, or tell me what to change.")
    if not r["items"]:
        rule = f"I found no sales of {r['what']} in the last 12 months. Try a wider area, a bigger budget or fewer conditions."
        return dict(find=r, reply=rule, ai=False, ctx=dict(find=spec))
    top = r["by_place"][:3]
    if r["over"]:
        rule = (f"Nothing like that sold within your budget in the last 12 months: of {r['all_sales']:,} sales of this kind, the cheapest went for {fd.kfmt(min(x['low'] for x in r['items']))}. "
                f"The card shows the ones that came closest. " + ask)
    else:
        rule = (f"{r['sales']:,} sale{'s' if r['sales'] != 1 else ''} in {r['blocks']:,} {place}{'s' if r['blocks'] != 1 else ''} fit that in the last 12 months" + (", most of them in " + ", ".join(f"{x['place']} (around {fd.kfmt(x['median'])})" for x in top) if top else "")
                + f". These are past sales, not homes on sale today: look for a unit on sale in one of these {place}s, then send it to me to check the price. " + ask)
    view = find_view(r, ask)
    if not comment or (who and (acc.over_limit(who, ip) or acc.budget_spent())):
        reply, made = rule, False
    else:
        reply, cost, made = llm.find_reply(view, text, rule, detail=detail)
        if who:
            acc.record(who, "ask" if detail == "long" else "area", cost, made, ip)
    return dict(find=r, reply=reply, ai=made, ctx=dict(find=spec))


class Find(BaseModel):
    spec: dict


@app.post("/api/find")
def find(body: Find):
    """The same search again with the requirements changed (the sort buttons on the card). No AI call."""
    spec = {**fd.EMPTY, **{k: v for k, v in body.spec.items() if k in fd.EMPTY}}
    if spec["kind"] not in ("hdb", "condo", "both") or spec["sort"] not in (None, *fd.ORDERS) or not isinstance(spec["towns"], list) or not isinstance(spec["flat_types"], list):
        raise HTTPException(422, "Those requirements are not in a form I can search.")
    try:
        return clean(find_answer(spec, "", None, comment=False))
    except (TypeError, ValueError, KeyError):
        raise HTTPException(422, "Those requirements are not in a form I can search.")


# ---- how close our estimates have been for homes like this one, on sales we estimated before they happened
@functools.lru_cache(maxsize=1)
def oos_track():
    d = pd.concat([pd.read_parquet(f, columns=["sale_date", "town", "flat_type", "lease_commence_date", "pred", "resale_price"]) for f in sorted((ROOT / "data/processed/oos").glob("chunk_*.parquet"))], ignore_index=True)
    d = d[d["sale_date"] > d["sale_date"].max() - pd.DateOffset(months=18)]
    return pd.DataFrame(dict(town=d["town"].astype(str), flat_type=d["flat_type"].astype(str), age=d["sale_date"].dt.year - d["lease_commence_date"], pred=d["pred"],
                             miss=(d["resale_price"] / d["pred"] - 1).abs()))


def track_summary(g, group, months=18):
    share = float((g["miss"] > 0.10).mean() * 100)
    return dict(sales_we_estimated_beforehand=int(len(g)), months=months, homes_like_this=group, half_were_within_pct=round(float(g["miss"].median() * 100), 1),
                within_5pct_share_pct=round(float((g["miss"] <= 0.05).mean() * 100)), off_by_more_than_10pct_share_pct=round(share, 1),
                off_by_more_than_10pct_in_words="fewer than 1 in 100" if share < 0.8 else f"about 1 in {round(100 / share)}")


def track_record(town, flat_type, lease_commence_date, fair):
    d, age, ft = oos_track(), int(meta()["trained_through"][:4]) - int(lease_commence_date), flat_type.title()
    same, near, price = (d["town"] == town) & (d["flat_type"] == flat_type), (d["age"] - age).abs() <= 8, (d["pred"] / fair - 1).abs() <= 0.20
    span = f"{max(0, age - 8)} to {age + 8} years old"
    for mask, group in ((same & near & price, f"{ft} flats in {town.title()}, {span}, estimated within 20% of this one"), (same & near, f"{ft} flats in {town.title()}, {span}"),
                        (same, f"{ft} flats in {town.title()}"), ((d["flat_type"] == flat_type) & near & price, f"{ft} flats across Singapore, {span}, estimated within 20% of this one"),
                        (d["flat_type"] == flat_type, f"{ft} flats across Singapore")):
        if mask.sum() >= 150:
            return track_summary(d[mask], group)
    return track_summary(d, "HDB flats across Singapore")


@functools.lru_cache(maxsize=1)
def condo_track_data():
    files = sorted((ROOT / "data/processed/condo_oos").glob("chunk_*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["sale_date", "project", "property_type", "market_segment", "area", "lp", "pred_lp"]) for f in files], ignore_index=True)
    d = d[d["sale_date"] > d["sale_date"].max() - pd.DateOffset(months=18)]
    return pd.DataFrame(dict(project=d["project"].astype(str), property_type=d["property_type"].astype(str), market_segment=d["market_segment"].astype(str), area=d["area"],
                             miss=(np.exp(d["lp"] - d["pred_lp"]) - 1).abs()))


def condo_track(project, property_type, market_segment, area):
    d = condo_track_data()
    seg, size = (d["market_segment"] == market_segment) & (d["property_type"] == property_type), (d["area"] / area - 1).abs() <= 0.30
    kind = {"Condominium": "condos", "Apartment": "apartments", "Executive Condominium": "executive condos"}.get(property_type, "units")
    for mask, group, need in ((d["project"] == project, f"resales in {project.title()}", 40), (seg & size, f"{market_segment} {kind} of a similar size", 100), (seg, f"{market_segment} {kind}", 100)):
        if mask.sum() >= need:
            return track_summary(d[mask], group)
    return track_summary(d, "private condos and apartments")


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
    base = dict(found=False, role=p["role"], age=p["age"], profile=p["profile"], price=None, assumed=[], changed=[], reply=None, watch=bool(WATCH.search(m.text[:600])))
    who = visitor(request, response)[0] if request is not None else None
    ip = client_ip(request)
    w = WORK.search(m.text[:600])
    base["work"] = re.sub(r"\s+(?:and|but|so|then|every|daily|now)\b.*$", "", w.group(1).strip(), flags=re.I) if w else None
    if base["work"] and not p["block"]:  # "I work at one-north": the page works out the commute
        return base
    if base["watch"] and not p["block"]:  # "alert me when this block has a new sale": the page sets the alert up
        return base
    asks = bool(QUESTION.search(m.text[:600]))
    same = cur and p["block"] == cur["block"] and p["street_name"] == cur["street_name"]

    # A condo is on screen: the message changes it, names another home, or is a question about it.
    on_condo = m.current if (m.current or {}).get("kind") == "condo" and condo_ready() and m.current.get("project") in cc.projects().index else None
    named = cc.find_in_text(m.text[:2000]) if condo_ready() and not p["block"] else None
    on_screen = bool(cur or on_condo)
    compare = on_screen and bool(COMPARE.search(m.text[:600]))  # "is Parc Esta better?": put the two side by side, keep the first on screen
    details = bool(p["floor_area_sqm"] or p["floor"] or p["asking"] or p["storey_hint"] or BEDS.search(m.text))
    # Another condo is named. Go to it, unless this reads as a question about the home on screen that merely mentions it.
    if named and not (on_condo and named == on_condo["project"]) and (not on_screen or not asks or compare or details or SWITCH.search(m.text[:600])):
        return condo_answer(named, p, {**base, "compare": compare}, m.text)
    if on_condo:
        if not p["block"] and not asks and details:
            return condo_answer(on_condo["project"], p, base, m.text, cur=on_condo)
        if not p["block"]:
            return clean(base)  # a question about the condo on screen
    if not on_screen and not p["block"] and TIMING.search(m.text[:600]):  # "is now a good time?": what the market has done, never a forecast
        return clean({**base, **market_answer(p["town_hint"], m.text, who, ip)})
    # "find me a 4 room under 600k near an MRT", or a change to the last search ("only tampines", "make it 650k")
    last = (m.context or {}).get("find") if isinstance((m.context or {}).get("find"), dict) else None
    if not p["block"] and not named and p["role"] != "seller":
        want = fd.intent(m.text)
        if want == "strong":  # "find me...", "recommend...": a new search, not a change to the last one
            last = None
        spec, said = fd.parse(m.text[:2000], p["asking"], last)
        beyond = any(spec[k] for k in FIND_BEYOND) or len(spec["towns"]) >= 2
        if said and ((last and not on_screen) or want == "strong" or (want and beyond and (not on_screen or re.search(r"\b(looking for|budget)\b", m.text, re.I)))):
            return clean({**base, **find_answer(spec, m.text, who, ip, detail=m.detail)})

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
        return new_flat(p, {**base, "compare": compare})

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
            return clean({**base, **area_answer(area_stats(town, street, p["flat_type"], p["asking"]), m.text, who, ip)})

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
    if (not cur or (not same and OTHER_FLAT.search(m.text[:600]))) and not (who and (acc.over_limit(who, ip) or acc.budget_spent())):
        r, cost = llm.route_message(m.text, m.history)
        base = {**base, "compare": compare}
        if who:
            acc.record(who, "route", cost, bool(r), ip)
        if r and isinstance(r.get("search"), dict) and not on_screen and p["role"] != "seller":  # the AI read a request to find homes that the rules missed
            spec = fd.from_ai(r["search"], fd.parse(m.text[:2000], p["asking"], last)[0])
            if any(spec[k] for k in FIND_BEYOND) or spec["towns"] or spec["flat_types"]:
                return clean({**base, **find_answer(spec, m.text, who, ip, detail=m.detail)})
            return {**base, "reply": "Happy to look. Tell me your budget, the flat type or number of bedrooms, and the area you have in mind, and say if it must be near an MRT."}
        if r and r.get("condo") and condo_ready() and cc.find_project(str(r["condo"])[:80]):  # the AI read a condo name the rules missed (a typo, a nickname)
            q = dict(p, floor=p["floor"] or num(r.get("floor"), 1, 80, int), floor_area_sqm=p["floor_area_sqm"] or num(r.get("sqm"), 20, 1000),
                     asking=p["asking"] or num(r.get("price"), 100_000, 30_000_000, int))
            return condo_answer(cc.find_project(str(r["condo"])[:80]), q, {**base, "role": p["role"] or (r.get("role") if r.get("role") in ("buyer", "seller") else None)}, m.text)
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
        deals=deals_nearby(f.town), trend=trend(f.town, f.flat_type), meta=meta(), nearby=nb.around_block(f.block, f.street_name),
        market=market_for_flat(f.town, f.flat_type, f.lease_commence_date), track=track_record(f.town, f.flat_type, f.lease_commence_date, a["fair"])))



@app.post("/api/assess")
def assess(f: Flat):
    return facts_for(*flat_key(f))


def flat_key(f):
    return (f.town, f.street_name, f.block, f.flat_type, f.flat_model, float(f.floor_area_sqm), f.storey_range, int(f.lease_commence_date), f.price or None)


class Ask(Flat):
    role: str = "buyer"
    detail: str = "short"  # "long": the Research option, a longer and more detailed answer
    age: float | None = None  # the youngest buyer's age, only if they told us
    profile: dict | None = None  # what else they told us: who (alone or couple), first_home, timeline
    question: str | None = None
    history: list[dict] | None = None
    assumed: list[str] | None = None  # details the person did not give and we filled in


@app.post("/api/ask")
def ask(a: Ask, request: Request, response: Response):
    """Ask Oikos: a short plain-English explanation of this flat's options, written by a model from the facts."""
    if a.role not in ("buyer", "seller"):
        raise HTTPException(422, "role must be buyer or seller")
    who, ip = gate(request, response)
    facts = facts_for(*flat_key(a))
    price, question = a.price or facts["fair"], a.question
    if not a.price:  # nothing to judge yet: say what the flat is worth instead of giving a verdict on a made-up price
        question = ("(No price has been given yet, so price_being_checked is only our fair value. Do not give a verdict on it. Say what this flat is worth, what to "
                    + ("offer" if a.role == "buyer" else "list it at") + ", and ask for the " + ("asking" if a.role == "buyer" else "listing") + " price.) " + (question or "What is this flat worth?"))
    if a.assumed:
        question = ("(The person did not say these, so we assumed: " + "; ".join(str(x)[:120] for x in a.assumed[:4])
                    + ". Mention that in one short sentence and invite a correction.) " + (question or "Explain my options."))
    out = answer(who, ip, facts, a, price, question)
    if not a.question and out.get("source") == "model":  # the opening advice, kept so a shared link can show it
        acc.put_advice(advice_key(a, a.role), out["text"])
    return clean(out)


def advice_key(f, role):
    return json.dumps([*flat_key(f), role])


class Commute(BaseModel):
    place: str
    block: str | None = None
    street_name: str | None = None
    project: str | None = None  # a condo, instead of an HDB block


# ---- private condos and apartments: the same shape of facts as an HDB flat, so the chat treats both alike
class CondoSpec(BaseModel):
    project: str
    area: float
    floor: int | None = None
    price: float | None = None


class CondoAsk(CondoSpec):
    role: str = "buyer"
    detail: str = "short"
    age: float | None = None
    profile: dict | None = None
    question: str | None = None
    history: list[dict] | None = None
    assumed: list[str] | None = None


def condo_ready():
    return (ROOT / "models/condo.json").exists()


def condo_latlon(name):
    pr = cc.projects().loc[name]
    if pr["x"] != pr["x"]:
        return None
    lat, lon = cf.svy21_to_latlon([pr["x"]], [pr["y"]])[0]
    return float(lat), float(lon)


@functools.lru_cache(maxsize=64)
def condo_facts(project, area, floor, price):
    try:
        a = cc.assess(project, area, floor or 0, price)
    except ValueError as e:
        raise HTTPException(404, str(e))
    m, pos = cc.meta(), condo_latlon(a["project"])
    band = a["floor_band"] or "not given"
    return clean(dict(
        kind="condo", condo=dict(project=a["project"], area=a["area"], floor=floor), project=a["project"], street=a["street"], district=a["district"],
        property_type=a["property_type"], tenure=a["tenure"], freehold=a["freehold"],
        flat=dict(block="", street_name=a["project"], town=f"District {a['district']}", flat_type=a["property_type"], flat_model="", floor_area_sqm=a["area"], storey=band,
                  lease_commence_date=None),
        price=price, fair=a["fair"], lease_now=999.0 if a["freehold"] or a["lease_left"] is None else a["lease_left"], quantiles=[[lvl, v] for lvl, v in a["quantiles"]],
        reliability=a["badge"], notes=a["notes"], checks=[], drivers=[],
        recent=[dict(month=r["month"], storey=r["floor"], floor_area_sqm=r["area"], price=r["price"], sale_type=r["sale_type"]) for r in a["recent"]],
        meta=dict(trained_through=m["trained_through"], holdout_median_error=m["holdout_median_error"], total_sales=m["rows"]),
        nearby=dict(nb.around(*pos), lat=pos[0], lon=pos[1]) if pos else None,
        track=condo_track(a["project"], a["property_type"], a["market_segment"], a["area"])))


def condo_answer(name, p, base, text, cur=None):
    """The reply for a message that names a condo (or changes the one on screen): the unit, with anything not said filled in."""
    pr, assumed, changed = cc.projects().loc[name], [], []
    beds = re.search(r"\b([1-5])[\s-]*(?:bed|bedroom|bedder|br)s?\b", text, re.I)
    area = p["floor_area_sqm"] or (cur or {}).get("area")
    if not area and beds and cc.size_for_bedrooms(name, int(beds.group(1))):
        area = cc.size_for_bedrooms(name, int(beds.group(1)))
        assumed.append(f"about {round(area)} sqm for a {beds.group(1)}-bedroom here, which is my estimate from the sizes sold in this project")
    if not area:
        area = float(round(pr["typical_area"]))
        assumed.append(f"{round(area)} sqm, the typical unit sold here")
    floor = p["floor"] or (cur or {}).get("floor")
    top = int(pr["top_floor"]) if pr["top_floor"] == pr["top_floor"] and pr["top_floor"] > 0 else None
    if not p["floor"] and p["storey_hint"] and top:  # "high floor": the top band sold in this project; "low": the bottom one
        floor = top if p["storey_hint"] == "high" else 3 if p["storey_hint"] == "low" else max(3, top // 2)
        assumed.append(f"floors {(floor - 1) // 5 * 5 + 1} to {(floor - 1) // 5 * 5 + 5} for a {p['storey_hint']} floor")
        if cur:
            changed.append(f"{p['storey_hint']} floor")
    if not floor:
        assumed.append("no floor, so a mid-level unit")
    if cur:
        changed += [x for x in ([f"{round(p['floor_area_sqm'])} sqm"] if p["floor_area_sqm"] else []) + ([f"floor {p['floor']}"] if p["floor"] else [])
                   + ([f"price ${p['asking']:,}"] if p["asking"] else []) + ([f"{beds.group(1)}-bedroom"] if beds and not p["floor_area_sqm"] else [])]
    return clean({**base, "found": True, "kind": "condo", "condo": dict(project=name, area=float(area), floor=int(floor) if floor else None), "price": p["asking"],
                  "assumed": [] if cur and changed else assumed, "changed": changed})


@app.post("/api/condo/assess")
def condo_assess(c: CondoSpec):
    return condo_facts(c.project, float(c.area), c.floor, c.price or None)


@app.post("/api/condo/ask")
def condo_ask(a: CondoAsk, request: Request, response: Response):
    """The AI's advice on a condo unit, with every figure checked, under the same daily limits as HDB advice."""
    if a.role not in ("buyer", "seller"):
        raise HTTPException(422, "role must be buyer or seller")
    who, ip = gate(request, response)
    facts = condo_facts(a.project, float(a.area), a.floor, a.price or None)
    price = a.price or facts["fair"]
    note = ("(This is a private condo unit, not an HDB flat. Do not mention HDB, the HDB-registered size or CPF. Stamp duties (including ABSD), loans and "
            "en-bloc prospects are outside what you cover: do not guess at them. Call it a unit or a home, not a flat. " + ("It is freehold or as good as, so the lease is not a concern. " if facts["freehold"] else "")
            + ("No price has been given yet, so price_being_checked is only our fair value: do not give a verdict on it; say what the unit is worth and ask for the price. " if not a.price else "")
            + ("The person did not say these, so we assumed: " + "; ".join(str(x)[:140] for x in a.assumed[:4]) + ". Mention that in one short sentence and invite a correction. " if a.assumed else "")
            + ("The size is our guess and the verdict depends heavily on it: call the verdict provisional until they confirm the size, and do not tell them to move fast. " if any("sqm" in str(x) for x in (a.assumed or [])) else "") + ") ")
    return clean(answer(who, ip, facts, a, price, note + (a.question or "Explain my options.")))


class Home(BaseModel):
    kind: str = "hdb"
    flat: dict | None = None   # the Flat fields, for an HDB flat
    condo: dict | None = None  # project, area, floor
    price: float | None = None


class Pair(BaseModel):
    a: Home
    b: Home
    role: str = "buyer"
    detail: str = "short"
    question: str | None = None


def home_facts(h):
    try:
        if h.kind == "condo":
            c = CondoSpec(**(h.condo or {}))
            return condo_facts(c.project, float(c.area), c.floor, h.price or None), c.project.title()
        f = Flat(**(h.flat or {}), price=h.price or None)
        return facts_for(*flat_key(f)), f"Blk {f.block} {f.street_name.title()}"
    except (TypeError, ValueError):
        raise HTTPException(422, "That home's details are incomplete.")


@app.post("/api/compare")
def compare(c: Pair, request: Request, response: Response):
    """Two homes side by side, in the AI's words with every figure checked. Counts as one AI answer."""
    if c.role not in ("buyer", "seller"):
        raise HTTPException(422, "role must be buyer or seller")
    who, ip = gate(request, response)
    (fa, na), (fb, nb_) = home_facts(c.a), home_facts(c.b)
    if acc.budget_spent():
        text, cost, made = llm.compare_fallback(fa, fb, c.a.price, c.b.price, na, nb_), 0.0, False
    else:
        text, cost, made = llm.compare(fa, fb, c.role, c.a.price, c.b.price, na, nb_, (c.question or "Which is the better deal?")[:600], detail=c.detail)
    acc.record(who, "ask", cost, made, ip)
    return dict(text=text, source="model" if made else "rules", names=[na, nb_])


@app.post("/api/commute")
def commute(c: Commute):
    """Public transport time from a block to a workplace or landmark, on a weekday morning."""
    pos = condo_latlon(c.project) if c.project and c.project in cc.projects().index else nb.block_coords().get((c.block, c.street_name))
    if not pos:
        raise HTTPException(404, "I don't have a map position for that home.")
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
