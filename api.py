"""Oikos API: the trained models behind a small web service, plus the front end in web/.

Run:  python -m uvicorn api:app --port 8700     then open http://localhost:8700
Needs the data and models built first (python src/update.py).

  GET  /api/options   lists for the flat picker (town > street > block > flat type) and sensible defaults
  POST /api/parse     pasted listing text -> flat details
  POST /api/understand  a chat message -> the flat it describes, a change to the flat on screen, or an AI reply
  POST /api/ask       the AI's advice on a flat, every figure checked against the facts
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

app = FastAPI(title="Oikos")


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
    return dict(user=dict(email=user["email"]) if user else None, used_today=acc.used_today(who), limit=acc.limit_for(who))


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
    base = dict(found=False, role=p["role"], age=p["age"], price=None, assumed=[], changed=[], reply=None)
    asks = bool(QUESTION.search(m.text[:600]))
    same = cur and p["block"] == cur["block"] and p["street_name"] == cur["street_name"]

    if p["block"] and not same:  # a new flat
        return new_flat(p, base)

    if cur and not asks:  # a correction to the flat on screen: "it's on the 11th floor", "asking is 400k", "it's 68 sqm"
        flat, changed = dict(cur), []
        if p["floor"]:
            flat["storey_range"] = band(p["floor"])
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
    who = visitor(request, response)[0] if request is not None else None
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
        deals=deals_nearby(f.town), trend=trend(f.town, f.flat_type), meta=meta()))



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
        raise HTTPException(429, f"You've used today's {cap} AI answers. " + ("They reset at midnight." if user else "Create a free account for more, or come back tomorrow."))
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
    return clean(out)


@app.get("/api/deals")
def deals(town: str):
    """Recent sales in a town that closed 8% or more under our estimate at the time."""
    return clean(deals_nearby(town))


@app.get("/api/trend")
def trend_api(town: str, flat_type: str):
    return clean(trend(town, flat_type))


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
