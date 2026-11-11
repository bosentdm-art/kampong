"""Accounts, sessions, saved flats and usage counts for Oikos, in one SQLite file (data/oikos.db, not in Git).

Passwords are stored only as salted scrypt hashes. A session is a random token kept in an HttpOnly cookie; the
database holds only its SHA-256. Visitors without an account get a random anonymous id so their usage can be counted.

Plans set how many AI answers a person gets per day (Singapore time): see PLANS. OIKOS_LIMIT_ANON and
OIKOS_LIMIT_USER override the visitor and free numbers. There is no payment yet: "I want this plan" is recorded in
the interest table, and a plan is switched by hand:  python src/accounts.py set-plan someone@example.com plus
No email verification or password reset yet: both need a mail service.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time

from geocode_blocks import ROOT

DB = ROOT / "data/oikos.db"
SESSION_DAYS = 30
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s.]{2,}$")
# name, price in S$ a month, AI answers a day. "visitor" is anyone not signed in.
PLANS = {"visitor": dict(name="Visitor", price=0, answers=3, alerts=0, blurb="Try it without an account"),
         "free": dict(name="Free", price=0, answers=10, alerts=2, blurb="For checking a flat or two"),
         "plus": dict(name="Plus", price=9, answers=100, alerts=10, blurb="For an active search or sale"),
         "pro": dict(name="Pro", price=39, answers=500, alerts=50, blurb="For property agents")}
_fails = {}  # email -> times of recent failed sign-ins (in memory; resets when the server restarts)


def db():
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    DB.parent.mkdir(exist_ok=True)
    with db() as c:
        c.executescript("""
            create table if not exists users (id integer primary key, email text unique not null, pw text not null, created real not null);
            create table if not exists sessions (token_hash text primary key, user_id integer not null, expires real not null);
            create table if not exists usage (id integer primary key, who text not null, kind text not null, at real not null, cost_cents real not null, answered integer not null);
            create index if not exists usage_who_at on usage (who, at);
            create table if not exists flats (user_id integer primary key, body text not null, updated real not null);
            create table if not exists interest (id integer primary key, user_id integer not null, plan text not null, at real not null);
            create table if not exists advice (key text primary key, body text not null, at real not null);
            create table if not exists shares (id text primary key, body text not null, created real not null, views integer not null default 0);
            create table if not exists alerts (id integer primary key, user_id integer not null, kind text not null, spec text not null, baseline text not null, seen text not null, created real not null);
        """)
        if "plan" not in [r["name"] for r in c.execute("pragma table_info(users)")]:
            c.execute("alter table users add column plan text not null default 'free'")


def _hash(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _matches(password, stored):
    _, salt, _ = stored.split("$")
    return hmac.compare_digest(_hash(password, bytes.fromhex(salt)), stored)


_DUMMY = _hash("not a real password")  # checked when the email is unknown, so both cases take the same time


def signup(email, password):
    email = email.strip().lower()
    if not EMAIL.match(email):
        raise ValueError("That email address doesn't look right.")
    if not 8 <= len(password) <= 200:
        raise ValueError("Use a password of at least 8 characters.")
    try:
        with db() as c:
            return c.execute("insert into users (email, pw, created) values (?, ?, ?)", (email, _hash(password), time.time())).lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("There is already an account with that email. Sign in instead.")


def login(email, password):
    """The user's id, or None. Five wrong passwords in 15 minutes locks that email for the rest of the 15 minutes."""
    email, now = email.strip().lower(), time.time()
    recent = [t for t in _fails.get(email, []) if now - t < 900]
    if len(recent) >= 5:
        raise ValueError("Too many attempts. Try again in 15 minutes.")
    with db() as c:
        u = c.execute("select id, pw from users where email = ?", (email,)).fetchone()
    if _matches(password[:200], u["pw"] if u else _DUMMY) and u:
        _fails.pop(email, None)
        return u["id"]
    _fails[email] = recent + [now]
    return None


def _th(token):
    return hashlib.sha256(token.encode()).hexdigest()


def new_session(user_id):
    token = secrets.token_urlsafe(32)
    with db() as c:
        c.execute("delete from sessions where expires < ?", (time.time(),))
        c.execute("insert into sessions values (?, ?, ?)", (_th(token), user_id, time.time() + SESSION_DAYS * 86400))
    return token


def user_for(token):
    if not token:
        return None
    with db() as c:
        return c.execute("select u.id, u.email, u.plan from sessions s join users u on u.id = s.user_id where s.token_hash = ? and s.expires > ?",
                         (_th(token), time.time())).fetchone()


def user_for_id(user_id):
    with db() as c:
        return c.execute("select id, email, plan from users where id = ?", (user_id,)).fetchone()


def set_plan(email, plan):
    if plan not in PLANS or plan == "visitor":
        raise ValueError("plan must be one of: free, plus, pro")
    with db() as c:
        if not c.execute("update users set plan = ? where email = ?", (plan, email.strip().lower())).rowcount:
            raise ValueError("no account with that email")


def want_plan(user_id, plan):
    """Record that someone asked for a paid plan (there is no payment yet), so demand can be counted."""
    if plan not in ("plus", "pro"):
        raise ValueError("unknown plan")
    with db() as c:
        c.execute("insert into interest (user_id, plan, at) values (?, ?, ?)", (user_id, plan, time.time()))


def end_session(token):
    if token:
        with db() as c:
            c.execute("delete from sessions where token_hash = ?", (_th(token),))


# ---- usage and limits
def _limit(name):
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else None


def plan_of(who):
    if not who.startswith("u:"):
        return "visitor"
    u = user_for_id(int(who[2:]))
    return u["plan"] if u and u["plan"] in PLANS else "free"


def limit_for(who):
    plan = plan_of(who)
    override = _limit("OIKOS_LIMIT_ANON") if plan == "visitor" else _limit("OIKOS_LIMIT_USER") if plan == "free" else None
    return override if override is not None else PLANS[plan]["answers"]


def _sg_midnight():
    now = time.time()
    return (now + 8 * 3600) // 86400 * 86400 - 8 * 3600


def record(who, kind, cost_cents, answered):
    with db() as c:
        c.execute("insert into usage (who, kind, at, cost_cents, answered) values (?, ?, ?, ?, ?)", (who, kind, time.time(), float(cost_cents or 0), int(bool(answered))))


def used_today(who):
    """AI answers this person has had since midnight, Singapore time."""
    with db() as c:
        return c.execute("select count(*) from usage where who = ? and at >= ? and answered = 1 and kind = 'ask'", (who, _sg_midnight())).fetchone()[0]


def over_limit(who):
    cap = limit_for(who)
    return cap is not None and used_today(who) >= cap


# ---- saved flats
# ---- shared links. A link holds the flat, the price, the role and the advice Oikos gave (taken from the server's own
# copy, never from the browser, so a link cannot carry text we did not write).
def put_advice(key, text):
    with db() as c:
        c.execute("insert into advice values (?, ?, ?) on conflict(key) do update set body = excluded.body, at = excluded.at", (key, text, time.time()))


def get_advice(key):
    with db() as c:
        r = c.execute("select body from advice where key = ?", (key,)).fetchone()
    return r["body"] if r else None


def new_share(body):
    sid = secrets.token_urlsafe(6)
    with db() as c:
        c.execute("insert into shares (id, body, created) values (?, ?, ?)", (sid, json.dumps(body), time.time()))
    return sid


def get_share(sid):
    with db() as c:
        r = c.execute("select body, created from shares where id = ?", (sid,)).fetchone()
        if not r:
            return None
        c.execute("update shares set views = views + 1 where id = ?", (sid,))
    return dict(json.loads(r["body"]), created=time.strftime("%Y-%m-%d", time.gmtime(r["created"] + 8 * 3600)))


def recent_count(who, kind, seconds=86400):
    with db() as c:
        return c.execute("select count(*) from usage where who = ? and kind = ? and at >= ?", (who, kind, time.time() - seconds)).fetchone()[0]


# ---- alerts. An alert remembers the newest sale month when it was set (baseline) and which matching sales the person
# has already seen; anything else from that month onward is new. Matching itself is done in api.py, against the sales data.
def alerts_for(user_id):
    with db() as c:
        rows = c.execute("select * from alerts where user_id = ? order by id desc", (user_id,)).fetchall()
    return [dict(id=r["id"], kind=r["kind"], spec=json.loads(r["spec"]), baseline=r["baseline"], seen=set(json.loads(r["seen"])),
                 created=time.strftime("%Y-%m-%d", time.gmtime(r["created"] + 8 * 3600))) for r in rows]


def add_alert(user_id, kind, spec, baseline, seen):
    cap = PLANS[plan_of(f"u:{user_id}")]["alerts"]
    with db() as c:
        if c.execute("select count(*) from alerts where user_id = ?", (user_id,)).fetchone()[0] >= cap:
            raise ValueError(f"Your plan allows {cap} alerts. Remove one, or see Plans for more.")
        if c.execute("select 1 from alerts where user_id = ? and kind = ? and spec = ?", (user_id, kind, json.dumps(spec, sort_keys=True))).fetchone():
            raise ValueError("You already have this alert.")
        return c.execute("insert into alerts (user_id, kind, spec, baseline, seen, created) values (?, ?, ?, ?, ?, ?)",
                         (user_id, kind, json.dumps(spec, sort_keys=True), baseline, json.dumps(sorted(seen)), time.time())).lastrowid


def mark_seen(user_id, alert_id, keys):
    with db() as c:
        r = c.execute("select seen from alerts where id = ? and user_id = ?", (alert_id, user_id)).fetchone()
        if r:
            c.execute("update alerts set seen = ? where id = ?", (json.dumps(sorted(set(json.loads(r["seen"])) | set(keys))), alert_id))


def drop_alert(user_id, alert_id):
    with db() as c:
        c.execute("delete from alerts where id = ? and user_id = ?", (alert_id, user_id))


def get_flats(user_id):
    with db() as c:
        r = c.execute("select body from flats where user_id = ?", (user_id,)).fetchone()
    return json.loads(r["body"]) if r else []


def put_flats(user_id, flats):
    body = json.dumps(flats)
    if len(body) > 2_000_000:
        raise ValueError("Too many saved flats.")
    with db() as c:
        c.execute("insert into flats values (?, ?, ?) on conflict(user_id) do update set body = excluded.body, updated = excluded.updated", (user_id, body, time.time()))


init()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "set-plan":
        set_plan(sys.argv[2], sys.argv[3])
        print("done")
    elif len(sys.argv) == 2 and sys.argv[1] == "interest":
        with db() as c:
            for r in c.execute("select plan, count(distinct user_id) n from interest group by plan"):
                print(r["plan"], r["n"], "people")
    else:
        print("usage: python src/accounts.py set-plan <email> <free|plus|pro>   |   python src/accounts.py interest")
