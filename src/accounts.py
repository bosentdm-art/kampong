"""Accounts, sessions, saved flats and usage counts for Oikos, in one SQLite file (data/oikos.db, not in Git).

Passwords are stored only as salted scrypt hashes. A session is a random token kept in an HttpOnly cookie; the
database holds only its SHA-256. Visitors without an account get a random anonymous id so their usage can be counted.

Limits are off unless set:  OIKOS_LIMIT_ANON / OIKOS_LIMIT_USER = AI answers per day (Singapore time).
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
        """)


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
        return c.execute("select u.id, u.email from sessions s join users u on u.id = s.user_id where s.token_hash = ? and s.expires > ?",
                         (_th(token), time.time())).fetchone()


def user_for_id(user_id):
    with db() as c:
        return c.execute("select id, email from users where id = ?", (user_id,)).fetchone()


def end_session(token):
    if token:
        with db() as c:
            c.execute("delete from sessions where token_hash = ?", (_th(token),))


# ---- usage and limits
def _limit(name):
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else None


def limit_for(who):
    return _limit("OIKOS_LIMIT_USER" if who.startswith("u:") else "OIKOS_LIMIT_ANON")


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
