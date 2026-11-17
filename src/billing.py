"""Paid plans through Stripe: Checkout to subscribe, the customer portal to change plan, card or cancel, and the
subscription's state copied onto the account so limits change straight away.

On when STRIPE_SECRET_KEY is in .env (a test key, sk_test_..., while building). STRIPE_WEBHOOK_SECRET (whsec_...) is
optional: with it, Stripe's messages about renewals, failed payments and cancellations arrive at /api/stripe/webhook.
Without it, Oikos still stays correct: it confirms each checkout itself when the person comes back, and re-checks a paid
account's subscription with Stripe at most once an hour.

Plans are found by the price's lookup key (oikos_plus_monthly, oikos_pro_monthly), so no Stripe ids live in the code.
Card details never touch Oikos: people pay on Stripe's own page. Talks to Stripe's REST API with `requests` directly."""
import hashlib
import hmac
import json
import os
import time

import requests

import accounts as acc

API = "https://api.stripe.com/v1"
LOOKUP = {"plus": "oikos_plus_monthly", "pro": "oikos_pro_monthly"}
PAID = {"active", "trialing", "past_due"}  # past_due: Stripe is retrying the card, so keep the plan until it gives up
RECHECK = 3600
_price_ids, _portal = {}, {}


def on():
    return os.environ.get("STRIPE_SECRET_KEY", "").strip().startswith(("sk_", "rk_"))


def test_mode():
    return os.environ.get("STRIPE_SECRET_KEY", "").strip().startswith(("sk_test_", "rk_test_"))


def _flat(d, prefix=""):
    """Stripe's form encoding: {"line_items": [{"price": "p"}]} -> line_items[0][price]=p."""
    out = []
    for k, v in d.items():
        key = f"{prefix}[{k}]" if prefix else str(k)
        if isinstance(v, dict):
            out += _flat(v, key)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                out += _flat(x, f"{key}[{i}]") if isinstance(x, dict) else [(f"{key}[{i}]", x)]
        elif isinstance(v, bool):
            out.append((key, "true" if v else "false"))
        elif v is not None:
            out.append((key, v))
    return out


def stripe(method, path, params=None):
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    kw = dict(params=_flat(params or {})) if method == "GET" else dict(data=_flat(params or {}))
    r = requests.request(method, API + path, auth=(key, ""), timeout=30, headers={"Stripe-Version": "2025-03-31.basil"}, **kw)
    data = r.json()
    if r.status_code >= 400:
        raise RuntimeError((data.get("error") or {}).get("message") or f"Stripe HTTP {r.status_code}")
    return data


def price_id(plan):
    if plan not in LOOKUP:
        raise ValueError("unknown plan")
    if plan not in _price_ids:
        found = stripe("GET", "/prices", {"lookup_keys": [LOOKUP[plan]], "active": True, "limit": 1})["data"]
        if not found:
            raise RuntimeError(f"No Stripe price with lookup key {LOOKUP[plan]}")
        _price_ids[plan] = found[0]["id"]
    return _price_ids[plan]


def plan_for_price(pid):
    for plan in LOOKUP:
        try:
            if price_id(plan) == pid:
                return plan
        except RuntimeError:
            pass
    return None


def customer_for(user):
    """The person's Stripe customer, made the first time they pay. Only the email and Oikos account id are sent."""
    b = acc.billing_of(user["id"])
    if b and b["stripe_customer"]:
        return b["stripe_customer"]
    c = stripe("POST", "/customers", {"email": user["email"], "metadata": {"oikos_user": str(user["id"])}})
    acc.set_billing(user["id"], stripe_customer=c["id"])
    return c["id"]


def checkout_url(user, plan, base):
    """A Stripe Checkout page for this plan. Someone already paying goes to the portal instead, to switch plans there."""
    b = acc.billing_of(user["id"])
    if b and b["stripe_sub"] and b["sub_status"] in PAID:
        return portal_url(user, base)
    s = stripe("POST", "/checkout/sessions", {
        "mode": "subscription", "customer": customer_for(user), "client_reference_id": str(user["id"]),
        "line_items": [{"price": price_id(plan), "quantity": 1}], "allow_promotion_codes": True,
        "subscription_data": {"metadata": {"oikos_user": str(user["id"]), "plan": plan}},
        "success_url": f"{base}/?checkout=done&session_id={{CHECKOUT_SESSION_ID}}", "cancel_url": f"{base}/?checkout=cancelled"})
    return s["url"]


def _portal_config():
    """The portal's settings: cancel, update the card, see invoices, switch between Plus and Pro. Made once if the account has none."""
    if "id" not in _portal:
        have = stripe("GET", "/billing_portal/configurations", {"is_default": True, "limit": 1})["data"]
        if have:
            _portal["id"] = have[0]["id"]
        else:
            prods = {p: stripe("GET", f"/prices/{price_id(p)}")["product"] for p in LOOKUP}
            c = stripe("POST", "/billing_portal/configurations", {
                "business_profile": {"headline": "Oikos: manage your plan"},
                "features": {"invoice_history": {"enabled": True}, "payment_method_update": {"enabled": True},
                             "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
                             "subscription_update": {"enabled": True, "default_allowed_updates": ["price"], "proration_behavior": "create_prorations",
                                                     "products": [{"product": prods[p], "prices": [price_id(p)]} for p in LOOKUP]}}})
            _portal["id"] = c["id"]
    return _portal["id"]


def portal_url(user, base):
    b = acc.billing_of(user["id"])
    if not b or not b["stripe_customer"]:
        raise ValueError("There's no billing to manage yet.")
    return stripe("POST", "/billing_portal/sessions", {"customer": b["stripe_customer"], "return_url": f"{base}/?billing=back", "configuration": _portal_config()})["url"]


def apply_subscription(sub, user_id=None):
    """Copy a Stripe subscription onto the account: plan, status and when the paid period ends."""
    if user_id is None:
        user_id = int((sub.get("metadata") or {}).get("oikos_user") or 0) or acc.user_for_customer(sub.get("customer"))
    if not user_id:
        return None
    items = (sub.get("items") or {}).get("data") or []
    plan = plan_for_price(items[0]["price"]["id"]) if items else None
    status = sub.get("status")
    ends = items[0].get("current_period_end") if items else sub.get("current_period_end")
    acc.set_billing(user_id, stripe_customer=sub.get("customer"), stripe_sub=sub.get("id"), sub_status=status, paid_until=ends, checked=time.time(),
                    cancel_at_end=1 if sub.get("cancel_at_period_end") else 0)
    acc.set_plan_for(user_id, plan if plan and status in PAID else "free")
    return plan if status in PAID else "free"


def confirm_checkout(user, session_id):
    """Called when the person comes back from Checkout: Oikos asks Stripe itself rather than trusting the page."""
    s = stripe("GET", f"/checkout/sessions/{session_id}", {"expand": ["subscription"]})
    if str(s.get("client_reference_id")) != str(user["id"]):
        raise ValueError("That checkout belongs to another account.")
    if s.get("status") != "complete" or not isinstance(s.get("subscription"), dict):
        return None
    return apply_subscription(s["subscription"], user["id"])


def refresh(user_id):
    """Re-read a paying account's subscription from Stripe, at most once an hour (renewals, failed cards, cancellations)."""
    b = acc.billing_of(user_id)
    if not on() or not b or not b["stripe_sub"] or time.time() - (b["checked"] or 0) < RECHECK:
        return
    try:
        apply_subscription(stripe("GET", f"/subscriptions/{b['stripe_sub']}"), user_id)
    except Exception:
        acc.set_billing(user_id, checked=time.time())  # Stripe unreachable: keep what we have and try again later


def verify_webhook(payload, header, secret, tolerance=300):
    """Stripe's signature check: HMAC-SHA256 of "timestamp.payload" with the endpoint secret, within five minutes."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
        sigs = [v for k, v in (p.split("=", 1) for p in header.split(",")) if k == "v1"]
    except Exception:
        return False
    want = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return abs(time.time() - ts) <= tolerance and any(hmac.compare_digest(want, s) for s in sigs)


def handle_event(event):
    t, obj = event.get("type"), (event.get("data") or {}).get("object") or {}
    if t in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        apply_subscription(obj)
    elif t == "checkout.session.completed" and obj.get("subscription"):
        apply_subscription(stripe("GET", f"/subscriptions/{obj['subscription']}"), int(obj.get("client_reference_id") or 0) or None)
    elif t in ("invoice.paid", "invoice.payment_failed"):
        sid = obj.get("subscription") or (((obj.get("parent") or {}).get("subscription_details") or {}).get("subscription"))
        if sid:
            apply_subscription(stripe("GET", f"/subscriptions/{sid}"))
