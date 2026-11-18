"""Reading the one listing page a person pastes a link to, the way a link preview does.

Rules (agreed with the owner, 2026-11-18):
  1. Only when a person pastes a link, and only that page: no following links, no searching a site, no background fetching.
  2. Kampong says who it is in every request (the user agent below), and never pretends to be a normal browser.
  3. If the site refuses (blocked, a challenge page, a CAPTCHA, an error), Kampong stops, says why, and asks for the text
     or a screenshot. No retries and no workarounds.
  4. Only the listing's facts are read (price, address, size, floor, the description). Agent names, phone numbers and
     emails are removed. Nothing from the page is kept except that cleaned text, as for any pasted listing.
  Off switches in .env: OIKOS_LINK_READING=off turns it off everywhere; OIKOS_LINK_SKIP=site1.com,site2.com skips sites.
  A site can also opt out in its robots.txt with "User-agent: Kampong" and "Disallow: /". https://kampong.com.de/bosan explains.
"""
import html
import ipaddress
import json
import os
import re
import socket
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import requests

import listings

USER_AGENT = "Kampong/1.0 (+https://kampong.com.de/bosan; reads one page a person pasted into Kampong)"
MAX_BYTES = 2_000_000
TIMEOUT = 10
PER_DAY = 30  # links one person can have read a day
CHALLENGE = re.compile(r"captcha|cf-chl|just a moment|attention required|access denied|are you a robot|verify you are human|"
                       r"datadome|px-captcha|perimeterx|request unsuccessful|unusual traffic|enable javascript and cookies", re.I)
_count = {}


class Refused(Exception):
    """The page couldn't be read. str(e) is the plain-English reason shown to the person."""


def on():
    return os.environ.get("OIKOS_LINK_READING", "on").strip().lower() not in ("off", "0", "no", "false")


def _site(host):
    return re.sub(r"^www\.", "", host.lower())


def _skipped(host):
    skip = [s.strip().lower() for s in os.environ.get("OIKOS_LINK_SKIP", "").split(",") if s.strip()]
    h = _site(host)
    return any(h == s or h.endswith("." + s) for s in skip)


def _public(host):
    """Never fetch this computer or a private network (a pasted link must not reach inside)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise Refused("That link's website couldn't be found.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise Refused("Kampong only reads public web pages.")


def _robots_allows(url):
    """Only rules written for Kampong by name count, so a site can opt out of Kampong specifically."""
    p = urlparse(url)
    try:
        r = requests.get(f"{p.scheme}://{p.netloc}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=5)
        if r.status_code != 200:
            return True
        lines = r.text.splitlines()
        if not any(re.match(r"\s*user-agent\s*:\s*kampong\b", ln, re.I) for ln in lines):
            return True
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(lines)
        return rp.can_fetch("Kampong", url)
    except requests.RequestException:
        return True


def fetch(url, who=None):
    """The page's HTML, or Refused with the reason to show."""
    if not on():
        raise Refused("Reading links is switched off.")
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https") or not p.hostname:
        raise Refused("That doesn't look like a web link.")
    if _skipped(p.hostname):
        raise Refused(f"Kampong doesn't read pages from {_site(p.hostname)}.")
    if who:
        day = time.strftime("%Y-%m-%d")
        n = _count.get((who, day), 0)
        if n >= PER_DAY:
            raise Refused("You've reached today's limit for reading links.")
        _count[(who, day)] = n + 1
    if not _robots_allows(url):
        raise Refused(f"{_site(p.hostname)} has asked Kampong not to read its pages.")
    target = url
    for _ in range(3):  # follow at most a few redirects, checking each address is public
        _public(urlparse(target).hostname)
        try:
            r = requests.get(target, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                             timeout=TIMEOUT, allow_redirects=False, stream=True)
        except requests.RequestException:
            raise Refused(f"{_site(p.hostname)} didn't answer.")
        if r.is_redirect and r.headers.get("location"):
            target = urljoin(target, r.headers["location"])
            continue
        break
    else:
        raise Refused(f"{_site(p.hostname)} kept redirecting.")
    if r.status_code in (401, 403, 429, 451, 503) or r.status_code >= 400:
        raise Refused(f"{_site(p.hostname)} didn't let Kampong read this page.")
    if "html" not in r.headers.get("content-type", "html"):
        raise Refused("That link isn't a web page.")
    body = r.raw.read(MAX_BYTES + 1, decode_content=True)[:MAX_BYTES].decode(r.encoding or "utf-8", errors="replace")
    if CHALLENGE.search(body[:20000]) and len(re.sub(r"<[^>]+>", " ", body)) < 20000:
        raise Refused(f"{_site(p.hostname)} didn't let Kampong read this page.")
    return body, _site(urlparse(target).hostname)


def _meta(body, name):
    m = re.search(r'<meta[^>]+(?:property|name)=["\']' + re.escape(name) + r'["\'][^>]*content=["\']([^"\']*)', body, re.I) \
        or re.search(r'<meta[^>]+content=["\']([^"\']*)["\'][^>]*(?:property|name)=["\']' + re.escape(name), body, re.I)
    return html.unescape(m.group(1)).strip() if m else ""


def _jsonld(body):
    """Facts from schema.org data that listing pages often include: price, address, size."""
    facts = []
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', body, re.I | re.S):
        try:
            data = json.loads(block.strip())
        except ValueError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            d = stack.pop(0)
            if isinstance(d, list):
                stack.extend(d)
                continue
            if not isinstance(d, dict):
                continue
            kind = " ".join(d.get("@type", [])) if isinstance(d.get("@type"), list) else str(d.get("@type", ""))
            if re.search(r"Person|Organization|Agent|Brand|Corporation|ContactPoint", kind):
                continue  # agents and agencies: rule 4, people's names and contacts are never kept
            stack.extend(v for v in d.values() if isinstance(v, (dict, list)))
            if kind == "Question":  # FAQ blocks: keep the answer, not the question ("What is the address...?")
                continue
            for k in ("description", "streetAddress", "addressLocality", "postalCode", "price", "floorSize", "numberOfRooms", "value", "unitText", "text"):
                v = d.get(k)
                if isinstance(v, (str, int, float)) and str(v).strip():
                    facts.append(f"{v}" if k == "text" else f"{k}: {v}")
    return list(dict.fromkeys(facts))[:40]


def listing_text(body):
    """The listing's facts as plain text: title, description, structured data and the start of the visible text."""
    title = _meta(body, "og:title") or (re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S) or [None, ""])[1]
    desc = _meta(body, "og:description") or _meta(body, "description")
    visible = re.sub(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", body, flags=re.I | re.S)
    visible = html.unescape(re.sub(r"<[^>]+>", " ", visible))
    visible = re.sub(r"\s+", " ", visible).strip()
    facts = ". ".join(_jsonld(body))
    # the visible text is a fallback only: on listing pages it mixes in agents, ads and other listings
    parts = [html.unescape(str(title)).strip(), desc, facts] + ([visible[:3000]] if len(desc) + len(facts) < 300 else [])
    text = " \n".join(p for p in parts if p)
    text = re.sub(r"\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s*\([^)]{0,40}\))?", "by [agent]", text)  # "listed by Jane Tan (Jane)"
    return listings.scrub(text)[:9000]
