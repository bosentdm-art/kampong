"""Download HDB resale flat price datasets from data.gov.sg into data/raw.

Rerunnable: skips files that already exist unless --force is passed.
"""
import argparse
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://api-open.data.gov.sg/v1/public/api/datasets/{id}/{action}"


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_with_backoff(url, attempts=6):
    """GET, waiting and retrying when data.gov.sg answers 429 (rate limited)."""
    for i in range(attempts):
        r = requests.get(url, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = 10 * 2**i
        print(f"rate limited, waiting {wait}s")
        time.sleep(wait)
    r.raise_for_status()
    return r


def get_download_url(dataset_id, retries=12):
    """Start a download and poll until data.gov.sg returns the file URL."""
    r = get_with_backoff(API.format(id=dataset_id, action="initiate-download"))
    body = r.json()
    if body.get("data", {}).get("url"):
        return body["data"]["url"]
    for _ in range(retries):
        time.sleep(5)  # data.gov.sg prepares the file; this can take a few minutes
        r = get_with_backoff(API.format(id=dataset_id, action="poll-download"))
        url = r.json().get("data", {}).get("url")
        if url:
            return url
    raise RuntimeError(f"No download URL for {dataset_id}")


PAGED = "https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit=10000&offset={off}&sort=_id%20asc"


def fetch_paged(dataset_id, out):
    """Fallback when the file link stalls: page through the query API and rebuild the CSV.

    Writes to a temporary file first and refuses to replace an existing file with a much smaller one."""
    import pandas as pd

    rows, off = [], 0
    while True:
        result = get_with_backoff(PAGED.format(id=dataset_id, off=off)).json()["result"]
        rows += result["records"]
        off += len(result["records"])
        if not result["records"] or off >= result["total"]:
            break
        time.sleep(2)
    df = pd.DataFrame(rows).drop(columns=["_id"], errors="ignore")
    if len(df) < result["total"]:
        raise RuntimeError(f"paged download incomplete: {len(df)} of {result['total']} rows")
    if out.exists() and len(df) < 0.95 * len(pd.read_csv(out)):
        raise RuntimeError("paged download is much smaller than the existing file; keeping the old one")
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"saved  {out.name} via paged query ({len(df)} rows)")


def download(name, dataset_id, out_dir, force):
    out = out_dir / f"{name}.csv"
    if out.exists() and not force:
        print(f"skip   {name} (already downloaded)")
        return out
    try:
        url = get_download_url(dataset_id)
    except Exception as e:
        print(f"download link failed ({e}); falling back to the paged query API")
        fetch_paged(dataset_id, out)
        return out
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"saved  {name} -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these dataset names from config.yaml")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = ROOT / cfg["paths"]["raw"]
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, dataset_id in cfg["datasets"].items():
        if args.only and name not in args.only:
            continue
        try:
            download(name, dataset_id, out_dir, args.force)
            time.sleep(5)  # be gentle with the API
        except Exception as e:  # keep going so one bad ID does not block the rest
            print(f"FAILED {name} ({dataset_id}): {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        sys.exit(f"{len(failed)} download(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
