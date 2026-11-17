"""Back up Kampong's database (data/oikos.db: accounts, chats, saved homes, alerts, billing links) to Cloudflare R2.

    python scripts/backup.py             take a backup now and keep the last 30 days of daily backups
    python scripts/backup.py --list      show the backups in R2
    python scripts/backup.py --restore   download the newest backup to data/oikos.db.restored (never overwrites the live file)

The copy is taken with SQLite's own backup API, so it is consistent even while Kampong is running. It is gzipped and
uploaded to backups/oikos-YYYY-MM-DD.db.gz in the bucket from R2_BUCKET, using R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
and R2_ENDPOINT from .env. Run nightly by the Windows task "Kampong backup" (set up by scripts/kampong.ps1 install-backup)."""
import gzip
import io
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "oikos.db"
KEEP_DAYS = 30
PREFIX = "backups/"


def env():
    vals = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(R2_[A-Z_]+)\s*=\s*(.*)", line)
        if m:
            vals[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    missing = [k for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET") if not vals.get(k)]
    if missing:
        sys.exit("Missing in .env: " + ", ".join(missing))
    return vals


def client(v):
    import boto3
    return boto3.client("s3", endpoint_url=v["R2_ENDPOINT"].rstrip("/"), aws_access_key_id=v["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=v["R2_SECRET_ACCESS_KEY"], region_name="auto")


def backups(s3, bucket):
    out, token = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=PREFIX, **({"ContinuationToken": token} if token else {}))
        r = s3.list_objects_v2(**kw)
        out += [o for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return sorted(out, key=lambda o: o["Key"])
        token = r["NextContinuationToken"]


def snapshot():
    """A consistent copy of the live database, as gzipped bytes."""
    tmp = ROOT / "data" / f".backup-{os.getpid()}.db"
    try:
        src, dst = sqlite3.connect(DB), sqlite3.connect(tmp)
        with dst:
            src.backup(dst)
        src.close(); dst.close()
        chk = sqlite3.connect(tmp)
        ok = chk.execute("pragma integrity_check").fetchone()[0]
        chk.close()
        if ok != "ok":
            sys.exit(f"The copy failed its integrity check ({ok}); nothing uploaded.")
        return gzip.compress(tmp.read_bytes(), compresslevel=9)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    v = env()
    s3, bucket = client(v), v["R2_BUCKET"]
    if "--list" in sys.argv:
        for o in backups(s3, bucket):
            print(f"{o['Key']}  {o['Size'] / 1024:,.0f} KB  {o['LastModified']:%Y-%m-%d %H:%M}")
        return
    if "--restore" in sys.argv:
        found = backups(s3, bucket)
        if not found:
            sys.exit("No backups in R2 yet.")
        key = found[-1]["Key"]
        data = gzip.decompress(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        out = DB.with_name("oikos.db.restored")
        out.write_bytes(data)
        print(f"Downloaded {key} to {out}. Stop Kampong, then replace data/oikos.db with it to restore.")
        return
    if not DB.exists():
        sys.exit(f"No database at {DB}.")
    body = snapshot()
    key = f"{PREFIX}oikos-{time.strftime('%Y-%m-%d')}.db.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/gzip")
    print(f"Backed up {DB.name} ({DB.stat().st_size / 1024:,.0f} KB, {len(body) / 1024:,.0f} KB compressed) to R2: {key}")
    found = backups(s3, bucket)
    old = found[:-KEEP_DAYS] if len(found) > KEEP_DAYS else []
    for o in old:
        s3.delete_object(Bucket=bucket, Key=o["Key"])
    print(f"{len(found) - len(old)} backups kept" + (f", {len(old)} older ones removed." if old else "."))


if __name__ == "__main__":
    main()
