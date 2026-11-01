"""Download amenity datasets (MRT exits, hawker centres, schools) into data/raw/amenities."""
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import ROOT, get_download_url, load_config  # noqa: E402


def main():
    cfg = load_config()
    out_dir = ROOT / cfg["paths"]["raw"] / "amenities"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, dataset_id in cfg["amenities"].items():
        try:
            url = get_download_url(dataset_id)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            ext = "geojson" if r.content.lstrip()[:1] == b"{" else "csv"
            out = out_dir / f"{name}.{ext}"
            out.write_bytes(r.content)
            print(f"saved  {name} -> {out.relative_to(ROOT)} ({len(r.content) / 1e3:.0f} KB)")
        except Exception as e:
            print(f"FAILED {name} ({dataset_id}): {e}", file=sys.stderr)
        time.sleep(5)


if __name__ == "__main__":
    main()
