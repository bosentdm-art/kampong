"""One command to refresh everything: python src/update.py

  1. download the latest HDB resale file     4. rebuild location features
  2. clean and combine                       5. retrain the production model + holdout check
  6. walk-forward out-of-sample errors (new chunks only), then the flat-specific range models
  3. geocode any new blocks (needs ONEMAP_TOKEN in .env; skipped with a warning if it fails)

A step that fails stops the run, except geocoding: without it, sales in brand-new blocks are
left out until the token is renewed.
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ("download latest data", ["download_data.py", "--force", "--only", "resale_2017_onwards"], True),
    ("clean and combine", ["clean_data.py"], True),
    ("geocode new blocks", ["geocode_blocks.py"], False),
    ("build location features", ["build_features.py"], True),
    ("retrain production model", ["train_production.py"], True),
    ("walk-forward errors for ranges", ["oos_residuals.py"], True),
    ("flat-specific range models", ["range_model.py"], True),
]


def main():
    t0 = time.time()
    for name, cmd, required in STEPS:
        print(f"\n=== {name} ===", flush=True)
        rc = subprocess.run([sys.executable, str(SRC / cmd[0]), *cmd[1:]]).returncode
        if rc != 0:
            if required:
                sys.exit(f"step failed: {name} (exit {rc})")
            print(f"WARNING: {name} failed (exit {rc}); continuing. New blocks stay unmatched until this works.")
    print(f"\nall done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
