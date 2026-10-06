"""Download hourly German electricity market data from SMARD (Bundesnetzagentur).

SMARD serves data in weekly JSON chunks:
  index:  https://www.smard.de/app/chart_data/{filter}/{region}/index_hour.json
  chunk:  https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_hour_{ts}.json
Each chunk has "series": [[timestamp_ms, value_or_null], ...].

Usage:
    python scripts/download_smard.py --start 2019-01-01 --out data/raw/smard.parquet
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from depower.config import SMARD_SERIES

BASE = "https://www.smard.de/app/chart_data"

# SMARD filter ids (hourly, region DE or DE-LU) live in depower.config.SMARD_SERIES.
SERIES = SMARD_SERIES

log = logging.getLogger("smard")


def _get_json(url: str, session: requests.Session, retries: int = 3) -> dict:
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"giving up on {url}")


def fetch_series(filter_id: int, region: str, start: pd.Timestamp, session: requests.Session) -> pd.Series:
    index = _get_json(f"{BASE}/{filter_id}/{region}/index_hour.json", session)["timestamps"]
    # keep chunks that can contain data on/after start (each chunk covers one week)
    start_ms = int(start.timestamp() * 1000) - 7 * 24 * 3600 * 1000
    chunks = [ts for ts in index if ts >= start_ms]
    rows = []
    for i, ts in enumerate(chunks, 1):
        data = _get_json(f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_hour_{ts}.json", session)
        rows.extend(data["series"])
        if i % 50 == 0:
            log.info("  %d/%d chunks", i, len(chunks))
        time.sleep(0.1)  # be polite to the public API
    s = pd.DataFrame(rows, columns=["ts", "value"]).dropna()
    s["ts"] = pd.to_datetime(s["ts"], unit="ms", utc=True)
    return s.drop_duplicates("ts").set_index("ts")["value"].sort_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--out", default="data/raw/smard.parquet")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = pd.Timestamp(args.start, tz="UTC")
    frames = {}
    with requests.Session() as session:
        for name, (fid, region) in SERIES.items():
            log.info("downloading %s (filter %d, %s)", name, fid, region)
            frames[name] = fetch_series(fid, region, start, session)
    df = pd.DataFrame(frames).loc[start:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    log.info("saved %s rows x %s cols to %s (%s to %s)", len(df), df.shape[1], out, df.index.min(), df.index.max())
    log.info("missing values per column:\n%s", df.isna().sum())


if __name__ == "__main__":
    main()
