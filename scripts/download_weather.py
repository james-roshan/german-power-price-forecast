"""Download hourly weather for a few representative German locations from Open-Meteo.

Two sources, on purpose:
  --source forecast  : archived forecasts at a conservative 48-hour lead time.
                       Uses the Previous Runs API, not the stitched historical endpoint.
  --source archive   : observed reanalysis weather (back to 2019). Only for EDA or an
                       explicitly labelled "oracle" upper-bound experiment.

Usage:
    python scripts/download_weather.py --source forecast --start 2024-04-01 --out data/raw/weather_forecast.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

URLS = {
    "forecast": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
}
# wind-heavy north, solar-heavy south, demand centres west/east
LOCATIONS = {
    "north_sea_coast": (53.9, 8.7),
    "brandenburg": (52.4, 13.1),
    "ruhr": (51.5, 7.2),
    "bavaria": (48.4, 11.6),
    "baden_wuerttemberg": (48.7, 9.2),
}
VARIABLES = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]

log = logging.getLogger("weather")


def fetch(source: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    variables = [f"{v}_previous_day2" for v in VARIABLES] if source == "forecast" else VARIABLES
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(variables), "timezone": "UTC",
        "temperature_unit": "celsius", "wind_speed_unit": "kmh",
    }
    if source == "forecast":
        params["models"] = "ecmwf_ifs025"
    for attempt in range(3):
        try:
            r = requests.get(URLS[source], params=params, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").rename(columns=dict(zip(variables, VARIABLES)))
    if list(df.columns) != VARIABLES or df.empty or df.isna().all().any():
        raise ValueError(f"Missing weather variables or coverage for {source}: {start} to {end}")
    df.attrs["hourly_units"] = {
        v: r.json()["hourly_units"][raw] for v, raw in zip(VARIABLES, variables)
    }
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(URLS), default="forecast")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.start = args.start or ("2024-04-01" if args.source == "forecast" else "2019-01-01")
    args.out = args.out or f"data/raw/weather_{args.source}.parquet"
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start > end:
        p.error("--start must be on or before --end")
    parts = []
    for name, (lat, lon) in LOCATIONS.items():
        log.info("downloading %s weather for %s", args.source, name)
        chunks = []
        for chunk_start in pd.date_range(start, end, freq="90D"):
            chunk_end = min(chunk_start + pd.Timedelta(days=89), end)
            log.info("  %s to %s", chunk_start.date(), chunk_end.date())
            chunks.append(fetch(args.source, lat, lon, str(chunk_start.date()), str(chunk_end.date())))
            time.sleep(0.2)
        df = pd.concat(chunks).sort_index()
        parts.append(df.add_prefix(f"{name}__"))
    out_df = pd.concat(parts, axis=1)
    expected = pd.date_range(start.tz_localize("UTC"), end.tz_localize("UTC") + pd.Timedelta(hours=23), freq="h")
    if not out_df.index.equals(expected) or out_df.index.has_duplicates:
        raise ValueError("Unexpected weather timestamp coverage")
    metadata = {
        "source": args.source, "endpoint": URLS[args.source],
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "requested_start": args.start, "requested_end": args.end,
        "locations": LOCATIONS, "variables": VARIABLES,
        "units": {"temperature_2m": "degC", "wind_speed_100m": "km/h", "shortwave_radiation": "W/m2", "cloud_cover": "%"},
        "model": "ecmwf_ifs025" if args.source == "forecast" else "best_match",
        "lead_hours": 48 if args.source == "forecast" else None,
        "availability_note": "Forecast values use API previous_day2 semantics (valid time minus 48 hours). Exact publication timestamps are not supplied; this assumes publication lag is shorter than the remaining pre-auction buffer. This is a conservative fixed-lead experiment, not the latest D-1 run.",
        "radiation_interval": "preceding hour mean; retained at the API timestamp",
        "missing_by_column": {c: int(n) for c, n in out_df.isna().sum().items()},
    }
    out_df.attrs = metadata
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out)
    out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("saved %s rows x %s cols to %s", len(out_df), out_df.shape[1], out)


if __name__ == "__main__":
    main()
