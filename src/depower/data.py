"""Loading raw inputs, provenance hashing and the data-quality audit."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TARGET, TZ

MARKET_DEFINITIONS = {
    TARGET: ("EUR/MWh", "Wholesale day-ahead price; the prediction target."),
    "load_mwh": ("MWh", "Actual grid load; system consumption with provider accounting boundaries."),
    "residual_load_mwh": ("MWh", "Grid load minus solar, onshore wind and offshore wind."),
    "solar_mwh": ("MWh", "Electricity produced by photovoltaic installations during the interval."),
    "wind_onshore_mwh": ("MWh", "Electricity produced by wind turbines on land."),
    "wind_offshore_mwh": ("MWh", "Electricity produced by offshore wind turbines."),
}
WEATHER_DEFINITIONS = {
    "temperature_2m": ("degC", "Air temperature 2 m above ground; may affect heating/cooling demand."),
    "wind_speed_100m": ("km/h", "Wind 100 m above ground; proxy for wind-turbine conditions."),
    "shortwave_radiation": ("W/m2", "Incoming horizontal solar radiation; proxy for available sunlight."),
    "cloud_cover": ("%", "Sky cloud fraction; 100 means fully cloud-covered, not zero radiation."),
}


@dataclass
class Datasets:
    """All raw inputs plus provenance."""

    market: pd.DataFrame
    archive: pd.DataFrame
    forecast: pd.DataFrame
    forecast_meta: dict
    manifest: dict

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return {"market": self.market, "weather_archive": self.archive,
                "weather_forecast": self.forecast}


def market_path(raw: Path) -> Path:
    """Prefer the newer `smard_sample.parquet` snapshot; fall back to `smard.parquet`.

    Overlapping snapshots are never concatenated.
    """
    sample = raw / "smard_sample.parquet"
    return sample if sample.exists() else raw / "smard.parquet"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_datasets(root: Path) -> Datasets:
    raw = Path(root) / "data/raw"
    paths = {"market": market_path(raw), "weather_archive": raw / "weather_archive.parquet",
             "weather_forecast": raw / "weather_forecast.parquet"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files (see README for download steps): {missing}")
    frames = {k: pd.read_parquet(p) for k, p in paths.items()}
    meta = json.loads((raw / "weather_forecast.metadata.json").read_text(encoding="utf-8"))
    if meta.get("source") != "forecast" or meta.get("lead_hours") != 48:
        raise ValueError("weather_forecast must be the 48-hour-lead forecast archive.")
    manifest = {k: {"file": str(p.relative_to(root)), "sha256": file_sha256(p)} for k, p in paths.items()}
    return Datasets(frames["market"], frames["weather_archive"], frames["weather_forecast"], meta, manifest)


# ---------------------------------------------------------------- dictionaries
def market_dictionary(market: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    row = market.loc[at]
    return pd.DataFrame([{"column": c, "unit": u, "meaning": m, "example_value": row[c]}
                         for c, (u, m) in MARKET_DEFINITIONS.items()]).set_index("column")


def weather_dictionary(archive: pd.DataFrame, forecast: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for column in archive:
        location, variable = column.split("__")
        unit, meaning = WEATHER_DEFINITIONS[variable]
        rows.append({"column": column, "location": location, "unit": unit, "meaning": meaning,
                     "archive_example": archive.loc[at, column],
                     "forecast_example_48h": forecast.loc[at, column]})
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------- quality audit
def quality_audit(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-dataset structural checks and per-column missing-value summary."""
    quality_rows, missing_rows = [], []
    for name, df in frames.items():
        if not isinstance(df.index, pd.DatetimeIndex) or str(df.index.tz) != "UTC":
            raise ValueError(f"{name}: index must be a UTC DatetimeIndex")
        expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
        quality_rows.append({
            "dataset": name, "rows": len(df), "columns": df.shape[1],
            "start_utc": df.index.min(), "end_utc": df.index.max(),
            "duplicates": int(df.index.duplicated().sum()), "sorted": df.index.is_monotonic_increasing,
            "missing_timestamps": len(expected.difference(df.index)),
            "non_finite_non_null": int((~np.isfinite(df) & df.notna()).sum().sum()),
        })
        for column in df:
            mask = df[column].isna()
            missing_rows.append({"dataset": name, "column": column, "missing": int(mask.sum()),
                                 "missing_pct": float(mask.mean() * 100),
                                 "first_missing": df.index[mask].min(), "last_missing": df.index[mask].max()})
    return pd.DataFrame(quality_rows).set_index("dataset"), pd.DataFrame(missing_rows)


def assert_quality(quality: pd.DataFrame) -> None:
    if not ((quality.duplicates == 0).all() and quality["sorted"].all()
            and (quality.missing_timestamps == 0).all() and (quality.non_finite_non_null == 0).all()):
        raise AssertionError(f"Data-quality audit failed:\n{quality}")


def delivery_day_coverage(market: pd.DataFrame) -> pd.DataFrame:
    """Observed vs expected hours per Berlin delivery day (23/24/25 around DST)."""
    local_index = market.index.tz_convert(TZ)
    counts = market.groupby(local_index.normalize()).size().rename("observed_hours").to_frame()
    counts["expected_hours"] = [len(pd.date_range(d, d + pd.DateOffset(days=1), freq="h", inclusive="left"))
                                for d in counts.index]
    counts["complete"] = counts.observed_hours == counts.expected_hours
    return counts


def weather_range_violations(df: pd.DataFrame) -> int:
    """Count physically impossible weather values (negative wind/radiation, cloud outside 0-100)."""
    total = 0
    for c in df:
        if c.endswith("cloud_cover"):
            total += int(((df[c] < 0) | (df[c] > 100)).sum())
        elif c.endswith(("wind_speed_100m", "shortwave_radiation")):
            total += int((df[c] < 0).sum())
    return total


def residual_load_flags(market: pd.DataFrame, tolerance: float = 0.1) -> pd.DataFrame:
    """Rows where published residual load differs from load - solar - wind by more than `tolerance` MWh."""
    recomputed = (market.load_mwh - market.solar_mwh - market.wind_onshore_mwh - market.wind_offshore_mwh)
    diff = (market.residual_load_mwh - recomputed).rename("residual_difference_mwh")
    flags = market.loc[diff.abs() > tolerance].copy()
    flags["residual_difference_mwh"] = diff
    return flags


def forecast_buffer_hours(forecast: pd.DataFrame, lead_hours: int) -> pd.Series:
    """Hours between the nominal forecast time and the 11:00 D-1 issue time for every valid hour."""
    local_days = forecast.index.tz_convert(TZ).normalize()
    issue_local = (local_days.tz_localize(None) - pd.Timedelta(days=1) + pd.Timedelta(hours=11)).tz_localize(TZ)
    nominal = forecast.index - pd.Timedelta(hours=lead_hours)
    return pd.Series((issue_local.tz_convert("UTC") - nominal).total_seconds() / 3600, index=forecast.index)
