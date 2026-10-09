"""Fail closed on structural input changes; missing observations are never fabricated."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_frame(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if frame.empty or list(frame.columns) != columns:
        raise ValueError(f"{name}: empty input or schema changed")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC" or index.hasnans:
        raise ValueError(f"{name}: UTC timestamps required")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name}: duplicate or unsorted timestamps")
    if not index.equals(pd.date_range(index.min(), index.max(), freq="h")):
        raise ValueError(f"{name}: missing or non-hourly timestamps")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
        raise ValueError(f"{name}: non-numeric values")
    if ((~np.isfinite(frame)) & frame.notna()).any().any():
        raise ValueError(f"{name}: infinite values")


def validate_weather_units(payload: dict, variables: list[str], units: dict) -> None:
    actual = payload.get("hourly_units", {})
    for variable in variables:
        # Open-Meteo labels Celsius with the SI symbol; historical project
        # metadata uses degC. These are equivalent, unlike m/s versus km/h.
        unit = actual.get(f"{variable}_previous_day2")
        unit = {"°C": "degC", "W/m²": "W/m2"}.get(unit, unit)
        if unit != units[variable]:
            raise ValueError(f"Weather unit changed: {variable}")
