"""Feature engineering for day-ahead price forecasting without look-ahead leakage.

The day-ahead auction for delivery day D closes at 12:00 (Europe/Berlin) on D-1.
So for every hour of day D we may only use:
  * prices/load/generation observed up to ~D-1 11:00 local  -> we use lags of >= 2 days
    for hourly values (lag 48h, 168h) and the full D-2 daily profile, which is always known;
  * weather FORECASTS issued before the auction (not observed weather for day D);
  * calendar information (hour, weekday, holidays), which is known in advance.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from .config import LAG_COLUMNS, SAFE_LAGS_H, TARGET, TZ


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(TZ)
    de_holidays = holidays.Germany(years=range(local.year.min(), local.year.max() + 1))
    f = pd.DataFrame(index=index)
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["month"] = local.month
    f["is_weekend"] = (local.dayofweek >= 5).astype(int)
    f["is_holiday"] = pd.Index(local.date).isin(list(de_holidays)).astype(int)
    # cyclic encodings help linear models
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    f["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    return f


def lag_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for c in cols:
        for lag in SAFE_LAGS_H:
            f[f"{c}_lag{lag}h"] = df[c].shift(lag)
    # rolling statistics that end 48h before the target hour
    past = df[TARGET].shift(48)
    f["price_roll24_mean"] = past.rolling(24).mean()
    f["price_roll24_std"] = past.rolling(24).std()
    f["price_roll168_mean"] = past.rolling(168).mean()
    return f


def build_features(market: pd.DataFrame, weather_forecast: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a feature matrix aligned to `market.index` (hourly, UTC)."""
    market = market.asfreq("h")
    lag_cols = [c for c in market.columns if c in LAG_COLUMNS]
    parts = [calendar_features(market.index), lag_features(market, lag_cols)]
    if weather_forecast is not None:
        parts.append(weather_forecast.reindex(market.index))
    X = pd.concat(parts, axis=1)
    X[TARGET] = market[TARGET]
    return X


def enhanced_features(market: pd.DataFrame, weather_forecast: pd.DataFrame | None) -> pd.DataFrame:
    """`build_features` plus a 24-hour price lag (day-ahead prices for D-1 are public before D's auction).

    The 24-hour lag is *not* applied to load or generation, whose D-1 afternoon is unobserved.
    """
    frame = build_features(market, weather_forecast=weather_forecast)
    frame[f"{TARGET}_lag24h"] = market[TARGET].asfreq("h").shift(24)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != TARGET]


def cloud_columns(weather: pd.DataFrame) -> list[str]:
    """Forecast cloud cover has source gaps; LightGBM handles them natively."""
    return [c for c in weather.columns if c.endswith("cloud_cover")]


def required_columns(frame: pd.DataFrame, weather: pd.DataFrame) -> list[str]:
    """Columns that must be present for a row to be eligible (everything except cloud cover)."""
    clouds = set(cloud_columns(weather))
    return [c for c in frame.columns if c not in clouds]


def naive_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Parameter-free baselines built from elapsed-hour price lags."""
    use_previous_day = frame.index.tz_convert(TZ).dayofweek.isin([1, 2, 3, 4])
    return pd.DataFrame({
        "Weekly naive": frame[f"{TARGET}_lag168h"],
        "Two-day naive": frame[f"{TARGET}_lag48h"],
        "Previous-day naive": frame[f"{TARGET}_lag24h"],
        "Weekday-aware naive": np.where(use_previous_day, frame[f"{TARGET}_lag24h"],
                                        frame[f"{TARGET}_lag168h"]),
    }, index=frame.index)


def feature_group(column: str) -> str:
    if "__" in column:
        return "weather forecast"
    if "lag" in column or "roll" in column:
        return "market history"
    return "calendar"
