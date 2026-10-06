"""Forward-validation workflow: strict 11:00 Berlin cutoff, recorded inputs, immutable forecasts.

Nothing here runs unless explicitly called. `run_live_cycle` needs network access and must be
invoked before 11:00 Europe/Berlin on the day before delivery. Historical replays (`replay_day`)
exercise the same feature construction offline and are never counted as live evidence.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import SMARD_SERIES, TARGET, TZ
from .features import enhanced_features
from .metrics import regression_metrics

SMARD_BASE = "https://www.smard.de/app/chart_data"


@dataclass
class LiveContext:
    """Everything the live cycle needs, taken from the development-time selection."""

    live_dir: Path
    market: pd.DataFrame            # seed history used if no live snapshot exists yet
    forecast: pd.DataFrame          # seed weather forecasts
    forecast_meta: dict
    feature_columns: list[str]
    required_columns: list[str]     # all frame columns except cloud cover
    cloud_columns: list[str]
    selected_name: str
    selected_spec: dict             # factory, window_days, retrain_days[, columns]
    best_params: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.selected_spec.get("columns", self.feature_columns)


# ---------------------------------------------------------------- time helpers
def _berlin_day(delivery_day) -> pd.Timestamp:
    day = pd.Timestamp(delivery_day)
    day = day.tz_localize(TZ) if day.tzinfo is None else day.tz_convert(TZ)
    return day.normalize()


def delivery_hours(delivery_day) -> pd.DatetimeIndex:
    """UTC hours of one Berlin delivery day (23, 24 or 25 of them)."""
    day = _berlin_day(delivery_day)
    return pd.date_range(day, day + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")


def issue_cutoff(delivery_day) -> pd.Timestamp:
    """11:00 Berlin on the day before delivery, as a UTC timestamp."""
    day = _berlin_day(delivery_day)
    return (pd.Timestamp(day.date()) - pd.DateOffset(days=1) + pd.Timedelta(hours=11)).tz_localize(TZ).tz_convert("UTC")


# ---------------------------------------------------------------- features
def prospective_features(market_history: pd.DataFrame, weather_forecasts: pd.DataFrame, delivery_day,
                         feature_columns: list[str], cloud_columns: list[str]):
    """Features for a delivery day whose target prices are unknown.

    Any price labels at or after the first delivery hour are dropped before features are built.
    Returns (X for the delivery hours, the full feature frame for training).
    """
    future = delivery_hours(delivery_day)
    history = market_history.loc[market_history.index < future[0]].copy()
    index = pd.date_range(history.index.min(), future[-1], freq="h", tz="UTC")
    features = enhanced_features(history.reindex(index), weather_forecasts)
    requested = features.loc[future, feature_columns]
    required = [c for c in feature_columns if c not in cloud_columns]
    if requested[required].isna().any().any():
        bad = requested[required].columns[requested[required].isna().any()].tolist()
        raise ValueError(f"Missing required live features: {bad}")
    if not np.isfinite(requested.fillna(0).to_numpy()).all():
        raise ValueError("Non-finite live features.")
    if not features.loc[future, TARGET].isna().all():
        raise AssertionError("Target labels leaked into the delivery window.")
    return requested, features


def replay_day(ctx: LiveContext, delivery_day):
    """Offline replay of one historical delivery day with its targets hidden.

    Returns (predictions frame, fitted model, training frame). Never written to the live ledger.
    """
    hours = delivery_hours(delivery_day)
    day = _berlin_day(delivery_day)
    X, frame = prospective_features(ctx.market.loc[ctx.market.index < hours[0]], ctx.forecast, day,
                                    ctx.feature_columns, ctx.cloud_columns)
    train = frame.loc[frame.index < hours[0]].dropna(subset=ctx.required_columns)
    train = train.loc[train.index >= (day - pd.DateOffset(days=ctx.selected_spec["window_days"])).tz_convert("UTC")]
    model = ctx.selected_spec["factory"]().fit(train[ctx.columns], train[TARGET])
    out = pd.DataFrame({"prediction": model.predict(X[ctx.columns]),
                        "actual_for_later_scoring": ctx.market.reindex(hours)[TARGET],
                        "mode": "historical_replay"}, index=hours)
    return out, model, train


# ---------------------------------------------------------------- recording / collection
def save_json_exclusive(path, document) -> None:
    """Write JSON, refusing to overwrite an existing file."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, default=str)


def fetch_recorded_json(session, url: str, params, folder: Path, label: str):
    response = session.get(url, params=params, timeout=45)
    received_at = pd.Timestamp.now(tz="UTC")
    response.raise_for_status()
    payload = response.json()
    save_json_exclusive(Path(folder) / f"{label}.json", {
        "url": response.url, "received_at_utc": received_at,
        "sha256": hashlib.sha256(response.content).hexdigest(), "response": payload})
    return payload


def collect_live_inputs(ctx: LiveContext, folder: Path, session=None):
    """Refresh the last 7 days of SMARD and 48-hour-lead Open-Meteo data, recording every response."""
    import requests

    now = pd.Timestamp.now(tz="UTC")
    tomorrow = now.tz_convert(TZ).normalize() + pd.DateOffset(days=1)
    live = ctx.live_dir
    market_base = pd.read_parquet(live / "market_latest.parquet") if (live / "market_latest.parquet").exists() else ctx.market.copy()
    weather_base = pd.read_parquet(live / "weather_latest.parquet") if (live / "weather_latest.parquet").exists() else ctx.forecast.copy()
    start = min(market_base.index.max(), weather_base.index.max()) - pd.Timedelta(days=7)
    meta = ctx.forecast_meta
    own_session = session is None
    session = session or requests.Session()
    try:
        series = {}
        for name, (fid, region) in SMARD_SERIES.items():
            base = f"{SMARD_BASE}/{fid}/{region}"
            index = fetch_recorded_json(session, f"{base}/index_hour.json", None, folder, f"{name}_index")["timestamps"]
            rows = []
            for stamp in index:
                if int(start.timestamp() * 1000) - 7 * 86400000 <= stamp <= int(now.timestamp() * 1000):
                    rows.extend(fetch_recorded_json(session, f"{base}/{fid}_{region}_hour_{stamp}.json", None,
                                                    folder, f"{name}_{stamp}")["series"])
            df = pd.DataFrame(rows, columns=["timestamp", "value"])
            if df.empty:
                raise ValueError(f"No refreshed market rows for {name}")
            df["timestamp"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
            series[name] = df.drop_duplicates("timestamp").set_index("timestamp").value.sort_index()
        recent_market = pd.DataFrame(series)
        pieces = []
        for location, (lat, lon) in meta["locations"].items():
            fields = [f"{v}_previous_day2" for v in meta["variables"]]
            params = {"latitude": lat, "longitude": lon, "hourly": ",".join(fields), "models": meta["model"],
                      "start_date": start.strftime("%Y-%m-%d"),
                      "end_date": (tomorrow + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                      "timezone": "UTC", "temperature_unit": "celsius", "wind_speed_unit": "kmh"}
            payload = fetch_recorded_json(session, meta["endpoint"], params, folder, f"weather_{location}")
            hourly = pd.DataFrame(payload["hourly"])
            hourly.index = pd.to_datetime(hourly.pop("time"), utc=True)
            pieces.append(hourly.rename(columns={f"{v}_previous_day2": f"{location}__{v}" for v in meta["variables"]}))
    finally:
        if own_session:
            session.close()
    # The latest refresh replaces even null values in its overlap, so old non-null revisions
    # cannot hide new missingness.
    recent_weather = pd.concat(pieces, axis=1)
    merged_market = pd.concat([market_base.loc[~market_base.index.isin(recent_market.index)], recent_market]).sort_index()
    merged_weather = pd.concat([weather_base.loc[~weather_base.index.isin(recent_weather.index)], recent_weather]).sort_index()
    merged_market.to_parquet(live / "market_latest.parquet")
    merged_weather.to_parquet(live / "weather_latest.parquet")
    merged_market.to_parquet(Path(folder) / "market_snapshot.parquet")
    merged_weather.to_parquet(Path(folder) / "weather_snapshot.parquet")
    return merged_market, merged_weather


def score_live_ledger(ctx: LiveContext, actual_prices: pd.Series, as_of=None) -> pd.DataFrame:
    """Score every issued live forecast whose delivery hours are complete and now observed."""
    live = ctx.live_dir
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of).tz_convert("UTC")
    rows = []
    for record_path in sorted(live.glob("forecast_*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["mode"] != "live":
            continue
        prediction = pd.read_parquet(live / record["prediction_file"])
        actual = actual_prices.reindex(prediction.index)
        ready = actual.notna() & (prediction.index + pd.Timedelta(hours=1) <= as_of)
        row = {"delivery_day": record["delivery_day"], "issued_at_utc": record["issued_at_utc"],
               "on_time": pd.Timestamp(record["issued_at_utc"]) < pd.Timestamp(record["cutoff_utc"]),
               "expected_hours": len(prediction), "scored_hours": int(ready.sum()), "outcomes_as_of_utc": as_of}
        if ready.any():
            row.update(regression_metrics(actual.loc[ready], prediction.loc[ready, "prediction"]))
            row["weekly_baseline_MAE"] = regression_metrics(actual.loc[ready], prediction.loc[ready, "weekly_baseline"])["MAE"]
            row["previous_day_MAE"] = regression_metrics(actual.loc[ready], prediction.loc[ready, "previous_day_baseline"])["MAE"]
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores.to_csv(live / "scores.csv", index=False)
    return scores


def run_live_cycle(ctx: LiveContext, session=None):
    """Issue tomorrow's forecast. Refuses to run after the cutoff or to overwrite an issued forecast."""
    live = ctx.live_dir
    live.mkdir(parents=True, exist_ok=True)
    spec = ctx.selected_spec
    started = pd.Timestamp.now(tz="UTC")
    day = started.tz_convert(TZ).normalize() + pd.DateOffset(days=1)
    cutoff = issue_cutoff(day)
    if started >= cutoff:
        raise RuntimeError("Past 11:00 Berlin cutoff: no live forecast was issued. Run tomorrow before 11:00.")
    day_key = day.strftime("%Y-%m-%d")
    record_file = live / f"forecast_{day_key}.json"
    if record_file.exists():
        raise FileExistsError("An immutable forecast already exists for this delivery day.")
    folder = live / f"capture_{started.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    folder.mkdir()
    try:
        fresh_market, fresh_weather = collect_live_inputs(ctx, folder, session)
        X_future, training_features = prospective_features(fresh_market, fresh_weather, day,
                                                           ctx.feature_columns, ctx.cloud_columns)
        columns = ctx.columns
        train = training_features.loc[
            (training_features.index < delivery_hours(day)[0])
            & (training_features.index >= (day - pd.DateOffset(days=spec["window_days"])).tz_convert("UTC"))]
        train = train.dropna(subset=ctx.required_columns)
        if train.empty:
            raise ValueError("No eligible live training rows.")
        config_payload = {"selected_name": ctx.selected_name, "columns": columns,
                          "window_days": spec["window_days"], "retrain_days": spec["retrain_days"],
                          "best_params": ctx.best_params}
        config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
        state_path = live / "model_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        refit = (state is None or state["config_hash"] != config_hash
                 or (day.date() - pd.Timestamp(state["fit_day"]).date()).days >= spec["retrain_days"])
        if refit:
            model = spec["factory"]().fit(train[columns], train[TARGET])
            model_file = f"model_{started.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.joblib"
            joblib.dump(model, live / model_file)
            state = {"fit_day": day_key, "model_file": model_file, "config_hash": config_hash,
                     "train_start": str(train.index.min()), "train_end": str(train.index.max()),
                     "n_train": len(train)}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        else:
            model = joblib.load(live / state["model_file"])
        if pd.Timestamp.now(tz="UTC") >= cutoff:
            raise RuntimeError("Collection/training finished after cutoff; forecast withheld.")
        output = pd.DataFrame({"prediction": model.predict(X_future[columns]),
                               "weekly_baseline": X_future[f"{TARGET}_lag168h"],
                               "previous_day_baseline": X_future[f"{TARGET}_lag24h"]}, index=X_future.index)
        if not np.isfinite(output.to_numpy()).all():
            raise ValueError("Non-finite predictions; forecast withheld.")
        prediction_file = f"predictions_{day_key}.parquet"
        if (live / prediction_file).exists():
            raise FileExistsError("Prediction artifact already exists; inspect the previous incomplete cycle.")
        X_future.to_parquet(folder / "forecast_features.parquet")
        output.to_parquet(live / prediction_file)
        issued = pd.Timestamp.now(tz="UTC")
        if issued >= cutoff:
            raise RuntimeError("Persistence passed the cutoff; this artifact is not a valid live forecast.")
        save_json_exclusive(record_file, {
            "mode": "live", "delivery_day": day_key, "issued_at_utc": issued, "cutoff_utc": cutoff,
            "prediction_file": prediction_file,
            "prediction_sha256": hashlib.sha256((live / prediction_file).read_bytes()).hexdigest(),
            "capture_directory": str(folder.relative_to(live)), "model_state": state,
            "configuration": config_payload, "weather_policy": ctx.forecast_meta.get("availability_note")})
        save_json_exclusive(folder / "status.json", {"status": "issued", "issued_at_utc": issued})
        return output, score_live_ledger(ctx, fresh_market[TARGET])
    except Exception as exc:
        save_json_exclusive(folder / "failure.json", {
            "status": "failed", "at_utc": pd.Timestamp.now(tz="UTC"),
            "error_type": type(exc).__name__, "message": str(exc)})
        raise
