"""Live-only matched-pair monitoring, with descriptive distribution diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import TARGET
from .metrics import regression_metrics


def matched_pairs(live: Path, actual: pd.Series, as_of: pd.Timestamp) -> pd.DataFrame:
    pieces = []
    for path in sorted(live.glob("forecast_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") != "live":
            continue
        issued = pd.Timestamp(record["issued_at_utc"])
        if issued >= pd.Timestamp(record["cutoff_utc"]):
            raise ValueError("Late forecast in live ledger")
        artifact = live / record["prediction_file"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != record["prediction_sha256"]:
            raise ValueError("Forecast integrity check failed")
        frame = pd.read_parquet(artifact)
        if issued >= frame.index.min():
            raise ValueError("Forecast created after target began")
        frame["actual"] = actual.reindex(frame.index)
        frame = frame.loc[frame.actual.notna() & (frame.index + pd.Timedelta(hours=1) <= as_of)].copy()
        frame["issued_at_utc"] = issued.isoformat()
        frame["model_version"] = record["model_state"]["config_hash"]
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces).sort_index()
    if result.index.has_duplicates:
        raise ValueError("Multiple production forecasts for one target")
    result["error"] = result.prediction - result.actual
    result["absolute_error"] = result.error.abs()
    result["rolling_7d_mae"] = result.absolute_error.rolling("7D", min_periods=24).mean()
    result["rolling_30d_mae"] = result.absolute_error.rolling("30D", min_periods=24).mean()
    return result


def distribution_shift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Mean shift in historical SD units; diagnostic, not a significance test.

    Hourly autocorrelation, seasonality and small samples invalidate naive p-values.
    Report sample counts and suppress comparisons below 24 finite observations.
    """
    report = {}
    for c in reference.columns.intersection(current.columns):
        old, new = reference[c].dropna(), current[c].dropna()
        sd = old.std()
        shift = float(abs(new.mean() - old.mean()) / sd) if len(old) >= 168 and len(new) >= 24 and sd > 0 else None
        report[c] = {
            "standardized_mean_shift": shift,
            "reference_n": len(old),
            "current_n": len(new),
            "flag": shift is not None and shift > 3,
        }
    return report


def export_dashboard(live: Path, market: pd.DataFrame, as_of=None) -> dict:
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of).tz_convert("UTC")
    pairs = matched_pairs(live, market[TARGET], now)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(live.glob("forecast_*.json"))]
    records = [r for r in records if r.get("mode") == "live"]
    latest = records[-1] if records else None
    alerts = []
    observed = market[TARGET].dropna().loc[lambda s: s.index + pd.Timedelta(hours=1) <= now]
    if observed.empty or now - observed.index.max() > pd.Timedelta(hours=48):
        alerts.append("Missing or stale observations (>48 hours)")
    if latest is None or now - pd.Timestamp(latest["issued_at_utc"]) > pd.Timedelta(hours=36):
        alerts.append("No fresh successful forecast (>36 hours)")
    metrics = regression_metrics(pairs.actual, pairs.prediction) if not pairs.empty else {}
    if not pairs.empty:
        metrics["weekly_baseline_MAE"] = float((pairs.weekly_baseline - pairs.actual).abs().mean())
        metrics["previous_day_MAE"] = float((pairs.previous_day_baseline - pairs.actual).abs().mean())
    if len(pairs) >= 168:
        recent = pairs.loc[pairs.index >= pairs.index.max() - pd.Timedelta(days=7)]
        baseline_mae = float((recent.weekly_baseline - recent.actual).abs().mean())
        metrics["weekly_baseline_7d_MAE"] = baseline_mae
        if len(recent) >= 168 and recent.absolute_error.mean() > 1.5 * max(baseline_mae, 1):
            alerts.append("7-day MAE exceeds 1.5x weekly baseline (floor 1 EUR/MWh)")
        older = pairs.loc[
            (pairs.index < recent.index.min()) & (pairs.index >= recent.index.min() - pd.Timedelta(days=30))
        ]
        if len(older) >= 168 and recent.absolute_error.mean() > 1.5 * max(older.absolute_error.mean(), 1):
            alerts.append("7-day MAE exceeds 1.5x preceding 30-day MAE")
    forecast = pd.read_parquet(live / latest["prediction_file"]) if latest else pd.DataFrame()
    drift = {}
    if latest:
        captured = live / latest["capture_directory"] / "forecast_features.parquet"
        reference = live / "feature_reference.parquet"
        if captured.exists() and reference.exists():
            drift = distribution_shift(pd.read_parquet(reference), pd.read_parquet(captured))
            if any(item["flag"] for item in drift.values()):
                alerts.append("Feature mean shift exceeds 3 historical standard deviations; investigate seasonality")
    if len(pairs) >= 336:
        drift["prediction"] = distribution_shift(pairs.iloc[:-168][["prediction"]], pairs.iloc[-168:][["prediction"]])[
            "prediction"
        ]

    def rows(frame):
        if frame.empty:
            return []
        return json.loads(
            frame.rename_axis("target_time_utc").reset_index().to_json(orient="records", date_format="iso")
        )

    document = {
        "updated_at_utc": now.isoformat(),
        "latest": latest,
        "metrics": metrics,
        "alerts": alerts,
        "drift": drift,
        "forecasts": rows(forecast),
        "observations": rows(observed.tail(168).to_frame("actual")),
        "comparison": rows(pairs.tail(24 * 366)),
        "input_missing": {c: int(v) for c, v in market.tail(168).isna().sum().items()},
        "model_versions": sorted({r["model_state"]["config_hash"] for r in records}),
    }
    # Public JSON is data-only. No model unpickling in the web application.
    temp = live / "dashboard.json.tmp"
    temp.write_text(json.dumps(document, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(live / "dashboard.json")
    if not pairs.empty:
        pairs.to_csv(live / "comparison.csv", index_label="target_time_utc")
    snapshot = {"as_of_utc": now.isoformat(), **metrics, "alerts": "; ".join(alerts)}
    history_file = live / "monitoring_history.csv"
    history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
    history = pd.concat([history, pd.DataFrame([snapshot])], ignore_index=True).tail(1500)
    history.to_csv(history_file, index=False)
    return document
