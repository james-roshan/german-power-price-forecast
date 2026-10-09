"""Deployment runner using the research model's frozen selection and live workflow.

Persistent state contains text histories and immutable forecast bundles. Joblib
artifacts are locally generated, never downloaded or loaded from dashboard data.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import SMARD_SERIES, TARGET, TZ
from .data import market_path
from .features import cloud_columns, enhanced_features
from .live import LiveContext, collect_live_inputs, run_live_cycle
from .models import lgb_factory
from .monitoring import export_dashboard
from .validation import validate_frame

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def read_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def context(state: Path, runtime: Path) -> LiveContext:
    selection = json.loads((ROOT / "production/selection.json").read_text(encoding="utf-8"))
    weather_meta = json.loads((ROOT / "production/weather.json").read_text(encoding="utf-8"))
    market, weather = read_history(state / "market.csv"), read_history(state / "weather.csv")
    validate_frame(market, list(SMARD_SERIES), "market")
    weather_cols = [f"{s}__{v}" for s in weather_meta["locations"] for v in weather_meta["variables"]]
    validate_frame(weather, weather_cols, "weather")
    columns = selection["selected_features"]
    cloud = cloud_columns(weather)
    return LiveContext(
        runtime,
        market,
        weather,
        weather_meta,
        columns,
        [TARGET, *[c for c in columns if c not in cloud]],
        cloud,
        selection["selected_name"],
        {
            "factory": lgb_factory(selection["best_lgbm_params"]),
            "window_days": selection["selected_window_days"],
            "retrain_days": selection["selected_retrain_days"],
        },
        selection["best_lgbm_params"],
    )


def restore_ledger(state: Path, runtime: Path) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    for path in sorted((state / "forecasts").glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        record = bundle["record"]
        # Parquet byte hashes cannot survive a CSV/JSON round-trip. Recompute
        # locally; the authoritative durable bundle preserves the actual values.
        prediction = pd.DataFrame(bundle["predictions"])
        prediction.index = pd.to_datetime(prediction.pop("target_time_utc"), utc=True)
        prediction.index.name = None
        target = runtime / record["prediction_file"]
        prediction.to_parquet(target)
        from .data import file_sha256

        record["prediction_sha256"] = file_sha256(target)
        (runtime / f"forecast_{record['delivery_day']}.json").write_text(json.dumps(record), encoding="utf-8")
    for name in ["monitoring_history.csv"]:
        if (state / name).exists():
            shutil.copyfile(state / name, runtime / name)


def persist(ctx: LiveContext, state: Path, document: dict) -> None:
    live = ctx.live_dir
    (state / "forecasts").mkdir(parents=True, exist_ok=True)
    for record_path in sorted(live.glob("forecast_*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("mode") != "live":
            continue
        destination = state / "forecasts" / f"{record['delivery_day']}.json"
        if destination.exists():
            continue
        frame = pd.read_parquet(live / record["prediction_file"])
        frame.index.name = "target_time_utc"
        values = json.loads(frame.reset_index().to_json(orient="records", date_format="iso"))
        with destination.open("x", encoding="utf-8") as stream:
            json.dump({"record": record, "predictions": values}, stream, indent=2)
    for label in ["market", "weather"]:
        snapshot = live / f"{label}_latest.parquet"
        frame = (
            pd.read_parquet(snapshot)
            if snapshot.exists()
            else getattr(ctx, "market" if label == "market" else "forecast")
        )
        # Training window plus lags/rolling support. Keep hourly gaps as nulls.
        begin = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=760)
        frame.loc[frame.index >= begin].to_csv(state / f"{label}.csv", index_label="timestamp_utc")
    for name in ["dashboard.json", "monitoring_history.csv", "comparison.csv"]:
        if (live / name).exists():
            shutil.copyfile(live / name, state / name)


def seed(state: Path, root: Path) -> None:
    """Explicit local bootstrap; no network or historical forecast generation."""
    state.mkdir(parents=True, exist_ok=True)
    for name in ["market.csv", "weather.csv"]:
        if (state / name).exists():
            raise FileExistsError(f"Seed would overwrite {name}")
    raw = root / "data/raw"
    market = pd.read_parquet(market_path(raw))
    weather = pd.read_parquet(raw / "weather_forecast.parquet")
    ctx_meta = json.loads((ROOT / "production/weather.json").read_text(encoding="utf-8"))
    validate_frame(market, list(SMARD_SERIES), "market")
    validate_frame(weather, [f"{s}__{v}" for s in ctx_meta["locations"] for v in ctx_meta["variables"]], "weather")
    start = min(market.index.max(), weather.index.max()) - pd.Timedelta(days=760)
    market.loc[market.index >= start].to_csv(state / "market.csv", index_label="timestamp_utc")
    weather.loc[weather.index >= start].to_csv(state / "weather.csv", index_label="timestamp_utc")
    (state / "README.md").write_text(
        "# Forecast state\nHistorical seed inputs are retrospective, not live observations or forecasts.\n"
        "Automated runs append immutable forecast bundles and refresh histories.\n",
        encoding="utf-8",
    )


def run(state: Path, runtime: Path, mode="forecast") -> dict:
    state, runtime = state.resolve(), runtime.resolve()
    if runtime.exists() and any(runtime.iterdir()):
        raise ValueError("Use a new empty runtime directory for each run")
    restore_ledger(state, runtime)
    ctx = context(state, runtime)
    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        tomorrow = (pd.Timestamp.now(tz=TZ).normalize() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        already_issued = (state / "forecasts" / f"{tomorrow}.json").exists()
        if mode == "forecast" and not already_issued:
            run_live_cycle(ctx, session)
        else:
            folder = runtime / "monitor_capture"
            folder.mkdir()
            collect_live_inputs(ctx, folder, session)
    market = pd.read_parquet(runtime / "market_latest.parquet")
    # Reference only earlier training rows, never delivery-day labels.
    weather = pd.read_parquet(runtime / "weather_latest.parquet")
    features = enhanced_features(market, weather)
    reference = features.loc[features.index < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2), ctx.columns].tail(
        24 * 30
    )
    reference.to_parquet(runtime / "feature_reference.parquet")
    document = export_dashboard(runtime, market)
    document["pipeline_status"] = "alert" if document["alerts"] else "success"
    (runtime / "dashboard.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    persist(ctx, state, document)
    if document["alerts"]:
        raise RuntimeError("; ".join(document["alerts"]))
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seed", "forecast", "monitor"])
    parser.add_argument("--state", type=Path, default=Path("data/state"))
    parser.add_argument("--runtime", type=Path, default=Path("data/runtime"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.command == "seed":
            seed(args.state, args.root)
        else:
            run(args.state, args.runtime, args.command)
    except Exception:
        LOG.exception("Production pipeline failed")
        args.runtime.mkdir(parents=True, exist_ok=True)
        (args.runtime / "pipeline_failure.json").write_text(
            json.dumps({"status": "failed", "at_utc": pd.Timestamp.now(tz="UTC").isoformat()}), encoding="utf-8"
        )
        # Persist any issued forecast even when subsequent monitoring fails.
        if args.command != "seed":
            try:
                ctx = context(args.state, args.runtime)
                snapshot = args.runtime / "market_latest.parquet"
                market = pd.read_parquet(snapshot) if snapshot.exists() else ctx.market
                doc = export_dashboard(args.runtime, market)
                doc["pipeline_status"] = "failed"
                doc["alerts"].append("Latest pipeline failed; inspect GitHub Actions diagnostics")
                (args.runtime / "dashboard.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
                persist(ctx, args.state, doc)
            except Exception:
                LOG.exception("Failed to persist diagnostics; existing state remains authoritative")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
