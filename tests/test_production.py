"""Production regressions: real model, DST leakage, integrity and hosted app states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from depower.config import TARGET
from depower.features import enhanced_features
from depower.live import delivery_hours
from depower.monitoring import distribution_shift, export_dashboard, matched_pairs
from depower.production import restore_ledger
from depower.validation import validate_frame, validate_weather_units


def ledger(tmp_path):
    hours = delivery_hours("2026-10-25")
    frame = pd.DataFrame({"prediction": -5.0, "weekly_baseline": 0.0, "previous_day_baseline": -2.0}, index=hours)
    file = tmp_path / "predictions_2026-10-25.parquet"
    frame.to_parquet(file)
    record = {
        "mode": "live",
        "status": "issued",
        "delivery_day": "2026-10-25",
        "issued_at_utc": "2026-10-24T07:00:00+00:00",
        "cutoff_utc": "2026-10-24T09:00:00+00:00",
        "prediction_file": file.name,
        "prediction_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "model_state": {"config_hash": "test"},
        "capture_directory": "absent",
    }
    (tmp_path / "forecast_2026-10-25.json").write_text(json.dumps(record))
    return hours, record, frame


def test_fall_dst_last_hour_price_lag_is_known_before_issue():
    index = pd.date_range("2026-10-01", "2026-10-27", freq="h", tz="UTC")
    market = pd.DataFrame({TARGET: np.arange(len(index), dtype=float)}, index=index)
    hours = delivery_hours("2026-10-25")
    first = enhanced_features(market, None).loc[hours]
    market.loc[market.index >= hours[0], TARGET] = 1e9
    second = enhanced_features(market, None).loc[hours]
    pd.testing.assert_series_equal(first[f"{TARGET}_lag24h"], second[f"{TARGET}_lag24h"])
    assert first.loc[hours[-1], f"{TARGET}_lag24h"] == market.loc[hours[-1] - pd.Timedelta(hours=25), TARGET]


def test_matching_uses_completed_hours_and_keeps_negative_prices(tmp_path):
    hours, _, _ = ledger(tmp_path)
    actual = pd.Series(-7.0, index=hours)
    pairs = matched_pairs(tmp_path, actual, hours[10] + pd.Timedelta(hours=1))
    assert len(pairs) == 11 and pairs.error.eq(2).all()
    doc = export_dashboard(tmp_path, actual.to_frame(TARGET), hours[-1] + pd.Timedelta(hours=1))
    assert doc["metrics"]["MAE"] == 2 and doc["metrics"]["RMSE"] == 2 and doc["metrics"]["bias"] == 2
    assert len(doc["comparison"]) == 25
    assert json.loads((tmp_path / "dashboard.json").read_text())["forecasts"][0]["prediction"] == -5


def test_integrity_and_late_issue_rejected(tmp_path):
    hours, record, frame = ledger(tmp_path)
    frame["prediction"] = 99
    frame.to_parquet(tmp_path / record["prediction_file"])
    with pytest.raises(ValueError, match="integrity"):
        matched_pairs(tmp_path, pd.Series(1.0, index=hours), hours[-1] + pd.Timedelta(days=1))
    record["issued_at_utc"] = record["cutoff_utc"]
    (tmp_path / "forecast_2026-10-25.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Late"):
        matched_pairs(tmp_path, pd.Series(1.0, index=hours), hours[-1] + pd.Timedelta(days=1))


def test_bundle_round_trip_preserves_values_and_issue_time(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    hours, record, frame = ledger(live)
    state = tmp_path / "state"
    (state / "forecasts").mkdir(parents=True)
    predictions = json.loads(
        frame.rename_axis("target_time_utc").reset_index().to_json(orient="records", date_format="iso")
    )
    (state / "forecasts/2026-10-25.json").write_text(json.dumps({"record": record, "predictions": predictions}))
    restored = tmp_path / "restored"
    restore_ledger(state, restored)
    result = matched_pairs(restored, pd.Series(-7.0, index=hours), hours[-1] + pd.Timedelta(days=1))
    assert len(result) == 25 and result.prediction.eq(-5).all()
    assert result.issued_at_utc.eq(record["issued_at_utc"]).all()


@pytest.mark.parametrize("bad", ["naive", "duplicates", "gap", "infinite", "schema"])
def test_validation_fails_closed(bad):
    frame = pd.DataFrame({TARGET: [-5.0, 0.0, 2.0]}, index=pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"))
    if bad == "naive":
        frame.index = frame.index.tz_localize(None)
    elif bad == "duplicates":
        frame = pd.concat([frame, frame.iloc[-1:]])
    elif bad == "gap":
        frame = frame.iloc[[0, 2]]
    elif bad == "infinite":
        frame.iloc[0, 0] = np.inf
    else:
        frame = frame.rename(columns={TARGET: "changed"})
    with pytest.raises(ValueError):
        validate_frame(frame, [TARGET], "market")


def test_units_and_drift_small_samples():
    validate_weather_units(
        {"hourly_units": {"temperature_2m_previous_day2": "°C", "shortwave_radiation_previous_day2": "W/m²"}},
        ["temperature_2m", "shortwave_radiation"],
        {"temperature_2m": "degC", "shortwave_radiation": "W/m2"},
    )
    with pytest.raises(ValueError, match="unit"):
        validate_weather_units(
            {"hourly_units": {"wind_speed_100m_previous_day2": "m/s"}}, ["wind_speed_100m"], {"wind_speed_100m": "km/h"}
        )
    old = pd.DataFrame({"x": np.arange(200, dtype=float)})
    assert distribution_shift(old, old.head(23))["x"]["standardized_mean_shift"] is None
    assert distribution_shift(old, pd.DataFrame({"x": np.full(24, 10000.0)}))["x"]["flag"]


def test_full_live_cycle_deadline_and_immutable_record(tmp_path, monkeypatch):
    from depower import live
    from depower.features import cloud_columns, feature_columns, required_columns
    from depower.models import lgb_factory

    day = "2024-05-10"
    hours = delivery_hours(day)
    index = pd.date_range("2024-01-01", "2024-05-11", freq="h", tz="UTC")
    market = pd.DataFrame({TARGET: 30 + 40 * np.sin(np.arange(len(index)) / 24)}, index=index)
    forecast = pd.DataFrame({"test__temperature_2m": 15.0, "test__cloud_cover": np.nan}, index=index)
    market.loc[market.index >= hours[0], TARGET] = np.nan
    features = enhanced_features(market, forecast)
    ctx = live.LiveContext(
        tmp_path,
        market,
        forecast,
        {},
        feature_columns(features),
        required_columns(features, forecast),
        cloud_columns(forecast),
        "test model",
        {
            "factory": lgb_factory({"n_estimators": 20, "verbosity": -1, "n_jobs": 1}),
            "window_days": 120,
            "retrain_days": 1,
        },
    )
    monkeypatch.setattr(live, "utc_now", lambda: pd.Timestamp("2024-05-09T08:00:00Z"))

    def collect(ctx, folder, session):
        market.to_parquet(ctx.live_dir / "market_latest.parquet")
        forecast.to_parquet(ctx.live_dir / "weather_latest.parquet")
        market.to_parquet(folder / "market_snapshot.parquet")
        forecast.to_parquet(folder / "weather_snapshot.parquet")
        return market, forecast

    monkeypatch.setattr(live, "collect_live_inputs", collect)
    output, _ = live.run_live_cycle(ctx)
    assert len(output) == 24 and np.isfinite(output.prediction).all()
    record = json.loads((tmp_path / "forecast_2024-05-10.json").read_text())
    assert record["status"] == "issued" and record["unit"] == "EUR/MWh"
    with pytest.raises(FileExistsError):
        live.run_live_cycle(ctx)
    monkeypatch.setattr(live, "utc_now", lambda: pd.Timestamp("2024-05-09T09:00:00Z"))
    with pytest.raises(RuntimeError, match="cutoff"):
        live.run_live_cycle(ctx)


def test_dashboard_all_views_and_missing_data(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEPOWER_DASHBOARD_FILE", str(tmp_path / "missing.json"))
    monkeypatch.delenv("DEPOWER_DATA_URL", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception and app.warning
    for name in [
        "Predicted vs actual",
        "Model performance",
        "Data and model monitoring",
        "Architecture and methodology",
    ]:
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception


def test_dashboard_with_live_data(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    hours, _, _ = ledger(tmp_path)
    export_dashboard(tmp_path, pd.Series(-7.0, index=hours).to_frame(TARGET), hours[-1] + pd.Timedelta(hours=1))
    monkeypatch.setenv("DEPOWER_DASHBOARD_FILE", str(tmp_path / "dashboard.json"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    for name in ["Forecast overview", "Predicted vs actual", "Model performance", "Data and model monitoring"]:
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception
