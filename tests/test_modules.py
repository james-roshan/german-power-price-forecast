"""Offline unit tests for the data, metrics, selection, interval, covariate and live modules."""
import json

import numpy as np
import pandas as pd
import pytest

from depower import data as dm
from depower.backtest import LagBaseline, audit_backtest
from depower.config import TARGET, TZ, evaluation_window, find_root
from depower.covariates import select_available_covariates
from depower.features import (
    cloud_columns,
    enhanced_features,
    feature_columns,
    feature_group,
    naive_predictions,
    required_columns,
)
from depower.intervals import quantile_intervals
from depower.live import (
    LiveContext,
    delivery_hours,
    issue_cutoff,
    prospective_features,
    replay_day,
    save_json_exclusive,
    score_live_ledger,
)
from depower.metrics import (
    diagnostic_frame,
    error_slice,
    interval_summary,
    interval_table,
    paired_block_interval,
    regression_metrics,
)
from depower.models import candidate_specs, lgb_factory, ridge_factory
from depower.selection import baseline_cv, cv_evaluate, leaderboard, make_folds


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", "2024-06-30 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    hour = idx.tz_convert(TZ).hour
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 30000
    load = 55000 + 10000 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 1000, len(idx))
    price = 80 + (load - solar - 40000) / 400 + rng.normal(0, 5, len(idx))
    return pd.DataFrame({TARGET: price, "load_mwh": load, "solar_mwh": solar}, index=idx)


@pytest.fixture(scope="module")
def forecast(market) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    f = pd.DataFrame({"a__temperature_2m": rng.normal(10, 5, len(market)),
                      "a__cloud_cover": rng.uniform(0, 100, len(market))}, index=market.index)
    f.iloc[:10, 1] = np.nan  # cloud gaps are tolerated
    return f


# ------------------------------------------------------------------ config / data
def test_test_window_has_8760_hours_for_the_default_year():
    begin, stop, expected = evaluation_window()
    assert len(expected) == 8760 and begin < stop


def test_find_root_locates_data_raw(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    nested = tmp_path / "a/b"
    nested.mkdir(parents=True)
    assert find_root(nested) == tmp_path.resolve()


def test_market_path_prefers_newer_snapshot(tmp_path):
    (tmp_path / "smard.parquet").write_bytes(b"x")
    assert dm.market_path(tmp_path).name == "smard.parquet"
    (tmp_path / "smard_sample.parquet").write_bytes(b"x")
    assert dm.market_path(tmp_path).name == "smard_sample.parquet"


def test_quality_audit_flags_gaps_and_duplicates():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    good = pd.DataFrame({"x": np.arange(10.0)}, index=idx)
    quality, missing = dm.quality_audit({"good": good})
    dm.assert_quality(quality)
    assert missing.loc[0, "missing"] == 0
    gappy = good.drop(idx[4])
    quality, _ = dm.quality_audit({"gappy": gappy})
    assert quality.loc["gappy", "missing_timestamps"] == 1
    with pytest.raises(AssertionError):
        dm.assert_quality(quality)
    with pytest.raises(ValueError):
        dm.quality_audit({"naive": good.tz_localize(None)})


def test_delivery_day_coverage_marks_dst_days():
    idx = pd.date_range("2024-03-30", "2024-04-01 23:00", freq="h", tz="Europe/Berlin").tz_convert("UTC")
    cov = dm.delivery_day_coverage(pd.DataFrame({"x": 1.0}, index=idx))
    assert cov.expected_hours.tolist() == [24, 23, 24] and cov.complete.all()


def test_weather_range_violations_and_residual_flags():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    w = pd.DataFrame({"a__cloud_cover": [10, 120, -1], "a__wind_speed_100m": [1, -2, 3]}, index=idx)
    assert dm.weather_range_violations(w) == 3
    m = pd.DataFrame({"load_mwh": [100.0, 100.0, 100.0], "solar_mwh": 10.0, "wind_onshore_mwh": 10.0,
                      "wind_offshore_mwh": 10.0, "residual_load_mwh": [70.0, 70.0, 50.0]}, index=idx)
    assert len(dm.residual_load_flags(m)) == 1


def test_forecast_buffer_is_positive_for_48h_lead():
    idx = pd.date_range("2025-01-02", periods=24, freq="h", tz="UTC")
    assert dm.forecast_buffer_hours(pd.DataFrame({"x": 1.0}, index=idx), 48).min() > 0


# ------------------------------------------------------------------ features
def test_enhanced_features_add_only_a_price_24h_lag(market, forecast):
    frame = enhanced_features(market, forecast)
    cols = feature_columns(frame)
    assert f"{TARGET}_lag24h" in cols and "load_mwh_lag24h" not in cols and TARGET not in cols
    t = pd.Timestamp("2023-06-01 10:00", tz="UTC")
    assert frame.loc[t, f"{TARGET}_lag24h"] == market.loc[t - pd.Timedelta(hours=24), TARGET]


def test_cloud_and_required_columns(market, forecast):
    frame = enhanced_features(market, forecast)
    assert cloud_columns(forecast) == ["a__cloud_cover"]
    assert "a__cloud_cover" not in required_columns(frame, forecast)
    assert feature_group("a__cloud_cover") == "weather forecast"
    assert feature_group("price_roll24_mean") == "market history" and feature_group("hour") == "calendar"


def test_naive_predictions_weekday_rule(market, forecast):
    frame = enhanced_features(market, forecast).dropna(subset=[f"{TARGET}_lag336h"])
    naive = naive_predictions(frame)
    dow = frame.index.tz_convert(TZ).dayofweek
    tue = np.flatnonzero(dow == 1)[0]
    sat = np.flatnonzero(dow == 5)[0]
    assert naive["Weekday-aware naive"].iloc[tue] == frame[f"{TARGET}_lag24h"].iloc[tue]
    assert naive["Weekday-aware naive"].iloc[sat] == frame[f"{TARGET}_lag168h"].iloc[sat]


# ------------------------------------------------------------------ metrics
def test_regression_metrics_known_values():
    m = regression_metrics([0, 0, 0, 0], [1, -1, 3, -3])
    assert m["MAE"] == 2 and m["bias"] == 0 and m["n_hours"] == 4
    assert np.isclose(m["RMSE"], np.sqrt(5))
    with pytest.raises(ValueError):
        regression_metrics([1.0], [np.nan])


def test_paired_bootstrap_detects_better_model():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    actual = pd.Series(np.zeros(len(idx)), index=idx)
    good, bad = actual + 1.0, actual + 5.0
    r = paired_block_interval(actual, good, bad, samples=100)
    assert np.isclose(r["MAE_improvement"], 4.0) and r["lower_95"] > 0


def test_error_slices_and_regimes():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    actual = pd.Series(np.linspace(-10, 250, 48), index=idx)
    d = diagnostic_frame(actual, actual + 2)
    assert (d.absolute_error == 2).all()
    assert set(d.price_regime.dropna().astype(str)) == {"negative", "0 to <100", "100 to <200", ">=200"}
    assert np.allclose(error_slice(d, "hour").MAE, 2)


def test_interval_table_repairs_crossing_and_scores():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    actual = pd.Series([5.0, 50.0, 5.0], index=idx)
    t = interval_table(actual, pd.Series([0.0, 40.0, 10.0], index=idx), pd.Series([10.0, 20.0, 0.0], index=idx))
    assert (t.lower <= t.upper).all()
    s = interval_summary(t)
    assert np.isclose(s.crossing_pct, 100 / 3 * 2) and 0 <= s.empirical_coverage_pct <= 100


# ------------------------------------------------------------------ models / selection
def test_candidate_specs_include_original_and_are_reproducible():
    a, b = candidate_specs(3), candidate_specs(3)
    assert list(a) == ["LightGBM trial 00", "LightGBM trial 01", "LightGBM trial 02", "LightGBM trial 03"]
    assert a == b and a["LightGBM trial 00"]["window_days"] == 730
    assert all("window_days" not in s["params"] for s in a.values())


def test_ridge_pipeline_handles_missing_values():
    X = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})
    assert np.isfinite(ridge_factory().fit(X, [1.0, 2.0, 3.0, 4.0]).predict(X)).all()


def test_make_folds_rejects_overlap_with_test(market):
    begin = pd.Timestamp("2024-02-01", tz=TZ).tz_convert("UTC")
    assert len(make_folds(begin, ["2024-01-01"], 28)) == 1
    with pytest.raises(ValueError):
        make_folds(begin, ["2024-01-20"], 28)


def test_audit_backtest_logs_fits_and_covers_every_hour(market, forecast):
    frame = enhanced_features(market, forecast).dropna(subset=required_columns(enhanced_features(market, forecast), forecast))
    pred, log = audit_backtest(frame, lambda: LagBaseline(168), "2024-03-01", "2024-03-15", window_days=60)
    assert len(pred) == 14 * 24 and len(log) == 2
    assert (pd.to_datetime(log.train_end) < pd.to_datetime(log.fit_day)).all()
    assert {"train_MAE", "validation_MAE", "n_validation"} <= set(log.columns)
    with pytest.raises(ValueError, match="Missing evaluation hours"):
        audit_backtest(frame, lambda: LagBaseline(168), "2024-06-25", "2024-07-10", window_days=60)
    with pytest.raises(ValueError):
        audit_backtest(frame, lambda: LagBaseline(168), "2024-03-15", "2024-03-01")


def test_cv_evaluate_and_leaderboard(market, forecast):
    full = enhanced_features(market, forecast)
    frame = full.dropna(subset=required_columns(full, forecast))
    folds = make_folds(pd.Timestamp("2024-06-01", tz=TZ).tz_convert("UTC"), ["2024-04-01", "2024-05-01"], 7)
    spec = {"n_estimators": 20, "num_leaves": 7, "verbosity": -1, "n_jobs": 1}
    summary, pred, log = cv_evaluate("tiny", lgb_factory(spec), frame, folds, window_days=90, verbose=False)
    assert len(summary) == 2 and len(pred) == 2 * 7 * 24 and set(pred.model) == {"tiny"}
    board = leaderboard(summary)
    assert board.loc["tiny", "CV_MAE"] == pytest.approx(summary.MAE.mean())


def test_baseline_cv_matches_manual_weekly_naive(market):
    folds = make_folds(pd.Timestamp("2024-06-01", tz=TZ).tz_convert("UTC"), ["2024-04-01"], 7)
    out = baseline_cv(market[TARGET], folds)
    weekly = out.loc[out.model == "Weekly naive", "MAE"].iloc[0]
    hours = pd.date_range(folds[0]["start"], folds[0]["stop"], freq="h", inclusive="left").tz_convert("UTC")
    manual = (market.loc[hours, TARGET] - market[TARGET].shift(168).loc[hours]).abs().mean()
    assert weekly == pytest.approx(manual)


def test_quantile_intervals_are_ordered(market, forecast):
    full = enhanced_features(market, forecast)
    frame = full.dropna(subset=required_columns(full, forecast))
    params = {"n_estimators": 20, "num_leaves": 7, "verbosity": -1, "n_jobs": 1}
    t = quantile_intervals(frame, params, pd.Timestamp("2024-04-01", tz=TZ), pd.Timestamp("2024-04-08", tz=TZ),
                           window_days=90, verbose=False)
    assert (t.lower <= t.upper).all() and len(t) == 7 * 24


# ------------------------------------------------------------------ covariates
def test_covariates_require_explicit_availability_and_reject_ambiguity():
    t = pd.Timestamp("2025-01-02 12:00", tz="UTC")
    base = pd.DataFrame({"valid_time": [t, t], "available_at": [pd.Timestamp("2025-01-01 08:00", tz="UTC")] * 2,
                         "feature": ["w", "w"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Ambiguous"):
        select_available_covariates(base, pd.DatetimeIndex([t]), pd.Timestamp("2025-01-01 10:00", tz="UTC"))
    with pytest.raises(ValueError, match="Required columns"):
        select_available_covariates(base.drop(columns="value"), pd.DatetimeIndex([t]), t)
    missing = base.assign(available_at=[None, None])
    with pytest.raises(ValueError, match="availability must be explicit"):
        select_available_covariates(missing, pd.DatetimeIndex([t]), t)


def test_covariates_pick_latest_revision_available_by_cutoff():
    t = pd.Timestamp("2025-01-02 12:00", tz="UTC")
    rev = pd.DataFrame({"valid_time": [t] * 3,
                        "available_at": [pd.Timestamp(f"2025-01-01 {h}:00", tz="UTC") for h in (6, 9, 12)],
                        "feature": ["w"] * 3, "value": [1.0, 2.0, 3.0]})
    out = select_available_covariates(rev, pd.DatetimeIndex([t]), pd.Timestamp("2025-01-01 10:00", tz="UTC"))
    assert out.loc[t, "w"] == 2.0


# ------------------------------------------------------------------ live
def _ctx(tmp_path, market, forecast):
    full = enhanced_features(market, forecast)
    spec = {"factory": lgb_factory({"n_estimators": 20, "num_leaves": 7, "verbosity": -1, "n_jobs": 1}),
            "window_days": 120, "retrain_days": 1}
    return LiveContext(live_dir=tmp_path, market=market, forecast=forecast, forecast_meta={},
                       feature_columns=feature_columns(full), required_columns=required_columns(full, forecast),
                       cloud_columns=cloud_columns(forecast), selected_name="tiny", selected_spec=spec)


def test_delivery_hours_and_cutoff_honor_dst():
    assert len(delivery_hours("2026-03-29")) == 23 and len(delivery_hours("2025-10-26")) == 25
    assert len(delivery_hours("2025-06-01")) == 24
    cut = issue_cutoff("2026-03-29")
    assert cut.tz_convert(TZ).hour == 11 and cut.tz_convert(TZ).day == 28


def test_prospective_features_exclude_future_labels_and_match_replay(market, forecast, tmp_path):
    ctx = _ctx(tmp_path, market, forecast)
    day = "2024-05-10"
    hours = delivery_hours(day)
    poisoned = market.copy()
    poisoned.loc[poisoned.index >= hours[0], TARGET] = 1e9
    X1, _ = prospective_features(market, forecast, day, ctx.feature_columns, ctx.cloud_columns)
    X2, _ = prospective_features(poisoned, forecast, day, ctx.feature_columns, ctx.cloud_columns)
    pd.testing.assert_frame_equal(X1, X2)
    out, model, train = replay_day(ctx, day)
    assert len(out) == 24 and (out["mode"] == "historical_replay").all() and np.isfinite(out.prediction).all()
    assert train.index.max() < hours[0]


def test_prospective_features_reject_missing_required_inputs(market, forecast, tmp_path):
    ctx = _ctx(tmp_path, market, forecast)
    broken = forecast.copy()
    broken.loc[delivery_hours("2024-05-10")[3], "a__temperature_2m"] = np.nan
    with pytest.raises(ValueError, match="Missing required live features"):
        prospective_features(market, broken, "2024-05-10", ctx.feature_columns, ctx.cloud_columns)


def test_save_json_exclusive_never_overwrites(tmp_path):
    path = tmp_path / "x.json"
    save_json_exclusive(path, {"a": pd.Timestamp("2024-01-01", tz="UTC")})
    assert json.loads(path.read_text())["a"].startswith("2024-01-01")
    with pytest.raises(FileExistsError):
        save_json_exclusive(path, {})


def test_score_live_ledger_scores_only_completed_hours(market, forecast, tmp_path):
    ctx = _ctx(tmp_path, market, forecast)
    hours = delivery_hours("2024-05-10")
    pred = pd.DataFrame({"prediction": 100.0, "weekly_baseline": 90.0, "previous_day_baseline": 95.0}, index=hours)
    pred.to_parquet(tmp_path / "predictions_2024-05-10.parquet")
    save_json_exclusive(tmp_path / "forecast_2024-05-10.json", {
        "mode": "live", "delivery_day": "2024-05-10", "issued_at_utc": "2024-05-09T08:00:00+00:00",
        "cutoff_utc": "2024-05-09T09:00:00+00:00", "prediction_file": "predictions_2024-05-10.parquet"})
    actual = pd.Series(100.0, index=hours)
    scores = score_live_ledger(ctx, actual, as_of=hours[11] + pd.Timedelta(hours=1))
    assert scores.loc[0, "scored_hours"] == 12 and scores.loc[0, "MAE"] == 0 and scores.loc[0, "on_time"]
    assert scores.loc[0, "weekly_baseline_MAE"] == 10
    # a replay record must never be scored as live evidence
    save_json_exclusive(tmp_path / "forecast_2024-05-11.json", {"mode": "historical_replay"})
    assert len(score_live_ledger(ctx, actual, as_of=hours[-1] + pd.Timedelta(days=2))) == 1
