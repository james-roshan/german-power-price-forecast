"""End-to-end experiment: data audit -> EDA -> original comparison -> extended selection -> checks.

Mirrors `notebooks/end_to_end_forecast.ipynb`, but built from the reusable `depower` modules.
All outputs are written under `out_dir` (default `data/processed/pipeline`); raw files are read-only.
"""
from __future__ import annotations

import importlib.metadata
import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .backtest import LagBaseline, SeasonalNaive, audit_backtest, rolling_backtest
from .config import (
    FOLD_STARTS,
    MODEL_PARAMS,
    RETRAIN_EVERY_DAYS,
    SEED,
    TARGET,
    TEST_END,
    TEST_START,
    TRAIN_WINDOW_DAYS,
    TZ,
    evaluation_window,
)
from .covariates import select_available_covariates
from .data import (
    Datasets,
    assert_quality,
    delivery_day_coverage,
    forecast_buffer_hours,
    load_datasets,
    market_dictionary,
    quality_audit,
    residual_load_flags,
    weather_dictionary,
    weather_range_violations,
)
from .features import (
    build_features,
    cloud_columns,
    enhanced_features,
    feature_columns,
    feature_group,
    naive_predictions,
    required_columns,
)
from .intervals import quantile_intervals
from .live import LiveContext, delivery_hours, issue_cutoff, replay_day
from .metrics import diagnostic_frame, error_slice, interval_summary, paired_block_interval, regression_metrics
from .models import make_lgbm, ridge_factory
from .selection import baseline_cv, learning_and_capacity, make_folds, run_selection

EXAMPLE_TIME = pd.Timestamp("2024-06-15 12:00", tz="UTC")
ORIGINAL_NAMES = ["Seasonal naive (168h)", "Two-day naive (48h)", "LightGBM market",
                  "LightGBM market + forecast weather"]
WEATHER_MODEL = "LightGBM market + forecast weather"


@dataclass
class RunConfig:
    """Experiment settings. Defaults reproduce the full notebook experiment."""

    test_start: str = TEST_START
    test_end: str = TEST_END
    fold_starts: list[str] = field(default_factory=lambda: list(FOLD_STARTS))
    fold_days: int = 28
    n_trials: int = 7
    include_competitors: bool = True
    bootstrap_samples: int = 1000
    run_intervals: bool = True
    run_capacity: bool = True
    run_extended: bool = True

    @classmethod
    def quick(cls) -> RunConfig:
        """Small smoke-test configuration: one test week, two folds, three trials, no neural net."""
        return cls(test_start="2025-09-25", test_end="2025-10-01", fold_starts=["2025-05-01", "2025-08-01"],
                   fold_days=14, n_trials=2, include_competitors=False, bootstrap_samples=50)


@dataclass
class Results:
    """Everything produced by a run, for programmatic inspection."""

    out: Path
    tables: dict = field(default_factory=dict)
    predictions: pd.DataFrame | None = None
    extended_predictions: pd.DataFrame | None = None
    selection: object | None = None
    intervals: pd.DataFrame | None = None
    checks: pd.DataFrame | None = None


class Runner:
    def __init__(self, root: Path, cfg: RunConfig | None = None, out_dir: Path | None = None, verbose: bool = True):
        self.root = Path(root)
        self.cfg = cfg or RunConfig()
        self.out = Path(out_dir) if out_dir else self.root / "data/processed/pipeline"
        self.fig_dir = self.out / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.ds: Datasets = load_datasets(self.root)
        self.begin, self.stop, self.expected_test = evaluation_window(self.cfg.test_start, self.cfg.test_end)
        self.full_period = (self.cfg.test_start, self.cfg.test_end) == (TEST_START, TEST_END)
        self.res = Results(self.out)
        plots.set_style()

    # ------------------------------------------------------------------ helpers
    def log(self, *args):
        if self.verbose:
            print(*args, flush=True)

    def table(self, name: str, df: pd.DataFrame | pd.Series, index: bool = True) -> None:
        self.res.tables[name] = df
        df.to_csv(self.out / f"{name}.csv", index=index)

    def fig(self, fig, name: str) -> None:
        plots.save_figure(fig, name, self.fig_dir)

    # ------------------------------------------------------------------ stage 1
    def audit_and_eda(self) -> None:
        ds = self.ds
        self.log("[1/4] data audit and EDA")
        quality, missing = quality_audit(ds.frames)
        assert_quality(quality)
        self.table("data_quality", quality)
        self.table("missing_values", missing, index=False)
        self.table("market_dictionary", market_dictionary(ds.market, EXAMPLE_TIME))
        self.table("weather_dictionary", weather_dictionary(ds.archive, ds.forecast, EXAMPLE_TIME))
        self.table("delivery_day_coverage", delivery_day_coverage(ds.market))
        self.table("residual_load_flags", residual_load_flags(ds.market))
        for name, df in [("archive", ds.archive), ("forecast", ds.forecast)]:
            if weather_range_violations(df):
                raise AssertionError(f"{name} weather has range violations")
        buffer = forecast_buffer_hours(ds.forecast, ds.forecast_meta["lead_hours"])
        if buffer.min() <= 0:
            raise AssertionError("Forecast lead leaves no buffer before the 11:00 D-1 cutoff.")
        self.buffer_hours_min = float(buffer.min())

        eda = ds.market.loc[ds.market.index < self.begin].copy()
        eda_local = eda.tz_convert(TZ)
        self.fig(plots.workflow(), "00_workflow")
        self.fig(plots.price_history(eda[TARGET]), "01_price_history")
        self.fig(plots.price_distribution(eda[TARGET]), "02_price_distribution")
        self.fig(plots.calendar_patterns(eda_local), "03_calendar_patterns")
        self.fig(plots.hour_by_year(eda_local), "04_hour_by_year")
        annual = plots.annual_prices(eda_local)
        self.table("development_annual_prices", annual)
        self.fig(plots.annual_regimes(annual), "05_annual_regimes")
        fig, corr = plots.market_relationships(eda)
        self.fig(fig, "06_market_relationships")
        weather_eda = eda.join(ds.archive, how="inner")
        self.fig(plots.weather_relationships(weather_eda), "07_weather_relationships")
        self.table("development_weather_correlations",
                   weather_eda.corr().loc[ds.archive.columns, [TARGET, "solar_mwh", "wind_onshore_mwh", "load_mwh"]])

    # ------------------------------------------------------------------ stage 2
    def evaluate(self, name, data, factory, results: dict, allow_missing_features=False):
        started = time.perf_counter()
        result = rolling_backtest(data, factory, self.cfg.test_start, self.cfg.test_end,
                                  train_window_days=TRAIN_WINDOW_DAYS, retrain_every_days=RETRAIN_EVERY_DAYS,
                                  allow_missing_features=allow_missing_features)
        pd.testing.assert_index_equal(result.predictions.index.as_unit("ns"), self.expected_test.as_unit("ns"),
                                      check_names=False)
        if not np.isfinite(result.predictions.to_numpy()).all():
            raise ValueError(f"Non-finite predictions from {name}")
        results[name] = result
        self.log(" ", name, {k: round(v, 3) for k, v in result.metrics().items()},
                 f"elapsed {time.perf_counter() - started:.1f}s")

    def original_experiment(self) -> None:
        ds, cfg = self.ds, self.cfg
        self.log("[2/4] original four-model comparison")
        X_market = build_features(ds.market)
        X_weather = build_features(ds.market, weather_forecast=ds.forecast)
        if len(X_market.columns) - 1 != 36 or len(X_weather.columns) - 1 != 56:
            raise AssertionError("Unexpected feature counts")
        required = list(X_market.columns) + [c for c in ds.forecast if not c.endswith("cloud_cover")]
        eligible = X_weather.dropna(subset=required).index
        missing_test = self.expected_test.difference(eligible)
        if len(missing_test):
            raise AssertionError(f"Missing eligible test hours: {list(missing_test[:10])}")
        if not ds.market.loc[self.expected_test, TARGET].notna().all():
            raise AssertionError("Missing test targets")
        self.X_market, self.X_weather, self.eligible = X_market, X_weather, eligible
        learn_market, learn_weather = X_market.loc[eligible], X_weather.loc[eligible]
        self.original_columns = list(learn_weather.columns.drop(TARGET))

        results: dict = {}
        self.evaluate(ORIGINAL_NAMES[0], X_market[[TARGET, f"{TARGET}_lag168h"]], SeasonalNaive, results)
        self.evaluate(ORIGINAL_NAMES[1], X_market[[TARGET, f"{TARGET}_lag48h"]], partial(LagBaseline, 48), results)
        self.evaluate(ORIGINAL_NAMES[2], learn_market, make_lgbm, results, allow_missing_features=True)
        self.evaluate(ORIGINAL_NAMES[3], learn_weather, make_lgbm, results, allow_missing_features=True)
        self.results = results

        scores = pd.DataFrame({n: r.metrics() for n, r in results.items()}).T
        scores["n_hours"] = scores.n_hours.astype(int)
        scores["MAE_improvement_vs_weekly_pct"] = 100 * (1 - scores.MAE / scores.loc[ORIGINAL_NAMES[0], "MAE"])
        self.scores = scores
        self.table("model_results", scores)
        predictions = pd.DataFrame({"actual": ds.market.loc[self.expected_test, TARGET]}, index=self.expected_test)
        for name, result in results.items():
            predictions[name] = result.predictions.y_pred
        predictions.to_parquet(self.out / "test_predictions.parquet")
        self.res.predictions = predictions
        self.fig(plots.model_comparison(scores), "08_model_comparison")

        errors = predictions.drop(columns="actual").sub(predictions.actual, axis=0).abs()
        local = predictions.index.tz_convert(TZ)
        hourly = errors.groupby(local.hour).mean()
        monthly = errors.groupby(local.strftime("%Y-%m")).mean()
        self.table("mae_by_local_hour", hourly)
        self.table("mae_by_weekend", errors.groupby(np.where(local.dayofweek >= 5, "weekend", "weekday")).mean())
        self.table("mae_by_price_regime",
                   errors.groupby(np.where(predictions.actual < 0, "negative price", "nonnegative price")).mean())
        self.fig(plots.error_slices(hourly, monthly), "09_error_slices")
        week_stop = (pd.Timestamp(cfg.test_start, tz=TZ) + pd.DateOffset(days=7)).tz_convert("UTC")
        week = predictions.loc[predictions.index < week_stop].tz_convert(TZ)
        self.fig(plots.forecast_week(week, ["actual", ORIGINAL_NAMES[0], WEATHER_MODEL]), "10_forecast_week")
        self.fig(plots.evaluation_timeline(eligible.min().tz_convert(TZ), self.begin.tz_convert(TZ),
                                           self.stop.tz_convert(TZ)), "11_evaluation_timeline")
        self.fig(plots.prediction_diagnostics(predictions.actual, predictions[WEATHER_MODEL]),
                 "12_prediction_diagnostics")
        gain = 100 * (1 - scores.loc[WEATHER_MODEL, "MAE"] / scores.loc["LightGBM market", "MAE"])
        self.log(f"  forecast weather changes MAE by {gain:+.2f}% vs market-only LightGBM")

    # ------------------------------------------------------------------ stage 3
    def extended_experiment(self) -> None:
        ds, cfg = self.ds, self.cfg
        self.log("[3/4] development-only selection, test comparison, intervals")
        enhanced_all = enhanced_features(ds.market, ds.forecast)
        self.feature_cols = feature_columns(enhanced_all)
        self.cloud_cols = cloud_columns(ds.forecast)
        self.required_cols = required_columns(enhanced_all, ds.forecast)
        enhanced = enhanced_all.dropna(subset=self.required_cols)
        if len(self.feature_cols) != 57 or TARGET in self.feature_cols:
            raise AssertionError("Expected 57 predictors without the target")
        if not self.expected_test.difference(enhanced.index).empty:
            raise AssertionError("Test hours missing from the enhanced feature table")
        self.enhanced = enhanced
        development = enhanced.loc[enhanced.index < self.begin].copy()
        folds = make_folds(self.begin, cfg.fold_starts, cfg.fold_days)
        self.folds = folds
        self.table("feature_inventory", pd.DataFrame({"feature": self.feature_cols,
                                                      "group": [feature_group(c) for c in self.feature_cols]}), index=False)

        sel = run_selection(development, folds, self.original_columns, n_trials=cfg.n_trials,
                            include_competitors=cfg.include_competitors, verbose=self.verbose)
        self.res.selection = sel
        self.sel = sel
        self.log(f"  development-selected model: {sel.selected_name}")
        self.table("cv_fold_results", sel.cv.folds, index=False)
        self.table("cv_leaderboard", sel.board)
        self.table("cv_fit_diagnostics", sel.cv.fit_logs, index=False)
        sel.cv.fold_predictions.to_parquet(self.out / "cv_predictions.parquet")
        self.table("hyperparameter_folds", sel.cv.folds.loc[sel.cv.folds.model.isin(sel.candidate_specs)], index=False)
        (self.out / "hyperparameter_candidates.json").write_text(
            json.dumps(sel.candidate_specs, indent=2), encoding="utf-8")
        self.table("baseline_cv_results", baseline_cv(ds.market[TARGET], folds), index=False)
        board = self.res.tables["baseline_cv_results"].groupby("model").agg(
            CV_MAE=("MAE", "mean"), sd_fold_MAE=("MAE", "std"))
        self.table("cv_baseline_comparison", pd.concat(
            [board, sel.board.loc[[sel.selected_name], ["CV_MAE", "sd_fold_MAE"]]]))

        if cfg.run_capacity:
            curve, capacity = learning_and_capacity(development, folds, self.feature_cols, sel.best_params)
            self.table("learning_curve", curve, index=False)
            self.table("capacity_diagnostics", capacity, index=False)
            self.fig(plots.learning_and_capacity(curve, capacity), "13_learning_and_capacity")
            self.capacity = capacity

        # ---- retrospective test comparison, after development choices are frozen
        test_ext = self.res.predictions.copy()
        for name, values in naive_predictions(enhanced.loc[self.expected_test]).items():
            test_ext[name] = values
        test_logs = {}
        names = [n for n in dict.fromkeys([sel.best_trial, sel.daily_name, "Ridge", "Pooled Lasso",
                                           "Small MLP", sel.selected_name]) if n in sel.family_specs]
        for name in names:
            started = time.perf_counter()
            pred, log = audit_backtest(enhanced, begin=self.begin, stop=self.stop, **sel.family_specs[name])
            test_ext[name] = pred.prediction
            test_logs[name] = log
            self.log(f"  test: {name} | MAE {regression_metrics(pred.actual, pred.prediction)['MAE']:.3f}"
                     f" | {time.perf_counter() - started:.1f}s")
        rows = []
        for name in test_ext.columns.drop("actual"):
            row = {"model": name, **regression_metrics(test_ext.actual, test_ext[name])}
            if name in test_logs:
                log = test_logs[name]
                row.update(train_MAE=np.average(log.train_MAE, weights=log.n_train),
                           train_RMSE=np.sqrt(np.average(log.train_RMSE ** 2, weights=log.n_train)),
                           n_fits=len(log), fits_with_warnings=int(log.warnings.ne("").sum()))
            rows.append(row)
        ext_scores = pd.DataFrame(rows).set_index("model")
        ext_scores["MAE_improvement_vs_weekly_pct"] = 100 * (1 - ext_scores.MAE / ext_scores.loc["Weekly naive", "MAE"])
        self.ext_scores, self.test_ext = ext_scores, test_ext
        self.table("extended_model_results", ext_scores)
        test_ext.to_parquet(self.out / "extended_test_predictions.parquet")
        self.res.extended_predictions = test_ext
        pd.concat([log.assign(model=n) for n, log in test_logs.items()]).to_csv(self.out / "test_fit_diagnostics.csv")
        self.fig(plots.extended_comparison(ext_scores), "14_extended_comparison")

        # ---- robustness and error concentration
        selected = sel.selected_name
        robustness = pd.DataFrame([
            {"reference": ref, **paired_block_interval(test_ext.actual, test_ext[selected], test_ext[ref],
                                                       samples=cfg.bootstrap_samples)}
            for ref in ["Weekly naive", "Previous-day naive", WEATHER_MODEL]]).set_index("reference")
        self.table("paired_bootstrap", robustness)
        diag = diagnostic_frame(test_ext.actual, test_ext[selected])
        self.diag = diag
        for group in ["hour", "month", "weekday", "price_regime"]:
            self.table(f"errors_by_{group}", error_slice(diag, group))
        self.fig(plots.error_heatmap_and_bias(diag), "15_error_heatmap_and_bias")
        self.table("largest_errors", diag.nlargest(20, "absolute_error"))
        self.fig(plots.error_drift(diag), "16_error_drift")

        if cfg.run_intervals:
            intervals = quantile_intervals(enhanced, sel.best_params, self.begin, self.stop,
                                           window_days=sel.best_spec["window_days"], verbose=self.verbose)
            self.res.intervals = intervals
            by_hour = intervals.groupby(intervals.index.tz_convert(TZ).hour).agg(
                hours=("covered", "size"), coverage=("covered", "mean"), width=("width", "mean"))
            self.table("interval_summary", interval_summary(intervals).to_frame("value"))
            self.table("interval_coverage_by_hour", by_hour)
            self.table("interval_coverage_by_regime", intervals.groupby(diag.price_regime, observed=True).agg(
                hours=("covered", "size"), coverage=("covered", "mean"), width=("width", "mean")))
            intervals.to_parquet(self.out / "interval_predictions.parquet")
            self.fig(plots.prediction_intervals(intervals, by_hour), "17_prediction_intervals")
            self.log(f"  80% interval empirical coverage: {100 * intervals.covered.mean():.1f}%")

        driver_status = pd.DataFrame([
            ("Market history", "Used", "SMARD snapshot; historical revisions not reconstructed"),
            ("Five-site forecast weather", "Used", "48-hour lead; exact historical publication not known"),
            ("Load and renewable forecasts", "Pending source integration", "ENTSO-E/SMARD availability/version audit required"),
            ("Outages and cross-border capacity", "Pending source integration", "Use only announcements available before issue"),
            ("Fuel and carbon signals", "Pending source integration", "Choose accessible series and publication convention"),
            ("Capacity-weighted spatial weather", "Pending source integration", "Requires additional historical sites/capacity weights"),
        ], columns=["driver", "status", "requirement"])
        self.table("driver_status", driver_status, index=False)

        # ---- historical replay with hidden targets (offline; not live evidence)
        self.live_ctx = LiveContext(
            live_dir=self.out / "live", market=ds.market, forecast=ds.forecast, forecast_meta=ds.forecast_meta,
            feature_columns=self.feature_cols, required_columns=self.required_cols, cloud_columns=self.cloud_cols,
            selected_name=selected, selected_spec=sel.selected_spec, best_params=sel.best_params)
        replay, model, train = replay_day(self.live_ctx, cfg.test_start)
        if not np.allclose(replay.prediction, test_ext.loc[replay.index, selected]):
            raise AssertionError("Historical replay differs from the backtest prediction.")
        replay.to_parquet(self.out / "historical_replay.parquet")
        self.log("  historical replay matches the backtest (not live evidence)")

        record = {"created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "selected_name": selected,
                  "best_lgbm_trial": sel.best_trial, "best_lgbm_params": sel.best_params,
                  "selected_window_days": sel.selected_spec["window_days"],
                  "selected_retrain_days": sel.selected_spec["retrain_days"],
                  "selected_features": sel.selected_spec.get("columns", self.feature_cols),
                  "cv_folds": [{k: str(v) for k, v in f.items()} for f in folds],
                  "input_manifest": ds.manifest,
                  "test_status": "Previously inspected retrospective benchmark; not a fresh holdout",
                  "live_status": "Implemented in depower.live; not exercised by this offline run",
                  "external_drivers": "Not integrated; as-of selection helper implemented"}
        (self.out / "selection_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ stage 4
    def validation_checks(self) -> pd.DataFrame:
        self.log("[4/4] validation checks")
        ds, checks = self.ds, []

        def passed(label):
            checks.append((label, "PASS"))

        small = ds.market.loc["2024-05-01":"2024-06-30"].copy()
        shock = pd.Timestamp("2024-06-15 12:00", tz="UTC")
        original = build_features(small)
        changed = small.copy()
        changed.loc[shock, TARGET] += 10000
        cols = original.columns.drop(TARGET)
        pd.testing.assert_frame_equal(original.loc[:shock, cols], build_features(changed).loc[:shock, cols])
        passed("No changed-price influence on current/past predictors")
        assert original.loc[shock, f"{TARGET}_lag48h"] == small.loc[shock - pd.Timedelta(hours=48), TARGET]
        passed("48-hour lag matches the exact historical timestamp")

        toy_index = pd.date_range("2024-03-01", "2024-04-02", freq="h", tz=TZ).tz_convert("UTC")
        toy = pd.DataFrame({TARGET: 1.0, "cloud": np.nan}, index=toy_index)
        fit_days = []

        class MissingAware:
            def fit(self, X, y):
                assert X.cloud.isna().all()
                fit_days.append(X.index[-1].tz_convert(TZ).date())
                return self

            def predict(self, X):
                return np.ones(len(X))

        toy_result = rolling_backtest(toy, MissingAware, "2024-03-25", "2024-04-01", allow_missing_features=True)
        assert fit_days == [pd.Timestamp("2024-03-24").date(), pd.Timestamp("2024-03-31").date()]
        assert len(toy_result.predictions) == 8 * 24 - 1 and toy_result.metrics()["MAE"] == 0
        passed("Calendar-based retraining, 23-hour DST day and native missing-feature handling")

        preds, n = self.res.predictions, len(self.expected_test)
        assert len(preds) == n and preds.index.is_unique and np.isfinite(preds.to_numpy()).all()
        if self.full_period:
            assert n == 8760
        for name, result in self.results.items():
            pd.testing.assert_index_equal(result.predictions.index.as_unit("ns"), self.expected_test.as_unit("ns"),
                                          check_names=False)
            e = preds.actual - preds[name]
            assert np.isclose(e.abs().mean(), self.scores.loc[name, "MAE"])
            assert np.isclose(np.sqrt(np.mean(e ** 2)), self.scores.loc[name, "RMSE"])
        passed("All original models cover identical finite predictions; MAE/RMSE recomputed independently")
        assert ds.forecast_meta["lead_hours"] == 48 and self.buffer_hours_min > 0
        assert not set(ds.archive.columns).intersection(self.X_market.columns)
        passed("Documented forecast lead; no weather in market-only inputs")

        if self.cfg.run_extended:
            assert all(f["stop"] <= self.begin.tz_convert(TZ) for f in self.folds)
            logs = self.sel.cv.fit_logs
            assert (pd.to_datetime(logs.train_end, utc=True) < pd.to_datetime(logs.fit_day, utc=True)).all()
            passed("Every CV fit precedes its forecast; all CV folds precede the test period")
            changed_market = ds.market.copy()
            changed_market.loc[changed_market.index >= self.begin, TARGET] += 10000
            before = enhanced_features(ds.market, ds.forecast).loc[:self.begin, self.feature_cols]
            after = enhanced_features(changed_market, ds.forecast).loc[:self.begin, self.feature_cols]
            pd.testing.assert_frame_equal(before, after)
            passed("Current/future target changes cannot alter current/past predictors")
            te = self.test_ext
            assert np.isfinite(te.to_numpy()).all() and len(te) == n
            for name in te.columns.drop("actual"):
                assert np.isclose(regression_metrics(te.actual, te[name])["MAE"], self.ext_scores.loc[name, "MAE"])
            passed("All compared models score identical hours with independently checked metrics")
            if self.res.intervals is not None:
                iv = self.res.intervals
                assert (iv.lower <= iv.upper).all()
                passed("Interval bounds ordered")
            assert len(delivery_hours("2026-03-29")) == 23 and len(delivery_hours("2025-10-26")) == 25
            assert issue_cutoff("2026-03-29").tz_convert(TZ).hour == 11
            passed("Prospective forecast indices and cutoff honor DST")
            fake = pd.Timestamp("2025-01-02 12:00", tz="UTC")
            rev = pd.DataFrame({"valid_time": [fake] * 2,
                                "available_at": [pd.Timestamp("2025-01-01 08:00", tz="UTC"),
                                                 pd.Timestamp("2025-01-01 12:00", tz="UTC")],
                                "feature": ["wind_forecast"] * 2, "value": [10.0, 999.0]})
            sel = select_available_covariates(rev, pd.DatetimeIndex([fake]), pd.Timestamp("2025-01-01 10:00", tz="UTC"))
            assert sel.iloc[0, 0] == 10.0
            passed("Late covariate revisions are excluded at the issue cutoff")
            pipe = ridge_factory().fit(pd.DataFrame({"x": [1.0, np.nan, 3.0]}), [1.0, 2.0, 3.0])
            before_stats = pipe.named_steps["simpleimputer"].statistics_.copy()
            pipe.predict(pd.DataFrame({"x": [1e9, np.nan]}))
            assert np.array_equal(before_stats, pipe.named_steps["simpleimputer"].statistics_)
            passed("Prediction cannot refit training imputation statistics")
            assert not (self.out / "live" / f"forecast_{self.cfg.test_start}.json").exists()
            passed("Historical replay is not entered in the live ledger")

        result = pd.DataFrame(checks, columns=["check", "result"])
        self.table("validation_checks", result, index=False)
        self.res.checks = result
        return result

    # ------------------------------------------------------------------ driver
    def run(self) -> Results:
        self.audit_and_eda()
        self.original_experiment()
        if self.cfg.run_extended:
            self.extended_experiment()
        self.validation_checks()
        metadata = {
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "inputs": self.ds.manifest,
            "test_start_local": self.cfg.test_start, "test_end_local": self.cfg.test_end, "timezone": TZ,
            "expected_test_hours": len(self.expected_test), "model_params": MODEL_PARAMS, "seed": SEED,
            "train_window_days": TRAIN_WINDOW_DAYS, "retrain_every_days": RETRAIN_EVERY_DAYS,
            "weather_metadata": self.ds.forecast_meta, "config": self.cfg.__dict__,
            "versions": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "lightgbm", "scikit-learn"]},
        }
        (self.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        self.log("done:", self.out)
        return self.res


def run_pipeline(root: Path, cfg: RunConfig | None = None, out_dir: Path | None = None,
                 verbose: bool = True) -> Results:
    return Runner(root, cfg, out_dir, verbose).run()
