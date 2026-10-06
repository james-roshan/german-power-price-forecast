"""Rolling-origin backtest: retrain periodically, forecast one delivery day at a time."""
from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .config import TARGET, TZ
from .metrics import regression_metrics


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class BacktestResult:
    predictions: pd.DataFrame  # columns: y_true, y_pred

    def metrics(self) -> dict[str, float | int]:
        e = self.predictions["y_true"] - self.predictions["y_pred"]
        return {
            "MAE": float(e.abs().mean()),
            "RMSE": float(np.sqrt((e ** 2).mean())),
            "n_hours": len(e),
        }


def rolling_backtest(
    data: pd.DataFrame,
    make_model: Callable[[], Model],
    test_start: str,
    test_end: str,
    train_window_days: int = 365 * 2,
    retrain_every_days: int = 7,
    allow_missing_features: bool = False,
) -> BacktestResult:
    """For each delivery day D in [test_start, test_end], train on data strictly before D
    (retraining every `retrain_every_days`) and predict all hours of D.

    Set `allow_missing_features` only for models that natively support NaNs.
    Missing targets are always excluded; callers should assert expected coverage.
    """
    data = data.dropna(subset=[TARGET])
    local_day = data.index.tz_convert(TZ).normalize()
    days = pd.date_range(test_start, test_end, freq="D", tz=TZ)
    feature_cols = [c for c in data.columns if c != TARGET]

    model, last_fit, out = None, None, []
    for day in days:
        if model is None or (day.date() - last_fit.date()).days >= retrain_every_days:
            train = data[(local_day < day) & (local_day >= day - pd.DateOffset(days=train_window_days))]
            if not allow_missing_features:
                train = train.dropna()
            model = make_model().fit(train[feature_cols], train[TARGET])
            last_fit = day
        test = data[local_day == day]
        if not allow_missing_features:
            test = test.dropna(subset=feature_cols)
        if test.empty:
            continue
        out.append(pd.DataFrame({"y_true": test[TARGET], "y_pred": model.predict(test[feature_cols])},
                                index=test.index))
    if not out:
        raise ValueError("No evaluable hours in the requested test period")
    return BacktestResult(pd.concat(out))


class SeasonalNaive:
    """Baseline: price at the same hour one week earlier (feature price_eur_mwh_lag168h)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[f"{TARGET}_lag168h"].to_numpy()


class LagBaseline:
    """Baseline: price `lag` hours earlier (feature `price_eur_mwh_lag{lag}h`)."""

    def __init__(self, lag: int):
        self.column = f"{TARGET}_lag{lag}h"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LagBaseline:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy()


def _berlin(ts) -> pd.Timestamp:
    """Timestamp in Europe/Berlin; tz-naive inputs are interpreted as Berlin local time."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(TZ) if ts.tzinfo is None else ts.tz_convert(TZ)


def audit_backtest(data: pd.DataFrame, factory: Callable[[], Model], begin, stop,
                   window_days: int = 730, retrain_days: int = 7, columns: list[str] | None = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin backtest that also records every fit (training errors, timing, warnings).

    `stop` is exclusive. Every hour of [begin, stop) must exist in `data`; the test set is never
    silently shrunk. Returns (predictions[actual, prediction, fit_id], fit_log).
    """
    begin, stop = _berlin(begin), _berlin(stop)
    columns = list(columns if columns is not None else data.columns.drop(TARGET))
    if TARGET in columns or not begin < stop:
        raise ValueError("Target must not be a feature and begin must precede stop.")
    expected = pd.date_range(begin, stop, freq="h", inclusive="left").tz_convert("UTC")
    if not expected.difference(data.index).empty:
        raise ValueError("Missing evaluation hours; do not silently shrink the test set.")
    local_day = data.index.tz_convert(TZ).normalize()
    outputs, fit_rows = [], []
    model, last_fit, fit_id = None, None, -1
    for day in pd.date_range(begin, stop, freq="D", inclusive="left"):
        if model is None or (day.date() - last_fit.date()).days >= retrain_days:
            train = data.loc[(local_day < day) & (local_day >= day - pd.DateOffset(days=window_days))
                             ].dropna(subset=[TARGET])
            if len(train) < 100:
                raise ValueError("Insufficient training observations.")
            if train.index.max() >= day.tz_convert("UTC"):
                raise AssertionError("Training data must precede the delivery day.")
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = factory().fit(train[columns], train[TARGET])
            fit_id = len(fit_rows)
            tr = regression_metrics(train[TARGET], model.predict(train[columns]))
            fit_rows.append({"fit_id": fit_id, "fit_day": day, "train_start": train.index.min(),
                             "train_end": train.index.max(), "n_train": len(train),
                             "train_MAE": tr["MAE"], "train_RMSE": tr["RMSE"],
                             "fit_seconds": time.perf_counter() - started,
                             "warnings": " | ".join(sorted({str(w.message) for w in caught}))})
            last_fit = day
        test = data.loc[local_day == day]
        outputs.append(pd.DataFrame({"actual": test[TARGET], "prediction": model.predict(test[columns]),
                                     "fit_id": fit_id}, index=test.index))
    pred = pd.concat(outputs)
    pd.testing.assert_index_equal(pred.index.as_unit("ns"), expected.as_unit("ns"), check_names=False)
    if not np.isfinite(pred[["actual", "prediction"]].to_numpy()).all():
        raise ValueError("Non-finite actuals or predictions.")
    fit_log = pd.DataFrame(fit_rows).set_index("fit_id")
    for fid, group in pred.groupby("fit_id"):
        m = regression_metrics(group.actual, group.prediction)
        fit_log.loc[fid, ["validation_MAE", "validation_RMSE", "n_validation"]] = [
            m["MAE"], m["RMSE"], m["n_hours"]]
    return pred, fit_log
