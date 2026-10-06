"""Development-only model selection: chronological CV, bounded search, learning/capacity curves.

Everything here uses delivery days *before* the frozen test period.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import audit_backtest
from .config import FOLD_DAYS, FOLD_STARTS, MODEL_PARAMS, TARGET, TZ
from .metrics import regression_metrics
from .models import candidate_specs, lasso_factory, lgb_factory, make_lgbm, mlp_factory, ridge_factory


def make_folds(test_begin_utc: pd.Timestamp, starts: list[str] = FOLD_STARTS,
               days: int = FOLD_DAYS) -> list[dict]:
    folds = []
    for number, start in enumerate(starts, 1):
        begin = pd.Timestamp(start, tz=TZ)
        stop = begin + pd.DateOffset(days=days)
        if stop > test_begin_utc.tz_convert(TZ):
            raise ValueError(f"Fold {number} overlaps the test period.")
        folds.append({"fold": number, "start": begin, "stop": stop})
    return folds


@dataclass
class CVResults:
    """Accumulates fold summaries, predictions and fit logs across models."""

    summaries: list = field(default_factory=list)
    predictions: list = field(default_factory=list)
    logs: list = field(default_factory=list)

    def add(self, result: tuple) -> pd.DataFrame:
        summary, pred, log = result
        self.summaries.append(summary)
        self.predictions.append(pred)
        self.logs.append(log)
        return summary

    @property
    def folds(self) -> pd.DataFrame:
        return pd.concat(self.summaries, ignore_index=True)

    @property
    def fold_predictions(self) -> pd.DataFrame:
        return pd.concat(self.predictions)

    @property
    def fit_logs(self) -> pd.DataFrame:
        return pd.concat(self.logs, ignore_index=True)


def cv_evaluate(name: str, factory: Callable, development: pd.DataFrame, folds: list[dict],
                window_days: int = 730, retrain_days: int = 7, columns: list[str] | None = None,
                verbose: bool = True):
    """Rolling-origin evaluation of one model over every fold -> (summary, predictions, fit log)."""
    summaries, predictions_out, logs = [], [], []
    for fold in folds:
        pred, log = audit_backtest(development, factory, fold["start"], fold["stop"],
                                   window_days, retrain_days, columns)
        summaries.append({
            "model": name, "fold": fold["fold"], **regression_metrics(pred.actual, pred.prediction),
            "train_MAE": np.average(log.train_MAE, weights=log.n_train),
            "train_RMSE": np.sqrt(np.average(log.train_RMSE ** 2, weights=log.n_train)),
            "n_fits": len(log), "fit_seconds": log.fit_seconds.sum(),
            "fits_with_warnings": int(log.warnings.ne("").sum())})
        predictions_out.append(pred.assign(fold=fold["fold"], model=name))
        logs.append(log.reset_index().assign(fold=fold["fold"], model=name))
    summary = pd.DataFrame(summaries)
    if verbose:
        print(name, "| CV MAE", round(np.average(summary.MAE, weights=summary.n_hours), 3),
              "| fits", int(summary.n_fits.sum()), flush=True)
    return summary, pd.concat(predictions_out), pd.concat(logs, ignore_index=True)


def weighted_mae(group: pd.DataFrame) -> float:
    return float(np.average(group.MAE, weights=group.n_hours))


def leaderboard(cv_folds: pd.DataFrame) -> pd.DataFrame:
    board = cv_folds.groupby("model").agg(
        mean_fold_MAE=("MAE", "mean"), sd_fold_MAE=("MAE", "std"), train_MAE=("train_MAE", "mean"),
        total_fits=("n_fits", "sum"), fit_seconds=("fit_seconds", "sum"),
        fits_with_warnings=("fits_with_warnings", "sum"))
    board["CV_MAE"] = cv_folds.groupby("model").apply(weighted_mae, include_groups=False)
    board["CV_RMSE"] = cv_folds.groupby("model").apply(
        lambda g: np.sqrt(np.average(g.RMSE ** 2, weights=g.n_hours)), include_groups=False)
    board["validation_minus_train_MAE"] = board.CV_MAE - board.train_MAE
    return board


@dataclass
class Selection:
    """Outcome of the development-only selection."""

    cv: CVResults
    board: pd.DataFrame
    search_rank: pd.Series
    candidate_specs: dict
    family_specs: dict
    best_trial: str
    best_params: dict
    selected_name: str
    daily_name: str = "Selected LightGBM daily"

    @property
    def best_spec(self) -> dict:
        return self.candidate_specs[self.best_trial]

    @property
    def selected_spec(self) -> dict:
        return self.family_specs[self.selected_name]


def run_selection(development: pd.DataFrame, folds: list[dict], original_columns: list[str],
                  n_trials: int = 7, include_competitors: bool = True, verbose: bool = True) -> Selection:
    """Bounded LightGBM search, then Ridge/Lasso/MLP, daily refit and the original 56-feature model.

    The final family is chosen by CV MAE only; no test-year score is used.
    """
    cv = CVResults()
    specs = candidate_specs(n_trials)
    for name, spec in specs.items():
        cv.add(cv_evaluate(name, lgb_factory(spec["params"]), development, folds,
                           window_days=spec["window_days"], verbose=verbose))
    search_folds = cv.folds
    search_rank = search_folds.groupby("model").apply(weighted_mae, include_groups=False).sort_values()
    best_trial = search_rank.index[0]
    best_spec = specs[best_trial]
    best_params = best_spec["params"].copy()

    family = {best_trial: {"factory": lgb_factory(best_params), "window_days": best_spec["window_days"],
                           "retrain_days": 7}}
    if include_competitors:
        for name, factory in [("Ridge", ridge_factory), ("Pooled Lasso", lasso_factory),
                              ("Small MLP", mlp_factory)]:
            family[name] = {"factory": factory, "window_days": 730, "retrain_days": 7}
    daily_name = "Selected LightGBM daily"
    family[daily_name] = dict(family[best_trial], retrain_days=1)
    family["Original 56-feature LightGBM"] = {
        "factory": make_lgbm, "window_days": 730, "retrain_days": 7, "columns": list(original_columns)}
    for name, spec in family.items():
        if name == best_trial:
            continue  # already evaluated in the search
        cv.add(cv_evaluate(name, development=development, folds=folds, verbose=verbose, **spec))
    board = leaderboard(cv.folds)
    selected_name = board.loc[list(family), "CV_MAE"].idxmin()
    return Selection(cv, board, search_rank, specs, family, best_trial, best_params, selected_name, daily_name)


def learning_and_capacity(development: pd.DataFrame, folds: list[dict], feature_cols: list[str],
                          best_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learning curve over trailing history (90/180/365 d) and small/original/large tree capacity."""
    def split(fold, history=None):
        begin = fold["start"].tz_convert("UTC")
        lo = (fold["start"] - pd.DateOffset(days=history)).tz_convert("UTC") if history else None
        train = development.loc[(development.index < begin) if lo is None
                                else (development.index >= lo) & (development.index < begin)]
        valid = development.loc[(development.index >= begin) & (development.index < fold["stop"].tz_convert("UTC"))]
        return train, valid

    def score(model, train, valid):
        return (regression_metrics(train[TARGET], model.predict(train[feature_cols]))["MAE"],
                regression_metrics(valid[TARGET], model.predict(valid[feature_cols]))["MAE"])

    curve_rows, capacity_rows = [], []
    for fold in folds:
        for history in [90, 180, 365]:
            train, valid = split(fold, history)
            model = lgb_factory(best_params)().fit(train[feature_cols], train[TARGET])
            tr, va = score(model, train, valid)
            curve_rows.append({"fold": fold["fold"], "history_days": history, "n_train": len(train),
                               "train_MAE": tr, "validation_MAE": va})
        train, valid = split(fold)
        for label, changes in [("small", {"num_leaves": 7, "n_estimators": 100, "min_child_samples": 200}),
                               ("original", {}),
                               ("large", {"num_leaves": 63, "n_estimators": 500, "min_child_samples": 20})]:
            model = lgb_factory(dict(MODEL_PARAMS, **changes))().fit(train[feature_cols], train[TARGET])
            tr, va = score(model, train, valid)
            capacity_rows.append({"fold": fold["fold"], "capacity": label, "train_MAE": tr, "validation_MAE": va})
    return pd.DataFrame(curve_rows), pd.DataFrame(capacity_rows)


def baseline_cv(market_price: pd.Series, folds: list[dict]) -> pd.DataFrame:
    """Apply the four naive rules directly on the development folds (no fitting)."""
    y = market_price.asfreq("h")
    frame = pd.DataFrame({"actual": y, "Weekly naive": y.shift(168), "Two-day naive": y.shift(48),
                          "Previous-day naive": y.shift(24)})
    frame["Weekday-aware naive"] = np.where(frame.index.tz_convert(TZ).dayofweek.isin([1, 2, 3, 4]),
                                            frame["Previous-day naive"], frame["Weekly naive"])
    rows = []
    for fold in folds:
        hours = pd.date_range(fold["start"], fold["stop"], freq="h", inclusive="left").tz_convert("UTC")
        sample = frame.loc[hours]
        if not sample.notna().all().all():
            raise ValueError("Baseline CV hours must be complete.")
        for name in sample.columns.drop("actual"):
            error = sample[name] - sample.actual
            rows.append({"model": name, "fold": fold["fold"], "MAE": error.abs().mean(),
                         "RMSE": float(np.sqrt((error ** 2).mean())), "n_hours": len(error)})
    return pd.DataFrame(rows)
