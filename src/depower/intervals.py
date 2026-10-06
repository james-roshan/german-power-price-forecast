"""Quantile (q10/q90) LightGBM models giving a nominal 80% prediction interval."""
from __future__ import annotations

import pandas as pd

from .backtest import audit_backtest
from .metrics import interval_table
from .models import lgb_factory


def quantile_intervals(data: pd.DataFrame, best_params: dict, begin, stop, window_days: int,
                       retrain_days: int = 7, verbose: bool = True) -> pd.DataFrame:
    """Rolling-origin q10 and q90 models; returns the interval table (see `metrics.interval_table`)."""
    preds = {}
    for q in (0.1, 0.9):
        params = dict(best_params, objective="quantile", alpha=q)
        pred, _ = audit_backtest(data, lgb_factory(params), begin, stop,
                                 window_days=window_days, retrain_days=retrain_days)
        preds[q] = pred
        if verbose:
            print("Quantile complete:", q, flush=True)
    return interval_table(preds[0.1].actual, preds[0.1].prediction, preds[0.9].prediction)
