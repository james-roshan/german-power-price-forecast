"""Point-forecast, interval and error-slice metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from .config import SEED, TZ


def regression_metrics(actual, predicted) -> dict[str, float | int]:
    """MAE/RMSE/bias etc. No percentage errors: prices can be zero or negative."""
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Metrics require nonempty, finite, aligned predictions.")
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(error.mean()), "median_AE": float(np.median(np.abs(error))),
            "p95_AE": float(np.quantile(np.abs(error), 0.95)),
            "within_20_pct": float(100 * np.mean(np.abs(error) <= 20)), "n_hours": len(error)}


def paired_block_interval(actual: pd.Series, challenger: pd.Series, reference: pd.Series,
                          samples: int = 1000, block_days: int = 7, seed: int = SEED) -> dict[str, float]:
    """Weekly-block bootstrap of the MAE improvement of `challenger` over `reference`.

    Positive improvement means the challenger has lower MAE. Daily loss differences are resampled in
    contiguous blocks because hours within an episode are dependent.
    """
    d = pd.DataFrame({"improvement": (reference - actual).abs() - (challenger - actual).abs()},
                     index=actual.index)
    daily = d.groupby(d.index.tz_convert(TZ).normalize()).improvement.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        starts = rng.integers(0, len(daily), size=int(np.ceil(len(daily) / block_days)))
        indices = np.concatenate([(s + np.arange(block_days)) % len(daily) for s in starts])[:len(daily)]
        draw = daily.iloc[indices]
        estimates.append(draw["sum"].sum() / draw["count"].sum())
    return {"MAE_improvement": float(d.improvement.mean()),
            "lower_95": float(np.quantile(estimates, 0.025)), "upper_95": float(np.quantile(estimates, 0.975))}


def diagnostic_frame(actual: pd.Series, prediction: pd.Series) -> pd.DataFrame:
    """Per-hour errors with Berlin hour/month/weekday and a post-hoc price-regime label."""
    d = pd.DataFrame({"actual": actual, "prediction": prediction})
    d["error"] = d.prediction - d.actual
    d["absolute_error"] = d.error.abs()
    local = d.index.tz_convert(TZ)
    d["hour"], d["month"], d["weekday"] = local.hour, local.strftime("%Y-%m"), local.day_name()
    d["price_regime"] = pd.cut(d.actual, [-np.inf, 0, 100, 200, np.inf], right=False,
                               labels=["negative", "0 to <100", "100 to <200", ">=200"])
    return d


def error_slice(diagnostic: pd.DataFrame, column: str) -> pd.DataFrame:
    return diagnostic.groupby(column, observed=True).agg(
        hours=("error", "size"), MAE=("absolute_error", "mean"),
        median_AE=("absolute_error", "median"), p95_AE=("absolute_error", lambda s: s.quantile(0.95)),
        bias=("error", "mean"))


def interval_table(actual: pd.Series, q10: pd.Series, q90: pd.Series) -> pd.DataFrame:
    """Combine two quantile forecasts into a (crossing-repaired) nominal-80% interval."""
    t = pd.DataFrame({"actual": actual, "q10": q10, "q90": q90})
    t["lower"] = t[["q10", "q90"]].min(axis=1)
    t["upper"] = t[["q10", "q90"]].max(axis=1)
    t["covered"] = t.actual.between(t.lower, t.upper)
    t["width"] = t.upper - t.lower
    t["interval_score"] = (t.width + 10 * (t.lower - t.actual).clip(lower=0)
                           + 10 * (t.actual - t.upper).clip(lower=0))
    return t


def interval_summary(t: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "nominal_coverage_pct": 80.0, "empirical_coverage_pct": 100 * t.covered.mean(),
        "mean_width": t.width.mean(), "mean_interval_score": t.interval_score.mean(),
        "pinball_q10": mean_pinball_loss(t.actual, t.q10, alpha=0.1),
        "pinball_q90": mean_pinball_loss(t.actual, t.q90, alpha=0.9),
        "crossing_pct": 100 * float((t.q10 > t.q90).mean()),
    })
