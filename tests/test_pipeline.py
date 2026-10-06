"""Offline tests on synthetic data: no network, runs in seconds."""
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from depower.backtest import SeasonalNaive, rolling_backtest
from depower.features import SAFE_LAGS_H, TARGET, build_features


@pytest.fixture
def market() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", "2024-03-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    hour = idx.tz_convert("Europe/Berlin").hour
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 30000
    load = 55000 + 10000 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 1000, len(idx))
    price = 80 + (load - solar - 40000) / 400 + rng.normal(0, 5, len(idx))
    return pd.DataFrame({TARGET: price, "load_mwh": load, "solar_mwh": solar}, index=idx)


def test_no_lag_shorter_than_48h():
    # day-ahead bids are placed ~12h-36h before delivery; 48h guarantees the value is known
    assert min(SAFE_LAGS_H) >= 48


def test_features_do_not_use_future(market):
    X = build_features(market)
    t = pd.Timestamp("2023-06-15 12:00", tz="UTC")
    # changing the price at t must not change any feature at t or earlier
    shocked = market.copy()
    shocked.loc[t, TARGET] += 10_000
    X2 = build_features(shocked)
    cols = [c for c in X.columns if c != TARGET]
    pd.testing.assert_frame_equal(X.loc[:t, cols], X2.loc[:t, cols])


def test_backtest_beats_seasonal_naive(market):
    X = build_features(market)
    naive = rolling_backtest(X, SeasonalNaive, "2024-02-01", "2024-02-14", train_window_days=180)
    gbm = rolling_backtest(X, lambda: LGBMRegressor(n_estimators=200, verbose=-1),
                           "2024-02-01", "2024-02-14", train_window_days=180)
    assert gbm.metrics()["MAE"] < naive.metrics()["MAE"]
    assert gbm.metrics()["n_hours"] == 14 * 24


def test_smard_parser(monkeypatch):
    """Parser handles SMARD's weekly chunk format, including null values and duplicates."""
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("dl", pathlib.Path(__file__).parents[1] / "scripts/download_smard.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    week = 7 * 24 * 3600 * 1000
    t0 = 1788127200000
    fake = {
        "index_hour.json": {"timestamps": [t0, t0 + week]},
        f"_hour_{t0}.json": {"series": [[t0, 174.98], [t0 + 3600000, None]]},
        f"_hour_{t0 + week}.json": {"series": [[t0 + week, -5.0], [t0 + week, -5.0]]},
    }
    monkeypatch.setattr(dl, "_get_json", lambda url, s: next(v for k, v in fake.items() if url.endswith(k)))
    monkeypatch.setattr(dl.time, "sleep", lambda *_: None)
    s = dl.fetch_series(4169, "DE-LU", pd.Timestamp("2026-08-01", tz="UTC"), None)
    assert list(s.values) == [174.98, -5.0]  # null dropped, duplicate removed, negative price kept
