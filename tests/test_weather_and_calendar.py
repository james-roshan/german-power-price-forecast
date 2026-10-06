"""Regressions for weather lead time and calendar-based retraining."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from depower.backtest import rolling_backtest
from depower.features import TARGET


def test_forecast_uses_archived_48_hour_lead(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "weather", Path(__file__).parents[1] / "scripts/download_weather.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def get(url, params, timeout):
        assert url == "https://previous-runs-api.open-meteo.com/v1/forecast"
        assert params["models"] == "ecmwf_ifs025"
        assert params["wind_speed_unit"] == "kmh"
        fields = params["hourly"].split(",")
        assert fields == [f"{v}_previous_day2" for v in module.VARIABLES]

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "hourly": {"time": ["2024-04-01T00:00"], **{v: [1.0] for v in fields}},
                    "hourly_units": dict(zip(fields, ["degC", "km/h", "W/m2", "%"])),
                }

        return Response()

    monkeypatch.setattr(module.requests, "get", get)
    result = module.fetch("forecast", 53.9, 8.7, "2024-04-01", "2024-04-01")
    assert list(result.columns) == module.VARIABLES
    assert str(result.index.tz) == "UTC"
    assert result.attrs["hourly_units"]["wind_speed_100m"] == "km/h"


def test_weekly_retraining_and_hour_coverage_across_spring_dst():
    index = pd.date_range("2024-03-01", "2024-04-02", freq="h", tz="Europe/Berlin")
    data = pd.DataFrame({TARGET: 1.0, "x": 1.0}, index=index.tz_convert("UTC"))
    fits = []

    class Model:
        def fit(self, X, y):
            fits.append(X.index[-1].tz_convert("Europe/Berlin").date())
            return self

        def predict(self, X):
            return np.ones(len(X))

    result = rolling_backtest(data, Model, "2024-03-25", "2024-04-01")
    assert fits == [pd.Timestamp("2024-03-24").date(), pd.Timestamp("2024-03-31").date()]
    assert len(result.predictions) == 8 * 24 - 1


def test_empty_backtest_has_clear_error():
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    data = pd.DataFrame({TARGET: 1.0, "x": 1.0}, index=index)
    from depower.backtest import SeasonalNaive

    with pytest.raises(ValueError, match="No evaluable hours"):
        rolling_backtest(data, SeasonalNaive, "2024-02-01", "2024-02-02")


def test_missing_feature_support_preserves_hours_without_filling():
    index = pd.date_range("2024-01-01", "2024-01-04", freq="h", tz="Europe/Berlin")
    data = pd.DataFrame({TARGET: 1.0, "cloud": np.nan}, index=index.tz_convert("UTC"))

    class NativeMissingModel:
        def fit(self, X, y):
            assert X.cloud.isna().all()
            return self

        def predict(self, X):
            assert X.cloud.isna().all()
            return np.ones(len(X))

    result = rolling_backtest(data, NativeMissingModel, "2024-01-03", "2024-01-03",
                              allow_missing_features=True)
    assert result.metrics()["n_hours"] == 24
    assert result.metrics()["MAE"] == 0
