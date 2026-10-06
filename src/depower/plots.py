"""Figures used in the analysis. Each function returns a matplotlib Figure; `save_figure` writes it."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .config import SEED, TARGET, TZ
from .metrics import error_slice

SOURCE_NOTE = "Sources: Bundesnetzagentur | SMARD.de; weather: Open-Meteo"


def set_style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2})


def save_figure(fig, name: str, out: Path, show: bool = False) -> Path:
    fig.text(0.01, 0.005, SOURCE_NOTE, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = Path(out) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def workflow():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set(xlim=(0, 13), ylim=(0, 3.5))
    ax.axis("off")
    boxes = [(0.2, "Market history\n+ weather forecasts", "#e7f0fa"),
             (3.4, "Validate data\n+ build available features", "#e7f0fa"),
             (6.6, "Train using\nearlier delivery days", "#e4f2e9"),
             (9.8, "Predict next day\n+ measure errors", "#fff0d9")]
    for x, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.4), 2.8, 1.0, boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor="#516579"))
        ax.text(x + 1.4, 1.9, label, ha="center", va="center", fontsize=10)
    for x in [3.1, 6.3, 9.5]:
        ax.annotate("", xy=(x + .2, 1.9), xytext=(x - .1, 1.9), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.text(6.5, .55, "Repeat through the test year; retrain every seven local calendar days.",
            ha="center", fontsize=11)
    ax.set_title("From raw data to an honest rolling evaluation", fontsize=14)
    return fig


def price_history(price: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    price.resample("D").mean().plot(ax=axes[0], linewidth=.7, label="Daily mean")
    axes[0].set(title="Development-period daily average price", ylabel="EUR/MWh")
    axes[0].axvspan(pd.Timestamp("2022-02-24", tz="UTC"), pd.Timestamp("2023-06-01", tz="UTC"),
                    color="tomato", alpha=.15, label="Illustrative crisis window")
    axes[0].legend()
    price.resample("MS").std().plot(ax=axes[1], color="darkorange")
    axes[1].set(title="Within-month hourly price variability", ylabel="Standard deviation, EUR/MWh")
    return fig


def price_distribution(price: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(price, bins=100, color="steelblue")
    axes[0].set(title="Full development distribution", xlabel="EUR/MWh", ylabel="Hours")
    without_2022 = price.loc[price.index.tz_convert(TZ).year != 2022]
    central = without_2022.loc[without_2022.between(-50, 200)]
    axes[1].hist(central, bins=70, color="teal")
    axes[1].set(title="Central range [-50, 200], excluding 2022", xlabel="EUR/MWh", ylabel="Hours")
    for ax in axes:
        ax.axvline(0, color="firebrick", linestyle="--")
    return fig


def calendar_patterns(eda_local: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, ax, title in [(eda_local.index.hour, axes[0], "Local hour"),
                           (eda_local.index.dayofweek, axes[1], "Weekday (0=Monday)"),
                           (eda_local.index.month, axes[2], "Month")]:
        eda_local.groupby(key)[TARGET].agg(["mean", "median"]).plot(ax=ax, marker="o", title=title)
        ax.set_ylabel("EUR/MWh")
    return fig


def hour_by_year(eda_local: pd.DataFrame):
    pivot = eda_local.pivot_table(values=TARGET, index=eda_local.index.hour,
                                  columns=eda_local.index.year, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 4))
    pivot.plot(ax=ax, title="Daily price shape changes across development years")
    ax.set(xlabel="Local hour", ylabel="Mean EUR/MWh")
    return fig


def annual_prices(eda_local: pd.DataFrame) -> pd.DataFrame:
    annual = eda_local.groupby(eda_local.index.year)[TARGET].agg(
        hours="size", mean="mean", median="median", minimum="min", maximum="max")
    annual["negative_hours"] = eda_local[TARGET].lt(0).groupby(eda_local.index.year).sum()
    annual["negative_pct"] = 100 * annual.negative_hours / annual.hours
    first, last = annual.index.min(), annual.index.max()
    annual["coverage_note"] = ["partial first local year" if y == first else
                               "partial year before holdout" if y == last else "full local year"
                               for y in annual.index]
    return annual


def annual_regimes(annual: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    annual["mean"].plot.bar(ax=axes[0], color="steelblue", title="Annual mean: development data")
    annual.negative_pct.plot.bar(ax=axes[1], color="teal",
                                 title="Negative-price share (partial years marked in table)")
    axes[0].set_ylabel("EUR/MWh")
    axes[1].set_ylabel("% of available hours")
    return fig


def market_relationships(eda: pd.DataFrame, seed: int = SEED):
    clean = eda.dropna()
    sample = clean.sample(min(6000, len(clean)), random_state=seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for is_crisis, label, color in [(False, "Other development years", "steelblue"), (True, "2022", "tomato")]:
        part = sample.loc[(sample.index.tz_convert(TZ).year == 2022) == is_crisis]
        axes[0].scatter(part.residual_load_mwh / 1000, part[TARGET], s=5, alpha=.25, label=label, color=color)
    axes[0].set(xlabel="Residual load, GWh in hour", ylabel="EUR/MWh", title="Price versus residual load")
    axes[0].legend()
    corr = eda.corr()
    im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = ["Price", "Load", "Residual", "Solar", "Wind land", "Wind sea"]
    axes[1].set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels,
                title="Development-period Pearson correlation")
    axes[1].tick_params(axis="x", rotation=45)
    for i in range(6):
        for j in range(6):
            axes[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], shrink=.8)
    return fig, corr


def weather_relationships(weather_eda: pd.DataFrame, seed: int = SEED):
    clean = weather_eda.dropna()
    sample = clean.sample(min(6000, len(clean)), random_state=seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [("brandenburg__shortwave_radiation", "solar_mwh", "Radiation, W/m2"),
             ("north_sea_coast__wind_speed_100m", "wind_onshore_mwh", "Wind speed, km/h"),
             ("ruhr__temperature_2m", "load_mwh", "Temperature, degC")]
    for ax, (x, y, label) in zip(axes, pairs):
        ax.scatter(sample[x], sample[y] / 1000, s=5, alpha=.15)
        ax.set(xlabel=label, ylabel=f"{y}: GWh in hour", title=f'{x.split("__")[0]} weather')
    return fig


def evaluation_timeline(start, cutoff, stop):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axvspan(start, cutoff, color="#4478a9", alpha=.25, label="Initial training; fixed parameters")
    ax.axvspan(cutoff, stop, color="#d99b35", alpha=.25, label="Rolling test year")
    ax.axvline(cutoff, color="black", linestyle="--")
    ax.set(ylim=(0, 1), yticks=[], title="Chronological evaluation: no random train/test shuffle",
           xlabel="Berlin delivery date")
    ax.legend(loc="upper left")
    return fig


def model_comparison(scores: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    scores.MAE.plot.barh(ax=axes[0], color="steelblue", title="Test MAE (lower is better)")
    scores.RMSE.plot.barh(ax=axes[1], color="teal", title="Test RMSE (lower is better)")
    for ax in axes:
        ax.set_xlabel("EUR/MWh")
    return fig


def error_slices(hourly_mae: pd.DataFrame, monthly_mae: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    hourly_mae.plot(ax=axes[0], marker=".", title="MAE by local hour")
    monthly_mae.plot(ax=axes[1], marker=".", title="MAE by test month")
    for ax in axes:
        ax.set_ylabel("EUR/MWh")
        ax.legend(fontsize=7)
    return fig


def forecast_week(example_week: pd.DataFrame, columns: list[str]):
    fig, ax = plt.subplots(figsize=(14, 4))
    example_week[columns].plot(ax=ax, linewidth=1)
    ax.set(title="First seven test delivery days: predictions versus actual", ylabel="EUR/MWh",
           xlabel="Berlin delivery time")
    return fig


def prediction_diagnostics(actual: pd.Series, predicted: pd.Series):
    signed = predicted - actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(actual, predicted, s=5, alpha=.15, color="#287d70")
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1, label="Perfect prediction")
    axes[0].set(xlabel="Actual EUR/MWh", ylabel="Predicted EUR/MWh", title="Weather model: actual versus predicted")
    axes[0].legend()
    axes[1].hist(signed, bins=90, color="#4478a9")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Prediction minus actual, EUR/MWh", ylabel="Hours", title="Signed test errors")
    return fig


def learning_and_capacity(learning_curve: pd.DataFrame, capacity: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    learning_curve.groupby("history_days")[["train_MAE", "validation_MAE"]].mean().plot(marker="o", ax=axes[0])
    axes[0].set(title="Development learning curve (four-fold average)",
                xlabel="Requested trailing history, days", ylabel="MAE, EUR/MWh")
    means = capacity.groupby("capacity")[["train_MAE", "validation_MAE"]].mean().reindex(["small", "original", "large"])
    means.plot.bar(ax=axes[1], rot=0)
    axes[1].set(title="Capacity versus training and future error", ylabel="MAE, EUR/MWh")
    axes[1].set_ylim(0, means.max().max() * 1.35)
    return fig


def extended_comparison(extended_scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    extended_scores.MAE.sort_values(ascending=False).plot.barh(ax=ax, color="teal")
    ax.set(title="Same-hour retrospective comparison; selected using development CV", xlabel="MAE, EUR/MWh")
    return fig


def error_heatmap_and_bias(diagnostic: pd.DataFrame):
    heat = diagnostic.groupby(["month", "hour"]).absolute_error.mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    im = axes[0].imshow(heat, aspect="auto", cmap="YlOrRd")
    axes[0].set(yticks=range(len(heat)), yticklabels=heat.index, xticks=range(0, 24, 3),
                xlabel="Berlin hour", title="MAE by month and hour")
    fig.colorbar(im, ax=axes[0], label="EUR/MWh")
    error_slice(diagnostic, "hour").bias.plot.bar(ax=axes[1], color="steelblue", rot=0)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(title="Bias by hour: negative means underprediction", ylabel="Prediction minus actual, EUR/MWh")
    return fig


def error_drift(diagnostic: pd.DataFrame):
    daily = diagnostic.absolute_error.groupby(diagnostic.index.tz_convert(TZ).normalize()).mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    daily.plot(ax=ax, alpha=.35, label="Daily MAE")
    daily.rolling(28, min_periods=7).mean().plot(ax=ax, label="Trailing 28-day MAE", lw=2)
    ax.set(title="Error drift through the retrospective test year", ylabel="EUR/MWh")
    ax.legend()
    return fig


def prediction_intervals(intervals: pd.DataFrame, by_hour: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    first = intervals.iloc[:168].tz_convert(TZ)
    axes[0].fill_between(first.index, first.lower, first.upper, alpha=.25, label="Nominal 80% interval")
    axes[0].plot(first.index, first.actual, lw=1, label="Actual")
    axes[0].legend()
    axes[0].set(title="First test week: interval forecasts", ylabel="EUR/MWh", xlabel="Berlin delivery day")
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=TZ))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=TZ))
    by_hour.coverage.mul(100).plot(ax=axes[1], marker="o")
    axes[1].axhline(80, color="black", ls="--", label="Nominal coverage")
    axes[1].legend()
    axes[1].set(title="Coverage by Berlin hour", ylabel="Percent", ylim=(0, 100))
    return fig


