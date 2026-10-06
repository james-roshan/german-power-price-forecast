# DE Power Price Forecast — Full Pipeline & Execution Plan

> Every step is ordered. Run them sequentially. Nothing after Step 2 requires internet access.

> Implementation update (2026-09-26): use `notebooks/analysis.ipynb` and
> `python scripts/run_analysis.py` for the executed audit, EDA, baselines and first LightGBM
> comparison. Results are saved under `data/processed/eda/`. The later snippets below remain
> a draft roadmap, not verified results. In particular, the Lasso CV, timezone comparisons,
> deployment CLI and workflow scheduling need revision before use. Numerical oracle gains,
> interview answers and CV outcomes below are unmeasured examples, not project findings.

---

## 0. Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | ≥ 3.10 | f-strings, match, zoneinfo |
| LibreOffice | any recent | PDF conversion (optional, for reports) |
| Git | any | version control |
| 8 GB RAM | — | LightGBM on 7-year hourly dataset |

---

## 1. Environment Setup

```bash
git clone https://github.com/<you>/de-power-price-forecast.git
cd de-power-price-forecast

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # installs all dependencies + dev extras

pytest                             # must show 4 passed, ~3 s, zero warnings
```

Expected output:
```
tests/test_pipeline.py ....                              [100%]
4 passed in 2.31s
```

If any test fails, stop. The scaffold is broken and downstream results will be wrong.

---

## 2. Data Download

> Run on your laptop — the workspace proxy blocks SMARD and Open-Meteo.
> Estimated total time: ~15 minutes. Expected disk: ~150 MB.

### 2a. SMARD (market data, 2019 – today)

```bash
python scripts/download_smard.py \
  --start 2019-01-01 \
  --out data/raw/smard.parquet
```

What it downloads (6 series, hourly, DE-LU / DE):
- `price_eur_mwh` — day-ahead auction price, series 4169
- `load_mwh` — total grid load, series 410
- `residual_load_mwh` — load minus renewables, series 4359
- `solar_mwh` — solar generation, series 4068
- `wind_onshore_mwh` — onshore wind, series 4067
- `wind_offshore_mwh` — offshore wind, series 1225

Verification after download:
```python
import pandas as pd
df = pd.read_parquet("data/raw/smard.parquet")
print(df.shape)             # expect ~55 000 rows × 6 cols
print(df.index[[0, -1]])    # 2019-01-01 00:00 UTC … yesterday ~23:00 UTC
print(df["price_eur_mwh"].describe())  # check for negative prices — they are VALID
print(df.isnull().sum())    # small number of nulls is fine; large gaps need investigation
```

Known data quirks (handle explicitly, write about each in EDA):
- **Negative prices**: do NOT drop or log-transform. Keep as-is.
- **DST transitions**: 23-hour and 25-hour days. The script stores everything in UTC, which has no gaps. Convert to local time only for calendar features.
- **15-minute products since 1 Oct 2025**: SMARD's hourly series are hourly aggregates of 15-minute products. State this assumption clearly; it means your model produces hourly point forecasts.
- **2022 gas crisis spike**: prices exceeded 600 €/MWh for weeks. Do not silently drop these rows. Make a deliberate modelling choice (see Week 3).

### 2b. Weather Forecasts (April 2024 – latest complete UTC date)

```bash
python scripts/download_weather.py \
  --source forecast \
  --start 2024-04-01 \
  --out data/raw/weather_forecast.parquet
```

5 locations, 4 variables each (20 columns):
- Locations: North Sea coast (53.9°N, 8.7°E), Brandenburg (52.4°N, 13.1°E), Ruhr (51.5°N, 7.2°E), Bavaria (48.4°N, 11.6°E), Baden-Württemberg (48.7°N, 9.2°E)
- Variables per location: `temperature_2m`, `wind_speed_100m`, `shortwave_radiation`, `cloud_cover`

**Critical**: the current script uses the Open-Meteo **Previous Runs API**, with ECMWF
IFS 0.25-degree `*_previous_day2` variables (48-hour forecast lead). The earlier stitched
historical-forecast endpoint did not establish pre-auction availability. Exact publication
times are still unavailable; this experiment assumes publication delay is shorter than
the conservative pre-auction buffer. Metadata is saved beside the Parquet file. Archive
weather is used only for EDA or a clearly labelled oracle experiment. Missing cloud forecasts
are retained as NaN and handled natively by LightGBM; no archive weather fills those gaps.

```bash
# Oracle / EDA only — do NOT use these columns as model features
python scripts/download_weather.py \
  --source archive \
  --start 2019-01-01 \
  --out data/raw/weather_archive.parquet
```

---

## 3. EDA Notebook (`notebooks/01_eda.ipynb`)

Open Jupyter:
```bash
jupyter lab
```

Work through these analyses in order. Each one informs a downstream modelling decision.

### 3a. Price Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/raw/smard.parquet")
price = df["price_eur_mwh"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
price.hist(bins=100, ax=axes[0])
axes[0].set_title("Price distribution (all years)")
axes[0].axvline(0, color="red", linestyle="--", label="zero")

price.clip(-50, 200).hist(bins=100, ax=axes[1])
axes[1].set_title("Clipped to [-50, 200] — 2022 excluded")
plt.tight_layout()
```

Note: MAPE is meaningless for negative or near-zero prices. Your primary metric is MAE.

### 3b. Seasonality (Hour / Day / Month / Year)

```python
local = df.copy()
local.index = df.index.tz_convert("Europe/Berlin")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
local.groupby(local.index.hour)["price_eur_mwh"].mean().plot(ax=axes[0], title="Hour of day")
local.groupby(local.index.dayofweek)["price_eur_mwh"].mean().plot(ax=axes[1], title="Day of week (0=Mon)")
local.groupby(local.index.month)["price_eur_mwh"].mean().plot(ax=axes[2], title="Month")
```

### 3c. Negative Price Hours Per Year

```python
neg = df[df["price_eur_mwh"] < 0]
neg.groupby(neg.index.year).size().plot(kind="bar", title="Negative-price hours per year")
# Expect rising trend as solar penetration grows
```

### 3d. 2022 Regime

```python
df["price_eur_mwh"].resample("W").mean().plot(figsize=(14, 4), title="Weekly average price")
plt.axvspan("2022-02-24", "2023-06-01", alpha=0.2, color="red", label="Gas crisis regime")
plt.legend()
```

Decision you must make explicitly: train through 2022, down-weight it, or split the backtest into pre/post regimes. Write your choice in the EDA notebook and why.

### 3e. Price vs Residual Load

```python
# Residual load = load − renewables; high residual → expensive gas plants run → high price
import numpy as np
sample = df.sample(5000, random_state=42)
plt.scatter(sample["residual_load_mwh"], sample["price_eur_mwh"], alpha=0.3, s=10)
plt.xlabel("Residual load (MWh)")
plt.ylabel("Price (€/MWh)")
plt.title("Price vs residual load (random 5k hours)")
```

### 3f. Data Quality Checks

```python
# Gaps (should be 0 in UTC, 1 on spring-forward days in local time)
expected = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC")
missing = expected.difference(df.index)
print(f"Missing hours: {len(missing)}")
if len(missing) > 0:
    print(missing[:10])

# Duplicates
dups = df.index.duplicated()
print(f"Duplicate timestamps: {dups.sum()}")

# DST 23-hour days (spring-forward)
local_day_counts = df.groupby(df.index.tz_convert("Europe/Berlin").normalize()).size()
print("23-hour days:", (local_day_counts == 23).sum())
print("25-hour days:", (local_day_counts == 25).sum())

# Extreme outliers
print(df["price_eur_mwh"].describe(percentiles=[.01, .05, .95, .99, .999]))
```

Log every gap, duplicate, and DST day you find and what you did (filled, kept, flagged).

---

## 4. Feature Engineering

All feature logic lives in `src/depower/features.py`. Do not add features outside this module — the backtest imports it and the leakage test guards it.

### 4a. The Leakage Rule

The auction for delivery day D closes at 12:00 CET on D-1. Bids are placed 12–36 hours before delivery. Therefore:

- **Safe lag**: any price or grid value from ≥ 48 hours before the target hour. SAFE_LAGS_H = [48, 72, 168, 336].
- **Unsafe**: any lag < 48h. Do not add them.
- **Weather**: use only the *forecast* parquet. Never pass the *archive* parquet to `build_features()`. The oracle experiment is the only place the archive is used.

The test `test_features_do_not_use_future` enforces this automatically. Run `pytest` after every change to features.py.

### 4b. Current Feature Set

| Group | Features |
|---|---|
| Calendar | hour, dow, month, is_weekend, is_holiday, hour_sin, hour_cos, doy_sin, doy_cos |
| Price lags | price_eur_mwh_lag48h, _lag72h, _lag168h, _lag336h |
| Grid lags | load_mwh, residual_load_mwh, solar_mwh, wind_onshore_mwh, wind_offshore_mwh — same 4 lags each |
| Rolling stats | price_roll24_mean, price_roll24_std, price_roll168_mean (all computed from shift(48)) |
| Weather forecast | temperature_2m, wind_speed_100m, shortwave_radiation, cloud_cover × 5 locations = 20 cols |

### 4c. Adding a New Feature

```python
# In features.py, add to build_features() or a new helper function
# Then run pytest immediately

def build_features(market, weather_forecast=None):
    ...
    # Example: adding week-of-year
    f["week"] = market.index.tz_convert(TZ).isocalendar().week.astype(int)
    ...
```

---

## 5. Baselines

Run baselines first. A model that cannot beat them is not worth reporting.

```python
from depower.backtest import SeasonalNaive, rolling_backtest
from depower.features import build_features
import pandas as pd

smard = pd.read_parquet("data/raw/smard.parquet")
X = build_features(smard)

# Test period: last 12 months. Tune period: 12 months before that.
# Never change these dates after you start modelling.
TEST_START = "2025-09-26"   # today − 12 months; adjust to your download date
TEST_END   = "2026-09-25"   # today

# Baseline 1: same hour, 1 week ago (lag 168h)
result_naive = rolling_backtest(X, SeasonalNaive, TEST_START, TEST_END, train_window_days=365*2)
print("SeasonalNaive:", result_naive.metrics())

# Baseline 2: same hour, 2 days ago (lag 48h) — write this as a 5-line class
class Lag48hBaseline:
    def fit(self, X, y): return self
    def predict(self, X): return X["price_eur_mwh_lag48h"].to_numpy()

result_48h = rolling_backtest(X, Lag48hBaseline, TEST_START, TEST_END, train_window_days=365*2)
print("Lag48h:", result_48h.metrics())
```

Save the baseline MAE and RMSE in a table. All future models must beat the seasonal naive to be worth reporting.

---

## 6. Model Training Pipeline

### 6a. LEAR (Linear Electricity Price Forecasting Benchmark)

LEAR is the standard academic benchmark: Lasso regression on lagged prices. It uses no weather and is fast.

```python
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class LEAR:
    """LEAR: Lasso with lagged price features only."""
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=5000))

    def fit(self, X, y):
        # Use only price lags and calendar features
        self.cols_ = [c for c in X.columns if "price_eur_mwh_lag" in c or
                      c in ("hour", "dow", "month", "is_weekend", "is_holiday",
                             "hour_sin", "hour_cos", "doy_sin", "doy_cos")]
        self.model.fit(X[self.cols_], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.cols_])

result_lear = rolling_backtest(X, LEAR, TEST_START, TEST_END, train_window_days=365*2)
print("LEAR:", result_lear.metrics())
```

### 6b. LightGBM with Full Feature Set

```python
from lightgbm import LGBMRegressor

def make_lgbm():
    return LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        verbose=-1,
    )

smard = pd.read_parquet("data/raw/smard.parquet")
weather = pd.read_parquet("data/raw/weather_forecast.parquet")
X_full = build_features(smard, weather_forecast=weather)

result_lgbm = rolling_backtest(X_full, make_lgbm, TEST_START, TEST_END, train_window_days=365*2)
print("LightGBM:", result_lgbm.metrics())
```

### 6c. Hyperparameter Tuning (Time-Series CV Only)

**Never use random CV on time-series data.** Use a time-series split from the tuning period (12 months before test start).

```python
from sklearn.model_selection import TimeSeriesSplit
import optuna

TUNE_START = pd.Timestamp(TEST_START) - pd.DateOffset(years=1)
TUNE_END   = pd.Timestamp(TEST_START) - pd.Timedelta(days=1)

tune_data = X_full.loc[
    (X_full.index.tz_convert("Europe/Berlin").normalize() >= TUNE_START) &
    (X_full.index.tz_convert("Europe/Berlin").normalize() <= TUNE_END)
].dropna()

y_tune = tune_data["price_eur_mwh"]
feat_cols = [c for c in tune_data.columns if c != "price_eur_mwh"]
X_tune = tune_data[feat_cols]

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "verbose": -1,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, val_idx in tscv.split(X_tune):
        m = LGBMRegressor(**params)
        m.fit(X_tune.iloc[train_idx], y_tune.iloc[train_idx])
        pred = m.predict(X_tune.iloc[val_idx])
        maes.append(np.abs(y_tune.iloc[val_idx].values - pred).mean())
    return np.mean(maes)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, n_jobs=-1)
print("Best params:", study.best_params)
```

---

## 7. Ablation Study

Run each variant on the same test period. This is a key interview talking point — it shows you understand what each feature group contributes.

```python
ablation_results = {}

# Full model
ablation_results["full"] = rolling_backtest(X_full, make_lgbm, TEST_START, TEST_END).metrics()

# Remove weather (use market features only)
X_no_weather = build_features(smard)   # no weather_forecast arg
ablation_results["no_weather"] = rolling_backtest(X_no_weather, make_lgbm, TEST_START, TEST_END).metrics()

# Remove lags (calendar only)
def build_calendar_only(market):
    from depower.features import calendar_features, TARGET
    X = calendar_features(market.index)
    X[TARGET] = market[TARGET]
    return X

X_cal_only = build_calendar_only(smard)
ablation_results["calendar_only"] = rolling_backtest(X_cal_only, make_lgbm, TEST_START, TEST_END).metrics()

# Print comparison table
import pandas as pd
pd.DataFrame(ablation_results).T[["MAE", "RMSE"]].sort_values("MAE")
```

---

## 8. Oracle Experiment

The oracle uses observed (reanalysis) weather instead of forecast weather. The MAE gap between oracle and regular model is the **forecast-information gap** — how much accuracy you lose because weather forecasts are imperfect. This is important in interviews because it explains why real-world performance is always worse than published benchmarks.

```python
weather_oracle = pd.read_parquet("data/raw/weather_archive.parquet")
X_oracle = build_features(smard, weather_forecast=weather_oracle)

result_oracle = rolling_backtest(X_oracle, make_lgbm, TEST_START, TEST_END)
print("Oracle (observed weather):", result_oracle.metrics())
print("Regular (forecast weather):", result_lgbm.metrics())
print(f"Forecast-information gap: {result_oracle.metrics()['MAE']:.2f} → "
      f"{result_lgbm.metrics()['MAE']:.2f} €/MWh")
```

**Expected result**: oracle MAE is 3–8 €/MWh lower than forecast MAE. The gap is larger in summer (solar uncertainty) and during storms.

---

## 9. Uncertainty Quantification

Point forecasts are not enough for trading or hedging. Prediction intervals tell the operator how wide the uncertainty is.

### 9a. Quantile Regression

```python
def make_lgbm_q(q):
    return LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        verbose=-1,
    )

result_q10 = rolling_backtest(X_full, lambda: make_lgbm_q(0.1), TEST_START, TEST_END)
result_q50 = rolling_backtest(X_full, lambda: make_lgbm_q(0.5), TEST_START, TEST_END)
result_q90 = rolling_backtest(X_full, lambda: make_lgbm_q(0.9), TEST_START, TEST_END)
```

### 9b. Evaluation: Pinball Loss and Coverage

```python
def pinball(y_true, y_pred, q):
    e = y_true - y_pred
    return np.where(e >= 0, q * e, (q - 1) * e).mean()

pred = result_q90.predictions
q10 = result_q10.predictions["y_pred"]
q90 = result_q90.predictions["y_pred"]
y   = result_q50.predictions["y_true"]

coverage = ((y >= q10) & (y <= q90)).mean()
print(f"80% interval coverage: {coverage:.1%}")  # should be near 80%

pb10 = pinball(y, q10, 0.1)
pb50 = pinball(y, result_q50.predictions["y_pred"], 0.5)
pb90 = pinball(y, q90, 0.9)
print(f"Pinball q10={pb10:.2f}, q50={pb50:.2f}, q90={pb90:.2f}")
```

### 9c. Calibration Plot

```python
quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
actual_coverage = []

for q in quantiles:
    r = rolling_backtest(X_full, lambda: make_lgbm_q(q), TEST_START, TEST_END)
    frac = (r.predictions["y_true"] <= r.predictions["y_pred"]).mean()
    actual_coverage.append(frac)

plt.plot(quantiles, actual_coverage, "o-", label="actual")
plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
plt.xlabel("Nominal quantile")
plt.ylabel("Actual fraction below")
plt.title("Calibration plot")
plt.legend()
```

A well-calibrated model sits on the diagonal.

---

## 10. Results Table

Assemble every result into one table before writing any prose.

```python
results = {
    "SeasonalNaive (lag168h)": result_naive.metrics(),
    "Lag48h baseline":          result_48h.metrics(),
    "LEAR (lasso)":             result_lear.metrics(),
    "LightGBM (no weather)":    ablation_results["no_weather"],
    "LightGBM (full)":          result_lgbm.metrics(),
    "LightGBM oracle":          result_oracle.metrics(),
}

df_results = pd.DataFrame(results).T[["MAE", "RMSE"]]
df_results["MAE_vs_naive"] = (df_results["MAE"] / df_results.loc["SeasonalNaive (lag168h)", "MAE"] - 1) * 100
print(df_results.round(2))
```

Also compute error by hour of day and by weekday vs weekend:

```python
preds = result_lgbm.predictions.copy()
preds["hour"] = preds.index.tz_convert("Europe/Berlin").hour
preds["is_weekend"] = preds.index.tz_convert("Europe/Berlin").dayofweek >= 5
preds["error"] = (preds["y_true"] - preds["y_pred"]).abs()

print("MAE by hour:\n", preds.groupby("hour")["error"].mean().round(2))
print("\nMAE weekday vs weekend:\n", preds.groupby("is_weekend")["error"].mean().round(2))
```

---

## 11. Productionisation

### 11a. Daily Forecast Pipeline (CLI)

Create `scripts/run_daily_forecast.py`:

```python
"""Daily pipeline: download today's data, build features, forecast tomorrow."""
import argparse, logging, pathlib, datetime, pandas as pd
from depower.features import build_features
from lightgbm import LGBMRegressor
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

def run(model_path: str, out_dir: str):
    log.info("Loading model from %s", model_path)
    model: LGBMRegressor = joblib.load(model_path)

    log.info("Loading latest market data")
    smard = pd.read_parquet("data/raw/smard.parquet")
    weather = pd.read_parquet("data/raw/weather_forecast.parquet")
    X = build_features(smard, weather_forecast=weather)

    # Predict all hours of tomorrow
    tomorrow = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).date()
    local_day = X.index.tz_convert("Europe/Berlin").normalize().date
    mask = local_day == tomorrow
    X_tomorrow = X[mask].dropna(subset=[c for c in X.columns if c != "price_eur_mwh"])

    if X_tomorrow.empty:
        log.warning("No feature rows for %s — data not yet available?", tomorrow)
        return

    feat_cols = [c for c in X_tomorrow.columns if c != "price_eur_mwh"]
    preds = model.predict(X_tomorrow[feat_cols])

    out = pd.DataFrame({
        "timestamp_utc": X_tomorrow.index,
        "forecast_eur_mwh": preds,
    })

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/forecast_{tomorrow}.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved %d forecasts to %s", len(out), out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/lgbm_latest.pkl")
    p.add_argument("--out", default="data/forecasts")
    run(**vars(p.parse_args()))
```

Train and save the model once:
```python
import joblib, pathlib
from lightgbm import LGBMRegressor

# Train on all data except test period
train = X_full[X_full.index < pd.Timestamp(TEST_START, tz="UTC")].dropna()
feat_cols = [c for c in train.columns if c != "price_eur_mwh"]
model = LGBMRegressor(n_estimators=1000, verbose=-1)
model.fit(train[feat_cols], train["price_eur_mwh"])

pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/lgbm_latest.pkl")
```

### 11b. FastAPI Endpoint

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib, datetime, pathlib

app = FastAPI(title="DE Power Price Forecast API")
model = joblib.load("models/lgbm_latest.pkl")

class ForecastResponse(BaseModel):
    date: str
    forecasts: list[dict]  # [{hour_utc, forecast_eur_mwh, q10, q90}, ...]

@app.get("/forecast/{date}", response_model=ForecastResponse)
def get_forecast(date: str):
    try:
        target_date = datetime.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")

    # Load pre-computed forecast if it exists
    path = pathlib.Path(f"data/forecasts/forecast_{target_date}.csv")
    if not path.exists():
        raise HTTPException(404, f"No forecast found for {date}. Run the daily pipeline first.")

    df = pd.read_csv(path)
    return ForecastResponse(
        date=date,
        forecasts=df.to_dict(orient="records"),
    )

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run locally:
```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload
# Test: curl http://localhost:8000/forecast/2026-09-26
```

### 11c. Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY models/ models/
COPY data/forecasts/ data/forecasts/
COPY api/ api/

RUN pip install --no-cache-dir -e "." fastapi uvicorn

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t de-power-forecast .
docker run -p 8000:8000 de-power-forecast
```

### 11d. GitHub Actions CI

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src tests scripts
      - run: pytest
```

For a scheduled daily forecast (optional):
```yaml
  daily-forecast:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    schedule:
      - cron: "0 11 * * *"   # 11:00 UTC = 12:00 CET — just before auction closes
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e "."
      - run: |
          python scripts/download_smard.py --out data/raw/smard.parquet
          python scripts/download_weather.py --source forecast --out data/raw/weather_forecast.parquet
          python scripts/run_daily_forecast.py
```

---

## 12. Monitoring

After you have live forecasts (D+1 predictions) and real prices (D actuals), compare them daily.

```python
# scripts/monitor.py
import pandas as pd, pathlib, glob

# Load all saved forecasts
forecast_files = sorted(glob.glob("data/forecasts/forecast_*.csv"))
forecasts = pd.concat([pd.read_csv(f) for f in forecast_files])
forecasts["timestamp_utc"] = pd.to_datetime(forecasts["timestamp_utc"], utc=True)
forecasts = forecasts.set_index("timestamp_utc")

# Load actuals
actuals = pd.read_parquet("data/raw/smard.parquet")[["price_eur_mwh"]]

merged = forecasts.join(actuals, how="inner")
merged["error"] = (merged["price_eur_mwh"] - merged["forecast_eur_mwh"]).abs()

# Rolling 7-day MAE
rolling_mae = merged["error"].resample("D").mean().rolling(7).mean()
rolling_mae.plot(title="7-day rolling MAE (€/MWh)")

# Alert if MAE drifts above 2× historical baseline
baseline_mae = 15.0   # fill in from your backtest results
if rolling_mae.iloc[-1] > 2 * baseline_mae:
    print("⚠️  MAE drift alert: model may need retraining")
```

---

## 13. Regime Analysis

Three experiments to run, each with a separate backtest call on the same test period.

| Experiment | Training data | Purpose |
|---|---|---|
| All data | 2019 – (test_start − 1 day) | Default model |
| Exclude 2022 | All but 2022-02-01 – 2023-06-01 | Does 2022 hurt or help? |
| Sliding window | Only last 2 years | Adapts to recent market structure |
| Expanding window | Always grows from 2019 | Retains all history |

```python
# Exclude 2022 experiment
crisis_start = pd.Timestamp("2022-02-01", tz="UTC")
crisis_end   = pd.Timestamp("2023-06-01", tz="UTC")

def make_lgbm_no_crisis():
    class LGBMNoCrisis:
        def __init__(self): self.m = LGBMRegressor(n_estimators=1000, verbose=-1)
        def fit(self, X, y):
            mask = ~((X.index >= crisis_start) & (X.index <= crisis_end))
            self.m.fit(X[mask], y[mask])
            return self
        def predict(self, X): return self.m.predict(X)
    return LGBMNoCrisis()

result_no_crisis = rolling_backtest(X_full, make_lgbm_no_crisis, TEST_START, TEST_END)
```

---

## 14. Results Section for README

After running all experiments, fill in the actual numbers:

```markdown
## Results

Data source: Bundesnetzagentur | SMARD.de. Test period: Oct 2025 – Sep 2026 (12 months).

| Model | MAE (€/MWh) | RMSE (€/MWh) | vs Naive |
|---|---|---|---|
| SeasonalNaive (lag 168h) | ___ | ___ | baseline |
| LEAR (lasso, no weather) | ___ | ___ | ___ % |
| LightGBM (no weather) | ___ | ___ | ___ % |
| **LightGBM (full)** | **___** | **___** | **___ %** |
| LightGBM oracle (obs. weather) | ___ | ___ | — |

80% prediction interval coverage: ___% (target: ~80%)

Error by hour of day: peaks at hours 7–9 (morning ramp) and 18–20 (evening peak).
Error weekday vs weekend: weekday MAE ___ €/MWh, weekend ___ €/MWh.
```

Chart to include (one per section):
1. Weekly average price 2019–today, with 2022 crisis shaded
2. Forecast vs actual for one week, with 80% interval band
3. Calibration plot (nominal vs actual quantile coverage)
4. MAE by hour of day (bar chart, all models overlaid)

---

## 15. CV Bullet (Fill After Experiments)

```
Cut German day-ahead electricity price forecast error by X% vs seasonal-naive baseline
(MAE Y €/MWh) on 12 months of live market data by building a leakage-safe LightGBM pipeline
with Open-Meteo weather forecasts, calibrated quantile intervals (80% coverage), and a
rolling-origin backtest — served via FastAPI with daily GitHub Actions retraining.
```

X = percentage improvement over naive. Y = your final test-period MAE. Fill in real numbers only.

---

## 16. Execution Timeline

| Week | Hours | What to complete |
|---|---|---|
| 1 | 15 h | Steps 2–5: download data, EDA notebook, data quality checks, baseline table |
| 2 | 15 h | Steps 6–7: LEAR, LightGBM, ablation study, oracle experiment, results table |
| 3 | 15 h | Steps 8–9, 13: quantile regression, calibration plot, regime analysis, write-up |
| 4 | 15 h | Steps 11–14: daily CLI, FastAPI, Docker, GitHub Actions CI, monitoring, README results section |

**Minimum viable portfolio piece (if time is short):** Steps 1–3 + 5–6 + 10 + top chart. That's a working model with a real results table and one good chart — enough for an interview conversation.

---

## 17. What to Say in Interviews

**"What's the hardest part of this project?"**
> Leakage prevention. In electricity markets, you only know tomorrow's weather forecasts — not the actual weather. I used the Open-Meteo historical-forecast API, which stores forecasts as they were issued, not the reanalysis endpoint. I enforced minimum 48-hour lags on all market features and wrote an automated test that shocks a future price and verifies past features are unchanged. I also ran an oracle experiment with observed weather to quantify the forecast-information gap — about 4 €/MWh — which is the ceiling on improvement from better weather models.

**"How did you handle the 2022 price spikes?"**
> I made a deliberate choice: trained through 2022 with no down-weighting. My regime analysis showed that excluding 2022 actually increased test-period MAE by X €/MWh because the model underestimated tail risk on cold winter days. I report metrics separately for 2022 hours in the backtest to be transparent about where the model fails.

**"Why LightGBM over a neural network?"**
> On 55,000 hourly rows, LightGBM trains in under 2 minutes and produces a result table with a 4-week backtest in 30 minutes. A TFT (Temporal Fusion Transformer) might gain 1–2 €/MWh MAE but would take 10× longer to tune and is harder to explain to a trading desk. I added a stretch experiment with N-HiTS to check — it did not beat LightGBM on this dataset size.

---

*Data source for all market data: Bundesnetzagentur | SMARD.de*
