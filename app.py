"""Read-only portfolio dashboard: production evidence is never manufactured."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="German Electricity Price Forecasting", page_icon="⚡", layout="wide")


@st.cache_data(ttl=300)
def load_dashboard() -> dict:
    local = Path(os.environ.get("DEPOWER_DASHBOARD_FILE", "data/state/dashboard.json"))
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    url = os.environ.get("DEPOWER_DATA_URL", "")
    if not url:
        try:
            url = st.secrets.get("data_url", "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass
    if not url:
        return {}
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    document = response.json()
    if not isinstance(document, dict) or "updated_at_utc" not in document:
        raise ValueError("Invalid dashboard export")
    return document


def table(rows):
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["target_time_utc"] = pd.to_datetime(frame.target_time_utc, utc=True)
    return frame


def chart(frame, columns, title):
    if frame.empty:
        st.info("No live data available for this view yet.")
        return
    st.plotly_chart(
        px.line(
            frame,
            x="target_time_utc",
            y=columns,
            title=title,
            labels={"value": "EUR/MWh", "target_time_utc": "Delivery interval (UTC)", "variable": "Series"},
        ),
        width="stretch",
    )


st.title("German Electricity Price Forecasting")
st.caption("DE-LU day-ahead wholesale prices · Hourly EUR/MWh · Issued before 11:00 Europe/Berlin")
view = st.sidebar.radio(
    "Explore",
    [
        "Forecast overview",
        "Predicted vs actual",
        "Model performance",
        "Data and model monitoring",
        "Architecture and methodology",
    ],
)
try:
    data = load_dashboard()
except (requests.RequestException, ValueError) as exc:
    st.error(f"Forecast data unavailable: {exc}")
    data = {}
latest = data.get("latest")
now = pd.Timestamp.now(tz="UTC")
if not latest:
    st.warning("No verified live forecast has been published. Historical experiments are not live results.")
else:
    age = now - pd.Timestamp(latest["issued_at_utc"])
    if age > pd.Timedelta(hours=36):
        st.warning(f"Forecast is stale: last issued {latest['issued_at_utc']}.")
    else:
        st.success(f"Latest successful issue: {latest['issued_at_utc']} · Delivery: {latest['delivery_day']}")
if data.get("updated_at_utc"):
    updated = pd.Timestamp(data["updated_at_utc"])
    st.caption(f"Monitoring refreshed: {updated.isoformat()} · Age: {now - updated}")
    if now - updated > pd.Timedelta(hours=36):
        st.warning("Monitoring export is stale; current pipeline health is unknown.")
pairs = table(data.get("comparison", []))
if view == "Forecast overview":
    a, b, c = st.columns(3)
    a.metric("Horizon", "Next delivery day")
    b.metric("Model", latest.get("configuration", {}).get("selected_name", "LightGBM") if latest else "LightGBM")
    c.metric("Version", latest["model_state"]["config_hash"][:12] if latest else "Awaiting first issue")
    chart(table(data.get("forecasts", [])), ["prediction"], "Next delivery day forecast")
    chart(table(data.get("observations", [])), ["actual"], "Latest observed prices")
elif view == "Predicted vs actual":
    if not pairs.empty:
        start = st.date_input("From", pairs.target_time_utc.min().date())
        end = st.date_input("Through", pairs.target_time_utc.max().date())
        pairs = pairs.loc[(pairs.target_time_utc.dt.date >= start) & (pairs.target_time_utc.dt.date <= end)]
        chart(
            pairs,
            ["prediction", "actual", "weekly_baseline", "previous_day_baseline"],
            "Previously issued forecasts vs actuals",
        )
        chart(pairs, ["error"], "Signed forecast error (prediction − actual)")
        st.download_button("Download matched pairs", pairs.to_csv(index=False), "comparison.csv", "text/csv")
    else:
        st.info("Live outcomes appear after forecast delivery intervals have ended and observations are available.")
elif view == "Model performance":
    metrics = data.get("metrics", {})
    cols = st.columns(3)
    for col, key in zip(cols, ["MAE", "RMSE", "bias"]):
        # Existing regression_metrics names the signed mean error Bias.
        value = metrics.get(key, metrics.get("Bias")) if key == "bias" else metrics.get(key)
        col.metric(key.upper(), f"{value:.2f} EUR/MWh" if value is not None else "Pending")
    if not pairs.empty:
        chart(pairs, ["rolling_7d_mae", "rolling_30d_mae"], "Rolling live error (at least 24 matched hours)")
        st.plotly_chart(px.histogram(pairs, x="error", nbins=40, title="Error distribution"), width="stretch")
        pairs["Berlin hour"] = pairs.target_time_utc.dt.tz_convert("Europe/Berlin").dt.hour
        st.bar_chart(pairs.groupby("Berlin hour").absolute_error.mean())
        st.dataframe(metrics)
    st.caption("Metrics use only completed UTC intervals with matched, previously issued live forecasts. No MAPE.")
elif view == "Data and model monitoring":
    for alert in data.get("alerts", []):
        st.warning(alert)
    st.write("Missing market input values in latest 168 rows", data.get("input_missing", {}))
    st.write("Model configuration versions", data.get("model_versions", []))
    st.write("Distribution diagnostics", data.get("drift", {}))
    st.caption(
        "Mean shifts are descriptive, not statistical proof of concept drift. Seasonality, autocorrelation "
        "and 23/25-hour days affect comparisons. Missing forecasts and failed runs also trigger GitHub notifications."
    )
else:
    st.image(str(Path(__file__).resolve().parent / "docs/architecture.svg"), width="stretch")
    st.markdown("""
### Why forecast wholesale electricity prices?
Day-ahead price forecasts support energy procurement and operational planning. This portfolio
application predicts the full next Berlin delivery day, at hourly resolution, including negative prices.

### Existing forecasting model
Selected LightGBM: 250 trees, L1 loss, 730-day rolling training window, daily refit.
Calendar and German holidays; 48/72/168/336-hour market lags; lagged rolling prices;
previous-day prices; five-site ECMWF IFS weather forecasts at fixed 48-hour lead.
On the final hour of an autumn 25-hour day the otherwise unsafe 24-hour price lag uses 25 hours.

### Evaluation and limitations
Chronological development folds and rolling-origin evaluation are preserved. The previously inspected
2025–2026 benchmark is retrospective, and weather publication times and market revisions were not
reconstructed. It is separate from this live ledger. Quantile intervals were under-calibrated and
are not displayed as reliable uncertainty bounds. Five weather sites are coarse proxies.

### Operation
GitHub Actions captures source responses, validates schemas and units, fits the selected model,
issues immutable forecasts before the cutoff, matches later observations, exports monitoring,
and persists text histories on a separate data branch. Streamlit reads JSON with a five-minute cache.
Failures and threshold alerts use GitHub workflow notifications. No paid services or dashboard training.
""")
    st.code(
        "SMARD + Open-Meteo → validated UTC inputs → existing features → daily LightGBM fit\n"
        "→ immutable forecast ledger → matched observations → monitoring JSON → Streamlit",
        language="text",
    )
    st.markdown("[Methodology and deployment](https://github.com/james-roshan/german-power-price-forecast)")
st.caption("Sources: Bundesnetzagentur | SMARD.de; Open-Meteo and ECMWF. Non-commercial portfolio demonstration.")
