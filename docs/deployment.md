# Deployment runbook

Status updated 2026-10-09: the owner approved publishing. Code is committed and
pushed to `main` (initial implementation `4e36c1e`), and `forecast-data` was created
with source histories and an observations-only export. GitHub authentication is
verified and the repository was already public; visibility was not changed.
Remote CI passed on Linux. The remote monitor fetched and persisted observations
and uploaded diagnostics; it exited with the expected missing-live-forecast alert.
The published JSON was verified to return HTTP 200. The corrected dashboard is
deployed at https://hpu4hyd22wxk8puvkmhww7.streamlit.app/ . Public metadata confirms
`app.py`, Python 3.12, and Streamlit 1.65; runtime health returns `ok`. The owner
confirms the dashboard renders. The first genuine pre-cutoff forecast and hosted
restart recovery still need verification.

### Resolved Streamlit configuration issue

The owner supplied
`https://german-power-price-forecast-fuiwd7mymph2hfbf4cvpp9.streamlit.app/`.
Read-only public hosting metadata confirms Python 3.12, but the deployed main
module is **`src/depower/__init__.py`**, not **`app.py`**. The initializer only
contains a package docstring, so a running Streamlit process displays a blank
page. This mistaken deployment was replaced with the corrected `app.py` deployment
at https://hpu4hyd22wxk8puvkmhww7.streamlit.app/ . Preserve this diagnosis for future
troubleshooting; the old blank app is not the portfolio dashboard.

Create a corrected deployment using this precise GitHub file URL:

`https://github.com/james-roshan/german-power-price-forecast/blob/main/app.py`

Before clicking Deploy, verify repository, `main`, **`app.py`**, Python 3.12 and
the `data_url` setting below. A new URL may be assigned; verify that app before
considering removal of the mistaken deployment. No Python package change is
needed to work around an incorrect hosting entrypoint.

## Local installation and first run

Use Python **3.12** on Windows or Linux:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m streamlit run app.py
```

If the Python environment was created by uv and has no pip, use
`uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt`.
On Linux replace `.venv/Scripts/python.exe` with `.venv/bin/python`.
Direct dependencies are pinned to the versions verified locally. Linux installation
has passed in GitHub CI; transitive dependencies are
resolved by the installer. The research package's broader constraints remain usable.

This checkout already has **data/state** with ~3.4 MB of seeded/refreshed CSV
history and a dashboard export containing observations, with no live forecasts.
Do not run `seed` over existing state. For a new clone without raw data, run the
existing downloaders once (start ~760 days before today), then:

```bash
python -m depower.production seed --state data/state
python -m depower.production forecast --state data/state --runtime data/runtime-first
```

Run forecast only before **11:00 Europe/Berlin**. A realistic target needs all
required lag/weather inputs. Missing data or already published delivery-day prices
cause failure. A monitor-only run is permitted at any hour:

```bash
python -m depower.production monitor --state data/state --runtime data/runtime-monitor
```

Use a different empty runtime directory on every invocation. An existing forecast
for tomorrow is restored and monitored without issuing another forecast. No seed
history/replay is promoted into the live ledger. A monitor run before the first
forecast exits 1 for the expected missing-forecast alert while persisting observations.

## Publish code and initialize persistent state (approval required)

Review `git diff` and `git status`, then commit the code and push `main` only after
approval. This bootstrap was performed with approval on 2026-10-09; do not repeat
initialization against the existing `forecast-data` branch. If Git reports
dubious ownership, use the one-command option
`git -c safe.directory=C:/Users/rosha/OneDrive/Documents/de-power-price-forecast ...`;
there is no need to change global Git configuration.

After the approved code push, create the dedicated branch from the local seed using
a separate staging repository. These PowerShell commands avoid altering main:

```powershell
New-Item -ItemType Directory -Path data/publish-state
Copy-Item -Path data/state/* -Destination data/publish-state -Recurse
git -C data/publish-state init -b forecast-data
git -C data/publish-state config user.name "YOUR NAME"
git -C data/publish-state config user.email "YOUR GITHUB EMAIL"
git -C data/publish-state add .
git -C data/publish-state commit -m "Initialize forecast history; no live forecasts"
git -C data/publish-state remote add origin https://github.com/james-roshan/german-power-price-forecast.git
git -C data/publish-state push -u origin forecast-data
```

Do not run this against an existing remote `forecast-data` branch; inspect and reuse
its ledger instead. Publishing the seed data also makes it available under source
licences; preserve attribution to Bundesnetzagentur | SMARD.de and Open-Meteo/ECMWF.

## GitHub settings and verification

1. Confirm desired repository visibility yourself; this task does not change it.
2. Settings → Actions → General: permit the supplied checkout/setup/upload actions.
   The forecast job requests `contents: write`; CI requests read-only access.
   Repository or organization policy may restrict the job token; never work around
   that with a broad personal access token.
3. Ensure `forecast-data` permits the bot's normal fast-forward push. Protect main
   with a rule requiring the CI `test` check, where the GitHub plan permits it.
4. Actions → Production forecast and monitoring → Run workflow → `main` → `monitor`.
   Before the first issue, the missing forecast alert is expected; confirm actual
   observations and `dashboard.json` exist on the data branch.
5. Before 11:00 Berlin, manually run with `forecast`. Verify the immutable bundle in
   `forecasts/YYYY-MM-DD.json`, issue timestamp, hashes, 23/24/25 targets, absence of
   delivery-day actual labels, successful push and uploaded diagnostic artifact.
6. Run again: verify the same forecast bundle remains unchanged. After delivery,
   run monitor and verify matched actuals and scores. Turn on GitHub failed-workflow
   notifications in your account notification preferences.

Schedule: **09:17 and 10:17 Berlin** (second is retry/monitor), plus **18:17 Berlin**
monitoring, every day. The IANA timezone schedule follows DST. GitHub schedules
are best-effort and may be delayed/dropped. Runs that reach cutoff withhold forecasts.
Public repositories inactive for 60 days may have schedules disabled; inspect
Actions regularly. Data branch commits do not trigger main CI or commit loops.
Concurrency serializes all state updates; non-fast-forward pushes fail and require
inspection, never force pushes or automatic conflict resolution.

## Streamlit Community Cloud (owner authentication required)

1. Sign in at https://share.streamlit.io and connect the approved GitHub repository.
2. Create app: repository `james-roshan/german-power-price-forecast`, branch `main`,
   entrypoint **app.py**. Advanced settings: Python **3.12**.
3. Add this non-secret configuration in Advanced settings → Secrets:

```toml
data_url = "https://raw.githubusercontent.com/james-roshan/german-power-price-forecast/forecast-data/dashboard.json"
```

4. Deploy only after approving public access. For this unauthenticated data URL,
   the data branch must be publicly readable. If the repo is private, stop and decide
   how to expose a sanitized data export; do not embed a GitHub token in a public URL.
5. Verify all five views, negative prices, staleness alerts, comparisons/downloads
   and error metrics. Restart the app and confirm it reloads the same data branch.
6. Only then add the actual verified app URL to README. No example URL represents
   an existing deployed service.

The app does not write local state, train or unpickle a remote model. It downloads
JSON with a five-minute cache. Raw branch URLs reflect subsequent data commits;
app restarts recover from Git history rather than ephemeral Streamlit storage.

## Monitoring and incident handling

MAE, RMSE and bias compare completed hourly UTC intervals with immutable live
forecasts. Baselines use exactly the same pairs. Rolling windows use elapsed
7/30-day intervals and at least 24 matched hours; counts may differ around DST or
missing observations. Published day-ahead price actuals are conservatively scored
only after the delivery interval ends. Snapshots retain their evaluation timestamp;
later source revisions can change a subsequent evaluation, never original forecasts.

Failure thresholds: observations >48 hours old; successful issue >36 hours old;
7-day MAE >1.5x weekly baseline (minimum 1 EUR/MWh) with at least 168 pairs;
7-day MAE >1.5x preceding 30-day MAE with at least 168 reference pairs. Feature
mean shift >3 historical standard deviations is a diagnostic alert, not proof of
drift. Prediction shifts are reported once at least 336 pairs exist. Seasonality,
autocorrelation and small samples require human interpretation.

Failed jobs retain capture/model files for seven days and return non-zero.
If a push fails, inspect and recover the issued record from diagnostics before the
next run; never invent a replacement forecast timestamp. Source outages lead to
missing forecasts and visible alerts. Check expected units/schema before retrying.

## Cost and storage

Target recurring cost **€0** for a modest non-commercial portfolio on a public
repository: standard GitHub-hosted Linux runners are free for public repositories;
Streamlit Community Cloud is free; Open-Meteo permits limited non-commercial API
access. There is no SLA. Private repositories have runner quotas; inspect usage
and set a spending limit of zero. API counting can weight long historical requests.

Normal runs use six SMARD indexes plus overlapping weekly chunks and five weather
requests per cycle. Weather history refresh is incremental with seven-day overlap;
bootstrap is a one-time download. Histories retain 760 days, approximately 3.4 MB
of text in this checkout. Git can delta-compress CSV updates. Model binaries and
full captures are not committed every day; seven-day diagnostic retention avoids
unbounded artifact storage. Forecast bundles are small; the dashboard retains at
most one year of hourly comparison data and monitoring history 1,500 snapshots.
Git history still grows: review size quarterly and archive/reinitialize the data
branch annually only after backing up the complete immutable ledger. Long-term
storage/free-tier sustainability must be measured; no unlimited-storage claim.

Official references (checked 2026-10-09):

- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- https://docs.github.com/en/actions/concepts/billing-and-usage
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://open-meteo.com/en/docs/previous-runs-api
- https://open-meteo.com/en/pricing
