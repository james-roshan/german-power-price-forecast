# Improvement plan

Updated: 2026-10-09. This document plans future work; unchecked items are not
implemented or verified. Keep the existing hourly next-day target, selected
LightGBM methodology, leakage controls and €0 recurring-cost objective.

Public dashboard: https://hpu4hyd22wxk8puvkmhww7.streamlit.app/

## Current automation and timing

The configured schedule uses **Europe/Berlin**, including daylight saving time:

| Time | Operation |
|---|---|
| 09:17 daily | First attempt to issue the next delivery day's hourly forecast |
| 10:17 daily | Retry if no forecast exists; otherwise refresh monitoring without replacing it |
| 11:00 | Hard issue cutoff, not a scheduled execution time |
| 18:17 daily | Refresh input data and score completed delivery intervals |

First eligible scheduled attempt after deployment: **10 October 2026 at 09:17**,
predicting the Berlin delivery day **11 October 2026**. GitHub scheduling is
best-effort: an attempt is not a promise of a successful or precisely timed run.
The workflow must finish recording the forecast before the cutoff.

Every forecasting attempt automatically:

1. Restores persistent market/weather history and previously issued forecasts.
2. Fetches recent SMARD prices, load and generation series.
3. Fetches weather forecasts from Open-Meteo for all five configured sites, with
   coverage through the delivery day. It joins these by UTC target timestamp.
4. Validates timestamps, schemas, units, missingness and duplicates.
5. Builds the original features and refits the selected LightGBM on eligible
   history using its existing daily-refit policy and 730-day window.
6. Predicts and records immutable hourly prices with issue time, model/configuration
   version, input/feature hashes and run identifier.
7. Matches older forecasts to completed, observed intervals and updates monitoring.
8. Commits persistent CSV/JSON state to `forecast-data`; Streamlit reads that export.

Weather requests are newly fetched on each run, but preserve the model's
**fixed 48-hour lead** (`previous_day2`) and ECMWF IFS model. They are not an
automatic switch to the newest available D-1 weather run. That change would
require a new feature/training specification and comparative evaluation.
Forecast weather is used; observed/reanalysis weather is not substituted.

Normal operation needs no daily upload, manual weather attachment, model button
click or local computer left running. GitHub Actions executes remotely. Streamlit
does not train the model. Refreshing an open dashboard reruns its read; the data
cache lasts five minutes. An unattended open tab is not currently guaranteed to
refresh itself. Retries, empty/stale data and missing required inputs can withhold
a forecast. Human work remains necessary for outages, account/policy changes,
source schema changes, candidate promotion and periodic storage maintenance.

## Priority 0 — Verify the first genuine live forecast

**Do this before tuning or expanding the application.**

- [ ] Inspect the first morning workflow and confirm ingestion, fitting, inference,
  persistence and the data-branch push all complete successfully.
- [ ] Confirm `forecasts/YYYY-MM-DD.json` exists for the intended delivery day.
- [ ] Verify issue/persistence is before 11:00 Berlin and before target prices
  become known; target timestamps cover exactly 23/24/25 delivery hours.
- [ ] Check finite outputs, valid negative prices, source/feature hashes and model
  training metadata. Never replace failure with fabricated or retrospective prices.
- [ ] Confirm the public app displays the same forecast after refreshing.
- [ ] Verify the 10:17 run leaves the issued bundle unchanged.
- [ ] After delivery, verify matching, MAE/RMSE/bias and naive baselines on identical
  UTC intervals. Inspect partial versus complete delivery-day coverage.
- [ ] Reboot the hosted app and confirm the persisted forecast is recovered.

Acceptance: one authentic persisted forecast, a duplicate-run check, hosted
restart recovery, and a subsequently scored delivery day. Preserve run links and
diagnostic evidence in `docs/verification.md`. Full live performance remains
unproven until enough prospective outcomes exist.

## Priority 1 — Make operational state clear

Files: `src/depower/production.py`, `monitoring.py`, `app.py`, production tests.

- [ ] Separate execution result from monitoring alerts. Represent successful
  ingestion with no first forecast as `awaiting_first_forecast`, rather than a
  generic execution crash. Preserve nonzero failure/notification behavior for
  actual exceptions and overdue forecasts after the expected issue window.
- [ ] Export a structured last-run status: phase, started/finished times, mode,
  outcome, safe error summary and GitHub run URL.
- [ ] Display the next scheduled attempt and cutoff in Berlin time; show UTC
  target timestamps alongside a Berlin delivery-time display where useful.
- [ ] Distinguish issue freshness, observation freshness and export freshness.
- [ ] Show coverage counts and the model configuration versus fitted-artifact
  version separately. Label retrospective results explicitly.
- [ ] Enable owner GitHub failed-run notifications and document incident handling.

Acceptance: users can distinguish an expected startup state, missing inputs,
late run, model failure, stale export and missing observations. Tests cover these
states without suppressing genuine failures or rewriting any issued forecast.

## Priority 2 — Improve dashboard use and freshness

Files: `app.py`, `.streamlit/config.toml`, `tests/test_production.py`.

- [ ] Add a refresh button that clears the data cache and reruns the read.
- [ ] Consider bounded automatic refresh while the app is open; verify that it
  does not trigger training or excessive GitHub/raw-data requests.
- [ ] Add delivery-day and timezone selection, readable issue/target tooltips,
  negative-price highlighting and forecast downloads containing provenance.
- [ ] Show matched-hour counts next to metrics, comparable baseline scores,
  rolling-window coverage and clear empty/sparse-data states.
- [ ] Improve small-screen layout and accessibility; validate contrast and chart
  labels. Capture screenshots only once real forecasts are available.
- [ ] Offer a short methodology summary and links to detailed source/limitations.

Acceptance: all five views pass smoke tests with empty, populated, stale, failed
and DST fixtures. Charts/downloads agree with persisted values; refresh remains
read-only and costs stay within the existing service limits.

## Priority 3 — Strengthen reliability and provenance

Files: `live.py`, `production.py`, `validation.py`, workflow and deployment docs.

- [ ] Add tests for failure between local forecast persistence and remote push;
  document recovery from diagnostics without changing the original issue time.
- [ ] Audit interrupted writes and state restoration; stage multi-file updates
  before publication and preserve the last valid export on failures.
- [ ] Separate required historical feature availability from incidental missing
  current-hour source values; fail precisely on inputs actually needed for inference.
- [ ] Measure API/run durations, retry behavior and publication delays under live
  operation. Add source revision/receipt metadata to evaluation snapshots.
- [ ] Add a true runner-to-dashboard integration check against persisted bundles.
- [ ] Review action-version pinning and dependency lock options for reproducibility.
- [ ] Evaluate CI protection for main without disrupting the data-branch bot.

Acceptance: simulated API timeouts, conflicting revisions, missing features,
cutoff crossings and push failures produce auditable outcomes. Immutable records
survive retries. No source failure silently fabricates data or hides missingness.

## Priority 4 — Establish trustworthy live evaluation

Files: monitoring, research features/backtests, original notebooks and documentation.

- [ ] Accumulate at least several weeks of prospective matched results before
  presenting live accuracy claims or selecting a replacement model.
- [ ] Recompute the historical benchmark after the autumn DST lag correction;
  preserve old outputs and explain why original numbers differ.
- [ ] Audit teaching notebooks for the same DST risk without erasing experiments.
- [ ] Report weekly/previous-day baseline performance on identical forecast hours,
  hour-of-day bias, negative-price periods, spikes and sample counts.
- [ ] Review threshold alerts against observed seasonal variability; document
  decisions before changing thresholds.
- [ ] Distinguish input/prediction distribution change from observed performance
  degradation. Keep descriptive drift limitations visible.

Acceptance: results are reproducible from issued bundles and dated observations;
live and retrospective evidence remain separate. Revised benchmark figures have
their own provenance and do not masquerade as untouched holdout performance.

## Priority 5 — Evaluate model and feature improvements

Dependent on Priority 4; follow `docs/retraining.md`.

- [ ] Evaluate publication-versioned demand/renewable forecasts and additional
  weather sites or capacity-weighted regional weather, with a free source audit.
- [ ] Evaluate more recent weather runs only with matching historical run data,
  explicit availability timestamps and a new reviewed configuration version.
- [ ] Revisit calibrated uncertainty intervals; existing nominal 80% intervals
  under-covered and should not be presented as reliable until calibrated/tested.
- [ ] Compare candidates with chronological windows, common eligible hours,
  regime checks and weekly block-bootstrap uncertainty.
- [ ] Add an optional manual candidate-training workflow if measured runtime fits
  the free-tier budget. Keep training and human promotion separate.

Acceptance: a candidate passes predeclared acceptance criteria, leakage tests,
schema checks and runtime/cost constraints. Human review promotes the selection;
successful training alone never replaces production. Rollback restores the former
configuration while preserving all earlier issued forecasts.

## Priority 6 — Maintain €0 operations and portfolio quality

- [ ] Review quarterly: runner minutes, API calls, artifact retention and Git size.
- [ ] Keep seven-day diagnostics and the bounded histories/export; verify backups
  before any annual data-branch archival or reinitialization.
- [ ] Document the lack of guaranteed scheduling/API uptime and private-repository
  quotas if repository visibility ever changes.
- [ ] Optionally add Docker packaging once the live pipeline is stable.
- [ ] Add screenshots, a concise case study and real live metrics to the portfolio.
- [ ] Add MLflow, a database or larger hosting only if measured needs justify it;
  do not introduce recurring paid services as a cosmetic improvement.

Acceptance: restart/recovery instructions work, complete forecast evidence is backed
up, expected usage remains inside free limits, and documentation matches reality.

## Suggested rollout order

1. Next eligible morning: Priority 0 issuance and duplicate-run verification.
2. After delivery: first live scoring and public restart check.
3. First week: Priorities 1–2, then targeted reliability improvements from Priority 3.
4. After several weeks: Priority 4 evaluation, followed by justified Priority 5 work.
5. Quarterly: Priority 6 operations review; portfolio documentation as evidence grows.

This is a sequence of acceptance gates, not a promise that evaluation or deployment
will succeed by a particular date. Implement each change in a small reviewed
commit, run relevant tests, verify CI and check the public dashboard afterward.

## Operational links

- Actions: https://github.com/james-roshan/german-power-price-forecast/actions
- Workflow: `.github/workflows/forecast.yml`
- Deployment/recovery: `docs/deployment.md`
- Verification evidence: `docs/verification.md`
- Retraining/promotion: `docs/retraining.md`
