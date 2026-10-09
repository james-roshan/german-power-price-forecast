# Repository assessment and implementation plan

Inspected on 2026-10-09 before architecture changes. The clean local `main` branch
has origin `https://github.com/james-roshan/german-power-price-forecast.git`.
Remote synchronization and repository visibility have not been verified.

The target is hourly DE-LU wholesale day-ahead price, EUR/MWh, for the entire next
Berlin delivery day (23/24/25 hours), issued before 11:00 on D-1. Preserve this
hourly representation despite quarter-hour market products. The selected model
is LightGBM with the original 250-tree L1 parameters, 730-day window and daily
refitting. Selection metadata in `data/processed/pipeline/selection_record.json`
contains 57 features. Existing joblib is a historical replay artifact, not a
production model. Daily refitting is already the selected methodology and is
necessary to establish a current artifact; no new architecture or tuning is needed.

Reuse `depower.features`, `live`, `metrics`, data definitions, model factories,
downloaders, chronological CV and all experiments. Existing tests cover leakage,
DST, metrics, covariate availability and replay equivalence. Preserve both notebooks
(35 and 85 cells), rendered walkthrough, experiments and retrospective results.

SMARD provides price, load, residual load and three generation series. Open-Meteo
provides five-site ECMWF IFS weather at fixed 48-hour lead. Calendar, market lags
(48/72/168/336 hours), lagged rolling statistics and price-only 24-hour lag are
reused. Cloud gaps are native LightGBM missing values; other missing features fail.
Raw snapshots end in September 2026. Historical revisions/publication times are
not reconstructible; training history remains retrospective. Future live inputs
must actually be received before cutoff, future target labels removed, and
publication completion checked again after writing predictions. Inspection also
found that t-24 on the final hour of an autumn 25-hour day can belong to the
delivery day itself; the correction uses t-25 for that one hour. Preserve original
benchmark outputs but do not assert they describe the corrected implementation. Negative prices
are valid. No replay may enter the production ledger.

Plan: harden capture/schema validation and cutoff enforcement; add a deployable
CLI around the existing live cycle; make JSON forecast records authoritative;
persist CSV histories on a separate Git branch with serialized workflow writes;
add matched-pair monitoring, drift diagnostics, five-view Streamlit dashboard,
controlled dependencies, CI/schedule, realistic replay verification and runbooks.
Use GitHub failures for alerts. Avoid a database account/dependency for this small
daily workload. Keep model and raw capture files as short-lived diagnostic artifacts;
refit daily from persistent history. Dashboard reads only a compact JSON export.

No pushes, visibility changes or public hosting without explicit approval.
