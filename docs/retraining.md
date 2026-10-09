# Refitting, candidate selection and rollback

Daily fitting on the selected 730-day window is the existing model's methodology,
not automatic promotion of newly tuned hyperparameters. Each issued forecast
records the configuration hash, model SHA-256, training interval, input/feature
hashes and run ID. Model binaries remain in seven-day diagnostic artifacts.

Candidate retraining is intentionally manual; no costly scheduled search is added.
Reuse `depower run` / `depower.selection` with dated, non-overlapping chronological
development folds. Adapt evaluation dates for a genuinely untouched future period;
the existing 2025–2026 test has already been inspected. Refit preprocessing inside
each training window. Never feed archive/reanalysis weather into candidate inference.
Record code commit, exact dependencies, source hashes, fold cutoffs and candidate
selection JSON alongside a trusted, local fitted model. The first production model
is fitted by the same existing factory because the supplied model is a replay artifact.

Acceptance proposal before promotion: complete 23/24/25-hour delivery coverage,
no leakage or missing required features, all CI checks pass, at least eight weeks
of untouched rolling-origin/shadow evaluation, MAE no worse than current production,
RMSE no more than 5% worse, and competitive MAE against weekly/previous-day naive
on identical pairs. Use weekly block-bootstrap intervals for paired improvements;
reject a candidate when uncertainty or regime regressions are operationally material.
Do not compare on different available-hour subsets. Inspect negative-price hours,
spikes, hour-of-day bias and run-time/free-tier impact. These are proposed acceptance
criteria, not claims that any new candidate has passed them.

Review changes to `production/selection.json` in a PR. Archive the former selection
and diagnostic model, then require explicit human promotion by merging/publishing
the reviewed configuration. Successful training alone never promotes a candidate.
Rollback by reverting that selection commit; the next daily run fits the previous
factory configuration on current eligible history. Restore a byte-identical prior
model only from a trusted diagnostic artifact with its matching dependency versions
and checksum. Previously issued forecasts must never be rewritten or re-scored as
if the rollback model had made them.

The autumn DST correction invalidates any claim of exact equivalence with the
old benchmark on all days. Keep old notebooks/results as historical evidence,
and recompute a corrected benchmark before publishing revised accuracy claims.
