"""Project-wide constants: experiment definition, paths and default model parameters."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "price_eur_mwh"
TZ = "Europe/Berlin"
SEED = 42
# Hours between the value at t-lag and delivery hour t that are guaranteed to be known.
SAFE_LAGS_H = [48, 72, 168, 336]

# Frozen retrospective test period (Berlin delivery days, inclusive).
TEST_START, TEST_END = "2025-09-25", "2026-09-24"
# Non-overlapping 28-day development folds, all before the test period.
FOLD_STARTS = ["2024-11-01", "2025-02-01", "2025-05-01", "2025-08-01"]
FOLD_DAYS = 28

TRAIN_WINDOW_DAYS = 730
RETRAIN_EVERY_DAYS = 7

# Original, fixed (untuned) LightGBM parameters.
MODEL_PARAMS: dict = dict(
    n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=40,
    reg_lambda=1.0, objective="regression_l1", random_state=SEED, n_jobs=2, verbosity=-1,
)

# SMARD filter ids: name -> (filter id, region).
SMARD_SERIES = {
    TARGET: (4169, "DE-LU"),
    "load_mwh": (410, "DE"),
    "residual_load_mwh": (4359, "DE"),
    "solar_mwh": (4068, "DE"),
    "wind_onshore_mwh": (4067, "DE"),
    "wind_offshore_mwh": (1225, "DE"),
}
LAG_COLUMNS = {TARGET, "load_mwh", "residual_load_mwh", "solar_mwh",
               "wind_onshore_mwh", "wind_offshore_mwh"}


def find_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) to the first folder that contains data/raw."""
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "data/raw").is_dir():
            return p
    raise FileNotFoundError("Could not find a data/raw folder; pass --root explicitly.")


def evaluation_window(test_start: str = TEST_START, test_end: str = TEST_END):
    """Return (begin_utc, stop_utc_exclusive, expected hourly UTC index) for the test period."""
    begin = pd.Timestamp(test_start, tz=TZ).tz_convert("UTC")
    stop = (pd.Timestamp(test_end, tz=TZ) + pd.DateOffset(days=1)).tz_convert("UTC")
    return begin, stop, pd.date_range(begin, stop, freq="h", inclusive="left")
