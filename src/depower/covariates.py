"""As-of selection of forecast covariates that carry explicit availability timestamps."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"valid_time", "available_at", "feature", "value"}


def select_available_covariates(revisions: pd.DataFrame, delivery_index: pd.DatetimeIndex,
                                issue_time) -> pd.DataFrame:
    """Return the latest revision of each feature that was available by `issue_time`.

    `revisions` is long-form with columns valid_time, available_at, feature, value. Rows published
    after the cutoff are ignored; missing or ambiguous availability timestamps are rejected.
    """
    if not REQUIRED_COLUMNS.issubset(revisions.columns):
        raise ValueError(f"Required columns: {sorted(REQUIRED_COLUMNS)}")
    table = revisions.copy()
    for column in ["valid_time", "available_at"]:
        table[column] = pd.to_datetime(table[column], utc=True)
        if table[column].isna().any():
            raise ValueError(f"Missing {column}; availability must be explicit.")
    if table.duplicated(["valid_time", "available_at", "feature"]).any():
        raise ValueError("Ambiguous revisions with identical availability timestamps.")
    eligible = table.loc[(table.available_at <= pd.Timestamp(issue_time).tz_convert("UTC"))
                         & table.valid_time.isin(delivery_index)]
    latest = eligible.sort_values("available_at").drop_duplicates(["valid_time", "feature"], keep="last")
    return latest.pivot(index="valid_time", columns="feature", values="value").reindex(delivery_index)
