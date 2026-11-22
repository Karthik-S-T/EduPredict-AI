from __future__ import annotations

import pandas as pd


ACTIVITY_TERMS = (
    "visit",
    "click",
    "activity",
    "engagement",
    "submission",
    "participation",
    "interaction",
    "vle",
)


def suggest_activity_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if any(term in str(column).lower() for term in ACTIVITY_TERMS)
    ]


def score_engagement(frame: pd.DataFrame, activity_columns: list[str]) -> pd.DataFrame:
    if len(activity_columns) < 2:
        raise ValueError("Select at least two numeric activity indicators; visits alone are insufficient.")
    invalid = [
        column
        for column in activity_columns
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if invalid:
        raise ValueError(f"Engagement indicators must be numeric columns: {', '.join(invalid)}")
    indicators = frame[activity_columns].copy()
    usable = [column for column in activity_columns if indicators[column].notna().any()]
    if len(usable) < 2:
        raise ValueError("At least two selected activity indicators need observed values.")
    percentiles = indicators[usable].rank(pct=True, method="average") * 100
    scores = percentiles.mean(axis=1, skipna=True)
    labels = pd.cut(
        scores,
        bins=[-0.01, 33.333, 66.667, 100],
        labels=["Low Engagement", "Moderate Engagement", "High Engagement"],
        include_lowest=True,
    ).astype("object")
    result = frame.copy()
    result["Engagement Score"] = scores.round(1)
    result["Engagement Level"] = labels
    return result
