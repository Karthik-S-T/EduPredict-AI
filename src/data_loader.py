from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd


def read_dataset(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file and return a non-empty dataframe."""
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()
    if name.endswith(".csv"):
        frame = pd.read_csv(BytesIO(content))
    elif name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(BytesIO(content))
    else:
        raise ValueError("Unsupported file format. Upload a CSV or Excel workbook.")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The uploaded file contains no data.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise ValueError("Every dataset column must have a name.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    return frame


def make_demo_dataset(rows: int = 360, seed: int = 27) -> pd.DataFrame:
    """Create clearly synthetic academic data for demonstrating the workflow."""
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(82, 12, rows), 35, 100)
    assignments = np.clip(rng.normal(78, 16, rows), 15, 100)
    visits = np.clip(rng.poisson(34, rows), 1, None)
    clicks = np.clip(visits * rng.normal(3.2, 0.8, rows) + rng.normal(0, 10, rows), 2, None)
    participation = np.clip(rng.normal(75, 17, rows), 10, 100)
    previous = np.clip(rng.normal(70, 18, rows), 0, 100)
    score = np.clip(
        0.30 * attendance
        + 0.25 * assignments
        + 0.20 * participation
        + 0.20 * previous
        + 0.05 * np.minimum(clicks, 100)
        + rng.normal(0, 9, rows),
        0,
        100,
    )
    result = np.select([score >= 75, score >= 55], ["Pass", "Borderline"], default="Fail")
    return pd.DataFrame(
        {
            "Student ID": [f"DEMO-{index:04d}" for index in range(1, rows + 1)],
            "Assessment Score": np.round(score, 1),
            "Attendance Percentage": np.round(attendance, 1),
            "Assignment Submission Percentage": np.round(assignments, 1),
            "Platform Visits": visits,
            "Total Clicks": np.round(clicks).astype(int),
            "Assessment Participation": np.round(participation, 1),
            "Previous Academic Result": np.round(previous, 1),
            "Final Result": result,
        }
    )


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=np.number).columns.tolist()


def find_column(frame: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    """Find a column by semantic name fragments, ignoring case and spacing."""
    for column in frame.columns:
        normalized = str(column).lower().replace("_", " ").replace("-", " ")
        if any(term in normalized for term in terms):
            return column
    return None


def suggest_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    return [
        column
        for column in frame.columns
        if column != target and not is_restricted_predictor(column, target)
    ]


def is_identifier_column(column: str) -> bool:
    normalized = str(column).strip().lower().replace("_", " ").replace("-", " ")
    return (
        normalized in {"id", "student id", "id student"}
        or normalized.endswith(" id")
        or normalized.startswith("id ")
    )


def is_restricted_predictor(column: str, target: str) -> bool:
    """Reject IDs and obvious outcome aliases from model input features."""
    normalized = str(column).strip().lower().replace("_", " ").replace("-", " ")
    target_normalized = str(target).strip().lower().replace("_", " ").replace("-", " ")
    if normalized == target_normalized:
        return True
    if normalized in {
        "id",
        "student id",
        "id student",
        "class",
        "label",
        "target",
        "outcome",
        "final result",
        "final grade",
        "final score",
        "target score",
        "result category",
        "performance category",
        "performance label",
        "outcome category",
        "pass status",
        "pass fail",
        "pass/fail",
    }:
        return True
    if normalized.startswith("final ") and any(
        term in normalized for term in ("result", "grade", "score", "outcome", "status")
    ):
        return True
    return is_identifier_column(column)
