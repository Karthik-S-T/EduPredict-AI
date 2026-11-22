from __future__ import annotations

import re
from numbers import Number
from typing import Any

import pandas as pd

from src.engagement import score_engagement, suggest_activity_columns
from src.prediction import global_feature_importance, logistic_logit_contributions, predict_student
from src.risk import assess_student_risk


_PRIVATE_COLUMN_TERMS = (
    "name",
    "email",
    "address",
    "phone",
    "birth",
    "gender",
    "ethnicity",
    "nationality",
    "disability",
)

_METRIC_TERMS = ("model", "accuracy", "precision", "recall", "f1", "f-score", "confusion", "classifier", "train")
_FEATURE_TERMS = ("factor", "feature", "importance", "influence", "contribute", "affect")
_DATA_TERMS = ("dataset", "data quality", "missing", "column", "field", "numeric", "categorical", "uploaded")
_ENGAGEMENT_TERMS = ("engagement", "attendance", "activity", "participation", "click", "visit", "submission")
_STUDENT_TERMS = (
    "student",
    "students",
    "learner",
    "learners",
    "strength",
    "weakness",
    "student performance",
    "record",
    "their",
    "they",
    "risk",
    "intervention",
    "prediction",
    "predict",
    "support",
    "improve",
)


def build_assistant_context(
    question: str,
    frame: pd.DataFrame | None,
    bundle: dict[str, Any] | None,
    prediction_history: list[dict[str, Any]],
    student_index: int | None = None,
) -> dict[str, Any]:
    """Build compact, query-specific context without exporting the dataset."""
    normalized_question = question.casefold()
    context: dict[str, Any] = {}
    is_dataset_question = _contains_any(normalized_question, _DATA_TERMS)
    is_model_question = _contains_any(normalized_question, _METRIC_TERMS + _FEATURE_TERMS)
    is_student_question = _contains_any(normalized_question, _STUDENT_TERMS)
    is_engagement_question = _contains_any(normalized_question, _ENGAGEMENT_TERMS)
    is_feature_question = _contains_any(normalized_question, _FEATURE_TERMS)

    if frame is not None and (is_dataset_question or is_student_question or is_engagement_question):
        context["dataset"] = {
            "source": "Loaded dataset",
            "record_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
        }

    if frame is not None and is_dataset_question:
        visible_columns = [column for column in frame.columns if not _is_private_column(str(column))]
        missing = frame[visible_columns].isna().sum()
        context["dataset"]["columns"] = [
            {
                "name": str(column),
                "type": _column_type(frame[column]),
                "missing": int(missing[column]),
            }
            for column in visible_columns[:30]
        ]
        context["dataset"]["columns_omitted"] = max(0, len(visible_columns) - 30)
        if _contains_any(normalized_question, ("missing", "quality", "statistics", "summary")):
            context["dataset"]["duplicate_rows"] = int(frame.duplicated().sum())
            numeric_columns = [
                column
                for column in frame.select_dtypes(include="number").columns
                if not _is_private_column(str(column))
            ][:12]
            context["dataset"]["numeric_summary"] = {
                str(column): {
                    "observed": int(frame[column].count()),
                    "missing": int(frame[column].isna().sum()),
                    "mean": _json_number(frame[column].mean()),
                    "median": _json_number(frame[column].median()),
                }
                for column in numeric_columns
            }

    if frame is not None and is_engagement_question:
        activity_columns = suggest_activity_columns(frame)
        if len(activity_columns) >= 2:
            try:
                engagement = score_engagement(frame, activity_columns)
                context["engagement_cohort"] = {
                    "indicators": [str(column) for column in activity_columns[:8]],
                    "average_score": _json_number(engagement["Engagement Score"].mean()),
                    "category_counts": {
                        str(label): int(count)
                        for label, count in engagement["Engagement Level"].value_counts().items()
                    },
                    "note": "Experimental cohort-relative indicator, not a validated measure of motivation or learning.",
                }
                separate_columns = [
                    column
                    for column in frame.select_dtypes(include="number").columns
                    if any(term in str(column).casefold() for term in ("attendance", "assessment score"))
                    and not _is_private_column(str(column))
                ][:6]
                context["engagement_cohort"]["related_academic_indicators"] = {
                    str(column): {
                        "mean": _json_number(frame[column].mean()),
                        "missing": int(frame[column].isna().sum()),
                    }
                    for column in separate_columns
                }
            except ValueError:
                context["engagement_cohort"] = {"available": False}
        else:
            context["engagement_cohort"] = {"available": False}

    selected_row = None
    if frame is not None and student_index is not None and 0 <= student_index < len(frame) and is_student_question:
        selected_row = frame.iloc[student_index]

    model_name = None
    if bundle and bundle.get("metrics"):
        model_name = max(bundle["metrics"], key=lambda name: bundle["metrics"][name].get("f1", 0))
        needs_student_model = selected_row is not None and (
            is_feature_question
            or _contains_any(
                normalized_question,
                ("risk", "support", "intervention", "improve", "prediction", "predict", "performance", "outcome"),
            )
        )
        if is_model_question or needs_student_model:
            best_f1 = max(values.get("f1", 0) for values in bundle["metrics"].values())
            tied_models = [
                name
                for name, values in bundle["metrics"].items()
                if values.get("f1", 0) == best_f1
            ]
            context["models"] = {
                "target": str(bundle.get("target", "Not recorded")),
                "test_records": int(bundle.get("test_records", 0)),
                "best_by_weighted_f1": model_name,
                "weighted_f1_ties": tied_models,
                "best_weighted_f1": _json_number(best_f1),
                "evaluation": {
                    name: {
                        key: _json_number(values.get(key))
                        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
                        if values.get(key) is not None
                    }
                    for name, values in bundle["metrics"].items()
                },
            }
            if _contains_any(normalized_question, ("confusion",)):
                metrics = bundle["metrics"][model_name]
                context["models"]["confusion_matrix"] = {
                    "classes": [str(label) for label in bundle.get("classes", [])],
                    "counts": metrics.get("confusion_matrix", []),
                }
        if is_feature_question:
            importance = global_feature_importance(bundle["models"][model_name], bundle["feature_columns"])
            context["models"] = context.get("models", {"best_by_weighted_f1": model_name})
            context["models"]["feature_importance"] = [
                {"feature": feature, "importance": _json_number(value)}
                for feature, value in importance
                if not _is_private_column(feature)
            ][:10]
            context["models"]["feature_importance_available"] = bool(importance)

    if selected_row is not None:
        student_context: dict[str, Any] = {
            "reference": "Selected record",
            "observed_academic_indicators": {},
        }
        candidate_columns = (
            bundle.get("feature_columns", [])
            if bundle
            else frame.select_dtypes(include="number").columns.tolist()
        )
        visible_student_columns = [
            column
            for column in candidate_columns
            if column in frame.columns and not _is_private_column(str(column))
        ][:12]
        student_context["observed_academic_indicators"] = {
            str(column): _json_value(selected_row[column]) for column in visible_student_columns
        }
        cohort_percentiles = {}
        for column in visible_student_columns:
            if not pd.api.types.is_numeric_dtype(frame[column].dtype) or pd.isna(selected_row[column]):
                continue
            cohort = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not cohort.empty:
                cohort_percentiles[str(column)] = _json_number(
                    (cohort <= float(selected_row[column])).mean() * 100
                )
        if cohort_percentiles:
            student_context["cohort_percentiles"] = cohort_percentiles
        if bundle and bundle.get("target") in frame.columns:
            student_context["observed_target"] = _json_value(selected_row[bundle["target"]])

        if bundle and model_name:
            model = bundle["models"][model_name]
            features = bundle["feature_columns"]
            values = selected_row[features].to_dict()
            try:
                predicted, probabilities = predict_student(model, features, values)
                student_context["prediction"] = {
                    "target": str(bundle["target"]),
                    "predicted_outcome": predicted,
                    "model": model_name,
                    "class_probabilities": probabilities,
                    "note": "Model estimate, not a guaranteed outcome.",
                }
                if is_feature_question:
                    contributions = logistic_logit_contributions(model, values, predicted)
                    if contributions:
                        student_context["individual_logit_terms"] = [
                            {"feature": feature, "term": _json_number(value)}
                            for feature, value in contributions[:8]
                            if not _is_private_column(feature)
                        ]
            except (ValueError, TypeError):
                student_context["prediction"] = {"available": False}

            if _contains_any(normalized_question, ("risk", "support", "intervention", "improve")):
                try:
                    risk = assess_student_risk(frame, bundle, model_name).iloc[student_index]
                    risk_reasons = [
                        reason
                        for reason in str(risk["Risk reasons"]).split("; ")
                        if not any(
                            _is_private_column(feature)
                            for feature in bundle.get("numeric_features", [])
                            if feature in reason
                        )
                    ]
                    student_context["risk_assessment"] = {
                        "category": str(risk["Risk category"]),
                        "reasons": "; ".join(risk_reasons),
                        "suggested_support": str(risk["Suggested support"]),
                        "probability": _json_number(risk["Risk probability"]),
                        "note": "Decision-support indicator for educator review, not a definitive judgment.",
                    }
                except ValueError:
                    student_context["risk_assessment"] = {
                        "available": False,
                        "reason": "The trained target labels do not support a safe risk mapping.",
                    }

        if is_engagement_question:
            activity_columns = suggest_activity_columns(frame)
            if len(activity_columns) >= 2:
                try:
                    engagement = score_engagement(frame, activity_columns)
                    result = engagement.iloc[student_index]
                    student_context["engagement"] = {
                        "score": _json_number(result["Engagement Score"]),
                        "cohort_relative_level": _json_value(result["Engagement Level"]),
                        "indicators": {
                            str(column): _json_value(selected_row[column])
                            for column in activity_columns[:8]
                            if not _is_private_column(str(column))
                        },
                        "note": "Experimental cohort-relative indicator, not a validated measure of motivation or learning.",
                    }
                    related_columns = [
                        column
                        for column in frame.select_dtypes(include="number").columns
                        if any(term in str(column).casefold() for term in ("attendance", "assessment score"))
                        and not _is_private_column(str(column))
                    ][:6]
                    student_context["engagement"]["related_academic_indicators"] = {
                        str(column): _json_value(selected_row[column]) for column in related_columns
                    }
                except ValueError:
                    student_context["engagement"] = {"available": False}
            else:
                student_context["engagement"] = {"available": False}
        context["selected_student"] = student_context

    if frame is None and is_dataset_question:
        context["dataset"] = {"available": False}
    if not bundle and is_model_question:
        context["models"] = {"available": False}
    if selected_row is None and _contains_any(normalized_question, ("last prediction", "recent prediction", "my prediction")):
        if prediction_history:
            latest = prediction_history[-1]
            context["recent_prediction"] = {
                key: _json_value(latest[key])
                for key in ("Predicted result", "Target", "Model", "Highest class probability", "Risk interpretation")
                if key in latest
            }
        else:
            context["recent_prediction"] = {"available": False}
    return context


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _is_private_column(column: str) -> bool:
    normalized = column.casefold().replace("_", " ").replace("-", " ")
    return (
        normalized in {"id", "student id", "id student"}
        or normalized.endswith(" id")
        or normalized.startswith("id ")
        or any(term in normalized for term in _PRIVATE_COLUMN_TERMS)
        or bool(re.search(r"\b(age|race)\b", normalized))
    )


def _column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numerical"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "date/time"
    return "categorical/text"


def _json_number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _json_value(value: Any) -> str | float | int | bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        return _json_number(value)
    return str(value)
