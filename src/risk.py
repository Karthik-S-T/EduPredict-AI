from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data_loader import find_column


HIGH_RISK_TERMS = (
    "fail",
    "withdraw",
    "at risk",
    "high risk",
    "not pass",
    "non pass",
    "very low",
    "low performance",
    "poor",
)
MODERATE_RISK_TERMS = (
    "borderline",
    "conditional",
    "moderate",
    "average",
    "needs support",
    "moderate performance",
)
LOW_RISK_TERMS = (
    "pass",
    "distinction",
    "excellent",
    "good",
    "high performance",
    "low risk",
    "satisfactory",
)


def class_risk_band(label: str) -> str | None:
    normalized = re.sub(r"[_-]+", " ", str(label).strip().lower())
    if any(term in normalized for term in HIGH_RISK_TERMS):
        return "High Risk"
    if any(term in normalized for term in MODERATE_RISK_TERMS):
        return "Moderate Risk"
    if any(term in normalized for term in LOW_RISK_TERMS):
        return "Low Risk"
    if normalized in {"f", "grade f", "d", "grade d"}:
        return "High Risk" if normalized in {"f", "grade f"} else "Moderate Risk"
    if normalized in {"a", "b", "grade a", "grade b"}:
        return "Low Risk"
    if normalized in {"c", "grade c"}:
        return "Moderate Risk"
    return None


def assess_student_risk(frame: pd.DataFrame, bundle: dict, model_name: str | None = None) -> pd.DataFrame:
    """Assess risk only when model class labels have a documented semantic mapping."""
    features = bundle["feature_columns"]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Dataset does not contain model features: {', '.join(missing)}")
    if model_name is None:
        model_name = max(bundle["metrics"], key=lambda name: bundle["metrics"][name]["f1"])
    model = bundle["models"][model_name]
    class_bands = {str(label): class_risk_band(str(label)) for label in model.classes_}
    mapped = {label: band for label, band in class_bands.items() if band is not None}
    if not mapped:
        raise ValueError(
            "The trained target classes do not identify academic risk (for example Pass/Borderline/Fail). "
            "Risk categories cannot be inferred safely from these labels."
        )

    predictions = model.predict(frame[features]).astype(str)
    probabilities = model.predict_proba(frame[features]) if hasattr(model, "predict_proba") else None
    classes = [str(label) for label in model.classes_]
    high_classes = [label for label, band in mapped.items() if band == "High Risk"]
    moderate_classes = [label for label, band in mapped.items() if band == "Moderate Risk"]
    risk_probability = None
    if probabilities is not None and len(mapped) == len(classes):
        risk_probability = np.zeros(len(frame))
        for label in high_classes:
            risk_probability += probabilities[:, classes.index(label)]
        for label in moderate_classes:
            risk_probability += 0.5 * probabilities[:, classes.index(label)]

    rows = []
    numeric_features = bundle.get("numeric_features", [])
    for row_index, (index, row) in enumerate(frame.iterrows()):
        predicted_label = predictions[row_index]
        predicted_band = class_bands.get(predicted_label)
        if risk_probability is not None:
            score = float(risk_probability[row_index])
            if score >= 0.6:
                category = "High Risk"
            elif score >= 0.25:
                category = "Moderate Risk"
            else:
                category = "Low Risk"
        else:
            score = None
            category = predicted_band or "Unclassified"
        reasons = [f"Model outcome: {predicted_label}"]
        if score is not None:
            reasons.append(f"Mapped risk probability: {score:.1%}")
        for feature in numeric_features:
            if feature not in features or feature not in frame.columns:
                continue
            values = pd.to_numeric(frame[feature], errors="coerce").dropna()
            value = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
            if not values.empty and pd.notna(value) and value <= values.quantile(0.25):
                reasons.append(f"{feature} is in the cohort's lower quartile")
        recommendations = _recommendations(category, reasons)
        rows.append(
            {
                "_row_index": index,
                "Predicted outcome": predicted_label,
                "Risk probability": score,
                "Risk category": category,
                "Risk reasons": "; ".join(reasons),
                "Suggested support": "; ".join(recommendations),
                "Model": model_name,
            }
        )
    return pd.DataFrame(rows)


def student_identifier_column(frame: pd.DataFrame) -> str | None:
    return find_column(frame, ("student id", "id student", "student_id", "id_student"))


def _recommendations(category: str, reasons: list[str]) -> list[str]:
    if category == "Low Risk":
        return ["Continue regular progress reviews"]
    suggestions = ["Schedule supportive academic mentoring", "Monitor future assessment performance"]
    combined = " ".join(reasons).lower()
    if "assignment" in combined or "submission" in combined:
        suggestions.append("Encourage completion of outstanding assignments")
    if "assessment" in combined or "score" in combined:
        suggestions.append("Review challenging assessment topics and offer additional materials")
    if "attendance" in combined:
        suggestions.append("Discuss attendance barriers and agree on a practical attendance plan")
    if "click" in combined or "activity" in combined or "engagement" in combined:
        suggestions.append("Check whether the learner can access and navigate course learning resources")
    return suggestions
