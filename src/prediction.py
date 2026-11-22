from __future__ import annotations

import numpy as np
import pandas as pd


def predict_student(model, feature_columns: list[str], values: dict) -> tuple[str, dict]:
    missing = [column for column in feature_columns if column not in values]
    if missing:
        raise ValueError(f"Prediction values are missing: {', '.join(missing)}")
    row = pd.DataFrame([{column: values[column] for column in feature_columns}])
    row = row.where(pd.notna(row), np.nan)
    prediction = str(model.predict(row)[0])
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, model.predict_proba(row)[0])
        }
    return prediction, probabilities


def global_feature_importance(model, feature_columns: list[str]) -> list[tuple[str, float]]:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        return []
    grouped = {column: 0.0 for column in feature_columns}
    for transformed_name, importance in zip(transformed_names, importances):
        source = str(transformed_name).split("__", 1)[-1]
        matched = next(
            (
                column
                for column in feature_columns
                if source == column or source.startswith(f"{column}_")
            ),
            None,
        )
        if matched is not None:
            grouped[matched] += float(importance)
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def logistic_logit_contributions(model, values: dict, predicted_label: str) -> list[tuple[str, float]]:
    """Return additive feature terms for a logistic class logit, when available."""
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    row = pd.DataFrame([values])
    transformed = model.named_steps["preprocessor"].transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed).ravel()
    coefficients = np.asarray(classifier.coef_)
    classes = [str(label) for label in classifier.classes_]
    if coefficients.shape[0] == 1 and len(classes) == 2:
        class_index = classes.index(predicted_label)
        coefficients = coefficients[0] * (1 if class_index == 1 else -1)
    else:
        coefficients = coefficients[classes.index(predicted_label)]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    totals: dict[str, float] = {}
    for name, value in zip(names, transformed * coefficients):
        feature = str(name).split("__", 1)[-1]
        totals[feature] = totals.get(feature, 0.0) + float(value)
    return sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)


def risk_label(prediction: str) -> str:
    normalized = prediction.strip().lower()
    if any(word in normalized for word in ("fail", "withdraw", "at risk", "at-risk")):
        return "Higher academic risk"
    if any(word in normalized for word in ("borderline", "conditional", "near pass")):
        return "Needs monitoring"
    if any(word in normalized for word in ("pass", "distinction", "excellent", "high")):
        return "Lower academic risk"
    return "Not inferred from this label"
