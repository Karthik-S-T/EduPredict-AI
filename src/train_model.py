from __future__ import annotations

import warnings
from datetime import datetime, timezone
from time import perf_counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import is_restricted_predictor
from src.preprocessing import build_preprocessor

MODEL_NAMES = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
)


def train_models(
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    test_size: float = 0.2,
    selected_models: Sequence[str] | None = None,
    random_state: int = 42,
    random_forest_estimators: int = 180,
    random_forest_max_depth: int | None = 8,
    decision_tree_max_depth: int | None = 8,
    gradient_boosting_estimators: int = 100,
    dataset_name: str = "Uploaded dataset",
) -> dict:
    if target not in frame.columns:
        raise ValueError("The selected target column is not in this dataset.")
    if target in feature_columns or not feature_columns:
        raise ValueError("Choose one or more features, excluding the target column.")
    restricted = [column for column in feature_columns if is_restricted_predictor(column, target)]
    if restricted:
        raise ValueError(
            "Remove identifier or outcome-alias columns from predictors: "
            + ", ".join(restricted)
        )
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Unknown feature columns: {', '.join(missing_columns)}")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Choose a test split between 10% and 40%.")
    if not 0 <= random_state <= 2**32 - 1:
        raise ValueError("Random state must be between 0 and 4,294,967,295.")
    if not 50 <= random_forest_estimators <= 1000:
        raise ValueError("Random Forest estimators must be between 50 and 1,000.")
    if random_forest_max_depth is not None and not 2 <= random_forest_max_depth <= 50:
        raise ValueError("Random Forest max depth must be between 2 and 50.")
    if decision_tree_max_depth is not None and not 2 <= decision_tree_max_depth <= 50:
        raise ValueError("Decision Tree max depth must be between 2 and 50.")
    if not 50 <= gradient_boosting_estimators <= 500:
        raise ValueError("Gradient Boosting estimators must be between 50 and 500.")

    clean = frame.drop_duplicates()[feature_columns + [target]].dropna(subset=[target]).copy()
    feature_columns = [
        column for column in feature_columns if not clean[column].isna().all()
    ]
    if not feature_columns:
        raise ValueError("All selected features are empty. Choose columns with values.")
    clean = clean[feature_columns + [target]]
    if len(clean) < 8:
        raise ValueError("At least 8 complete target records are required for a train/test split.")
    if clean[target].nunique() < 2:
        raise ValueError("The target must contain at least two distinct classes.")
    maximum_classes = min(50, max(10, len(clean) // 5))
    if clean[target].nunique() > maximum_classes:
        raise ValueError(
            f"The target has {clean[target].nunique()} distinct classes, too many for this classification workflow. "
            "Choose a categorical outcome such as a final result or convert a continuous score into justified categories."
        )

    X, y = clean[feature_columns], clean[target].astype(str)
    class_counts = y.value_counts()
    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    stratify = (
        y
        if class_counts.min() >= 2
        and test_count >= class_counts.size
        and train_count >= class_counts.size
        else None
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as error:
        raise ValueError(f"Unable to create the requested train/test split: {error}") from error

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=decision_tree_max_depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=random_forest_estimators,
            max_depth=random_forest_max_depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gradient_boosting_estimators,
            random_state=random_state,
        ),
    }
    if selected_models is None:
        selected_models = list(estimators)
    if not selected_models:
        raise ValueError("Select at least one classifier.")
    unknown_models = sorted(set(selected_models) - set(estimators))
    if unknown_models:
        raise ValueError(f"Unknown classifiers: {', '.join(unknown_models)}")
    models, metrics = {}, {}
    labels = sorted(y.unique())
    for name in estimators:
        if name not in selected_models:
            continue
        estimator = estimators[name]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )
        started = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "Gradient Boosting":
                sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
                pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
            else:
                pipeline.fit(X_train, y_train)
        training_seconds = perf_counter() - started
        predicted = pipeline.predict(X_test)
        models[name] = pipeline
        probabilities = (
            pipeline.predict_proba(X_test)
            if hasattr(pipeline, "predict_proba")
            else None
        )
        class_labels = [str(label) for label in pipeline.classes_]
        roc_auc = None
        if probabilities is not None and len(class_labels) >= 2:
            try:
                if len(class_labels) == 2:
                    roc_auc = float(
                        roc_auc_score(
                            (y_test.astype(str) == class_labels[1]).astype(int),
                            probabilities[:, 1],
                        )
                    )
                else:
                    roc_auc = float(
                        roc_auc_score(
                            y_test.astype(str),
                            probabilities,
                            labels=class_labels,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )
            except ValueError:
                roc_auc = None
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
            "classification_report": classification_report(
                y_test, predicted, labels=labels, output_dict=True, zero_division=0
            ),
            "training_seconds": float(training_seconds),
            "roc_auc": roc_auc,
            "roc_classes": class_labels,
            "test_actual": y_test.astype(str).tolist(),
            "test_probabilities": probabilities.tolist() if probabilities is not None else None,
        }
    return {
        "models": models,
        "metrics": metrics,
        "target": target,
        "feature_columns": feature_columns,
        "classes": labels,
        "test_size": test_size,
        "stratified_split": stratify is not None,
        "test_records": len(X_test),
        "train_records": len(X_train),
        "averaging": "weighted",
        "random_state": random_state,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "selected_models": list(models),
        "training_parameters": {
            "random_forest_estimators": random_forest_estimators,
            "random_forest_max_depth": random_forest_max_depth,
            "decision_tree_max_depth": decision_tree_max_depth,
            "gradient_boosting_estimators": gradient_boosting_estimators,
        },
        "feature_defaults": _feature_defaults(X_train),
        "numeric_features": X_train.select_dtypes(include="number").columns.tolist(),
    }


def _feature_defaults(features: pd.DataFrame) -> dict:
    defaults = {}
    for column in features.columns:
        values = features[column].dropna()
        if pd.api.types.is_numeric_dtype(features[column]):
            defaults[column] = float(values.median()) if not values.empty else 0.0
        else:
            modes = values.mode()
            defaults[column] = str(modes.iloc[0]) if not modes.empty else ""
    return defaults
