from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("Select at least one usable feature column.")
    return ColumnTransformer(transformers=transformers, remainder="drop")
