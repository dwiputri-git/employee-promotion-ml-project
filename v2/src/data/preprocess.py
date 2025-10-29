from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


def build_preprocess_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: str = "standard",
    encode_categorical: str = "onehot",
    impute_numeric: str = "median",
    impute_categorical: str = "most_frequent",
) -> ColumnTransformer:
    if scale_numeric == "standard":
        scaler = StandardScaler()
    elif scale_numeric == "robust":
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    if encode_categorical == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_numeric)),
            ("scaler", scaler),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_categorical)),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df2 = df.dropna(subset=[target_col]).copy()
    y = df2[target_col]
    X = df2.drop(columns=[target_col])
    if y.dtype != np.int64 and y.dtype != np.int32:
        y = y.astype(int)
    return X, y


