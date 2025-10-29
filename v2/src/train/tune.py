from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from data.preprocess import build_preprocess_pipeline, split_features_target
from eval.metrics import compute_classification_metrics
from models.candidates import get_candidate_models
from utils.io import Config


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def stratified_split(df: pd.DataFrame, target_col: str, test_size: float, seed: int) -> DatasetSplit:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X, y = split_features_target(df, target_col)
    idx_train, idx_test = next(splitter.split(X, y))
    return DatasetSplit(X.iloc[idx_train], X.iloc[idx_test], y.iloc[idx_train], y.iloc[idx_test])


def build_pipeline(preprocessor, model, imbalance_strategy: str, categorical_features: Tuple[int, ...] | None = None):
    if imbalance_strategy == "smote" and categorical_features is not None:
        smote = SMOTENC(categorical_features=categorical_features, random_state=0)
        pipe = ImbPipeline(steps=[("preprocess", preprocessor), ("smote", smote), ("model", model)])
    else:
        pipe = ImbPipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe


def get_feature_lists(df: pd.DataFrame, target_col: str) -> Tuple[list[str], list[str]]:
    features = [c for c in df.columns if c != target_col]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    return numeric, categorical


def cv_iterator(n_splits: int, n_repeats: int, seed: int):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def evaluate_model(pipe, X: pd.DataFrame, y: pd.Series, cv, primary: str) -> Tuple[float, Dict[str, float]]:
    scores: Dict[str, float] = {"accuracy": [], "f1_macro": [], "f1_weighted": [], "rocauc": [], "prauc": [], "brier": []}
    for train_idx, valid_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_metrics = compute_classification_metrics(y_va.to_numpy(), proba, pred)
        for k in scores:
            scores[k].append(fold_metrics.get(k, np.nan))
    mean_scores = {k: float(np.nanmean(v)) for k, v in scores.items()}
    return float(mean_scores.get(primary, np.nan)), mean_scores


def run_broad_search(df: pd.DataFrame, cfg: Config, target_col: str = "Promotion_Eligible") -> pd.DataFrame:
    seed = cfg.get("seed", 42)
    test_size = cfg.get("cv.test_size", 0.2)
    primary = cfg.get("metrics.primary", "prauc")
    imb_strategy = cfg.get("imbalance.strategy", "class_weight")

    split = stratified_split(df, target_col, test_size, seed)
    numeric, categorical = get_feature_lists(df, target_col)

    preprocessor = build_preprocess_pipeline(
        numeric_features=numeric,
        categorical_features=categorical,
        scale_numeric=cfg.get("preprocessing.scale_numeric", "standard"),
        encode_categorical=cfg.get("preprocessing.encode_categorical", "onehot"),
        impute_numeric=cfg.get("preprocessing.impute_numeric", "median"),
        impute_categorical=cfg.get("preprocessing.impute_categorical", "most_frequent"),
    )

    models = get_candidate_models(
        class_weight_strategy="class_weight" if imb_strategy == "class_weight" else "none"
    )

    cv = cv_iterator(
        n_splits=int(cfg.get("cv.n_splits", 5)),
        n_repeats=int(cfg.get("cv.n_repeats", 3)),
        seed=seed,
    )

    results = []
    for name, model in models.items():
        pipe = build_pipeline(preprocessor, model, imb_strategy, categorical_features=None)
        score, metrics = evaluate_model(pipe, split.X_train, split.y_train, cv, primary)
        row = {"model": name, **metrics}
        results.append(row)

    return pd.DataFrame(results).sort_values(primary, ascending=False).reset_index(drop=True)


