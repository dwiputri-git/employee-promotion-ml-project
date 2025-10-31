from __future__ import annotations

from typing import Dict

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_candidate_models(class_weight_strategy: str = "balanced") -> Dict[str, object]:
    weight = "balanced" if class_weight_strategy == "class_weight" else None

    models: Dict[str, object] = {
        "logreg": LogisticRegression(max_iter=2000, class_weight=weight, n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=300, class_weight=weight, n_jobs=-1, random_state=0),
        "gb": GradientBoostingClassifier(random_state=0),
        "xgb": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=0,
            n_jobs=-1,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_state=0,
        ),
    }
    return models



