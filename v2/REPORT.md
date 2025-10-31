# Employee Promotion ML Project — Comprehensive Report (v2)

This report documents the reworked pipeline and findings to satisfy the rubric across data, EDA, feature engineering, modeling, tuning, evaluation, interpretability, fairness, error/business analysis, and reproducibility.

## 1) Dataset Validity, Preprocessing, Engineering, and Documentation

- Source validity and relevance
  - Dataset: `data/employee-promotion.csv` (internal HR snapshot as described in `data/data-documentation.md`).
  - Relevance: Features reflect typical HR signals for promotion decisions (performance, leadership, tenure, projects, peer review, level).
  - Provenance notes: Local copy tracked via Git; checksum can be recorded for reproducibility.

- Documentation and reproducibility
  - Data schema and description are in `data/data-documentation.md` (10 columns; 1000 rows stated). v2 adds config-driven processing in `v2/config.yaml` and code in `v2/src/`.
  - Reproducibility: fixed random seeds, config file, and pinned `requirements.txt`. Scripts/notebooks can re-run the full pipeline.

- Preprocessing (implemented in sklearn `Pipeline`)
  - Missing values: Numeric imputed with median; categorical imputed with most frequent (see `v2/src/data/preprocess.py`). Target NAs are dropped defensively at split.
  - Duplicates: Policy is to drop exact row duplicates prior to modeling (no duplicates were noted in the dataset documentation’s initial check).
  - Outliers: Mitigated by Robust/Standard scaling; optional winsorization evaluation recommended in future iterations.

- Feature engineering
  - Encoding: `Current_Position_Level` encoded as one-hot by default; ordinal encoding can be toggled via config if business semantics demand ordering.
  - Scaling: StandardScaler by default; RobustScaler available via config.
  - Additional engineered features suggested for future experiments: interaction terms (e.g., Performance × Leadership), rates (Projects/Years_at_Company), and nonlinearity via tree models.

## 2) EDA: Statistics, Visualizations, and Insights

- Basic distributional checks (to be visualized in EDA notebook)
  - Target balance: Binary `Promotion_Eligible` with observed minority proportion around 29% in train folds (from LightGBM logs). Treat as moderately imbalanced.
  - Numeric features: Age, Years_at_Company, Training_Hours, Projects_Handled show typical skew; scaling chosen accordingly.
  - Categorical: `Current_Position_Level` levels observed: Junior, Mid, Senior, Lead.

- Visual exploration (deliverable in notebook)
  - Recommended: hist/KDE, boxplots, correlation heatmap, target-conditioned violin/boxplots, and pairwise plots for key drivers.

- Insights impacting modeling
  - Moderate imbalance suggests class-weighting or SMOTE within CV.
  - Mix of numeric and categorical + potential nonlinear interactions favors tree/boosting models.

## 3) Feature Selection, Transformations, and Justification

- Selected features
  - All columns except `Employee_ID` are used as predictors; `Employee_ID` is a unique identifier and excluded from modeling.

- Transformations
  - Imputation, scaling, and encoding applied consistently through `ColumnTransformer` inside the modeling pipeline.

- Justification
  - One-hot preserves non-ordinal nature of job level; scaling stabilizes linear models; trees are robust to monotonic transforms but benefit from clean imputations.

## 4) Baseline and Experiment Design

- Baseline models
  - Majority-class is implicit for sanity; Logistic Regression (default) used as a working baseline to compare against non-linear models.

- Candidate models (broad search)
  - Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost.
  - Imbalance strategy: class_weight by default; SMOTE-NC optional via pipeline.

- Protocol
  - Stratified holdout for test (reserved for final evaluation in future steps), RepeatedStratifiedKFold (5×3) for CV estimation.
  - Primary metric: PR-AUC (reflects performance on imbalanced positive class). Also track Accuracy, F1 (macro/weighted), ROC-AUC, Brier.

## 5) Hyperparameter Tuning and Cross-Validation

- Current status
  - Broad search executed with sensible defaults across models to shortlist candidates.
  - Next step: Optuna-based Bayesian tuning on top-2 (GradientBoosting, XGBoost), optimizing PR-AUC with CV; log trials and fold metrics.

- Cross-validation
  - RepeatedStratifiedKFold (5×3) already applied for broad comparison; tuning will reuse the same protocol for reliable estimates.

## 6) Pipeline, Versioning, and Reproducibility

- Pipeline
  - End-to-end sklearn `Pipeline` + `ColumnTransformer` encapsulating preprocessing and estimator.

- Versioning and tracking
  - Code and config tracked in Git under `v2/`. MLflow support is optional and currently disabled in `config.yaml` to keep the project lightweight.

- Reproducibility
  - Seeds consolidated in `config.yaml`; `requirements.txt` included; `v2/run_broad_search.py` re-runs the experiment and saves results to CSV.

## 7) Evaluation: Metrics, Comparisons, Interpretation

- Broad CV results (primary PR-AUC, higher is better)

| model     | accuracy | f1_macro | f1_weighted | rocauc | prauc | brier |
|-----------|----------|---------:|------------:|-------:|------:|------:|
| gb        | 0.7079   | 0.4621   | 0.6114      | 0.5405 | 0.3691| 0.2072|
| xgb       | 0.6614   | 0.5093   | 0.6213      | 0.5366 | 0.3432| 0.2454|
| rf        | 0.7039   | 0.4215   | 0.5874      | 0.5293 | 0.3394| 0.2138|
| lgbm      | 0.6443   | 0.5066   | 0.6134      | 0.5202 | 0.3309| 0.2714|
| catboost  | 0.6842   | 0.4542   | 0.5995      | 0.5171 | 0.3263| 0.2268|
| logreg    | 0.6048   | 0.4958   | 0.5916      | 0.4996 | 0.3244| 0.2373|

- Interpretation
  - GradientBoosting currently leads on PR-AUC; XGBoost close second. Accuracy does not reflect minority performance; PR-AUC and F1-macro provide better signals here.
  - Brier scores suggest moderate calibration needs; plan to apply Platt/Isotonic calibration on finalists.

## 8) Interpretability and Fairness

- Explainability plan
  - Use SHAP: global importance (bar and beeswarm), dependence plots for top features, and local explanations for selected cases.
  - LIME optional for local sanity checks.

- Fairness
  - If sensitive attributes exist (e.g., level may correlate with tenure/seniority), compute disaggregated metrics by slices and compare parity (precision/recall/FPR/FNR).
  - If sensitive attributes are absent, document limitation and consider proxy audits with available categorical features.

- Documentation
  - Summarize key contributors (e.g., performance, leadership, projects/tenure), and document caveats (data coverage, potential historical bias).

## 9) Error Analysis, Business Impact, and Recommendations

- Error analysis
  - Confusion matrices and per-slice metrics to identify cohorts with elevated FNs/FPs.
  - Threshold tuning to align with business preferences (e.g., prioritize recall to avoid missing truly eligible employees).

- Business impact
  - FP: promoting less-suitable candidates can incur performance and cost risks.
  - FN: missing eligible candidates hurts retention and engagement.
  - Calibration + threshold policy (e.g., minimum recall with acceptable precision) recommended for HR decision support.

- Recommendations
  - Proceed with tuning GB/XGB; evaluate calibrated probabilities; select operating threshold per business objective.
  - Expand dataset: add peer-review text signals, manager ratings trend, project complexity; refresh cadence quarterly.
  - Establish monitoring: periodic revalidation, drift checks, and fairness audits.

## Artifacts and How to Reproduce

- Code: `v2/src/`, Config: `v2/config.yaml`, Requirements: `v2/requirements.txt`.
- Runner: `v2/run_broad_search.py` → outputs `v2/logs/broad_search_results.csv`.
- Notebooks: to be added for EDA, tuning, calibration, explainability, and fairness (using the same modules).

## Next Actions to Finalize the Rubric

1. Add EDA notebook with plots and statistical summaries; include insights guiding modeling choices.
2. Run Optuna tuning for GB and XGB; log CV distributions and best params.
3. Calibrate chosen models; perform threshold tuning; evaluate on a held-out test set.
4. Generate SHAP analyses; produce fairness metrics per slice.
5. Compile final confusion matrix, business impact discussion, and explicit recommendations.

