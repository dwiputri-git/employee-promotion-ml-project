# V3 Notebooks Plan

This version mirrors the original notebooks flow, upgraded to satisfy the full rubric. Artifacts are written under `v3_artifacts/` and cleaned/engineered data under `v3_data/`.

Order to run:
1. 01_data_intro.ipynb
2. 02_data_cleaning.ipynb
3. 03_feature_engineering.ipynb
4. 04_model_pipeline_baselines.ipynb
5. 05_model_selection_cv.ipynb
6. 06_hyperparameter_tuning.ipynb
7. 07_calibration_threshold.ipynb
8. 08_explainability_fairness.ipynb
9. 09_error_business_analysis.ipynb

Notes:
- Primary metric: PR-AUC (with class imbalance). Also track Accuracy, F1 macro/weighted, ROC-AUC, Brier.
- File I/O is consistent and config variables are at the top of each notebook.

