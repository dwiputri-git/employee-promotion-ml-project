# Employee Promotion ML Project - v2 (Enhanced)

This folder contains a reworked, fully reproducible pipeline to search for the best-performing model with robust evaluation and documentation.

## Highlights
- Config-driven preprocessing, model selection, and evaluation.
- Cross-validated hyperparameter tuning (Optuna) with class imbalance options.
- Optional experiment tracking (enable/disable in `config.yaml`).
- Clear separation from the original project version.

## Structure
```
v2/
  ├── README.md
  ├── requirements.txt
  ├── config.yaml
  └── src/
      ├── __init__.py
      ├── data/
      │   └── preprocess.py
      ├── eval/
      │   └── metrics.py
      ├── models/
      │   └── candidates.py
      ├── train/
      │   └── tune.py
      └── utils/
          └── io.py
```

## Quickstart
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure `config.yaml` (paths, CV, metrics, tracking).
3. Use modules in `src/` within notebooks or scripts to run tuning and evaluation.

## Notes on Tracking
- MLflow is optional and disabled by default via `config.yaml`.
- If disabled, runs are logged to CSV/JSON locally.



