#!/usr/bin/env python3
"""Execute V3 notebooks in order."""
import subprocess
import sys
import os
from pathlib import Path

NOTEBOOKS = [
    '01_data_intro.ipynb',
    '02_data_cleaning.ipynb',
    '03_feature_engineering.ipynb',
    '04_model_pipeline_baselines.ipynb',
    '05_model_selection_cv.ipynb',
    '06_hyperparameter_tuning.ipynb',
    '07_calibration_threshold.ipynb',
    '08_explainability_fairness.ipynb',
    '09_error_business_analysis.ipynb',
]

V3_DIR = Path(__file__).parent
os.chdir(V3_DIR)

for nb in NOTEBOOKS:
    print(f'\nExecuting {nb}...')
    result = subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'notebook', '--execute', '--inplace', nb
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'ERROR in {nb}:', result.stderr[:500])
        sys.exit(1)
    print(f'✓ {nb} completed')

print('\nAll notebooks executed successfully!')

