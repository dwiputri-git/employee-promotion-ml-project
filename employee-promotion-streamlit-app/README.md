# Employee Promotion Prediction App

A Streamlit application for predicting employee promotion eligibility using machine learning.

## Features

- **Dashboard**: View KPIs, model performance, and prediction tables
- **Predictions**: Upload CSV or input data manually for predictions
- **Model Analysis**: Detailed model evaluation and fairness analysis
- **AI Insights**: AI-powered recommendations and pattern analysis

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Usage

1. **Dashboard**: View overall model performance and recent predictions
2. **Predictions**: Upload new data or use form input to get predictions
3. **Model Analysis**: Explore model metrics, confusion matrix, and fairness analysis
4. **AI Insights**: Get AI-generated recommendations based on predictions

## Model

Uses the trained Logistic Regression model from V3 with:
- PR-AUC: 0.350
- Accuracy: 0.544
- Calibrated threshold: 0.209
