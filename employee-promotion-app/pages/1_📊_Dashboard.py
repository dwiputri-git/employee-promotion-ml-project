"""
Dashboard page for the Employee Promotion Prediction App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.visualizations import (
    create_position_level_analysis, create_performance_analysis,
    create_prediction_table, create_kpi_cards
)

def show_dashboard():
    """Main dashboard page"""
    st.header("📊 Dashboard")
    
    # Load real data
    @st.cache_data
    def load_real_data():
        # Load the actual employee promotion data
        data_path = Path('employee-promotion-app/data/employee-promotion.csv')

        # Handle semicolon delimiter
        df = pd.read_csv(data_path, sep=';')

        # Clean the data (basic cleaning)
        df = df.dropna(subset=['Promotion_Eligible'])
        df['Promotion_Eligible'] = df['Promotion_Eligible'].astype(int)

        # Remove rows with negative values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        neg_mask = (df[numeric_cols] < 0).any(axis=1)
        df = df[~neg_mask]

        # Outliers by IQR (drop if <5%, else winsorize)
        for c in numeric_cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df[c] < lb) | (df[c] > ub)
            pct = 100 * mask.mean()
            if pct < 5:
                df = df[~mask]
            else:
                df[c] = np.where(df[c] < lb, lb, np.where(df[c] > ub, ub, df[c]))

        return df
    
    # Call function
    sample_data = load_real_data()

    # Mock predictions for demonstration
    np.random.seed(42)
    mock_predictions = np.random.choice([0, 1], len(sample_data), p=[0.7, 0.3])
    mock_probabilities = np.random.beta(2, 5, len(sample_data))
    
    sample_data['prediction'] = mock_predictions
    sample_data['probability'] = mock_probabilities
    sample_data['confidence'] = ['High' if p > 0.7 or p < 0.3 else 'Medium' for p in mock_probabilities]
    sample_data['recommendation'] = ['Promote' if p == 1 else 'Not Ready' for p in mock_predictions]
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(sample_data))
    
    with col2:
        st.metric("Predicted Promotions", int(sample_data['prediction'].sum()))
    
    with col3:
        promotion_rate = (sample_data['prediction'].sum() / len(sample_data)) * 100
        st.metric("Promotion Rate", f"{promotion_rate:.1f}%")
    
    with col4:
        high_conf = (sample_data['confidence'] == 'High').sum()
        st.metric("High Confidence", f"{high_conf}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Position level analysis
        fig = create_position_level_analysis(sample_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance analysis
        fig = create_performance_analysis(sample_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Prediction tables
    st.subheader("Recent Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Historical Predictions (Sample Data)**")
        historical_df = create_prediction_table(sample_data.head(10))
        st.dataframe(historical_df, use_container_width=True)
    
    with col2:
        st.write("**New Predictions (Uploaded Data)**")
        st.info("Upload new data in the Predictions page to see results here")
        
        # Show some mock new predictions
        new_predictions = sample_data.tail(5).copy()
        new_predictions['Employee_ID'] = [f'NEW_{i:03d}' for i in range(1, 6)]
        new_df = create_prediction_table(new_predictions)
        st.dataframe(new_df, use_container_width=True)

if __name__ == "__main__":
    show_dashboard()
