"""
Predictions page for the Employee Promotion Prediction App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.insights.ai_generator import AIInsightsGenerator
from src.utils.visualizations import create_prediction_table

def show_predictions():
    """Predictions page"""
    st.header("🔮 Predictions")
    
    # Data input methods
    input_method = st.radio(
        "Choose input method:",
        ["Upload CSV File", "Manual Form Input"]
    )
    
    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with employee data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} rows")
                
                # Show data preview
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head())
                
                # Data validation
                required_columns = ['Age', 'Years_at_Company', 'Performance_Score', 
                                  'Leadership_Score', 'Training_Hours', 'Projects_Handled',
                                  'Current_Position_Level', 'Peer_Review_Score']
                
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing required columns: {missing_cols}")
                else:
                    st.success("All required columns present")
                
                # Process and predict
                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Processing data and generating predictions..."):
                        # Mock predictions for demo
                        np.random.seed(42)
                        predictions = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
                        probabilities = np.random.beta(2, 5, len(df))
                        
                        df['prediction'] = predictions
                        df['probability'] = probabilities
                        df['confidence'] = ['High' if p > 0.7 or p < 0.3 else 'Medium' for p in probabilities]
                        df['recommendation'] = ['Promote' if p == 1 else 'Not Ready' for p in predictions]
                        
                        st.success("Predictions generated successfully!")
                        
                        # Show results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Predictions", len(df))
                            st.metric("Promotions Recommended", int(predictions.sum()))
                            promotion_rate = (predictions.sum() / len(df)) * 100
                            st.metric("Promotion Rate", f"{promotion_rate:.1f}%")
                        
                        with col2:
                            high_conf = (np.array(df['confidence']) == 'High').sum()
                            st.metric("High Confidence", high_conf)
                            avg_prob = probabilities.mean()
                            st.metric("Average Probability", f"{avg_prob:.3f}")
                        
                        # Results table
                        st.subheader("Prediction Results")
                        st.dataframe(create_prediction_table(df), use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="promotion_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    elif input_method == "Manual Form Input":
        st.subheader("Employee Information Form")
        
        col1, col2 = st.columns(2)
        
        with col1:
            employee_id = st.text_input("Employee ID", value="EMP_001")
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=30, value=3)
            performance_score = st.slider("Performance Score", 0, 100, 75)
            leadership_score = st.slider("Leadership Score", 0, 100, 70)
        
        with col2:
            training_hours = st.number_input("Training Hours", min_value=0, max_value=200, value=40)
            projects_handled = st.number_input("Projects Handled", min_value=0, max_value=20, value=5)
            position_level = st.selectbox("Position Level", ["Junior", "Mid", "Senior", "Lead"])
            peer_review_score = st.slider("Peer Review Score", 0, 100, 80)
        
        if st.button("Predict Promotion", type="primary"):
            # Create single employee data
            employee_data = {
                'Employee_ID': employee_id,
                'Age': age,
                'Years_at_Company': years_at_company,
                'Performance_Score': performance_score,
                'Leadership_Score': leadership_score,
                'Training_Hours': training_hours,
                'Projects_Handled': projects_handled,
                'Current_Position_Level': position_level,
                'Peer_Review_Score': peer_review_score
            }
            
            # Mock prediction
            np.random.seed(42)
            probability = np.random.beta(2, 5)
            prediction = 1 if probability > 0.5 else 0
            confidence = 'High' if probability > 0.7 or probability < 0.3 else 'Medium'
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "Promote" if prediction == 1 else "Not Ready")
            
            with col2:
                st.metric("Probability", f"{probability:.3f}")
            
            with col3:
                st.metric("Confidence", confidence)
            
            # Generate insight
            ai_generator = AIInsightsGenerator()
            insight = ai_generator.generate_employee_insight(employee_data, prediction, probability)
            
            st.info(insight)
            
            # Show employee details
            with st.expander("Employee Details", expanded=False):
                for key, value in employee_data.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    else:  # Use Sample Data
        st.subheader("Sample Data Predictions")
        st.write("Using sample data for demonstration")
        
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
            for col in numeric_cols:
                if col != 'Promotion_Eligible':
                    df = df[df[col] >= 0]
            
            return df
        
        sample_data = load_real_data()
        
        if st.button("Generate Sample Predictions", type="primary"):
            # Mock predictions
            np.random.seed(42)
            predictions = np.random.choice([0, 1], len(sample_data), p=[0.7, 0.3])
            probabilities = np.random.beta(2, 5, len(sample_data))
            
            sample_data['prediction'] = predictions
            sample_data['probability'] = probabilities
            sample_data['confidence'] = ['High' if p > 0.7 or p < 0.3 else 'Medium' for p in probabilities]
            sample_data['recommendation'] = ['Promote' if p == 1 else 'Not Ready' for p in predictions]
            
            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Employees", len(sample_data))
            
            with col2:
                st.metric("Promotions", int(predictions.sum()))
            
            with col3:
                promotion_rate = (predictions.sum() / len(sample_data)) * 100
                st.metric("Promotion Rate", f"{promotion_rate:.1f}%")
            
            with col4:
                high_conf = (np.array(sample_data['confidence']) == 'High').sum()
                st.metric("High Confidence", high_conf)
            
            # Show results table
            st.subheader("Prediction Results")
            st.dataframe(create_prediction_table(sample_data), use_container_width=True)
            
            # Download option
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="sample_promotion_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    show_predictions()
