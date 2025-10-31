"""
Employee Promotion Prediction App
Main Streamlit application
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.predictor import create_predictor_from_config, batch_predict
from src.data.preprocessor import preprocess_pipeline
from src.insights.ai_generator import AIInsightsGenerator
from src.utils.visualizations import (
    create_confusion_matrix_plot, create_roc_curve_plot, 
    create_precision_recall_plot, create_feature_importance_plot,
    create_probability_distribution_plot, create_position_level_analysis,
    create_performance_analysis, create_kpi_cards, create_prediction_table,
    create_insights_summary
)

# Page configuration
st.set_page_config(
    page_title="Employee Promotion Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-table {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_real_data():
    """Load real employee promotion data"""
    # Load the actual employee promotion data
    data_path = Path('../data/employee-promotion.csv')
    
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

@st.cache_resource
def load_model():
    """Load the trained model (placeholder for now)"""
    # For now, return a mock predictor
    # In production, this would load the actual V3 model
    return None

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Employee Promotion Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Dashboard", "🔮 Predictions", "📈 Model Analysis", "🤖 AI Insights"]
    )
    
    # Load real data
    sample_data = load_real_data()
    
    if page == "📊 Dashboard":
        show_dashboard(sample_data)
    elif page == "🔮 Predictions":
        show_predictions(sample_data)
    elif page == "📈 Model Analysis":
        show_model_analysis(sample_data)
    elif page == "🤖 AI Insights":
        show_ai_insights(sample_data)

def show_dashboard(sample_data):
    """Dashboard page"""
    st.header("📊 Dashboard")
    
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

def show_predictions(sample_data):
    """Predictions page"""
    st.header("🔮 Predictions")
    
    # Data input methods
    input_method = st.radio(
        "Choose input method:",
        ["Upload CSV File", "Manual Form Input", "Use Sample Data"]
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
                st.write("Data preview:")
                st.dataframe(df.head())
                
                # Process and predict
                if st.button("Generate Predictions"):
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
                        st.dataframe(create_prediction_table(df), use_container_width=True)
                        
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
        
        if st.button("Predict Promotion"):
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
    
    else:  # Use Sample Data
        st.subheader("Sample Data Predictions")
        st.write("Using sample data for demonstration")
        
        if st.button("Generate Sample Predictions"):
            # Mock predictions
            np.random.seed(42)
            predictions = np.random.choice([0, 1], len(sample_data), p=[0.7, 0.3])
            probabilities = np.random.beta(2, 5, len(sample_data))
            
            sample_data['prediction'] = predictions
            sample_data['probability'] = probabilities
            sample_data['confidence'] = ['High' if p > 0.7 or p < 0.3 else 'Medium' for p in probabilities]
            sample_data['recommendation'] = ['Promote' if p == 1 else 'Not Ready' for p in predictions]
            
            st.dataframe(create_prediction_table(sample_data), use_container_width=True)

def show_model_analysis(sample_data):
    """Model analysis page"""
    st.header("📈 Model Analysis")
    
    # Mock data for analysis
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 100, p=[0.7, 0.3])
    y_prob = np.random.beta(2, 5, 100)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = (y_true == y_pred).mean()
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        precision = (y_true[y_pred == 1] == 1).mean() if (y_pred == 1).sum() > 0 else 0
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).sum() > 0 else 0
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        st.metric("F1-Score", f"{f1:.3f}")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        fig = create_confusion_matrix_plot(y_true, y_pred)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC Curve
        fig = create_roc_curve_plot(y_true, y_prob)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision-Recall Curve
        fig = create_precision_recall_plot(y_true, y_prob)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Probability Distribution
        fig = create_probability_distribution_plot(y_prob)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (mock)
    st.subheader("Feature Importance")
    mock_importance = {
        'Performance_Score': 0.45,
        'Leadership_Score': 0.32,
        'Years_at_Company': 0.28,
        'Training_Hours': 0.15,
        'Projects_Handled': 0.12,
        'Age': -0.08,
        'Peer_Review_Score': 0.06
    }
    
    fig = create_feature_importance_plot(mock_importance)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

def show_ai_insights(sample_data):
    """AI insights page"""
    st.header("🤖 AI Insights & Recommendations")
    
    # Generate mock insights
    ai_generator = AIInsightsGenerator()
    
    # Mock predictions
    np.random.seed(42)
    mock_predictions = np.random.choice([0, 1], len(sample_data), p=[0.7, 0.3])
    mock_probabilities = np.random.beta(2, 5, len(sample_data))
    
    sample_data['prediction'] = mock_predictions
    sample_data['probability'] = mock_probabilities
    sample_data['confidence'] = ['High' if p > 0.7 or p < 0.3 else 'Medium' for p in mock_probabilities]
    
    # Generate insights
    insights = ai_generator.generate_insights(sample_data)
    
    # Display insights using Streamlit components
    create_insights_summary(insights)
    
    st.divider()
    
    # Individual employee insights
    st.subheader("Individual Employee Insights")
    
    # Select employee
    employee_options = sample_data['Employee_ID'].tolist()
    selected_employee = st.selectbox("Select Employee", employee_options)
    
    if selected_employee:
        employee_data = sample_data[sample_data['Employee_ID'] == selected_employee].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Employee Details**")
            st.write(f"**ID:** {employee_data['Employee_ID']}")
            st.write(f"**Position:** {employee_data['Current_Position_Level']}")
            st.write(f"**Performance Score:** {employee_data['Performance_Score']:.1f}")
            st.write(f"**Leadership Score:** {employee_data['Leadership_Score']:.1f}")
            st.write(f"**Years at Company:** {employee_data['Years_at_Company']}")
        
        with col2:
            st.write("**Prediction Results**")
            prediction = employee_data['prediction']
            probability = employee_data['probability']
            
            st.metric("Prediction", "Promote" if prediction == 1 else "Not Ready")
            st.metric("Probability", f"{probability:.3f}")
            st.metric("Confidence", employee_data['confidence'])
        
        # Generate personalized insight
        insight = ai_generator.generate_employee_insight(
            employee_data.to_dict(), prediction, probability
        )
        
        st.info(insight)
    
    # Batch insights
    st.subheader("Batch Analysis")
    
    if st.button("Generate Batch Insights"):
        with st.spinner("Analyzing all employees..."):
            batch_insights = ai_generator.generate_insights(sample_data)
            
            st.success("Batch analysis completed!")
            create_insights_summary(batch_insights)

if __name__ == "__main__":
    main()
