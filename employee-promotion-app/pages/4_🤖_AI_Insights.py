"""
AI Insights page for the Employee Promotion Prediction App
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
from src.utils.visualizations import create_insights_summary

def show_ai_insights():
    """AI insights page"""
    st.header("🤖 AI Insights & Recommendations")
    
    # Load real data
    @st.cache_data
    def load_real_data():
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
    
    sample_data = load_real_data()
    
    # Generate mock predictions
    np.random.seed(42)
    mock_predictions = np.random.choice([0, 1], len(sample_data), p=[0.7, 0.3])
    mock_probabilities = np.random.beta(2, 5, len(sample_data))
    
    sample_data['prediction'] = mock_predictions
    sample_data['probability'] = mock_probabilities
    sample_data['confidence'] = ['High' if p > 0.7 or p < 0.3 else 'Medium' for p in mock_probabilities]
    
    # Initialize AI generator
    ai_generator = AIInsightsGenerator()
    
    # Generate insights
    insights = ai_generator.generate_insights(sample_data)
    
    # Display insights using Streamlit components
    create_insights_summary(insights)
    
    st.divider()
    
    # Individual employee insights
    st.subheader("Individual Employee Analysis")
    
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
            st.write(f"**Training Hours:** {employee_data['Training_Hours']:.1f}")
            st.write(f"**Projects Handled:** {employee_data['Projects_Handled']}")
        
        with col2:
            st.write("**Prediction Results**")
            prediction = employee_data['prediction']
            probability = employee_data['probability']
            
            st.metric("Prediction", "Promote" if prediction == 1 else "Not Ready")
            st.metric("Probability", f"{probability:.3f}")
            st.metric("Confidence", employee_data['confidence'])
            
            # Additional metrics
            st.write("**Performance vs Peers**")
            perf_percentile = (sample_data['Performance_Score'] <= employee_data['Performance_Score']).mean() * 100
            st.write(f"Performance Percentile: {perf_percentile:.1f}%")
            
            lead_percentile = (sample_data['Leadership_Score'] <= employee_data['Leadership_Score']).mean() * 100
            st.write(f"Leadership Percentile: {lead_percentile:.1f}%")
        
        # Generate personalized insight
        insight = ai_generator.generate_employee_insight(
            employee_data.to_dict(), prediction, probability
        )
        
        st.info(insight)
        
        # Detailed analysis
        with st.expander("Detailed Analysis", expanded=False):
            st.write("**Strengths:**")
            if employee_data['Performance_Score'] > sample_data['Performance_Score'].quantile(0.75):
                st.write("✅ High performance score")
            if employee_data['Leadership_Score'] > sample_data['Leadership_Score'].quantile(0.75):
                st.write("✅ Strong leadership skills")
            if employee_data['Years_at_Company'] > sample_data['Years_at_Company'].quantile(0.75):
                st.write("✅ Long tenure with company")
            
            st.write("**Areas for Improvement:**")
            if employee_data['Performance_Score'] < sample_data['Performance_Score'].quantile(0.5):
                st.write("⚠️ Performance score below median")
            if employee_data['Leadership_Score'] < sample_data['Leadership_Score'].quantile(0.5):
                st.write("⚠️ Leadership score below median")
            if employee_data['Training_Hours'] < sample_data['Training_Hours'].quantile(0.5):
                st.write("⚠️ Training hours below median")
    
    st.divider()
    
    # Batch insights
    st.subheader("Batch Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Batch Insights", type="primary"):
            with st.spinner("Analyzing all employees..."):
                batch_insights = ai_generator.generate_insights(sample_data)
                
                st.success("Batch analysis completed!")
                create_insights_summary(batch_insights)
    
    with col2:
        if st.button("Export Insights Report"):
            # Create a comprehensive report
            report = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_employees': len(sample_data),
                'insights': insights
            }
            
            # Convert to JSON for download
            import json
            json_report = json.dumps(report, indent=2, default=str)
            
            st.download_button(
                label="Download Insights Report",
                data=json_report,
                file_name=f"insights_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.divider()
    
    # Pattern analysis
    st.subheader("Pattern Analysis")
    
    # Performance vs Promotion correlation
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Score Distribution**")
        high_perf = sample_data[sample_data['Performance_Score'] >= sample_data['Performance_Score'].quantile(0.75)]
        high_perf_promotion = (high_perf['prediction'] == 1).mean() * 100
        st.metric("High Performers Promotion Rate", f"{high_perf_promotion:.1f}%")
        
        low_perf = sample_data[sample_data['Performance_Score'] < sample_data['Performance_Score'].quantile(0.25)]
        low_perf_promotion = (low_perf['prediction'] == 1).mean() * 100
        st.metric("Low Performers Promotion Rate", f"{low_perf_promotion:.1f}%")
    
    with col2:
        st.write("**Leadership Score Distribution**")
        high_lead = sample_data[sample_data['Leadership_Score'] >= sample_data['Leadership_Score'].quantile(0.75)]
        high_lead_promotion = (high_lead['prediction'] == 1).mean() * 100
        st.metric("High Leadership Promotion Rate", f"{high_lead_promotion:.1f}%")
        
        low_lead = sample_data[sample_data['Leadership_Score'] < sample_data['Leadership_Score'].quantile(0.25)]
        low_lead_promotion = (low_lead['prediction'] == 1).mean() * 100
        st.metric("Low Leadership Promotion Rate", f"{low_lead_promotion:.1f}%")
    
    # Position level analysis
    st.write("**Promotion Rate by Position Level**")
    position_analysis = sample_data.groupby('Current_Position_Level').agg({
        'prediction': ['count', 'sum', 'mean'],
        'probability': 'mean'
    }).round(3)
    
    position_analysis.columns = ['Total', 'Promotions', 'Promotion_Rate', 'Avg_Probability']
    position_analysis = position_analysis.reset_index()
    
    st.dataframe(position_analysis, use_container_width=True)
    
    # Risk analysis
    st.subheader("Risk Analysis")
    
    # High probability but low performance
    concerning = sample_data[
        (sample_data['probability'] > 0.7) & 
        (sample_data['Performance_Score'] < sample_data['Performance_Score'].median())
    ]
    
    if len(concerning) > 0:
        st.warning(f"⚠️ {len(concerning)} employees have high promotion probability but below-average performance")
        st.dataframe(concerning[['Employee_ID', 'Performance_Score', 'probability']], use_container_width=True)
    else:
        st.success("✅ No concerning cases detected")
    
    # Low probability but high performance
    missed_opportunities = sample_data[
        (sample_data['probability'] < 0.3) & 
        (sample_data['Performance_Score'] > sample_data['Performance_Score'].quantile(0.75))
    ]
    
    if len(missed_opportunities) > 0:
        st.info(f"💡 {len(missed_opportunities)} high performers have low promotion probability - potential missed opportunities")
        st.dataframe(missed_opportunities[['Employee_ID', 'Performance_Score', 'probability']], use_container_width=True)
    else:
        st.success("✅ No missed opportunities detected")

if __name__ == "__main__":
    show_ai_insights()
