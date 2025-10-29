"""
Visualization utilities for the Streamlit app
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt


def create_confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix"):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        colorscale='Reds',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=400,
        height=400
    )
    
    return fig


def create_roc_curve_plot(y_true, y_prob, title="ROC Curve"):
    """Create ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = np.trapz(tpr, fpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=500,
        height=400
    )
    
    return fig


def create_precision_recall_plot(y_true, y_prob, title="Precision-Recall Curve"):
    """Create precision-recall curve plot"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(precision, recall)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {pr_auc:.3f})',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=500,
        height=400
    )
    
    return fig


def create_feature_importance_plot(importance_dict, title="Feature Importance", max_features=10):
    """Create feature importance plot"""
    if not importance_dict:
        return None
    
    # Sort by absolute importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    features, importances = zip(*sorted_features[:max_features])
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker_color=['red' if x < 0 else 'blue' for x in importances]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Features",
        width=600,
        height=400
    )
    
    return fig


def create_probability_distribution_plot(probabilities, title="Probability Distribution"):
    """Create probability distribution plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=probabilities,
        nbinsx=20,
        name="Probability Distribution",
        marker_color='lightblue'
    ))
    
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold (0.5)"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Prediction Probability",
        yaxis_title="Count",
        width=500,
        height=400
    )
    
    return fig


def create_position_level_analysis(df, title="Promotion Rate by Position Level"):
    """Create promotion rate analysis by position level"""
    if 'Current_Position_Level' not in df.columns:
        return None
    
    position_analysis = df.groupby('Current_Position_Level').agg({
        'prediction': ['count', 'sum', 'mean'],
        'probability': 'mean'
    }).round(3)
    
    position_analysis.columns = ['Total', 'Promotions', 'Promotion_Rate', 'Avg_Probability']
    position_analysis = position_analysis.reset_index()
    
    fig = go.Figure()
    
    # Promotion rate
    fig.add_trace(go.Bar(
        x=position_analysis['Current_Position_Level'],
        y=position_analysis['Promotion_Rate'],
        name='Promotion Rate',
        marker_color='lightblue'
    ))
    
    # Average probability
    fig.add_trace(go.Scatter(
        x=position_analysis['Current_Position_Level'],
        y=position_analysis['Avg_Probability'],
        mode='lines+markers',
        name='Avg Probability',
        yaxis='y2',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Position Level",
        yaxis_title="Promotion Rate",
        yaxis2=dict(
            title="Average Probability",
            overlaying="y",
            side="right"
        ),
        width=600,
        height=400
    )
    
    return fig


def create_performance_analysis(df, title="Performance vs Promotion Probability"):
    """Create performance analysis plot"""
    if 'Performance_Score' not in df.columns:
        return None
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df['Performance_Score'],
        y=df['probability'],
        mode='markers',
        marker=dict(
            color=df['prediction'],
            colorscale=['red', 'green'],
            size=8,
            opacity=0.7
        ),
        text=df.get('Employee_ID', ''),
        hovertemplate='Performance: %{x}<br>Probability: %{y:.3f}<br>Prediction: %{marker.color}<extra></extra>'
    ))
    
    # Threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="black",
        annotation_text="Threshold"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Performance Score",
        yaxis_title="Promotion Probability",
        width=600,
        height=400
    )
    
    return fig


def create_kpi_cards(metrics_dict):
    """Create KPI cards for dashboard"""
    cards = []
    
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            value = f"{value:.2f}"
        
        cards.append({
            'title': key.replace('_', ' ').title(),
            'value': value
        })
    
    return cards


def create_prediction_table(df, max_rows=20):
    """Create styled prediction table"""
    # Select relevant columns for display
    display_cols = ['Employee_ID', 'prediction', 'probability', 'confidence', 'recommendation']
    available_cols = [col for col in display_cols if col in df.columns]
    
    if not available_cols:
        return df.head(max_rows)
    
    display_df = df[available_cols].head(max_rows).copy()
    
    # Format columns
    if 'probability' in display_df.columns:
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.3f}")
    
    if 'prediction' in display_df.columns:
        display_df['prediction'] = display_df['prediction'].apply(lambda x: 'Yes' if x == 1 else 'No')
    
    return display_df


def create_insights_summary(insights_dict):
    """Create insights summary for display using Streamlit components"""
    import streamlit as st
    
    # Summary metrics
    if 'summary' in insights_dict:
        st.subheader("📊 Summary")
        col1, col2 = st.columns(2)
        
        summary_items = list(insights_dict['summary'].items())
        for i, (key, value) in enumerate(summary_items):
            with col1 if i % 2 == 0 else col2:
                st.metric(
                    label=key.replace('_', ' ').title(),
                    value=value
                )
    
    # Patterns
    if 'patterns' in insights_dict and insights_dict['patterns']:
        st.subheader("🔍 Patterns")
        for pattern in insights_dict['patterns']:
            st.write(f"• {pattern}")
    
    # Risks
    if 'risks' in insights_dict and insights_dict['risks']:
        st.subheader("⚠️ Risks")
        for risk in insights_dict['risks']:
            st.warning(f"• {risk}")
    
    # Recommendations
    if 'recommendations' in insights_dict and insights_dict['recommendations']:
        st.subheader("💡 Recommendations")
        for rec in insights_dict['recommendations']:
            st.info(f"• {rec}")


def create_insights_summary_html(insights_dict):
    """Create insights summary for display as HTML (fallback)"""
    summary_html = """
    <div style='font-family: Arial, sans-serif; line-height: 1.6;'>
    """
    
    # Summary metrics
    if 'summary' in insights_dict:
        summary_html += """
        <h3 style='color: #1f77b4; margin-top: 20px; margin-bottom: 10px;'>📊 Summary</h3>
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;'>
        """
        for key, value in insights_dict['summary'].items():
            summary_html += f"<div style='margin: 8px 0;'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
        summary_html += "</div>"
    
    # Patterns
    if 'patterns' in insights_dict and insights_dict['patterns']:
        summary_html += """
        <h3 style='color: #28a745; margin-top: 20px; margin-bottom: 10px;'>🔍 Patterns</h3>
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;'>
        """
        for pattern in insights_dict['patterns']:
            summary_html += f"<div style='margin: 8px 0;'>• {pattern}</div>"
        summary_html += "</div>"
    
    # Risks
    if 'risks' in insights_dict and insights_dict['risks']:
        summary_html += """
        <h3 style='color: #dc3545; margin-top: 20px; margin-bottom: 10px;'>⚠️ Risks</h3>
        <div style='background-color: #fff5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545;'>
        """
        for risk in insights_dict['risks']:
            summary_html += f"<div style='margin: 8px 0; color: #dc3545;'>• {risk}</div>"
        summary_html += "</div>"
    
    # Recommendations
    if 'recommendations' in insights_dict and insights_dict['recommendations']:
        summary_html += """
        <h3 style='color: #ffc107; margin-top: 20px; margin-bottom: 10px;'>💡 Recommendations</h3>
        <div style='background-color: #fffbf0; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;'>
        """
        for rec in insights_dict['recommendations']:
            summary_html += f"<div style='margin: 8px 0; color: #856404;'>• {rec}</div>"
        summary_html += "</div>"
    
    summary_html += "</div>"
    
    return summary_html
