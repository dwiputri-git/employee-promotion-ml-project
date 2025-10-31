"""
AI-powered insights and recommendations generator
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json


class AIInsightsGenerator:
    """Generate AI-powered insights and recommendations"""
    
    def __init__(self):
        self.insights_templates = {
            'promotion_patterns': [
                "High performers with leadership scores above {threshold} show {pct}% promotion rate",
                "Employees with {feature} in top quartile have {pct}% higher promotion probability",
                "Tenure of {years}+ years combined with high performance shows strong promotion signal"
            ],
            'risk_factors': [
                "Low performance scores (< {threshold}) indicate {pct}% lower promotion probability",
                "Employees in {position} position show concerning promotion patterns",
                "Recent performance decline detected in {count} employees"
            ],
            'recommendations': [
                "Consider targeted training for {count} high-potential employees",
                "Implement leadership development program for {position} level",
                "Review promotion criteria for {feature} to ensure fairness",
                "Schedule performance reviews for {count} borderline cases"
            ]
        }
    
    def generate_insights(self, predictions_df: pd.DataFrame, 
                         historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate comprehensive insights from predictions
        
        Args:
            predictions_df: DataFrame with predictions and probabilities
            historical_data: Historical data for comparison
            
        Returns:
            Dictionary of insights and recommendations
        """
        insights = {
            'summary': self._generate_summary(predictions_df),
            'patterns': self._analyze_patterns(predictions_df),
            'risks': self._identify_risks(predictions_df),
            'recommendations': self._generate_recommendations(predictions_df),
            'metrics': self._calculate_metrics(predictions_df)
        }
        
        return insights
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_employees = len(df)
        predicted_promotions = df['prediction'].sum()
        promotion_rate = (predicted_promotions / total_employees) * 100
        
        high_confidence = len(df[df['confidence'] == 'High'])
        avg_probability = df['probability'].mean()
        
        return {
            'total_employees': total_employees,
            'predicted_promotions': int(predicted_promotions),
            'promotion_rate': round(promotion_rate, 1),
            'high_confidence_predictions': int(high_confidence),
            'average_probability': round(avg_probability, 3)
        }
    
    def _analyze_patterns(self, df: pd.DataFrame) -> List[str]:
        """Analyze promotion patterns"""
        patterns = []
        
        # Performance score patterns
        if 'Performance_Score' in df.columns:
            high_perf = df[df['Performance_Score'] >= df['Performance_Score'].quantile(0.75)]
            high_perf_promotion = (high_perf['prediction'] == 1).mean() * 100
            patterns.append(f"High performers (top 25%) show {high_perf_promotion:.1f}% promotion rate")
        
        # Leadership score patterns
        if 'Leadership_Score' in df.columns:
            high_lead = df[df['Leadership_Score'] >= df['Leadership_Score'].quantile(0.75)]
            high_lead_promotion = (high_lead['prediction'] == 1).mean() * 100
            patterns.append(f"High leadership scores show {high_lead_promotion:.1f}% promotion rate")
        
        # Tenure patterns
        if 'Years_at_Company' in df.columns:
            senior_employees = df[df['Years_at_Company'] >= 5]
            if len(senior_employees) > 0:
                senior_promotion = (senior_employees['prediction'] == 1).mean() * 100
                patterns.append(f"Senior employees (5+ years) show {senior_promotion:.1f}% promotion rate")
        
        # Position level patterns
        if 'Current_Position_Level' in df.columns:
            for level in df['Current_Position_Level'].unique():
                level_data = df[df['Current_Position_Level'] == level]
                if len(level_data) > 0:
                    level_promotion = (level_data['prediction'] == 1).mean() * 100
                    patterns.append(f"{level} level shows {level_promotion:.1f}% promotion rate")
        
        return patterns
    
    def _identify_risks(self, df: pd.DataFrame) -> List[str]:
        """Identify potential risks and concerns"""
        risks = []
        
        # Low performance risks
        if 'Performance_Score' in df.columns:
            low_perf = df[df['Performance_Score'] < df['Performance_Score'].quantile(0.25)]
            if len(low_perf) > 0:
                low_perf_promotion = (low_perf['prediction'] == 1).mean() * 100
                risks.append(f"Low performers show {low_perf_promotion:.1f}% promotion rate - potential bias concern")
        
        # High probability but low performance
        if 'Performance_Score' in df.columns:
            concerning = df[(df['probability'] > 0.7) & (df['Performance_Score'] < df['Performance_Score'].median())]
            if len(concerning) > 0:
                risks.append(f"{len(concerning)} employees with high promotion probability but below-average performance")
        
        # Position level bias
        if 'Current_Position_Level' in df.columns:
            for level in df['Current_Position_Level'].unique():
                level_data = df[df['Current_Position_Level'] == level]
                if len(level_data) > 0:
                    level_promotion = (level_data['prediction'] == 1).mean() * 100
                    if level_promotion == 0:
                        risks.append(f"No promotions predicted for {level} level - potential bias")
        
        return risks
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High potential employees
        high_potential = df[(df['probability'] > 0.6) & (df['prediction'] == 1)]
        if len(high_potential) > 0:
            recommendations.append(f"Focus on {len(high_potential)} high-potential employees for immediate promotion")
        
        # Borderline cases
        borderline = df[(df['probability'] >= 0.3) & (df['probability'] <= 0.7)]
        if len(borderline) > 0:
            recommendations.append(f"Review {len(borderline)} borderline cases for additional assessment")
        
        # Development needs
        low_prob_high_perf = df[(df['probability'] < 0.3) & (df['Performance_Score'] >= df['Performance_Score'].quantile(0.75))]
        if len(low_prob_high_perf) > 0:
            recommendations.append(f"Provide leadership development for {len(low_prob_high_perf)} high performers with low promotion probability")
        
        # Training recommendations
        if 'Training_Hours' in df.columns:
            low_training = df[df['Training_Hours'] < df['Training_Hours'].quantile(0.25)]
            if len(low_training) > 0:
                recommendations.append(f"Consider additional training for {len(low_training)} employees with low training hours")
        
        return recommendations
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key metrics"""
        metrics = {
            'promotion_rate': (df['prediction'] == 1).mean() * 100,
            'avg_probability': df['probability'].mean(),
            'high_confidence_rate': (df['confidence'] == 'High').mean() * 100,
            'probability_std': df['probability'].std()
        }
        
        return {k: round(v, 3) for k, v in metrics.items()}
    
    def generate_employee_insight(self, employee_data: Dict[str, Any], 
                                 prediction: int, probability: float) -> str:
        """Generate personalized insight for a single employee"""
        name = employee_data.get('Employee_ID', 'Employee')
        
        if prediction == 1:
            if probability > 0.8:
                insight = f"🚀 {name} is a strong promotion candidate with {probability:.1%} probability. Consider fast-tracking their promotion."
            else:
                insight = f"✅ {name} shows promotion potential with {probability:.1%} probability. Monitor their performance closely."
        else:
            if probability < 0.2:
                insight = f"📈 {name} needs development before promotion consideration. Focus on skill building and performance improvement."
            else:
                insight = f"⏳ {name} is close to promotion readiness with {probability:.1%} probability. Provide targeted development opportunities."
        
        return insight


def create_sample_insights() -> Dict[str, Any]:
    """Create sample insights for demonstration"""
    return {
        'summary': {
            'total_employees': 50,
            'predicted_promotions': 15,
            'promotion_rate': 30.0,
            'high_confidence_predictions': 25,
            'average_probability': 0.45
        },
        'patterns': [
            "High performers (top 25%) show 65.2% promotion rate",
            "High leadership scores show 58.7% promotion rate",
            "Senior employees (5+ years) show 42.1% promotion rate"
        ],
        'risks': [
            "Low performers show 12.5% promotion rate - potential bias concern",
            "2 employees with high promotion probability but below-average performance"
        ],
        'recommendations': [
            "Focus on 8 high-potential employees for immediate promotion",
            "Review 12 borderline cases for additional assessment",
            "Provide leadership development for 3 high performers with low promotion probability"
        ],
        'metrics': {
            'promotion_rate': 30.0,
            'avg_probability': 0.45,
            'high_confidence_rate': 50.0,
            'probability_std': 0.23
        }
    }

