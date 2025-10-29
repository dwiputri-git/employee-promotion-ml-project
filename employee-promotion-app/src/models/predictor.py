"""
Prediction engine for employee promotion model
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional
import yaml


class PromotionPredictor:
    """Main prediction class for employee promotion model"""
    
    def __init__(self, model_path: str, preprocessor_path: str, 
                 feature_names_path: str, threshold: float = 0.209):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.feature_names_path = feature_names_path
        self.threshold = threshold
        
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
    
    def load_model(self):
        """Load trained model and preprocessor"""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load preprocessor
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            # Load feature names
            self.feature_names = joblib.load(self.feature_names_path)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X: Input features (DataFrame or array)
            
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Prediction probabilities
        """
        if not self.is_loaded:
            if not self.load_model():
                raise ValueError("Model not loaded successfully")
        
        # Transform features if needed
        if hasattr(self.preprocessor, 'transform'):
            X_transformed = self.preprocessor.transform(X)
        else:
            X_transformed = X
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_transformed)[:, 1]
        
        # Apply threshold
        predictions = (probabilities >= self.threshold).astype(int)
        
        return predictions, probabilities
    
    def predict_single(self, features: dict) -> Tuple[int, float, dict]:
        """
        Predict for a single employee
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            prediction: Binary prediction (0 or 1)
            probability: Prediction probability
            details: Additional prediction details
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        predictions, probabilities = self.predict(df)
        
        prediction = predictions[0]
        probability = probabilities[0]
        
        # Create details
        details = {
            'prediction': prediction,
            'probability': probability,
            'confidence': 'High' if probability > 0.7 or probability < 0.3 else 'Medium',
            'threshold_used': self.threshold,
            'above_threshold': probability >= self.threshold
        }
        
        return prediction, probability, details
    
    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance if available"""
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        if hasattr(self.model, 'coef_'):
            # Logistic regression coefficients
            importance = dict(zip(self.feature_names, self.model.coef_[0]))
            return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based feature importance
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return None
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if not self.is_loaded:
            return {'status': 'Not loaded'}
        
        info = {
            'model_type': type(self.model).__name__,
            'threshold': self.threshold,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names[:10] if self.feature_names else []  # First 10 features
        }
        
        return info


def load_config(config_path: str = "config/app_config.yaml") -> dict:
    """Load app configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_predictor_from_config(config_path: str = "config/app_config.yaml") -> PromotionPredictor:
    """Create predictor from configuration file"""
    config = load_config(config_path)
    model_config = config['model']
    
    return PromotionPredictor(
        model_path=model_config['path'],
        preprocessor_path=model_config['preprocessor_path'],
        feature_names_path=model_config['feature_names_path'],
        threshold=model_config['threshold']
    )


def batch_predict(df: pd.DataFrame, predictor: PromotionPredictor) -> pd.DataFrame:
    """
    Make batch predictions and return DataFrame with results
    
    Args:
        df: Input DataFrame
        predictor: Trained predictor
        
    Returns:
        DataFrame with predictions and probabilities
    """
    predictions, probabilities = predictor.predict(df)
    
    result_df = df.copy()
    result_df['prediction'] = predictions
    result_df['probability'] = probabilities
    result_df['confidence'] = result_df['probability'].apply(
        lambda x: 'High' if x > 0.7 or x < 0.3 else 'Medium'
    )
    result_df['recommendation'] = result_df['prediction'].apply(
        lambda x: 'Promote' if x == 1 else 'Not Ready'
    )
    
    return result_df
