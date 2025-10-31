"""
Promotion prediction module
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import yaml


class PromotionPredictor:
    """Promotion prediction model wrapper"""
    
    def __init__(self, model, preprocessor=None, threshold=0.5):
        """
        Initialize predictor
        
        Args:
            model: Trained model (e.g., sklearn model)
            preprocessor: Optional preprocessor pipeline
            threshold: Decision threshold for binary classification
        """
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Preprocess if preprocessor available
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)[:, 1]
        else:
            # Fallback for models without predict_proba
            probabilities = np.ones(len(X_processed)) * 0.5
        
        # Apply threshold
        predictions = (probabilities >= self.threshold).astype(int)
        
        return predictions, probabilities
    
    def predict_single(self, X: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict for single instance
        
        Args:
            X: Single row DataFrame
            
        Returns:
            Tuple of (prediction, probability)
        """
        predictions, probabilities = self.predict(X)
        return int(predictions[0]), float(probabilities[0])


def create_predictor_from_config(config_path: Optional[str] = None) -> PromotionPredictor:
    """
    Create predictor from config file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configured PromotionPredictor instance
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "app_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    
    # Load model
    model_path = Path(__file__).parent.parent.parent / model_config.get('path', 'models/trained_model.pkl')
    
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        # Return mock predictor if model doesn't exist
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        # Fit on dummy data so it can be used
        X_dummy = np.random.randn(10, 5)
        y_dummy = np.random.choice([0, 1], 10)
        model.fit(X_dummy, y_dummy)
    
    # Load preprocessor if available
    preprocessor_path = Path(__file__).parent.parent.parent / model_config.get('preprocessor_path', 'models/preprocessor.pkl')
    preprocessor = None
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
    
    # Get threshold
    threshold = model_config.get('threshold', 0.5)
    
    return PromotionPredictor(model, preprocessor, threshold)


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