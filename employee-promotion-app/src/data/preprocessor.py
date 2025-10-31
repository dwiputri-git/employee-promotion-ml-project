"""
Data preprocessing pipeline matching V3 workflow
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path


class DataPreprocessor:
    """Data preprocessor that replicates V3 pipeline"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Fit the preprocessor on training data"""
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
        
        # Fit preprocessor
        self.preprocessor.fit(X)
        
        # Get feature names after transformation
        self.feature_names = self._get_feature_names(numeric_cols, categorical_cols)
        self.is_fitted = True
        
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _get_feature_names(self, numeric_cols, categorical_cols):
        """Get feature names after transformation"""
        feature_names = []
        
        # Add numeric feature names
        feature_names.extend(numeric_cols)
        
        # Add categorical feature names (one-hot encoded)
        if categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        
        return feature_names
    
    def get_feature_names(self):
        """Return feature names after transformation"""
        return self.feature_names


def clean_data(df, target_col='Promotion_Eligible'):
    """
    Clean data following V3 pipeline
    """
    df = df.copy()
    
    # Drop rows with missing target
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Impute missing values
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle outliers using IQR method
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check outlier percentage
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_pct = 100 * outlier_mask.mean()
        
        if outlier_pct < 5:
            # Drop outliers if < 5%
            df = df[~outlier_mask]
        else:
            # Winsorize if >= 5%
            df[col] = np.where(df[col] < lower_bound, lower_bound, 
                              np.where(df[col] > upper_bound, upper_bound, df[col]))
    
    # Remove rows with negative values in numeric columns
    negative_mask = (df[numeric_cols] < 0).any(axis=1)
    df = df[~negative_mask]
    
    return df


def engineer_features(df):
    """
    Engineer features following V3 pipeline
    """
    df = df.copy()
    
    # Create binned features
    if 'Training_Hours' in df.columns:
        df['Training_Level'] = pd.qcut(df['Training_Hours'], q=5, 
                                      labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
    
    if 'Leadership_Score' in df.columns:
        df['Leadership_Level'] = pd.qcut(df['Leadership_Score'], q=4, 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Projects per years
    if 'Projects_Handled' in df.columns and 'Years_at_Company' in df.columns:
        df['Projects_per_Years'] = df['Projects_Handled'] / (df['Years_at_Company'] + 1)
        df['Project_Level'] = pd.qcut(df['Projects_per_Years'], q=4, 
                                     labels=['Low', 'Moderate', 'High', 'Very High'])
    
    # Tenure and age groups
    if 'Years_at_Company' in df.columns:
        df['Tenure_Level'] = pd.qcut(df['Years_at_Company'], q=4, 
                                    labels=['New', 'Mid', 'Senior', 'Veteran'])
    
    if 'Age' in df.columns:
        df['Age_Group'] = pd.qcut(df['Age'], q=4, 
                                 labels=['Young', 'Early Mid', 'Late Mid', 'Senior'])
    
    # Interaction features
    if 'Performance_Score' in df.columns and 'Leadership_Score' in df.columns:
        df['Perf_x_Leader'] = df['Performance_Score'] * df['Leadership_Score']
    
    # Log transformation for skewed features
    if 'Projects_per_Years' in df.columns:
        if (df['Projects_per_Years'] >= 0).all():
            df['Projects_per_Years_log'] = np.log1p(df['Projects_per_Years'])
    
    # Drop intermediate columns
    columns_to_drop = ['Employee_ID', 'Projects_per_Years']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    return df


def preprocess_pipeline(df, target_col='Promotion_Eligible', fit_preprocessor=True):
    """
    Complete preprocessing pipeline
    """
    # Clean data
    df_clean = clean_data(df, target_col)
    
    # Engineer features
    df_features = engineer_features(df_clean)
    
    # Separate features and target
    if target_col in df_features.columns:
        X = df_features.drop(columns=[target_col])
        y = df_features[target_col]
    else:
        X = df_features
        y = None
    
    # Create and fit preprocessor
    preprocessor = DataPreprocessor()
    if fit_preprocessor and y is not None:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X) if preprocessor.is_fitted else X
    
    return X_transformed, y, preprocessor, df_features

