"""Data preprocessing pipeline for property valuation model."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from src.utils.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PropertyDataPreprocessor:
    """Handles all data preprocessing for the property valuation model."""
    
    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.is_fitted = False
    
    def create_preprocessor(self) -> ColumnTransformer:
        """Create the preprocessing pipeline."""
        
        # Categorical transformer using Ordinal Encoding
        categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, CATEGORICAL_COLUMNS)
                # Note: We keep numerical columns as-is, but could add scaling here if needed
            ],
            remainder='passthrough'  # Keep numerical columns unchanged
        )
        
        logger.info(
            "preprocessor_created",
            categorical_columns=CATEGORICAL_COLUMNS,
            numerical_columns=NUMERICAL_COLUMNS
        )
        
        return preprocessor
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PropertyDataPreprocessor':
        """
        Fit the preprocessor to training data.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        logger.info("fitting_preprocessor", samples=len(X))
        
        # Ensure we have the right columns
        X_features = X[FEATURE_COLUMNS].copy()
        
        # Create and fit preprocessor
        self.preprocessor = self.create_preprocessor()
        self.preprocessor.fit(X_features, y)
        self.is_fitted = True
        
        logger.info("preprocessor_fitted", features=FEATURE_COLUMNS)
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed features array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Ensure we have the right columns
        X_features = X[FEATURE_COLUMNS].copy()
        
        # Transform
        X_transformed = self.preprocessor.transform(X_features)
        
        logger.debug(
            "features_transformed",
            input_shape=X_features.shape,
            output_shape=X_transformed.shape
        )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Fit preprocessor and transform features in one step.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Transformed features array
        """
        return self.fit(X, y).transform(X)


def prepare_training_data(
    train_df: pd.DataFrame, 
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare training and test data for model training.
    
    Args:
        train_df: Training DataFrame
        test_df: Optional test DataFrame
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    logger.info(
        "preparing_training_data",
        train_samples=len(train_df),
        test_samples=len(test_df) if test_df is not None else 0
    )
    
    # Prepare training data
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df['price'].copy()
    
    # Initialize preprocessor
    preprocessor = PropertyDataPreprocessor()
    
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Process test data if provided
    X_test_processed = None
    y_test = None
    if test_df is not None:
        X_test = test_df[FEATURE_COLUMNS].copy()
        y_test = test_df['price'].copy()
        X_test_processed = preprocessor.transform(X_test)
    
    logger.info(
        "data_preparation_complete",
        train_features_shape=X_train_processed.shape,
        test_features_shape=X_test_processed.shape if X_test_processed is not None else None
    )
    
    return X_train_processed, y_train.values, X_test_processed, y_test.values if y_test is not None else None


def prepare_prediction_data(features_dict: dict, fitted_preprocessor: PropertyDataPreprocessor) -> np.ndarray:
    """
    Prepare a single prediction input.
    
    Args:
        features_dict: Dictionary of feature values
        fitted_preprocessor: Already fitted preprocessor
        
    Returns:
        Processed features array
    """
    # Convert to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Transform using fitted preprocessor
    processed_features = fitted_preprocessor.transform(df)
    
    logger.debug(
        "prediction_data_prepared",
        input_features=features_dict,
        output_shape=processed_features.shape
    )
    
    return processed_features
