"""Property valuation model implementation."""

import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from src.models.preprocessor import PropertyDataPreprocessor
from src.utils.config import MODEL_PARAMS, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PropertyValuationModel:
    """Complete property valuation model with preprocessing and prediction."""
    
    def __init__(self):
        self.preprocessor = PropertyDataPreprocessor()
        self.model = GradientBoostingRegressor(**MODEL_PARAMS)
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.model_metrics: Dict[str, float] = {}
    
    def create_pipeline(self) -> Pipeline:
        """Create the complete ML pipeline."""
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor.create_preprocessor()),
            ('model', self.model)
        ])
        
        logger.info("pipeline_created", model_params=MODEL_PARAMS)
        return pipeline
    
    def train(
        self, 
        train_df: pd.DataFrame, 
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train the property valuation model.
        
        Args:
            train_df: Training DataFrame
            test_df: Optional test DataFrame for evaluation
            
        Returns:
            Dictionary of model metrics
        """
        logger.info(
            "training_started",
            train_samples=len(train_df),
            test_samples=len(test_df) if test_df is not None else 0
        )
        
        # Prepare features and target
        X_train = train_df[self.preprocessor.preprocessor.feature_names_in_ if hasattr(self.preprocessor.preprocessor, 'feature_names_in_') else train_df.columns[:-1]]
        y_train = train_df['price']
        
        # Create and train pipeline
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.pipeline.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_predictions, "train")
        
        # Calculate test metrics if test data provided
        test_metrics = {}
        if test_df is not None:
            X_test = test_df[X_train.columns]
            y_test = test_df['price']
            test_predictions = self.pipeline.predict(X_test)
            test_metrics = self._calculate_metrics(y_test, test_predictions, "test")
        
        # Combine metrics
        self.model_metrics = {**train_metrics, **test_metrics}
        
        logger.info(
            "training_completed",
            train_samples=len(train_df),
            metrics=self.model_metrics
        )
        
        return self.model_metrics
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Make a single prediction.
        
        Args:
            features: Dictionary of property features
            
        Returns:
            Predicted price
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.pipeline.predict(df)[0]
        
        logger.debug(
            "prediction_made",
            features=features,
            prediction=float(prediction)
        )
        
        return float(prediction)
    
    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make batch predictions.
        
        Args:
            features_df: DataFrame with multiple property features
            
        Returns:
            Array of predicted prices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.pipeline.predict(features_df)
        
        logger.debug(
            "batch_prediction_made",
            batch_size=len(features_df),
            predictions_count=len(predictions)
        )
        
        return predictions
    
    def save(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire pipeline
        joblib.dump(self.pipeline, model_path)
        
        # Also save metrics separately
        metrics_path = model_path.replace('.pkl', '_metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        
        logger.info(
            "model_saved",
            model_path=model_path,
            metrics_path=metrics_path,
            metrics=self.model_metrics
        )
    
    def load(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the pipeline
        self.pipeline = joblib.load(model_path)
        self.is_trained = True
        
        # Load metrics if available
        metrics_path = model_path.replace('.pkl', '_metrics.json')
        if Path(metrics_path).exists():
            import json
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
        
        logger.info(
            "model_loaded",
            model_path=model_path,
            has_metrics=bool(self.model_metrics)
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get the actual model from pipeline
        model = self.pipeline.named_steps['model']
        
        # Get feature names after preprocessing
        feature_names = getattr(self.pipeline.named_steps['preprocessor'], 'get_feature_names_out', lambda: [f'feature_{i}' for i in range(len(model.feature_importances_))])()
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {
            f"{prefix}_rmse": float(rmse),
            f"{prefix}_mape": float(mape),
            f"{prefix}_mae": float(mae)
        }
        
        logger.info(f"{prefix}_metrics_calculated", **metrics)
        return metrics


def load_model(model_path: Optional[str] = None) -> PropertyValuationModel:
    """
    Load a property valuation model.
    
    Args:
        model_path: Optional path to model file. Uses config default if not provided.
        
    Returns:
        Loaded PropertyValuationModel instance
    """
    if model_path is None:
        settings = get_settings()
        model_path = settings.model_path
    
    model = PropertyValuationModel()
    model.load(model_path)
    
    return model
