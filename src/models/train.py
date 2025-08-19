"""Training script for the property valuation model."""

import time
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.model import PropertyValuationModel
from src.utils.config import get_settings, FEATURE_COLUMNS
from src.utils.logger import configure_logging, get_logger, APILogger
from src.utils.validation import validate_training_data


def load_training_data(train_path: str, test_path: Optional[str] = None) -> tuple:
    """Load training and test data."""
    logger = get_logger(__name__)
    
    logger.info("loading_training_data", train_path=train_path, test_path=test_path)
    
    # Load training data
    train_df = pd.read_csv(train_path)
    logger.info("train_data_loaded", shape=train_df.shape)
    
    # Load test data if provided
    test_df = None
    if test_path and Path(test_path).exists():
        test_df = pd.read_csv(test_path)
        logger.info("test_data_loaded", shape=test_df.shape)
    
    return train_df, test_df


def validate_data(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> None:
    """Validate training and test data."""
    logger = get_logger(__name__)
    
    # Validate training data
    is_valid, errors = validate_training_data(train_df)
    if not is_valid:
        logger.error("training_data_validation_failed", errors=errors)
        raise ValueError(f"Training data validation failed: {errors}")
    
    logger.info("training_data_validation_passed", samples=len(train_df))
    
    # Validate test data if provided
    if test_df is not None:
        is_valid, errors = validate_training_data(test_df)
        if not is_valid:
            logger.error("test_data_validation_failed", errors=errors)
            raise ValueError(f"Test data validation failed: {errors}")
        
        logger.info("test_data_validation_passed", samples=len(test_df))


def train_model(
    train_path: str,
    test_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> PropertyValuationModel:
    """
    Train the property valuation model.
    
    Args:
        train_path: Path to training data CSV
        test_path: Optional path to test data CSV
        output_path: Optional path to save trained model
        
    Returns:
        Trained PropertyValuationModel
    """
    logger = get_logger(__name__)
    api_logger = APILogger()
    
    start_time = time.time()
    
    logger.info("training_started", train_path=train_path, test_path=test_path)
    
    # Load data
    train_df, test_df = load_training_data(train_path, test_path)
    
    # Validate data
    validate_data(train_df, test_df)
    
    # Initialize model
    model = PropertyValuationModel()
    
    # Train model
    training_start = time.time()
    metrics = model.train(train_df, test_df)
    training_time = (time.time() - training_start) * 1000  # ms
    
    # Log training completion
    api_logger.log_model_training(
        training_samples=len(train_df),
        training_time_ms=training_time,
        model_metrics=metrics
    )
    
    # Save model if output path provided
    if output_path:
        model.save(output_path)
        logger.info("model_saved", output_path=output_path)
    else:
        # Use default path from settings
        settings = get_settings()
        model.save(settings.model_path)
        logger.info("model_saved", output_path=settings.model_path)
    
    total_time = time.time() - start_time
    logger.info(
        "training_completed",
        total_time_seconds=total_time,
        metrics=metrics
    )
    
    return model


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train Property Valuation Model")
    parser.add_argument(
        "--train-data",
        default="train.csv",
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--test-data",
        default="test.csv",
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--output",
        help="Path to save trained model (optional)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    try:
        # Train model
        model = train_model(
            train_path=args.train_data,
            test_path=args.test_data,
            output_path=args.output
        )
        
        # Print summary
        print("\n=== Training Complete ===")
        print(f"Model metrics: {model.model_metrics}")
        
        # Show feature importance
        if model.is_trained:
            print("\n=== Feature Importance ===")
            importance = model.get_feature_importance()
            for feature, importance_score in list(importance.items())[:10]:  # Top 10
                print(f"{feature}: {importance_score:.4f}")
        
        logger.info("training_script_completed_successfully")
        
    except Exception as e:
        logger.error("training_script_failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
