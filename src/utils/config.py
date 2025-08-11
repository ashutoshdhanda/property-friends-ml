"""Configuration management for the Property Friends ML application."""

import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key_secret: str = "default-secret-change-in-production"
    debug: bool = False
    
    # Model Configuration
    model_path: str = "models/property_model.pkl"
    model_version: str = "1.0.0"
    
    # Data Configuration
    data_source: str = "csv"  # csv or database
    train_data_path: str = "train.csv"
    test_data_path: str = "test.csv"
    
    # Database Configuration (for future use)
    database_url: Optional[str] = None
    database_pool_size: int = 5
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Security
    secret_key: str = "your-jwt-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Feature configuration
FEATURE_COLUMNS = [
    "type", "sector", "net_usable_area", "net_area", 
    "n_rooms", "n_bathroom", "latitude", "longitude"
]

CATEGORICAL_COLUMNS = ["type", "sector"]
NUMERICAL_COLUMNS = [
    "net_usable_area", "net_area", "n_rooms", 
    "n_bathroom", "latitude", "longitude"
]

TARGET_COLUMN = "price"

# Model hyperparameters
MODEL_PARAMS = {
    "learning_rate": 0.01,
    "n_estimators": 300,
    "max_depth": 5,
    "loss": "absolute_error"
}
