"""Input validation utilities for the Property Friends ML application."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field, validator
from src.utils.config import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


class PropertyFeatures(BaseModel):
    """Validation model for property features."""
    
    type: str = Field(..., description="Property type (casa or departamento)")
    sector: str = Field(..., description="Property sector/neighborhood")
    net_usable_area: float = Field(..., gt=0, description="Net usable area in square meters")
    net_area: float = Field(..., gt=0, description="Total net area in square meters")
    n_rooms: int = Field(..., ge=1, description="Number of rooms")
    n_bathroom: int = Field(..., ge=1, description="Number of bathrooms")
    latitude: float = Field(..., ge=-90, le=90, description="Property latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Property longitude")
    
    @validator('type')
    def validate_property_type(cls, v):
        """Validate property type is one of the expected values."""
        valid_types = ['casa', 'departamento']
        if v.lower() not in valid_types:
            raise ValueError(f'Property type must be one of: {valid_types}')
        return v.lower()
    
    @validator('net_usable_area', 'net_area')
    def validate_areas(cls, v):
        """Validate area values are reasonable."""
        if v > 10000:  # 10,000 sq meters is very large
            raise ValueError('Area seems unreasonably large')
        return v
    
    @validator('n_rooms', 'n_bathroom')
    def validate_room_counts(cls, v):
        """Validate room counts are reasonable."""
        if v > 50:  # 50 rooms is very large
            raise ValueError('Room count seems unreasonably large')
        return v
    
    @validator('latitude')
    def validate_chile_latitude(cls, v):
        """Validate latitude is within Chile's approximate bounds."""
        if not (-56 <= v <= -17):  # Chile's approximate latitude range
            raise ValueError('Latitude should be within Chile\'s bounds (-56 to -17)')
        return v
    
    @validator('longitude')
    def validate_chile_longitude(cls, v):
        """Validate longitude is within Chile's approximate bounds."""
        if not (-76 <= v <= -66):  # Chile's approximate longitude range
            raise ValueError('Longitude should be within Chile\'s bounds (-76 to -66)')
        return v


class PredictionRequest(BaseModel):
    """Model for prediction API requests."""
    
    features: PropertyFeatures = Field(..., description="Property features for prediction")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")


class PredictionResponse(BaseModel):
    """Model for prediction API responses."""
    
    prediction: float = Field(..., description="Predicted property price")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    request_id: Optional[str] = Field(None, description="Request ID if provided")
    model_version: str = Field(..., description="Model version used for prediction")


def validate_training_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate training/test data format and quality.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required columns
    missing_cols = set(FEATURE_COLUMNS + ['price']) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataframe
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # Check for null values in critical columns
    null_counts = df[FEATURE_COLUMNS + ['price']].isnull().sum()
    if null_counts.any():
        errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check categorical values
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            valid_values = ['casa', 'departamento'] if col == 'type' else None
            if valid_values and not df[col].isin(valid_values).all():
                invalid_values = df[~df[col].isin(valid_values)][col].unique()
                errors.append(f"Invalid values in {col}: {invalid_values}")
    
    # Check numerical ranges
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            if (df[col] <= 0).any() and col in ['net_usable_area', 'net_area', 'n_rooms', 'n_bathroom']:
                errors.append(f"Non-positive values found in {col}")
    
    # Check price range (target variable)
    if 'price' in df.columns:
        if (df['price'] <= 0).any():
            errors.append("Non-positive prices found")
        if df['price'].max() > 1000000:  # Extremely high prices might be outliers
            errors.append("Extremely high prices detected (>1M), please verify")
    
    return len(errors) == 0, errors


def clean_features_dict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize feature dictionary for prediction."""
    cleaned = {}
    
    for key, value in features.items():
        if key in FEATURE_COLUMNS:
            if key in CATEGORICAL_COLUMNS:
                cleaned[key] = str(value).lower().strip()
            else:
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    raise ValueError(f"Cannot convert {key}={value} to numeric")
        
    return cleaned
