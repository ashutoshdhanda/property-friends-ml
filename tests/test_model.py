"""Tests for the ML model components."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.model import PropertyValuationModel
from src.models.preprocessor import PropertyDataPreprocessor
from src.utils.validation import PropertyFeatures, validate_training_data


class TestPropertyDataPreprocessor:
    """Tests for data preprocessing."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        return pd.DataFrame({
            'type': ['casa', 'departamento', 'casa'],
            'sector': ['vitacura', 'las condes', 'providencia'],
            'net_usable_area': [140.0, 80.0, 120.0],
            'net_area': [170.0, 95.0, 150.0],
            'n_rooms': [4, 2, 3],
            'n_bathroom': [3, 2, 2],
            'latitude': [-33.40, -33.41, -33.42],
            'longitude': [-70.58, -70.57, -70.59],
            'price': [15000, 8000, 12000]
        })
    
    def test_preprocessor_creation(self):
        """Test preprocessor creation."""
        preprocessor = PropertyDataPreprocessor()
        column_transformer = preprocessor.create_preprocessor()
        
        assert column_transformer is not None
        assert hasattr(column_transformer, 'transformers')
    
    def test_fit_transform(self, sample_data):
        """Test fit and transform."""
        preprocessor = PropertyDataPreprocessor()
        
        X = sample_data.drop('price', axis=1)
        y = sample_data['price']
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert preprocessor.is_fitted
        assert X_transformed.shape[0] == len(sample_data)
        assert isinstance(X_transformed, np.ndarray)
    
    def test_transform_without_fit(self, sample_data):
        """Test transform without fitting first."""
        preprocessor = PropertyDataPreprocessor()
        X = sample_data.drop('price', axis=1)
        
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(X)


class TestPropertyValuationModel:
    """Tests for the main model class."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        return pd.DataFrame({
            'type': ['casa', 'departamento', 'casa', 'departamento'],
            'sector': ['vitacura', 'las condes', 'providencia', 'vitacura'],
            'net_usable_area': [140.0, 80.0, 120.0, 90.0],
            'net_area': [170.0, 95.0, 150.0, 110.0],
            'n_rooms': [4, 2, 3, 2],
            'n_bathroom': [3, 2, 2, 2],
            'latitude': [-33.40, -33.41, -33.42, -33.39],
            'longitude': [-70.58, -70.57, -70.59, -70.58],
            'price': [15000, 8000, 12000, 9000]
        })
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = PropertyValuationModel()
        
        assert not model.is_trained
        assert model.model_metrics == {}
        assert model.pipeline is None
    
    def test_model_training(self, sample_data):
        """Test model training."""
        model = PropertyValuationModel()
        
        # Mock the feature columns to match our sample data
        with patch('src.models.model.PropertyDataPreprocessor'):
            with patch.object(model, 'create_pipeline') as mock_pipeline:
                mock_pipe = MagicMock()
                mock_pipe.fit = MagicMock()
                mock_pipe.predict = MagicMock(return_value=np.array([15000, 8000, 12000, 9000]))
                mock_pipeline.return_value = mock_pipe
                
                metrics = model.train(sample_data)
                
                assert model.is_trained
                assert isinstance(metrics, dict)
                assert 'train_rmse' in metrics
                assert 'train_mape' in metrics
                assert 'train_mae' in metrics
    
    def test_prediction_without_training(self):
        """Test prediction without training."""
        model = PropertyValuationModel()
        
        features = {
            'type': 'casa',
            'sector': 'vitacura',
            'net_usable_area': 140.0,
            'net_area': 170.0,
            'n_rooms': 4,
            'n_bathroom': 3,
            'latitude': -33.40,
            'longitude': -70.58
        }
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(features)


class TestPropertyFeatures:
    """Tests for feature validation."""
    
    def test_valid_features(self):
        """Test valid property features."""
        features = PropertyFeatures(
            type="casa",
            sector="vitacura",
            net_usable_area=140.0,
            net_area=170.0,
            n_rooms=4,
            n_bathroom=3,
            latitude=-33.40123,
            longitude=-70.58056
        )
        
        assert features.type == "casa"
        assert features.sector == "vitacura"
        assert features.net_usable_area == 140.0
    
    def test_invalid_property_type(self):
        """Test invalid property type."""
        with pytest.raises(ValueError, match="Property type must be"):
            PropertyFeatures(
                type="invalid_type",
                sector="vitacura",
                net_usable_area=140.0,
                net_area=170.0,
                n_rooms=4,
                n_bathroom=3,
                latitude=-33.40123,
                longitude=-70.58056
            )
    
    def test_negative_area(self):
        """Test negative area values."""
        with pytest.raises(ValueError):
            PropertyFeatures(
                type="casa",
                sector="vitacura",
                net_usable_area=-10.0,  # Negative area
                net_area=170.0,
                n_rooms=4,
                n_bathroom=3,
                latitude=-33.40123,
                longitude=-70.58056
            )
    
    def test_invalid_coordinates(self):
        """Test invalid coordinates."""
        with pytest.raises(ValueError, match="Chile's bounds"):
            PropertyFeatures(
                type="casa",
                sector="vitacura",
                net_usable_area=140.0,
                net_area=170.0,
                n_rooms=4,
                n_bathroom=3,
                latitude=0.0,  # Invalid latitude for Chile
                longitude=-70.58056
            )


class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_valid_training_data(self):
        """Test validation of valid training data."""
        df = pd.DataFrame({
            'type': ['casa', 'departamento'],
            'sector': ['vitacura', 'las condes'],
            'net_usable_area': [140.0, 80.0],
            'net_area': [170.0, 95.0],
            'n_rooms': [4, 2],
            'n_bathroom': [3, 2],
            'latitude': [-33.40, -33.41],
            'longitude': [-70.58, -70.57],
            'price': [15000, 8000]
        })
        
        is_valid, errors = validate_training_data(df)
        assert is_valid
        assert len(errors) == 0
    
    def test_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({
            'type': ['casa'],
            'sector': ['vitacura'],
            # Missing other required columns
        })
        
        is_valid, errors = validate_training_data(df)
        assert not is_valid
        assert "Missing required columns" in errors[0]
    
    def test_null_values(self):
        """Test validation with null values."""
        df = pd.DataFrame({
            'type': ['casa', None],
            'sector': ['vitacura', 'las condes'],
            'net_usable_area': [140.0, 80.0],
            'net_area': [170.0, 95.0],
            'n_rooms': [4, 2],
            'n_bathroom': [3, 2],
            'latitude': [-33.40, -33.41],
            'longitude': [-70.58, -70.57],
            'price': [15000, 8000]
        })
        
        is_valid, errors = validate_training_data(df)
        assert not is_valid
        assert "Null values found" in errors[0]
