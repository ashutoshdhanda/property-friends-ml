"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_healthy(self):
        """Test health check when model is loaded and healthy."""
        with patch('src.api.endpoints.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.is_trained = True
            mock_get_model.return_value = mock_model
            
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["model_trained"] is True
    
    def test_health_check_unhealthy(self):
        """Test health check when model fails to load."""
        with patch('src.api.endpoints.get_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Model not found")
            
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestAuthEndpoints:
    """Tests for authentication."""
    
    def test_unauthenticated_request(self):
        """Test request without authentication."""
        response = client.post("/api/v1/predict", json={
            "features": {
                "type": "casa",
                "sector": "vitacura",
                "net_usable_area": 140.0,
                "net_area": 170.0,
                "n_rooms": 4,
                "n_bathroom": 3,
                "latitude": -33.40123,
                "longitude": -70.58056
            }
        })
        assert response.status_code == 401
    
    def test_invalid_api_key(self):
        """Test request with invalid API key."""
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.post("/api/v1/predict", json={
            "features": {
                "type": "casa",
                "sector": "vitacura", 
                "net_usable_area": 140.0,
                "net_area": 170.0,
                "n_rooms": 4,
                "n_bathroom": 3,
                "latitude": -33.40123,
                "longitude": -70.58056
            }
        }, headers=headers)
        assert response.status_code == 401


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer demo-api-key-for-testing"}
    
    @pytest.fixture
    def valid_features(self):
        """Provide valid property features."""
        return {
            "type": "casa",
            "sector": "vitacura",
            "net_usable_area": 140.0,
            "net_area": 170.0,
            "n_rooms": 4,
            "n_bathroom": 3,
            "latitude": -33.40123,
            "longitude": -70.58056
        }
    
    def test_predict_valid_request(self, auth_headers, valid_features):
        """Test valid prediction request."""
        with patch('src.api.endpoints.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = 15000.0
            mock_get_model.return_value = mock_model
            
            response = client.post("/api/v1/predict", json={
                "features": valid_features
            }, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] == 15000.0
            assert "model_version" in data
    
    def test_predict_invalid_features(self, auth_headers):
        """Test prediction with invalid features."""
        invalid_features = {
            "type": "invalid_type",  # Invalid property type
            "sector": "vitacura",
            "net_usable_area": -10.0,  # Negative area
            "net_area": 170.0,
            "n_rooms": 4,
            "n_bathroom": 3,
            "latitude": -33.40123,
            "longitude": -70.58056
        }
        
        response = client.post("/api/v1/predict", json={
            "features": invalid_features
        }, headers=auth_headers)
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict(self, auth_headers, valid_features):
        """Test batch prediction."""
        with patch('src.api.endpoints.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.side_effect = [15000.0, 12000.0]
            mock_get_model.return_value = mock_model
            
            features_list = [valid_features, valid_features.copy()]
            
            response = client.post("/api/v1/predict/batch", 
                                 json=features_list, 
                                 headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert all("prediction" in item for item in data)


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""
    
    def test_model_info(self):
        """Test model info endpoint."""
        auth_headers = {"Authorization": "Bearer demo-api-key-for-testing"}
        
        with patch('src.api.endpoints.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.is_trained = True
            mock_model.model_metrics = {"test_rmse": 10254.15}
            mock_model.get_feature_importance.return_value = {
                "net_usable_area": 0.3,
                "latitude": 0.25
            }
            mock_get_model.return_value = mock_model
            
            response = client.get("/api/v1/model/info", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_trained"] is True
            assert "metrics" in data
            assert "feature_importance" in data
            assert "supported_features" in data


class TestRootEndpoints:
    """Tests for root and info endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "demo_credentials" in data
    
    def test_api_info_endpoint(self):
        """Test API info endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        
        data = response.json()
        assert "endpoints" in data
        assert "authentication" in data
        assert "docs" in data
