"""API endpoints for property valuation predictions."""

import time
import uuid
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse

from src.api.auth import get_current_user
from src.models.model import PropertyValuationModel, load_model
from src.utils.validation import PredictionRequest, PredictionResponse, PropertyFeatures
from src.utils.config import get_settings
from src.utils.logger import get_logger, APILogger

logger = get_logger(__name__)
api_logger = APILogger()
router = APIRouter()

# Global model instance (loaded once at startup)
_model: Optional[PropertyValuationModel] = None


def get_model() -> PropertyValuationModel:
    """Get the loaded model instance."""
    global _model
    if _model is None:
        settings = get_settings()
        try:
            _model = load_model(settings.model_path)
            logger.info("model_loaded_for_api", model_path=settings.model_path)
        except Exception as e:
            logger.error("model_loading_failed", error=str(e), model_path=settings.model_path)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not available: {str(e)}"
            )
    return _model


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Property Price",
    description="Predict the price of a single property based on its features."
)
async def predict_property_price(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user),
    http_request: Request = None
) -> PredictionResponse:
    """
    Predict property price for a single property.
    
    - **features**: Property features including type, sector, area, rooms, etc.
    - **request_id**: Optional request ID for tracking
    
    Returns predicted price in the same currency as training data.
    """
    start_time = time.time()
    
    # Generate request ID if not provided
    request_id = request.request_id or str(uuid.uuid4())
    
    # Get client IP
    client_ip = http_request.client.host if http_request else None
    
    # Log prediction request
    features_dict = request.features.dict()
    api_logger.log_prediction_request(
        request_id=request_id,
        features=features_dict,
        client_ip=client_ip
    )
    
    try:
        # Get model
        model = get_model()
        
        # Make prediction
        prediction = model.predict(features_dict)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log successful prediction
        api_logger.log_prediction_response(
            request_id=request_id,
            prediction=prediction,
            processing_time_ms=processing_time_ms
        )
        
        # Return response
        return PredictionResponse(
            prediction=prediction,
            request_id=request_id,
            model_version=get_settings().model_version
        )
        
    except Exception as e:
        # Log prediction error
        api_logger.log_prediction_error(
            request_id=request_id,
            error=str(e),
            features=features_dict
        )
        
        logger.error(
            "prediction_failed",
            request_id=request_id,
            error=str(e),
            features=features_dict
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    summary="Batch Predict Property Prices",
    description="Predict prices for multiple properties in a single request."
)
async def batch_predict_property_prices(
    requests: List[PropertyFeatures],
    current_user: dict = Depends(get_current_user),
    http_request: Request = None
) -> List[PredictionResponse]:
    """
    Predict property prices for multiple properties.
    
    - **requests**: List of property features for batch prediction
    
    Returns a list of predicted prices.
    """
    start_time = time.time()
    
    # Generate batch ID for logging
    batch_id = str(uuid.uuid4())
    client_ip = http_request.client.host if http_request else None
    
    logger.info(
        "batch_prediction_started",
        batch_id=batch_id,
        batch_size=len(requests),
        client_ip=client_ip
    )
    
    try:
        # Get model
        model = get_model()
        
        # Process predictions
        results = []
        for i, features in enumerate(requests):
            request_id = f"{batch_id}_{i}"
            
            try:
                # Convert to dict and predict
                features_dict = features.dict()
                prediction = model.predict(features_dict)
                
                # Log individual prediction
                api_logger.log_prediction_response(
                    request_id=request_id,
                    prediction=prediction
                )
                
                results.append(PredictionResponse(
                    prediction=prediction,
                    request_id=request_id,
                    model_version=get_settings().model_version
                ))
                
            except Exception as e:
                # Log individual prediction error
                api_logger.log_prediction_error(
                    request_id=request_id,
                    error=str(e),
                    features=features_dict if 'features_dict' in locals() else None
                )
                
                # Add error response
                results.append(PredictionResponse(
                    prediction=0.0,  # Default value for failed prediction
                    request_id=request_id,
                    model_version=get_settings().model_version
                ))
        
        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "batch_prediction_completed",
            batch_id=batch_id,
            batch_size=len(requests),
            processing_time_ms=processing_time_ms
        )
        
        return results
        
    except Exception as e:
        logger.error(
            "batch_prediction_failed",
            batch_id=batch_id,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/model/info",
    summary="Get Model Information",
    description="Get information about the currently loaded model."
)
async def get_model_info(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get information about the currently loaded model.
    
    Returns model version, metrics, and feature importance.
    """
    try:
        model = get_model()
        settings = get_settings()
        
        # Get feature importance if available
        feature_importance = {}
        try:
            feature_importance = model.get_feature_importance()
        except Exception as e:
            logger.warning("feature_importance_unavailable", error=str(e))
        
        return {
            "model_version": settings.model_version,
            "model_path": settings.model_path,
            "is_trained": model.is_trained,
            "metrics": model.model_metrics,
            "feature_importance": feature_importance,
            "supported_features": [
                "type", "sector", "net_usable_area", "net_area",
                "n_rooms", "n_bathroom", "latitude", "longitude"
            ]
        }
        
    except Exception as e:
        logger.error("model_info_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Check if the API and model are healthy."
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns the health status of the API and model.
    """
    try:
        # Check if model is loaded and healthy
        model = get_model()
        model_healthy = model.is_trained
        
        return {
            "status": "healthy" if model_healthy else "unhealthy",
            "model_loaded": True,
            "model_trained": model_healthy,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "model_trained": False,
            "error": str(e),
            "timestamp": time.time()
        }
