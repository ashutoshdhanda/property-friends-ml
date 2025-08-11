"""Structured logging configuration for the Property Friends ML application."""

import logging
import structlog
from typing import Any, Dict, Optional
from src.utils.config import get_settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class APILogger:
    """Logger specifically for API requests and responses."""
    
    def __init__(self):
        self.logger = get_logger("api")
    
    def log_prediction_request(
        self, 
        request_id: str, 
        features: Dict[str, Any],
        client_ip: Optional[str] = None
    ) -> None:
        """Log an incoming prediction request."""
        self.logger.info(
            "prediction_request",
            request_id=request_id,
            features=features,
            client_ip=client_ip
        )
    
    def log_prediction_response(
        self, 
        request_id: str, 
        prediction: float,
        confidence: Optional[float] = None,
        processing_time_ms: Optional[float] = None
    ) -> None:
        """Log a prediction response."""
        self.logger.info(
            "prediction_response",
            request_id=request_id,
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
    
    def log_prediction_error(
        self, 
        request_id: str, 
        error: str,
        features: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a prediction error."""
        self.logger.error(
            "prediction_error",
            request_id=request_id,
            error=error,
            features=features
        )
    
    def log_model_load(self, model_path: str, load_time_ms: float) -> None:
        """Log model loading event."""
        self.logger.info(
            "model_loaded",
            model_path=model_path,
            load_time_ms=load_time_ms
        )
    
    def log_model_training(
        self, 
        training_samples: int,
        training_time_ms: float,
        model_metrics: Dict[str, float]
    ) -> None:
        """Log model training completion."""
        self.logger.info(
            "model_training_completed",
            training_samples=training_samples,
            training_time_ms=training_time_ms,
            metrics=model_metrics
        )
