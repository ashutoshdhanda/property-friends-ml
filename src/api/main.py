"""Main FastAPI application for Property Friends ML API."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.api.endpoints import router as prediction_router
from src.api.auth import get_demo_credentials
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("api_startup_started")
    
    # Pre-load model during startup
    try:
        from src.api.endpoints import get_model
        model = get_model()
        logger.info("model_preloaded_successfully", is_trained=model.is_trained)
    except Exception as e:
        logger.error("model_preload_failed", error=str(e))
        # Don't fail startup - let individual requests handle model loading errors
    
    logger.info("api_startup_completed")
    
    yield
    
    # Shutdown
    logger.info("api_shutdown_started")
    # Cleanup code here if needed
    logger.info("api_shutdown_completed")


# Create FastAPI application
app = FastAPI(
    title="Property Friends ML API",
    description="Chilean Real Estate Property Valuation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "api_request",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response
    logger.info(
        "api_response",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.warning(
        "http_exception",
        method=request.method,
        url=str(request.url),
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled exceptions."""
    logger.error(
        "unhandled_exception",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        error_type=type(exc).__name__
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


# Include routers
app.include_router(
    prediction_router,
    prefix="/api/v1",
    tags=["predictions"]
)


@app.get("/", summary="Root Endpoint")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": "Property Friends ML API",
        "version": "1.0.0",
        "description": "Chilean Real Estate Property Valuation API",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "demo_credentials": get_demo_credentials()
    }


@app.get("/api", summary="API Information")
async def api_info() -> Dict[str, Any]:
    """API information and available endpoints."""
    return {
        "name": "Property Friends ML API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "model_info": "/api/v1/model/info",
            "health": "/api/v1/health"
        },
        "authentication": {
            "type": "Bearer Token (API Key)",
            "header": "Authorization: Bearer <api-key>",
            "demo_credentials": get_demo_credentials()
        },
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


def start_server():
    """Start the FastAPI server."""
    settings = get_settings()
    
    logger.info(
        "starting_api_server",
        host=settings.api_host,
        port=settings.api_port,
        debug=settings.debug
    )
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
