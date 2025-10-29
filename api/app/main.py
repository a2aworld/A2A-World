"""
A2A World Platform - Enhanced Main API Application

FastAPI application for the A2A World platform with comprehensive
data management, agent coordination, pattern discovery, and system monitoring.
"""

import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import uvicorn

try:
    from app.core.config import settings
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Fallback configuration
    class Settings:
        API_V1_STR = "/api/v1"
        BACKEND_CORS_ORIGINS = ["*"]
    settings = Settings()

from app.api.api_v1.api import api_router

try:
    from app.core.errors import create_error_response, ErrorHandlingMiddleware
    from app.core.serialization import ResponseProcessingMiddleware, performance_monitor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting A2A World Platform API v2.0")
    logger.info("Initializing system components...")
    
    if ENHANCED_FEATURES_AVAILABLE:
        logger.info("Enhanced features loaded: error handling, caching, monitoring")
    else:
        logger.warning("Enhanced features not available - running in basic mode")
    
    yield
    
    # Shutdown  
    logger.info("Shutting down A2A World Platform API")
    logger.info("Cleanup completed")

# Create FastAPI application with comprehensive configuration
app = FastAPI(
    title="A2A World Platform API",
    description="""
    # A2A World Platform REST API v2.0
    
    Comprehensive REST API for data access and agent management in the A2A World platform.
    
    ## Features
    
    - **Data Management**: Upload, process, and manage geospatial datasets (KML, GeoJSON, CSV)
    - **Agent Coordination**: Lifecycle management and monitoring of autonomous agents
    - **Pattern Discovery**: Advanced pattern discovery with HDBSCAN clustering and validation
    - **System Monitoring**: Health checks, metrics, and operational management
    
    ## API Enhancements (Phase 2)
    
    This version includes comprehensive enhancements:
    
    ### Data Access API
    - Advanced filtering with geospatial queries (bounding box, radius search)
    - Complex search with text, properties, and date range filters
    - Bulk data export in multiple formats (GeoJSON, KML, CSV)
    - Comprehensive dataset statistics and analytics
    - Real-time processing status and progress tracking
    
    ### Agent Management API
    - Complete agent lifecycle control (start, stop, restart)
    - Real-time health monitoring and performance metrics  
    - Task assignment and queue management
    - Agent configuration management
    - Comprehensive status reporting with capabilities
    
    ### Pattern Discovery API
    - Enhanced search and filtering capabilities
    - Batch validation operations for multiple patterns
    - Detailed validation history and consensus tracking
    - Pattern export with visualization data
    - Advanced statistical analysis integration
    
    ### System Management API
    - Comprehensive health monitoring with dependency checks
    - System metrics and performance analytics
    - Configuration management with dynamic updates
    - System logs with filtering and search
    - Maintenance operations and task scheduling
    
    ## Authentication
    
    The API supports multiple authentication methods:
    - API Key authentication for programmatic access
    - Session-based authentication for web applications
    
    ## Rate Limiting
    
    All endpoints are rate-limited to ensure fair usage:
    - Standard endpoints: 100 requests per minute
    - Data upload endpoints: 10 requests per minute
    - Export endpoints: 5 requests per minute
    
    ## Data Formats
    
    **Supported input formats:**
    - **KML/KMZ**: Google Earth format files with full geometry support
    - **GeoJSON**: Standard geographic JSON format
    - **CSV**: Comma-separated values with automatic coordinate detection
    - **ZIP**: Compressed archives containing supported formats
    
    **Supported output formats:**
    - **JSON**: Default response format with comprehensive metadata
    - **GeoJSON**: Geographic feature collections with styling
    - **CSV**: Tabular data export with all properties
    - **KML**: Google Earth compatible export with extended data
    
    ## Error Handling
    
    All errors follow a consistent format with:
    - Standardized error codes for programmatic handling
    - Human-readable messages with context
    - Detailed validation errors when applicable
    - Suggested actions for resolution
    - Request tracking with unique IDs
    
    ## Performance & Caching
    
    Responses are intelligently cached for improved performance:
    - List endpoints: 5 minutes (300s)
    - Data queries: 15 minutes (900s)  
    - Statistics: 1 hour (3600s)
    - Configuration: Until changed
    - Redis backend with local fallback
    
    ## Monitoring & Analytics
    
    Comprehensive monitoring capabilities:
    - Real-time performance metrics
    - Request/response analytics  
    - Error rate tracking
    - Cache hit/miss statistics
    - System resource monitoring
    - Agent health and status
    """,
    version="2.0.0",
    contact={
        "name": "A2A World Platform Team",
        "url": "https://github.com/a2a-world/platform",
        "email": "support@a2aworld.org"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.a2aworld.org",
            "description": "Production server"
        }
    ],
    openapi_tags=[
        {
            "name": "root",
            "description": "API information and navigation endpoints"
        },
        {
            "name": "health",
            "description": "System health monitoring and status endpoints"
        },
        {
            "name": "data", 
            "description": "Comprehensive data management and processing endpoints"
        },
        {
            "name": "agents",
            "description": "Agent lifecycle management and monitoring endpoints"
        },
        {
            "name": "patterns",
            "description": "Pattern discovery, validation, and analysis endpoints"
        },
        {
            "name": "monitoring",
            "description": "System monitoring and performance endpoints"
        }
    ],
    lifespan=lifespan
)

# Custom OpenAPI schema with enhanced documentation
def custom_openapi():
    """Generate custom OpenAPI schema with comprehensive documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for programmatic access"
        },
        "SessionAuth": {
            "type": "apiKey", 
            "in": "cookie",
            "name": "session_id",
            "description": "Session-based authentication for web applications"
        }
    }
    
    # Add global security requirement (optional)
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"SessionAuth": []},
        {}  # Allow unauthenticated access
    ]
    
    # Add enhanced response schemas
    openapi_schema["components"]["schemas"].update({
        "ErrorResponse": {
            "type": "object",
            "required": ["success", "error_code", "message", "timestamp"],
            "properties": {
                "success": {"type": "boolean", "example": False, "description": "Always false for errors"},
                "error_code": {"type": "string", "example": "VALIDATION_ERROR", "description": "Machine-readable error code"},
                "message": {"type": "string", "example": "Request validation failed", "description": "Human-readable error message"},
                "details": {"type": "object", "description": "Additional error context and debugging information"},
                "timestamp": {"type": "string", "format": "date-time", "description": "ISO timestamp when error occurred"},
                "request_id": {"type": "string", "format": "uuid", "description": "Unique request identifier for tracking"},
                "validation_errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "description": "Field that failed validation"},
                            "message": {"type": "string", "description": "Validation error message"},
                            "code": {"type": "string", "description": "Validation error code"}
                        }
                    },
                    "description": "Detailed validation errors for request parameters"
                },
                "suggested_action": {"type": "string", "description": "Suggested action to resolve the error"}
            },
            "description": "Standard error response format"
        },
        "SuccessResponse": {
            "type": "object",
            "required": ["success", "timestamp"],
            "properties": {
                "success": {"type": "boolean", "example": True, "description": "Always true for successful responses"},
                "data": {"type": "object", "description": "Response data payload"},
                "timestamp": {"type": "string", "format": "date-time", "description": "ISO timestamp of response"},
                "request_id": {"type": "string", "format": "uuid", "description": "Unique request identifier"},
                "execution_time_ms": {"type": "number", "description": "Request processing time in milliseconds"},
                "cached": {"type": "boolean", "description": "Whether response was served from cache"}
            },
            "description": "Standard success response format"
        },
        "PaginatedResponse": {
            "type": "object",
            "required": ["items", "total", "limit", "offset", "has_more"],
            "properties": {
                "items": {"type": "array", "items": {}, "description": "Array of result items"},
                "total": {"type": "integer", "description": "Total number of items available"},
                "limit": {"type": "integer", "description": "Maximum items per page"},
                "offset": {"type": "integer", "description": "Number of items skipped"},
                "has_more": {"type": "boolean", "description": "Whether more items are available"},
                "page_info": {
                    "type": "object",
                    "properties": {
                        "current_page": {"type": "integer", "description": "Current page number"},
                        "total_pages": {"type": "integer", "description": "Total number of pages"},
                        "items_on_page": {"type": "integer", "description": "Items on current page"}
                    },
                    "description": "Pagination metadata"
                }
            },
            "description": "Paginated response format for list endpoints"
        }
    })
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Configure CORS with production settings
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["x-request-id", "x-response-time", "x-api-version", "x-cache-status"]
    )
else:
    # Fallback CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add enhanced middleware if available
if ENHANCED_FEATURES_AVAILABLE:
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(ResponseProcessingMiddleware)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        request_id = getattr(request.state, "request_id", None)
        return await create_error_response(exc, request_id, include_traceback=False)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Enhanced root endpoint with comprehensive API information
@app.get(
    "/",
    summary="API Information and Navigation",
    description="Get comprehensive information about the A2A World Platform API including features, endpoints, and capabilities",
    response_description="Complete API metadata and navigation links",
    tags=["root"]
)
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing comprehensive API information and navigation.
    
    Returns detailed metadata about the API including:
    - Version and feature information
    - Available endpoint categories  
    - Documentation and monitoring links
    - Supported formats and limitations
    - System capabilities and status
    """
    return {
        "service": "A2A World Platform API",
        "version": "2.0.0",
        "phase": "2 - Comprehensive REST API",
        "description": "Enhanced REST API for data access and agent management",
        "status": "operational",
        "enhanced_features": ENHANCED_FEATURES_AVAILABLE,
        "documentation": {
            "interactive": "/docs",
            "alternative": "/redoc", 
            "openapi_spec": "/openapi.json",
            "readme": "https://github.com/a2a-world/platform/blob/main/README.md"
        },
        "endpoints": {
            "health_monitoring": f"{settings.API_V1_STR}/health",
            "data_management": f"{settings.API_V1_STR}/data",
            "agent_control": f"{settings.API_V1_STR}/agents",
            "pattern_discovery": f"{settings.API_V1_STR}/patterns"
        },
        "capabilities": {
            "data_processing": [
                "Geospatial file upload and validation",
                "Advanced filtering and search",
                "Multi-format export (GeoJSON, KML, CSV)",
                "Real-time processing status",
                "Comprehensive analytics"
            ],
            "agent_management": [
                "Lifecycle control (start/stop/restart)",
                "Real-time health monitoring", 
                "Task assignment and tracking",
                "Performance metrics collection",
                "Configuration management"
            ],
            "pattern_discovery": [
                "HDBSCAN clustering analysis",
                "Batch validation operations",
                "Consensus tracking and scoring",
                "Statistical significance testing",
                "Export with visualization data"
            ],
            "system_monitoring": [
                "Comprehensive health checks",
                "Performance analytics",
                "Configuration management",
                "Log aggregation and search",
                "Maintenance operations"
            ]
        },
        "supported_formats": {
            "input": ["KML", "KMZ", "GeoJSON", "CSV", "ZIP"],
            "output": ["JSON", "GeoJSON", "KML", "CSV"]
        },
        "api_limits": {
            "max_file_size_mb": 100,
            "rate_limit_per_minute": 100,
            "max_results_per_page": 1000,
            "max_export_features": 50000,
            "cache_ttl_seconds": 300
        },
        "performance": {
            "caching_enabled": ENHANCED_FEATURES_AVAILABLE,
            "error_tracking": ENHANCED_FEATURES_AVAILABLE,
            "request_monitoring": ENHANCED_FEATURES_AVAILABLE
        }
    }

# Enhanced health check endpoint
@app.get(
    "/health",
    summary="Basic Health Check", 
    description="Quick health check for load balancers and monitoring systems",
    tags=["health"],
    response_description="Basic service health status"
)
async def health_check():
    """
    Basic health check endpoint for load balancers and monitoring systems.
    
    Returns minimal response for quick availability checks without
    performing expensive dependency checks.
    """
    return {
        "status": "healthy",
        "service": "a2a-world-api",
        "version": "2.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }

# API performance metrics endpoint  
@app.get(
    "/metrics",
    summary="API Performance Metrics",
    description="Get comprehensive API performance metrics and statistics",
    tags=["monitoring"],
    response_description="Current performance metrics and cache statistics"
)
async def get_api_metrics():
    """
    Get comprehensive API performance metrics and statistics.
    
    Includes:
    - Request counts and response times
    - Error rates and status distributions
    - Cache hit/miss ratios
    - System resource utilization
    """
    base_metrics = {
        "service": "a2a-world-api",
        "version": "2.0.0",
        "enhanced_monitoring": ENHANCED_FEATURES_AVAILABLE
    }
    
    if ENHANCED_FEATURES_AVAILABLE:
        base_metrics["performance"] = performance_monitor.get_metrics()
    else:
        base_metrics["message"] = "Enhanced monitoring not available - install performance monitoring dependencies"
    
    return base_metrics

# Custom documentation endpoints with enhanced styling
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling and configuration."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Interactive Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "none",
            "operationsSorter": "method",
            "tagsSorter": "alpha",
            "filter": True,
            "tryItOutEnabled": True,
            "syntaxHighlight.theme": "tomorrow-night"
        }
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc documentation with enhanced configuration."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Comprehensive Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js",
        redoc_options={
            "theme": {
                "colors": {
                    "primary": {
                        "main": "#1976d2"
                    }
                }
            },
            "hideDownloadButton": False,
            "expandResponses": "200,201",
            "requiredPropsFirst": True,
            "sortPropsAlphabetically": True,
            "showExtensions": True
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False
    )