"""
A2A World Platform - Error Handling and Validation

Comprehensive error handling, custom exceptions, and validation utilities
for the A2A World REST API.
"""

from typing import Dict, Any, Optional, List
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

# Custom Exception Classes

class A2ABaseException(Exception):
    """Base exception class for A2A World platform."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.utcnow()
        super().__init__(message)

class DatabaseConnectionError(A2ABaseException):
    """Database connectivity issues."""
    
    def __init__(self, message: str = "Database connection failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_ERROR",
            details=details,
            status_code=503
        )

class AgentNotFoundError(A2ABaseException):
    """Agent not found or unavailable."""
    
    def __init__(self, agent_id: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Agent {agent_id} not found or unavailable",
            error_code="AGENT_NOT_FOUND",
            details={"agent_id": agent_id, **(details or {})},
            status_code=404
        )

class AgentOperationError(A2ABaseException):
    """Agent operation failed."""
    
    def __init__(self, agent_id: str, operation: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Agent {agent_id} {operation} failed: {reason}",
            error_code="AGENT_OPERATION_ERROR",
            details={"agent_id": agent_id, "operation": operation, "reason": reason, **(details or {})},
            status_code=400
        )

class DataValidationError(A2ABaseException):
    """Data validation failed."""
    
    def __init__(self, field: str, value: Any, reason: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Validation failed for {field}: {reason}",
            error_code="DATA_VALIDATION_ERROR",
            details={"field": field, "value": str(value), "reason": reason, **(details or {})},
            status_code=422
        )

class PatternNotFoundError(A2ABaseException):
    """Pattern not found."""
    
    def __init__(self, pattern_id: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Pattern {pattern_id} not found",
            error_code="PATTERN_NOT_FOUND",
            details={"pattern_id": pattern_id, **(details or {})},
            status_code=404
        )

class FileProcessingError(A2ABaseException):
    """File processing failed."""
    
    def __init__(self, filename: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Failed to process file {filename}: {reason}",
            error_code="FILE_PROCESSING_ERROR",
            details={"filename": filename, "reason": reason, **(details or {})},
            status_code=422
        )

class GeospatialError(A2ABaseException):
    """Geospatial operation failed."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Geospatial {operation} failed: {reason}",
            error_code="GEOSPATIAL_ERROR",
            details={"operation": operation, "reason": reason, **(details or {})},
            status_code=400
        )

class RateLimitExceededError(A2ABaseException):
    """Rate limit exceeded."""
    
    def __init__(self, limit: int, window: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}",
            error_code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window": window, **(details or {})},
            status_code=429
        )

# Error Response Models

class ErrorDetail(BaseModel):
    """Error detail structure."""
    field: Optional[str] = None
    message: str
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standardized error response structure."""
    success: bool = False
    error_code: str
    message: str
    details: Dict[str, Any] = {}
    timestamp: str
    request_id: Optional[str] = None
    validation_errors: Optional[List[ErrorDetail]] = None
    suggested_action: Optional[str] = None

# Error Handler Functions

async def create_error_response(
    exception: Exception,
    request_id: Optional[str] = None,
    include_traceback: bool = False
) -> JSONResponse:
    """Create standardized error response."""
    
    if isinstance(exception, A2ABaseException):
        error_response = ErrorResponse(
            error_code=exception.error_code,
            message=exception.message,
            details=exception.details,
            timestamp=exception.timestamp.isoformat(),
            request_id=request_id,
            suggested_action=_get_suggested_action(exception.error_code)
        )
        status_code = exception.status_code
        
    elif isinstance(exception, ValidationError):
        validation_errors = []
        for error in exception.errors():
            validation_errors.append(ErrorDetail(
                field=".".join(str(loc) for loc in error["loc"]),
                message=error["msg"],
                code=error["type"]
            ))
        
        error_response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"validation_count": len(validation_errors)},
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            validation_errors=validation_errors,
            suggested_action="Check request parameters and try again"
        )
        status_code = 422
        
    elif isinstance(exception, HTTPException):
        error_response = ErrorResponse(
            error_code="HTTP_ERROR",
            message=exception.detail,
            details={"status_code": exception.status_code},
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            suggested_action=_get_suggested_action_for_status(exception.status_code)
        )
        status_code = exception.status_code
        
    else:
        # Unknown exception
        logger.error(f"Unhandled exception: {exception}", exc_info=True)
        error_response = ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details={"exception_type": type(exception).__name__},
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            suggested_action="Please try again later or contact support"
        )
        status_code = 500
        
        if include_traceback:
            error_response.details["traceback"] = traceback.format_exc()
    
    # Log the error
    logger.error(
        f"API Error: {error_response.error_code} - {error_response.message}",
        extra={
            "error_code": error_response.error_code,
            "request_id": request_id,
            "details": error_response.details
        }
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )

def _get_suggested_action(error_code: str) -> Optional[str]:
    """Get suggested action for error code."""
    suggestions = {
        "DATABASE_CONNECTION_ERROR": "Check database connectivity and try again",
        "AGENT_NOT_FOUND": "Verify agent ID and ensure agent is registered",
        "AGENT_OPERATION_ERROR": "Check agent status and configuration",
        "DATA_VALIDATION_ERROR": "Review input data format and requirements", 
        "PATTERN_NOT_FOUND": "Verify pattern ID exists in the system",
        "FILE_PROCESSING_ERROR": "Check file format and content validity",
        "GEOSPATIAL_ERROR": "Verify coordinate format and spatial data validity",
        "RATE_LIMIT_EXCEEDED": "Reduce request frequency and try again later"
    }
    return suggestions.get(error_code)

def _get_suggested_action_for_status(status_code: int) -> Optional[str]:
    """Get suggested action for HTTP status code."""
    suggestions = {
        400: "Check request parameters and format",
        401: "Provide valid authentication credentials",
        403: "Check user permissions for this operation",
        404: "Verify resource ID and path",
        405: "Use correct HTTP method for this endpoint",
        409: "Resource conflict - check for duplicates",
        422: "Fix validation errors in request data",
        429: "Reduce request rate and try again later",
        500: "Try again later or contact support",
        502: "Service temporarily unavailable",
        503: "Service is under maintenance"
    }
    return suggestions.get(status_code)

# Validation Utilities

def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validate UUID format."""
    import uuid
    try:
        uuid_obj = uuid.UUID(value)
        return str(uuid_obj)
    except ValueError:
        raise DataValidationError(
            field=field_name,
            value=value,
            reason="Invalid UUID format"
        )

def validate_coordinates(longitude: float, latitude: float) -> tuple:
    """Validate geographic coordinates."""
    if not -180.0 <= longitude <= 180.0:
        raise GeospatialError(
            operation="coordinate_validation",
            reason=f"Longitude {longitude} out of valid range [-180, 180]"
        )
    
    if not -90.0 <= latitude <= 90.0:
        raise GeospatialError(
            operation="coordinate_validation", 
            reason=f"Latitude {latitude} out of valid range [-90, 90]"
        )
    
    return longitude, latitude

def validate_bbox(bbox: List[float]) -> List[float]:
    """Validate bounding box coordinates."""
    if len(bbox) != 4:
        raise GeospatialError(
            operation="bbox_validation",
            reason="Bounding box must contain exactly 4 values [min_lon, min_lat, max_lon, max_lat]"
        )
    
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Validate individual coordinates
    validate_coordinates(min_lon, min_lat)
    validate_coordinates(max_lon, max_lat)
    
    # Check logical consistency
    if min_lon >= max_lon:
        raise GeospatialError(
            operation="bbox_validation",
            reason="min_longitude must be less than max_longitude"
        )
    
    if min_lat >= max_lat:
        raise GeospatialError(
            operation="bbox_validation", 
            reason="min_latitude must be less than max_latitude"
        )
    
    return bbox

def validate_file_size(size_bytes: int, max_size_mb: int = 100) -> int:
    """Validate file size limits."""
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if size_bytes > max_size_bytes:
        raise FileProcessingError(
            filename="uploaded_file",
            reason=f"File size {size_bytes} bytes exceeds maximum {max_size_mb}MB",
            details={"size_bytes": size_bytes, "max_size_bytes": max_size_bytes}
        )
    
    return size_bytes

def validate_pagination(limit: int, offset: int, max_limit: int = 1000) -> tuple:
    """Validate pagination parameters."""
    if limit < 1:
        raise DataValidationError(
            field="limit",
            value=limit,
            reason="Limit must be at least 1"
        )
    
    if limit > max_limit:
        raise DataValidationError(
            field="limit", 
            value=limit,
            reason=f"Limit cannot exceed {max_limit}"
        )
    
    if offset < 0:
        raise DataValidationError(
            field="offset",
            value=offset, 
            reason="Offset cannot be negative"
        )
    
    return limit, offset

def validate_date_range(date_from: Optional[str], date_to: Optional[str]) -> tuple:
    """Validate date range parameters."""
    from datetime import datetime
    
    parsed_from = None
    parsed_to = None
    
    if date_from:
        try:
            parsed_from = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
        except ValueError:
            raise DataValidationError(
                field="date_from",
                value=date_from,
                reason="Invalid ISO format date"
            )
    
    if date_to:
        try:
            parsed_to = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
        except ValueError:
            raise DataValidationError(
                field="date_to",
                value=date_to,
                reason="Invalid ISO format date"
            )
    
    if parsed_from and parsed_to and parsed_from >= parsed_to:
        raise DataValidationError(
            field="date_range",
            value=f"{date_from} to {date_to}",
            reason="date_from must be earlier than date_to"
        )
    
    return date_from, date_to

# Request/Response Middleware

class ErrorHandlingMiddleware:
    """Middleware for handling errors across the application."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Generate request ID for tracking
        import uuid
        request_id = str(uuid.uuid4())
        scope["request_id"] = request_id
        
        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            # Handle any unhandled exceptions
            response = await create_error_response(exc, request_id)
            await send({
                "type": "http.response.start",
                "status": response.status_code,
                "headers": [[b"content-type", b"application/json"]]
            })
            await send({
                "type": "http.response.body",
                "body": response.body
            })

# Performance and Rate Limiting

class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self):
        self.requests = {}  # In production, use Redis
    
    def check_rate_limit(self, identifier: str, limit: int, window_seconds: int) -> bool:
        """Check if request exceeds rate limit."""
        import time
        
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier] 
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        if len(self.requests[identifier]) >= limit:
            raise RateLimitExceededError(
                limit=limit,
                window=f"{window_seconds}s",
                details={"current_requests": len(self.requests[identifier])}
            )
        
        # Add current request
        self.requests[identifier].append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()