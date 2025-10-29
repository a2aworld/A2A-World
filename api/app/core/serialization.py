"""
A2A World Platform - Request/Response Serialization and Caching

Comprehensive utilities for request/response serialization, caching,
and performance optimization for the A2A World REST API.
"""

import json
import pickle
import hashlib
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Redis connection (in production, use proper connection pooling)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    REDIS_AVAILABLE = True
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False

# Serialization Utilities

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for A2A World API responses."""
    
    def default(self, obj):
        """Handle special object types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
            # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            # Generic objects
            return obj.__dict__
        elif isinstance(obj, bytes):
            # Handle binary data
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return f"<binary data: {len(obj)} bytes>"
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def serialize_response(data: Any, pretty: bool = False) -> str:
    """Serialize response data to JSON with custom encoding."""
    try:
        if pretty:
            return json.dumps(data, cls=CustomJSONEncoder, indent=2, ensure_ascii=False)
        else:
            return json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Serialization error: {e}")
        # Fallback to basic serialization
        return json.dumps({"error": "Serialization failed", "details": str(e)})

def deserialize_request(data: str) -> Dict[str, Any]:
    """Deserialize request data from JSON."""
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.error(f"Deserialization error: {e}")
        raise ValueError(f"Invalid JSON: {str(e)}")

# Response Models for Consistent API Structure

class APIResponse(BaseModel):
    """Base API response structure."""
    success: bool = True
    timestamp: str = None
    request_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    cached: bool = False
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow().isoformat()
        super().__init__(**data)

class DataResponse(APIResponse):
    """Response for data endpoints."""
    data: Any
    total_count: Optional[int] = None
    page_info: Optional[Dict[str, Any]] = None

class ListResponse(APIResponse):
    """Response for list endpoints with pagination."""
    items: List[Any]
    total: int
    limit: int
    offset: int
    has_more: bool
    page_info: Dict[str, Any] = {}

class OperationResponse(APIResponse):
    """Response for operation endpoints."""
    operation: str
    status: str
    message: str
    details: Dict[str, Any] = {}

# Caching System

class CacheManager:
    """Comprehensive caching manager for API responses."""
    
    def __init__(self, redis_client=None, default_ttl: int = 300):
        self.redis_client = redis_client or globals().get('redis_client')
        self.default_ttl = default_ttl
        self.local_cache = {}  # Fallback local cache
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and parameters."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else []
        }
        key_string = json.dumps(key_data, sort_keys=True, cls=CustomJSONEncoder)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"a2a:{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client and REDIS_AVAILABLE:
                # Try Redis first
                value = self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    return json.loads(value)
            
            # Fallback to local cache
            if key in self.local_cache:
                entry = self.local_cache[key]
                if entry["expires_at"] > datetime.utcnow():
                    self.cache_stats["hits"] += 1
                    return entry["data"]
                else:
                    # Expired entry
                    del self.local_cache[key]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        
        try:
            serialized_value = serialize_response(value)
            
            if self.redis_client and REDIS_AVAILABLE:
                # Set in Redis
                self.redis_client.setex(key, ttl, serialized_value)
            
            # Set in local cache as fallback
            self.local_cache[key] = {
                "data": value,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl)
            }
            
            self.cache_stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client and REDIS_AVAILABLE:
                self.redis_client.delete(key)
            
            if key in self.local_cache:
                del self.local_cache[key]
            
            self.cache_stats["deletes"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        deleted_count = 0
        
        try:
            if self.redis_client and REDIS_AVAILABLE:
                # Get all matching keys
                keys = self.redis_client.keys(f"a2a:{pattern}:*")
                if keys:
                    deleted_count = self.redis_client.delete(*keys)
            
            # Clean local cache
            keys_to_delete = [k for k in self.local_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.local_cache[key]
                deleted_count += 1
            
            logger.info(f"Invalidated {deleted_count} cache entries for pattern: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache pattern invalidation error for {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "local_cache_size": len(self.local_cache),
            "redis_available": REDIS_AVAILABLE
        }

# Global cache manager instance
cache_manager = CacheManager()

# Caching Decorators

def cache_response(
    ttl: int = 300,
    key_prefix: str = "response",
    vary_on: Optional[List[str]] = None,
    skip_if: Optional[Callable] = None
):
    """Decorator to cache API endpoint responses."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request information for cache key
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Build cache key
            cache_key_parts = []
            if vary_on:
                for field in vary_on:
                    if field in kwargs:
                        cache_key_parts.append(f"{field}:{kwargs[field]}")
                    elif request and hasattr(request, field):
                        cache_key_parts.append(f"{field}:{getattr(request, field)}")
            
            cache_key = cache_manager._generate_cache_key(
                f"{key_prefix}:{func.__name__}",
                *cache_key_parts
            )
            
            # Check if we should skip caching
            if skip_if and skip_if(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                # Add cache indicators to response
                if isinstance(cached_result, dict):
                    cached_result["cached"] = True
                    cached_result["cache_key"] = cache_key
                return cached_result
            
            # Execute function and cache result
            start_time = datetime.utcnow()
            result = await func(*args, **kwargs)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Add execution time to response if it's a dict
            if isinstance(result, dict):
                result["execution_time_ms"] = round(execution_time, 2)
                result["cached"] = False
            
            # Cache the result
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

def invalidate_cache_on_change(patterns: List[str]):
    """Decorator to invalidate cache when data changes."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute the function first
            result = await func(*args, **kwargs)
            
            # Invalidate cache patterns
            for pattern in patterns:
                await cache_manager.invalidate_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator

# Response Processing Middleware

class ResponseProcessingMiddleware:
    """Middleware for processing and enhancing API responses."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Capture start time for performance metrics
        start_time = datetime.utcnow()
        
        # Process request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add standard headers
                headers = list(message.get("headers", []))
                headers.extend([
                    [b"x-api-version", b"1.0"],
                    [b"x-request-id", scope.get("request_id", "unknown").encode()],
                    [b"x-response-time", str(round((datetime.utcnow() - start_time).total_seconds() * 1000, 2)).encode()]
                ])
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

# Data Transformation Utilities

class DataTransformer:
    """Utilities for transforming data between different formats."""
    
    @staticmethod
    def to_geojson(features: List[Dict]) -> Dict[str, Any]:
        """Transform features to GeoJSON format."""
        geojson_features = []
        
        for feature in features:
            geojson_feature = {
                "type": "Feature",
                "id": feature.get("id"),
                "geometry": feature.get("geometry", {
                    "type": "Point",
                    "coordinates": [
                        feature.get("longitude", 0),
                        feature.get("latitude", 0)
                    ]
                }),
                "properties": {
                    k: v for k, v in feature.items() 
                    if k not in ["id", "geometry", "longitude", "latitude"]
                }
            }
            geojson_features.append(geojson_feature)
        
        return {
            "type": "FeatureCollection",
            "features": geojson_features
        }
    
    @staticmethod
    def to_csv_format(data: List[Dict]) -> List[List[str]]:
        """Transform data to CSV format (list of rows)."""
        if not data:
            return []
        
        # Get all unique keys for headers
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        headers = sorted(list(all_keys))
        rows = [headers]
        
        # Add data rows
        for item in data:
            row = []
            for header in headers:
                value = item.get(header, "")
                # Convert complex types to string
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, cls=CustomJSONEncoder)
                elif value is None:
                    value = ""
                else:
                    value = str(value)
                row.append(value)
            rows.append(row)
        
        return rows
    
    @staticmethod
    def paginate_results(
        items: List[Any], 
        limit: int, 
        offset: int
    ) -> Dict[str, Any]:
        """Apply pagination to results."""
        total = len(items)
        paginated_items = items[offset:offset + limit]
        
        return {
            "items": paginated_items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
            "page_info": {
                "current_page": (offset // limit) + 1 if limit > 0 else 1,
                "total_pages": (total + limit - 1) // limit if limit > 0 else 1,
                "items_on_page": len(paginated_items)
            }
        }

# Performance Monitoring

class PerformanceMonitor:
    """Monitor API performance and generate metrics."""
    
    def __init__(self):
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "average_response_time": 0.0,
            "slow_requests": 0,  # > 1000ms
            "error_count": 0,
            "endpoint_metrics": {}
        }
        self.slow_threshold_ms = 1000
    
    def record_request(
        self, 
        endpoint: str, 
        method: str, 
        response_time_ms: float, 
        status_code: int
    ):
        """Record request metrics."""
        self.metrics["request_count"] += 1
        self.metrics["total_response_time"] += response_time_ms
        self.metrics["average_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["request_count"]
        )
        
        if response_time_ms > self.slow_threshold_ms:
            self.metrics["slow_requests"] += 1
        
        if status_code >= 400:
            self.metrics["error_count"] += 1
        
        # Endpoint-specific metrics
        endpoint_key = f"{method} {endpoint}"
        if endpoint_key not in self.metrics["endpoint_metrics"]:
            self.metrics["endpoint_metrics"][endpoint_key] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "errors": 0
            }
        
        endpoint_metrics = self.metrics["endpoint_metrics"][endpoint_key]
        endpoint_metrics["count"] += 1
        endpoint_metrics["total_time"] += response_time_ms
        endpoint_metrics["average_time"] = (
            endpoint_metrics["total_time"] / endpoint_metrics["count"]
        )
        
        if status_code >= 400:
            endpoint_metrics["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.metrics,
            "error_rate": (
                self.metrics["error_count"] / self.metrics["request_count"] * 100
                if self.metrics["request_count"] > 0 else 0
            ),
            "slow_request_rate": (
                self.metrics["slow_requests"] / self.metrics["request_count"] * 100
                if self.metrics["request_count"] > 0 else 0
            ),
            "cache_stats": cache_manager.get_stats()
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "average_response_time": 0.0,
            "slow_requests": 0,
            "error_count": 0,
            "endpoint_metrics": {}
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Utility Functions

async def compress_response(data: str, compression: str = "gzip") -> bytes:
    """Compress response data."""
    import gzip
    import zlib
    
    if compression == "gzip":
        return gzip.compress(data.encode('utf-8'))
    elif compression == "deflate":
        return zlib.compress(data.encode('utf-8'))
    else:
        return data.encode('utf-8')

def format_response(
    data: Any,
    success: bool = True,
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Format standardized API response."""
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    
    if message:
        response["message"] = message
    
    if request_id:
        response["request_id"] = request_id
    
    response.update(kwargs)
    
    return response