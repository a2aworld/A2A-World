"""
A2A World Platform - Main FastAPI Application

This module provides the entry point for the A2A World API server.
Configures the FastAPI application with CORS, middleware, and routing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.api.api_v1.api import api_router

# Create FastAPI application instance
app = FastAPI(
    title="A2A World API",
    description="AI-driven system for discovering meaningful patterns across geospatial data, cultural mythology, and environmental phenomena",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.API_V1_STR else "/openapi.json",
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "A2A World API",
        "version": "0.1.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "a2a-world-api"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )