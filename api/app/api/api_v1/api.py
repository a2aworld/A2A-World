"""
A2A World Platform - API Router

Main API router that includes all API endpoints.
"""

from fastapi import APIRouter

from app.api.api_v1.endpoints import agents, patterns, data, health, validation, consensus

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(patterns.router, prefix="/patterns", tags=["patterns"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(validation.router, prefix="/validation", tags=["validation"])
api_router.include_router(consensus.router, prefix="/consensus", tags=["consensus"])