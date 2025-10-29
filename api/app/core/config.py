"""
A2A World Platform - Configuration Settings

This module contains all configuration settings for the A2A World API.
Uses Pydantic settings for environment variable management.
"""

import os
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseSettings, AnyHttpUrl, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "A2A World Platform"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "a2a_user"
    POSTGRES_PASSWORD: str = "a2a_password"
    POSTGRES_DB: str = "a2a_world"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}:{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"
    
    # Redis Configuration (for caching)
    REDIS_URL: str = "redis://localhost:6379"
    
    # NATS Configuration (for agent messaging)
    NATS_URL: str = "nats://localhost:4222"
    
    # Consul Configuration (for service discovery)
    CONSUL_HOST: str = "localhost"
    CONSUL_PORT: int = 8500
    
    # Security Configuration
    SECRET_KEY: str = "a2a-world-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # LLM Configuration
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    
    # File Storage Configuration
    DATA_STORAGE_PATH: str = "./data"
    MAX_FILE_SIZE_MB: int = 100
    
    # Agent Configuration
    MAX_CONCURRENT_AGENTS: int = 10
    AGENT_HEARTBEAT_INTERVAL: int = 30  # seconds
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()