"""
A2A World Platform - Agent Configuration Management

Configuration management system for agents with environment variables,
Consul KV store integration, and dynamic configuration updates.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
import asyncio

from pydantic import BaseSettings, Field, validator
import yaml

from agents.core.registry import ConsulRegistry


class AgentConfig(BaseSettings):
    """Base configuration class for all agents."""
    
    # Agent identification
    agent_id: Optional[str] = None
    agent_type: str = "base"
    agent_name: Optional[str] = None
    
    # Connection settings
    nats_url: str = Field(default="nats://localhost:4222", env="NATS_URL")
    consul_host: str = Field(default="localhost", env="CONSUL_HOST")
    consul_port: int = Field(default=8500, env="CONSUL_PORT")
    consul_token: Optional[str] = Field(default=None, env="CONSUL_TOKEN")
    
    # Database settings
    database_url: str = Field(
        default="postgresql://a2a_user:a2a_password@localhost:5432/a2a_world",
        env="DATABASE_URL"
    )
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Agent behavior settings
    heartbeat_interval: int = Field(default=30, env="AGENT_HEARTBEAT_INTERVAL")
    max_concurrent_tasks: int = Field(default=5, env="AGENT_MAX_CONCURRENT_TASKS")
    task_timeout: int = Field(default=3600, env="AGENT_TASK_TIMEOUT")
    retry_attempts: int = Field(default=3, env="AGENT_RETRY_ATTEMPTS")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Resource limits
    memory_limit_mb: Optional[int] = Field(default=None, env="AGENT_MEMORY_LIMIT_MB")
    cpu_limit_percent: Optional[int] = Field(default=None, env="AGENT_CPU_LIMIT_PERCENT")
    
    # Health check settings
    health_check_port: int = Field(default=0, env="AGENT_HEALTH_PORT")  # 0 = auto-assign
    health_check_path: str = Field(default="/health", env="AGENT_HEALTH_PATH")
    
    # Data paths
    data_path: Path = Field(default=Path("./data"), env="DATA_PATH")
    temp_path: Path = Field(default=Path("./temp"), env="TEMP_PATH")
    
    # Feature flags
    enable_metrics: bool = Field(default=True, env="AGENT_ENABLE_METRICS")
    enable_tracing: bool = Field(default=False, env="AGENT_ENABLE_TRACING")
    enable_profiling: bool = Field(default=False, env="AGENT_ENABLE_PROFILING")
    
    # Custom configuration
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow additional fields
    
    @validator("data_path", "temp_path", pre=True)
    def ensure_path(cls, v):
        """Ensure path fields are Path objects."""
        return Path(v) if not isinstance(v, Path) else v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    def get_consul_config_key(self) -> str:
        """Get the Consul configuration key for this agent."""
        return f"a2a-world/agents/{self.agent_type}/{self.agent_id or 'default'}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = self.dict()
        
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        return config_dict


class ParserAgentConfig(AgentConfig):
    """Configuration for KML/GeoJSON parser agents."""
    
    agent_type: str = "parser"
    
    # Parser-specific settings
    supported_formats: List[str] = Field(default=["kml", "kmz", "geojson"])
    max_file_size_mb: int = Field(default=100, env="PARSER_MAX_FILE_SIZE_MB")
    batch_size: int = Field(default=10, env="PARSER_BATCH_SIZE")
    
    # Processing settings
    validate_geometry: bool = Field(default=True, env="PARSER_VALIDATE_GEOMETRY")
    extract_metadata: bool = Field(default=True, env="PARSER_EXTRACT_METADATA")
    convert_coordinates: bool = Field(default=True, env="PARSER_CONVERT_COORDINATES")
    target_srid: int = Field(default=4326, env="PARSER_TARGET_SRID")


class DiscoveryAgentConfig(AgentConfig):
    """Configuration for pattern discovery agents."""
    
    agent_type: str = "discovery"
    
    # Discovery algorithm settings
    default_algorithm: str = Field(default="hdbscan", env="DISCOVERY_DEFAULT_ALGORITHM")
    min_cluster_size: int = Field(default=5, env="DISCOVERY_MIN_CLUSTER_SIZE")
    min_samples: int = Field(default=3, env="DISCOVERY_MIN_SAMPLES")
    confidence_threshold: float = Field(default=0.7, env="DISCOVERY_CONFIDENCE_THRESHOLD")
    
    # Spatial analysis settings
    search_radius_km: float = Field(default=50.0, env="DISCOVERY_SEARCH_RADIUS_KM")
    max_distance_km: float = Field(default=1000.0, env="DISCOVERY_MAX_DISTANCE_KM")
    
    # Performance settings
    max_features_per_analysis: int = Field(default=10000, env="DISCOVERY_MAX_FEATURES")
    enable_parallel_processing: bool = Field(default=True, env="DISCOVERY_ENABLE_PARALLEL")


class ValidationAgentConfig(AgentConfig):
    """Configuration for validation agents."""
    
    agent_type: str = "validation"
    
    # Statistical validation settings
    significance_level: float = Field(default=0.05, env="VALIDATION_SIGNIFICANCE_LEVEL")
    min_sample_size: int = Field(default=30, env="VALIDATION_MIN_SAMPLE_SIZE")
    bootstrap_iterations: int = Field(default=1000, env="VALIDATION_BOOTSTRAP_ITERATIONS")
    
    # Validation methods
    enable_morans_i: bool = Field(default=True, env="VALIDATION_ENABLE_MORANS_I")
    enable_getis_ord: bool = Field(default=True, env="VALIDATION_ENABLE_GETIS_ORD")
    enable_ripley_k: bool = Field(default=False, env="VALIDATION_ENABLE_RIPLEY_K")
    
    # Cross-validation settings
    cross_validation_folds: int = Field(default=5, env="VALIDATION_CV_FOLDS")
    test_split_ratio: float = Field(default=0.2, env="VALIDATION_TEST_SPLIT_RATIO")


class MonitorAgentConfig(AgentConfig):
    """Configuration for monitoring agents."""
    
    agent_type: str = "monitoring"
    
    # Monitoring intervals
    system_check_interval: int = Field(default=60, env="MONITOR_SYSTEM_CHECK_INTERVAL")
    agent_check_interval: int = Field(default=30, env="MONITOR_AGENT_CHECK_INTERVAL")
    health_check_timeout: int = Field(default=10, env="MONITOR_HEALTH_CHECK_TIMEOUT")
    
    # Alert thresholds
    cpu_threshold: float = Field(default=80.0, env="MONITOR_CPU_THRESHOLD")
    memory_threshold: float = Field(default=85.0, env="MONITOR_MEMORY_THRESHOLD")
    disk_threshold: float = Field(default=90.0, env="MONITOR_DISK_THRESHOLD")
    response_time_threshold: float = Field(default=5.0, env="MONITOR_RESPONSE_TIME_THRESHOLD")
    
    # Alert settings
    enable_alerts: bool = Field(default=True, env="MONITOR_ENABLE_ALERTS")
    alert_cooldown: int = Field(default=300, env="MONITOR_ALERT_COOLDOWN")  # 5 minutes


class ConfigurationManager:
    """
    Manages agent configuration with support for environment variables,
    file-based config, and Consul KV store integration.
    """
    
    def __init__(self, consul_registry: Optional[ConsulRegistry] = None):
        self.consul = consul_registry
        self.logger = logging.getLogger("config_manager")
        
        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        self._watch_tasks: Dict[str, asyncio.Task] = {}
        self._change_callbacks: Dict[str, List[Callable]] = {}
        
        # Configuration file paths
        self.config_paths = [
            Path("./config/agent.yaml"),
            Path("./config/agent.yml"),
            Path("./agent.yaml"),
            Path("./agent.yml")
        ]
    
    def load_config(self, agent_type: str, config_file: Optional[Path] = None) -> AgentConfig:
        """Load configuration for a specific agent type."""
        try:
            # Determine config class based on agent type
            config_class = self._get_config_class(agent_type)
            
            # Load base configuration from environment variables
            config_data = {}
            
            # Load from configuration file if specified or found
            if config_file:
                config_data.update(self._load_from_file(config_file))
            else:
                for path in self.config_paths:
                    if path.exists():
                        config_data.update(self._load_from_file(path))
                        break
            
            # Create configuration instance
            config = config_class(**config_data)
            
            # Cache the configuration
            cache_key = f"{agent_type}:{config.agent_id or 'default'}"
            self._config_cache[cache_key] = config
            
            self.logger.info(f"Loaded configuration for {agent_type} agent: {config.agent_id}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration for {agent_type}: {e}")
            # Return default configuration as fallback
            config_class = self._get_config_class(agent_type)
            return config_class()
    
    async def load_from_consul(self, agent_type: str, agent_id: Optional[str] = None) -> Optional[AgentConfig]:
        """Load configuration from Consul KV store."""
        if not self.consul:
            self.logger.warning("Consul registry not available for configuration loading")
            return None
        
        try:
            config_key = f"a2a-world/agents/{agent_type}/{agent_id or 'default'}"
            config_data = await self.consul.get_configuration(config_key)
            
            if config_data:
                config_class = self._get_config_class(agent_type)
                config = config_class(**config_data)
                
                # Cache the configuration
                cache_key = f"{agent_type}:{agent_id or 'default'}"
                self._config_cache[cache_key] = config
                
                self.logger.info(f"Loaded configuration from Consul: {config_key}")
                return config
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from Consul: {e}")
            return None
    
    async def save_to_consul(self, config: AgentConfig) -> bool:
        """Save configuration to Consul KV store."""
        if not self.consul:
            self.logger.warning("Consul registry not available for configuration saving")
            return False
        
        try:
            config_key = config.get_consul_config_key()
            config_data = config.to_dict()
            
            success = await self.consul.set_configuration(config_key, config_data)
            
            if success:
                self.logger.info(f"Saved configuration to Consul: {config_key}")
            else:
                self.logger.error(f"Failed to save configuration to Consul: {config_key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to Consul: {e}")
            return False
    
    async def watch_config(self, config: AgentConfig, callback: Callable[[AgentConfig], None]):
        """Watch for configuration changes in Consul."""
        if not self.consul:
            return
        
        config_key = config.get_consul_config_key()
        
        async def config_change_handler(key: str, new_value: Any, old_value: Any):
            try:
                if new_value != old_value:
                    # Parse new configuration
                    config_class = self._get_config_class(config.agent_type)
                    new_config = config_class(**new_value)
                    
                    # Update cache
                    cache_key = f"{config.agent_type}:{config.agent_id or 'default'}"
                    self._config_cache[cache_key] = new_config
                    
                    # Notify callback
                    await callback(new_config)
                    
                    self.logger.info(f"Configuration updated: {key}")
            except Exception as e:
                self.logger.error(f"Error handling configuration change: {e}")
        
        # Start watching
        watch_task = asyncio.create_task(
            self.consul.watch_configuration(config_key, config_change_handler)
        )
        self._watch_tasks[config_key] = watch_task
    
    def register_change_callback(self, config_key: str, callback: Callable):
        """Register a callback for configuration changes."""
        if config_key not in self._change_callbacks:
            self._change_callbacks[config_key] = []
        self._change_callbacks[config_key].append(callback)
    
    def get_cached_config(self, agent_type: str, agent_id: Optional[str] = None) -> Optional[AgentConfig]:
        """Get cached configuration."""
        cache_key = f"{agent_type}:{agent_id or 'default'}"
        return self._config_cache.get(cache_key)
    
    def validate_config(self, config: AgentConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        try:
            # Validate paths exist or can be created
            for path_field in ["data_path", "temp_path"]:
                path = getattr(config, path_field)
                if path and not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        issues.append(f"Cannot create {path_field}: {path} - {e}")
            
            # Validate network connectivity (basic checks)
            if config.heartbeat_interval <= 0:
                issues.append("Heartbeat interval must be positive")
            
            if config.max_concurrent_tasks <= 0:
                issues.append("Max concurrent tasks must be positive")
            
            if config.task_timeout <= 0:
                issues.append("Task timeout must be positive")
            
            # Agent-specific validation
            if isinstance(config, DiscoveryAgentConfig):
                if config.confidence_threshold < 0 or config.confidence_threshold > 1:
                    issues.append("Confidence threshold must be between 0 and 1")
                
                if config.min_cluster_size <= 0:
                    issues.append("Min cluster size must be positive")
            
            elif isinstance(config, ValidationAgentConfig):
                if config.significance_level <= 0 or config.significance_level >= 1:
                    issues.append("Significance level must be between 0 and 1")
                
                if config.min_sample_size <= 0:
                    issues.append("Min sample size must be positive")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues
    
    def _get_config_class(self, agent_type: str) -> type:
        """Get the appropriate configuration class for agent type."""
        config_classes = {
            "parser": ParserAgentConfig,
            "kml_parser": ParserAgentConfig,
            "discovery": DiscoveryAgentConfig,
            "pattern_discovery": DiscoveryAgentConfig,
            "validation": ValidationAgentConfig,
            "monitoring": MonitorAgentConfig,
            "monitor": MonitorAgentConfig
        }
        
        return config_classes.get(agent_type, AgentConfig)
    
    def _load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    self.logger.warning(f"Unsupported config file format: {file_path}")
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to load config file {file_path}: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup configuration manager resources."""
        # Cancel watch tasks
        for task in self._watch_tasks.values():
            task.cancel()
        
        if self._watch_tasks:
            await asyncio.gather(*self._watch_tasks.values(), return_exceptions=True)
        
        self._watch_tasks.clear()
        self._config_cache.clear()
        self._change_callbacks.clear()


# Global configuration manager instance
_global_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(consul_registry: Optional[ConsulRegistry] = None) -> ConfigurationManager:
    """Get or create global configuration manager."""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager(consul_registry)
    
    return _global_config_manager


async def cleanup_config_manager():
    """Cleanup global configuration manager."""
    global _global_config_manager
    
    if _global_config_manager:
        await _global_config_manager.cleanup()
        _global_config_manager = None


def load_agent_config(agent_type: str, config_file: Optional[str] = None) -> AgentConfig:
    """Convenience function to load agent configuration."""
    config_manager = get_config_manager()
    config_path = Path(config_file) if config_file else None
    return config_manager.load_config(agent_type, config_path)