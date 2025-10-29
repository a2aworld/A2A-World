"""
A2A World Platform - Agent Health Check Server

HTTP server for agent health monitoring and status reporting.
Provides endpoints for external monitoring systems, load balancers,
and orchestration platforms.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from aiohttp import web, ClientSession
import aiohttp_cors


class HealthCheckServer:
    """
    HTTP server for agent health checks and status reporting.
    
    Provides endpoints:
    - GET /health - Basic health check
    - GET /status - Detailed agent status
    - GET /metrics - Agent metrics
    - GET /ready - Readiness probe
    - GET /alive - Liveness probe
    """
    
    def __init__(
        self,
        agent,
        host: str = "0.0.0.0",
        port: int = 0,  # 0 = auto-assign
        enable_cors: bool = True
    ):
        self.agent = agent
        self.host = host
        self.port = port
        self.enable_cors = enable_cors
        self.logger = logging.getLogger(f"health_server.{agent.agent_id}")
        
        # Server components
        self.app: Optional[web.Application] = None
        self.server: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.actual_port: Optional[int] = None
        
        # Health check functions
        self.custom_health_checks: Dict[str, Callable] = {}
        
    async def start(self) -> bool:
        """Start the health check server."""
        try:
            # Create web application
            self.app = web.Application()
            
            # Setup routes
            self._setup_routes()
            
            # Setup CORS if enabled
            if self.enable_cors:
                self._setup_cors()
            
            # Create and start server
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            # Find available port if auto-assign
            if self.port == 0:
                import socket
                sock = socket.socket()
                sock.bind(('', 0))
                self.actual_port = sock.getsockname()[1]
                sock.close()
            else:
                self.actual_port = self.port
            
            # Start site
            site = web.TCPSite(runner, self.host, self.actual_port)
            await site.start()
            
            self.server = runner
            self.site = site
            
            self.logger.info(f"Health check server started on {self.host}:{self.actual_port}")
            
            # Update agent config with actual port
            if hasattr(self.agent.config, 'health_check_port'):
                self.agent.config.health_check_port = self.actual_port
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start health check server: {e}")
            return False
    
    async def stop(self):
        """Stop the health check server."""
        try:
            if self.site:
                await self.site.stop()
                self.site = None
            
            if self.server:
                await self.server.cleanup()
                self.server = None
            
            self.logger.info("Health check server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping health check server: {e}")
    
    def _setup_routes(self):
        """Setup HTTP routes for health checks."""
        # Basic health check
        self.app.router.add_get('/health', self._health_handler)
        self.app.router.add_get('/', self._health_handler)  # Root endpoint
        
        # Detailed status
        self.app.router.add_get('/status', self._status_handler)
        
        # Metrics endpoint
        self.app.router.add_get('/metrics', self._metrics_handler)
        
        # Kubernetes-style probes
        self.app.router.add_get('/ready', self._readiness_handler)
        self.app.router.add_get('/alive', self._liveness_handler)
        
        # Configuration endpoint
        self.app.router.add_get('/config', self._config_handler)
        
        # Version/info endpoint
        self.app.router.add_get('/info', self._info_handler)
    
    def _setup_cors(self):
        """Setup CORS for cross-origin requests."""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def _health_handler(self, request: web.Request) -> web.Response:
        """Basic health check endpoint."""
        try:
            # Check agent health
            health_status = self.agent.health_status
            agent_status = self.agent.status
            
            # Determine HTTP status code
            if health_status == "healthy" and agent_status == "running":
                status_code = 200
                status = "healthy"
            elif health_status in ["degraded", "warning"]:
                status_code = 200  # Still responding but degraded
                status = "degraded"
            else:
                status_code = 503  # Service unavailable
                status = "unhealthy"
            
            response_data = {
                "status": status,
                "agent_id": self.agent.agent_id,
                "agent_type": self.agent.agent_type,
                "health_status": health_status,
                "agent_status": agent_status,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": self.agent.metrics.get("uptime_seconds", 0)
            }
            
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return web.json_response(
                {"status": "error", "error": str(e)},
                status=500
            )
    
    async def _status_handler(self, request: web.Request) -> web.Response:
        """Detailed agent status endpoint."""
        try:
            status_data = self.agent.get_status()
            
            # Add health check server info
            status_data["health_server"] = {
                "host": self.host,
                "port": self.actual_port,
                "endpoints": ["/health", "/status", "/metrics", "/ready", "/alive", "/config", "/info"]
            }
            
            # Add custom health checks
            if self.custom_health_checks:
                custom_results = {}
                for name, check_func in self.custom_health_checks.items():
                    try:
                        result = await check_func()
                        custom_results[name] = result
                    except Exception as e:
                        custom_results[name] = {"status": "error", "error": str(e)}
                
                status_data["custom_health_checks"] = custom_results
            
            return web.json_response(status_data)
            
        except Exception as e:
            self.logger.error(f"Status check error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _metrics_handler(self, request: web.Request) -> web.Response:
        """Agent metrics endpoint."""
        try:
            # Get agent metrics
            base_metrics = self.agent.metrics.copy()
            
            # Get custom metrics if available
            custom_metrics = await self.agent.collect_metrics()
            if custom_metrics:
                base_metrics.update(custom_metrics)
            
            # Add timestamp
            base_metrics["timestamp"] = datetime.utcnow().isoformat()
            base_metrics["agent_id"] = self.agent.agent_id
            base_metrics["agent_type"] = self.agent.agent_type
            
            return web.json_response(base_metrics)
            
        except Exception as e:
            self.logger.error(f"Metrics error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _readiness_handler(self, request: web.Request) -> web.Response:
        """Kubernetes-style readiness probe."""
        try:
            # Agent is ready if it's running and healthy
            ready = (
                self.agent.status == "running" and
                self.agent.health_status in ["healthy", "degraded"]
            )
            
            if ready:
                return web.json_response(
                    {
                        "status": "ready",
                        "agent_id": self.agent.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    status=200
                )
            else:
                return web.json_response(
                    {
                        "status": "not_ready",
                        "agent_id": self.agent.agent_id,
                        "agent_status": self.agent.status,
                        "health_status": self.agent.health_status,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    status=503
                )
                
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": str(e)},
                status=500
            )
    
    async def _liveness_handler(self, request: web.Request) -> web.Response:
        """Kubernetes-style liveness probe."""
        try:
            # Agent is alive if it's not stopped or in error state
            alive = self.agent.status not in ["stopped", "error"]
            
            if alive:
                return web.json_response(
                    {
                        "status": "alive",
                        "agent_id": self.agent.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    status=200
                )
            else:
                return web.json_response(
                    {
                        "status": "dead",
                        "agent_id": self.agent.agent_id,
                        "agent_status": self.agent.status,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    status=503
                )
                
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": str(e)},
                status=500
            )
    
    async def _config_handler(self, request: web.Request) -> web.Response:
        """Agent configuration endpoint."""
        try:
            # Return sanitized configuration (remove sensitive data)
            config_data = self.agent.config.to_dict()
            
            # Remove sensitive fields
            sensitive_fields = ["consul_token", "secret_key", "password"]
            for field in sensitive_fields:
                if field in config_data:
                    config_data[field] = "***REDACTED***"
            
            return web.json_response({
                "agent_id": self.agent.agent_id,
                "agent_type": self.agent.agent_type,
                "config": config_data,
                "capabilities": self.agent.capabilities,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _info_handler(self, request: web.Request) -> web.Response:
        """Agent information endpoint."""
        try:
            import platform
            import sys
            
            info_data = {
                "agent": {
                    "id": self.agent.agent_id,
                    "type": self.agent.agent_type,
                    "version": "1.0.0",
                    "status": self.agent.status,
                    "health_status": self.agent.health_status,
                    "capabilities": self.agent.capabilities,
                    "start_time": self.agent.start_time.isoformat() if self.agent.start_time else None,
                    "uptime_seconds": self.agent.metrics.get("uptime_seconds", 0)
                },
                "system": {
                    "platform": platform.platform(),
                    "python_version": sys.version,
                    "architecture": platform.architecture()[0],
                    "processor": platform.processor(),
                    "hostname": platform.node()
                },
                "health_server": {
                    "host": self.host,
                    "port": self.actual_port,
                    "cors_enabled": self.enable_cors
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return web.json_response(info_data)
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    def add_custom_health_check(self, name: str, check_func: Callable):
        """Add a custom health check function."""
        self.custom_health_checks[name] = check_func
        self.logger.info(f"Added custom health check: {name}")
    
    def remove_custom_health_check(self, name: str):
        """Remove a custom health check function."""
        if name in self.custom_health_checks:
            del self.custom_health_checks[name]
            self.logger.info(f"Removed custom health check: {name}")
    
    def get_health_check_url(self) -> Optional[str]:
        """Get the health check URL for this server."""
        if self.actual_port:
            return f"http://{self.host if self.host != '0.0.0.0' else 'localhost'}:{self.actual_port}/health"
        return None


async def create_health_server(agent, **kwargs) -> HealthCheckServer:
    """
    Factory function to create and start a health check server for an agent.
    """
    # Determine port from agent config
    port = 0
    if hasattr(agent.config, 'health_check_port'):
        port = agent.config.health_check_port
    
    # Create server
    server = HealthCheckServer(agent, port=port, **kwargs)
    
    # Start server
    success = await server.start()
    if not success:
        raise RuntimeError("Failed to start health check server")
    
    return server


# Health check utilities

async def check_service_health(url: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Check the health of a service via HTTP.
    """
    try:
        async with ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        return {
                            "healthy": True,
                            "status_code": response.status,
                            "response": data
                        }
                    except Exception:
                        return {
                            "healthy": True,
                            "status_code": response.status,
                            "response": await response.text()
                        }
                else:
                    return {
                        "healthy": False,
                        "status_code": response.status,
                        "error": f"HTTP {response.status}"
                    }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }


async def check_multiple_services(urls: List[str], timeout: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Check the health of multiple services concurrently.
    """
    tasks = [check_service_health(url, timeout) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    health_results = {}
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            health_results[url] = {
                "healthy": False,
                "error": str(result)
            }
        else:
            health_results[url] = result
    
    return health_results


# Example custom health check functions

async def database_health_check(db_url: str) -> Dict[str, Any]:
    """Example database health check."""
    try:
        # This is a placeholder - implement actual database connection check
        return {
            "status": "healthy",
            "database": "connected",
            "response_time_ms": 10
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def external_api_health_check(api_url: str) -> Dict[str, Any]:
    """Example external API health check."""
    try:
        result = await check_service_health(api_url)
        return {
            "status": "healthy" if result["healthy"] else "unhealthy",
            "api_response": result
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }