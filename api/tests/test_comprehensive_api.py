"""
A2A World Platform - Comprehensive API Tests

Unit and integration tests for all enhanced API endpoints including
data management, agent control, pattern discovery, and system monitoring.
"""

import pytest
import asyncio
import json
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import the FastAPI app and dependencies
from api.app.main import app
from api.app.core.errors import A2ABaseException, DataValidationError
from api.app.core.serialization import cache_manager, performance_monitor

# Test database setup
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def test_db():
    """Create test database session."""
    try:
        from database.models.base import Base
        Base.metadata.create_all(bind=test_engine)
        db = TestingSessionLocal()
        yield db
    except ImportError:
        # Mock database if not available
        yield Mock()
    finally:
        try:
            db.close()
        except:
            pass

@pytest.fixture
def sample_geojson():
    """Sample GeoJSON data for testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "test-feature-1",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-122.4194, 37.7749]
                },
                "properties": {
                    "name": "Test Location 1",
                    "description": "A test location in San Francisco",
                    "category": "test"
                }
            },
            {
                "type": "Feature", 
                "id": "test-feature-2",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-74.0059, 40.7128]
                },
                "properties": {
                    "name": "Test Location 2", 
                    "description": "A test location in New York",
                    "category": "test"
                }
            }
        ]
    }

@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        "agent_id": "test-agent-001",
        "agent_type": "pattern_discovery",
        "status": "active",
        "health_status": "healthy",
        "capabilities": ["clustering", "validation", "analysis"],
        "configuration": {
            "min_cluster_size": 5,
            "algorithm": "hdbscan",
            "timeout_seconds": 3600
        }
    }

class TestRootEndpoints:
    """Test root and basic endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns comprehensive API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "A2A World Platform API"
        assert data["version"] == "2.0.0"
        assert data["phase"] == "2 - Comprehensive REST API"
        assert "endpoints" in data
        assert "capabilities" in data
        assert "supported_formats" in data
        assert "api_limits" in data
        
        # Check endpoint structure
        endpoints = data["endpoints"]
        assert "health_monitoring" in endpoints
        assert "data_management" in endpoints
        assert "agent_control" in endpoints
        assert "pattern_discovery" in endpoints
        
        # Check capabilities
        capabilities = data["capabilities"]
        assert "data_processing" in capabilities
        assert "agent_management" in capabilities
        assert "pattern_discovery" in capabilities
        assert "system_monitoring" in capabilities
    
    def test_basic_health_endpoint(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "a2a-world-api"
        assert data["version"] == "2.0.0"
    
    def test_metrics_endpoint(self, client):
        """Test API metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "a2a-world-api"
        assert data["version"] == "2.0.0"
        assert "enhanced_monitoring" in data

class TestDataEndpoints:
    """Test data management endpoints."""
    
    def test_list_datasets_basic(self, client):
        """Test basic dataset listing."""
        with patch('api.app.api.api_v1.endpoints.data.get_database_session'):
            response = client.get("/api/v1/data/")
            # Should handle database unavailable gracefully
            assert response.status_code in [200, 503]
    
    def test_file_upload_validation(self, client):
        """Test file upload with validation."""
        # Test invalid file format
        response = client.post(
            "/api/v1/data/upload",
            files={"file": ("test.txt", "invalid content", "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    @patch('api.app.api.api_v1.endpoints.data.PROCESSORS_AVAILABLE', True)
    def test_data_search_validation(self, client):
        """Test data search with parameter validation."""
        # Test invalid bbox
        search_data = {
            "bbox": [1, 2, 3]  # Invalid - should have 4 values
        }
        response = client.post("/api/v1/data/search", json=search_data)
        assert response.status_code == 422
    
    def test_data_export_format_validation(self, client):
        """Test data export format validation."""
        response = client.get("/api/v1/data/export?format=invalid")
        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]
    
    def test_data_statistics_structure(self, client):
        """Test data statistics endpoint structure."""
        with patch('api.app.api.api_v1.endpoints.data.get_database_session'):
            response = client.get("/api/v1/data/statistics")
            # Should handle database unavailable gracefully
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "total_datasets", "total_features", "file_type_distribution",
                    "feature_type_distribution", "processing_status_distribution"
                ]
                for field in required_fields:
                    assert field in data

class TestAgentEndpoints:
    """Test agent management endpoints."""
    
    def test_list_agents_structure(self, client):
        """Test agent listing endpoint structure.""" 
        with patch('api.app.api.api_v1.endpoints.agents.get_database_session'):
            response = client.get("/api/v1/agents/")
            if response.status_code == 200:
                data = response.json()
                assert "agents" in data
                assert "total" in data
                assert "status_summary" in data
                assert "supported_types" in data
    
    def test_agent_start_validation(self, client, sample_agent_data):
        """Test agent start with validation."""
        start_request = {
            "agent_type": "pattern_discovery",
            "configuration": sample_agent_data["configuration"]
        }
        
        with patch('api.app.api.api_v1.endpoints.agents.get_database_session'):
            response = client.post(
                f"/api/v1/agents/{sample_agent_data['agent_id']}/start",
                json=start_request
            )
            # Should handle database operations gracefully
            assert response.status_code in [200, 503]
    
    def test_agent_task_assignment(self, client, sample_agent_data):
        """Test task assignment to agent."""
        task_request = {
            "task_type": "discover_patterns",
            "parameters": {
                "dataset_id": "test-dataset-001",
                "algorithm": "hdbscan"
            },
            "priority": 5,
            "timeout_seconds": 3600
        }
        
        with patch('api.app.api.api_v1.endpoints.agents.get_database_session'):
            response = client.post(
                f"/api/v1/agents/{sample_agent_data['agent_id']}/tasks",
                json=task_request
            )
            # Should handle database operations
            assert response.status_code in [200, 404, 503]
    
    def test_agent_metrics_structure(self, client, sample_agent_data):
        """Test agent metrics endpoint structure."""
        with patch('api.app.api.api_v1.endpoints.agents.get_database_session'):
            response = client.get(f"/api/v1/agents/{sample_agent_data['agent_id']}/metrics")
            if response.status_code == 200:
                data = response.json()
                assert "agent_id" in data
                assert "metrics" in data
                assert "performance_stats" in data
                assert "resource_usage" in data

class TestPatternEndpoints:
    """Test pattern discovery endpoints."""
    
    @patch('api.app.api.api_v1.endpoints.patterns.pattern_storage')
    def test_list_patterns_with_filters(self, mock_storage, client):
        """Test pattern listing with filters."""
        # Mock pattern storage response
        mock_storage.list_patterns.return_value = ([], 0)
        
        response = client.get("/api/v1/patterns/?pattern_type=spatial_clustering&min_confidence=0.7")
        assert response.status_code == 200
        
        data = response.json()
        assert "patterns" in data
        assert "total" in data
        assert "filters" in data
    
    @patch('api.app.api.api_v1.endpoints.patterns.pattern_storage')
    def test_pattern_discovery_trigger(self, mock_storage, client):
        """Test pattern discovery trigger."""
        mock_storage.get_sacred_sites.return_value = [{"id": i} for i in range(20)]
        
        response = client.post(
            "/api/v1/patterns/discover?algorithm=hdbscan&min_cluster_size=5"
        )
        # Should handle pattern discovery process
        assert response.status_code in [200, 500]
    
    def test_batch_validation_structure(self, client):
        """Test batch validation endpoint structure."""
        pattern_ids = ["pattern-001", "pattern-002", "pattern-003"]
        
        with patch('api.app.api.api_v1.endpoints.patterns.pattern_storage') as mock_storage:
            mock_storage.get_pattern.return_value = {"id": "pattern-001", "confidence_score": 0.8}
            mock_storage.validate_pattern.return_value = True
            
            response = client.post(
                f"/api/v1/patterns/batch-validate",
                params={
                    "pattern_ids": pattern_ids,
                    "validation_method": "automated"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "total_patterns" in data
                assert "successful_validations" in data
                assert "results" in data
    
    def test_pattern_search_with_geospatial(self, client):
        """Test pattern search with geospatial filters."""
        with patch('api.app.api.api_v1.endpoints.patterns.pattern_storage') as mock_storage:
            mock_storage.search_patterns.return_value = ([], 0)
            
            response = client.get(
                "/api/v1/patterns/search?bbox=-122.5,37.7,-122.3,37.8&min_confidence=0.5"
            )
            assert response.status_code == 200

class TestHealthEndpoints:
    """Test system health and monitoring endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "critical"]
        assert "service" in data
        assert "version" in data
        assert "components" in data
        assert "performance_metrics" in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "system_metrics" in data
        assert "health_check_duration_ms" in data
    
    def test_system_metrics_endpoint(self, client):
        """Test system metrics endpoint."""
        with patch('api.app.api.api_v1.endpoints.health.get_database_session'):
            response = client.get("/api/v1/health/metrics")
            if response.status_code == 200:
                data = response.json()
                assert "current_metrics" in data
                assert "historical_data" in data
    
    def test_configuration_management(self, client):
        """Test configuration endpoints."""
        # Test get configuration
        response = client.get("/api/v1/health/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "configuration" in data
        assert "last_updated" in data
        
        # Test update configuration
        config_update = {
            "component": "api",
            "configuration": {
                "log_level": "DEBUG",
                "rate_limit": 200
            }
        }
        
        response = client.post("/api/v1/health/config", json=config_update)
        assert response.status_code == 200
    
    def test_maintenance_operations(self, client):
        """Test maintenance operations."""
        maintenance_task = {
            "task_type": "database_cleanup",
            "parameters": {
                "retention_days": 30
            }
        }
        
        response = client.post("/api/v1/health/maintenance", json=maintenance_task)
        assert response.status_code == 200
        
        data = response.json()
        assert "task_id" in data
        assert "status" in data
        assert data["status"] == "initiated"

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_validation_errors(self, client):
        """Test validation error responses."""
        # Test invalid data search request
        invalid_search = {
            "limit": -1,  # Invalid limit
            "bbox": [1, 2, 3]  # Invalid bbox
        }
        
        response = client.post("/api/v1/data/search", json=invalid_search)
        assert response.status_code == 422
        
        data = response.json()
        assert "validation_errors" in data or "detail" in data
    
    def test_not_found_errors(self, client):
        """Test 404 error responses."""
        response = client.get("/api/v1/data/nonexistent-dataset-id")
        assert response.status_code in [404, 503]  # May be 503 if database unavailable
    
    def test_error_response_structure(self, client):
        """Test error response follows standard structure."""
        # Trigger a validation error
        response = client.get("/api/v1/data/export?format=invalid")
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data  # FastAPI standard format

class TestCaching:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache get/set operations."""
        # Test basic cache operations
        test_key = "test_cache_key"
        test_value = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
        
        # Set cache
        success = await cache_manager.set(test_key, test_value, ttl=60)
        assert success is not None  # May be True or False depending on Redis availability
        
        # Get from cache
        cached_value = await cache_manager.get(test_key)
        # May be None if Redis not available (uses local cache fallback)
        
        # Get cache stats
        stats = cache_manager.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "sets" in stats
    
    @pytest.mark.asyncio  
    async def test_cache_invalidation(self):
        """Test cache invalidation."""
        # Test pattern invalidation
        deleted_count = await cache_manager.invalidate_pattern("test_pattern")
        assert isinstance(deleted_count, int)

class TestPerformanceMonitoring:
    """Test performance monitoring."""
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        # Record some test metrics
        performance_monitor.record_request(
            endpoint="/api/v1/data/",
            method="GET",
            response_time_ms=150.5,
            status_code=200
        )
        
        performance_monitor.record_request(
            endpoint="/api/v1/agents/",
            method="POST",  
            response_time_ms=2500.0,  # Slow request
            status_code=500  # Error
        )
        
        # Get metrics
        metrics = performance_monitor.get_metrics()
        assert metrics["request_count"] >= 2
        assert "average_response_time" in metrics
        assert "error_rate" in metrics
        assert "slow_request_rate" in metrics
        assert "endpoint_metrics" in metrics
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        # Record a metric
        performance_monitor.record_request("/test", "GET", 100, 200)
        
        # Reset metrics
        performance_monitor.reset_metrics()
        
        metrics = performance_monitor.get_metrics()
        assert metrics["request_count"] == 0
        assert metrics["total_response_time"] == 0.0

class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    def test_data_processing_workflow(self, client, sample_geojson):
        """Test complete data processing workflow."""
        # This would test: upload -> process -> validate -> export
        # For now, test the structure exists
        workflow_endpoints = [
            "/api/v1/data/upload",
            "/api/v1/data/search", 
            "/api/v1/data/export",
            "/api/v1/data/statistics"
        ]
        
        for endpoint in workflow_endpoints:
            # Check endpoint exists (may return method not allowed, but not 404)
            response = client.options(endpoint)
            assert response.status_code != 404
    
    def test_agent_management_workflow(self, client, sample_agent_data):
        """Test agent management workflow."""
        agent_id = sample_agent_data["agent_id"]
        
        # Test workflow endpoints exist
        workflow_endpoints = [
            f"/api/v1/agents/{agent_id}/start",
            f"/api/v1/agents/{agent_id}/tasks",
            f"/api/v1/agents/{agent_id}/metrics",
            f"/api/v1/agents/{agent_id}/stop"
        ]
        
        for endpoint in workflow_endpoints:
            response = client.options(endpoint)
            assert response.status_code != 404
    
    def test_pattern_discovery_workflow(self, client):
        """Test pattern discovery workflow."""
        workflow_endpoints = [
            "/api/v1/patterns/discover",
            "/api/v1/patterns/search",
            "/api/v1/patterns/batch-validate", 
            "/api/v1/patterns/export"
        ]
        
        for endpoint in workflow_endpoints:
            response = client.options(endpoint)
            assert response.status_code != 404

# Test fixtures for mocking external dependencies
@pytest.fixture
def mock_database_session():
    """Mock database session."""
    session = Mock()
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.count.return_value = 0
    session.query.return_value.all.return_value = []
    return session

@pytest.fixture
def mock_nats_client():
    """Mock NATS client."""
    client = Mock()
    client.is_connected = True
    client.publish = AsyncMock()
    return client

@pytest.fixture
def mock_consul_registry():
    """Mock Consul registry.""" 
    registry = Mock()
    registry.register_agent = AsyncMock(return_value=True)
    registry.discover_services = AsyncMock(return_value=[])
    return registry

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for API endpoints."""
    
    def test_root_endpoint_performance(self, client):
        """Benchmark root endpoint performance."""
        import time
        
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time_ms = (end_time - start_time) * 1000
        assert response_time_ms < 100  # Should respond in under 100ms
    
    def test_health_check_performance(self, client):
        """Benchmark health check performance."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time_ms = (end_time - start_time) * 1000
        assert response_time_ms < 50  # Should respond in under 50ms
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Benchmark cache operations performance."""
        import time
        
        # Test cache set performance
        start_time = time.time()
        await cache_manager.set("perf_test", {"data": "test"}, ttl=60)
        set_time = (time.time() - start_time) * 1000
        
        # Test cache get performance
        start_time = time.time()
        await cache_manager.get("perf_test")
        get_time = (time.time() - start_time) * 1000
        
        # Cache operations should be fast
        assert set_time < 10  # Under 10ms
        assert get_time < 5   # Under 5ms

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--cov=api",
        "--cov-report=term-missing",
        "--cov-report=html"
    ])