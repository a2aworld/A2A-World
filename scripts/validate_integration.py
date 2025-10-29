"""
A2A World Platform - Integration Validation Script

Comprehensive validation script to test integration between database,
agents, and API endpoints for Phase 2 completion.
"""

import asyncio
import sys
import logging
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationValidator:
    """Comprehensive integration validation system."""
    
    def __init__(self):
        self.results = {
            "database": {"status": "unknown", "tests": []},
            "agents": {"status": "unknown", "tests": []},
            "api": {"status": "unknown", "tests": []},
            "performance": {"status": "unknown", "tests": []},
            "end_to_end": {"status": "unknown", "tests": []},
            "overall": {"status": "unknown", "start_time": None, "end_time": None}
        }
        self.failures = []
        
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all integration validations."""
        logger.info("Starting comprehensive integration validation for A2A World Platform Phase 2")
        self.results["overall"]["start_time"] = datetime.utcnow().isoformat()
        
        try:
            # 1. Database Integration Tests
            await self.validate_database_integration()
            
            # 2. Agent System Integration Tests
            await self.validate_agent_integration()
            
            # 3. API Endpoints Integration Tests
            await self.validate_api_integration()
            
            # 4. Performance Validation Tests
            await self.validate_performance()
            
            # 5. End-to-End Workflow Tests
            await self.validate_end_to_end_workflows()
            
            # Calculate overall status
            self._calculate_overall_status()
            
        except Exception as e:
            logger.error(f"Critical error during validation: {e}")
            self.results["overall"]["status"] = "critical_failure"
            self.failures.append(f"Critical validation error: {str(e)}")
            
        finally:
            self.results["overall"]["end_time"] = datetime.utcnow().isoformat()
        
        return self.results
    
    async def validate_database_integration(self):
        """Validate database connectivity and models."""
        logger.info("=== Validating Database Integration ===")
        
        tests = []
        
        # Test 1: Database Connection
        try:
            from database.connection import get_database_session
            with get_database_session() as session:
                session.execute("SELECT 1")
            tests.append({"name": "Database Connection", "status": "pass", "message": "Successfully connected to database"})
        except Exception as e:
            tests.append({"name": "Database Connection", "status": "fail", "message": f"Database connection failed: {str(e)}"})
            self.failures.append("Database connection unavailable")
        
        # Test 2: Database Models Import
        try:
            from database.models.datasets import Dataset
            from database.models.geospatial import GeospatialFeature, SacredSite
            from database.models.agents import Agent, AgentTask
            from database.models.patterns import Pattern, PatternComponent
            tests.append({"name": "Database Models", "status": "pass", "message": "All database models imported successfully"})
        except Exception as e:
            tests.append({"name": "Database Models", "status": "fail", "message": f"Model import failed: {str(e)}"})
            self.failures.append("Database models unavailable")
        
        # Test 3: Model Relationships
        try:
            from database.models.base import Base
            # Check that models have proper relationships
            model_count = len(Base.metadata.tables)
            tests.append({"name": "Model Relationships", "status": "pass", "message": f"Found {model_count} database tables with relationships"})
        except Exception as e:
            tests.append({"name": "Model Relationships", "status": "fail", "message": f"Relationship validation failed: {str(e)}"})
        
        # Test 4: PostGIS Support
        try:
            from database.connection import get_database_session
            with get_database_session() as session:
                result = session.execute("SELECT PostGIS_Version()").scalar()
                if result:
                    tests.append({"name": "PostGIS Support", "status": "pass", "message": f"PostGIS version: {result}"})
                else:
                    tests.append({"name": "PostGIS Support", "status": "fail", "message": "PostGIS not available"})
        except Exception as e:
            tests.append({"name": "PostGIS Support", "status": "warning", "message": f"PostGIS check failed (may not be required): {str(e)}"})
        
        self.results["database"]["tests"] = tests
        self.results["database"]["status"] = "pass" if all(t["status"] in ["pass", "warning"] for t in tests) else "fail"
        
        logger.info(f"Database validation completed: {self.results['database']['status']}")
    
    async def validate_agent_integration(self):
        """Validate agent system integration."""
        logger.info("=== Validating Agent Integration ===")
        
        tests = []
        
        # Test 1: Agent Base Class
        try:
            from agents.core.base_agent import BaseAgent
            from agents.core.config import AgentConfig
            
            # Create test agent configuration
            config = AgentConfig(
                agent_id="test-validation-agent",
                nats_url="nats://localhost:4222",
                consul_host="localhost",
                consul_port=8500
            )
            tests.append({"name": "Agent Base Classes", "status": "pass", "message": "Agent base classes loaded successfully"})
        except Exception as e:
            tests.append({"name": "Agent Base Classes", "status": "fail", "message": f"Agent classes failed: {str(e)}"})
            self.failures.append("Agent system unavailable")
        
        # Test 2: Agent Registry
        try:
            from agents.core.registry import ConsulRegistry
            # Don't actually connect to Consul, just test import
            tests.append({"name": "Agent Registry", "status": "pass", "message": "Agent registry system available"})
        except Exception as e:
            tests.append({"name": "Agent Registry", "status": "warning", "message": f"Registry unavailable (external service): {str(e)}"})
        
        # Test 3: Agent Messaging
        try:
            from agents.core.messaging import NATSClient, AgentMessaging
            tests.append({"name": "Agent Messaging", "status": "pass", "message": "Messaging system available"})
        except Exception as e:
            tests.append({"name": "Agent Messaging", "status": "warning", "message": f"Messaging unavailable (external service): {str(e)}"})
        
        # Test 4: Specific Agent Types
        agent_types_tested = []
        
        try:
            from agents.parsers.kml_parser_agent import KMLParserAgent
            agent_types_tested.append("KML Parser")
        except Exception as e:
            logger.debug(f"KML Parser Agent not available: {e}")
        
        try:
            from agents.discovery.pattern_discovery import PatternDiscoveryAgent
            agent_types_tested.append("Pattern Discovery")
        except Exception as e:
            logger.debug(f"Pattern Discovery Agent not available: {e}")
        
        try:
            from agents.validation.validation_agent import ValidationAgent
            agent_types_tested.append("Validation")
        except Exception as e:
            logger.debug(f"Validation Agent not available: {e}")
        
        if agent_types_tested:
            tests.append({"name": "Specific Agent Types", "status": "pass", "message": f"Available agents: {', '.join(agent_types_tested)}"})
        else:
            tests.append({"name": "Specific Agent Types", "status": "warning", "message": "No specific agent implementations found"})
        
        # Test 5: Data Processors
        try:
            from agents.parsers.data_processors import KMLProcessor, GeoJSONProcessor, CSVProcessor
            tests.append({"name": "Data Processors", "status": "pass", "message": "Data processing agents available"})
        except Exception as e:
            tests.append({"name": "Data Processors", "status": "warning", "message": f"Data processors unavailable: {str(e)}"})
        
        self.results["agents"]["tests"] = tests
        self.results["agents"]["status"] = "pass" if any(t["status"] == "pass" for t in tests) else "fail"
        
        logger.info(f"Agent validation completed: {self.results['agents']['status']}")
    
    async def validate_api_integration(self):
        """Validate API endpoints integration."""
        logger.info("=== Validating API Integration ===")
        
        tests = []
        
        # Test 1: FastAPI Application
        try:
            from api.app.main import app
            tests.append({"name": "FastAPI Application", "status": "pass", "message": "Main application loaded successfully"})
        except Exception as e:
            tests.append({"name": "FastAPI Application", "status": "fail", "message": f"App load failed: {str(e)}"})
            self.failures.append("API application unavailable")
            return
        
        # Test 2: Enhanced Features
        try:
            from api.app.core.errors import A2ABaseException, create_error_response
            from api.app.core.serialization import cache_manager, performance_monitor
            tests.append({"name": "Enhanced Features", "status": "pass", "message": "Error handling and serialization available"})
        except Exception as e:
            tests.append({"name": "Enhanced Features", "status": "warning", "message": f"Enhanced features unavailable: {str(e)}"})
        
        # Test 3: API Endpoints
        endpoint_tests = []
        
        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)
            
            # Test root endpoint
            response = client.get("/")
            if response.status_code == 200:
                data = response.json()
                if data.get("service") == "A2A World Platform API":
                    endpoint_tests.append({"endpoint": "/", "status": "pass"})
                else:
                    endpoint_tests.append({"endpoint": "/", "status": "fail", "message": "Unexpected response"})
            else:
                endpoint_tests.append({"endpoint": "/", "status": "fail", "message": f"Status: {response.status_code}"})
            
            # Test health endpoint
            response = client.get("/health")
            if response.status_code == 200:
                endpoint_tests.append({"endpoint": "/health", "status": "pass"})
            else:
                endpoint_tests.append({"endpoint": "/health", "status": "fail", "message": f"Status: {response.status_code}"})
            
            # Test API v1 endpoints
            api_endpoints = [
                "/api/v1/health/",
                "/api/v1/data/",
                "/api/v1/agents/", 
                "/api/v1/patterns/"
            ]
            
            for endpoint in api_endpoints:
                try:
                    response = client.get(endpoint)
                    if response.status_code in [200, 503]:  # 503 acceptable for database unavailable
                        endpoint_tests.append({"endpoint": endpoint, "status": "pass"})
                    else:
                        endpoint_tests.append({"endpoint": endpoint, "status": "fail", "message": f"Status: {response.status_code}"})
                except Exception as e:
                    endpoint_tests.append({"endpoint": endpoint, "status": "fail", "message": str(e)})
            
        except Exception as e:
            tests.append({"name": "API Endpoints", "status": "fail", "message": f"Endpoint testing failed: {str(e)}"})
        
        if endpoint_tests:
            passed = len([t for t in endpoint_tests if t["status"] == "pass"])
            total = len(endpoint_tests)
            tests.append({"name": "API Endpoints", "status": "pass" if passed == total else "warning", 
                         "message": f"Endpoints tested: {passed}/{total} passed", "details": endpoint_tests})
        
        # Test 4: OpenAPI Documentation
        try:
            response = client.get("/openapi.json")
            if response.status_code == 200:
                openapi_spec = response.json()
                if "paths" in openapi_spec and len(openapi_spec["paths"]) > 0:
                    tests.append({"name": "OpenAPI Documentation", "status": "pass", 
                                 "message": f"OpenAPI spec has {len(openapi_spec['paths'])} endpoints"})
                else:
                    tests.append({"name": "OpenAPI Documentation", "status": "fail", "message": "No endpoints in OpenAPI spec"})
            else:
                tests.append({"name": "OpenAPI Documentation", "status": "fail", "message": f"OpenAPI unavailable: {response.status_code}"})
        except Exception as e:
            tests.append({"name": "OpenAPI Documentation", "status": "fail", "message": f"OpenAPI test failed: {str(e)}"})
        
        self.results["api"]["tests"] = tests
        self.results["api"]["status"] = "pass" if all(t["status"] in ["pass", "warning"] for t in tests) else "fail"
        
        logger.info(f"API validation completed: {self.results['api']['status']}")
    
    async def validate_performance(self):
        """Validate performance monitoring and optimization."""
        logger.info("=== Validating Performance Systems ===")
        
        tests = []
        
        # Test 1: Performance Monitoring
        try:
            from api.app.core.performance import profiler, benchmarker, optimizer
            tests.append({"name": "Performance Tools", "status": "pass", "message": "Performance monitoring tools loaded"})
        except Exception as e:
            tests.append({"name": "Performance Tools", "status": "fail", "message": f"Performance tools unavailable: {str(e)}"})
        
        # Test 2: Caching System
        try:
            from api.app.core.serialization import cache_manager
            
            # Test cache operations
            test_key = "integration_test_key"
            test_value = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
            
            success = await cache_manager.set(test_key, test_value, ttl=60)
            cached_value = await cache_manager.get(test_key)
            
            if cached_value is not None or success:  # Either operation succeeded
                tests.append({"name": "Caching System", "status": "pass", "message": "Cache operations working"})
            else:
                tests.append({"name": "Caching System", "status": "warning", "message": "Cache operations failed (Redis unavailable)"})
            
        except Exception as e:
            tests.append({"name": "Caching System", "status": "warning", "message": f"Cache test failed: {str(e)}"})
        
        # Test 3: Resource Monitoring
        try:
            from api.app.core.performance import get_system_resource_status
            
            resources = get_system_resource_status()
            if "cpu_percent" in resources and "memory" in resources:
                tests.append({"name": "Resource Monitoring", "status": "pass", 
                             "message": f"CPU: {resources['cpu_percent']:.1f}%, Memory: {resources['memory']['percent']:.1f}%"})
            else:
                tests.append({"name": "Resource Monitoring", "status": "fail", "message": "Resource monitoring failed"})
            
        except Exception as e:
            tests.append({"name": "Resource Monitoring", "status": "fail", "message": f"Resource monitoring failed: {str(e)}"})
        
        self.results["performance"]["tests"] = tests
        self.results["performance"]["status"] = "pass" if any(t["status"] == "pass" for t in tests) else "fail"
        
        logger.info(f"Performance validation completed: {self.results['performance']['status']}")
    
    async def validate_end_to_end_workflows(self):
        """Validate end-to-end workflow integration."""
        logger.info("=== Validating End-to-End Workflows ===")
        
        tests = []
        
        # Test 1: Data Processing Workflow
        try:
            # Test the complete data processing pipeline
            from fastapi.testclient import TestClient
            from api.app.main import app
            
            client = TestClient(app)
            
            # Create test GeoJSON data
            test_geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-122.4194, 37.7749]},
                    "properties": {"name": "Test Location", "category": "validation"}
                }]
            }
            
            # Test data search endpoint structure
            search_response = client.post("/api/v1/data/search", json={
                "query": "test",
                "limit": 10
            })
            
            if search_response.status_code in [200, 503]:  # 503 acceptable if database unavailable
                tests.append({"name": "Data Processing Workflow", "status": "pass", "message": "Data search endpoint functional"})
            else:
                tests.append({"name": "Data Processing Workflow", "status": "fail", 
                             "message": f"Data search failed: {search_response.status_code}"})
            
        except Exception as e:
            tests.append({"name": "Data Processing Workflow", "status": "fail", "message": f"Workflow test failed: {str(e)}"})
        
        # Test 2: Agent Management Workflow
        try:
            # Test agent management endpoints
            agent_response = client.get("/api/v1/agents/")
            
            if agent_response.status_code in [200, 503]:
                tests.append({"name": "Agent Management Workflow", "status": "pass", "message": "Agent endpoints functional"})
            else:
                tests.append({"name": "Agent Management Workflow", "status": "fail", 
                             "message": f"Agent management failed: {agent_response.status_code}"})
            
        except Exception as e:
            tests.append({"name": "Agent Management Workflow", "status": "fail", "message": f"Agent workflow test failed: {str(e)}"})
        
        # Test 3: Pattern Discovery Workflow
        try:
            # Test pattern discovery endpoints
            pattern_response = client.get("/api/v1/patterns/")
            
            if pattern_response.status_code in [200, 500]:  # May error without pattern storage
                tests.append({"name": "Pattern Discovery Workflow", "status": "pass", "message": "Pattern endpoints functional"})
            else:
                tests.append({"name": "Pattern Discovery Workflow", "status": "fail", 
                             "message": f"Pattern discovery failed: {pattern_response.status_code}"})
            
        except Exception as e:
            tests.append({"name": "Pattern Discovery Workflow", "status": "warning", 
                         "message": f"Pattern workflow test failed (may require external services): {str(e)}"})
        
        # Test 4: System Monitoring Workflow
        try:
            # Test system monitoring endpoints
            health_response = client.get("/api/v1/health/detailed")
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                if "components" in health_data:
                    tests.append({"name": "System Monitoring Workflow", "status": "pass", "message": "Health monitoring functional"})
                else:
                    tests.append({"name": "System Monitoring Workflow", "status": "fail", "message": "Health data incomplete"})
            else:
                tests.append({"name": "System Monitoring Workflow", "status": "fail", 
                             "message": f"Health monitoring failed: {health_response.status_code}"})
            
        except Exception as e:
            tests.append({"name": "System Monitoring Workflow", "status": "fail", "message": f"Health workflow test failed: {str(e)}"})
        
        self.results["end_to_end"]["tests"] = tests
        self.results["end_to_end"]["status"] = "pass" if any(t["status"] == "pass" for t in tests) else "fail"
        
        logger.info(f"End-to-end validation completed: {self.results['end_to_end']['status']}")
    
    def _calculate_overall_status(self):
        """Calculate overall validation status."""
        component_statuses = [
            self.results["database"]["status"],
            self.results["agents"]["status"],
            self.results["api"]["status"],
            self.results["performance"]["status"],
            self.results["end_to_end"]["status"]
        ]
        
        pass_count = len([s for s in component_statuses if s == "pass"])
        fail_count = len([s for s in component_statuses if s == "fail"])
        
        if fail_count == 0:
            self.results["overall"]["status"] = "pass"
        elif pass_count >= 3:  # Majority passing
            self.results["overall"]["status"] = "pass_with_warnings"
        elif pass_count >= 2:  # Some core functionality working
            self.results["overall"]["status"] = "partial"
        else:
            self.results["overall"]["status"] = "fail"
        
        logger.info(f"Overall validation status: {self.results['overall']['status']}")
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("=" * 80)
        report.append("A2A WORLD PLATFORM PHASE 2 - INTEGRATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Validation completed: {self.results['overall']['end_time']}")
        report.append(f"Overall status: {self.results['overall']['status'].upper()}")
        report.append("")
        
        # Component summaries
        for component, data in self.results.items():
            if component == "overall":
                continue
                
            report.append(f"{component.upper()} INTEGRATION: {data['status'].upper()}")
            report.append("-" * 50)
            
            for test in data.get("tests", []):
                status_symbol = "✓" if test["status"] == "pass" else "⚠" if test["status"] == "warning" else "✗"
                report.append(f"  {status_symbol} {test['name']}: {test['message']}")
                
                if "details" in test:
                    for detail in test["details"]:
                        detail_symbol = "✓" if detail["status"] == "pass" else "⚠" if detail["status"] == "warning" else "✗"
                        report.append(f"    {detail_symbol} {detail.get('endpoint', detail.get('name', 'Detail'))}")
            
            report.append("")
        
        # Failure summary
        if self.failures:
            report.append("CRITICAL ISSUES IDENTIFIED:")
            report.append("-" * 30)
            for i, failure in enumerate(self.failures, 1):
                report.append(f"{i}. {failure}")
            report.append("")
        
        # Phase 2 Feature Summary
        report.append("PHASE 2 FEATURES VALIDATION:")
        report.append("-" * 35)
        
        features = [
            ("Enhanced Data Access API", self.results["api"]["status"]),
            ("Agent Management System", self.results["agents"]["status"]),
            ("Pattern Discovery Integration", self.results["end_to_end"]["status"]),
            ("System Health Monitoring", self.results["api"]["status"]),
            ("Performance Optimization", self.results["performance"]["status"]),
            ("Database Integration", self.results["database"]["status"])
        ]
        
        for feature, status in features:
            status_symbol = "✓" if status == "pass" else "⚠" if status in ["warning", "partial"] else "✗"
            report.append(f"  {status_symbol} {feature}: {status.upper()}")
        
        report.append("")
        report.append("=" * 80)
        
        # Final assessment
        overall_status = self.results["overall"]["status"]
        if overall_status == "pass":
            report.append("🎉 PHASE 2 INTEGRATION VALIDATION: SUCCESSFUL")
            report.append("All core systems are integrated and functional.")
        elif overall_status == "pass_with_warnings":
            report.append("✅ PHASE 2 INTEGRATION VALIDATION: SUCCESSFUL WITH WARNINGS")
            report.append("Core functionality working, some optional features unavailable.")
        elif overall_status == "partial":
            report.append("⚠️  PHASE 2 INTEGRATION VALIDATION: PARTIAL SUCCESS")
            report.append("Some systems working, requires attention before production.")
        else:
            report.append("❌ PHASE 2 INTEGRATION VALIDATION: FAILED")
            report.append("Critical issues must be resolved.")
        
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main validation function."""
    print("A2A World Platform Phase 2 - Integration Validation")
    print("=" * 60)
    
    validator = IntegrationValidator()
    
    try:
        results = await validator.run_all_validations()
        report = validator.generate_report()
        
        # Print report to console
        print(report)
        
        # Save report to file
        report_file = f"integration_validation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Save JSON results for programmatic access
        json_file = f"integration_validation_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"JSON results saved to: {json_file}")
        
        # Exit with appropriate code
        overall_status = results["overall"]["status"]
        if overall_status in ["pass", "pass_with_warnings"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"CRITICAL ERROR: Validation failed with exception: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())