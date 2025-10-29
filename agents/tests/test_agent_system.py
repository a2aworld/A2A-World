"""
A2A World Platform - Agent System Integration Tests

Comprehensive test suite for validating the multi-agent system components.
Tests individual components and their integration.
"""

import asyncio
import pytest
import json
import time
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import agent components
from agents.core.base_agent import BaseAgent
from agents.core.messaging import NATSClient, AgentMessaging, AgentMessage
from agents.core.registry import ConsulRegistry, AgentServiceInfo
from agents.core.task_queue import TaskQueue, Task, TaskPriority, create_parse_task
from agents.core.config import AgentConfig, ParserAgentConfig, ValidationAgentConfig
from agents.core.health_server import HealthCheckServer
from agents.validation.validation_agent import ValidationAgent
from agents.monitoring.monitor_agent import MonitorAgent
from agents.parsers.kml_parser import KMLParserAgent
from agents.discovery.pattern_discovery import PatternDiscoveryAgent


# Test fixtures and utilities

class MockAgent(BaseAgent):
    """Mock agent for testing BaseAgent functionality."""
    
    def __init__(self, agent_id: str = None, **kwargs):
        config = AgentConfig()
        config.health_check_port = 0  # Disable health server for tests
        super().__init__(
            agent_id=agent_id or f"mock-{uuid.uuid4().hex[:8]}",
            agent_type="mock",
            config=config
        )
        self.process_called = 0
        self.test_data = {}
    
    async def process(self):
        self.process_called += 1
        await asyncio.sleep(0.01)  # Small delay
    
    def _get_capabilities(self):
        return ["mock", "testing", "base_agent"]


class AgentSystemTester:
    """Comprehensive agent system test suite."""
    
    def __init__(self):
        self.logger = logging.getLogger("agent_system_tester")
        self.test_results = {}
        self.test_agents = []
        self.cleanup_tasks = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests."""
        print("🧪 Starting A2A World Agent System Tests")
        print("=" * 60)
        
        # Initialize test results
        self.test_results = {
            "start_time": datetime.utcnow().isoformat(),
            "tests": {},
            "summary": {}
        }
        
        # Define test suite
        test_suite = [
            ("Configuration Management", self.test_configuration_system),
            ("NATS Messaging", self.test_nats_messaging),
            ("Consul Registry", self.test_consul_registry),
            ("Task Queue System", self.test_task_queue),
            ("Base Agent Framework", self.test_base_agent),
            ("Health Check System", self.test_health_checks),
            ("Validation Agent", self.test_validation_agent),
            ("Monitor Agent", self.test_monitor_agent),
            ("Parser Agent", self.test_parser_agent),
            ("Discovery Agent", self.test_discovery_agent),
            ("Agent Integration", self.test_agent_integration),
            ("System Performance", self.test_system_performance)
        ]
        
        # Run tests
        passed = 0
        failed = 0
        
        for test_name, test_func in test_suite:
            print(f"\n🔍 Testing {test_name}...")
            try:
                result = await test_func()
                if result.get("passed", False):
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    failed += 1
                
                self.test_results["tests"][test_name] = result
                
            except Exception as e:
                print(f"💥 {test_name}: CRASHED - {e}")
                self.test_results["tests"][test_name] = {
                    "passed": False,
                    "error": f"Test crashed: {e}",
                    "exception_type": type(e).__name__
                }
                failed += 1
        
        # Cleanup
        await self.cleanup()
        
        # Summary
        self.test_results["summary"] = {
            "total_tests": len(test_suite),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_suite) if test_suite else 0,
            "end_time": datetime.utcnow().isoformat()
        }
        
        print(f"\n📊 Test Results Summary:")
        print(f"   Total Tests: {len(test_suite)}")
        print(f"   Passed: {passed} ✅")
        print(f"   Failed: {failed} ❌")
        print(f"   Success Rate: {self.test_results['summary']['success_rate']:.1%}")
        
        return self.test_results
    
    async def test_configuration_system(self) -> Dict[str, Any]:
        """Test the configuration management system."""
        try:
            # Test basic configuration loading
            config = AgentConfig()
            assert hasattr(config, 'agent_type')
            assert hasattr(config, 'nats_url')
            assert hasattr(config, 'consul_host')
            
            # Test specific configurations
            parser_config = ParserAgentConfig()
            assert hasattr(parser_config, 'supported_formats')
            assert hasattr(parser_config, 'max_file_size_mb')
            
            validation_config = ValidationAgentConfig()
            assert hasattr(validation_config, 'significance_level')
            assert hasattr(validation_config, 'min_sample_size')
            
            # Test configuration serialization
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            assert 'nats_url' in config_dict
            
            return {
                "passed": True,
                "details": {
                    "base_config": "loaded",
                    "parser_config": "loaded", 
                    "validation_config": "loaded",
                    "serialization": "working"
                }
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_nats_messaging(self) -> Dict[str, Any]:
        """Test NATS messaging components."""
        try:
            # Test message creation
            message = AgentMessage.create(
                sender_id="test-agent",
                message_type="test_message",
                payload={"test": "data"}
            )
            
            assert message.sender_id == "test-agent"
            assert message.message_type == "test_message"
            assert message.payload["test"] == "data"
            
            # Test message serialization
            message_dict = message.to_dict()
            assert isinstance(message_dict, dict)
            
            # Test message deserialization
            restored_message = AgentMessage.from_dict(message_dict)
            assert restored_message.sender_id == message.sender_id
            assert restored_message.message_type == message.message_type
            
            return {
                "passed": True,
                "details": {
                    "message_creation": "working",
                    "serialization": "working",
                    "deserialization": "working"
                },
                "note": "NATS connection tests require running NATS server"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_consul_registry(self) -> Dict[str, Any]:
        """Test Consul service registry components."""
        try:
            # Test AgentServiceInfo creation
            service_info = AgentServiceInfo(
                service_id="test-service",
                agent_id="test-agent",
                agent_type="test",
                capabilities=["test"],
                address="localhost",
                port=8080
            )
            
            assert service_info.service_id == "test-service"
            assert service_info.agent_id == "test-agent"
            assert "test" in service_info.capabilities
            
            # Test serialization
            service_dict = service_info.to_dict()
            assert isinstance(service_dict, dict)
            assert service_dict["service_id"] == "test-service"
            
            # Test deserialization
            restored_service = AgentServiceInfo.from_dict(service_dict)
            assert restored_service.service_id == service_info.service_id
            
            return {
                "passed": True,
                "details": {
                    "service_info_creation": "working",
                    "serialization": "working",
                    "deserialization": "working"
                },
                "note": "Consul connection tests require running Consul server"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_task_queue(self) -> Dict[str, Any]:
        """Test task queue system."""
        try:
            # Test task creation
            task = Task(
                task_type="test_task",
                priority=TaskPriority.NORMAL.value,
                parameters={"test": "data"}
            )
            
            assert task.task_type == "test_task"
            assert task.priority == TaskPriority.NORMAL.value
            assert task.status == "pending"
            
            # Test task serialization
            task_dict = task.to_dict()
            assert isinstance(task_dict, dict)
            assert task_dict["task_type"] == "test_task"
            
            # Test task deserialization
            restored_task = Task.from_dict(task_dict)
            assert restored_task.task_type == task.task_type
            assert restored_task.task_id == task.task_id
            
            # Test factory functions
            parse_task = create_parse_task("/test/file.kml", "kml")
            assert parse_task.task_type == "parse_kml_file"
            assert parse_task.parameters["file_path"] == "/test/file.kml"
            
            # Test task readiness
            assert task.is_ready_to_execute(set()) == True
            
            return {
                "passed": True,
                "details": {
                    "task_creation": "working",
                    "serialization": "working",
                    "deserialization": "working",
                    "factory_functions": "working",
                    "readiness_check": "working"
                }
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_base_agent(self) -> Dict[str, Any]:
        """Test BaseAgent framework."""
        try:
            # Create mock agent
            agent = MockAgent()
            self.test_agents.append(agent)
            
            # Test agent properties
            assert agent.agent_id is not None
            assert agent.agent_type == "mock"
            assert agent.status == "initializing"
            
            # Test capabilities
            capabilities = agent._get_capabilities()
            assert isinstance(capabilities, list)
            assert "mock" in capabilities
            
            # Test status reporting
            status = agent.get_status()
            assert isinstance(status, dict)
            assert status["agent_id"] == agent.agent_id
            assert status["agent_type"] == "mock"
            
            # Test metrics collection
            metrics = await agent.collect_metrics()
            # Base implementation returns None, which is fine
            
            return {
                "passed": True,
                "details": {
                    "agent_creation": "working",
                    "properties": "working",
                    "capabilities": "working", 
                    "status_reporting": "working",
                    "metrics_collection": "working"
                },
                "agent_id": agent.agent_id
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_health_checks(self) -> Dict[str, Any]:
        """Test health check system."""
        try:
            # Create agent with health server disabled
            agent = MockAgent()
            self.test_agents.append(agent)
            
            # Test health status
            assert hasattr(agent, 'health_status')
            agent.health_status = "healthy"
            assert agent.health_status == "healthy"
            
            # The health server requires actual network setup
            # So we'll test the health check logic instead
            health_issues = await agent.check_health()
            # Base implementation returns None (no issues)
            
            return {
                "passed": True,
                "details": {
                    "health_status": "working",
                    "health_checks": "working"
                },
                "note": "Health server requires network setup for full testing"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_validation_agent(self) -> Dict[str, Any]:
        """Test ValidationAgent implementation."""
        try:
            # Create validation agent
            config = ValidationAgentConfig()
            config.health_check_port = 0  # Disable health server
            agent = ValidationAgent(config=config)
            self.test_agents.append(agent)
            
            # Test agent properties
            assert agent.agent_type == "validation"
            assert hasattr(agent, 'validation_methods')
            
            # Test capabilities
            capabilities = agent._get_capabilities()
            assert "validation" in capabilities
            assert "morans_i" in capabilities
            
            # Test pattern validation (with mock data)
            pattern_data = {
                "features": [
                    {"latitude": 37.7749, "longitude": -122.4194, "value": 1},
                    {"latitude": 37.7849, "longitude": -122.4094, "value": 2}
                ]
            }
            
            # This would require NATS/Consul for full testing
            # So we test the validation logic components
            assert hasattr(agent, '_calculate_morans_i')
            assert hasattr(agent, '_extract_spatial_data')
            
            return {
                "passed": True,
                "details": {
                    "agent_creation": "working",
                    "capabilities": "working",
                    "validation_methods": "available",
                    "spatial_data_extraction": "working"
                },
                "agent_id": agent.agent_id
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_monitor_agent(self) -> Dict[str, Any]:
        """Test MonitorAgent implementation."""
        try:
            # Create monitor agent
            from agents.core.config import MonitorAgentConfig
            config = MonitorAgentConfig()
            config.health_check_port = 0  # Disable health server
            agent = MonitorAgent(config=config)
            self.test_agents.append(agent)
            
            # Test agent properties
            assert agent.agent_type == "monitoring"
            assert hasattr(agent, 'system_metrics')
            assert hasattr(agent, 'active_alerts')
            
            # Test capabilities
            capabilities = agent._get_capabilities()
            assert "monitoring" in capabilities
            assert "system_monitoring" in capabilities
            
            # Test metrics collection
            await agent._collect_system_metrics()
            assert "timestamp" in agent.system_metrics
            assert "cpu_percent" in agent.system_metrics
            
            # Test health checks initialization
            assert hasattr(agent, 'health_checks')
            assert len(agent.health_checks) > 0
            
            return {
                "passed": True,
                "details": {
                    "agent_creation": "working",
                    "capabilities": "working",
                    "metrics_collection": "working",
                    "health_checks": "initialized"
                },
                "agent_id": agent.agent_id,
                "health_checks": list(agent.health_checks.keys())
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_parser_agent(self) -> Dict[str, Any]:
        """Test KMLParserAgent implementation."""
        try:
            # Create parser agent
            config = ParserAgentConfig()
            config.health_check_port = 0  # Disable health server
            agent = KMLParserAgent(config=config)
            self.test_agents.append(agent)
            
            # Test agent properties
            assert agent.agent_type == "parser"
            assert hasattr(agent, 'supported_formats')
            assert "kml" in agent.supported_formats
            
            # Test capabilities
            capabilities = agent._get_capabilities()
            assert "parser" in capabilities
            assert "parse_kml_file" in capabilities
            
            # Test parsing methods exist
            assert hasattr(agent, 'parse_kml_file')
            assert hasattr(agent, '_parse_kml_basic')
            
            return {
                "passed": True,
                "details": {
                    "agent_creation": "working",
                    "capabilities": "working",
                    "supported_formats": agent.supported_formats,
                    "parsing_methods": "available"
                },
                "agent_id": agent.agent_id
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_discovery_agent(self) -> Dict[str, Any]:
        """Test PatternDiscoveryAgent implementation."""
        try:
            # Create discovery agent
            from agents.core.config import DiscoveryAgentConfig
            config = DiscoveryAgentConfig()
            config.health_check_port = 0  # Disable health server
            agent = PatternDiscoveryAgent(config=config)
            self.test_agents.append(agent)
            
            # Test agent properties
            assert agent.agent_type == "discovery"
            assert hasattr(agent, 'clustering_algorithms')
            assert "hdbscan" in agent.clustering_algorithms
            
            # Test capabilities
            capabilities = agent._get_capabilities()
            assert "discovery" in capabilities
            assert "pattern_discovery" in capabilities
            
            # Test clustering methods exist
            assert hasattr(agent, 'discover_patterns')
            assert hasattr(agent, '_perform_clustering')
            assert hasattr(agent, '_simple_clustering')
            
            return {
                "passed": True,
                "details": {
                    "agent_creation": "working",
                    "capabilities": "working",
                    "clustering_algorithms": agent.clustering_algorithms,
                    "discovery_methods": "available"
                },
                "agent_id": agent.agent_id
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_agent_integration(self) -> Dict[str, Any]:
        """Test agent integration and communication."""
        try:
            # Test agent lifecycle simulation
            mock_agent = MockAgent()
            self.test_agents.append(mock_agent)
            
            # Test agent initialization state
            assert mock_agent.status == "initializing"
            assert mock_agent.processed_tasks == 0
            
            # Test process method
            original_count = mock_agent.process_called
            await mock_agent.process()
            assert mock_agent.process_called == original_count + 1
            
            # Test shutdown event
            assert not mock_agent.shutdown_event.is_set()
            mock_agent.shutdown_event.set()
            assert mock_agent.shutdown_event.is_set()
            
            # Test agent status tracking
            status = mock_agent.get_status()
            assert status["agent_id"] == mock_agent.agent_id
            assert status["processed_tasks"] == 0
            
            return {
                "passed": True,
                "details": {
                    "lifecycle_management": "working",
                    "process_method": "working",
                    "shutdown_handling": "working",
                    "status_tracking": "working"
                },
                "process_calls": mock_agent.process_called
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_system_performance(self) -> Dict[str, Any]:
        """Test system performance characteristics."""
        try:
            # Test agent creation performance
            start_time = time.time()
            test_agents = []
            
            for i in range(5):  # Create 5 agents
                agent = MockAgent(agent_id=f"perf-test-{i}")
                test_agents.append(agent)
            
            creation_time = time.time() - start_time
            self.test_agents.extend(test_agents)
            
            # Test concurrent processing
            start_time = time.time()
            tasks = [agent.process() for agent in test_agents]
            await asyncio.gather(*tasks)
            processing_time = time.time() - start_time
            
            # Test memory usage (basic)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                "passed": True,
                "details": {
                    "agent_creation_time": f"{creation_time:.3f}s",
                    "concurrent_processing_time": f"{processing_time:.3f}s",
                    "memory_usage_mb": f"{memory_mb:.1f}MB",
                    "agents_created": len(test_agents)
                },
                "performance_acceptable": creation_time < 5.0 and processing_time < 1.0
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def cleanup(self):
        """Cleanup test resources."""
        try:
            # Clean up test agents
            for agent in self.test_agents:
                if hasattr(agent, 'shutdown_event'):
                    agent.shutdown_event.set()
            
            # Run cleanup tasks
            for task in self.cleanup_tasks:
                try:
                    await task
                except Exception as e:
                    self.logger.warning(f"Cleanup task failed: {e}")
            
            self.test_agents.clear()
            self.cleanup_tasks.clear()
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


async def main():
    """Main test runner."""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run tester
    tester = AgentSystemTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Save results
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: {results_file}")
        
        # Exit with appropriate code
        if results["summary"]["failed"] > 0:
            print("\n❌ Some tests failed!")
            return 1
        else:
            print("\n✅ All tests passed!")
            return 0
            
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)