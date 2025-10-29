"""
Tests for Enhanced KML Parser Agent

Tests for agent functionality including task processing, database integration,
and messaging system integration.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Try to import agent components
try:
    from agents.parsers.kml_parser_agent import KMLParserAgent
    from agents.core.task_queue import Task
    from agents.core.messaging import AgentMessage
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


@pytest.fixture
def sample_kml_file():
    """Create a temporary KML file for testing."""
    kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Test Document</name>
    <Placemark>
      <name>Test Site</name>
      <description>A test location</description>
      <Point>
        <coordinates>-74.0060,40.7128,0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.kml', delete=False) as f:
        f.write(kml_content)
        temp_file = f.name
    
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    csv_content = """name,latitude,longitude,description,culture
Test Site 1,40.7128,-74.0060,A test site in NYC,Modern
Test Site 2,51.5074,-0.1278,A test site in London,Modern"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_file = f.name
    
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def mock_database_session():
    """Mock database session."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None
    return mock_session


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
class TestKMLParserAgentBasic:
    """Basic tests for KML Parser Agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = KMLParserAgent(agent_id="test-parser-1")
        
        assert agent.agent_id == "test-parser-1"
        assert agent.agent_type == "parser"
        assert hasattr(agent, 'kml_processor')
        assert hasattr(agent, 'geojson_processor')
        assert hasattr(agent, 'csv_processor')
        assert hasattr(agent, 'geometry_validator')
        assert hasattr(agent, 'quality_checker')
    
    def test_get_capabilities(self):
        """Test agent capabilities reporting."""
        agent = KMLParserAgent()
        capabilities = agent._get_capabilities()
        
        expected_capabilities = [
            "enhanced_parser",
            "kml_parser", 
            "geojson_parser",
            "csv_parser",
            "batch_processing",
            "database_integration",
            "quality_assessment"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    def test_detect_file_format(self):
        """Test file format detection."""
        agent = KMLParserAgent()
        
        assert agent._detect_file_format(Path("test.kml")) == "kml"
        assert agent._detect_file_format(Path("test.kmz")) == "kmz"
        assert agent._detect_file_format(Path("test.geojson")) == "geojson"
        assert agent._detect_file_format(Path("test.json")) == "geojson"
        assert agent._detect_file_format(Path("test.csv")) == "csv"
        assert agent._detect_file_format(Path("test.txt")) is None
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self):
        """Test metrics collection."""
        agent = KMLParserAgent()
        
        # Set some test values
        agent.files_processed = 5
        agent.features_extracted = 25
        agent.database_inserts = 20
        agent.processing_errors = 1
        
        metrics = await agent.collect_metrics()
        
        assert metrics["files_processed"] == 5
        assert metrics["features_extracted"] == 25
        assert metrics["database_inserts"] == 20
        assert metrics["processing_errors"] == 1
        assert metrics["avg_features_per_file"] == 5.0
        assert metrics["error_rate"] == 1/6  # 1 error out of 6 total operations


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
class TestKMLParserAgentFileProcessing:
    """Test file processing functionality."""
    
    @pytest.mark.asyncio
    async def test_parse_kml_file_success(self, sample_kml_file):
        """Test successful KML file parsing."""
        agent = KMLParserAgent()
        
        result = await agent.parse_file(
            sample_kml_file,
            store_in_database=False,
            generate_quality_report=True
        )
        
        assert result["success"]
        assert result["file_format"] == "kml"
        assert result["features_count"] > 0
        assert "features" in result
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_parse_csv_file_success(self, sample_csv_file):
        """Test successful CSV file parsing."""
        agent = KMLParserAgent()
        
        result = await agent.parse_file(
            sample_csv_file,
            store_in_database=False,
            generate_quality_report=True
        )
        
        assert result["success"]
        assert result["file_format"] == "csv" 
        assert result["features_count"] == 2
        assert "quality_report" in result
    
    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self):
        """Test parsing non-existent file."""
        agent = KMLParserAgent()
        
        result = await agent.parse_file("/nonexistent/file.kml")
        
        assert not result["success"]
        assert "not found" in result["errors"][0].lower()
    
    @pytest.mark.asyncio
    async def test_parse_batch_files(self, sample_kml_file, sample_csv_file):
        """Test batch file parsing."""
        agent = KMLParserAgent()
        
        file_paths = [sample_kml_file, sample_csv_file]
        result = await agent.parse_batch(
            file_paths,
            store_in_database=False
        )
        
        assert result["success"]
        assert result["batch_size"] == 2
        assert result["total_features"] > 0
        assert result["total_errors"] == 0
        assert len(result["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_parse_batch_with_failures(self, sample_kml_file):
        """Test batch parsing with some failures."""
        agent = KMLParserAgent()
        
        file_paths = [sample_kml_file, "/nonexistent/file.csv"]
        result = await agent.parse_batch(file_paths, store_in_database=False)
        
        assert result["success"]
        assert result["batch_size"] == 2
        assert result["total_errors"] == 1
        assert result["files_processed"] == 1


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
class TestKMLParserAgentDatabaseIntegration:
    """Test database integration functionality."""
    
    @pytest.mark.asyncio
    async def test_store_features_in_database_success(self, mock_database_session):
        """Test successful feature storage in database."""
        agent = KMLParserAgent()
        agent.db_session_factory = lambda: mock_database_session
        
        features = [
            {
                "name": "Test Site",
                "description": "A test site",
                "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
                "properties": {"culture": "Test", "significance": "high"}
            }
        ]
        
        with patch('agents.parsers.kml_parser_agent.Dataset') as mock_dataset, \
             patch('agents.parsers.kml_parser_agent.GeospatialFeature') as mock_feature:
            
            mock_dataset_instance = MagicMock()
            mock_dataset_instance.id = "test-dataset-id"
            mock_dataset.return_value = mock_dataset_instance
            
            result = await agent.store_features_in_database(features, "Test Dataset")
            
            assert result["database_stored"]
            assert result["dataset_id"] == "test-dataset-id"
            assert result["features_stored"] == 1
    
    @pytest.mark.asyncio
    async def test_store_features_no_database(self):
        """Test feature storage when database is not available."""
        agent = KMLParserAgent()
        agent.db_session_factory = None
        
        features = [{"name": "Test", "geometry": {"type": "Point", "coordinates": [0, 0]}}]
        result = await agent.store_features_in_database(features, "Test Dataset")
        
        assert not result["database_stored"]
        assert "not available" in result["error"]
    
    @pytest.mark.asyncio
    async def test_validate_dataset(self, mock_database_session):
        """Test dataset validation functionality."""
        agent = KMLParserAgent()
        agent.db_session_factory = lambda: mock_database_session
        
        # Mock dataset and features
        mock_dataset = MagicMock()
        mock_dataset.name = "Test Dataset"
        mock_database_session.query.return_value.filter_by.return_value.first.return_value = mock_dataset
        mock_database_session.query.return_value.filter_by.return_value.all.return_value = []
        
        with patch.object(agent.quality_checker, 'check_dataset_quality') as mock_quality_check:
            mock_quality_report = MagicMock()
            mock_quality_report.metrics.overall_quality_score = 85.0
            mock_quality_report.issues = []
            mock_quality_check.return_value = mock_quality_report
            
            result = await agent.validate_dataset("test-dataset-id")
            
            assert result["success"]
            assert result["dataset_id"] == "test-dataset-id"
            assert "quality_report" in result


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
class TestKMLParserAgentTaskProcessing:
    """Test task processing functionality."""
    
    @pytest.mark.asyncio
    async def test_handle_parse_file_task(self, sample_kml_file):
        """Test handling parse file task."""
        agent = KMLParserAgent()
        
        task = Task(
            task_id="test-task-1",
            task_type="parse_file",
            parameters={"file_path": sample_kml_file, "store_in_db": False},
            input_data={}
        )
        
        with patch.object(agent, 'task_queue') as mock_queue:
            await agent.handle_task(task)
            
            # Check that task completion was called
            mock_queue.complete_task.assert_called_once()
            args = mock_queue.complete_task.call_args[0]
            assert args[0] == "test-task-1"  # task_id
            assert args[2] == agent.agent_id  # agent_id
            
            result = args[1]  # result
            assert result["success"]
    
    @pytest.mark.asyncio
    async def test_handle_parse_batch_task(self, sample_kml_file, sample_csv_file):
        """Test handling batch parse task."""
        agent = KMLParserAgent()
        
        task = Task(
            task_id="test-batch-1",
            task_type="parse_batch",
            parameters={
                "file_paths": [sample_kml_file, sample_csv_file],
                "store_in_db": False
            },
            input_data={}
        )
        
        with patch.object(agent, 'task_queue') as mock_queue:
            await agent.handle_task(task)
            
            mock_queue.complete_task.assert_called_once()
            result = mock_queue.complete_task.call_args[0][1]
            assert result["success"]
            assert result["batch_size"] == 2
    
    @pytest.mark.asyncio
    async def test_handle_invalid_task_type(self):
        """Test handling invalid task type."""
        agent = KMLParserAgent()
        
        task = Task(
            task_id="test-invalid",
            task_type="invalid_task_type",
            parameters={},
            input_data={}
        )
        
        with patch.object(agent, 'task_queue') as mock_queue:
            await agent.handle_task(task)
            
            # Should call fail_task due to invalid task type
            mock_queue.fail_task.assert_called_once()
            args = mock_queue.fail_task.call_args[0]
            assert args[0] == "test-invalid"
            assert "unknown" in args[1].lower()  # error message


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
class TestKMLParserAgentMessaging:
    """Test messaging functionality."""
    
    @pytest.mark.asyncio
    async def test_handle_parse_request_message(self, sample_kml_file):
        """Test handling parse request via messaging."""
        agent = KMLParserAgent()
        agent.nats_client = AsyncMock()
        
        # Create test message
        message = AgentMessage.create(
            sender_id="test-sender",
            receiver_id=agent.agent_id,
            message_type="parse_request",
            payload={"file_path": sample_kml_file, "store_in_db": False}
        )
        message.reply_to = "test.reply"
        
        await agent._handle_parse_request(message)
        
        # Check that response was published
        agent.nats_client.publish.assert_called_once()
        args = agent.nats_client.publish.call_args[0]
        assert args[0] == "test.reply"  # reply subject
        
        response_message = args[1]  # response message
        assert response_message.message_type == "parse_response"
        assert response_message.payload["success"]
    
    @pytest.mark.asyncio
    async def test_handle_file_upload_message(self, sample_kml_file):
        """Test handling file upload notification."""
        agent = KMLParserAgent()
        agent.nats_client = AsyncMock()
        
        message = AgentMessage.create(
            sender_id="upload-service",
            message_type="file_uploaded",
            payload={
                "file_path": sample_kml_file,
                "file_type": "kml"
            }
        )
        
        await agent._handle_file_upload(message)
        
        # Check that parsing result was published
        agent.nats_client.publish.assert_called_once()
        args = agent.nats_client.publish.call_args[0]
        assert args[0] == "agents.parsers.results"
        
        result_message = args[1]
        assert result_message.message_type == "file_parsed"
        assert result_message.payload["auto_parsed"]
    
    @pytest.mark.asyncio
    async def test_handle_batch_request_message(self, sample_kml_file):
        """Test handling batch processing request."""
        agent = KMLParserAgent()
        agent.nats_client = AsyncMock()
        
        message = AgentMessage.create(
            sender_id="batch-service",
            message_type="batch_request", 
            payload={"file_paths": [sample_kml_file], "store_in_db": False}
        )
        message.reply_to = "batch.reply"
        
        await agent._handle_batch_request(message)
        
        # Check response
        agent.nats_client.publish.assert_called_once()
        args = agent.nats_client.publish.call_args[0]
        response_message = args[1]
        assert response_message.message_type == "batch_response"
        assert response_message.payload["success"]


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
class TestKMLParserAgentUtilities:
    """Test utility functions."""
    
    def test_is_sacred_site_detection(self):
        """Test sacred site detection heuristics."""
        agent = KMLParserAgent()
        
        # Test positive cases
        temple_feature = {
            "name": "Ancient Temple",
            "description": "A sacred temple complex",
            "properties": {"site_type": "religious"}
        }
        assert agent._is_sacred_site(temple_feature)
        
        church_feature = {
            "name": "Cathedral of Notre Dame",
            "description": "Historic church",
            "properties": {}
        }
        assert agent._is_sacred_site(church_feature)
        
        # Test negative cases
        office_feature = {
            "name": "Office Building",
            "description": "Modern office complex",
            "properties": {"type": "commercial"}
        }
        assert not agent._is_sacred_site(office_feature)
    
    def test_determine_site_type(self):
        """Test site type determination."""
        agent = KMLParserAgent()
        
        temple_feature = {"name": "Hindu Temple", "description": "Ancient temple"}
        assert agent._determine_site_type(temple_feature) == "temple"
        
        monument_feature = {"name": "War Memorial", "description": "Monument to fallen soldiers"}
        assert agent._determine_site_type(monument_feature) == "monument"
        
        burial_feature = {"name": "Ancient Tomb", "description": "Burial site of ancient king"}
        assert agent._determine_site_type(burial_feature) == "burial_ground"
        
        unknown_feature = {"name": "Mystery Site", "description": "Unknown purpose"}
        assert agent._determine_site_type(unknown_feature) == "historical"
    
    def test_geometry_to_wkt_conversion(self):
        """Test geometry to WKT conversion."""
        agent = KMLParserAgent()
        
        # Test Point
        point_geom = {"type": "Point", "coordinates": [-74.006, 40.7128]}
        wkt = agent._geometry_to_wkt(point_geom)
        assert wkt == "POINT(-74.006 40.7128)"
        
        # Test empty geometry
        empty_wkt = agent._geometry_to_wkt(None)
        assert empty_wkt == "POINT EMPTY"
        
        # Test invalid geometry
        invalid_geom = {"type": "Point", "coordinates": []}
        invalid_wkt = agent._geometry_to_wkt(invalid_geom)
        assert invalid_wkt == "POINT EMPTY"


# Performance and load testing helpers

@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
@pytest.mark.slow
class TestKMLParserAgentPerformance:
    """Performance tests for the agent."""
    
    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self, sample_kml_file, sample_csv_file):
        """Test concurrent processing of multiple files."""
        agent = KMLParserAgent()
        
        # Create multiple parsing tasks
        tasks = []
        for i in range(5):
            file_path = sample_kml_file if i % 2 == 0 else sample_csv_file
            task = agent.parse_file(file_path, store_in_database=False)
            tasks.append(task)
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all tasks completed successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successful_results) == 5
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_dataset(self):
        """Test memory efficiency with large dataset."""
        # This would test memory usage patterns with large files
        # In practice, this would create/use large sample files
        pass


# Integration test with full agent lifecycle

@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent not available")
@pytest.mark.integration
class TestKMLParserAgentIntegration:
    """Full integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, sample_kml_file):
        """Test complete agent startup, processing, and shutdown."""
        agent = KMLParserAgent(agent_id="integration-test-agent")
        
        try:
            # Initialize agent (without actually starting NATS/messaging)
            await agent.agent_initialize()
            
            # Process a file
            result = await agent.parse_file(sample_kml_file, store_in_database=False)
            assert result["success"]
            
            # Check metrics
            metrics = await agent.collect_metrics()
            assert metrics["files_processed"] == 1
            
        finally:
            # Cleanup would happen here in a real scenario
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])