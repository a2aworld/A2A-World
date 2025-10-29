"""
Integration Tests for Enhanced Data API Endpoints

Tests for file upload, processing status, dataset management, and validation endpoints.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

# Try to import API components
try:
    from api.app.main import app
    from api.app.api.api_v1.endpoints.data import upload_progress, processing_tasks
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.fixture
def client():
    """Create test client."""
    if not API_AVAILABLE:
        pytest.skip("API not available")
    return TestClient(app)


@pytest.fixture
def sample_kml_content():
    """Sample KML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Test Sacred Sites</name>
    <Placemark>
      <name>Test Site 1</name>
      <description>A test sacred site</description>
      <Point>
        <coordinates>-1.826215,51.178882,0</coordinates>
      </Point>
      <ExtendedData>
        <Data name="culture">
          <value>Test Culture</value>
        </Data>
        <Data name="site_type">
          <value>test_site</value>
        </Data>
      </ExtendedData>
    </Placemark>
    <Placemark>
      <name>Test Site 2</name>
      <description>Another test sacred site</description>
      <Point>
        <coordinates>31.134202,29.979175,0</coordinates>
      </Point>
      <ExtendedData>
        <Data name="culture">
          <value>Test Culture</value>
        </Data>
        <Data name="site_type">
          <value>test_site</value>
        </Data>
      </ExtendedData>
    </Placemark>
  </Document>
</kml>"""


@pytest.fixture
def sample_geojson_content():
    """Sample GeoJSON content for testing."""
    return {
        "type": "FeatureCollection",
        "name": "Test Cultural Sites",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Test Monument",
                    "description": "A test cultural monument",
                    "culture": "Test Culture",
                    "site_type": "monument"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [2.294481, 48.858370]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Test Temple",
                    "description": "A test temple site",
                    "culture": "Test Culture",
                    "site_type": "temple"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [12.492269, 41.890251]
                }
            }
        ]
    }


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """name,latitude,longitude,culture,time_period,site_type,description
Test Site 1,40.7128,-74.0060,Test Culture,Modern,test_site,A test site in New York
Test Site 2,51.5074,-0.1278,Test Culture,Modern,test_site,A test site in London
Test Site 3,48.8566,2.3522,Test Culture,Modern,test_site,A test site in Paris"""


@pytest.fixture
def invalid_kml_content():
    """Invalid KML content for error testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Invalid Test File</name>
    <Placemark>
      <name>Invalid Coordinates</name>
      <Point>
        <coordinates>-200,95,0</coordinates>
      </Point>
    </Placemark>
  <!-- Missing closing Document and kml tags"""


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestDataUploadEndpoint:
    """Test cases for data upload endpoint."""
    
    def test_upload_valid_kml_file(self, client, sample_kml_content):
        """Test uploading a valid KML file."""
        # Clear any existing progress data
        upload_progress.clear()
        
        with patch('agents.parsers.data_processors.KMLProcessor') as mock_processor:
            # Mock successful processing
            mock_instance = MagicMock()
            mock_instance.process_file.return_value = MagicMock(
                success=True,
                features=[{"name": "Test Site", "geometry": {"type": "Point", "coordinates": [0, 0]}}],
                errors=[],
                warnings=[],
                quality_report=None
            )
            mock_processor.return_value = mock_instance
            
            files = {"file": ("test.kml", sample_kml_content, "application/vnd.google-earth.kml+xml")}
            response = client.post("/api/v1/data/upload", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "upload_id" in data
            assert data["filename"] == "test.kml"
            assert data["file_type"] == "kml"
            assert data["status"] == "queued"
    
    def test_upload_valid_geojson_file(self, client, sample_geojson_content):
        """Test uploading a valid GeoJSON file."""
        upload_progress.clear()
        
        geojson_str = json.dumps(sample_geojson_content)
        files = {"file": ("test.geojson", geojson_str, "application/geo+json")}
        
        response = client.post("/api/v1/data/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["file_type"] == "geojson"
        assert data["status"] == "queued"
    
    def test_upload_valid_csv_file(self, client, sample_csv_content):
        """Test uploading a valid CSV file."""
        upload_progress.clear()
        
        files = {"file": ("test.csv", sample_csv_content, "text/csv")}
        response = client.post("/api/v1/data/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["file_type"] == "csv"
        assert data["status"] == "queued"
    
    def test_upload_unsupported_file_type(self, client):
        """Test uploading an unsupported file type."""
        files = {"file": ("test.txt", "This is a text file", "text/plain")}
        response = client.post("/api/v1/data/upload", files=files)
        
        assert response.status_code == 400
        assert "unsupported file format" in response.json()["detail"].lower()
    
    def test_upload_oversized_file(self, client):
        """Test uploading a file that exceeds size limit."""
        # Create a large content string (>100MB)
        large_content = "x" * (101 * 1024 * 1024)  # 101MB
        files = {"file": ("large.kml", large_content, "application/vnd.google-earth.kml+xml")}
        
        response = client.post("/api/v1/data/upload", files=files)
        
        assert response.status_code == 400
        assert "file too large" in response.json()["detail"].lower()
    
    def test_upload_no_filename(self, client):
        """Test uploading without filename."""
        files = {"file": ("", "content", "application/vnd.google-earth.kml+xml")}
        response = client.post("/api/v1/data/upload", files=files)
        
        assert response.status_code == 400
        assert "filename" in response.json()["detail"].lower()


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestUploadStatusEndpoint:
    """Test cases for upload status endpoint."""
    
    def test_get_upload_status_existing(self, client):
        """Test getting status for existing upload."""
        upload_id = "test-upload-123"
        upload_progress[upload_id] = {
            "status": "processing",
            "progress": {"stage": "parsing", "progress": 50},
            "filename": "test.kml",
            "file_size": 1024
        }
        
        response = client.get(f"/api/v1/data/upload/{upload_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["upload_id"] == upload_id
        assert data["status"] == "processing"
        assert data["progress"]["stage"] == "parsing"
        assert data["progress"]["progress"] == 50
    
    def test_get_upload_status_completed(self, client):
        """Test getting status for completed upload."""
        upload_id = "test-upload-completed"
        upload_progress[upload_id] = {
            "status": "completed",
            "progress": {"stage": "completed", "progress": 100},
            "result": {
                "success": True,
                "features_count": 5,
                "quality_score": 85.0
            }
        }
        
        response = client.get(f"/api/v1/data/upload/{upload_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "completed"
        assert data["result"]["success"]
        assert data["result"]["features_count"] == 5
    
    def test_get_upload_status_not_found(self, client):
        """Test getting status for non-existent upload."""
        response = client.get("/api/v1/data/upload/nonexistent-id/status")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_upload_status_error(self, client):
        """Test getting status for failed upload."""
        upload_id = "test-upload-error"
        upload_progress[upload_id] = {
            "status": "error",
            "progress": {"stage": "error", "progress": 0},
            "error": "File parsing failed due to invalid format"
        }
        
        response = client.get(f"/api/v1/data/upload/{upload_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "error"
        assert data["error"] == "File parsing failed due to invalid format"


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available") 
class TestValidateFileEndpoint:
    """Test cases for file validation endpoint."""
    
    def test_validate_geojson_structure_valid(self, client, sample_geojson_content):
        """Test validation of valid GeoJSON structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            json.dump(sample_geojson_content, f)
            temp_file = f.name
        
        try:
            request_data = {"file_path": temp_file, "file_type": "geojson"}
            response = client.post("/api/v1/data/validate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["valid"]
            assert data["file_format"] == "geojson"
            assert "structure_validation" in data
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_geojson_structure_invalid(self, client):
        """Test validation of invalid GeoJSON structure."""
        invalid_geojson = {"type": "FeatureCollection", "features": "not_an_array"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            json.dump(invalid_geojson, f)
            temp_file = f.name
        
        try:
            request_data = {"file_path": temp_file, "file_type": "geojson"}
            response = client.post("/api/v1/data/validate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert not data["valid"]
            assert len(data["structure_validation"]["issues"]) > 0
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_csv_structure(self, client, sample_csv_content):
        """Test validation of CSV structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            temp_file = f.name
        
        try:
            request_data = {"file_path": temp_file, "file_type": "csv"}
            response = client.post("/api/v1/data/validate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["valid"]
            assert data["file_format"] == "csv"
            assert "preview" in data
            assert "detected_columns" in data["preview"]
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_kml_structure(self, client, sample_kml_content):
        """Test validation of KML structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kml', delete=False) as f:
            f.write(sample_kml_content)
            temp_file = f.name
        
        try:
            request_data = {"file_path": temp_file, "file_type": "kml"}
            response = client.post("/api/v1/data/validate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["valid"]
            assert data["file_format"] == "kml"
            assert "preview" in data
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_nonexistent_file(self, client):
        """Test validation of non-existent file."""
        request_data = {"file_path": "/nonexistent/file.kml", "file_type": "kml"}
        response = client.post("/api/v1/data/validate", json=request_data)
        
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestDatasetManagementEndpoints:
    """Test cases for dataset management endpoints."""
    
    @patch('api.app.api.api_v1.endpoints.data.get_database_session')
    def test_list_datasets(self, mock_db_session, client):
        """Test listing datasets."""
        # Mock database session and query results
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock dataset objects
        mock_dataset1 = MagicMock()
        mock_dataset1.id = "dataset-1"
        mock_dataset1.name = "Test Dataset 1"
        mock_dataset1.file_type = "kml"
        mock_dataset1.status = "completed"
        mock_dataset1.created_at = None
        mock_dataset1.updated_at = None
        mock_dataset1.metadata = {"features_count": 10}
        
        mock_dataset2 = MagicMock()
        mock_dataset2.id = "dataset-2" 
        mock_dataset2.name = "Test Dataset 2"
        mock_dataset2.file_type = "geojson"
        mock_dataset2.status = "completed"
        mock_dataset2.created_at = None
        mock_dataset2.updated_at = None
        mock_dataset2.metadata = {"features_count": 5}
        
        mock_query = MagicMock()
        mock_query.count.return_value = 2
        mock_query.offset.return_value.limit.return_value.all.return_value = [mock_dataset1, mock_dataset2]
        mock_session.query.return_value = mock_query
        
        response = client.get("/api/v1/data/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "datasets" in data
        assert data["total"] == 2
        assert len(data["datasets"]) == 2
        assert data["datasets"][0]["name"] == "Test Dataset 1"
        assert data["datasets"][1]["file_type"] == "geojson"
    
    @patch('api.app.api.api_v1.endpoints.data.get_database_session')
    def test_get_dataset_by_id(self, mock_db_session, client):
        """Test getting specific dataset by ID."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.id = "test-dataset-id"
        mock_dataset.name = "Test Dataset"
        mock_dataset.description = "A test dataset"
        mock_dataset.file_type = "kml"
        mock_dataset.status = "completed"
        mock_dataset.created_at = None
        mock_dataset.updated_at = None
        mock_dataset.metadata = {"features_count": 15}
        mock_dataset.processing_log = "Processing completed successfully"
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_dataset
        mock_session.query.return_value.filter.return_value.count.return_value = 15
        
        response = client.get("/api/v1/data/test-dataset-id")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test-dataset-id"
        assert data["name"] == "Test Dataset"
        assert data["file_type"] == "kml"
        assert data["feature_count"] == 15
    
    @patch('api.app.api.api_v1.endpoints.data.get_database_session')
    def test_get_dataset_not_found(self, mock_db_session, client):
        """Test getting non-existent dataset."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        response = client.get("/api/v1/data/nonexistent-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('api.app.api.api_v1.endpoints.data.get_database_session')
    def test_delete_dataset(self, mock_db_session, client):
        """Test deleting a dataset."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.name = "Test Dataset"
        mock_session.query.return_value.filter.return_value.first.return_value = mock_dataset
        mock_session.query.return_value.filter.return_value.count.return_value = 10
        
        response = client.delete("/api/v1/data/test-dataset-id")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"]
        assert "Test Dataset" in data["message"]
        assert data["features_deleted"] == 10


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestDataSummaryEndpoint:
    """Test cases for data summary statistics endpoint."""
    
    @patch('api.app.api.api_v1.endpoints.data.get_database_session')
    def test_get_data_summary(self, mock_db_session, client):
        """Test getting data summary statistics."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock database query results
        mock_session.query.return_value.count.side_effect = [25, 20, 3, 150]  # total, completed, failed, features
        mock_session.query.return_value.group_by.return_value.all.return_value = [
            ("kml", 15), ("geojson", 8), ("csv", 2)
        ]
        mock_session.query.return_value.filter.return_value.count.return_value = 5
        
        response = client.get("/api/v1/data/stats/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "datasets" in data
        assert "features" in data
        assert "file_types" in data
        assert "recent_activity" in data
        assert "processing_capabilities" in data
        
        assert data["datasets"]["total"] == 25
        assert data["datasets"]["completed"] == 20
        assert data["datasets"]["failed"] == 3
        assert data["features"]["total"] == 150
        assert data["file_types"]["kml"] == 15


# Additional test utilities and fixtures

@pytest.fixture
def mock_successful_processing():
    """Mock successful file processing."""
    return {
        "success": True,
        "features": [
            {
                "name": "Test Feature 1",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"culture": "Test"}
            }
        ],
        "metadata": {"processor": "test", "features_count": 1},
        "errors": [],
        "warnings": [],
        "quality_report": None
    }


@pytest.fixture 
def mock_failed_processing():
    """Mock failed file processing."""
    return {
        "success": False,
        "features": [],
        "metadata": {},
        "errors": ["File format is invalid", "Could not parse coordinates"],
        "warnings": []
    }


def test_error_handling_integration():
    """Test that error handling works across the system."""
    # This would be an integration test that tests error propagation
    # from processors through API endpoints to client responses
    pass


def test_concurrent_upload_handling():
    """Test handling of concurrent file uploads."""
    # This would test that multiple simultaneous uploads are handled correctly
    pass


def test_large_file_processing():
    """Test processing of large files with memory efficiency."""
    # This would test memory usage and performance with large datasets
    pass


if __name__ == "__main__":
    pytest.main([__file__])