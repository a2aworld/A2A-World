"""
Unit Tests for Enhanced Data Processors

Tests for KMLProcessor, GeoJSONProcessor, CSVProcessor, GeometryValidator, and QualityChecker.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the processors
try:
    from agents.parsers.data_processors import (
        KMLProcessor, GeoJSONProcessor, CSVProcessor,
        GeometryValidator, QualityChecker, ValidationResult
    )
    PROCESSORS_AVAILABLE = True
except ImportError:
    PROCESSORS_AVAILABLE = False


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Data processors not available")
class TestGeometryValidator:
    """Test cases for GeometryValidator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = GeometryValidator()
    
    def test_validate_valid_point(self):
        """Test validation of valid point geometry."""
        geometry = {
            "type": "Point",
            "coordinates": [-74.006, 40.7128]  # New York
        }
        
        result = self.validator.validate_geometry(geometry)
        
        assert result.is_valid
        assert len(result.issues) == 0
        assert result.geometry == geometry
    
    def test_validate_invalid_coordinates_out_of_range(self):
        """Test validation of point with coordinates out of range."""
        geometry = {
            "type": "Point",
            "coordinates": [-200, 95]  # Invalid longitude and latitude
        }
        
        result = self.validator.validate_geometry(geometry)
        
        assert not result.is_valid
        assert len(result.issues) >= 2  # Both longitude and latitude issues
        assert any("longitude" in issue.lower() for issue in result.issues)
        assert any("latitude" in issue.lower() for issue in result.issues)
    
    def test_validate_missing_coordinates(self):
        """Test validation of geometry without coordinates."""
        geometry = {
            "type": "Point"
        }
        
        result = self.validator.validate_geometry(geometry)
        
        assert not result.is_valid
        assert any("coordinates" in issue.lower() for issue in result.issues)
    
    def test_validate_insufficient_coordinates(self):
        """Test validation of point with insufficient coordinates."""
        geometry = {
            "type": "Point",
            "coordinates": [40.7128]  # Missing longitude
        }
        
        result = self.validator.validate_geometry(geometry)
        
        assert not result.is_valid
        assert any("2 coordinates" in issue for issue in result.issues)
    
    def test_validate_linestring_valid(self):
        """Test validation of valid LineString."""
        geometry = {
            "type": "LineString",
            "coordinates": [[-74.006, 40.7128], [-73.9857, 40.7489]]
        }
        
        result = self.validator.validate_geometry(geometry)
        
        assert result.is_valid
        assert len(result.issues) == 0
    
    def test_validate_linestring_insufficient_points(self):
        """Test validation of LineString with insufficient points."""
        geometry = {
            "type": "LineString",
            "coordinates": [[-74.006, 40.7128]]  # Need at least 2 points
        }
        
        result = self.validator.validate_geometry(geometry)
        
        assert not result.is_valid
        assert any("at least 2" in issue for issue in result.issues)
    
    def test_calculate_bounds_point(self):
        """Test bounding box calculation for point."""
        geometry = {
            "type": "Point",
            "coordinates": [-74.006, 40.7128]
        }
        
        bounds = self.validator.calculate_bounds(geometry)
        
        assert bounds is not None
        assert bounds == (-74.006, 40.7128, -74.006, 40.7128)
    
    def test_calculate_bounds_invalid_geometry(self):
        """Test bounding box calculation for invalid geometry."""
        geometry = {"type": "Point"}  # Missing coordinates
        
        bounds = self.validator.calculate_bounds(geometry)
        
        assert bounds is None


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Data processors not available")
class TestQualityChecker:
    """Test cases for QualityChecker."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.quality_checker = QualityChecker()
    
    def test_check_quality_valid_dataset(self):
        """Test quality check on valid dataset."""
        data = [
            {
                "name": "Test Site 1",
                "description": "A test sacred site with good data quality",
                "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
                "properties": {"culture": "Test", "significance": "high"}
            },
            {
                "name": "Test Site 2", 
                "description": "Another test site with complete information",
                "geometry": {"type": "Point", "coordinates": [-73.9857, 40.7489]},
                "properties": {"culture": "Test", "significance": "medium"}
            }
        ]
        
        report = self.quality_checker.check_dataset_quality(data, "Test Dataset")
        
        assert report.dataset_name == "Test Dataset"
        assert report.metrics.total_records == 2
        assert report.metrics.valid_geometries == 2
        assert report.metrics.invalid_geometries == 0
        assert report.metrics.overall_quality_score > 80  # Should be high quality
    
    def test_check_quality_dataset_with_issues(self):
        """Test quality check on dataset with various issues."""
        data = [
            {
                "name": "Good Site",
                "description": "Complete site data",
                "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
                "properties": {"culture": "Test"}
            },
            {
                "name": "",  # Missing name
                "description": "",  # Missing description
                "geometry": {"type": "Point", "coordinates": [0, 0]},  # Suspicious coordinates
                "properties": {}
            },
            {
                "name": "Invalid Coords Site",
                "description": "Site with invalid coordinates",
                "geometry": {"type": "Point", "coordinates": [-200, 95]},  # Invalid
                "properties": {}
            },
            {
                # Missing geometry entirely
                "name": "No Geometry Site",
                "description": "Site without geometry",
                "properties": {}
            }
        ]
        
        report = self.quality_checker.check_dataset_quality(data, "Problem Dataset")
        
        assert report.dataset_name == "Problem Dataset"
        assert report.metrics.total_records == 4
        assert report.metrics.invalid_geometries > 0
        assert report.metrics.missing_names > 0
        assert report.metrics.missing_descriptions > 0
        assert report.metrics.overall_quality_score < 70  # Should be lower quality
        assert len(report.issues) > 0
        assert len(report.recommendations) > 0
    
    def test_check_quality_duplicate_detection(self):
        """Test duplicate record detection."""
        data = [
            {
                "name": "Duplicate Site",
                "description": "First instance",
                "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
                "properties": {}
            },
            {
                "name": "Duplicate Site",  # Same name and coordinates
                "description": "Second instance", 
                "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
                "properties": {}
            }
        ]
        
        report = self.quality_checker.check_dataset_quality(data, "Duplicate Test")
        
        assert report.metrics.duplicate_records > 0
        assert any("duplicate" in issue.description.lower() for issue in report.issues)


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Data processors not available")
class TestGeoJSONProcessor:
    """Test cases for GeoJSONProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = GeoJSONProcessor()
    
    def test_process_valid_geojson_string(self):
        """Test processing valid GeoJSON string."""
        geojson_string = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": "Test Site",
                        "description": "A test location"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-74.006, 40.7128]
                    }
                }
            ]
        })
        
        result = self.processor.process_geojson_string(geojson_string)
        
        assert result.success
        assert len(result.features) == 1
        assert result.features[0]["name"] == "Test Site"
        assert result.features[0]["geometry"]["type"] == "Point"
        assert len(result.errors) == 0
    
    def test_process_invalid_json(self):
        """Test processing invalid JSON string."""
        invalid_json = '{"type": "FeatureCollection", "features": [{'
        
        result = self.processor.process_geojson_string(invalid_json)
        
        assert not result.success
        assert len(result.errors) > 0
        assert any("json" in error.lower() for error in result.errors)
    
    def test_process_feature_collection(self):
        """Test processing FeatureCollection."""
        geojson_data = {
            "type": "FeatureCollection",
            "name": "Test Collection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Site 1"},
                    "geometry": {"type": "Point", "coordinates": [0, 0]}
                },
                {
                    "type": "Feature", 
                    "properties": {"name": "Site 2"},
                    "geometry": {"type": "Point", "coordinates": [1, 1]}
                }
            ]
        }
        
        result = self.processor._process_geojson_data(geojson_data)
        
        assert result.success
        assert len(result.features) == 2
        assert result.metadata["geojson_type"] == "FeatureCollection"
        assert result.metadata["feature_count"] == 2
    
    def test_validate_geojson_structure_valid(self):
        """Test validation of valid GeoJSON structure."""
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Test"},
                    "geometry": {"type": "Point", "coordinates": [0, 0]}
                }
            ]
        }
        
        validation = self.processor.validate_geojson_structure(geojson_data)
        
        assert validation["valid"]
        assert validation["type"] == "FeatureCollection"
        assert len(validation["issues"]) == 0
    
    def test_validate_geojson_structure_invalid(self):
        """Test validation of invalid GeoJSON structure."""
        invalid_geojson = {
            "type": "FeatureCollection",
            "features": "not_an_array"  # Should be array
        }
        
        validation = self.processor.validate_geojson_structure(invalid_geojson)
        
        assert not validation["valid"]
        assert len(validation["issues"]) > 0


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Data processors not available")
class TestCSVProcessor:
    """Test cases for CSVProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = CSVProcessor()
    
    def test_auto_detect_columns(self):
        """Test automatic column detection."""
        headers = ["site_name", "lat", "lon", "description", "culture"]
        
        mapping = self.processor._auto_detect_columns(headers)
        
        assert mapping.latitude == "lat"
        assert mapping.longitude == "lon"
        assert mapping.name == "site_name"
        assert mapping.description == "description"
        assert "culture" in mapping.properties
    
    def test_parse_coordinates_decimal(self):
        """Test parsing decimal coordinate formats."""
        # Test valid decimal coordinates
        assert self.processor._parse_coordinate("40.7128", "latitude") == 40.7128
        assert self.processor._parse_coordinate("-74.0060", "longitude") == -74.0060
        
        # Test with cardinal directions
        assert self.processor._parse_coordinate("40.7128N", "latitude") == 40.7128
        assert self.processor._parse_coordinate("74.0060W", "longitude") == -74.0060
        assert self.processor._parse_coordinate("40.7128S", "latitude") == -40.7128
        assert self.processor._parse_coordinate("74.0060E", "longitude") == 74.0060
    
    def test_parse_coordinates_invalid(self):
        """Test parsing invalid coordinate formats."""
        assert self.processor._parse_coordinate("not-a-number", "latitude") is None
        assert self.processor._parse_coordinate("", "longitude") is None
        assert self.processor._parse_coordinate("95.0", "latitude") is None  # Out of range handled elsewhere
    
    def test_preview_file_functionality(self):
        """Test CSV file preview functionality."""
        # Create temporary CSV file
        csv_content = """name,latitude,longitude,description
Test Site 1,40.7128,-74.0060,A test location in NYC
Test Site 2,51.5074,-0.1278,A test location in London
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            preview = self.processor.preview_file(temp_file)
            
            assert 'error' not in preview
            assert preview['headers'] == ['name', 'latitude', 'longitude', 'description']
            assert len(preview['sample_data']) == 2
            assert preview['detected_mapping']['latitude'] == 'latitude'
            assert preview['detected_mapping']['longitude'] == 'longitude'
            assert preview['detected_mapping']['name'] == 'name'
            assert preview['total_rows'] == 2
        
        finally:
            os.unlink(temp_file)


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Data processors not available")
class TestKMLProcessor:
    """Test cases for KMLProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = KMLProcessor()
    
    def test_process_basic_kml_content(self):
        """Test processing basic KML content."""
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
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
</kml>'''
        
        result = self.processor._process_kml_content_basic(kml_content)
        
        assert result.success
        assert len(result.features) == 1
        assert result.features[0]["name"] == "Test Site"
        assert result.features[0]["geometry"]["type"] == "Point"
        assert result.metadata["library"] == "xml.etree"
    
    def test_process_malformed_kml(self):
        """Test processing malformed KML content."""
        malformed_kml = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Test Document</name>
    <Placemark>
      <name>Test Site</name>
      <Point>
        <coordinates>invalid-coordinates</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>'''
        
        result = self.processor._process_kml_content_basic(malformed_kml)
        
        # Should still succeed but may have warnings
        assert result.success
        # The invalid coordinates should be handled gracefully


@pytest.fixture
def sample_data_files():
    """Fixture providing paths to sample data files."""
    return {
        'kml': 'sample_data/sacred_sites.kml',
        'geojson': 'sample_data/cultural_landmarks.geojson', 
        'csv': 'sample_data/ancient_sites.csv',
        'invalid_kml': 'sample_data/malformed_test.kml',
        'invalid_csv': 'sample_data/invalid_coordinates.csv',
        'invalid_geojson': 'sample_data/empty_features.geojson'
    }


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Data processors not available")
class TestIntegrationWithSampleFiles:
    """Integration tests using sample data files."""
    
    def test_process_sample_kml_file(self, sample_data_files):
        """Test processing the sample KML file."""
        if not Path(sample_data_files['kml']).exists():
            pytest.skip("Sample KML file not found")
        
        processor = KMLProcessor()
        result = processor.process_file(sample_data_files['kml'])
        
        assert result.success
        assert len(result.features) > 0
        assert all('name' in feature for feature in result.features)
        assert all('geometry' in feature for feature in result.features)
    
    def test_process_sample_geojson_file(self, sample_data_files):
        """Test processing the sample GeoJSON file."""
        if not Path(sample_data_files['geojson']).exists():
            pytest.skip("Sample GeoJSON file not found")
        
        processor = GeoJSONProcessor()
        result = processor.process_file(sample_data_files['geojson'])
        
        assert result.success
        assert len(result.features) > 0
        assert result.quality_report is not None
        assert result.quality_report.metrics.overall_quality_score > 70
    
    def test_process_sample_csv_file(self, sample_data_files):
        """Test processing the sample CSV file."""
        if not Path(sample_data_files['csv']).exists():
            pytest.skip("Sample CSV file not found")
        
        processor = CSVProcessor()
        result = processor.process_file(sample_data_files['csv'])
        
        assert result.success
        assert len(result.features) > 0
        assert all('latitude' in feature and 'longitude' in feature for feature in result.features)
    
    def test_process_invalid_files_error_handling(self, sample_data_files):
        """Test error handling with intentionally invalid files."""
        processors = [
            (KMLProcessor(), sample_data_files['invalid_kml']),
            (CSVProcessor(), sample_data_files['invalid_csv']),
            (GeoJSONProcessor(), sample_data_files['invalid_geojson'])
        ]
        
        for processor, file_path in processors:
            if not Path(file_path).exists():
                continue
                
            result = processor.process_file(file_path)
            
            # Files may succeed but should have warnings/issues
            if result.success:
                assert len(result.warnings) > 0 or (
                    result.quality_report and len(result.quality_report.issues) > 0
                )


if __name__ == "__main__":
    pytest.main([__file__])