"""
A2A World Platform - GeoJSON Processor

Advanced GeoJSON parsing with feature collection processing, validation, and error handling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .geometry_validator import GeometryValidator, ValidationResult
from .quality_checker import QualityChecker, QualityReport


@dataclass
class ProcessingResult:
    """Result of GeoJSON processing."""
    success: bool
    features: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    quality_report: Optional[QualityReport] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class GeoJSONProcessor:
    """
    Enhanced GeoJSON processor with comprehensive parsing capabilities.
    
    Features:
    - Handles GeoJSON Feature Collections and individual features
    - Validates GeoJSON structure and geometry
    - Extracts properties and metadata
    - Supports various geometry types
    - Data quality assessment and reporting
    - Error recovery and correction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.geometry_validator = GeometryValidator()
        self.quality_checker = QualityChecker()
        
    def process_file(
        self,
        file_path: str,
        validate_geometry: bool = True,
        generate_quality_report: bool = True
    ) -> ProcessingResult:
        """
        Process GeoJSON file.
        
        Args:
            file_path: Path to GeoJSON file
            validate_geometry: Whether to validate geometries
            generate_quality_report: Whether to generate quality report
            
        Returns:
            ProcessingResult with extracted data and analysis
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return ProcessingResult(
                    success=False,
                    features=[],
                    metadata={},
                    errors=[f"File not found: {file_path}"]
                )
            
            self.logger.info(f"Processing GeoJSON file: {file_path}")
            
            # Read and parse JSON
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                try:
                    geojson_data = json.load(f)
                except json.JSONDecodeError as e:
                    return ProcessingResult(
                        success=False,
                        features=[],
                        metadata={'file_path': str(file_path_obj)},
                        errors=[f"Invalid JSON format: {str(e)}"]
                    )
            
            # Process the GeoJSON data
            result = self._process_geojson_data(geojson_data)
            
            # Validate geometries if requested
            if validate_geometry and result.success:
                result = self._validate_features(result)
            
            # Generate quality report if requested
            if generate_quality_report and result.success:
                quality_report = self.quality_checker.check_dataset_quality(
                    result.features,
                    f"GeoJSON: {file_path_obj.name}"
                )
                result.quality_report = quality_report
            
            # Add file metadata
            result.metadata.update({
                'file_path': str(file_path_obj),
                'file_size': file_path_obj.stat().st_size,
                'processed_at': datetime.utcnow().isoformat(),
                'processor': 'GeoJSONProcessor'
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing GeoJSON file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                features=[],
                metadata={'file_path': file_path},
                errors=[f"Processing failed: {str(e)}"]
            )
    
    def process_geojson_string(
        self,
        geojson_string: str,
        validate_geometry: bool = True,
        generate_quality_report: bool = True
    ) -> ProcessingResult:
        """
        Process GeoJSON from string.
        
        Args:
            geojson_string: GeoJSON content as string
            validate_geometry: Whether to validate geometries
            generate_quality_report: Whether to generate quality report
            
        Returns:
            ProcessingResult with extracted data and analysis
        """
        try:
            # Parse JSON string
            try:
                geojson_data = json.loads(geojson_string)
            except json.JSONDecodeError as e:
                return ProcessingResult(
                    success=False,
                    features=[],
                    metadata={},
                    errors=[f"Invalid JSON format: {str(e)}"]
                )
            
            # Process the GeoJSON data
            result = self._process_geojson_data(geojson_data)
            
            # Validate geometries if requested
            if validate_geometry and result.success:
                result = self._validate_features(result)
            
            # Generate quality report if requested
            if generate_quality_report and result.success:
                quality_report = self.quality_checker.check_dataset_quality(
                    result.features,
                    "GeoJSON String"
                )
                result.quality_report = quality_report
            
            # Add metadata
            result.metadata.update({
                'source': 'string',
                'processed_at': datetime.utcnow().isoformat(),
                'processor': 'GeoJSONProcessor'
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing GeoJSON string: {e}")
            return ProcessingResult(
                success=False,
                features=[],
                metadata={},
                errors=[f"Processing failed: {str(e)}"]
            )
    
    def _process_geojson_data(self, geojson_data: Dict[str, Any]) -> ProcessingResult:
        """Process parsed GeoJSON data structure."""
        features = []
        metadata = {}
        warnings = []
        errors = []
        
        try:
            # Validate basic GeoJSON structure
            if not isinstance(geojson_data, dict):
                return ProcessingResult(
                    success=False,
                    features=[],
                    metadata={},
                    errors=["GeoJSON must be an object"]
                )
            
            geojson_type = geojson_data.get('type')
            if not geojson_type:
                return ProcessingResult(
                    success=False,
                    features=[],
                    metadata={},
                    errors=["GeoJSON must have a 'type' property"]
                )
            
            # Process based on GeoJSON type
            if geojson_type == 'FeatureCollection':
                result = self._process_feature_collection(geojson_data)
                features = result['features']
                metadata.update(result['metadata'])
                warnings.extend(result.get('warnings', []))
                errors.extend(result.get('errors', []))
                
            elif geojson_type == 'Feature':
                feature_data = self._process_feature(geojson_data)
                if feature_data:
                    features.append(feature_data)
                else:
                    errors.append("Failed to process feature")
                
            elif geojson_type in ['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon', 'GeometryCollection']:
                # Bare geometry - wrap in feature
                feature_data = {
                    'name': 'Geometry',
                    'description': f'Bare {geojson_type} geometry',
                    'geometry': geojson_data,
                    'properties': {}
                }
                features.append(feature_data)
                warnings.append(f"Converted bare {geojson_type} geometry to feature")
                
            else:
                return ProcessingResult(
                    success=False,
                    features=[],
                    metadata={},
                    errors=[f"Unsupported GeoJSON type: {geojson_type}"]
                )
            
            # Extract CRS information if present
            if 'crs' in geojson_data:
                metadata['crs'] = geojson_data['crs']
            
            # Extract bbox if present
            if 'bbox' in geojson_data:
                metadata['bbox'] = geojson_data['bbox']
            
            metadata.update({
                'geojson_type': geojson_type,
                'feature_count': len(features)
            })
            
            return ProcessingResult(
                success=True,
                features=features,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata={},
                errors=[f"GeoJSON data processing failed: {str(e)}"]
            )
    
    def _process_feature_collection(self, feature_collection: Dict[str, Any]) -> Dict[str, Any]:
        """Process GeoJSON FeatureCollection."""
        features = []
        metadata = {}
        warnings = []
        errors = []
        
        try:
            # Validate FeatureCollection structure
            geojson_features = feature_collection.get('features', [])
            if not isinstance(geojson_features, list):
                return {
                    'features': [],
                    'metadata': {},
                    'errors': ["FeatureCollection 'features' must be an array"]
                }
            
            # Extract collection-level properties
            if 'name' in feature_collection:
                metadata['collection_name'] = feature_collection['name']
            
            if 'description' in feature_collection:
                metadata['collection_description'] = feature_collection['description']
            
            # Process each feature
            for i, geojson_feature in enumerate(geojson_features):
                try:
                    feature_data = self._process_feature(geojson_feature, i)
                    if feature_data:
                        features.append(feature_data)
                    else:
                        warnings.append(f"Skipped invalid feature at index {i}")
                except Exception as e:
                    errors.append(f"Error processing feature {i}: {str(e)}")
            
            metadata.update({
                'total_features_in_collection': len(geojson_features),
                'successfully_processed': len(features)
            })
            
            return {
                'features': features,
                'metadata': metadata,
                'warnings': warnings,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'features': [],
                'metadata': {},
                'errors': [f"FeatureCollection processing failed: {str(e)}"]
            }
    
    def _process_feature(self, geojson_feature: Dict[str, Any], index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Process individual GeoJSON Feature."""
        try:
            # Validate feature structure
            if not isinstance(geojson_feature, dict):
                return None
            
            feature_type = geojson_feature.get('type')
            if feature_type != 'Feature':
                return None
            
            # Extract basic feature data
            feature_data = {
                'properties': geojson_feature.get('properties', {}),
                'geometry': geojson_feature.get('geometry'),
            }
            
            # Add index if provided
            if index is not None:
                feature_data['feature_index'] = index
            
            # Extract name from properties or use default
            properties = feature_data['properties']
            name = None
            
            # Try common name fields
            for name_field in ['name', 'Name', 'title', 'Title', 'label', 'Label']:
                if name_field in properties and properties[name_field]:
                    name = str(properties[name_field])
                    break
            
            if not name:
                name = f"Feature {index}" if index is not None else "Unnamed Feature"
            
            feature_data['name'] = name
            
            # Extract description
            description = None
            for desc_field in ['description', 'Description', 'desc', 'Desc', 'info', 'Info']:
                if desc_field in properties and properties[desc_field]:
                    description = str(properties[desc_field])
                    break
            
            feature_data['description'] = description or ''
            
            # Add coordinate info for Point geometries
            if feature_data['geometry'] and feature_data['geometry'].get('type') == 'Point':
                coordinates = feature_data['geometry'].get('coordinates', [])
                if len(coordinates) >= 2:
                    feature_data['longitude'] = coordinates[0]
                    feature_data['latitude'] = coordinates[1]
                    if len(coordinates) > 2:
                        feature_data['altitude'] = coordinates[2]
            
            # Extract ID if present
            if 'id' in geojson_feature:
                feature_data['feature_id'] = geojson_feature['id']
            
            return feature_data
            
        except Exception as e:
            self.logger.warning(f"Error processing GeoJSON feature: {e}")
            return None
    
    def _validate_features(self, result: ProcessingResult) -> ProcessingResult:
        """Validate geometries in processed features."""
        if not result.success or not result.features:
            return result
        
        validated_features = []
        validation_errors = []
        validation_warnings = []
        
        for i, feature in enumerate(result.features):
            if 'geometry' in feature and feature['geometry']:
                validation_result = self.geometry_validator.validate_geometry(feature['geometry'])
                
                # Use corrected geometry if available and valid
                if validation_result.corrected_geometry:
                    feature['geometry'] = validation_result.corrected_geometry
                    validation_warnings.append(f"Feature {i}: Geometry corrected")
                
                # Add validation info to feature
                feature['geometry_valid'] = validation_result.is_valid
                
                if validation_result.issues:
                    feature['geometry_issues'] = validation_result.issues
                    validation_errors.extend([f"Feature {i}: {issue}" for issue in validation_result.issues])
                
                if validation_result.warnings:
                    feature['geometry_warnings'] = validation_result.warnings
                    validation_warnings.extend([f"Feature {i}: {warning}" for warning in validation_result.warnings])
            else:
                feature['geometry_valid'] = False
                validation_errors.append(f"Feature {i}: Missing or null geometry")
            
            validated_features.append(feature)
        
        result.features = validated_features
        result.errors.extend(validation_errors)
        result.warnings.extend(validation_warnings)
        
        return result
    
    def create_geojson_feature_collection(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a GeoJSON FeatureCollection from processed features.
        
        Args:
            features: List of processed feature dictionaries
            
        Returns:
            GeoJSON FeatureCollection dictionary
        """
        geojson_features = []
        
        for feature in features:
            geojson_feature = {
                'type': 'Feature',
                'geometry': feature.get('geometry'),
                'properties': feature.get('properties', {})
            }
            
            # Add name to properties if not already there
            if 'name' in feature and 'name' not in geojson_feature['properties']:
                geojson_feature['properties']['name'] = feature['name']
            
            # Add description to properties if not already there
            if 'description' in feature and 'description' not in geojson_feature['properties']:
                geojson_feature['properties']['description'] = feature['description']
            
            # Add feature ID if present
            if 'feature_id' in feature:
                geojson_feature['id'] = feature['feature_id']
            
            geojson_features.append(geojson_feature)
        
        return {
            'type': 'FeatureCollection',
            'features': geojson_features
        }
    
    def validate_geojson_structure(self, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate GeoJSON structure without full processing.
        
        Args:
            geojson_data: GeoJSON data to validate
            
        Returns:
            Validation result with status and issues
        """
        issues = []
        warnings = []
        
        try:
            # Check basic structure
            if not isinstance(geojson_data, dict):
                issues.append("GeoJSON must be an object")
                return {'valid': False, 'issues': issues}
            
            geojson_type = geojson_data.get('type')
            if not geojson_type:
                issues.append("Missing 'type' property")
                return {'valid': False, 'issues': issues}
            
            # Validate based on type
            if geojson_type == 'FeatureCollection':
                features = geojson_data.get('features')
                if not isinstance(features, list):
                    issues.append("FeatureCollection 'features' must be an array")
                else:
                    for i, feature in enumerate(features):
                        if not isinstance(feature, dict) or feature.get('type') != 'Feature':
                            issues.append(f"Feature {i} is not a valid Feature object")
                        elif not feature.get('geometry'):
                            warnings.append(f"Feature {i} has no geometry")
            
            elif geojson_type == 'Feature':
                if not geojson_data.get('geometry'):
                    warnings.append("Feature has no geometry")
                if not isinstance(geojson_data.get('properties'), dict):
                    warnings.append("Feature properties should be an object")
            
            elif geojson_type not in ['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon', 'GeometryCollection']:
                issues.append(f"Unknown GeoJSON type: {geojson_type}")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'type': geojson_type
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': warnings
            }