"""
A2A World Platform - CSV Processor

Processes CSV files with coordinate data and metadata for geospatial analysis.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

from .geometry_validator import GeometryValidator, ValidationResult
from .quality_checker import QualityChecker, QualityReport


@dataclass
class ProcessingResult:
    """Result of CSV processing."""
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


@dataclass
class ColumnMapping:
    """Mapping of CSV columns to standard fields."""
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    id: Optional[str] = None
    elevation: Optional[str] = None
    type: Optional[str] = None
    properties: List[str] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = []


class CSVProcessor:
    """
    Enhanced CSV processor for geospatial coordinate data.
    
    Features:
    - Auto-detects coordinate columns
    - Handles various coordinate formats (decimal, DMS)
    - Extracts metadata and properties
    - Data validation and quality assessment
    - Supports custom column mappings
    - Handles different delimiters and encodings
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.geometry_validator = GeometryValidator()
        self.quality_checker = QualityChecker()
        
        # Common column name patterns for auto-detection
        self.latitude_patterns = [
            r'lat(?:itude)?$', r'y$', r'north(?:ing)?$', r'lat_deg$'
        ]
        self.longitude_patterns = [
            r'lon(?:g|gitude)?$', r'lng$', r'x$', r'east(?:ing)?$', r'lon_deg$'
        ]
        self.name_patterns = [
            r'name$', r'title$', r'label$', r'site$', r'location$', r'place$'
        ]
        self.description_patterns = [
            r'desc(?:ription)?$', r'info$', r'notes?$', r'comment$', r'details?$'
        ]
        
    def process_file(
        self,
        file_path: str,
        delimiter: str = 'auto',
        encoding: str = 'utf-8',
        column_mapping: Optional[ColumnMapping] = None,
        validate_geometry: bool = True,
        generate_quality_report: bool = True
    ) -> ProcessingResult:
        """
        Process CSV file with coordinate data.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter ('auto' for auto-detection)
            encoding: File encoding
            column_mapping: Custom column mapping
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
            
            self.logger.info(f"Processing CSV file: {file_path}")
            
            # Read CSV content
            try:
                with open(file_path_obj, 'r', encoding=encoding, newline='') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try different encodings
                for alt_encoding in ['latin-1', 'cp1252', 'utf-8-sig']:
                    try:
                        with open(file_path_obj, 'r', encoding=alt_encoding, newline='') as f:
                            content = f.read()
                        encoding = alt_encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return ProcessingResult(
                        success=False,
                        features=[],
                        metadata={'file_path': str(file_path_obj)},
                        errors=["Could not decode file with any supported encoding"]
                    )
            
            # Detect delimiter if needed
            if delimiter == 'auto':
                delimiter = self._detect_delimiter(content)
            
            # Process CSV content
            result = self._process_csv_content(
                content, delimiter, column_mapping, validate_geometry, generate_quality_report
            )
            
            # Add file metadata
            result.metadata.update({
                'file_path': str(file_path_obj),
                'file_size': file_path_obj.stat().st_size,
                'encoding': encoding,
                'delimiter': delimiter,
                'processed_at': datetime.utcnow().isoformat(),
                'processor': 'CSVProcessor'
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                features=[],
                metadata={'file_path': file_path},
                errors=[f"Processing failed: {str(e)}"]
            )
    
    def _detect_delimiter(self, content: str) -> str:
        """Auto-detect CSV delimiter."""
        try:
            sniffer = csv.Sniffer()
            sample = content[:2048]  # Use first 2KB for detection
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except Exception:
            # Fallback to comma
            return ','
    
    def _process_csv_content(
        self,
        content: str,
        delimiter: str,
        column_mapping: Optional[ColumnMapping],
        validate_geometry: bool,
        generate_quality_report: bool
    ) -> ProcessingResult:
        """Process CSV content."""
        features = []
        metadata = {}
        warnings = []
        errors = []
        
        try:
            # Parse CSV
            csv_reader = csv.reader(content.splitlines(), delimiter=delimiter)
            rows = list(csv_reader)
            
            if not rows:
                return ProcessingResult(
                    success=False,
                    features=[],
                    metadata={},
                    errors=["CSV file is empty"]
                )
            
            # Extract headers
            headers = rows[0]
            data_rows = rows[1:]
            
            # Auto-detect or use provided column mapping
            if column_mapping is None:
                column_mapping = self._auto_detect_columns(headers)
                if not column_mapping.latitude or not column_mapping.longitude:
                    return ProcessingResult(
                        success=False,
                        features=[],
                        metadata={'headers': headers},
                        errors=["Could not detect latitude/longitude columns"]
                    )
            
            # Validate column mapping
            missing_columns = self._validate_column_mapping(column_mapping, headers)
            if missing_columns:
                return ProcessingResult(
                    success=False,
                    features=[],
                    metadata={'headers': headers},
                    errors=[f"Missing required columns: {', '.join(missing_columns)}"]
                )
            
            # Process data rows
            for row_index, row in enumerate(data_rows):
                try:
                    feature = self._process_csv_row(row, headers, column_mapping, row_index)
                    if feature:
                        features.append(feature)
                    else:
                        warnings.append(f"Skipped invalid row {row_index + 2}")
                except Exception as e:
                    errors.append(f"Error processing row {row_index + 2}: {str(e)}")
            
            # Update metadata
            metadata.update({
                'total_rows': len(data_rows),
                'processed_rows': len(features),
                'headers': headers,
                'column_mapping': {
                    'latitude': column_mapping.latitude,
                    'longitude': column_mapping.longitude,
                    'name': column_mapping.name,
                    'description': column_mapping.description,
                    'id': column_mapping.id,
                    'elevation': column_mapping.elevation,
                    'type': column_mapping.type,
                    'properties': column_mapping.properties
                }
            })
            
            # Validate geometries if requested
            if validate_geometry and features:
                features, validation_errors, validation_warnings = self._validate_features(features)
                errors.extend(validation_errors)
                warnings.extend(validation_warnings)
            
            # Generate quality report if requested
            quality_report = None
            if generate_quality_report and features:
                quality_report = self.quality_checker.check_dataset_quality(
                    features,
                    "CSV Dataset"
                )
            
            return ProcessingResult(
                success=True,
                features=features,
                metadata=metadata,
                quality_report=quality_report,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata={},
                errors=[f"CSV content processing failed: {str(e)}"]
            )
    
    def _auto_detect_columns(self, headers: List[str]) -> ColumnMapping:
        """Auto-detect column mappings from headers."""
        mapping = ColumnMapping()
        
        # Convert headers to lowercase for matching
        header_map = {header.lower(): header for header in headers}
        
        # Detect latitude column
        for pattern in self.latitude_patterns:
            for header_lower, header_original in header_map.items():
                if re.search(pattern, header_lower, re.IGNORECASE):
                    mapping.latitude = header_original
                    break
            if mapping.latitude:
                break
        
        # Detect longitude column
        for pattern in self.longitude_patterns:
            for header_lower, header_original in header_map.items():
                if re.search(pattern, header_lower, re.IGNORECASE):
                    mapping.longitude = header_original
                    break
            if mapping.longitude:
                break
        
        # Detect name column
        for pattern in self.name_patterns:
            for header_lower, header_original in header_map.items():
                if re.search(pattern, header_lower, re.IGNORECASE):
                    mapping.name = header_original
                    break
            if mapping.name:
                break
        
        # Detect description column
        for pattern in self.description_patterns:
            for header_lower, header_original in header_map.items():
                if re.search(pattern, header_lower, re.IGNORECASE):
                    mapping.description = header_original
                    break
            if mapping.description:
                break
        
        # Detect other common columns
        for header_lower, header_original in header_map.items():
            if re.search(r'id$', header_lower, re.IGNORECASE):
                mapping.id = header_original
            elif re.search(r'(?:elev|alt)(?:ation)?$', header_lower, re.IGNORECASE):
                mapping.elevation = header_original
            elif re.search(r'type$', header_lower, re.IGNORECASE):
                mapping.type = header_original
        
        # All other columns become properties
        used_columns = {mapping.latitude, mapping.longitude, mapping.name, 
                       mapping.description, mapping.id, mapping.elevation, mapping.type}
        mapping.properties = [h for h in headers if h not in used_columns]
        
        return mapping
    
    def _validate_column_mapping(self, mapping: ColumnMapping, headers: List[str]) -> List[str]:
        """Validate that required columns exist in headers."""
        missing = []
        
        if mapping.latitude and mapping.latitude not in headers:
            missing.append(mapping.latitude)
        
        if mapping.longitude and mapping.longitude not in headers:
            missing.append(mapping.longitude)
        
        return missing
    
    def _process_csv_row(
        self, 
        row: List[str], 
        headers: List[str], 
        mapping: ColumnMapping, 
        row_index: int
    ) -> Optional[Dict[str, Any]]:
        """Process individual CSV row."""
        try:
            # Create header-value mapping
            if len(row) != len(headers):
                self.logger.warning(f"Row {row_index + 2} has {len(row)} values but {len(headers)} headers")
                # Pad with empty strings if row is shorter
                while len(row) < len(headers):
                    row.append('')
            
            row_data = dict(zip(headers, row))
            
            # Extract coordinates
            lat_str = row_data.get(mapping.latitude, '').strip()
            lon_str = row_data.get(mapping.longitude, '').strip()
            
            if not lat_str or not lon_str:
                return None  # Skip rows with missing coordinates
            
            # Parse coordinates
            latitude = self._parse_coordinate(lat_str, 'latitude')
            longitude = self._parse_coordinate(lon_str, 'longitude')
            
            if latitude is None or longitude is None:
                return None
            
            # Validate coordinate ranges
            if not (-90 <= latitude <= 90):
                self.logger.warning(f"Row {row_index + 2}: Latitude {latitude} out of range")
                return None
            
            if not (-180 <= longitude <= 180):
                self.logger.warning(f"Row {row_index + 2}: Longitude {longitude} out of range")
                return None
            
            # Build feature
            feature = {
                'name': row_data.get(mapping.name, '').strip() or f"Point {row_index + 1}",
                'description': row_data.get(mapping.description, '').strip(),
                'latitude': latitude,
                'longitude': longitude,
                'geometry': {
                    'type': 'Point',
                    'coordinates': [longitude, latitude]
                },
                'properties': {}
            }
            
            # Add elevation if present
            if mapping.elevation and mapping.elevation in row_data:
                elev_str = row_data[mapping.elevation].strip()
                if elev_str:
                    try:
                        elevation = float(elev_str)
                        feature['elevation'] = elevation
                        feature['geometry']['coordinates'] = [longitude, latitude, elevation]
                    except ValueError:
                        pass
            
            # Add ID if present
            if mapping.id and mapping.id in row_data:
                feature_id = row_data[mapping.id].strip()
                if feature_id:
                    feature['feature_id'] = feature_id
            
            # Add type if present
            if mapping.type and mapping.type in row_data:
                feature_type = row_data[mapping.type].strip()
                if feature_type:
                    feature['feature_type'] = feature_type
            
            # Add all other columns as properties
            for prop_column in mapping.properties:
                if prop_column in row_data:
                    value = row_data[prop_column].strip()
                    if value:  # Only add non-empty values
                        feature['properties'][prop_column] = value
            
            return feature
            
        except Exception as e:
            self.logger.warning(f"Error processing CSV row {row_index + 2}: {e}")
            return None
    
    def _parse_coordinate(self, coord_str: str, coord_type: str) -> Optional[float]:
        """
        Parse coordinate string, handling various formats.
        
        Supports:
        - Decimal degrees: 40.7128
        - Degrees with cardinal direction: 40.7128N, 74.0060W
        - Degrees, minutes, seconds: 40°42'46"N
        """
        try:
            coord_str = coord_str.strip()
            if not coord_str:
                return None
            
            # Simple decimal format
            try:
                return float(coord_str)
            except ValueError:
                pass
            
            # Handle cardinal directions
            cardinal_multiplier = 1
            if coord_str[-1].upper() in ['S', 'W']:
                cardinal_multiplier = -1
                coord_str = coord_str[:-1].strip()
            elif coord_str[-1].upper() in ['N', 'E']:
                coord_str = coord_str[:-1].strip()
            
            # Try decimal again after removing cardinal
            try:
                return float(coord_str) * cardinal_multiplier
            except ValueError:
                pass
            
            # Handle DMS (Degrees, Minutes, Seconds) format
            # Pattern: 40°42'46"N or 40:42:46N or 40 42 46 N
            dms_patterns = [
                r"(\d+)[°:]?\s*(\d+)[':′]?\s*(\d+(?:\.\d+)?)[\"″]?",
                r"(\d+)[°:]?\s*(\d+(?:\.\d+)?)[':′]?",
                r"(\d+(?:\.\d+)?)[°]?"
            ]
            
            for pattern in dms_patterns:
                match = re.match(pattern, coord_str)
                if match:
                    parts = [float(p) for p in match.groups()]
                    
                    if len(parts) == 3:  # DMS
                        decimal = parts[0] + parts[1]/60 + parts[2]/3600
                    elif len(parts) == 2:  # DM
                        decimal = parts[0] + parts[1]/60
                    else:  # D
                        decimal = parts[0]
                    
                    return decimal * cardinal_multiplier
            
            return None
            
        except Exception:
            return None
    
    def _validate_features(self, features: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Validate geometries in processed features."""
        validated_features = []
        validation_errors = []
        validation_warnings = []
        
        for i, feature in enumerate(features):
            if 'geometry' in feature and feature['geometry']:
                validation_result = self.geometry_validator.validate_geometry(feature['geometry'])
                
                # Use corrected geometry if available and valid
                if validation_result.corrected_geometry:
                    feature['geometry'] = validation_result.corrected_geometry
                    # Update coordinate fields
                    coords = validation_result.corrected_geometry['coordinates']
                    feature['longitude'] = coords[0]
                    feature['latitude'] = coords[1]
                    if len(coords) > 2:
                        feature['elevation'] = coords[2]
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
                validation_errors.append(f"Feature {i}: Missing geometry")
            
            validated_features.append(feature)
        
        return validated_features, validation_errors, validation_warnings
    
    def create_column_mapping(
        self,
        latitude: str,
        longitude: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        id: Optional[str] = None,
        elevation: Optional[str] = None,
        type_column: Optional[str] = None,
        properties: Optional[List[str]] = None
    ) -> ColumnMapping:
        """
        Create a custom column mapping.
        
        Args:
            latitude: Name of latitude column
            longitude: Name of longitude column
            name: Name of name/title column
            description: Name of description column
            id: Name of ID column
            elevation: Name of elevation/altitude column
            type_column: Name of type/category column
            properties: List of additional property columns
            
        Returns:
            ColumnMapping object
        """
        return ColumnMapping(
            latitude=latitude,
            longitude=longitude,
            name=name,
            description=description,
            id=id,
            elevation=elevation,
            type=type_column,
            properties=properties or []
        )
    
    def preview_file(
        self,
        file_path: str,
        delimiter: str = 'auto',
        encoding: str = 'utf-8',
        max_rows: int = 10
    ) -> Dict[str, Any]:
        """
        Preview CSV file structure and detect columns.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter ('auto' for auto-detection)
            encoding: File encoding
            max_rows: Maximum number of data rows to preview
            
        Returns:
            Preview information including headers, sample data, and detected mappings
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return {'error': f'File not found: {file_path}'}
            
            # Read file
            with open(file_path_obj, 'r', encoding=encoding, newline='') as f:
                content = f.read(8192)  # Read first 8KB for preview
            
            # Detect delimiter
            if delimiter == 'auto':
                delimiter = self._detect_delimiter(content)
            
            # Parse preview
            csv_reader = csv.reader(content.splitlines(), delimiter=delimiter)
            rows = list(csv_reader)
            
            if not rows:
                return {'error': 'CSV file is empty'}
            
            headers = rows[0]
            sample_data = rows[1:max_rows+1]
            
            # Auto-detect columns
            detected_mapping = self._auto_detect_columns(headers)
            
            return {
                'file_path': str(file_path_obj),
                'file_size': file_path_obj.stat().st_size,
                'delimiter': delimiter,
                'encoding': encoding,
                'headers': headers,
                'sample_data': sample_data,
                'detected_mapping': {
                    'latitude': detected_mapping.latitude,
                    'longitude': detected_mapping.longitude,
                    'name': detected_mapping.name,
                    'description': detected_mapping.description,
                    'id': detected_mapping.id,
                    'elevation': detected_mapping.elevation,
                    'type': detected_mapping.type,
                    'properties': detected_mapping.properties
                },
                'total_rows': len(rows) - 1,  # Exclude header
                'preview_complete': len(rows) <= max_rows + 1
            }
            
        except Exception as e:
            return {'error': f'Preview failed: {str(e)}'}