"""
A2A World Platform - Enhanced KML Processor

Advanced KML parsing with extended data extraction, validation, and error handling.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import zipfile
import re

try:
    from fastkml import kml, features
    from fastkml import styles, atom
    FASTKML_AVAILABLE = True
except ImportError:
    FASTKML_AVAILABLE = False

from .geometry_validator import GeometryValidator, ValidationResult
from .quality_checker import QualityChecker, QualityReport


@dataclass
class ProcessingResult:
    """Result of KML processing."""
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


class KMLProcessor:
    """
    Enhanced KML processor with advanced parsing capabilities.
    
    Features:
    - Handles complex KML structures with nested folders
    - Extracts extended data and custom schemas
    - Processes KMZ files (zipped KML)
    - Style and icon information extraction
    - Network link resolution
    - Robust error handling and recovery
    - Data validation and quality assessment
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.geometry_validator = GeometryValidator()
        self.quality_checker = QualityChecker()
        
        # KML namespaces
        self.namespaces = {
            'kml': 'http://www.opengis.net/kml/2.2',
            'gx': 'http://www.google.com/kml/ext/2.2',
            'atom': 'http://www.w3.org/2005/Atom'
        }
        
    def process_file(
        self, 
        file_path: str,
        validate_geometry: bool = True,
        generate_quality_report: bool = True
    ) -> ProcessingResult:
        """
        Process KML or KMZ file.
        
        Args:
            file_path: Path to KML/KMZ file
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
            
            self.logger.info(f"Processing KML file: {file_path}")
            
            # Determine file type and process accordingly
            if file_path_obj.suffix.lower() == '.kmz':
                return self._process_kmz_file(file_path_obj, validate_geometry, generate_quality_report)
            else:
                return self._process_kml_file(file_path_obj, validate_geometry, generate_quality_report)
                
        except Exception as e:
            self.logger.error(f"Error processing KML file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                features=[],
                metadata={},
                errors=[f"Processing failed: {str(e)}"]
            )
    
    def _process_kml_file(
        self, 
        file_path: Path, 
        validate_geometry: bool,
        generate_quality_report: bool
    ) -> ProcessingResult:
        """Process a single KML file."""
        try:
            # Try fastkml first if available, fallback to basic parsing
            if FASTKML_AVAILABLE:
                result = self._process_with_fastkml(file_path)
            else:
                result = self._process_with_basic_xml(file_path)
            
            # Validate geometries if requested
            if validate_geometry and result.success:
                result = self._validate_features(result)
            
            # Generate quality report if requested
            if generate_quality_report and result.success:
                quality_report = self.quality_checker.check_dataset_quality(
                    result.features, 
                    f"KML: {file_path.name}"
                )
                result.quality_report = quality_report
            
            # Add file metadata
            result.metadata.update({
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'processed_at': datetime.utcnow().isoformat(),
                'processor': 'KMLProcessor'
            })
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata={'file_path': str(file_path)},
                errors=[f"KML processing failed: {str(e)}"]
            )
    
    def _process_kmz_file(
        self, 
        file_path: Path, 
        validate_geometry: bool,
        generate_quality_report: bool
    ) -> ProcessingResult:
        """Process a KMZ (zipped KML) file."""
        try:
            with zipfile.ZipFile(file_path, 'r') as kmz:
                # Find the main KML file (usually doc.kml)
                kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
                
                if not kml_files:
                    return ProcessingResult(
                        success=False,
                        features=[],
                        metadata={'file_path': str(file_path)},
                        errors=["No KML files found in KMZ archive"]
                    )
                
                # Process the first/main KML file
                main_kml = kml_files[0]
                if 'doc.kml' in kml_files:
                    main_kml = 'doc.kml'
                
                kml_content = kmz.read(main_kml).decode('utf-8')
                
                # Process KML content
                if FASTKML_AVAILABLE:
                    result = self._process_kml_content_fastkml(kml_content)
                else:
                    result = self._process_kml_content_basic(kml_content)
                
                # Extract additional files info
                result.metadata['kmz_files'] = kmz.namelist()
                result.metadata['main_kml'] = main_kml
                
                # Validate geometries if requested
                if validate_geometry and result.success:
                    result = self._validate_features(result)
                
                # Generate quality report if requested
                if generate_quality_report and result.success:
                    quality_report = self.quality_checker.check_dataset_quality(
                        result.features, 
                        f"KMZ: {file_path.name}"
                    )
                    result.quality_report = quality_report
                
                result.metadata.update({
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'processed_at': datetime.utcnow().isoformat(),
                    'processor': 'KMLProcessor'
                })
                
                return result
                
        except zipfile.BadZipFile:
            return ProcessingResult(
                success=False,
                features=[],
                metadata={'file_path': str(file_path)},
                errors=["File is not a valid KMZ archive"]
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata={'file_path': str(file_path)},
                errors=[f"KMZ processing failed: {str(e)}"]
            )
    
    def _process_with_fastkml(self, file_path: Path) -> ProcessingResult:
        """Process KML using fastkml library."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._process_kml_content_fastkml(content)
            
        except Exception as e:
            self.logger.error(f"FastKML processing failed: {e}")
            return ProcessingResult(
                success=False,
                features=[],
                metadata={},
                errors=[f"FastKML processing failed: {str(e)}"]
            )
    
    def _process_kml_content_fastkml(self, content: str) -> ProcessingResult:
        """Process KML content using fastkml."""
        features = []
        metadata = {'library': 'fastkml'}
        warnings = []
        
        try:
            k = kml.KML()
            k.from_string(content)
            
            # Extract document-level metadata
            for feature in k.features():
                if isinstance(feature, features.Document):
                    metadata.update({
                        'document_name': getattr(feature, 'name', None),
                        'document_description': getattr(feature, 'description', None),
                        'document_author': getattr(feature, 'author', None)
                    })
                    
                    # Process features in document
                    features.extend(self._extract_features_fastkml(feature, []))
                    
                elif isinstance(feature, features.Folder):
                    # Process folder
                    folder_features = self._extract_features_fastkml(feature, [feature.name or 'Unnamed Folder'])
                    features.extend(folder_features)
                    
                elif isinstance(feature, features.Placemark):
                    # Individual placemark
                    placemark_data = self._extract_placemark_fastkml(feature, [])
                    if placemark_data:
                        features.append(placemark_data)
            
            return ProcessingResult(
                success=True,
                features=features,
                metadata=metadata,
                warnings=warnings
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata=metadata,
                errors=[f"FastKML content processing failed: {str(e)}"]
            )
    
    def _extract_features_fastkml(self, container, folder_path: List[str]) -> List[Dict[str, Any]]:
        """Recursively extract features from KML container using fastkml."""
        features = []
        
        try:
            for feature in container.features():
                if isinstance(feature, features.Folder):
                    # Recursive folder processing
                    sub_folder_path = folder_path + [feature.name or 'Unnamed Folder']
                    sub_features = self._extract_features_fastkml(feature, sub_folder_path)
                    features.extend(sub_features)
                    
                elif isinstance(feature, features.Placemark):
                    placemark_data = self._extract_placemark_fastkml(feature, folder_path)
                    if placemark_data:
                        features.append(placemark_data)
        
        except Exception as e:
            self.logger.warning(f"Error extracting features from container: {e}")
        
        return features
    
    def _extract_placemark_fastkml(self, placemark, folder_path: List[str]) -> Optional[Dict[str, Any]]:
        """Extract data from placemark using fastkml."""
        try:
            feature_data = {
                'name': getattr(placemark, 'name', None) or 'Unnamed',
                'description': getattr(placemark, 'description', None) or '',
                'folder_path': folder_path.copy() if folder_path else [],
                'properties': {},
                'style_info': {}
            }
            
            # Extract geometry
            if hasattr(placemark, '_geometry') and placemark._geometry:
                geometry_data = self._convert_geometry_to_geojson(placemark._geometry)
                if geometry_data:
                    feature_data['geometry'] = geometry_data
                    
                    # Add coordinate info for easier access
                    if geometry_data['type'] == 'Point':
                        coords = geometry_data['coordinates']
                        feature_data['longitude'] = coords[0]
                        feature_data['latitude'] = coords[1]
                        if len(coords) > 2:
                            feature_data['altitude'] = coords[2]
            
            # Extract extended data
            if hasattr(placemark, 'extended_data') and placemark.extended_data:
                extended_data = placemark.extended_data
                if hasattr(extended_data, 'elements'):
                    for element in extended_data.elements:
                        if hasattr(element, 'name') and hasattr(element, 'value'):
                            feature_data['properties'][element.name] = element.value
            
            # Extract style information
            if hasattr(placemark, 'styleUrl') and placemark.styleUrl:
                feature_data['style_info']['style_url'] = placemark.styleUrl
            
            if hasattr(placemark, '_styles') and placemark._styles:
                for style in placemark._styles:
                    style_data = self._extract_style_info(style)
                    feature_data['style_info'].update(style_data)
            
            # Extract timestamps if present
            if hasattr(placemark, 'timeStamp') and placemark.timeStamp:
                feature_data['timestamp'] = str(placemark.timeStamp.timestamp)
            
            if hasattr(placemark, 'timeSpan') and placemark.timeSpan:
                feature_data['time_span'] = {
                    'begin': str(placemark.timeSpan.begin) if placemark.timeSpan.begin else None,
                    'end': str(placemark.timeSpan.end) if placemark.timeSpan.end else None
                }
            
            return feature_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting placemark data: {e}")
            return None
    
    def _process_with_basic_xml(self, file_path: Path) -> ProcessingResult:
        """Process KML using basic XML parsing (fallback)."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            content = file_path.read_text(encoding='utf-8')
            return self._process_kml_content_basic(content)
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata={},
                errors=[f"Basic XML processing failed: {str(e)}"]
            )
    
    def _process_kml_content_basic(self, content: str) -> ProcessingResult:
        """Process KML content using basic XML parsing."""
        features = []
        metadata = {'library': 'xml.etree'}
        warnings = []
        
        try:
            root = ET.fromstring(content)
            
            # Extract document information
            doc_name = self._find_element_text(root, './/kml:Document/kml:name')
            doc_desc = self._find_element_text(root, './/kml:Document/kml:description')
            
            if doc_name:
                metadata['document_name'] = doc_name
            if doc_desc:
                metadata['document_description'] = doc_desc
            
            # Find all placemarks
            placemarks = root.findall('.//kml:Placemark', self.namespaces)
            
            for placemark in placemarks:
                placemark_data = self._extract_placemark_basic(placemark)
                if placemark_data:
                    features.append(placemark_data)
            
            return ProcessingResult(
                success=True,
                features=features,
                metadata=metadata,
                warnings=warnings
            )
            
        except ET.ParseError as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata=metadata,
                errors=[f"XML parsing error: {str(e)}"]
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                features=[],
                metadata=metadata,
                errors=[f"Basic XML content processing failed: {str(e)}"]
            )
    
    def _extract_placemark_basic(self, placemark: ET.Element) -> Optional[Dict[str, Any]]:
        """Extract placemark data using basic XML parsing."""
        try:
            feature_data = {
                'name': self._find_element_text(placemark, 'kml:name') or 'Unnamed',
                'description': self._find_element_text(placemark, 'kml:description') or '',
                'properties': {},
                'style_info': {}
            }
            
            # Find folder path
            folder_path = []
            current = placemark.getparent() if hasattr(placemark, 'getparent') else None
            while current is not None:
                if current.tag.endswith('Folder'):
                    folder_name = self._find_element_text(current, 'kml:name')
                    if folder_name:
                        folder_path.insert(0, folder_name)
                current = current.getparent() if hasattr(current, 'getparent') else None
            
            feature_data['folder_path'] = folder_path
            
            # Extract geometry (Point only for basic parser)
            point_elem = placemark.find('.//kml:Point/kml:coordinates', self.namespaces)
            if point_elem is not None and point_elem.text:
                coords_text = point_elem.text.strip()
                coords = self._parse_coordinates(coords_text)
                if coords:
                    feature_data['geometry'] = {
                        'type': 'Point',
                        'coordinates': coords
                    }
                    feature_data['longitude'] = coords[0]
                    feature_data['latitude'] = coords[1]
                    if len(coords) > 2:
                        feature_data['altitude'] = coords[2]
            
            # Extract extended data
            extended_data = placemark.find('kml:ExtendedData', self.namespaces)
            if extended_data is not None:
                for data_elem in extended_data.findall('kml:Data', self.namespaces):
                    name = data_elem.get('name')
                    value_elem = data_elem.find('kml:value', self.namespaces)
                    if name and value_elem is not None:
                        feature_data['properties'][name] = value_elem.text
            
            # Extract style URL
            style_url = self._find_element_text(placemark, 'kml:styleUrl')
            if style_url:
                feature_data['style_info']['style_url'] = style_url
            
            return feature_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting placemark (basic): {e}")
            return None
    
    def _find_element_text(self, parent: ET.Element, xpath: str) -> Optional[str]:
        """Find element text using xpath."""
        try:
            element = parent.find(xpath, self.namespaces)
            return element.text if element is not None else None
        except Exception:
            return None
    
    def _parse_coordinates(self, coords_text: str) -> Optional[List[float]]:
        """Parse coordinate string."""
        try:
            # Handle different coordinate formats
            coords_text = coords_text.strip()
            
            # Split by whitespace or comma
            parts = re.split(r'[,\s]+', coords_text)
            coords = [float(p) for p in parts if p.strip()]
            
            if len(coords) >= 2:
                return coords[:3]  # lon, lat, alt (if present)
            
            return None
            
        except (ValueError, IndexError):
            return None
    
    def _convert_geometry_to_geojson(self, geometry) -> Optional[Dict[str, Any]]:
        """Convert geometry object to GeoJSON format."""
        try:
            # This would depend on the specific geometry library
            # For now, implement basic conversion
            if hasattr(geometry, 'geom_type'):
                geom_type = geometry.geom_type
                
                if geom_type == 'Point':
                    coords = list(geometry.coords)[0]
                    return {
                        'type': 'Point',
                        'coordinates': list(coords)
                    }
                elif geom_type == 'LineString':
                    coords = list(geometry.coords)
                    return {
                        'type': 'LineString',
                        'coordinates': [list(c) for c in coords]
                    }
                elif geom_type == 'Polygon':
                    exterior = list(geometry.exterior.coords)
                    holes = [list(hole.coords) for hole in geometry.interiors]
                    return {
                        'type': 'Polygon',
                        'coordinates': [exterior] + holes
                    }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error converting geometry: {e}")
            return None
    
    def _extract_style_info(self, style) -> Dict[str, Any]:
        """Extract style information."""
        style_data = {}
        
        try:
            if hasattr(style, 'id'):
                style_data['id'] = style.id
            
            # Extract icon style
            if hasattr(style, 'iconstyle') and style.iconstyle:
                icon_style = style.iconstyle
                style_data['icon'] = {
                    'href': getattr(icon_style, 'icon_href', None),
                    'scale': getattr(icon_style, 'scale', None),
                    'color': getattr(icon_style, 'color', None)
                }
            
            # Extract line style
            if hasattr(style, 'linestyle') and style.linestyle:
                line_style = style.linestyle
                style_data['line'] = {
                    'color': getattr(line_style, 'color', None),
                    'width': getattr(line_style, 'width', None)
                }
            
            # Extract polygon style
            if hasattr(style, 'polystyle') and style.polystyle:
                poly_style = style.polystyle
                style_data['polygon'] = {
                    'color': getattr(poly_style, 'color', None),
                    'fill': getattr(poly_style, 'fill', None),
                    'outline': getattr(poly_style, 'outline', None)
                }
        
        except Exception as e:
            self.logger.warning(f"Error extracting style info: {e}")
        
        return style_data
    
    def _validate_features(self, result: ProcessingResult) -> ProcessingResult:
        """Validate geometries in processed features."""
        if not result.success or not result.features:
            return result
        
        validated_features = []
        validation_errors = []
        validation_warnings = []
        
        for i, feature in enumerate(result.features):
            if 'geometry' in feature:
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
            
            validated_features.append(feature)
        
        result.features = validated_features
        result.errors.extend(validation_errors)
        result.warnings.extend(validation_warnings)
        
        return result