"""
A2A World Platform - Geometry Validator

Validates and corrects geospatial geometries for data quality assurance.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import math

try:
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
    from shapely.validation import make_valid
    from shapely.ops import transform
    import pyproj
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of geometry validation."""
    is_valid: bool
    geometry: Optional[Dict[str, Any]] = None
    corrected_geometry: Optional[Dict[str, Any]] = None
    issues: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []


class GeometryValidator:
    """
    Validates and corrects geospatial geometries.
    
    Features:
    - Coordinate system validation and conversion
    - Geometry validity checking
    - Common issue correction (self-intersecting polygons, etc.)
    - Coordinate range validation
    - Duplicate coordinate removal
    - Topology validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_geometry(self, geometry: Dict[str, Any]) -> ValidationResult:
        """
        Validate a GeoJSON geometry object.
        
        Args:
            geometry: GeoJSON geometry dict
            
        Returns:
            ValidationResult with validation status and corrections
        """
        if not geometry:
            return ValidationResult(
                is_valid=False,
                issues=["Geometry is None or empty"]
            )
            
        try:
            geom_type = geometry.get("type")
            coordinates = geometry.get("coordinates")
            
            if not geom_type or not coordinates:
                return ValidationResult(
                    is_valid=False,
                    geometry=geometry,
                    issues=["Missing geometry type or coordinates"]
                )
            
            # Dispatch to specific validation method
            if geom_type == "Point":
                return self._validate_point(geometry)
            elif geom_type == "LineString":
                return self._validate_linestring(geometry)
            elif geom_type == "Polygon":
                return self._validate_polygon(geometry)
            elif geom_type == "MultiPoint":
                return self._validate_multipoint(geometry)
            elif geom_type == "MultiLineString":
                return self._validate_multilinestring(geometry)
            elif geom_type == "MultiPolygon":
                return self._validate_multipolygon(geometry)
            else:
                return ValidationResult(
                    is_valid=False,
                    geometry=geometry,
                    issues=[f"Unsupported geometry type: {geom_type}"]
                )
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                geometry=geometry,
                issues=[f"Validation error: {str(e)}"]
            )
    
    def _validate_point(self, geometry: Dict[str, Any]) -> ValidationResult:
        """Validate Point geometry."""
        coordinates = geometry["coordinates"]
        issues = []
        warnings = []
        
        if len(coordinates) < 2:
            return ValidationResult(
                is_valid=False,
                geometry=geometry,
                issues=["Point must have at least 2 coordinates (longitude, latitude)"]
            )
        
        lon, lat = coordinates[0], coordinates[1]
        
        # Validate coordinate ranges
        if not (-180 <= lon <= 180):
            issues.append(f"Longitude {lon} is out of valid range (-180 to 180)")
        
        if not (-90 <= lat <= 90):
            issues.append(f"Latitude {lat} is out of valid range (-90 to 90)")
        
        # Check for common coordinate swap issue
        if abs(lat) > 90 and abs(lon) <= 90:
            warnings.append("Possible coordinate swap detected (latitude > 90)")
            
        # Try to correct if issues found
        corrected_geometry = None
        if issues:
            # Attempt to fix common issues
            corrected_lon = max(-180, min(180, lon))
            corrected_lat = max(-90, min(90, lat))
            
            corrected_geometry = {
                "type": "Point",
                "coordinates": [corrected_lon, corrected_lat]
            }
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            geometry=geometry,
            corrected_geometry=corrected_geometry,
            issues=issues,
            warnings=warnings
        )
    
    def _validate_linestring(self, geometry: Dict[str, Any]) -> ValidationResult:
        """Validate LineString geometry."""
        coordinates = geometry["coordinates"]
        issues = []
        warnings = []
        
        if len(coordinates) < 2:
            return ValidationResult(
                is_valid=False,
                geometry=geometry,
                issues=["LineString must have at least 2 coordinate pairs"]
            )
        
        # Validate each coordinate pair
        valid_coords = []
        for i, coord in enumerate(coordinates):
            if len(coord) < 2:
                issues.append(f"Coordinate {i} must have at least 2 values")
                continue
                
            lon, lat = coord[0], coord[1]
            
            if not (-180 <= lon <= 180):
                issues.append(f"Coordinate {i}: longitude {lon} out of range")
            
            if not (-90 <= lat <= 90):
                issues.append(f"Coordinate {i}: latitude {lat} out of range")
            else:
                valid_coords.append(coord)
        
        # Remove duplicate consecutive coordinates
        cleaned_coords = self._remove_duplicate_coordinates(coordinates)
        if len(cleaned_coords) != len(coordinates):
            warnings.append(f"Removed {len(coordinates) - len(cleaned_coords)} duplicate coordinates")
        
        corrected_geometry = None
        if warnings or (valid_coords and len(valid_coords) != len(coordinates)):
            corrected_geometry = {
                "type": "LineString", 
                "coordinates": valid_coords if valid_coords else cleaned_coords
            }
        
        return ValidationResult(
            is_valid=len(issues) == 0 and len(valid_coords) >= 2,
            geometry=geometry,
            corrected_geometry=corrected_geometry,
            issues=issues,
            warnings=warnings
        )
    
    def _validate_polygon(self, geometry: Dict[str, Any]) -> ValidationResult:
        """Validate Polygon geometry."""
        coordinates = geometry["coordinates"]
        issues = []
        warnings = []
        
        if not coordinates or len(coordinates) == 0:
            return ValidationResult(
                is_valid=False,
                geometry=geometry,
                issues=["Polygon must have at least one ring"]
            )
        
        # Validate exterior ring
        exterior_ring = coordinates[0]
        if len(exterior_ring) < 4:
            issues.append("Polygon exterior ring must have at least 4 coordinate pairs")
        
        # Check if ring is closed
        if exterior_ring and exterior_ring[0] != exterior_ring[-1]:
            issues.append("Polygon ring is not closed (first and last coordinates must be identical)")
        
        # Validate coordinates in rings
        corrected_rings = []
        for ring_idx, ring in enumerate(coordinates):
            valid_coords = []
            for coord_idx, coord in enumerate(ring):
                if len(coord) < 2:
                    issues.append(f"Ring {ring_idx}, coordinate {coord_idx} must have at least 2 values")
                    continue
                    
                lon, lat = coord[0], coord[1]
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    valid_coords.append(coord)
                else:
                    issues.append(f"Ring {ring_idx}, coordinate {coord_idx}: invalid coordinates {lon}, {lat}")
            
            # Remove duplicates but preserve closure
            if valid_coords:
                cleaned_coords = self._remove_duplicate_coordinates(valid_coords, preserve_closure=True)
                corrected_rings.append(cleaned_coords)
        
        # Use Shapely for advanced validation if available
        if SHAPELY_AVAILABLE and not issues:
            try:
                shapely_geom = Polygon(exterior_ring, coordinates[1:] if len(coordinates) > 1 else [])
                if not shapely_geom.is_valid:
                    issues.append(f"Polygon topology is invalid: {shapely_geom.explain_validity()}")
                    # Try to fix
                    fixed_geom = make_valid(shapely_geom)
                    if fixed_geom and fixed_geom.is_valid:
                        warnings.append("Polygon topology was corrected")
            except Exception as e:
                warnings.append(f"Could not validate topology: {e}")
        
        corrected_geometry = None
        if corrected_rings and (issues or warnings):
            corrected_geometry = {
                "type": "Polygon",
                "coordinates": corrected_rings
            }
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            geometry=geometry,
            corrected_geometry=corrected_geometry,
            issues=issues,
            warnings=warnings
        )
    
    def _validate_multipoint(self, geometry: Dict[str, Any]) -> ValidationResult:
        """Validate MultiPoint geometry."""
        coordinates = geometry["coordinates"]
        issues = []
        warnings = []
        valid_points = []
        
        for i, point_coords in enumerate(coordinates):
            point_geom = {"type": "Point", "coordinates": point_coords}
            result = self._validate_point(point_geom)
            
            if result.is_valid:
                valid_points.append(point_coords)
            else:
                issues.extend([f"Point {i}: {issue}" for issue in result.issues])
            
            warnings.extend([f"Point {i}: {warning}" for warning in result.warnings])
        
        corrected_geometry = None
        if valid_points and len(valid_points) != len(coordinates):
            corrected_geometry = {
                "type": "MultiPoint",
                "coordinates": valid_points
            }
        
        return ValidationResult(
            is_valid=len(issues) == 0 and len(valid_points) > 0,
            geometry=geometry,
            corrected_geometry=corrected_geometry,
            issues=issues,
            warnings=warnings
        )
    
    def _validate_multilinestring(self, geometry: Dict[str, Any]) -> ValidationResult:
        """Validate MultiLineString geometry."""
        coordinates = geometry["coordinates"]
        issues = []
        warnings = []
        valid_linestrings = []
        
        for i, linestring_coords in enumerate(coordinates):
            linestring_geom = {"type": "LineString", "coordinates": linestring_coords}
            result = self._validate_linestring(linestring_geom)
            
            if result.is_valid:
                valid_linestrings.append(linestring_coords)
            else:
                issues.extend([f"LineString {i}: {issue}" for issue in result.issues])
            
            warnings.extend([f"LineString {i}: {warning}" for warning in result.warnings])
        
        corrected_geometry = None
        if valid_linestrings and len(valid_linestrings) != len(coordinates):
            corrected_geometry = {
                "type": "MultiLineString",
                "coordinates": valid_linestrings
            }
        
        return ValidationResult(
            is_valid=len(issues) == 0 and len(valid_linestrings) > 0,
            geometry=geometry,
            corrected_geometry=corrected_geometry,
            issues=issues,
            warnings=warnings
        )
    
    def _validate_multipolygon(self, geometry: Dict[str, Any]) -> ValidationResult:
        """Validate MultiPolygon geometry."""
        coordinates = geometry["coordinates"]
        issues = []
        warnings = []
        valid_polygons = []
        
        for i, polygon_coords in enumerate(coordinates):
            polygon_geom = {"type": "Polygon", "coordinates": polygon_coords}
            result = self._validate_polygon(polygon_geom)
            
            if result.is_valid:
                valid_polygons.append(polygon_coords)
            else:
                issues.extend([f"Polygon {i}: {issue}" for issue in result.issues])
            
            warnings.extend([f"Polygon {i}: {warning}" for warning in result.warnings])
        
        corrected_geometry = None
        if valid_polygons and len(valid_polygons) != len(coordinates):
            corrected_geometry = {
                "type": "MultiPolygon",
                "coordinates": valid_polygons
            }
        
        return ValidationResult(
            is_valid=len(issues) == 0 and len(valid_polygons) > 0,
            geometry=geometry,
            corrected_geometry=corrected_geometry,
            issues=issues,
            warnings=warnings
        )
    
    def _remove_duplicate_coordinates(
        self, 
        coordinates: List[List[float]], 
        preserve_closure: bool = False
    ) -> List[List[float]]:
        """Remove duplicate consecutive coordinates."""
        if not coordinates:
            return coordinates
        
        cleaned = [coordinates[0]]
        
        for coord in coordinates[1:]:
            if coord != cleaned[-1]:
                cleaned.append(coord)
        
        # For polygons, ensure closure is preserved
        if preserve_closure and len(cleaned) > 1 and cleaned[0] != cleaned[-1]:
            cleaned.append(cleaned[0])
        
        return cleaned
    
    def convert_coordinate_system(
        self, 
        geometry: Dict[str, Any], 
        from_crs: str = "EPSG:4326", 
        to_crs: str = "EPSG:4326"
    ) -> ValidationResult:
        """
        Convert geometry between coordinate systems.
        
        Args:
            geometry: GeoJSON geometry
            from_crs: Source coordinate reference system
            to_crs: Target coordinate reference system
            
        Returns:
            ValidationResult with converted geometry
        """
        if not SHAPELY_AVAILABLE:
            return ValidationResult(
                is_valid=False,
                geometry=geometry,
                issues=["Shapely not available for coordinate system conversion"]
            )
        
        if from_crs == to_crs:
            return ValidationResult(is_valid=True, geometry=geometry)
        
        try:
            # Create transformer
            transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
            
            # Convert based on geometry type
            geom_type = geometry.get("type")
            coordinates = geometry.get("coordinates")
            
            if geom_type == "Point":
                lon, lat = coordinates[0], coordinates[1]
                new_x, new_y = transformer.transform(lon, lat)
                converted_coords = [new_x, new_y]
                
            elif geom_type in ["LineString", "MultiPoint"]:
                converted_coords = [
                    list(transformer.transform(coord[0], coord[1])) 
                    for coord in coordinates
                ]
                
            elif geom_type == "Polygon":
                converted_coords = []
                for ring in coordinates:
                    converted_ring = [
                        list(transformer.transform(coord[0], coord[1])) 
                        for coord in ring
                    ]
                    converted_coords.append(converted_ring)
                    
            elif geom_type in ["MultiLineString", "MultiPolygon"]:
                converted_coords = []
                for sub_geom in coordinates:
                    if geom_type == "MultiLineString":
                        converted_sub = [
                            list(transformer.transform(coord[0], coord[1])) 
                            for coord in sub_geom
                        ]
                    else:  # MultiPolygon
                        converted_sub = []
                        for ring in sub_geom:
                            converted_ring = [
                                list(transformer.transform(coord[0], coord[1])) 
                                for coord in ring
                            ]
                            converted_sub.append(converted_ring)
                    converted_coords.append(converted_sub)
            else:
                return ValidationResult(
                    is_valid=False,
                    geometry=geometry,
                    issues=[f"Coordinate conversion not supported for {geom_type}"]
                )
            
            converted_geometry = {
                "type": geom_type,
                "coordinates": converted_coords
            }
            
            return ValidationResult(
                is_valid=True,
                geometry=geometry,
                corrected_geometry=converted_geometry,
                warnings=[f"Converted from {from_crs} to {to_crs}"]
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                geometry=geometry,
                issues=[f"Coordinate conversion failed: {str(e)}"]
            )
    
    def calculate_bounds(self, geometry: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculate bounding box of geometry.
        
        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat) or None if invalid
        """
        try:
            coordinates = geometry.get("coordinates")
            if not coordinates:
                return None
            
            # Flatten all coordinates
            all_coords = self._flatten_coordinates(coordinates)
            
            if not all_coords:
                return None
            
            lons = [coord[0] for coord in all_coords]
            lats = [coord[1] for coord in all_coords]
            
            return (min(lons), min(lats), max(lons), max(lats))
            
        except Exception:
            return None
    
    def _flatten_coordinates(self, coordinates) -> List[List[float]]:
        """Recursively flatten nested coordinate arrays."""
        if not coordinates:
            return []
        
        # Check if this is a coordinate pair (list of numbers)
        if isinstance(coordinates[0], (int, float)):
            return [coordinates]
        
        flattened = []
        for item in coordinates:
            flattened.extend(self._flatten_coordinates(item))
        
        return flattened