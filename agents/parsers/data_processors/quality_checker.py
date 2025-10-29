"""
A2A World Platform - Data Quality Checker

Assesses data quality and generates comprehensive reports for geospatial datasets.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import statistics

from .geometry_validator import GeometryValidator, ValidationResult


@dataclass
class QualityMetrics:
    """Data quality metrics."""
    total_records: int = 0
    valid_geometries: int = 0
    invalid_geometries: int = 0
    correctable_geometries: int = 0
    duplicate_records: int = 0
    missing_coordinates: int = 0
    missing_names: int = 0
    missing_descriptions: int = 0
    coordinate_precision_issues: int = 0
    geometry_complexity_score: float = 0.0
    completeness_score: float = 0.0
    validity_score: float = 0.0
    overall_quality_score: float = 0.0
    

@dataclass
class QualityIssue:
    """Individual quality issue."""
    record_id: Optional[str]
    record_index: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    description: str
    suggested_fix: Optional[str] = None
    field_name: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    dataset_name: str
    check_timestamp: datetime
    metrics: QualityMetrics
    issues: List[QualityIssue]
    statistics: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "check_timestamp": self.check_timestamp.isoformat(),
            "metrics": {
                "total_records": self.metrics.total_records,
                "valid_geometries": self.metrics.valid_geometries,
                "invalid_geometries": self.metrics.invalid_geometries,
                "correctable_geometries": self.metrics.correctable_geometries,
                "duplicate_records": self.metrics.duplicate_records,
                "missing_coordinates": self.metrics.missing_coordinates,
                "missing_names": self.metrics.missing_names,
                "missing_descriptions": self.metrics.missing_descriptions,
                "coordinate_precision_issues": self.metrics.coordinate_precision_issues,
                "geometry_complexity_score": self.metrics.geometry_complexity_score,
                "completeness_score": self.metrics.completeness_score,
                "validity_score": self.metrics.validity_score,
                "overall_quality_score": self.metrics.overall_quality_score
            },
            "issue_summary": {
                "total_issues": len(self.issues),
                "errors": len([i for i in self.issues if i.severity == 'error']),
                "warnings": len([i for i in self.issues if i.severity == 'warning']),
                "info": len([i for i in self.issues if i.severity == 'info'])
            },
            "issues": [
                {
                    "record_id": issue.record_id,
                    "record_index": issue.record_index,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description,
                    "suggested_fix": issue.suggested_fix,
                    "field_name": issue.field_name
                }
                for issue in self.issues
            ],
            "statistics": self.statistics,
            "recommendations": self.recommendations
        }


class QualityChecker:
    """
    Comprehensive data quality checker for geospatial datasets.
    
    Checks:
    - Geometry validity and correctness
    - Data completeness
    - Duplicate detection
    - Coordinate precision and accuracy
    - Metadata quality
    - Consistency across records
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.geometry_validator = GeometryValidator()
        
    def check_dataset_quality(
        self, 
        data: List[Dict[str, Any]], 
        dataset_name: str = "Unknown Dataset"
    ) -> QualityReport:
        """
        Perform comprehensive quality check on dataset.
        
        Args:
            data: List of feature records to check
            dataset_name: Name of the dataset for reporting
            
        Returns:
            QualityReport with detailed analysis
        """
        self.logger.info(f"Starting quality check for dataset: {dataset_name}")
        
        metrics = QualityMetrics()
        issues = []
        statistics = {}
        
        # Initialize metrics
        metrics.total_records = len(data)
        
        # Track various data for statistics
        geometry_types = []
        name_lengths = []
        description_lengths = []
        coordinate_counts = []
        unique_names = set()
        duplicate_signatures = []
        
        # Process each record
        for i, record in enumerate(data):
            record_issues = self._check_record_quality(record, i)
            issues.extend(record_issues)
            
            # Update metrics based on record
            self._update_metrics_from_record(record, metrics, record_issues)
            
            # Collect statistics
            if 'geometry' in record:
                geom_type = record['geometry'].get('type', 'Unknown')
                geometry_types.append(geom_type)
                
                # Count coordinates for complexity
                coord_count = self._count_coordinates(record['geometry'])
                coordinate_counts.append(coord_count)
            
            if 'name' in record and record['name']:
                name_lengths.append(len(str(record['name'])))
                unique_names.add(str(record['name']).strip().lower())
            
            if 'description' in record and record['description']:
                description_lengths.append(len(str(record['description'])))
            
            # Create signature for duplicate detection
            signature = self._create_record_signature(record)
            duplicate_signatures.append(signature)
        
        # Check for duplicates
        duplicate_count = self._count_duplicates(duplicate_signatures)
        metrics.duplicate_records = duplicate_count
        if duplicate_count > 0:
            issues.append(QualityIssue(
                record_id=None,
                record_index=-1,
                issue_type="duplicates",
                severity="warning",
                description=f"Found {duplicate_count} duplicate records",
                suggested_fix="Review and remove duplicate entries"
            ))
        
        # Calculate quality scores
        self._calculate_quality_scores(metrics)
        
        # Generate statistics
        statistics = self._generate_statistics(
            geometry_types, name_lengths, description_lengths, 
            coordinate_counts, unique_names
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, issues, statistics)
        
        report = QualityReport(
            dataset_name=dataset_name,
            check_timestamp=datetime.utcnow(),
            metrics=metrics,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
        
        self.logger.info(
            f"Quality check complete. Overall score: {metrics.overall_quality_score:.2f}/100"
        )
        
        return report
    
    def _check_record_quality(self, record: Dict[str, Any], index: int) -> List[QualityIssue]:
        """Check quality of individual record."""
        issues = []
        record_id = record.get('id') or record.get('name', f"record_{index}")
        
        # Check for required fields
        if not record.get('name'):
            issues.append(QualityIssue(
                record_id=record_id,
                record_index=index,
                issue_type="missing_name",
                severity="warning",
                description="Record missing name/title",
                field_name="name",
                suggested_fix="Provide a meaningful name or title"
            ))
        
        # Check geometry
        if 'geometry' not in record or not record['geometry']:
            issues.append(QualityIssue(
                record_id=record_id,
                record_index=index,
                issue_type="missing_geometry",
                severity="error",
                description="Record missing geometry data",
                field_name="geometry",
                suggested_fix="Add valid geometry coordinates"
            ))
        else:
            # Validate geometry
            validation_result = self.geometry_validator.validate_geometry(record['geometry'])
            if not validation_result.is_valid:
                severity = "error" if not validation_result.corrected_geometry else "warning"
                issues.append(QualityIssue(
                    record_id=record_id,
                    record_index=index,
                    issue_type="invalid_geometry",
                    severity=severity,
                    description=f"Geometry validation failed: {'; '.join(validation_result.issues)}",
                    field_name="geometry",
                    suggested_fix="Use corrected geometry" if validation_result.corrected_geometry else "Fix geometry issues"
                ))
            
            # Check for warnings
            for warning in validation_result.warnings:
                issues.append(QualityIssue(
                    record_id=record_id,
                    record_index=index,
                    issue_type="geometry_warning",
                    severity="info",
                    description=f"Geometry warning: {warning}",
                    field_name="geometry"
                ))
        
        # Check description quality
        description = record.get('description', '')
        if not description:
            issues.append(QualityIssue(
                record_id=record_id,
                record_index=index,
                issue_type="missing_description",
                severity="info",
                description="Record missing description",
                field_name="description",
                suggested_fix="Add descriptive information about the location"
            ))
        elif len(description.strip()) < 10:
            issues.append(QualityIssue(
                record_id=record_id,
                record_index=index,
                issue_type="short_description",
                severity="info",
                description="Description is very short",
                field_name="description",
                suggested_fix="Provide more detailed description"
            ))
        
        # Check coordinate precision
        if 'geometry' in record and record['geometry']:
            precision_issues = self._check_coordinate_precision(record['geometry'])
            for issue in precision_issues:
                issues.append(QualityIssue(
                    record_id=record_id,
                    record_index=index,
                    issue_type="coordinate_precision",
                    severity="info",
                    description=issue,
                    field_name="geometry"
                ))
        
        # Check for suspicious data patterns
        suspicious_patterns = self._check_suspicious_patterns(record)
        for pattern in suspicious_patterns:
            issues.append(QualityIssue(
                record_id=record_id,
                record_index=index,
                issue_type="suspicious_pattern",
                severity="warning",
                description=pattern,
                suggested_fix="Review and verify data accuracy"
            ))
        
        return issues
    
    def _update_metrics_from_record(
        self, 
        record: Dict[str, Any], 
        metrics: QualityMetrics, 
        issues: List[QualityIssue]
    ) -> None:
        """Update metrics based on record analysis."""
        # Count geometry validity
        if 'geometry' in record and record['geometry']:
            validation_result = self.geometry_validator.validate_geometry(record['geometry'])
            if validation_result.is_valid:
                metrics.valid_geometries += 1
            else:
                metrics.invalid_geometries += 1
                if validation_result.corrected_geometry:
                    metrics.correctable_geometries += 1
        else:
            metrics.missing_coordinates += 1
        
        # Count missing data
        if not record.get('name'):
            metrics.missing_names += 1
        
        if not record.get('description'):
            metrics.missing_descriptions += 1
        
        # Count precision issues
        precision_issues = [i for i in issues if i.issue_type == "coordinate_precision"]
        metrics.coordinate_precision_issues += len(precision_issues)
    
    def _calculate_quality_scores(self, metrics: QualityMetrics) -> None:
        """Calculate overall quality scores."""
        if metrics.total_records == 0:
            return
        
        # Validity score (0-100)
        total_with_geometry = metrics.total_records - metrics.missing_coordinates
        if total_with_geometry > 0:
            metrics.validity_score = (metrics.valid_geometries / total_with_geometry) * 100
        
        # Completeness score (0-100)
        completeness_factors = [
            (metrics.total_records - metrics.missing_names) / metrics.total_records,
            (metrics.total_records - metrics.missing_descriptions) / metrics.total_records,
            (metrics.total_records - metrics.missing_coordinates) / metrics.total_records
        ]
        metrics.completeness_score = statistics.mean(completeness_factors) * 100
        
        # Geometry complexity score (normalized)
        # This is a placeholder - would need actual complexity calculation
        metrics.geometry_complexity_score = 50.0  # Default medium complexity
        
        # Overall quality score (weighted average)
        weights = {
            'validity': 0.4,
            'completeness': 0.4,
            'complexity': 0.1,
            'duplicates': 0.1
        }
        
        duplicate_penalty = min((metrics.duplicate_records / metrics.total_records) * 100, 30)
        
        metrics.overall_quality_score = (
            weights['validity'] * metrics.validity_score +
            weights['completeness'] * metrics.completeness_score +
            weights['complexity'] * metrics.geometry_complexity_score -
            weights['duplicates'] * duplicate_penalty
        )
        
        # Ensure score is between 0 and 100
        metrics.overall_quality_score = max(0, min(100, metrics.overall_quality_score))
    
    def _generate_statistics(
        self,
        geometry_types: List[str],
        name_lengths: List[int],
        description_lengths: List[int],
        coordinate_counts: List[int],
        unique_names: Set[str]
    ) -> Dict[str, Any]:
        """Generate dataset statistics."""
        stats = {}
        
        # Geometry type distribution
        if geometry_types:
            stats['geometry_types'] = dict(Counter(geometry_types))
        
        # Name statistics
        if name_lengths:
            stats['name_statistics'] = {
                'avg_length': statistics.mean(name_lengths),
                'min_length': min(name_lengths),
                'max_length': max(name_lengths),
                'unique_names': len(unique_names)
            }
        
        # Description statistics
        if description_lengths:
            stats['description_statistics'] = {
                'avg_length': statistics.mean(description_lengths),
                'min_length': min(description_lengths),
                'max_length': max(description_lengths)
            }
        
        # Geometry complexity
        if coordinate_counts:
            stats['geometry_complexity'] = {
                'avg_coordinates': statistics.mean(coordinate_counts),
                'min_coordinates': min(coordinate_counts),
                'max_coordinates': max(coordinate_counts)
            }
        
        return stats
    
    def _generate_recommendations(
        self,
        metrics: QualityMetrics,
        issues: List[QualityIssue],
        statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Geometry recommendations
        if metrics.invalid_geometries > 0:
            if metrics.correctable_geometries > 0:
                recommendations.append(
                    f"Fix {metrics.correctable_geometries} correctable geometry issues automatically"
                )
            
            uncorrectable = metrics.invalid_geometries - metrics.correctable_geometries
            if uncorrectable > 0:
                recommendations.append(
                    f"Manually review {uncorrectable} geometry records with unfixable issues"
                )
        
        # Completeness recommendations
        if metrics.missing_names > metrics.total_records * 0.1:
            recommendations.append("Consider adding names/titles to improve record identification")
        
        if metrics.missing_descriptions > metrics.total_records * 0.2:
            recommendations.append("Add descriptions to provide better context for locations")
        
        # Duplicate recommendations
        if metrics.duplicate_records > 0:
            recommendations.append("Remove or consolidate duplicate records to improve data quality")
        
        # Overall quality recommendations
        if metrics.overall_quality_score < 70:
            recommendations.append("Dataset quality is below recommended threshold - consider data cleanup")
        elif metrics.overall_quality_score < 85:
            recommendations.append("Dataset quality is good but has room for improvement")
        else:
            recommendations.append("Dataset quality is excellent")
        
        return recommendations
    
    def _count_coordinates(self, geometry: Dict[str, Any]) -> int:
        """Count total coordinates in geometry."""
        try:
            coordinates = geometry.get('coordinates', [])
            return len(self._flatten_coordinates(coordinates))
        except Exception:
            return 0
    
    def _flatten_coordinates(self, coordinates) -> List:
        """Flatten nested coordinate arrays."""
        if not coordinates:
            return []
        
        # Check if this is a coordinate pair
        if isinstance(coordinates[0], (int, float)):
            return [coordinates]
        
        flattened = []
        for item in coordinates:
            if isinstance(item[0], (int, float)):
                flattened.append(item)
            else:
                flattened.extend(self._flatten_coordinates(item))
        
        return flattened
    
    def _create_record_signature(self, record: Dict[str, Any]) -> str:
        """Create signature for duplicate detection."""
        # Use geometry coordinates and name for signature
        signature_parts = []
        
        if 'geometry' in record and record['geometry']:
            coords = record['geometry'].get('coordinates')
            if coords:
                signature_parts.append(str(coords))
        
        if 'name' in record and record['name']:
            signature_parts.append(str(record['name']).strip().lower())
        
        return "|".join(signature_parts)
    
    def _count_duplicates(self, signatures: List[str]) -> int:
        """Count duplicate records based on signatures."""
        signature_counts = Counter(signatures)
        duplicates = sum(count - 1 for count in signature_counts.values() if count > 1)
        return duplicates
    
    def _check_coordinate_precision(self, geometry: Dict[str, Any]) -> List[str]:
        """Check for coordinate precision issues."""
        issues = []
        
        try:
            coordinates = geometry.get('coordinates', [])
            flat_coords = self._flatten_coordinates(coordinates)
            
            for coord in flat_coords:
                if len(coord) >= 2:
                    lon, lat = coord[0], coord[1]
                    
                    # Check for suspiciously round numbers (possible low precision)
                    if lon == int(lon) and lat == int(lat):
                        issues.append("Coordinates appear to be rounded to integers (low precision)")
                        break
                    
                    # Check for excessive precision (more than ~11 decimal places)
                    lon_str = str(lon)
                    lat_str = str(lat)
                    
                    if '.' in lon_str and len(lon_str.split('.')[1]) > 8:
                        issues.append("Longitude has excessive decimal precision")
                    
                    if '.' in lat_str and len(lat_str.split('.')[1]) > 8:
                        issues.append("Latitude has excessive decimal precision")
        
        except Exception:
            issues.append("Could not analyze coordinate precision")
        
        return issues
    
    def _check_suspicious_patterns(self, record: Dict[str, Any]) -> List[str]:
        """Check for suspicious data patterns."""
        issues = []
        
        # Check for null island (0, 0) coordinates
        if 'geometry' in record and record['geometry']:
            coords = record['geometry'].get('coordinates')
            if coords and len(coords) >= 2:
                if coords[0] == 0 and coords[1] == 0:
                    issues.append("Coordinates at (0, 0) - possible data entry error")
        
        # Check for very short or generic names
        name = record.get('name', '')
        if name:
            if len(name.strip()) <= 2:
                issues.append("Name is unusually short")
            
            generic_names = ['point', 'location', 'place', 'site', 'marker', 'pin']
            if name.lower().strip() in generic_names:
                issues.append("Name appears to be generic placeholder")
        
        return issues