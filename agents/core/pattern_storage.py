"""
A2A World Platform - Pattern Storage Operations

Database operations for storing and retrieving discovered patterns.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2 import functions as geo_func

from agents.core.database_models import (
    Pattern, PatternComponent, ClusteringResult, SpatialAnalysis, 
    PatternValidation, SacredSite
)
from database.connection import get_db_session


class PatternStorage:
    """
    Database operations for pattern discovery storage and retrieval.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def store_pattern(self, pattern_data: Dict[str, Any], agent_id: str) -> str:
        """
        Store a discovered pattern in the database.
        
        Args:
            pattern_data: Pattern discovery results from agent
            agent_id: ID of the discovering agent
            
        Returns:
            Pattern ID if successful, empty string if failed
        """
        try:
            with get_db_session() as db:
                # Create main pattern record
                pattern = Pattern(
                    name=pattern_data.get("name", f"Pattern_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
                    description=pattern_data.get("description", "Auto-discovered spatial pattern"),
                    pattern_type=pattern_data.get("pattern_type", "spatial_clustering"),
                    confidence_score=float(pattern_data.get("confidence_score", 0.0)),
                    statistical_significance=pattern_data.get("statistical_significance"),
                    effect_size=pattern_data.get("effect_size"),
                    sample_size=pattern_data.get("sample_size"),
                    algorithm_used=pattern_data.get("algorithm", "hdbscan"),
                    algorithm_version=pattern_data.get("algorithm_version", "0.8.33"),
                    parameters=pattern_data.get("parameters", {}),
                    discovered_by_agent=agent_id,
                    metadata=pattern_data.get("metadata", {})
                )
                
                db.add(pattern)
                db.flush()  # Get the pattern ID
                
                # Store clustering results
                clustering_result = pattern_data.get("clustering_result", {})
                if clustering_result:
                    cluster_record = ClusteringResult(
                        pattern_id=pattern.id,
                        clustering_algorithm=clustering_result.get("algorithm", "hdbscan"),
                        num_clusters=clustering_result.get("n_clusters", 0),
                        cluster_centers=clustering_result.get("cluster_centers"),
                        cluster_labels=clustering_result.get("cluster_labels", []),
                        outliers_detected=clustering_result.get("n_noise", 0),
                        algorithm_parameters=clustering_result.get("cluster_info", {}),
                        data_dimensions=len(clustering_result.get("features_used", []))
                    )
                    db.add(cluster_record)
                
                # Store pattern components if available
                patterns_list = pattern_data.get("patterns", [])
                for discovered_pattern in patterns_list:
                    cluster_info = discovered_pattern.get("cluster_info", {})
                    points = cluster_info.get("points", [])
                    
                    for idx, point in enumerate(points):
                        component = PatternComponent(
                            pattern_id=pattern.id,
                            component_type="sacred_site",  # Default type
                            component_id=uuid.uuid4(),  # Would be actual site ID in production
                            relevance_score=1.0 / len(points),  # Simple relevance scoring
                            component_role="member",
                            distance_to_center=self._calculate_distance_to_center(point, cluster_info),
                            contribution_weight=1.0,
                            metadata={"point_data": point, "cluster_index": idx}
                        )
                        db.add(component)
                
                db.commit()
                self.logger.info(f"Stored pattern {pattern.id} with {len(patterns_list)} sub-patterns")
                return str(pattern.id)
                
        except Exception as e:
            self.logger.error(f"Failed to store pattern: {e}")
            return ""
    
    async def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific pattern by ID.
        
        Args:
            pattern_id: Pattern UUID
            
        Returns:
            Pattern data dictionary or None if not found
        """
        try:
            with get_db_session() as db:
                pattern = db.query(Pattern).filter(Pattern.id == pattern_id).first()
                
                if not pattern:
                    return None
                
                # Build comprehensive pattern data
                pattern_data = {
                    "id": str(pattern.id),
                    "name": pattern.name,
                    "description": pattern.description,
                    "pattern_type": pattern.pattern_type,
                    "confidence_score": float(pattern.confidence_score) if pattern.confidence_score else 0.0,
                    "statistical_significance": float(pattern.statistical_significance) if pattern.statistical_significance else None,
                    "algorithm_used": pattern.algorithm_used,
                    "discovered_by_agent": pattern.discovered_by_agent,
                    "discovery_timestamp": pattern.discovery_timestamp.isoformat() if pattern.discovery_timestamp else None,
                    "validation_status": pattern.validation_status,
                    "metadata": pattern.metadata or {}
                }
                
                # Add clustering results
                clustering_results = db.query(ClusteringResult).filter(
                    ClusteringResult.pattern_id == pattern.id
                ).all()
                
                pattern_data["clustering_results"] = []
                for result in clustering_results:
                    pattern_data["clustering_results"].append({
                        "algorithm": result.clustering_algorithm,
                        "num_clusters": result.num_clusters,
                        "cluster_centers": result.cluster_centers,
                        "cluster_labels": result.cluster_labels,
                        "outliers_detected": result.outliers_detected,
                        "parameters": result.algorithm_parameters
                    })
                
                # Add pattern components
                components = db.query(PatternComponent).filter(
                    PatternComponent.pattern_id == pattern.id
                ).all()
                
                pattern_data["components"] = []
                for component in components:
                    pattern_data["components"].append({
                        "id": str(component.id),
                        "type": component.component_type,
                        "relevance_score": float(component.relevance_score) if component.relevance_score else 0.0,
                        "role": component.component_role,
                        "distance_to_center": float(component.distance_to_center) if component.distance_to_center else None,
                        "metadata": component.metadata or {}
                    })
                
                return pattern_data
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve pattern {pattern_id}: {e}")
            return None
    
    async def list_patterns(self, 
                          limit: int = 50, 
                          offset: int = 0,
                          pattern_type: Optional[str] = None,
                          min_confidence: float = 0.0,
                          validation_status: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List discovered patterns with filtering options.
        
        Args:
            limit: Maximum number of patterns to return
            offset: Number of patterns to skip
            pattern_type: Filter by pattern type
            min_confidence: Minimum confidence score
            validation_status: Filter by validation status
            
        Returns:
            Tuple of (patterns list, total count)
        """
        try:
            with get_db_session() as db:
                query = db.query(Pattern)
                
                # Apply filters
                if pattern_type:
                    query = query.filter(Pattern.pattern_type == pattern_type)
                
                if min_confidence > 0:
                    query = query.filter(Pattern.confidence_score >= min_confidence)
                
                if validation_status:
                    query = query.filter(Pattern.validation_status == validation_status)
                
                # Get total count
                total_count = query.count()
                
                # Apply pagination and ordering
                patterns = query.order_by(Pattern.discovery_timestamp.desc()).offset(offset).limit(limit).all()
                
                pattern_list = []
                for pattern in patterns:
                    pattern_data = {
                        "id": str(pattern.id),
                        "name": pattern.name,
                        "description": pattern.description,
                        "pattern_type": pattern.pattern_type,
                        "confidence_score": float(pattern.confidence_score) if pattern.confidence_score else 0.0,
                        "algorithm_used": pattern.algorithm_used,
                        "discovered_by_agent": pattern.discovered_by_agent,
                        "discovery_timestamp": pattern.discovery_timestamp.isoformat() if pattern.discovery_timestamp else None,
                        "validation_status": pattern.validation_status,
                        "sample_size": pattern.sample_size
                    }
                    pattern_list.append(pattern_data)
                
                return pattern_list, total_count
                
        except Exception as e:
            self.logger.error(f"Failed to list patterns: {e}")
            return [], 0
    
    async def get_sacred_sites(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve sacred sites for pattern discovery.
        
        Args:
            limit: Maximum number of sites to return
            
        Returns:
            List of sacred site data
        """
        try:
            with get_db_session() as db:
                sites = db.query(SacredSite).limit(limit).all()
                
                site_list = []
                for site in sites:
                    site_data = {
                        "id": str(site.id),
                        "name": site.name,
                        "description": site.description,
                        "site_type": site.site_type,
                        "latitude": float(site.latitude) if site.latitude else None,
                        "longitude": float(site.longitude) if site.longitude else None,
                        "elevation": float(site.elevation) if site.elevation else None,
                        "cultural_context": site.cultural_context,
                        "significance_level": site.significance_level,
                        "metadata": site.metadata or {}
                    }
                    site_list.append(site_data)
                
                return site_list
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve sacred sites: {e}")
            return []
    
    async def create_sample_sacred_sites(self, count: int = 50) -> int:
        """
        Create sample sacred sites for testing pattern discovery.
        
        Args:
            count: Number of sample sites to create
            
        Returns:
            Number of sites created
        """
        try:
            # Sample sacred sites data with clustering patterns
            sample_sites = []
            
            # Cluster 1: Stonehenge area (UK)
            stonehenge_lat, stonehenge_lon = 51.1789, -1.8262
            for i in range(8):
                lat_offset = (i - 4) * 0.01
                lon_offset = (i % 3 - 1) * 0.01
                sample_sites.append({
                    "name": f"Ancient Stone Circle {i+1}",
                    "description": f"Neolithic stone circle site {i+1}",
                    "site_type": "stone_circle",
                    "latitude": stonehenge_lat + lat_offset,
                    "longitude": stonehenge_lon + lon_offset,
                    "cultural_context": "Neolithic Britain",
                    "significance_level": 4
                })
            
            # Cluster 2: Egyptian pyramids area
            giza_lat, giza_lon = 29.9792, 31.1342
            for i in range(6):
                lat_offset = (i - 3) * 0.005
                lon_offset = (i % 2) * 0.005
                sample_sites.append({
                    "name": f"Pyramid Complex {i+1}",
                    "description": f"Ancient Egyptian pyramid site {i+1}",
                    "site_type": "pyramid",
                    "latitude": giza_lat + lat_offset,
                    "longitude": giza_lon + lon_offset,
                    "cultural_context": "Ancient Egypt",
                    "significance_level": 5
                })
            
            # Cluster 3: Machu Picchu area (Peru)
            machu_lat, machu_lon = -13.1631, -72.5450
            for i in range(5):
                lat_offset = (i - 2) * 0.008
                lon_offset = (i % 2) * 0.008
                sample_sites.append({
                    "name": f"Inca Sacred Site {i+1}",
                    "description": f"Ancient Inca ceremonial site {i+1}",
                    "site_type": "temple",
                    "latitude": machu_lat + lat_offset,
                    "longitude": machu_lon + lon_offset,
                    "cultural_context": "Inca Empire",
                    "significance_level": 5
                })
            
            # Scattered sites for contrast
            import random
            random.seed(42)  # Reproducible results
            for i in range(31):  # Fill up to count
                sample_sites.append({
                    "name": f"Sacred Site {i+20}",
                    "description": f"Sacred site with historical significance {i+20}",
                    "site_type": random.choice(["temple", "burial_ground", "stone_circle", "sacred_grove"]),
                    "latitude": random.uniform(-60, 70),  # Global distribution
                    "longitude": random.uniform(-180, 180),
                    "cultural_context": random.choice(["Ancient", "Medieval", "Prehistoric", "Indigenous"]),
                    "significance_level": random.randint(2, 4)
                })
            
            # Store in database
            with get_db_session() as db:
                created_count = 0
                for site_data in sample_sites[:count]:
                    # Check if site already exists (by name)
                    existing = db.query(SacredSite).filter(SacredSite.name == site_data["name"]).first()
                    if existing:
                        continue
                    
                    site = SacredSite(
                        name=site_data["name"],
                        description=site_data["description"],
                        site_type=site_data["site_type"],
                        latitude=site_data["latitude"],
                        longitude=site_data["longitude"],
                        elevation=random.uniform(0, 2000) if random.random() > 0.5 else None,
                        cultural_context=site_data["cultural_context"],
                        significance_level=site_data["significance_level"],
                        verification_status="sample_data",
                        data_source="A2A Pattern Discovery Test Data"
                    )
                    
                    # Set PostGIS point geometry
                    site.location = f"POINT({site_data['longitude']} {site_data['latitude']})"
                    
                    db.add(site)
                    created_count += 1
                
                db.commit()
                self.logger.info(f"Created {created_count} sample sacred sites")
                return created_count
                
        except Exception as e:
            self.logger.error(f"Failed to create sample sacred sites: {e}")
            return 0
    
    def _calculate_distance_to_center(self, point: Dict[str, Any], cluster_info: Dict[str, Any]) -> float:
        """Calculate distance from point to cluster center."""
        try:
            centroid = cluster_info.get("centroid", {})
            if not centroid or "latitude" not in point or "longitude" not in point:
                return 0.0
            
            # Simple Euclidean distance (for more accuracy, use Haversine)
            lat_diff = point["latitude"] - centroid["latitude"]
            lon_diff = point["longitude"] - centroid["longitude"]
            distance = (lat_diff**2 + lon_diff**2)**0.5
            
            # Convert to approximate meters (rough conversion)
            return distance * 111320  # 1 degree ≈ 111.32 km
            
        except Exception:
            return 0.0
    
    async def validate_pattern(self, pattern_id: str, validator_id: str, 
                             validation_result: str, validation_score: float,
                             notes: Optional[str] = None) -> bool:
        """
        Add validation result for a pattern.
        
        Args:
            pattern_id: Pattern UUID
            validator_id: ID of validator (agent or human)
            validation_result: 'approved', 'rejected', 'needs_revision', 'inconclusive'
            validation_score: Score between 0.0 and 1.0
            notes: Optional validation notes
            
        Returns:
            True if validation was recorded successfully
        """
        try:
            with get_db_session() as db:
                validation = PatternValidation(
                    pattern_id=pattern_id,
                    validation_type="automated_validation",
                    validator_type="agent",
                    validator_id=validator_id,
                    validation_result=validation_result,
                    validation_score=validation_score,
                    validation_notes=notes,
                    confidence_level=validation_score
                )
                
                db.add(validation)
                db.commit()
                
                self.logger.info(f"Added validation for pattern {pattern_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to validate pattern {pattern_id}: {e}")
            return False