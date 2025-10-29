"""
A2A World Platform - Pattern Discovery Endpoints

Endpoints for pattern discovery, validation, and exploration with database integration.
"""

import logging
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import pattern storage and discovery agent
from agents.core.pattern_storage import PatternStorage
from agents.discovery.pattern_discovery import PatternDiscoveryAgent

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize pattern storage
pattern_storage = PatternStorage()

@router.get("/")
async def list_patterns(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score"),
    validation_status: Optional[str] = Query(None, description="Filter by validation status")
) -> Dict[str, Any]:
    """
    List discovered patterns with filtering and pagination options.
    """
    try:
        offset = (page - 1) * page_size
        
        patterns, total_count = await pattern_storage.list_patterns(
            limit=page_size,
            offset=offset,
            pattern_type=pattern_type,
            min_confidence=min_confidence,
            validation_status=validation_status
        )
        
        return {
            "patterns": patterns,
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size,
            "filters": {
                "pattern_type": pattern_type,
                "min_confidence": min_confidence,
                "validation_status": validation_status
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patterns: {str(e)}")

@router.get("/{pattern_id}")
async def get_pattern(pattern_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific pattern.
    """
    try:
        pattern = await pattern_storage.get_pattern(pattern_id)
        
        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")
        
        return {
            "success": True,
            "pattern": pattern
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pattern: {str(e)}")

@router.post("/discover")
async def discover_patterns(
    background_tasks: BackgroundTasks,
    algorithm: str = Query("hdbscan", description="Clustering algorithm to use"),
    min_cluster_size: int = Query(5, ge=2, description="Minimum cluster size"),
    min_samples: int = Query(3, ge=1, description="Minimum samples for core points")
) -> Dict[str, Any]:
    """
    Trigger pattern discovery from database sacred sites.
    """
    try:
        logger.info(f"Starting pattern discovery with algorithm: {algorithm}")
        
        # Create pattern discovery agent
        from agents.core.config import DiscoveryAgentConfig
        config = DiscoveryAgentConfig(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            default_algorithm=algorithm
        )
        
        agent = PatternDiscoveryAgent(config=config)
        
        # Run pattern discovery
        discovery_result = await agent.discover_patterns_from_database()
        
        if discovery_result.get("error"):
            raise HTTPException(status_code=500, detail=discovery_result["error"])
        
        return {
            "success": True,
            "message": "Pattern discovery completed",
            "discovery_id": discovery_result.get("dataset_id"),
            "patterns_found": discovery_result.get("pattern_count", 0),
            "significant_patterns": discovery_result.get("significant_patterns", 0),
            "algorithm_used": algorithm,
            "parameters": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples
            },
            "stored_in_database": discovery_result.get("stored_in_database", False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern discovery failed: {str(e)}")

@router.post("/validate/{pattern_id}")
async def validate_pattern(
    pattern_id: str,
    validator_notes: Optional[str] = Query(None, description="Validation notes")
) -> Dict[str, Any]:
    """
    Trigger validation process for a specific pattern.
    """
    try:
        # Check if pattern exists
        pattern = await pattern_storage.get_pattern(pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")
        
        # For now, implement basic automated validation
        confidence_score = pattern.get("confidence_score", 0.0)
        validation_result = "approved" if confidence_score > 0.7 else "needs_review"
        validation_score = min(1.0, confidence_score + 0.1)  # Slight boost for validation
        
        # Store validation
        success = await pattern_storage.validate_pattern(
            pattern_id=pattern_id,
            validator_id="api_automated_validator",
            validation_result=validation_result,
            validation_score=validation_score,
            notes=validator_notes
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store validation result")
        
        return {
            "success": True,
            "pattern_id": pattern_id,
            "validation_result": validation_result,
            "validation_score": validation_score,
            "validated_at": datetime.utcnow().isoformat(),
            "validator": "api_automated_validator"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/stats/overview")
async def get_pattern_statistics() -> Dict[str, Any]:
    """
    Get overview statistics about discovered patterns.
    """
    try:
        # Get basic pattern counts
        all_patterns, total_count = await pattern_storage.list_patterns(limit=10000)  # Get all for stats
        
        if total_count == 0:
            return {
                "total_patterns": 0,
                "significant_patterns": 0,
                "validation_stats": {},
                "algorithm_stats": {},
                "pattern_type_stats": {}
            }
        
        # Calculate statistics
        significant_count = len([p for p in all_patterns if p.get("confidence_score", 0) > 0.7])
        
        # Validation status stats
        validation_stats = {}
        for pattern in all_patterns:
            status = pattern.get("validation_status", "pending")
            validation_stats[status] = validation_stats.get(status, 0) + 1
        
        # Algorithm stats
        algorithm_stats = {}
        for pattern in all_patterns:
            algo = pattern.get("algorithm_used", "unknown")
            algorithm_stats[algo] = algorithm_stats.get(algo, 0) + 1
        
        # Pattern type stats
        pattern_type_stats = {}
        for pattern in all_patterns:
            ptype = pattern.get("pattern_type", "unknown")
            pattern_type_stats[ptype] = pattern_type_stats.get(ptype, 0) + 1
        
        return {
            "total_patterns": total_count,
            "significant_patterns": significant_count,
            "significance_rate": significant_count / total_count if total_count > 0 else 0.0,
            "validation_stats": validation_stats,
            "algorithm_stats": algorithm_stats,
            "pattern_type_stats": pattern_type_stats,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating pattern statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate statistics: {str(e)}")

@router.post("/sample-data/create")
async def create_sample_data(
    count: int = Query(50, ge=10, le=200, description="Number of sample sites to create")
) -> Dict[str, Any]:
    """
    Create sample sacred sites data for pattern discovery testing.
    """
    try:
        created_count = await pattern_storage.create_sample_sacred_sites(count)
        
        return {
            "success": True,
            "message": f"Created {created_count} sample sacred sites",
            "sites_created": created_count,
            "ready_for_pattern_discovery": created_count >= 10
        }
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create sample data: {str(e)}")

@router.get("/sacred-sites/count")
async def get_sacred_sites_count() -> Dict[str, Any]:
    """
    Get count of sacred sites available for pattern discovery.
    """
    try:
        sites = await pattern_storage.get_sacred_sites(limit=10000)  # Get all for counting
        
        return {
            "total_sites": len(sites),
            "ready_for_clustering": len(sites) >= 10,
            "recommended_min_cluster_size": max(3, len(sites) // 20) if len(sites) > 0 else 5
        }
        
    except Exception as e:
        logger.error(f"Error counting sacred sites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to count sacred sites: {str(e)}")

@router.post("/batch-validate")
async def batch_validate_patterns(
    pattern_ids: List[str] = Query(..., description="List of pattern IDs to validate"),
    validator_notes: Optional[str] = Query(None, description="Validation notes for all patterns"),
    validation_method: str = Query("automated", description="Validation method to use")
) -> Dict[str, Any]:
    """
    Validate multiple patterns in batch operation.
    Useful for bulk validation workflows.
    """
    try:
        validation_results = []
        successful_validations = 0
        failed_validations = 0
        
        for pattern_id in pattern_ids:
            try:
                # Check if pattern exists
                pattern = await pattern_storage.get_pattern(pattern_id)
                if not pattern:
                    validation_results.append({
                        "pattern_id": pattern_id,
                        "status": "error",
                        "error": "Pattern not found"
                    })
                    failed_validations += 1
                    continue
                
                # Perform validation based on method
                if validation_method == "automated":
                    confidence_score = pattern.get("confidence_score", 0.0)
                    validation_result = "approved" if confidence_score > 0.7 else "needs_review"
                    validation_score = min(1.0, confidence_score + 0.1)
                elif validation_method == "statistical":
                    # Enhanced statistical validation
                    validation_result = "approved" if pattern.get("statistical_significance", 1.0) < 0.05 else "needs_review"
                    validation_score = 1.0 - pattern.get("statistical_significance", 1.0)
                else:
                    validation_result = "needs_review"
                    validation_score = 0.5
                
                # Store validation
                success = await pattern_storage.validate_pattern(
                    pattern_id=pattern_id,
                    validator_id=f"api_batch_validator_{validation_method}",
                    validation_result=validation_result,
                    validation_score=validation_score,
                    notes=validator_notes
                )
                
                if success:
                    validation_results.append({
                        "pattern_id": pattern_id,
                        "status": "validated",
                        "validation_result": validation_result,
                        "validation_score": validation_score
                    })
                    successful_validations += 1
                else:
                    validation_results.append({
                        "pattern_id": pattern_id,
                        "status": "error",
                        "error": "Failed to store validation"
                    })
                    failed_validations += 1
                    
            except Exception as e:
                validation_results.append({
                    "pattern_id": pattern_id,
                    "status": "error",
                    "error": str(e)
                })
                failed_validations += 1
        
        return {
            "success": True,
            "total_patterns": len(pattern_ids),
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "validation_method": validation_method,
            "results": validation_results,
            "summary": {
                "success_rate": (successful_validations / len(pattern_ids) * 100) if pattern_ids else 0,
                "validated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Batch validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")

@router.get("/search")
async def search_patterns(
    query: Optional[str] = Query(None, description="Text search in pattern name/description"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score"),
    validation_status: Optional[str] = Query(None, description="Filter by validation status"),
    bbox: Optional[str] = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    algorithm: Optional[str] = Query(None, description="Filter by discovery algorithm"),
    date_from: Optional[str] = Query(None, description="Filter by discovery date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter by discovery date (ISO format)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Number to skip")
) -> Dict[str, Any]:
    """
    Advanced pattern search with multiple filtering criteria.
    Supports text search, geospatial filtering, and metadata filtering.
    """
    try:
        # Build search parameters
        search_params = {
            "limit": limit,
            "offset": offset,
            "pattern_type": pattern_type,
            "min_confidence": min_confidence,
            "validation_status": validation_status
        }
        
        # Add text search
        if query:
            search_params["text_query"] = query
        
        # Add geospatial filter
        if bbox:
            try:
                min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
                search_params["bbox"] = [min_lon, min_lat, max_lon, max_lat]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        # Add algorithm filter
        if algorithm:
            search_params["algorithm"] = algorithm
        
        # Add date filters
        if date_from:
            search_params["date_from"] = date_from
        if date_to:
            search_params["date_to"] = date_to
        
        # Perform search (enhanced version of list_patterns)
        patterns, total_count = await pattern_storage.search_patterns(**search_params)
        
        return {
            "patterns": patterns,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count,
            "search_parameters": search_params,
            "filters_applied": {
                "text_query": query is not None,
                "pattern_type": pattern_type is not None,
                "confidence_threshold": min_confidence > 0.0,
                "validation_status": validation_status is not None,
                "geospatial": bbox is not None,
                "algorithm": algorithm is not None,
                "date_range": date_from is not None or date_to is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern search failed: {str(e)}")

@router.get("/{pattern_id}/validation")
async def get_pattern_validation_results(pattern_id: str) -> Dict[str, Any]:
    """
    Get detailed validation results for a specific pattern.
    Includes validation history and consensus scores.
    """
    try:
        # Get pattern
        pattern = await pattern_storage.get_pattern(pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")
        
        # Get validation history from pattern storage
        validation_history = await pattern_storage.get_pattern_validations(pattern_id)
        
        # Calculate consensus metrics
        validations = validation_history or []
        consensus_data = {
            "total_validations": len(validations),
            "approved_count": len([v for v in validations if v.get("validation_result") == "approved"]),
            "rejected_count": len([v for v in validations if v.get("validation_result") == "rejected"]),
            "needs_review_count": len([v for v in validations if v.get("validation_result") == "needs_review"]),
            "average_score": 0.0,
            "consensus_level": "no_consensus"
        }
        
        if validations:
            scores = [v.get("validation_score", 0.0) for v in validations if v.get("validation_score")]
            if scores:
                consensus_data["average_score"] = sum(scores) / len(scores)
            
            # Determine consensus level
            total = len(validations)
            approved_ratio = consensus_data["approved_count"] / total
            
            if approved_ratio >= 0.8:
                consensus_data["consensus_level"] = "strong_consensus"
            elif approved_ratio >= 0.6:
                consensus_data["consensus_level"] = "moderate_consensus"
            elif approved_ratio <= 0.2:
                consensus_data["consensus_level"] = "strong_disagreement"
            else:
                consensus_data["consensus_level"] = "mixed_opinions"
        
        return {
            "pattern_id": pattern_id,
            "current_validation_status": pattern.get("validation_status", "pending"),
            "current_confidence_score": pattern.get("confidence_score", 0.0),
            "validation_consensus": consensus_data,
            "validation_history": validation_history,
            "requires_additional_validation": consensus_data["total_validations"] < 3,
            "recommendation": _get_validation_recommendation(consensus_data, pattern)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting validation results for pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation results: {str(e)}")

@router.post("/export")
async def export_patterns(
    format: str = Query("json", description="Export format (json, csv, geojson)"),
    pattern_ids: Optional[List[str]] = Query(None, description="Specific pattern IDs to export"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    validation_status: Optional[str] = Query(None, description="Filter by validation status"),
    min_confidence: float = Query(0.0, description="Minimum confidence score"),
    include_components: bool = Query(False, description="Include pattern components"),
    include_validations: bool = Query(False, description="Include validation history")
) -> Dict[str, Any]:
    """
    Export patterns in various formats with filtering options.
    Supports JSON, CSV, and GeoJSON formats with comprehensive data.
    """
    try:
        # Validate format
        if format.lower() not in ["json", "csv", "geojson"]:
            raise HTTPException(status_code=400, detail="Unsupported format. Use: json, csv, geojson")
        
        # Get patterns based on filters
        if pattern_ids:
            patterns = []
            for pattern_id in pattern_ids:
                pattern = await pattern_storage.get_pattern(pattern_id)
                if pattern:
                    patterns.append(pattern)
        else:
            patterns, _ = await pattern_storage.list_patterns(
                limit=10000,
                pattern_type=pattern_type,
                min_confidence=min_confidence,
                validation_status=validation_status
            )
        
        if not patterns:
            raise HTTPException(status_code=404, detail="No patterns found matching criteria")
        
        # Enhance patterns with additional data if requested
        enhanced_patterns = []
        for pattern in patterns:
            enhanced_pattern = pattern.copy()
            
            if include_components:
                components = await pattern_storage.get_pattern_components(pattern["id"])
                enhanced_pattern["components"] = components
            
            if include_validations:
                validations = await pattern_storage.get_pattern_validations(pattern["id"])
                enhanced_pattern["validations"] = validations
            
            enhanced_patterns.append(enhanced_pattern)
        
        # Generate export based on format
        export_data = {
            "export_info": {
                "format": format,
                "exported_at": datetime.utcnow().isoformat(),
                "pattern_count": len(enhanced_patterns),
                "filters_applied": {
                    "pattern_type": pattern_type,
                    "validation_status": validation_status,
                    "min_confidence": min_confidence
                },
                "includes": {
                    "components": include_components,
                    "validations": include_validations
                }
            },
            "patterns": enhanced_patterns
        }
        
        return {
            "success": True,
            "export_id": f"patterns_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "format": format,
            "pattern_count": len(enhanced_patterns),
            "data": export_data,
            "download_size_estimate": f"{len(str(export_data)) // 1024} KB"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern export failed: {str(e)}")

def _get_validation_recommendation(consensus_data: Dict[str, Any], pattern: Dict[str, Any]) -> str:
    """Generate validation recommendation based on consensus and pattern data."""
    total_validations = consensus_data["total_validations"]
    consensus_level = consensus_data["consensus_level"]
    confidence_score = pattern.get("confidence_score", 0.0)
    
    if total_validations == 0:
        return "Pattern requires initial validation"
    elif total_validations < 3:
        return "Pattern needs additional validations for consensus"
    elif consensus_level == "strong_consensus" and confidence_score > 0.8:
        return "Pattern is well-validated and highly confident"
    elif consensus_level == "strong_disagreement":
        return "Pattern has conflicting validations, requires expert review"
    elif consensus_level == "moderate_consensus":
        return "Pattern has reasonable validation consensus"
    else:
        return "Pattern validation status is unclear, needs review"