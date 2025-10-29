"""
A2A World Platform - Enhanced Statistical Validation Endpoints

API endpoints for comprehensive statistical validation including Moran's I analysis,
null hypothesis testing, significance classification, and validation reporting.
"""

import logging
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import uuid

# Import enhanced validation components
from agents.validation.enhanced_validation_agent import EnhancedValidationAgent
from agents.validation.statistical_validation import SignificanceLevel, PatternSignificance
from agents.core.pattern_storage import PatternStorage

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize components
pattern_storage = PatternStorage()

# Pydantic models for request/response validation
class ValidationRequest(BaseModel):
    pattern_id: str
    validation_methods: List[str] = Field(default=["full_statistical_suite"], description="Statistical validation methods to apply")
    store_results: bool = Field(default=True, description="Whether to store results in database")
    significance_level: float = Field(default=0.05, ge=0.001, le=0.1, description="Statistical significance threshold")

class BatchValidationRequest(BaseModel):
    pattern_ids: List[str] = Field(..., description="List of pattern IDs to validate")
    validation_methods: List[str] = Field(default=["comprehensive_morans_i", "csr_testing"], description="Validation methods")
    max_parallel: int = Field(default=4, ge=1, le=10, description="Maximum parallel validations")
    store_results: bool = Field(default=True, description="Store results in database")

class StatisticalAnalysisRequest(BaseModel):
    coordinates: List[List[float]] = Field(..., description="Array of [latitude, longitude] coordinates")
    values: List[float] = Field(..., description="Array of values for statistical analysis")
    analysis_methods: List[str] = Field(default=["comprehensive_morans_i", "csr_testing"], description="Analysis methods")
    significance_level: float = Field(default=0.05, description="Statistical significance threshold")

class ValidationConfigRequest(BaseModel):
    significance_levels: Dict[str, float] = Field(default={
        "very_high": 0.001,
        "high": 0.01,
        "moderate": 0.05,
        "low": 0.10
    }, description="Custom significance level thresholds")
    monte_carlo_iterations: int = Field(default=999, ge=99, le=9999, description="Monte Carlo iterations")
    bootstrap_iterations: int = Field(default=1000, ge=100, le=10000, description="Bootstrap iterations")

# Enhanced validation endpoints

@router.get("/")
async def list_validation_results(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    significance_classification: Optional[str] = Query(None, description="Filter by significance classification"),
    min_reliability_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum reliability score"),
    validation_success: Optional[bool] = Query(None, description="Filter by validation success")
) -> Dict[str, Any]:
    """
    List enhanced statistical validation results with filtering and pagination.
    """
    try:
        # This would integrate with the database to get validation results
        # For now, return a structured response that would come from the database
        
        return {
            "validation_results": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
            "filters": {
                "significance_classification": significance_classification,
                "min_reliability_score": min_reliability_score,
                "validation_success": validation_success
            },
            "available_methods": [
                "comprehensive_morans_i",
                "monte_carlo_validation",
                "bootstrap_validation",
                "csr_testing",
                "hotspot_analysis",
                "spatial_concentration",
                "pattern_significance",
                "full_statistical_suite"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing validation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list validation results: {str(e)}")

@router.get("/{pattern_id}")
async def get_validation_results(
    pattern_id: str,
    include_details: bool = Query(True, description="Include detailed statistical results"),
    include_reports: bool = Query(False, description="Include generated validation reports")
) -> Dict[str, Any]:
    """
    Get comprehensive statistical validation results for a specific pattern.
    """
    try:
        # Check if pattern exists
        pattern = await pattern_storage.get_pattern(pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")
        
        # This would fetch from enhanced_statistical_validations table
        # For now, create a structured response
        validation_results = {
            "pattern_id": pattern_id,
            "pattern_name": pattern.get("name", "Unknown Pattern"),
            "validation_status": "not_validated",
            "overall_significance_classification": "unknown",
            "reliability_score": 0.0,
            "total_statistical_tests": 0,
            "significant_tests": 0,
            "validation_timestamp": None,
            "performed_by_agent": None,
            "message": "Enhanced statistical validation not yet performed for this pattern"
        }
        
        if include_details:
            validation_results.update({
                "detailed_results": {
                    "morans_i_analysis": None,
                    "null_hypothesis_tests": None,
                    "spatial_statistics": None,
                    "significance_classification": None
                }
            })
        
        if include_reports:
            validation_results.update({
                "validation_reports": [],
                "dashboard_data": None
            })
        
        return {
            "success": True,
            "validation_results": validation_results,
            "recommendations": [
                "Run enhanced statistical validation to get comprehensive analysis",
                "Use POST /validation/{pattern_id} to trigger validation"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving validation results for pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve validation results: {str(e)}")

@router.post("/{pattern_id}")
async def validate_pattern_enhanced(
    pattern_id: str,
    background_tasks: BackgroundTasks,
    request: ValidationRequest = Body(...)
) -> Dict[str, Any]:
    """
    Trigger enhanced statistical validation for a specific pattern.
    """
    try:
        # Check if pattern exists
        pattern = await pattern_storage.get_pattern(pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")
        
        # Create enhanced validation agent
        validation_agent = EnhancedValidationAgent()
        
        # Get pattern data for validation
        pattern_data = await _prepare_pattern_data_for_validation(pattern_id)
        
        # Run enhanced validation
        logger.info(f"Starting enhanced validation for pattern {pattern_id}")
        
        validation_result = await validation_agent.validate_pattern_enhanced(
            pattern_id=pattern_id,
            pattern_data=pattern_data,
            validation_methods=request.validation_methods,
            store_results=request.store_results
        )
        
        # Process results
        if "error" in validation_result:
            raise HTTPException(status_code=500, detail=f"Validation failed: {validation_result['error']}")
        
        # Extract key metrics
        significance_classification = validation_result.get("significance_classification", {})
        enhanced_metrics = validation_result.get("enhanced_metrics", {})
        
        return {
            "success": True,
            "pattern_id": pattern_id,
            "validation_session_id": validation_result.get("validation_session_id"),
            "overall_significance": significance_classification.get("overall_classification", "unknown"),
            "reliability_score": significance_classification.get("reliability_score", 0.0),
            "statistical_tests_performed": len(validation_result.get("statistical_results", [])),
            "significant_tests": enhanced_metrics.get("significant_tests", 0),
            "highly_significant_tests": enhanced_metrics.get("highly_significant_tests", 0),
            "validation_timestamp": validation_result.get("validation_timestamp"),
            "processing_time_ms": enhanced_metrics.get("processing_time_ms"),
            "stored_in_database": validation_result.get("stored_validation_id") is not None,
            "recommendations": validation_result.get("recommendations", []),
            "detailed_results": validation_result if request.store_results else "Results not stored - set store_results=true to save"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced validation failed for pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced validation failed: {str(e)}")

@router.post("/batch")
async def batch_validate_patterns(
    background_tasks: BackgroundTasks,
    request: BatchValidationRequest = Body(...)
) -> Dict[str, Any]:
    """
    Validate multiple patterns using enhanced statistical validation in batch.
    """
    try:
        if len(request.pattern_ids) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 patterns per batch request")
        
        # Create enhanced validation agent
        validation_agent = EnhancedValidationAgent()
        
        logger.info(f"Starting batch validation for {len(request.pattern_ids)} patterns")
        
        # Run batch validation
        batch_result = await validation_agent.batch_validate_patterns(
            pattern_ids=request.pattern_ids,
            validation_methods=request.validation_methods,
            max_parallel=request.max_parallel
        )
        
        # Process results
        if "error" in batch_result:
            raise HTTPException(status_code=500, detail=f"Batch validation failed: {batch_result['error']}")
        
        batch_summary = batch_result.get("batch_summary", {})
        
        return {
            "success": True,
            "batch_id": batch_result.get("batch_id"),
            "total_patterns": batch_result.get("total_patterns", 0),
            "successful_validations": batch_summary.get("successful_validations", 0),
            "failed_validations": batch_summary.get("failed_validations", 0),
            "success_rate": batch_summary.get("success_rate", 0.0),
            "highly_significant_patterns": batch_summary.get("highly_significant_patterns", 0),
            "significance_rate": batch_summary.get("significance_rate", 0.0),
            "validation_timestamp": batch_result.get("timestamp"),
            "validation_methods": request.validation_methods,
            "pattern_results": batch_result.get("pattern_results", {}),
            "failed_patterns": batch_result.get("failed_patterns", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")

@router.post("/analyze")
async def statistical_analysis(
    request: StatisticalAnalysisRequest = Body(...)
) -> Dict[str, Any]:
    """
    Perform statistical analysis on provided coordinates and values.
    Direct analysis without requiring a stored pattern.
    """
    try:
        if len(request.coordinates) != len(request.values):
            raise HTTPException(status_code=400, detail="Coordinates and values arrays must have same length")
        
        if len(request.coordinates) < 3:
            raise HTTPException(status_code=400, detail="Minimum 3 data points required for statistical analysis")
        
        # Create enhanced validation agent
        validation_agent = EnhancedValidationAgent()
        
        # Convert coordinates to numpy arrays
        import numpy as np
        coordinates = np.array(request.coordinates)
        values = np.array(request.values)
        
        logger.info(f"Running statistical analysis on {len(coordinates)} data points")
        
        # Run comprehensive statistical analysis
        analysis_result = await validation_agent._run_comprehensive_statistical_analysis(
            coordinates, values, request.analysis_methods
        )
        
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=f"Statistical analysis failed: {analysis_result['error']}")
        
        return {
            "success": True,
            "analysis_id": str(uuid.uuid4()),
            "sample_size": len(coordinates),
            "analysis_methods": request.analysis_methods,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "statistical_results": analysis_result.get("statistical_results", []),
            "spatial_analysis": analysis_result.get("spatial_analysis", {}),
            "significance_summary": {
                "total_tests": len(analysis_result.get("statistical_results", [])),
                "significant_tests": len([r for r in analysis_result.get("statistical_results", []) if getattr(r, 'significant', False)]),
                "min_p_value": min([getattr(r, 'p_value', 1.0) for r in analysis_result.get("statistical_results", [])], default=1.0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Statistical analysis failed: {str(e)}")

@router.post("/configure")
async def configure_validation_parameters(
    request: ValidationConfigRequest = Body(...)
) -> Dict[str, Any]:
    """
    Configure statistical validation parameters for enhanced validation.
    """
    try:
        # Validate significance levels
        for level, threshold in request.significance_levels.items():
            if not (0.001 <= threshold <= 0.5):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid significance threshold for {level}: {threshold}. Must be between 0.001 and 0.5"
                )
        
        # This would update configuration in database or agent configuration
        config_data = {
            "significance_levels": request.significance_levels,
            "monte_carlo_iterations": request.monte_carlo_iterations,
            "bootstrap_iterations": request.bootstrap_iterations,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "message": "Validation parameters configured successfully",
            "configuration": config_data,
            "validation_impact": {
                "stricter_significance": request.significance_levels["high"] < 0.01,
                "high_precision_testing": request.monte_carlo_iterations > 999,
                "robust_bootstrapping": request.bootstrap_iterations > 1000
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@router.get("/statistics")
async def get_validation_statistics() -> Dict[str, Any]:
    """
    Get comprehensive statistical validation summary and performance metrics.
    """
    try:
        # This would query the validation performance metrics tables
        # For now, return structured mock data
        
        return {
            "validation_statistics": {
                "total_validations_performed": 0,
                "total_statistical_tests_executed": 0,
                "highly_significant_patterns": 0,
                "enhanced_significance_rate": 0.0,
                "avg_tests_per_validation": 0.0,
                "validation_success_rate": 0.0,
                "avg_processing_time_ms": 0,
                "last_updated": datetime.utcnow().isoformat()
            },
            "significance_distribution": {
                "very_high": 0,
                "high": 0,
                "moderate": 0,
                "low": 0,
                "not_significant": 0
            },
            "popular_methods": [
                {"method": "comprehensive_morans_i", "usage_count": 0},
                {"method": "csr_testing", "usage_count": 0},
                {"method": "full_statistical_suite", "usage_count": 0}
            ],
            "performance_metrics": {
                "cache_hit_rate": 0.0,
                "avg_reliability_score": 0.0,
                "statistical_framework_errors": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving validation statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve validation statistics: {str(e)}")

@router.get("/reports/{validation_id}")
async def get_validation_report(
    validation_id: str,
    report_type: str = Query("comprehensive", description="Type of report to generate"),
    format: str = Query("json", description="Report format (json, html, pdf)")
) -> Dict[str, Any]:
    """
    Generate and retrieve statistical validation report.
    """
    try:
        # Validate report parameters
        valid_report_types = ["comprehensive", "summary", "executive", "technical", "dashboard"]
        valid_formats = ["json", "html", "pdf", "csv"]
        
        if report_type not in valid_report_types:
            raise HTTPException(status_code=400, detail=f"Invalid report type. Use: {', '.join(valid_report_types)}")
        
        if format not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Invalid format. Use: {', '.join(valid_formats)}")
        
        # This would fetch validation results from database and generate report
        # For now, return structured mock report
        
        report_data = {
            "report_id": f"report_{validation_id}_{report_type}",
            "validation_id": validation_id,
            "report_type": report_type,
            "format": format,
            "generated_at": datetime.utcnow().isoformat(),
            "report_content": {
                "executive_summary": "Statistical validation report not yet available",
                "detailed_findings": {},
                "statistical_tables": {},
                "conclusions": [],
                "recommendations": [
                    "Complete enhanced statistical validation to generate comprehensive reports"
                ],
                "visualizations": {}
            },
            "metadata": {
                "report_version": "1.0",
                "generated_by": "enhanced_validation_api",
                "includes_visualizations": format in ["html", "pdf"]
            }
        }
        
        return {
            "success": True,
            "report": report_data,
            "download_info": {
                "format": format,
                "estimated_size": "N/A",
                "ready_for_download": False
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating validation report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate validation report: {str(e)}")

@router.get("/dashboard/data")
async def get_validation_dashboard_data(
    time_period: str = Query("30d", description="Time period for dashboard data"),
    include_charts: bool = Query(True, description="Include chart data for visualizations")
) -> Dict[str, Any]:
    """
    Get dashboard data for validation monitoring and analytics.
    """
    try:
        # Validate time period
        valid_periods = ["1d", "7d", "30d", "90d", "1y", "all"]
        if time_period not in valid_periods:
            raise HTTPException(status_code=400, detail=f"Invalid time period. Use: {', '.join(valid_periods)}")
        
        # This would query dashboard views and performance metrics
        # For now, return structured mock dashboard data
        
        dashboard_data = {
            "overview_metrics": {
                "total_validations": 0,
                "highly_significant_patterns": 0,
                "avg_reliability_score": 0.0,
                "validation_success_rate": 0.0,
                "active_validation_agents": 0
            },
            "significance_indicators": [],
            "recent_validations": [],
            "performance_trends": {
                "validation_rate_trend": "stable",
                "significance_rate_trend": "stable",
                "processing_time_trend": "stable"
            },
            "alerts": [
                {
                    "type": "info",
                    "message": "Enhanced statistical validation framework is ready",
                    "priority": "low"
                }
            ]
        }
        
        if include_charts:
            dashboard_data["charts"] = {
                "significance_distribution": {
                    "type": "pie",
                    "data": {"labels": [], "values": []},
                    "title": "Pattern Significance Distribution"
                },
                "validation_timeline": {
                    "type": "line",
                    "data": {"dates": [], "validations": []},
                    "title": "Validation Activity Over Time"
                },
                "reliability_scores": {
                    "type": "histogram",
                    "data": {"bins": [], "counts": []},
                    "title": "Reliability Score Distribution"
                }
            }
        
        return {
            "success": True,
            "time_period": time_period,
            "last_updated": datetime.utcnow().isoformat(),
            "dashboard_data": dashboard_data,
            "refresh_interval_seconds": 300
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")

@router.get("/methods")
async def get_available_validation_methods() -> Dict[str, Any]:
    """
    Get list of available statistical validation methods and their descriptions.
    """
    try:
        methods = {
            "comprehensive_morans_i": {
                "name": "Comprehensive Moran's I Analysis",
                "description": "Global and Local Moran's I spatial autocorrelation analysis with multiple weight methods",
                "output": ["Global Moran's I statistic", "Local Indicators of Spatial Association (LISA)", "Hot spots and cold spots"],
                "recommended_for": ["Spatial clustering patterns", "Geographic correlation analysis"],
                "statistical_tests": ["Global Moran's I", "Local Moran's I", "Monte Carlo significance testing"]
            },
            "monte_carlo_validation": {
                "name": "Monte Carlo Permutation Testing",
                "description": "Monte Carlo permutation tests for pattern significance assessment",
                "output": ["Permutation p-values", "Effect sizes", "Confidence intervals"],
                "recommended_for": ["Pattern significance validation", "Non-parametric testing"],
                "statistical_tests": ["Permutation test", "Bootstrap resampling"]
            },
            "bootstrap_validation": {
                "name": "Bootstrap Confidence Intervals",
                "description": "Bootstrap resampling for confidence interval estimation and pattern stability",
                "output": ["Bootstrap confidence intervals", "Bias estimates", "Standard errors"],
                "recommended_for": ["Confidence interval estimation", "Pattern stability assessment"],
                "statistical_tests": ["Bootstrap resampling", "Percentile method", "Bias-corrected intervals"]
            },
            "csr_testing": {
                "name": "Complete Spatial Randomness Testing",
                "description": "CSR testing using Ripley's K function and nearest neighbor analysis",
                "output": ["Ripley's K statistics", "L-function values", "Nearest neighbor ratios"],
                "recommended_for": ["Spatial randomness assessment", "Point pattern analysis"],
                "statistical_tests": ["Ripley's K test", "Nearest neighbor analysis", "Monte Carlo CSR test"]
            },
            "hotspot_analysis": {
                "name": "Getis-Ord Gi* Hotspot Analysis",
                "description": "Getis-Ord Gi* statistic for hot spot and cold spot identification",
                "output": ["Gi* statistics", "Hot spot locations", "Cold spot locations", "Statistical significance"],
                "recommended_for": ["Hot spot detection", "Spatial clustering identification"],
                "statistical_tests": ["Getis-Ord Gi* statistic", "Multiple comparison correction"]
            },
            "spatial_concentration": {
                "name": "Spatial Concentration Analysis", 
                "description": "Spatial concentration indices including Gini coefficient and location quotients",
                "output": ["Gini coefficient", "Location quotients", "Spatial association matrices"],
                "recommended_for": ["Concentration measurement", "Inequality analysis"],
                "statistical_tests": ["Gini coefficient", "Location quotient analysis", "Spatial association"]
            },
            "pattern_significance": {
                "name": "Pattern Significance Classification",
                "description": "Multi-tier significance classification with reliability scoring",
                "output": ["Significance classification", "Reliability scores", "Confidence metrics"],
                "recommended_for": ["Overall pattern assessment", "Decision support"],
                "statistical_tests": ["Multiple statistical test combination", "Reliability scoring"]
            },
            "full_statistical_suite": {
                "name": "Full Statistical Validation Suite",
                "description": "Comprehensive validation using all available statistical methods",
                "output": ["Complete statistical analysis", "All individual test results", "Comprehensive reports"],
                "recommended_for": ["Complete validation", "Research applications", "Publication preparation"],
                "statistical_tests": ["All available methods", "Comprehensive analysis", "Integrated reporting"]
            }
        }
        
        return {
            "success": True,
            "total_methods": len(methods),
            "methods": methods,
            "categories": {
                "spatial_autocorrelation": ["comprehensive_morans_i"],
                "null_hypothesis_testing": ["monte_carlo_validation", "bootstrap_validation", "csr_testing"],
                "spatial_analysis": ["hotspot_analysis", "spatial_concentration"],
                "overall_assessment": ["pattern_significance", "full_statistical_suite"]
            },
            "recommendations": {
                "quick_validation": ["comprehensive_morans_i", "csr_testing"],
                "thorough_validation": ["full_statistical_suite"],
                "research_grade": ["full_statistical_suite", "monte_carlo_validation", "bootstrap_validation"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving validation methods: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve validation methods: {str(e)}")

# Helper functions

async def _prepare_pattern_data_for_validation(pattern_id: str) -> Dict[str, Any]:
    """
    Prepare pattern data for statistical validation.
    """
    try:
        # Get pattern from database
        pattern = await pattern_storage.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        # Get pattern components (spatial data points)
        components = await pattern_storage.get_pattern_components(pattern_id)
        
        # Format pattern data for validation
        pattern_data = {
            "id": pattern_id,
            "name": pattern.get("name"),
            "pattern_type": pattern.get("pattern_type"),
            "confidence_score": pattern.get("confidence_score", 0.0),
            "pattern_components": components,
            "discovery_region": pattern.get("discovery_region"),
            "metadata": pattern.get("metadata", {})
        }
        
        # Add spatial features if available
        if components:
            spatial_features = []
            for component in components:
                if component.get("component_type") == "sacred_site":
                    # Get sacred site details
                    site_data = await pattern_storage.get_sacred_site(component["component_id"])
                    if site_data and "location" in site_data:
                        spatial_features.append({
                            "latitude": site_data["location"]["latitude"],
                            "longitude": site_data["location"]["longitude"],
                            "value": component.get("relevance_score", 1.0),
                            "component_id": component["component_id"],
                            "component_type": component["component_type"]
                        })
            
            pattern_data["features"] = spatial_features
        
        return pattern_data
        
    except Exception as e:
        logger.error(f"Error preparing pattern data for validation: {e}")
        raise

def _validate_statistical_methods(methods: List[str]) -> List[str]:
    """
    Validate that requested statistical methods are available.
    """
    available_methods = [
        "comprehensive_morans_i",
        "monte_carlo_validation", 
        "bootstrap_validation",
        "csr_testing",
        "hotspot_analysis",
        "spatial_concentration",
        "pattern_significance",
        "full_statistical_suite"
    ]
    
    invalid_methods = [m for m in methods if m not in available_methods]
    if invalid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid validation methods: {invalid_methods}. Available: {available_methods}"
        )
    
    return methods