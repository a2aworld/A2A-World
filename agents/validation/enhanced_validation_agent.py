"""
A2A World Platform - Enhanced Validation Agent

Enhanced validation agent that integrates comprehensive statistical validation framework
with rigorous Moran's I analysis, null hypothesis testing, and advanced statistical metrics.
This agent enhances the existing validation capabilities to provide Phase 3 statistical rigor.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

from agents.core.base_agent import BaseAgent
from agents.core.config import ValidationAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.pattern_storage import PatternStorage

# Import our new statistical validation framework
from .statistical_validation import (
    MoransIAnalyzer, NullHypothesisTests, StatisticalResult,
    SignificanceLevel, PatternSignificance
)
from .statistical_validation_extended import (
    SpatialStatistics, SignificanceClassifier, StatisticalReports
)

# Import cultural and ethical validation components
from .cultural_validation import CulturalRelevanceValidator, MythologicalContextAnalyzer
from .ethical_validation import HumanFlourishingValidator, BiasDiversityAssessor


class EnhancedValidationAgent(BaseAgent):
    """
    Enhanced validation agent with comprehensive statistical validation capabilities.
    
    This agent provides rigorous statistical validation of discovered patterns using:
    - Global and Local Moran's I spatial autocorrelation analysis
    - Monte Carlo permutation tests and bootstrap validation
    - Complete Spatial Randomness (CSR) testing using Ripley's K function
    - Getis-Ord Gi* statistic for hot spot analysis
    - Advanced statistical metrics with multiple comparison corrections
    - Multi-tier significance classification system
    - Comprehensive statistical reporting and visualization
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[ValidationAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="enhanced_validation",
            config=config or ValidationAgentConfig(),
            config_file=config_file
        )
        
        # Initialize statistical validation components
        self.morans_analyzer = MoransIAnalyzer(
            significance_level=self.config.significance_level,
            n_permutations=getattr(self.config, 'monte_carlo_iterations', 999)
        )
        
        self.null_hypothesis_tests = NullHypothesisTests(
            significance_level=self.config.significance_level,
            n_bootstrap=getattr(self.config, 'bootstrap_iterations', 1000),
            n_permutations=getattr(self.config, 'monte_carlo_iterations', 999)
        )
        
        self.spatial_statistics = SpatialStatistics(
            significance_level=self.config.significance_level
        )
        
        self.significance_classifier = SignificanceClassifier()
        self.statistical_reports = StatisticalReports()

        # Cultural and ethical validation components
        self.cultural_validator = CulturalRelevanceValidator()
        self.mythological_analyzer = MythologicalContextAnalyzer()
        self.ethical_validator = HumanFlourishingValidator()
        self.bias_assessor = BiasDiversityAssessor()

        # Pattern storage for database integration
        self.pattern_storage = PatternStorage()
        
        # Enhanced validation methods registry
        self.enhanced_validation_methods = {
            "comprehensive_morans_i": self._comprehensive_morans_i_analysis,
            "monte_carlo_validation": self._monte_carlo_pattern_validation,
            "bootstrap_validation": self._bootstrap_pattern_validation,
            "csr_testing": self._complete_spatial_randomness_testing,
            "hotspot_analysis": self._hotspot_analysis_gi_star,
            "spatial_concentration": self._spatial_concentration_analysis,
            "pattern_significance": self._pattern_significance_classification,
            "full_statistical_suite": self._full_statistical_validation_suite,
            # Cultural and ethical validation methods
            "cultural_relevance_check": self._cultural_relevance_validation,
            "mythological_context_analysis": self._mythological_context_validation,
            "human_flourishing_assessment": self._human_flourishing_validation,
            "bias_diversity_audit": self._bias_diversity_validation,
            "multidisciplinary_validation": self._multidisciplinary_validation
        }
        
        # Validation results cache with enhanced structure
        self.validation_cache: Dict[str, Dict[str, Any]] = {}
        self.batch_validation_queue: List[Dict[str, Any]] = []
        
        # Enhanced statistics tracking
        self.enhanced_validations_performed = 0
        self.highly_significant_patterns = 0
        self.statistical_tests_executed = 0
        self.validation_reports_generated = 0
        
        # Thread pool for parallel statistical computations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info(f"Enhanced ValidationAgent {self.agent_id} initialized with comprehensive statistical framework")
    
    async def process(self) -> None:
        """
        Main processing loop with enhanced validation capabilities.
        """
        try:
            # Process batch validation queue
            if self.batch_validation_queue:
                await self._process_batch_validation_queue()
            
            # Clean up old cache entries (every 100 iterations)
            if self.processed_tasks % 100 == 0:
                await self._cleanup_enhanced_cache()
            
            # Generate periodic validation reports
            if self.processed_tasks % 500 == 0 and self.enhanced_validations_performed > 0:
                await self._generate_periodic_validation_report()
                
        except Exception as e:
            self.logger.error(f"Error in enhanced validation processing: {e}")
    
    async def agent_initialize(self) -> None:
        """
        Enhanced validation agent specific initialization.
        """
        try:
            # Verify statistical dependencies
            self._verify_statistical_dependencies()
            
            # Initialize database connections
            await self._initialize_validation_storage()
            
            # Load cached validation results if available
            await self._load_enhanced_validation_cache()
            
            self.logger.info("Enhanced ValidationAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced ValidationAgent: {e}")
            raise
    
    async def setup_subscriptions(self) -> None:
        """
        Setup enhanced validation-specific message subscriptions.
        """
        if not self.messaging:
            return
        
        # Subscribe to enhanced validation requests
        enhanced_validation_sub_id = await self.nats_client.subscribe(
            "agents.validation.enhanced.request",
            self._handle_enhanced_validation_request,
            queue_group="enhanced-validation-workers"
        )
        self.subscription_ids.append(enhanced_validation_sub_id)
        
        # Subscribe to batch validation requests
        batch_validation_sub_id = await self.nats_client.subscribe(
            "agents.validation.batch.request",
            self._handle_batch_validation_request,
            queue_group="batch-validation-workers"
        )
        self.subscription_ids.append(batch_validation_sub_id)
        
        # Subscribe to pattern discovery events for automatic enhanced validation
        discovery_sub_id = await self.messaging.subscribe_to_discoveries(
            self._handle_discovery_event_enhanced
        )
        self.subscription_ids.append(discovery_sub_id)
    
    async def handle_task(self, task: Task) -> None:
        """
        Handle enhanced validation task processing.
        """
        self.logger.info(f"Processing enhanced validation task {task.task_id}: {task.task_type}")
        
        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)
            
            result = None
            
            if task.task_type == "enhanced_pattern_validation":
                result = await self._enhanced_pattern_validation_task(task)
            elif task.task_type == "comprehensive_statistical_analysis":
                result = await self._comprehensive_statistical_analysis_task(task)
            elif task.task_type == "batch_pattern_validation":
                result = await self._batch_pattern_validation_task(task)
            elif task.task_type == "generate_validation_report":
                result = await self._generate_validation_report_task(task)
            elif task.task_type == "statistical_significance_testing":
                result = await self._statistical_significance_testing_task(task)
            else:
                raise ValueError(f"Unknown enhanced validation task type: {task.task_type}")
            
            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)
            
            self.processed_tasks += 1
            self.enhanced_validations_performed += 1
            
            # Update statistics based on results
            if result and result.get("significance_classification"):
                classification = result["significance_classification"]["overall_classification"]
                if classification in ["very_high", "high"]:
                    self.highly_significant_patterns += 1
            
            if result and "statistical_results" in result:
                self.statistical_tests_executed += len(result["statistical_results"])
            
            self.logger.info(f"Completed enhanced validation task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing enhanced validation task {task.task_id}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)
            
            self.failed_tasks += 1
        
        finally:
            self.current_tasks.discard(task.task_id)
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect enhanced validation-specific metrics.
        """
        base_metrics = await super().collect_metrics() or {}
        
        enhanced_significance_rate = 0.0
        if self.enhanced_validations_performed > 0:
            enhanced_significance_rate = self.highly_significant_patterns / self.enhanced_validations_performed
        
        avg_tests_per_validation = 0.0
        if self.enhanced_validations_performed > 0:
            avg_tests_per_validation = self.statistical_tests_executed / self.enhanced_validations_performed
        
        enhanced_metrics = {
            "enhanced_validations_performed": self.enhanced_validations_performed,
            "highly_significant_patterns": self.highly_significant_patterns,
            "statistical_tests_executed": self.statistical_tests_executed,
            "validation_reports_generated": self.validation_reports_generated,
            "enhanced_significance_rate": enhanced_significance_rate,
            "avg_tests_per_validation": avg_tests_per_validation,
            "available_enhanced_methods": len(self.enhanced_validation_methods),
            "cache_size": len(self.validation_cache),
            "batch_queue_size": len(self.batch_validation_queue)
        }
        
        return {**base_metrics, **enhanced_metrics}
    
    def _get_capabilities(self) -> List[str]:
        """
        Get enhanced validation agent capabilities.
        """
        base_capabilities = [
            "enhanced_validation",
            "comprehensive_statistical_analysis",
            "enhanced_pattern_validation",
            "batch_pattern_validation",
            "generate_validation_report",
            "statistical_significance_testing",
            "spatial_autocorrelation_analysis",
            "monte_carlo_testing",
            "bootstrap_validation",
            "hotspot_analysis",
            "spatial_concentration_analysis",
            "pattern_significance_classification",
            "multiple_comparison_correction",
            "statistical_reporting",
            # Cultural and ethical validation capabilities
            "cultural_relevance_validation",
            "mythological_context_analysis",
            "human_flourishing_assessment",
            "bias_diversity_audit",
            "multidisciplinary_validation",
            "cultural_sensitivity_assessment",
            "ethical_impact_evaluation"
        ]

        # Add method-specific capabilities
        for method in self.enhanced_validation_methods:
            base_capabilities.append(f"method_{method}")

        return base_capabilities
    
    # Enhanced Validation Task Handlers
    
    async def _enhanced_pattern_validation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle enhanced pattern validation task.
        """
        pattern_id = task.parameters.get("pattern_id")
        pattern_data = task.input_data.get("pattern_data", {})
        validation_methods = task.parameters.get("methods", ["full_statistical_suite"])
        store_results = task.parameters.get("store_results", True)
        
        if not pattern_id:
            raise ValueError("Pattern ID is required for enhanced validation")
        
        return await self.validate_pattern_enhanced(
            pattern_id, pattern_data, validation_methods, store_results
        )
    
    async def _comprehensive_statistical_analysis_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle comprehensive statistical analysis task.
        """
        coordinates = np.array(task.input_data.get("coordinates", []))
        values = np.array(task.input_data.get("values", []))
        analysis_methods = task.parameters.get("methods", ["comprehensive_morans_i", "csr_testing"])
        
        if len(coordinates) == 0:
            raise ValueError("Coordinates are required for statistical analysis")
        
        return await self._run_comprehensive_statistical_analysis(coordinates, values, analysis_methods)
    
    async def _batch_pattern_validation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle batch pattern validation task.
        """
        pattern_ids = task.parameters.get("pattern_ids", [])
        validation_methods = task.parameters.get("methods", ["full_statistical_suite"])
        max_parallel = task.parameters.get("max_parallel", 4)
        
        if not pattern_ids:
            raise ValueError("Pattern IDs are required for batch validation")
        
        return await self.batch_validate_patterns(pattern_ids, validation_methods, max_parallel)
    
    async def _generate_validation_report_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle validation report generation task.
        """
        validation_results = task.input_data.get("validation_results", {})
        report_type = task.parameters.get("report_type", "comprehensive")
        include_visualizations = task.parameters.get("include_visualizations", True)
        
        return await self._generate_comprehensive_validation_report(
            validation_results, report_type, include_visualizations
        )
    
    async def _statistical_significance_testing_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle statistical significance testing task.
        """
        pattern_data = task.input_data.get("pattern_data", {})
        significance_tests = task.parameters.get("tests", ["morans_i", "monte_carlo", "csr"])
        correction_method = task.parameters.get("correction_method", "bonferroni")
        
        return await self._run_significance_testing_suite(pattern_data, significance_tests, correction_method)
    
    # Core Enhanced Validation Methods
    
    async def validate_pattern_enhanced(self, pattern_id: str, pattern_data: Dict[str, Any],
                                      validation_methods: List[str],
                                      store_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive enhanced pattern validation using multiple rigorous statistical methods.
        """
        try:
            # Check cache first
            cache_key = f"{pattern_id}:{':'.join(sorted(validation_methods))}"
            if cache_key in self.validation_cache:
                self.logger.debug(f"Returning cached enhanced validation for {pattern_id}")
                return self.validation_cache[cache_key]
            
            self.logger.info(f"Starting enhanced validation for pattern {pattern_id}")
            
            # Extract and validate spatial data
            spatial_data = await self._extract_and_validate_spatial_data(pattern_data)
            if not spatial_data:
                raise ValueError("No valid spatial data found for pattern validation")
            
            # Convert to coordinate arrays for analysis
            coordinates, values = self._prepare_analysis_data(spatial_data)
            
            # Initialize results structure
            enhanced_results = {
                "pattern_id": pattern_id,
                "validation_timestamp": datetime.utcnow().isoformat(),
                "validation_methods": validation_methods,
                "statistical_results": [],
                "spatial_analysis": {},
                "significance_classification": {},
                "validation_report": {},
                "enhanced_metrics": {},
                "recommendations": []
            }
            
            # Run statistical validation methods
            for method in validation_methods:
                if method in self.enhanced_validation_methods:
                    self.logger.info(f"Running {method} validation for pattern {pattern_id}")
                    
                    try:
                        method_result = await self.enhanced_validation_methods[method](
                            coordinates, values, spatial_data, pattern_data
                        )
                        
                        if method_result:
                            enhanced_results[f"{method}_results"] = method_result
                            
                            # Extract statistical results if present
                            if "statistical_results" in method_result:
                                enhanced_results["statistical_results"].extend(
                                    method_result["statistical_results"]
                                )
                            
                            # Extract individual result if it's a StatisticalResult
                            if isinstance(method_result, dict) and "statistic_name" in method_result:
                                stat_result = StatisticalResult(
                                    statistic_name=method_result["statistic_name"],
                                    statistic_value=method_result["statistic_value"],
                                    p_value=method_result["p_value"],
                                    significant=method_result.get("significant", False),
                                    interpretation=method_result.get("interpretation", ""),
                                    metadata=method_result.get("metadata", {})
                                )
                                enhanced_results["statistical_results"].append(stat_result)
                                
                    except Exception as e:
                        self.logger.error(f"Method {method} failed for pattern {pattern_id}: {e}")
                        enhanced_results[f"{method}_error"] = str(e)
                else:
                    self.logger.warning(f"Unknown validation method: {method}")
            
            # Perform significance classification if we have statistical results
            if enhanced_results["statistical_results"]:
                enhanced_results["significance_classification"] = self.significance_classifier.classify_pattern_significance(
                    enhanced_results["statistical_results"]
                )
            
            # Generate comprehensive validation report
            enhanced_results["validation_report"] = self.statistical_reports.generate_comprehensive_report(
                enhanced_results, pattern_data
            )
            
            # Calculate enhanced metrics
            enhanced_results["enhanced_metrics"] = self._calculate_enhanced_metrics(
                enhanced_results, len(spatial_data)
            )
            
            # Generate recommendations
            enhanced_results["recommendations"] = self._generate_enhanced_recommendations(
                enhanced_results
            )
            
            # Store results in database if requested
            if store_results and enhanced_results["statistical_results"]:
                try:
                    stored_validation_id = await self._store_enhanced_validation_results(
                        pattern_id, enhanced_results
                    )
                    enhanced_results["stored_validation_id"] = stored_validation_id
                    self.logger.info(f"Enhanced validation results stored with ID: {stored_validation_id}")

                    # Publish validation completion event for XAI agent
                    if self.messaging:
                        validation_event = {
                            "validation_id": stored_validation_id,
                            "pattern_id": pattern_id,
                            "validation_type": "enhanced_statistical",
                            "overall_classification": enhanced_results.get("significance_classification", {}).get("overall_classification", "unknown"),
                            "confidence_score": enhanced_results.get("enhanced_metrics", {}).get("reliability_score", 0.0),
                            "significant_tests": enhanced_results.get("enhanced_metrics", {}).get("significant_tests", 0),
                            "auto_explain": True,  # Enable automatic XAI explanation generation
                            "timestamp": datetime.utcnow().isoformat()
                        }

                        await self.messaging.publish_validation(validation_event)
                        self.logger.info(f"Published validation completion event for XAI processing: {stored_validation_id}")

                except Exception as e:
                    self.logger.error(f"Failed to store enhanced validation results: {e}")
                    enhanced_results["storage_error"] = str(e)
            
            # Cache results
            self.validation_cache[cache_key] = enhanced_results
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Enhanced pattern validation failed for {pattern_id}: {e}")
            return {
                "pattern_id": pattern_id,
                "error": str(e),
                "validation_timestamp": datetime.utcnow().isoformat(),
                "statistical_results": [],
                "enhanced_metrics": {"validation_success": False}
            }
    
    async def batch_validate_patterns(self, pattern_ids: List[str],
                                    validation_methods: List[str],
                                    max_parallel: int = 4) -> Dict[str, Any]:
        """
        Validate multiple patterns in parallel with enhanced statistical analysis.
        """
        try:
            self.logger.info(f"Starting batch validation for {len(pattern_ids)} patterns")
            
            batch_results = {
                "batch_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "total_patterns": len(pattern_ids),
                "validation_methods": validation_methods,
                "pattern_results": {},
                "batch_summary": {},
                "failed_patterns": []
            }
            
            # Semaphore to limit parallel validations
            semaphore = asyncio.Semaphore(max_parallel)
            
            async def validate_single_pattern(pattern_id: str) -> Tuple[str, Dict[str, Any]]:
                async with semaphore:
                    try:
                        # Get pattern data from database
                        pattern_data = await self._get_pattern_data_for_validation(pattern_id)
                        
                        if not pattern_data:
                            raise ValueError(f"Pattern data not found for {pattern_id}")
                        
                        # Perform enhanced validation
                        result = await self.validate_pattern_enhanced(
                            pattern_id, pattern_data, validation_methods, store_results=True
                        )
                        
                        return pattern_id, result
                        
                    except Exception as e:
                        self.logger.error(f"Batch validation failed for pattern {pattern_id}: {e}")
                        return pattern_id, {
                            "pattern_id": pattern_id,
                            "error": str(e),
                            "validation_timestamp": datetime.utcnow().isoformat()
                        }
            
            # Execute validations in parallel
            validation_tasks = [validate_single_pattern(pid) for pid in pattern_ids]
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            successful_validations = 0
            highly_significant_count = 0
            
            for result in results:
                if isinstance(result, Exception):
                    batch_results["failed_patterns"].append(f"Exception: {str(result)}")
                    continue
                
                pattern_id, validation_result = result
                batch_results["pattern_results"][pattern_id] = validation_result
                
                if "error" not in validation_result:
                    successful_validations += 1
                    
                    # Check if highly significant
                    sig_class = validation_result.get("significance_classification", {})
                    if sig_class.get("overall_classification") in ["very_high", "high"]:
                        highly_significant_count += 1
                else:
                    batch_results["failed_patterns"].append(pattern_id)
            
            # Generate batch summary
            batch_results["batch_summary"] = {
                "successful_validations": successful_validations,
                "failed_validations": len(batch_results["failed_patterns"]),
                "success_rate": successful_validations / len(pattern_ids) if pattern_ids else 0,
                "highly_significant_patterns": highly_significant_count,
                "significance_rate": highly_significant_count / successful_validations if successful_validations > 0 else 0
            }
            
            self.logger.info(f"Batch validation completed: {successful_validations}/{len(pattern_ids)} successful")
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            return {
                "batch_id": str(uuid.uuid4()),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Statistical Analysis Methods Implementation
    
    async def _comprehensive_morans_i_analysis(self, coordinates: np.ndarray, values: np.ndarray,
                                             spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive Moran's I analysis including global and local analysis.
        """
        try:
            results = {
                "analysis_type": "comprehensive_morans_i",
                "statistical_results": []
            }
            
            # Global Moran's I with different weight methods
            weight_methods = ["knn", "distance"]
            
            for method in weight_methods:
                global_result = self.morans_analyzer.calculate_global_morans_i(
                    coordinates, values, weights_method=method
                )
                results["statistical_results"].append(global_result)
                results[f"global_morans_i_{method}"] = global_result
            
            # Local Moran's I (LISA)
            local_result = self.morans_analyzer.calculate_local_morans_i(
                coordinates, values, weights_method="knn"
            )
            results["local_morans_i"] = local_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive Moran's I analysis failed: {e}")
            return {"error": str(e), "analysis_type": "comprehensive_morans_i"}
    
    async def _monte_carlo_pattern_validation(self, coordinates: np.ndarray, values: np.ndarray,
                                            spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Monte Carlo permutation test for pattern significance.
        """
        try:
            def test_statistic_func(coords, vals):
                # Use Moran's I as test statistic
                analyzer = MoransIAnalyzer()
                result = analyzer.calculate_global_morans_i(coords, vals)
                return result.statistic_value
            
            mc_result = self.null_hypothesis_tests.monte_carlo_permutation_test(
                coordinates, values, test_statistic_func
            )
            
            return {
                "analysis_type": "monte_carlo_validation",
                "statistical_results": [mc_result],
                "monte_carlo_result": mc_result
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo validation failed: {e}")
            return {"error": str(e), "analysis_type": "monte_carlo_validation"}
    
    async def _bootstrap_pattern_validation(self, coordinates: np.ndarray, values: np.ndarray,
                                          spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Bootstrap validation for pattern confidence intervals.
        """
        try:
            def statistic_func(data):
                if len(data) < 3:
                    return 0.0
                # Calculate a simple spatial statistic
                return np.std(data) / np.mean(data) if np.mean(data) != 0 else 0.0
            
            bootstrap_result = self.null_hypothesis_tests.bootstrap_confidence_intervals(
                values, statistic_func
            )
            
            return {
                "analysis_type": "bootstrap_validation", 
                "bootstrap_result": bootstrap_result
            }
            
        except Exception as e:
            self.logger.error(f"Bootstrap validation failed: {e}")
            return {"error": str(e), "analysis_type": "bootstrap_validation"}
    
    async def _complete_spatial_randomness_testing(self, coordinates: np.ndarray, values: np.ndarray,
                                                 spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Complete Spatial Randomness testing using Ripley's K and nearest neighbor analysis.
        """
        try:
            results = {
                "analysis_type": "csr_testing",
                "statistical_results": []
            }
            
            # Ripley's K test
            csr_result = self.null_hypothesis_tests.complete_spatial_randomness_test(coordinates)
            results["statistical_results"].append(csr_result)
            results["ripley_k_test"] = csr_result
            
            # Nearest neighbor analysis
            nn_result = self.null_hypothesis_tests.nearest_neighbor_analysis(coordinates)
            results["statistical_results"].append(nn_result)
            results["nearest_neighbor_test"] = nn_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"CSR testing failed: {e}")
            return {"error": str(e), "analysis_type": "csr_testing"}
    
    async def _hotspot_analysis_gi_star(self, coordinates: np.ndarray, values: np.ndarray,
                                      spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Getis-Ord Gi* hotspot analysis.
        """
        try:
            gi_star_result = self.spatial_statistics.getis_ord_gi_star(
                coordinates, values, weights_method="distance"
            )
            
            return {
                "analysis_type": "hotspot_analysis",
                "gi_star_result": gi_star_result
            }
            
        except Exception as e:
            self.logger.error(f"Hotspot analysis failed: {e}")
            return {"error": str(e), "analysis_type": "hotspot_analysis"}
    
    async def _spatial_concentration_analysis(self, coordinates: np.ndarray, values: np.ndarray,
                                            spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Spatial concentration analysis using Gini coefficient and location quotients.
        """
        try:
            results = {
                "analysis_type": "spatial_concentration"
            }
            
            # Gini coefficient
            gini_coeff = self.spatial_statistics.gini_coefficient(values)
            results["gini_coefficient"] = gini_coeff
            
            # Location quotients (if we have global reference data)
            if len(values) > 1:
                location_quotients = self.spatial_statistics.location_quotient(
                    values, values  # Using same data as reference for demonstration
                )
                results["location_quotients"] = location_quotients.tolist()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Spatial concentration analysis failed: {e}")
            return {"error": str(e), "analysis_type": "spatial_concentration"}
    
    async def _pattern_significance_classification(self, coordinates: np.ndarray, values: np.ndarray,
                                                 spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Pattern significance classification using multiple criteria.
        """
        try:
            # Run basic statistical tests first
            basic_results = []
            
            # Moran's I
            morans_result = self.morans_analyzer.calculate_global_morans_i(coordinates, values)
            basic_results.append(morans_result)
            
            # Nearest neighbor
            nn_result = self.null_hypothesis_tests.nearest_neighbor_analysis(coordinates)
            basic_results.append(nn_result)
            
            # Classify significance
            classification_result = self.significance_classifier.classify_pattern_significance(basic_results)
            
            return {
                "analysis_type": "pattern_significance",
                "statistical_results": basic_results,
                "significance_classification": classification_result
            }
            
        except Exception as e:
            self.logger.error(f"Pattern significance classification failed: {e}")
            return {"error": str(e), "analysis_type": "pattern_significance"}
    
    async def _full_statistical_validation_suite(self, coordinates: np.ndarray, values: np.ndarray,
                                                 spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Full statistical validation suite running all available methods.
        """
        try:
            suite_results = {
                "analysis_type": "full_statistical_suite",
                "statistical_results": [],
                "individual_analyses": {}
            }

            # Run all major analysis types
            analysis_methods = [
                "comprehensive_morans_i",
                "monte_carlo_validation",
                "csr_testing",
                "hotspot_analysis",
                "spatial_concentration"
            ]

            for method in analysis_methods:
                if method in self.enhanced_validation_methods and method != "full_statistical_suite":
                    try:
                        result = await self.enhanced_validation_methods[method](
                            coordinates, values, spatial_data, pattern_data
                        )
                        suite_results["individual_analyses"][method] = result

                        # Collect statistical results
                        if "statistical_results" in result:
                            suite_results["statistical_results"].extend(result["statistical_results"])

                    except Exception as e:
                        self.logger.error(f"Method {method} failed in full suite: {e}")
                        suite_results["individual_analyses"][method] = {"error": str(e)}

            return suite_results

        except Exception as e:
            self.logger.error(f"Full statistical validation suite failed: {e}")
            return {"error": str(e), "analysis_type": "full_statistical_suite"}

    # Cultural and Ethical Validation Methods

    async def _cultural_relevance_validation(self, coordinates: np.ndarray, values: np.ndarray,
                                           spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Cultural relevance validation for patterns.
        """
        try:
            cultural_assessment = await self.cultural_validator.assess_cultural_relevance(
                pattern_data, {}
            )

            return {
                "analysis_type": "cultural_relevance",
                "cultural_relevance_score": cultural_assessment.get("cultural_relevance_score", 0.0),
                "sensitivity_assessment": cultural_assessment.get("sensitivity_assessment", {}),
                "cultural_contexts": cultural_assessment.get("cultural_contexts", []),
                "sensitivity_concerns": cultural_assessment.get("sensitivity_concerns", []),
                "cultural_recommendations": cultural_assessment.get("cultural_recommendations", [])
            }

        except Exception as e:
            self.logger.error(f"Cultural relevance validation failed: {e}")
            return {"error": str(e), "analysis_type": "cultural_relevance"}

    async def _mythological_context_validation(self, coordinates: np.ndarray, values: np.ndarray,
                                             spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Mythological context analysis for patterns.
        """
        try:
            mythological_analysis = await self.mythological_analyzer.analyze_mythological_context(
                pattern_data
            )

            return {
                "analysis_type": "mythological_context",
                "mythological_alignment_score": mythological_analysis.get("mythological_alignment_score", 0.0),
                "identified_archetypes": mythological_analysis.get("identified_archetypes", []),
                "symbolic_elements": mythological_analysis.get("symbolic_elements", []),
                "cross_cultural_connections": mythological_analysis.get("cross_cultural_connections", []),
                "mythological_significance": mythological_analysis.get("mythological_significance", "unknown")
            }

        except Exception as e:
            self.logger.error(f"Mythological context validation failed: {e}")
            return {"error": str(e), "analysis_type": "mythological_context"}

    async def _human_flourishing_validation(self, coordinates: np.ndarray, values: np.ndarray,
                                          spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Human flourishing alignment validation.
        """
        try:
            flourishing_assessment = await self.ethical_validator.assess_human_flourishing(
                pattern_data
            )

            return {
                "analysis_type": "human_flourishing",
                "flourishing_score": flourishing_assessment.get("flourishing_score", 0.0),
                "dimension_scores": flourishing_assessment.get("dimension_scores", {}),
                "ethical_concerns": flourishing_assessment.get("ethical_concerns", []),
                "flourishing_alignment": flourishing_assessment.get("flourishing_alignment", {}),
                "ethical_recommendations": flourishing_assessment.get("recommendations", [])
            }

        except Exception as e:
            self.logger.error(f"Human flourishing validation failed: {e}")
            return {"error": str(e), "analysis_type": "human_flourishing"}

    async def _bias_diversity_validation(self, coordinates: np.ndarray, values: np.ndarray,
                                       spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Bias and diversity audit for validation results.
        """
        try:
            # Create validation data structure for bias audit
            validation_data = {
                "pattern_components": spatial_data,
                "description": pattern_data.get("description", ""),
                "name": pattern_data.get("name", ""),
                "validation_methods": ["statistical", "cultural", "ethical"],
                "statistical_results": []
            }

            bias_audit = await self.bias_assessor.audit_bias_diversity(validation_data)

            return {
                "analysis_type": "bias_diversity",
                "diversity_score": bias_audit.get("diversity_score", 0.0),
                "bias_score": bias_audit.get("bias_score", 1.0),
                "diversity_dimensions": bias_audit.get("diversity_dimensions", {}),
                "identified_biases": bias_audit.get("identified_biases", []),
                "diversity_recommendations": bias_audit.get("diversity_recommendations", [])
            }

        except Exception as e:
            self.logger.error(f"Bias diversity validation failed: {e}")
            return {"error": str(e), "analysis_type": "bias_diversity"}

    async def _multidisciplinary_validation(self, coordinates: np.ndarray, values: np.ndarray,
                                          spatial_data: List[Dict], pattern_data: Dict) -> Dict[str, Any]:
        """
        Integrated multidisciplinary validation combining all layers.
        """
        try:
            multidisciplinary_results = {
                "analysis_type": "multidisciplinary_validation",
                "layer_assessments": {},
                "integrated_score": 0.0,
                "overall_assessment": {},
                "multidisciplinary_recommendations": []
            }

            # Run all validation layers
            layers_to_run = [
                "cultural_relevance_check",
                "mythological_context_analysis",
                "human_flourishing_assessment",
                "bias_diversity_audit"
            ]

            layer_scores = {}
            for layer in layers_to_run:
                if layer in self.enhanced_validation_methods:
                    try:
                        result = await self.enhanced_validation_methods[layer](
                            coordinates, values, spatial_data, pattern_data
                        )
                        multidisciplinary_results["layer_assessments"][layer] = result

                        # Extract score based on layer type
                        if layer == "cultural_relevance_check":
                            layer_scores[layer] = result.get("cultural_relevance_score", 0.0)
                        elif layer == "mythological_context_analysis":
                            layer_scores[layer] = result.get("mythological_alignment_score", 0.0)
                        elif layer == "human_flourishing_assessment":
                            layer_scores[layer] = result.get("flourishing_score", 0.0)
                        elif layer == "bias_diversity_audit":
                            # Combine diversity and bias scores
                            diversity = result.get("diversity_score", 0.0)
                            bias = result.get("bias_score", 1.0)
                            layer_scores[layer] = (diversity + bias) / 2

                    except Exception as e:
                        self.logger.error(f"Layer {layer} failed in multidisciplinary validation: {e}")
                        multidisciplinary_results["layer_assessments"][layer] = {"error": str(e)}
                        layer_scores[layer] = 0.0

            # Calculate integrated score
            if layer_scores:
                multidisciplinary_results["integrated_score"] = sum(layer_scores.values()) / len(layer_scores)

                # Determine overall assessment
                integrated_score = multidisciplinary_results["integrated_score"]
                if integrated_score >= 0.8:
                    assessment = "excellent_multidisciplinary_alignment"
                    recommendations = ["Pattern shows excellent multidisciplinary validation"]
                elif integrated_score >= 0.6:
                    assessment = "good_multidisciplinary_alignment"
                    recommendations = ["Pattern shows good multidisciplinary validation with minor areas for improvement"]
                elif integrated_score >= 0.4:
                    assessment = "moderate_multidisciplinary_alignment"
                    recommendations = ["Pattern shows moderate multidisciplinary validation - consider improvements"]
                else:
                    assessment = "needs_multidisciplinary_improvement"
                    recommendations = ["Pattern requires significant multidisciplinary validation improvements"]

                multidisciplinary_results["overall_assessment"] = assessment
                multidisciplinary_results["multidisciplinary_recommendations"] = recommendations

                # Collect all recommendations from layers
                all_recommendations = []
                for layer_result in multidisciplinary_results["layer_assessments"].values():
                    if isinstance(layer_result, dict) and "recommendations" in layer_result:
                        all_recommendations.extend(layer_result["recommendations"])
                    elif isinstance(layer_result, dict) and "cultural_recommendations" in layer_result:
                        all_recommendations.extend(layer_result["cultural_recommendations"])
                    elif isinstance(layer_result, dict) and "ethical_recommendations" in layer_result:
                        all_recommendations.extend(layer_result["ethical_recommendations"])
                    elif isinstance(layer_result, dict) and "diversity_recommendations" in layer_result:
                        all_recommendations.extend(layer_result["diversity_recommendations"])

                multidisciplinary_results["multidisciplinary_recommendations"].extend(all_recommendations[:5])  # Limit to top 5

            return multidisciplinary_results

        except Exception as e:
            self.logger.error(f"Multidisciplinary validation failed: {e}")
            return {"error": str(e), "analysis_type": "multidisciplinary_validation"}
    
    # Helper Methods
    
    def _prepare_analysis_data(self, spatial_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare coordinate and value arrays from spatial data.
        """
        coordinates = []
        values = []
        
        for point in spatial_data:
            if "latitude" in point and "longitude" in point:
                coordinates.append([point["latitude"], point["longitude"]])
                values.append(point.get("value", point.get("significance_level", 1.0)))
        
        return np.array(coordinates), np.array(values)
    
    async def _extract_and_validate_spatial_data(self, pattern_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and validate spatial data from pattern data structure.
        """
        try:
            # Try different possible data structures
            spatial_data = None
            
            if "features" in pattern_data:
                spatial_data = pattern_data["features"]
            elif "points" in pattern_data:
                spatial_data = pattern_data["points"]
            elif "coordinates" in pattern_data:
                coords = pattern_data["coordinates"]
                spatial_data = [{"latitude": lat, "longitude": lon, "value": 1} 
                              for lat, lon in coords]
            elif "pattern_components" in pattern_data:
                # Extract from pattern components
                components = pattern_data["pattern_components"]
                spatial_data = []
                for comp in components:
                    if "coordinates" in comp:
                        spatial_data.append({
                            "latitude": comp["coordinates"][0],
                            "longitude": comp["coordinates"][1],
                            "value": comp.get("relevance_score", 1.0)
                        })
            
            # If no explicit spatial data, try to get from database using pattern ID
            if not spatial_data and "id" in pattern_data:
                spatial_data = await self._get_spatial_data_from_database(pattern_data["id"])
            
            # Validate spatial data
            if spatial_data:
                validated_data = []
                for point in spatial_data:
                    if ("latitude" in point and "longitude" in point and
                        isinstance(point["latitude"], (int, float)) and
                        isinstance(point["longitude"], (int, float))):
                        validated_data.append(point)
                
                return validated_data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error extracting spatial data: {e}")
            return []
    
    async def _get_spatial_data_from_database(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get spatial data from database for a given pattern ID.
        """
        try:
            # Use pattern storage to get pattern components
            pattern_components = await self.pattern_storage.get_pattern_components(pattern_id)
            
            spatial_data = []
            for component in pattern_components:
                if component.get("component_type") == "sacred_site":
                    # Get sacred site details
                    site_data = await self.pattern_storage.get_sacred_site(component["component_id"])
                    if site_data and "location" in site_data:
                        spatial_data.append({
                            "latitude": site_data["location"]["latitude"],
                            "longitude": site_data["location"]["longitude"], 
                            "value": component.get("relevance_score", 1.0),
                            "site_id": component["component_id"]
                        })
            
            return spatial_data
            
        except Exception as e:
            self.logger.error(f"Error getting spatial data from database: {e}")
            return []
    
    def _calculate_enhanced_metrics(self, validation_results: Dict[str, Any], sample_size: int) -> Dict[str, Any]:
        """
        Calculate enhanced validation metrics.
        """
        try:
            metrics = {
                "validation_success": True,
                "sample_size": sample_size,
                "total_statistical_tests": len(validation_results.get("statistical_results", [])),
                "significant_tests": 0,
                "highly_significant_tests": 0,
                "overall_significance_score": 0.0,
                "reliability_score": 0.0
            }
            
            # Count significant tests
            for result in validation_results.get("statistical_results", []):
                if hasattr(result, 'significant') and result.significant:
                    metrics["significant_tests"] += 1
                    
                    if hasattr(result, 'p_value') and result.p_value and result.p_value < 0.001:
                        metrics["highly_significant_tests"] += 1
            
            # Calculate significance scores
            if metrics["total_statistical_tests"] > 0:
                metrics["significance_rate"] = metrics["significant_tests"] / metrics["total_statistical_tests"]
                metrics["high_significance_rate"] = metrics["highly_significant_tests"] / metrics["total_statistical_tests"]
            
            # Get overall scores from classification if available
            if "significance_classification" in validation_results:
                sig_class = validation_results["significance_classification"]
                metrics["reliability_score"] = sig_class.get("reliability_score", 0.0)
                
                # Convert classification to numeric score
                classification = sig_class.get("overall_classification", "not_significant")
                classification_scores = {
                    "very_high": 1.0,
                    "high": 0.8,
                    "moderate": 0.6,
                    "low": 0.3,
                    "not_significant": 0.0
                }
                metrics["overall_significance_score"] = classification_scores.get(classification, 0.0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced metrics: {e}")
            return {"validation_success": False, "error": str(e)}
    
    def _generate_enhanced_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Generate enhanced recommendations based on validation results.
        """
        recommendations = []
        
        try:
            # Get basic metrics
            metrics = validation_results.get("enhanced_metrics", {})
            sig_class = validation_results.get("significance_classification", {})
            
            # Overall classification recommendations
            classification = sig_class.get("overall_classification", "unknown")
            reliability_score = sig_class.get("reliability_score", 0.0)
            
            if classification == "very_high":
                recommendations.extend([
                    "Pattern shows very high statistical significance - excellent candidate for publication",
                    "Consider replication studies with independent datasets",
                    "Suitable for use in evidence-based decision making"
                ])
            elif classification == "high": 
                recommendations.extend([
                    "Pattern shows high statistical significance - strong evidence",
                    "Additional validation with larger sample sizes recommended",
                    "Consider peer review for publication readiness"
                ])
            elif classification == "moderate":
                recommendations.extend([
                    "Pattern shows moderate statistical significance - proceed with caution",
                    "Additional statistical tests or methodological improvements recommended",
                    "Consider increasing sample size or refining analysis methods"
                ])
            elif classification == "low":
                recommendations.extend([
                    "Pattern shows low statistical significance - interpret results cautiously",
                    "Consider alternative analytical approaches or data collection methods",
                    "May require fundamental methodological review"
                ])
            else:
                recommendations.extend([
                    "Pattern does not show statistical significance",
                    "Review data quality, methodology, and analytical assumptions",
                    "Consider alternative hypotheses or research questions"
                ])
            
            # Reliability-based recommendations
            if reliability_score < 0.3:
                recommendations.append("Very low reliability score - comprehensive methodological review required")
            elif reliability_score < 0.5:
                recommendations.append("Moderate reliability - additional validation strongly recommended")
            elif reliability_score > 0.8:
                recommendations.append("High reliability score - results are trustworthy")
            
            # Sample size recommendations
            sample_size = metrics.get("sample_size", 0)
            if sample_size < 30:
                recommendations.append("Small sample size - consider increasing sample for more robust results")
            elif sample_size > 1000:
                recommendations.append("Large sample size provides excellent statistical power")
            
            # Testing completeness recommendations
            total_tests = metrics.get("total_statistical_tests", 0)
            if total_tests < 3:
                recommendations.append("Limited statistical testing - consider running additional validation methods")
            elif total_tests > 8:
                recommendations.append("Comprehensive statistical testing completed - excellent validation coverage")
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review required")
        
        return recommendations
    
    # Storage and Caching
    
    async def _store_enhanced_validation_results(self, pattern_id: str, validation_results: Dict[str, Any]) -> str:
        """
        Store enhanced validation results in database.
        """
        try:
            validation_record = {
                "pattern_id": pattern_id,
                "validation_type": "enhanced_statistical_validation",
                "validator_type": "agent",
                "validator_id": self.agent_id,
                "validation_result": validation_results.get("significance_classification", {}).get("overall_classification", "unknown"),
                "validation_score": validation_results.get("enhanced_metrics", {}).get("reliability_score", 0.0),
                "confidence_level": validation_results.get("significance_classification", {}).get("reliability_score", 0.0),
                "validation_method": "comprehensive_statistical_framework",
                "test_statistics": validation_results,
                "validation_notes": f"Enhanced validation with {len(validation_results.get('statistical_results', []))} statistical tests",
                "replication_successful": validation_results.get("enhanced_metrics", {}).get("validation_success", False),
                "recommendations": "; ".join(validation_results.get("recommendations", []))
            }
            
            # Store in database using pattern storage
            validation_id = await self.pattern_storage.store_pattern_validation(validation_record)
            
            return validation_id
            
        except Exception as e:
            self.logger.error(f"Failed to store enhanced validation results: {e}")
            raise
    
    async def _cleanup_enhanced_cache(self) -> None:
        """
        Clean up old enhanced validation cache entries.
        """
        current_time = datetime.utcnow()
        cache_ttl = 7200  # 2 hours for enhanced validation cache
        
        keys_to_remove = []
        for key, result in self.validation_cache.items():
            if "validation_timestamp" in result:
                try:
                    result_time = datetime.fromisoformat(result["validation_timestamp"])
                    if (current_time - result_time).seconds > cache_ttl:
                        keys_to_remove.append(key)
                except (ValueError, KeyError):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.validation_cache[key]
        
        if keys_to_remove:
            self.logger.info(f"Cleaned up {len(keys_to_remove)} enhanced validation cache entries")
    
    # Message Handlers
    
    async def _handle_enhanced_validation_request(self, message: AgentMessage) -> None:
        """
        Handle enhanced validation requests via NATS.
        """
        try:
            request_data = message.payload
            pattern_id = request_data.get("pattern_id")
            validation_methods = request_data.get("methods", ["full_statistical_suite"])
            pattern_data = request_data.get("pattern_data", {})
            
            # Perform enhanced validation
            results = await self.validate_pattern_enhanced(
                pattern_id, pattern_data, validation_methods
            )
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="enhanced_validation_response",
                payload={
                    "pattern_id": pattern_id,
                    "validation_results": results,
                    "timestamp": datetime.utcnow().isoformat()
                },
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling enhanced validation request: {e}")
    
    async def _handle_batch_validation_request(self, message: AgentMessage) -> None:
        """
        Handle batch validation requests.
        """
        try:
            request_data = message.payload
            pattern_ids = request_data.get("pattern_ids", [])
            validation_methods = request_data.get("methods", ["full_statistical_suite"])
            max_parallel = request_data.get("max_parallel", 4)
            
            # Perform batch validation
            results = await self.batch_validate_patterns(
                pattern_ids, validation_methods, max_parallel
            )
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="batch_validation_response",
                payload={
                    "batch_results": results,
                    "timestamp": datetime.utcnow().isoformat()
                },
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling batch validation request: {e}")
    
    async def _handle_discovery_event_enhanced(self, message: AgentMessage) -> None:
        """
        Handle pattern discovery events for automatic enhanced validation.
        """
        try:
            if message.message_type == "pattern_discovered":
                discovery_data = message.payload
                pattern_id = discovery_data.get("pattern_id")
                
                if pattern_id and self.config.auto_validate_discovered_patterns:
                    # Add to batch validation queue for processing
                    self.batch_validation_queue.append({
                        "pattern_id": pattern_id,
                        "discovery_data": discovery_data,
                        "timestamp": datetime.utcnow().isoformat(),
                        "methods": ["comprehensive_morans_i", "csr_testing"]  # Default methods for auto-validation
                    })
                    
                    self.logger.info(f"Added pattern {pattern_id} to enhanced validation queue")
                    
        except Exception as e:
            self.logger.error(f"Error handling enhanced discovery event: {e}")
    
    # Utility Methods
    
    def _verify_statistical_dependencies(self) -> None:
        """
        Verify that all required statistical dependencies are available.
        """
        try:
            import numpy as np
            import pandas as pd
            import scipy.stats
            import sklearn
            from sklearn.neighbors import NearestNeighbors
            from sklearn.metrics import silhouette_score
            self.logger.info("All enhanced statistical dependencies verified")
        except ImportError as e:
            self.logger.error(f"Missing statistical dependency: {e}")
            raise
    
    async def _initialize_validation_storage(self) -> None:
        """
        Initialize database connections and validation storage.
        """
        try:
            # Initialize pattern storage if not already done
            if not self.pattern_storage.initialized:
                await self.pattern_storage.initialize()
            self.logger.info("Enhanced validation storage initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize validation storage: {e}")
            raise
    
    async def _load_enhanced_validation_cache(self) -> None:
        """
        Load enhanced validation cache from database if available.
        """
        # Implementation would load recent validation results from database
        # For now, start with empty cache
        pass
    
    async def shutdown(self) -> None:
        """
        Enhanced shutdown procedure.
        """
        try:
            # Process any remaining batch validations
            if self.batch_validation_queue:
                self.logger.info(f"Processing {len(self.batch_validation_queue)} remaining batch validations")
                await self._process_batch_validation_queue()
            
            # Shutdown thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Call parent shutdown
            await super().shutdown()
            
            self.logger.info("Enhanced ValidationAgent shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during enhanced validation agent shutdown: {e}")


# Main entry point for running the enhanced agent
async def main():
    """
    Main entry point for running the Enhanced ValidationAgent.
    """
    import signal
    import sys
    
    # Create and configure agent
    agent = EnhancedValidationAgent()
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...")
        asyncio.create_task(agent.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the agent
        await agent.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Enhanced ValidationAgent failed: {e}")
        sys.exit(1)
    
    print("Enhanced ValidationAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())