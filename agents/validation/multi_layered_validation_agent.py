"""
A2A World Platform - Multi-Layered Validation Agent

Orchestrates comprehensive validation across statistical, cultural, and ethical dimensions.
Implements Phase 4 advanced features including cultural relevance, human flourishing alignment,
and bias diversity assessment for multidisciplinary protocol validation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
import traceback

from agents.core.base_agent import BaseAgent
from agents.core.config import ValidationAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.pattern_storage import PatternStorage

# Import existing validation frameworks
from .enhanced_validation_agent import EnhancedValidationAgent
from .consensus_validation_agent import ConsensusValidationAgent
from .statistical_validation_extended import SpatialStatistics, SignificanceClassifier

# Import cultural and ethical validation components
from .cultural_validation import CulturalRelevanceValidator, MythologicalContextAnalyzer
from .ethical_validation import HumanFlourishingValidator, BiasDiversityAssessor


class MultiLayeredValidationAgent(BaseAgent):
    """
    Multi-layered validation agent that orchestrates comprehensive validation
    across statistical, cultural, and ethical dimensions.

    This agent implements Phase 4 advanced features:
    - Statistical validation (existing framework)
    - Cultural relevance and sensitivity validation
    - Human flourishing alignment assessment
    - Bias diversity analysis
    - Consensus integration for multidisciplinary protocols
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[ValidationAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="multi_layered_validation",
            config=config or ValidationAgentConfig(),
            config_file=config_file
        )

        # Initialize validation layer components
        self._initialize_validation_layers()

        # Multi-layered validation registry
        self.validation_layers = {
            "statistical": self._statistical_validation_layer,
            "cultural": self._cultural_validation_layer,
            "ethical": self._ethical_validation_layer,
            "consensus": self._consensus_validation_layer,
            "integrated": self._integrated_validation_layer
        }

        # Validation orchestration settings
        self.layer_weights = {
            "statistical": 0.4,
            "cultural": 0.3,
            "ethical": 0.2,
            "consensus": 0.1
        }

        # Performance and caching
        self.validation_cache: Dict[str, Dict[str, Any]] = {}
        self.layer_execution_times: Dict[str, float] = {}
        self.validation_metrics = {
            "total_validations": 0,
            "layer_success_rates": {},
            "average_execution_time": 0.0,
            "multidisciplinary_score": 0.0
        }

        # Thread pool for parallel layer execution
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

        self.logger.info(f"Multi-layered ValidationAgent {self.agent_id} initialized")

    def _initialize_validation_layers(self) -> None:
        """Initialize all validation layer components."""
        try:
            # Statistical validation layer
            self.statistical_validator = EnhancedValidationAgent(
                agent_id=f"{self.agent_id}_statistical",
                config=self.config
            )

            # Cultural validation layer
            self.cultural_validator = CulturalRelevanceValidator()
            self.mythological_analyzer = MythologicalContextAnalyzer()

            # Ethical validation layer
            self.ethical_validator = HumanFlourishingValidator()
            self.bias_assessor = BiasDiversityAssessor()

            # Consensus integration
            self.consensus_validator = ConsensusValidationAgent(
                agent_id=f"{self.agent_id}_consensus",
                config=self.config
            )

            # Pattern storage for database integration
            self.pattern_storage = PatternStorage()

            self.logger.info("All validation layers initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize validation layers: {e}")
            raise

    async def process(self) -> None:
        """Main processing loop for multi-layered validation."""
        try:
            # Process any pending multi-layered validations
            await self._process_pending_validations()

            # Update validation metrics
            await self._update_validation_metrics()

            # Clean up old cache entries
            if self.processed_tasks % 200 == 0:
                await self._cleanup_validation_cache()

        except Exception as e:
            self.logger.error(f"Error in multi-layered validation processing: {e}")

    async def agent_initialize(self) -> None:
        """Initialize multi-layered validation agent."""
        try:
            # Initialize all validation layers
            await self.statistical_validator.agent_initialize()
            await self.consensus_validator.agent_initialize()

            # Initialize database connections
            await self._initialize_validation_storage()

            # Load cached validation results
            await self._load_validation_cache()

            self.logger.info("Multi-layered ValidationAgent initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize Multi-layered ValidationAgent: {e}")
            raise

    async def setup_subscriptions(self) -> None:
        """Setup multi-layered validation message subscriptions."""
        if not self.messaging:
            return

        # Subscribe to multi-layered validation requests
        multi_layer_sub_id = await self.nats_client.subscribe(
            "agents.validation.multi_layered.request",
            self._handle_multi_layered_validation_request,
            queue_group="multi-layered-validation-workers"
        )
        self.subscription_ids.append(multi_layer_sub_id)

        # Subscribe to pattern discovery for automatic validation
        discovery_sub_id = await self.messaging.subscribe_to_discoveries(
            self._handle_discovery_for_validation
        )
        self.subscription_ids.append(discovery_sub_id)

    async def handle_task(self, task: Task) -> None:
        """Handle multi-layered validation tasks."""
        self.logger.info(f"Processing multi-layered validation task {task.task_id}: {task.task_type}")

        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)

            result = None

            if task.task_type == "multi_layered_pattern_validation":
                result = await self.validate_pattern_multi_layered(
                    task.parameters.get("pattern_id"),
                    task.input_data.get("pattern_data", {}),
                    task.parameters.get("validation_layers", ["statistical", "cultural", "ethical"]),
                    task.parameters.get("store_results", True)
                )
            elif task.task_type == "cultural_relevance_assessment":
                result = await self.assess_cultural_relevance(
                    task.input_data.get("pattern_data", {}),
                    task.parameters.get("cultural_context", {})
                )
            elif task.task_type == "ethical_impact_evaluation":
                result = await self.evaluate_ethical_impact(
                    task.input_data.get("pattern_data", {}),
                    task.parameters.get("stakeholder_analysis", {})
                )
            elif task.task_type == "bias_diversity_audit":
                result = await self.audit_bias_diversity(
                    task.input_data.get("validation_results", {}),
                    task.parameters.get("diversity_criteria", {})
                )
            else:
                raise ValueError(f"Unknown multi-layered validation task type: {task.task_type}")

            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)

            self.processed_tasks += 1
            self.validation_metrics["total_validations"] += 1

            self.logger.info(f"Completed multi-layered validation task {task_id}")

        except Exception as e:
            self.logger.error(f"Error processing multi-layered validation task {task.task_id}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)

            self.failed_tasks += 1

        finally:
            self.current_tasks.discard(task.task_id)

    async def validate_pattern_multi_layered(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any],
        validation_layers: List[str],
        store_results: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive multi-layered pattern validation.

        Args:
            pattern_id: Pattern identifier
            pattern_data: Pattern data for validation
            validation_layers: List of validation layers to execute
            store_results: Whether to store results in database

        Returns:
            Multi-layered validation results
        """
        try:
            start_time = datetime.utcnow()
            self.logger.info(f"Starting multi-layered validation for pattern {pattern_id}")

            # Check cache first
            cache_key = f"{pattern_id}:{':'.join(sorted(validation_layers))}"
            if cache_key in self.validation_cache:
                self.logger.debug(f"Returning cached multi-layered validation for {pattern_id}")
                return self.validation_cache[cache_key]

            # Initialize results structure
            multi_layered_results = {
                "pattern_id": pattern_id,
                "validation_timestamp": start_time.isoformat(),
                "validation_layers": validation_layers,
                "layer_results": {},
                "integrated_assessment": {},
                "validation_metadata": {},
                "recommendations": [],
                "execution_times": {}
            }

            # Execute validation layers in parallel where possible
            layer_tasks = []
            for layer in validation_layers:
                if layer in self.validation_layers:
                    task = asyncio.create_task(
                        self._execute_validation_layer(layer, pattern_id, pattern_data)
                    )
                    layer_tasks.append((layer, task))

            # Wait for all layers to complete
            for layer, task in layer_tasks:
                try:
                    layer_start = datetime.utcnow()
                    layer_result = await task
                    layer_end = datetime.utcnow()

                    execution_time = (layer_end - layer_start).total_seconds()
                    self.layer_execution_times[layer] = execution_time

                    multi_layered_results["layer_results"][layer] = layer_result
                    multi_layered_results["execution_times"][layer] = execution_time

                    if layer_result and "error" not in layer_result:
                        self.logger.info(f"Layer {layer} validation completed successfully")
                    else:
                        self.logger.warning(f"Layer {layer} validation failed: {layer_result.get('error', 'Unknown error')}")

                except Exception as e:
                    self.logger.error(f"Layer {layer} execution failed: {e}")
                    multi_layered_results["layer_results"][layer] = {"error": str(e)}

            # Perform integrated assessment
            multi_layered_results["integrated_assessment"] = self._perform_integrated_assessment(
                multi_layered_results["layer_results"]
            )

            # Generate comprehensive recommendations
            multi_layered_results["recommendations"] = self._generate_multidisciplinary_recommendations(
                multi_layered_results
            )

            # Add validation metadata
            end_time = datetime.utcnow()
            total_execution_time = (end_time - start_time).total_seconds()

            multi_layered_results["validation_metadata"] = {
                "total_execution_time": total_execution_time,
                "layers_executed": len(validation_layers),
                "successful_layers": sum(1 for r in multi_layered_results["layer_results"].values()
                                       if r and "error" not in r),
                "validation_agent": self.agent_id,
                "validation_version": "1.0.0",
                "multidisciplinary_score": multi_layered_results["integrated_assessment"].get("overall_multidisciplinary_score", 0.0)
            }

            # Store results if requested
            if store_results:
                try:
                    stored_id = await self._store_multi_layered_validation_results(
                        pattern_id, multi_layered_results
                    )
                    multi_layered_results["stored_validation_id"] = stored_id
                    self.logger.info(f"Multi-layered validation results stored with ID: {stored_id}")
                except Exception as e:
                    self.logger.error(f"Failed to store multi-layered validation results: {e}")
                    multi_layered_results["storage_error"] = str(e)

            # Cache results
            self.validation_cache[cache_key] = multi_layered_results

            # Update metrics
            self.validation_metrics["average_execution_time"] = (
                (self.validation_metrics["average_execution_time"] * (self.validation_metrics["total_validations"] - 1)) +
                total_execution_time
            ) / self.validation_metrics["total_validations"]

            return multi_layered_results

        except Exception as e:
            self.logger.error(f"Multi-layered pattern validation failed for {pattern_id}: {e}")
            return {
                "pattern_id": pattern_id,
                "error": str(e),
                "validation_timestamp": datetime.utcnow().isoformat(),
                "validation_layers": validation_layers,
                "layer_results": {},
                "integrated_assessment": {"validation_success": False}
            }

    async def _execute_validation_layer(
        self,
        layer_name: str,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific validation layer."""
        try:
            if layer_name not in self.validation_layers:
                raise ValueError(f"Unknown validation layer: {layer_name}")

            return await self.validation_layers[layer_name](pattern_id, pattern_data)

        except Exception as e:
            self.logger.error(f"Validation layer {layer_name} failed: {e}")
            return {"error": str(e), "layer": layer_name}

    # Validation Layer Implementations

    async def _statistical_validation_layer(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute statistical validation layer."""
        try:
            # Use the enhanced statistical validator
            statistical_results = await self.statistical_validator.validate_pattern_enhanced(
                pattern_id=pattern_id,
                pattern_data=pattern_data,
                validation_methods=["full_statistical_suite"],
                store_results=False
            )

            return {
                "layer": "statistical",
                "validation_success": statistical_results.get("enhanced_metrics", {}).get("validation_success", False),
                "statistical_results": statistical_results.get("statistical_results", []),
                "significance_classification": statistical_results.get("significance_classification", {}),
                "reliability_score": statistical_results.get("enhanced_metrics", {}).get("reliability_score", 0.0),
                "detailed_results": statistical_results
            }

        except Exception as e:
            self.logger.error(f"Statistical validation layer failed: {e}")
            return {"layer": "statistical", "error": str(e)}

    async def _cultural_validation_layer(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute cultural validation layer."""
        try:
            # Assess cultural relevance
            cultural_relevance = await self.cultural_validator.assess_cultural_relevance(
                pattern_data, {}
            )

            # Analyze mythological context
            mythological_context = await self.mythological_analyzer.analyze_mythological_context(
                pattern_data
            )

            # Combine cultural assessments
            cultural_score = (
                cultural_relevance.get("cultural_relevance_score", 0.0) * 0.6 +
                mythological_context.get("mythological_alignment_score", 0.0) * 0.4
            )

            return {
                "layer": "cultural",
                "validation_success": cultural_score > 0.3,
                "cultural_relevance_score": cultural_score,
                "cultural_relevance_assessment": cultural_relevance,
                "mythological_context_analysis": mythological_context,
                "cultural_sensitivity_check": cultural_relevance.get("sensitivity_assessment", {}),
                "recommendations": cultural_relevance.get("cultural_recommendations", [])
            }

        except Exception as e:
            self.logger.error(f"Cultural validation layer failed: {e}")
            return {"layer": "cultural", "error": str(e)}

    async def _ethical_validation_layer(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute ethical validation layer."""
        try:
            # Assess human flourishing alignment
            flourishing_alignment = await self.ethical_validator.assess_human_flourishing(
                pattern_data
            )

            # Audit bias and diversity
            bias_audit = await self.bias_assessor.audit_bias_diversity(
                pattern_data
            )

            # Calculate ethical compliance score
            ethical_score = (
                flourishing_alignment.get("flourishing_score", 0.0) * 0.7 +
                bias_audit.get("diversity_score", 0.0) * 0.3
            )

            return {
                "layer": "ethical",
                "validation_success": ethical_score > 0.4,
                "ethical_compliance_score": ethical_score,
                "human_flourishing_assessment": flourishing_alignment,
                "bias_diversity_audit": bias_audit,
                "ethical_concerns": flourishing_alignment.get("ethical_concerns", []),
                "diversity_recommendations": bias_audit.get("diversity_recommendations", [])
            }

        except Exception as e:
            self.logger.error(f"Ethical validation layer failed: {e}")
            return {"layer": "ethical", "error": str(e)}

    async def _consensus_validation_layer(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute consensus validation layer."""
        try:
            # Use consensus validation with multidisciplinary focus
            consensus_results = await self.consensus_validator.validate_pattern_with_consensus(
                pattern_id=pattern_id,
                pattern_data=pattern_data,
                consensus_enabled=True,
                voting_mechanism="adaptive",
                min_participants=2,
                timeout_seconds=15
            )

            return {
                "layer": "consensus",
                "validation_success": consensus_results.get("consensus_results") is not None,
                "consensus_results": consensus_results.get("consensus_results"),
                "peer_agreement_score": consensus_results.get("consensus_metadata", {}).get("consensus_agreement", 0.0),
                "participating_agents": consensus_results.get("consensus_metadata", {}).get("participating_agents", []),
                "consensus_confidence": consensus_results.get("consensus_results", {}).get("confidence", 0.0)
            }

        except Exception as e:
            self.logger.error(f"Consensus validation layer failed: {e}")
            return {"layer": "consensus", "error": str(e)}

    async def _integrated_validation_layer(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute integrated validation across all layers."""
        try:
            # Run all layers and integrate results
            all_layers = ["statistical", "cultural", "ethical", "consensus"]
            integrated_results = {}

            for layer in all_layers:
                result = await self._execute_validation_layer(layer, pattern_id, pattern_data)
                integrated_results[layer] = result

            # Perform cross-layer integration
            integrated_assessment = self._perform_integrated_assessment(integrated_results)

            return {
                "layer": "integrated",
                "validation_success": integrated_assessment.get("overall_validation_success", False),
                "integrated_assessment": integrated_assessment,
                "layer_results": integrated_results,
                "multidisciplinary_score": integrated_assessment.get("overall_multidisciplinary_score", 0.0)
            }

        except Exception as e:
            self.logger.error(f"Integrated validation layer failed: {e}")
            return {"layer": "integrated", "error": str(e)}

    def _perform_integrated_assessment(self, layer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integrated assessment across all validation layers."""
        try:
            assessment = {
                "overall_validation_success": True,
                "layer_success_rates": {},
                "weighted_scores": {},
                "overall_multidisciplinary_score": 0.0,
                "validation_strengths": [],
                "validation_concerns": [],
                "integrated_classification": "unknown"
            }

            total_weight = 0.0
            weighted_sum = 0.0

            for layer_name, layer_result in layer_results.items():
                if layer_result and "error" not in layer_result:
                    success = layer_result.get("validation_success", False)
                    assessment["layer_success_rates"][layer_name] = success

                    # Calculate layer-specific score
                    layer_score = self._calculate_layer_score(layer_name, layer_result)
                    assessment["weighted_scores"][layer_name] = layer_score

                    # Apply layer weight
                    weight = self.layer_weights.get(layer_name, 0.25)
                    weighted_sum += layer_score * weight
                    total_weight += weight

                    # Track strengths and concerns
                    if success:
                        assessment["validation_strengths"].append(f"{layer_name} validation passed")
                    else:
                        assessment["validation_concerns"].append(f"{layer_name} validation failed")
                        assessment["overall_validation_success"] = False
                else:
                    assessment["layer_success_rates"][layer_name] = False
                    assessment["validation_concerns"].append(f"{layer_name} validation error")
                    assessment["overall_validation_success"] = False

            # Calculate overall score
            if total_weight > 0:
                assessment["overall_multidisciplinary_score"] = weighted_sum / total_weight

            # Determine integrated classification
            overall_score = assessment["overall_multidisciplinary_score"]
            if overall_score >= 0.8:
                assessment["integrated_classification"] = "excellent_multidisciplinary_validation"
            elif overall_score >= 0.6:
                assessment["integrated_classification"] = "good_multidisciplinary_validation"
            elif overall_score >= 0.4:
                assessment["integrated_classification"] = "moderate_multidisciplinary_validation"
            elif overall_score >= 0.2:
                assessment["integrated_classification"] = "weak_multidisciplinary_validation"
            else:
                assessment["integrated_classification"] = "poor_multidisciplinary_validation"

            return assessment

        except Exception as e:
            self.logger.error(f"Integrated assessment failed: {e}")
            return {"error": str(e), "overall_validation_success": False}

    def _calculate_layer_score(self, layer_name: str, layer_result: Dict[str, Any]) -> float:
        """Calculate score for a specific validation layer."""
        try:
            if layer_name == "statistical":
                return layer_result.get("reliability_score", 0.0)
            elif layer_name == "cultural":
                return layer_result.get("cultural_relevance_score", 0.0)
            elif layer_name == "ethical":
                return layer_result.get("ethical_compliance_score", 0.0)
            elif layer_name == "consensus":
                return layer_result.get("peer_agreement_score", 0.0)
            else:
                return 0.0
        except Exception:
            return 0.0

    def _generate_multidisciplinary_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate multidisciplinary recommendations based on validation results."""
        recommendations = []

        try:
            integrated_assessment = validation_results.get("integrated_assessment", {})
            layer_results = validation_results.get("layer_results", {})

            # Overall assessment recommendations
            overall_score = integrated_assessment.get("overall_multidisciplinary_score", 0.0)
            if overall_score >= 0.8:
                recommendations.append("Excellent multidisciplinary validation - pattern suitable for advanced applications")
            elif overall_score >= 0.6:
                recommendations.append("Good multidisciplinary validation - proceed with confidence")
            elif overall_score >= 0.4:
                recommendations.append("Moderate multidisciplinary validation - additional review recommended")
            else:
                recommendations.append("Poor multidisciplinary validation - comprehensive review required")

            # Layer-specific recommendations
            for layer_name, layer_result in layer_results.items():
                if layer_result and "error" in layer_result:
                    recommendations.append(f"Address {layer_name} validation errors before proceeding")
                elif not layer_result.get("validation_success", False):
                    recommendations.append(f"Improve {layer_name} validation compliance")

            # Cross-layer integration recommendations
            successful_layers = sum(1 for r in layer_results.values()
                                  if r and "error" not in r and r.get("validation_success", False))
            if successful_layers < len(layer_results):
                recommendations.append("Enhance integration between validation layers for better multidisciplinary assessment")

        except Exception as e:
            self.logger.error(f"Error generating multidisciplinary recommendations: {e}")
            recommendations.append("Manual review of validation results recommended due to processing error")

        return recommendations

    # Additional validation methods

    async def assess_cultural_relevance(
        self,
        pattern_data: Dict[str, Any],
        cultural_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess cultural relevance of patterns."""
        return await self.cultural_validator.assess_cultural_relevance(pattern_data, cultural_context)

    async def evaluate_ethical_impact(
        self,
        pattern_data: Dict[str, Any],
        stakeholder_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate ethical impact and human flourishing alignment."""
        flourishing = await self.ethical_validator.assess_human_flourishing(pattern_data)
        bias_audit = await self.bias_assessor.audit_bias_diversity(pattern_data)

        return {
            "ethical_evaluation": {
                "human_flourishing": flourishing,
                "bias_diversity": bias_audit,
                "overall_ethical_score": (flourishing.get("flourishing_score", 0.0) +
                                        bias_audit.get("diversity_score", 0.0)) / 2
            },
            "stakeholder_impact": stakeholder_analysis,
            "recommendations": flourishing.get("ethical_concerns", []) + bias_audit.get("diversity_recommendations", [])
        }

    async def audit_bias_diversity(
        self,
        validation_results: Dict[str, Any],
        diversity_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Audit bias and diversity in validation results."""
        return await self.bias_assessor.audit_bias_diversity(validation_results)

    # Database and storage methods

    async def _store_multi_layered_validation_results(
        self,
        pattern_id: str,
        validation_results: Dict[str, Any]
    ) -> str:
        """Store multi-layered validation results in database."""
        try:
            # Create comprehensive validation record
            validation_record = {
                "pattern_id": pattern_id,
                "validation_type": "multi_layered_validation",
                "validator_type": "agent",
                "validator_id": self.agent_id,
                "validation_result": validation_results["integrated_assessment"].get("integrated_classification", "unknown"),
                "validation_score": validation_results["integrated_assessment"].get("overall_multidisciplinary_score", 0.0),
                "confidence_level": validation_results["validation_metadata"].get("multidisciplinary_score", 0.0),
                "validation_method": "multidisciplinary_validation_framework",
                "test_statistics": validation_results,
                "validation_notes": f"Multi-layered validation across {len(validation_results['validation_layers'])} dimensions",
                "replication_successful": validation_results["integrated_assessment"].get("overall_validation_success", False),
                "recommendations": "; ".join(validation_results.get("recommendations", []))
            }

            # Store in database
            validation_id = await self.pattern_storage.store_pattern_validation(validation_record)

            return validation_id

        except Exception as e:
            self.logger.error(f"Failed to store multi-layered validation results: {e}")
            raise

    async def _initialize_validation_storage(self) -> None:
        """Initialize validation storage connections."""
        try:
            if not self.pattern_storage.initialized:
                await self.pattern_storage.initialize()
            self.logger.info("Multi-layered validation storage initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize validation storage: {e}")
            raise

    async def _load_validation_cache(self) -> None:
        """Load validation cache from database."""
        # Implementation would load recent validation results
        pass

    async def _cleanup_validation_cache(self) -> None:
        """Clean up old validation cache entries."""
        current_time = datetime.utcnow()
        cache_ttl = 3600  # 1 hour for multi-layered validation cache

        keys_to_remove = []
        for key, result in self.validation_cache.items():
            if "validation_timestamp" in result:
                try:
                    result_time = datetime.fromisoformat(result["validation_timestamp"])
                    if (current_time - result_time).total_seconds() > cache_ttl:
                        keys_to_remove.append(key)
                except (ValueError, KeyError):
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.validation_cache[key]

        if keys_to_remove:
            self.logger.info(f"Cleaned up {len(keys_to_remove)} multi-layered validation cache entries")

    # Message handlers

    async def _handle_multi_layered_validation_request(self, message: AgentMessage) -> None:
        """Handle multi-layered validation requests."""
        try:
            request_data = message.payload
            pattern_id = request_data.get("pattern_id")
            validation_layers = request_data.get("validation_layers", ["statistical", "cultural", "ethical"])
            pattern_data = request_data.get("pattern_data", {})

            # Perform multi-layered validation
            results = await self.validate_pattern_multi_layered(
                pattern_id, pattern_data, validation_layers
            )

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="multi_layered_validation_response",
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
            self.logger.error(f"Error handling multi-layered validation request: {e}")

    async def _handle_discovery_for_validation(self, message: AgentMessage) -> None:
        """Handle pattern discovery events for automatic validation."""
        try:
            if message.message_type == "pattern_discovered":
                discovery_data = message.payload
                pattern_id = discovery_data.get("pattern_id")

                if pattern_id:
                    # Queue multi-layered validation for newly discovered patterns
                    validation_task = Task(
                        task_id=str(uuid.uuid4()),
                        task_type="multi_layered_pattern_validation",
                        parameters={
                            "pattern_id": pattern_id,
                            "validation_layers": ["statistical", "cultural", "ethical"],
                            "store_results": True
                        },
                        input_data={"pattern_data": discovery_data},
                        priority=3
                    )

                    if self.task_queue:
                        await self.task_queue.add_task(validation_task)

                    self.logger.info(f"Queued multi-layered validation for discovered pattern {pattern_id}")

        except Exception as e:
            self.logger.error(f"Error handling discovery for validation: {e}")

    async def _process_pending_validations(self) -> None:
        """Process any pending multi-layered validations."""
        # Implementation for processing queued validations
        pass

    async def _update_validation_metrics(self) -> None:
        """Update validation performance metrics."""
        try:
            # Update layer success rates
            for layer_name in self.validation_layers.keys():
                if layer_name in self.validation_cache:
                    recent_results = [r for r in self.validation_cache.values()
                                    if layer_name in r.get("layer_results", {})]
                    if recent_results:
                        success_rate = sum(1 for r in recent_results
                                         if r["layer_results"][layer_name].get("validation_success", False)) / len(recent_results)
                        self.validation_metrics["layer_success_rates"][layer_name] = success_rate

        except Exception as e:
            self.logger.error(f"Error updating validation metrics: {e}")

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect multi-layered validation metrics."""
        base_metrics = await super().collect_metrics() or {}

        multi_layered_metrics = {
            "total_multilayered_validations": self.validation_metrics["total_validations"],
            "average_execution_time": self.validation_metrics["average_execution_time"],
            "multidisciplinary_score": self.validation_metrics["multidisciplinary_score"],
            "layer_success_rates": self.validation_metrics["layer_success_rates"],
            "cache_size": len(self.validation_cache),
            "layer_execution_times": self.layer_execution_times,
            "validation_layers_available": len(self.validation_layers)
        }

        return {**base_metrics, **multi_layered_metrics}

    def _get_capabilities(self) -> List[str]:
        """Get multi-layered validation agent capabilities."""
        base_capabilities = [
            "multi_layered_validation_agent",
            "multidisciplinary_validation",
            "cultural_relevance_assessment",
            "ethical_impact_evaluation",
            "bias_diversity_audit",
            "statistical_validation",
            "consensus_integration",
            "integrated_assessment"
        ]

        # Add layer-specific capabilities
        for layer in self.validation_layers.keys():
            base_capabilities.append(f"validation_layer_{layer}")

        return base_capabilities

    async def shutdown(self) -> None:
        """Shutdown multi-layered validation agent."""
        try:
            # Shutdown validation layers
            if hasattr(self, 'statistical_validator'):
                await self.statistical_validator.shutdown()
            if hasattr(self, 'consensus_validator'):
                await self.consensus_validator.shutdown()

            # Shutdown thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)

            # Call parent shutdown
            await super().shutdown()

            self.logger.info("Multi-layered ValidationAgent shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during multi-layered validation agent shutdown: {e}")


# Factory function
def create_multi_layered_validation_agent(agent_id: Optional[str] = None, **kwargs) -> MultiLayeredValidationAgent:
    """
    Factory function to create multi-layered validation agents.

    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured multi-layered validation agent
    """
    return MultiLayeredValidationAgent(agent_id=agent_id, **kwargs)


# Main entry point
async def main():
    """Main entry point for running the Multi-layered ValidationAgent."""
    import signal
    import sys

    # Create and configure agent
    agent = MultiLayeredValidationAgent()

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
        print(f"Multi-layered ValidationAgent failed: {e}")
        sys.exit(1)

    print("Multi-layered ValidationAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())