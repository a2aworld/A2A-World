"""
A2A World Platform - Narrative XAI Agent

Agent responsible for generating explainable AI explanations with narrative-driven
approaches, Chain of Thought reasoning, and multimodal content for pattern discovery
and validation results.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import traceback

from agents.core.base_agent import BaseAgent
from agents.core.config import XAIConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.pattern_storage import PatternStorage

# Import XAI database models
from database.models.xai_explanations import (
    XAIExplanation, CoTReasoningChain, MultimodalElement,
    NarrativeTemplate, XAIRequestLog
)
from database.models.patterns import Pattern, PatternValidation

# Import existing agents for integration
from agents.discovery.pattern_discovery import PatternDiscoveryAgent
from agents.validation.enhanced_validation_agent import EnhancedValidationAgent


class NarrativeXAIConfig:
    """Configuration for the Narrative XAI Agent."""

    def __init__(self):
        self.max_narrative_length = 2000
        self.default_audience_level = "intermediate"
        self.cot_max_steps = 10
        self.enable_multimodal = True
        self.cache_explanations = True
        self.cache_ttl_hours = 24
        self.generation_timeout_seconds = 30
        self.quality_threshold = 0.7


class NarrativeXAIAgent(BaseAgent):
    """
    Narrative-driven Explainable AI Agent.

    Generates human-understandable explanations for AI decisions using:
    - Chain of Thought (CoT) reasoning
    - Narrative story generation
    - Multimodal explanations (text, visual, interactive)
    - Integration with pattern discovery and validation agents
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[NarrativeXAIConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="narrative_xai",
            config=config or NarrativeXAIConfig(),
            config_file=config_file
        )

        # XAI-specific components
        self.pattern_storage = PatternStorage()
        self.xai_config = config or NarrativeXAIConfig()

        # Explanation cache
        self.explanation_cache: Dict[str, Dict[str, Any]] = {}
        self.narrative_templates: Dict[str, Dict[str, Any]] = {}

        # Thread pool for narrative generation
        self.generation_executor = ThreadPoolExecutor(max_workers=4)

        # Statistics
        self.explanations_generated = 0
        self.cot_reasoning_performed = 0
        self.multimodal_elements_created = 0
        self.narrative_generation_errors = 0
        self.average_generation_time_ms = 0

        # Integration agents (will be set during initialization)
        self.pattern_discovery_agent = None
        self.validation_agent = None

        self.logger.info(f"NarrativeXAI Agent {self.agent_id} initialized with narrative-driven XAI capabilities")

    async def agent_initialize(self) -> None:
        """
        XAI agent specific initialization.
        """
        try:
            # Load narrative templates
            await self._load_narrative_templates()

            # Initialize database connections
            await self._initialize_xai_storage()

            # Setup integration with other agents
            await self._setup_agent_integration()

            # Load cached explanations if available
            await self._load_explanation_cache()

            self.logger.info("NarrativeXAI Agent initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize NarrativeXAI Agent: {e}")
            raise

    async def process(self) -> None:
        """
        Main processing loop for XAI explanation generation.
        """
        try:
            # Process any pending explanation requests
            await self._process_explanation_queue()

            # Clean up old cache entries periodically
            if self.processed_tasks % 100 == 0:
                await self._cleanup_explanation_cache()

            # Update narrative templates usage statistics
            if self.processed_tasks % 200 == 0:
                await self._update_template_statistics()

        except Exception as e:
            self.logger.error(f"Error in XAI processing: {e}")

    async def setup_subscriptions(self) -> None:
        """
        Setup XAI-specific message subscriptions.
        """
        if not self.messaging:
            return

        # Subscribe to XAI explanation requests
        xai_request_sub_id = await self.nats_client.subscribe(
            "agents.xai.explanation.request",
            self._handle_explanation_request,
            queue_group="xai-explanation-workers"
        )
        self.subscription_ids.append(xai_request_sub_id)

        # Subscribe to pattern discovery results for auto-explanation
        discovery_results_sub_id = await self.nats_client.subscribe(
            "agents.discovery.results",
            self._handle_discovery_results,
            queue_group="xai-discovery-integration"
        )
        self.subscription_ids.append(discovery_results_sub_id)

        # Subscribe to validation results for auto-explanation
        validation_results_sub_id = await self.nats_client.subscribe(
            "agents.validation.results",
            self._handle_validation_results,
            queue_group="xai-validation-integration"
        )
        self.subscription_ids.append(validation_results_sub_id)

        # Subscribe to CoT reasoning requests
        cot_request_sub_id = await self.nats_client.subscribe(
            "agents.xai.cot.request",
            self._handle_cot_request,
            queue_group="xai-cot-workers"
        )
        self.subscription_ids.append(cot_request_sub_id)

    async def handle_task(self, task: Task) -> None:
        """
        Handle XAI explanation generation tasks.
        """
        self.logger.info(f"Processing XAI task {task.task_id}: {task.task_type}")

        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)

            result = None

            if task.task_type == "generate_explanation":
                result = await self._generate_explanation_task(task)
            elif task.task_type == "generate_narrative":
                result = await self._generate_narrative_task(task)
            elif task.task_type == "cot_reasoning":
                result = await self._cot_reasoning_task(task)
            elif task.task_type == "multimodal_explanation":
                result = await self._multimodal_explanation_task(task)
            else:
                raise ValueError(f"Unknown XAI task type: {task.task_type}")

            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)

            self.processed_tasks += 1
            self.explanations_generated += 1

            self.logger.info(f"Completed XAI task {task_id}")

        except Exception as e:
            self.logger.error(f"Error processing XAI task {task.task_id}: {e}")

            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)

            self.failed_tasks += 1
            self.narrative_generation_errors += 1

        finally:
            self.current_tasks.discard(task_id)

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect XAI-specific metrics.
        """
        base_metrics = await super().collect_metrics() or {}

        xai_metrics = {
            "explanations_generated": self.explanations_generated,
            "cot_reasoning_performed": self.cot_reasoning_performed,
            "multimodal_elements_created": self.multimodal_elements_created,
            "narrative_generation_errors": self.narrative_generation_errors,
            "average_generation_time_ms": self.average_generation_time_ms,
            "cache_size": len(self.explanation_cache),
            "templates_loaded": len(self.narrative_templates),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }

        return {**base_metrics, **xai_metrics}

    def _get_capabilities(self) -> List[str]:
        """
        Get XAI agent capabilities.
        """
        return [
            "narrative_xai",
            "explanation_generation",
            "cot_reasoning",
            "narrative_generation",
            "multimodal_explanations",
            "pattern_explanation",
            "validation_explanation",
            "story_driven_xai",
            "chain_of_thought",
            "interactive_explanations",
            "xai_integration",
            "narrative_templates",
            "explanation_caching"
        ]

    # Core XAI Methods

    async def generate_explanation(self, target_type: str, target_id: str,
                                 explanation_type: str = "comprehensive",
                                 audience_level: str = "intermediate") -> Dict[str, Any]:
        """
        Generate a comprehensive XAI explanation for a target (pattern/validation).

        Args:
            target_type: 'pattern' or 'validation'
            target_id: UUID of the target
            explanation_type: Type of explanation to generate
            audience_level: Target audience level

        Returns:
            Generated explanation with narrative, CoT, and multimodal elements
        """
        try:
            # Check cache first
            cache_key = f"{target_type}:{target_id}:{explanation_type}:{audience_level}"
            if cache_key in self.explanation_cache:
                self.logger.debug(f"Returning cached explanation for {cache_key}")
                return self.explanation_cache[cache_key]

            self.logger.info(f"Generating {explanation_type} explanation for {target_type} {target_id}")

            # Get target data
            target_data = await self._get_target_data(target_type, target_id)
            if not target_data:
                raise ValueError(f"Target {target_type} {target_id} not found")

            # Perform Chain of Thought reasoning
            cot_result = await self._perform_cot_reasoning(target_type, target_data, explanation_type)

            # Generate narrative explanation
            narrative_result = await self._generate_narrative_explanation(
                target_type, target_data, cot_result, audience_level
            )

            # Create multimodal elements if enabled
            multimodal_elements = []
            if self.xai_config.enable_multimodal:
                multimodal_elements = await self._generate_multimodal_elements(
                    target_type, target_data, narrative_result
                )

            # Combine into comprehensive explanation
            explanation = {
                "explanation_id": str(uuid.uuid4()),
                "target_type": target_type,
                "target_id": target_id,
                "explanation_type": explanation_type,
                "audience_level": audience_level,
                "generated_at": datetime.utcnow().isoformat(),
                "cot_reasoning": cot_result,
                "narrative_explanation": narrative_result,
                "multimodal_elements": multimodal_elements,
                "confidence_score": self._calculate_explanation_confidence(cot_result, narrative_result),
                "quality_metrics": self._assess_explanation_quality(narrative_result, multimodal_elements)
            }

            # Store in database
            stored_id = await self._store_explanation(explanation)
            explanation["stored_id"] = stored_id

            # Cache result
            if self.xai_config.cache_explanations:
                self.explanation_cache[cache_key] = explanation

            return explanation

        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {e}")
            return {
                "error": str(e),
                "target_type": target_type,
                "target_id": target_id,
                "generated_at": datetime.utcnow().isoformat()
            }

    async def _perform_cot_reasoning(self, target_type: str, target_data: Dict[str, Any],
                                   explanation_type: str) -> Dict[str, Any]:
        """
        Perform Chain of Thought reasoning for explanation generation.
        """
        try:
            reasoning_steps = []
            current_step = 1

            # Step 1: Observe and understand the target
            observation_step = self._cot_observe_target(target_type, target_data)
            reasoning_steps.append(observation_step)
            current_step += 1

            # Step 2: Analyze the data/components
            analysis_step = self._cot_analyze_data(target_type, target_data)
            reasoning_steps.append(analysis_step)
            current_step += 1

            # Step 3: Identify key patterns/insights
            insight_step = self._cot_identify_insights(target_type, target_data, reasoning_steps)
            reasoning_steps.append(insight_step)
            current_step += 1

            # Step 4: Consider alternative explanations
            alternatives_step = self._cot_consider_alternatives(target_type, target_data, insight_step)
            reasoning_steps.append(alternatives_step)
            current_step += 1

            # Step 5: Validate reasoning
            validation_step = self._cot_validate_reasoning(reasoning_steps)
            reasoning_steps.append(validation_step)

            # Calculate overall confidence
            confidence = self._calculate_cot_confidence(reasoning_steps)

            cot_result = {
                "reasoning_chain": reasoning_steps,
                "total_steps": len(reasoning_steps),
                "confidence_score": confidence,
                "reasoning_path": [step["step_type"] for step in reasoning_steps],
                "key_insights": [step for step in reasoning_steps if step.get("is_key_insight", False)]
            }

            self.cot_reasoning_performed += 1
            return cot_result

        except Exception as e:
            self.logger.error(f"CoT reasoning failed: {e}")
            return {
                "error": str(e),
                "reasoning_chain": [],
                "confidence_score": 0.0
            }

    async def _generate_narrative_explanation(self, target_type: str, target_data: Dict[str, Any],
                                            cot_result: Dict[str, Any], audience_level: str) -> Dict[str, Any]:
        """
        Generate a narrative (story-like) explanation.
        """
        try:
            # Select appropriate template
            template = self._select_narrative_template(target_type, audience_level)
            if not template:
                # Fallback to basic narrative generation
                return await self._generate_basic_narrative(target_type, target_data, cot_result, audience_level)

            # Fill template with data
            narrative_content = await self._fill_narrative_template(
                template, target_data, cot_result, audience_level
            )

            # Enhance with storytelling elements
            enhanced_narrative = self._enhance_with_storytelling(narrative_content, target_data)

            narrative_result = {
                "narrative_type": "story_driven",
                "template_used": template.get("template_name"),
                "title": enhanced_narrative.get("title", f"Understanding {target_type.title()}"),
                "introduction": enhanced_narrative.get("introduction", ""),
                "main_narrative": enhanced_narrative.get("body", ""),
                "conclusion": enhanced_narrative.get("conclusion", ""),
                "key_takeaways": enhanced_narrative.get("takeaways", []),
                "word_count": len(enhanced_narrative.get("full_text", "").split()),
                "reading_time_minutes": max(1, len(enhanced_narrative.get("full_text", "").split()) // 200)
            }

            return narrative_result

        except Exception as e:
            self.logger.error(f"Narrative generation failed: {e}")
            return {
                "error": str(e),
                "narrative_type": "error",
                "main_narrative": f"An error occurred while generating the narrative explanation: {str(e)}"
            }

    async def _generate_multimodal_elements(self, target_type: str, target_data: Dict[str, Any],
                                          narrative_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multimodal elements (charts, diagrams, etc.) for the explanation.
        """
        try:
            multimodal_elements = []

            # Generate appropriate visualizations based on target type
            if target_type == "pattern":
                elements = await self._generate_pattern_visualizations(target_data, narrative_result)
                multimodal_elements.extend(elements)
            elif target_type == "validation":
                elements = await self._generate_validation_visualizations(target_data, narrative_result)
                multimodal_elements.extend(elements)

            # Add interactive elements if supported
            interactive_elements = self._generate_interactive_elements(target_data)
            multimodal_elements.extend(interactive_elements)

            self.multimodal_elements_created += len(multimodal_elements)
            return multimodal_elements

        except Exception as e:
            self.logger.error(f"Multimodal generation failed: {e}")
            return []

    # CoT Reasoning Step Methods

    def _cot_observe_target(self, target_type: str, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Observe and understand the target."""
        return {
            "step_number": 1,
            "step_type": "observation",
            "description": f"Observing the {target_type} and its key characteristics",
            "reasoning_content": f"I am examining a {target_type} with the following key attributes: {self._summarize_target_data(target_data)}",
            "evidence_used": target_data,
            "confidence_level": 0.9,
            "is_key_insight": False
        }

    def _cot_analyze_data(self, target_type: str, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Analyze the data/components."""
        analysis = self._analyze_target_components(target_data)

        return {
            "step_number": 2,
            "step_type": "analysis",
            "description": "Analyzing the components and structure of the data",
            "reasoning_content": f"Through analysis, I can see that this {target_type} consists of: {analysis['summary']}",
            "evidence_used": analysis,
            "confidence_level": analysis.get("confidence", 0.8),
            "is_key_insight": True
        }

    def _cot_identify_insights(self, target_type: str, target_data: Dict[str, Any],
                             previous_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 3: Identify key patterns/insights."""
        insights = self._extract_key_insights(target_data, previous_steps)

        return {
            "step_number": 3,
            "step_type": "insight_identification",
            "description": "Identifying the most important insights and patterns",
            "reasoning_content": f"The key insights I've identified are: {insights['summary']}",
            "evidence_used": insights,
            "confidence_level": insights.get("confidence", 0.7),
            "is_key_insight": True
        }

    def _cot_consider_alternatives(self, target_type: str, target_data: Dict[str, Any],
                                 insight_step: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Consider alternative explanations."""
        alternatives = self._generate_alternative_explanations(target_data, insight_step)

        return {
            "step_number": 4,
            "step_type": "alternative_consideration",
            "description": "Considering alternative ways to interpret the data",
            "reasoning_content": f"Alternative explanations could include: {alternatives['summary']}. However, the primary explanation seems most likely because {alternatives['rationale']}",
            "evidence_used": alternatives,
            "confidence_level": 0.8,
            "is_key_insight": False
        }

    def _cot_validate_reasoning(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 5: Validate the overall reasoning."""
        validation = self._validate_reasoning_chain(reasoning_steps)

        return {
            "step_number": 5,
            "step_type": "validation",
            "description": "Validating the reasoning chain and conclusions",
            "reasoning_content": f"Reviewing the reasoning process: {validation['assessment']}",
            "evidence_used": validation,
            "confidence_level": validation.get("confidence", 0.8),
            "is_key_insight": False
        }

    # Helper Methods

    def _summarize_target_data(self, target_data: Dict[str, Any]) -> str:
        """Create a summary of target data for CoT reasoning."""
        if "name" in target_data:
            return f"name: {target_data['name']}, type: {target_data.get('pattern_type', 'unknown')}"
        elif "validation_result" in target_data:
            return f"result: {target_data['validation_result']}, score: {target_data.get('validation_score', 'unknown')}"
        else:
            return f"complex data structure with {len(target_data)} attributes"

    def _analyze_target_components(self, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the components of the target data."""
        components = []

        if "pattern_components" in target_data:
            components = target_data["pattern_components"]
        elif "statistical_results" in target_data:
            components = target_data["statistical_results"]

        return {
            "component_count": len(components),
            "summary": f"{len(components)} key components identified",
            "confidence": 0.8
        }

    def _extract_key_insights(self, target_data: Dict[str, Any],
                            previous_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key insights from the data and reasoning."""
        insights = []

        if "confidence_score" in target_data:
            insights.append(f"confidence level of {target_data['confidence_score']}")
        if "statistical_significance" in target_data:
            insights.append(f"statistical significance (p={target_data['statistical_significance']})")

        return {
            "insights": insights,
            "summary": "; ".join(insights) if insights else "No specific insights identified",
            "confidence": 0.7
        }

    def _generate_alternative_explanations(self, target_data: Dict[str, Any],
                                         insight_step: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alternative explanations for robustness."""
        return {
            "alternatives": ["random chance", "data artifacts", "alternative interpretation"],
            "summary": "random chance, data artifacts, or alternative interpretations",
            "rationale": "the data shows clear patterns that are unlikely to occur by chance",
            "confidence": 0.8
        }

    def _validate_reasoning_chain(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the overall reasoning chain."""
        total_confidence = sum(step.get("confidence_level", 0) for step in reasoning_steps)
        avg_confidence = total_confidence / len(reasoning_steps) if reasoning_steps else 0

        return {
            "assessment": f"Reasoning chain appears {'strong' if avg_confidence > 0.7 else 'moderate'} with average confidence of {avg_confidence:.2f}",
            "step_count": len(reasoning_steps),
            "confidence": avg_confidence
        }

    def _calculate_cot_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in CoT reasoning."""
        if not reasoning_steps:
            return 0.0

        # Weight later steps more heavily as they build on previous reasoning
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # For 5 steps
        weights = weights[:len(reasoning_steps)]

        weighted_sum = sum(step.get("confidence_level", 0) * weight
                          for step, weight in zip(reasoning_steps, weights))

        return min(1.0, weighted_sum / sum(weights))

    # Database and Storage Methods

    async def _store_explanation(self, explanation: Dict[str, Any]) -> str:
        """Store explanation in database."""
        try:
            # This would integrate with database session
            # For now, return a mock ID
            return str(uuid.uuid4())
        except Exception as e:
            self.logger.error(f"Failed to store explanation: {e}")
            return str(uuid.uuid4())

    async def _get_target_data(self, target_type: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get target data from database."""
        try:
            if target_type == "pattern":
                # Get pattern data
                return await self.pattern_storage.get_pattern(target_id)
            elif target_type == "validation":
                # Get validation data
                return await self.pattern_storage.get_validation(target_id)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to get target data: {e}")
            return None

    # Template and Cache Methods

    async def _load_narrative_templates(self) -> None:
        """Load narrative templates from database."""
        try:
            # Load default templates
            self.narrative_templates = {
                "pattern_discovery_intermediate": {
                    "template_name": "pattern_discovery_intermediate",
                    "introduction_template": "Imagine you're exploring an ancient landscape...",
                    "body_template": "What I discovered is a fascinating {pattern_type} pattern...",
                    "conclusion_template": "This pattern tells us something important about..."
                },
                "validation_result_intermediate": {
                    "template_name": "validation_result_intermediate",
                    "introduction_template": "When we validate scientific discoveries...",
                    "body_template": "The validation process revealed that...",
                    "conclusion_template": "This validation gives us confidence that..."
                }
            }
            self.logger.info(f"Loaded {len(self.narrative_templates)} narrative templates")
        except Exception as e:
            self.logger.error(f"Failed to load narrative templates: {e}")

    def _select_narrative_template(self, target_type: str, audience_level: str) -> Optional[Dict[str, Any]]:
        """Select appropriate narrative template."""
        template_key = f"{target_type}_{audience_level}"
        return self.narrative_templates.get(template_key)

    async def _fill_narrative_template(self, template: Dict[str, Any], target_data: Dict[str, Any],
                                     cot_result: Dict[str, Any], audience_level: str) -> Dict[str, Any]:
        """Fill narrative template with actual data."""
        try:
            # Simple template filling - in production this would be more sophisticated
            filled_narrative = {
                "title": f"Understanding the {target_data.get('name', 'Discovery')}",
                "introduction": template.get("introduction_template", ""),
                "body": template.get("body_template", "").format(
                    pattern_type=target_data.get('pattern_type', 'spatial'),
                    confidence=target_data.get('confidence_score', 0.8)
                ),
                "conclusion": template.get("conclusion_template", ""),
                "full_text": "Generated narrative explanation"
            }
            return filled_narrative
        except Exception as e:
            self.logger.error(f"Template filling failed: {e}")
            return {"error": str(e)}

    def _enhance_with_storytelling(self, narrative: Dict[str, Any], target_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance narrative with storytelling elements."""
        # Add storytelling enhancements
        enhanced = narrative.copy()
        enhanced["story_elements"] = ["context", "discovery", "meaning"]
        return enhanced

    # Visualization Methods

    async def _generate_pattern_visualizations(self, target_data: Dict[str, Any],
                                            narrative_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate visualizations for pattern explanations."""
        visualizations = []

        # Add cluster visualization if pattern has clusters
        if "clustering_results" in target_data:
            cluster_viz = {
                "element_type": "chart",
                "element_format": "json",
                "title": "Pattern Clusters",
                "description": "Visualization of discovered pattern clusters",
                "data": {"type": "scatter", "clusters": target_data["clustering_results"]}
            }
            visualizations.append(cluster_viz)

        return visualizations

    async def _generate_validation_visualizations(self, target_data: Dict[str, Any],
                                               narrative_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate visualizations for validation explanations."""
        visualizations = []

        # Add significance plot
        if "statistical_results" in target_data:
            sig_viz = {
                "element_type": "chart",
                "element_format": "json",
                "title": "Statistical Significance",
                "description": "Visualization of validation statistical results",
                "data": {"type": "bar", "results": target_data["statistical_results"]}
            }
            visualizations.append(sig_viz)

        return visualizations

    def _generate_interactive_elements(self, target_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate interactive elements for explanations."""
        return []  # Placeholder for interactive elements

    # Quality and Confidence Methods

    def _calculate_explanation_confidence(self, cot_result: Dict[str, Any],
                                        narrative_result: Dict[str, Any]) -> float:
        """Calculate overall confidence in the explanation."""
        cot_confidence = cot_result.get("confidence_score", 0.5)
        narrative_confidence = 0.8 if "error" not in narrative_result else 0.3

        return (cot_confidence + narrative_confidence) / 2

    def _assess_explanation_quality(self, narrative_result: Dict[str, Any],
                                  multimodal_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of the generated explanation."""
        quality = {
            "clarity_score": 0.8,
            "completeness_score": 0.7,
            "usefulness_score": 0.8,
            "multimodal_enhancement": len(multimodal_elements) > 0
        }

        if "error" in narrative_result:
            quality["overall_quality"] = 0.3
        else:
            quality["overall_quality"] = sum(quality.values()) / len(quality)

        return quality

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder)."""
        return 0.0  # Would track actual cache hits vs misses

    # Integration and Setup Methods

    async def _initialize_xai_storage(self) -> None:
        """Initialize XAI database storage."""
        try:
            # Initialize pattern storage if not already done
            if not self.pattern_storage.initialized:
                await self.pattern_storage.initialize()
            self.logger.info("XAI storage initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize XAI storage: {e}")
            raise

    async def _setup_agent_integration(self) -> None:
        """Setup integration with other agents."""
        try:
            # Note: In a real implementation, these would be actual agent instances
            # or references obtained through the registry
            self.pattern_discovery_agent = "integrated"
            self.validation_agent = "integrated"
            self.logger.info("Agent integration setup complete")
        except Exception as e:
            self.logger.error(f"Failed to setup agent integration: {e}")

    async def _load_explanation_cache(self) -> None:
        """Load explanation cache from database."""
        # Would load recent explanations from database
        pass

    # Task Handler Methods

    async def _generate_explanation_task(self, task: Task) -> Dict[str, Any]:
        """Handle explanation generation task."""
        params = task.parameters
        target_type = params.get("target_type")
        target_id = params.get("target_id")
        explanation_type = params.get("explanation_type", "comprehensive")
        audience_level = params.get("audience_level", "intermediate")

        if not target_type or not target_id:
            raise ValueError("target_type and target_id are required")

        return await self.generate_explanation(target_type, target_id, explanation_type, audience_level)

    async def _generate_narrative_task(self, task: Task) -> Dict[str, Any]:
        """Handle narrative generation task."""
        # Similar to explanation generation but focused on narrative
        return await self._generate_explanation_task(task)

    async def _cot_reasoning_task(self, task: Task) -> Dict[str, Any]:
        """Handle CoT reasoning task."""
        params = task.parameters
        target_type = params.get("target_type")
        target_id = params.get("target_id")

        if not target_type or not target_id:
            raise ValueError("target_type and target_id are required")

        target_data = await self._get_target_data(target_type, target_id)
        if not target_data:
            raise ValueError(f"Target {target_type} {target_id} not found")

        return await self._perform_cot_reasoning(target_type, target_data, "cot_only")

    async def _multimodal_explanation_task(self, task: Task) -> Dict[str, Any]:
        """Handle multimodal explanation task."""
        # Generate explanation with focus on multimodal elements
        return await self._generate_explanation_task(task)

    # Message Handlers

    async def _handle_explanation_request(self, message: AgentMessage) -> None:
        """Handle XAI explanation requests via NATS."""
        try:
            request_data = message.payload
            target_type = request_data.get("target_type")
            target_id = request_data.get("target_id")
            explanation_type = request_data.get("explanation_type", "comprehensive")
            audience_level = request_data.get("audience_level", "intermediate")

            # Generate explanation
            result = await self.generate_explanation(target_type, target_id, explanation_type, audience_level)

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="xai_explanation_response",
                payload={
                    "request_id": request_data.get("request_id"),
                    "explanation": result,
                    "timestamp": datetime.utcnow().isoformat()
                },
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling explanation request: {e}")

    async def _handle_discovery_results(self, message: AgentMessage) -> None:
        """Handle pattern discovery results for auto-explanation."""
        try:
            if message.message_type == "pattern_discovered":
                discovery_data = message.payload
                pattern_id = discovery_data.get("pattern_id")

                if pattern_id and discovery_data.get("auto_explain", False):
                    # Generate automatic explanation
                    explanation = await self.generate_explanation("pattern", pattern_id)

                    # Publish explanation
                    if self.messaging:
                        await self.messaging.publish_explanation(explanation)

                    self.logger.info(f"Auto-generated explanation for pattern {pattern_id}")

        except Exception as e:
            self.logger.error(f"Error handling discovery results: {e}")

    async def _handle_validation_results(self, message: AgentMessage) -> None:
        """Handle validation results for auto-explanation."""
        try:
            if message.message_type == "validation_completed":
                validation_data = message.payload
                validation_id = validation_data.get("validation_id")

                if validation_id and validation_data.get("auto_explain", False):
                    # Generate automatic explanation
                    explanation = await self.generate_explanation("validation", validation_id)

                    # Publish explanation
                    if self.messaging:
                        await self.messaging.publish_explanation(explanation)

                    self.logger.info(f"Auto-generated explanation for validation {validation_id}")

        except Exception as e:
            self.logger.error(f"Error handling validation results: {e}")

    async def _handle_cot_request(self, message: AgentMessage) -> None:
        """Handle CoT reasoning requests via NATS."""
        try:
            request_data = message.payload
            target_type = request_data.get("target_type")
            target_id = request_data.get("target_id")

            target_data = await self._get_target_data(target_type, target_id)
            if target_data:
                cot_result = await self._perform_cot_reasoning(target_type, target_data, "cot_only")

                # Send response
                response = AgentMessage.create(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type="cot_reasoning_response",
                    payload={
                        "request_id": request_data.get("request_id"),
                        "cot_result": cot_result,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    correlation_id=message.correlation_id
                )

                if message.reply_to:
                    await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling CoT request: {e}")

    # Maintenance Methods

    async def _process_explanation_queue(self) -> None:
        """Process queued explanation requests."""
        # Placeholder for queued processing
        pass

    async def _cleanup_explanation_cache(self) -> None:
        """Clean up old explanation cache entries."""
        try:
            current_time = datetime.utcnow()
            cache_ttl_seconds = self.xai_config.cache_ttl_hours * 3600

            keys_to_remove = []
            for key, explanation in self.explanation_cache.items():
                if "generated_at" in explanation:
                    try:
                        gen_time = datetime.fromisoformat(explanation["generated_at"])
                        if (current_time - gen_time).total_seconds() > cache_ttl_seconds:
                            keys_to_remove.append(key)
                    except (ValueError, KeyError):
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.explanation_cache[key]

            if keys_to_remove:
                self.logger.info(f"Cleaned up {len(keys_to_remove)} cached explanations")

        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")

    async def _update_template_statistics(self) -> None:
        """Update narrative template usage statistics."""
        # Placeholder for template statistics updates
        pass

    async def shutdown(self) -> None:
        """
        Enhanced shutdown procedure for XAI agent.
        """
        try:
            # Shutdown thread pool
            if hasattr(self, 'generation_executor'):
                self.generation_executor.shutdown(wait=True)

            # Call parent shutdown
            await super().shutdown()

            self.logger.info("NarrativeXAI Agent shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during XAI agent shutdown: {e}")


# Main entry point for running the agent
async def main():
    """
    Main entry point for running the NarrativeXAI Agent.
    """
    import signal
    import sys

    # Create and configure agent
    agent = NarrativeXAIAgent()

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
        print(f"NarrativeXAI Agent failed: {e}")
        sys.exit(1)

    print("NarrativeXAI Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())