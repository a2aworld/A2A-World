"""
A2A World Platform - Environmental Mythology Agent

Specialized agent for analyzing environmental determinants of mythology.
Examines how environmental factors, climate patterns, geological features,
and ecological conditions influence the development and content of mythological narratives.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import json
from collections import defaultdict
import statistics

from agents.core.base_agent import BaseAgent
from agents.core.config import DiscoveryAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.pattern_storage import PatternStorage

# Import integration components
from agents.parsers.data_processors.text_processor import TextProcessor
from agents.validation.multi_layered_validation_agent import MultiLayeredValidationAgent
from agents.discovery.pattern_discovery import PatternDiscoveryAgent


class EnvironmentalMythologyAgent(BaseAgent):
    """
    Agent specialized in environmental determinants of mythology.

    Capabilities:
    - Environmental-mythological correlations
    - Climate pattern analysis in narratives
    - Geological feature mythological associations
    - Ecological condition narrative influences
    - Seasonal cycle mythological representations
    - Natural disaster mythological explanations
    - Biodiversity mythological symbolism
    - Landscape transformation narrative analysis
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DiscoveryAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="environmental_mythology",
            config=config or DiscoveryAgentConfig(),
            config_file=config_file
        )

        # Environmental analysis parameters
        self.correlation_threshold = 0.6  # Threshold for environmental-mythological correlations
        self.climatic_influence_threshold = 0.5  # Threshold for climatic narrative influence
        self.geological_significance_threshold = 0.4  # Threshold for geological mythological significance
        self.ecological_representation_threshold = 0.3  # Threshold for ecological narrative representation

        # Environmental categories
        self.environmental_categories = {
            "climate": ["temperature", "precipitation", "seasonal_patterns", "extreme_weather"],
            "geology": ["mountains", "rivers", "caves", "volcanoes", "earthquakes"],
            "ecology": ["flora", "fauna", "biodiversity", "ecosystems", "habitats"],
            "hydrology": ["rivers", "lakes", "oceans", "water_sources", "flooding"],
            "atmospheric": ["storms", "lightning", "rainbows", "auroras", "celestial_events"]
        }

        # Mythological environmental motifs
        self.mythological_motifs = {
            "creation_myths": ["world_creation", "cosmic_eggs", "primordial_waters"],
            "destruction_myths": ["floods", "fires", "earthquakes", "apocalypses"],
            "seasonal_myths": ["death_rebirth", "fertility_cycles", "agricultural_cycles"],
            "landscape_myths": ["sacred_mountains", "holy_rivers", "enchanted_forests"]
        }

        # Performance tracking
        self.environmental_analyses_performed = 0
        self.mythological_correlations_found = 0
        self.climatic_patterns_identified = 0
        self.geological_associations_mapped = 0
        self.ecological_narratives_analyzed = 0

        # Integration components
        self.text_processor = None
        self.validation_agent = None
        self.pattern_discovery = None

        # Environmental knowledge base
        self.environmental_data = self._load_environmental_data()
        self.mythological_environmental_lexicon = self._load_mythological_environmental_lexicon()

        # Database integration
        self.pattern_storage = PatternStorage()

        self.logger.info(f"EnvironmentalMythologyAgent {self.agent_id} initialized")

    async def process(self) -> None:
        """
        Main processing loop for environmental mythology analysis.
        """
        try:
            # Process any pending environmental mythology requests
            await self._process_environmental_queue()

            # Perform periodic environmental-mythological correlation analysis
            if self.processed_tasks % 120 == 0:
                await self._perform_periodic_environmental_analysis()

        except Exception as e:
            self.logger.error(f"Error in environmental mythology process: {e}")

    async def agent_initialize(self) -> None:
        """
        Environmental mythology agent specific initialization.
        """
        try:
            # Initialize integration components
            await self._initialize_integrations()

            # Load environmental and mythological data
            await self._load_environmental_mythology_data()

            self.logger.info("EnvironmentalMythologyAgent initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize EnvironmentalMythologyAgent: {e}")
            raise

    async def _initialize_integrations(self) -> None:
        """Initialize integration with other agents and processors."""
        try:
            # Initialize text processor for mythological text analysis
            self.text_processor = TextProcessor()
            await self.text_processor.initialize()

            # Initialize validation agent for multi-layered validation
            self.validation_agent = MultiLayeredValidationAgent(
                agent_id=f"{self.agent_id}_validation"
            )
            await self.validation_agent.agent_initialize()

            # Initialize pattern discovery for environmental pattern analysis
            self.pattern_discovery = PatternDiscoveryAgent(
                agent_id=f"{self.agent_id}_discovery"
            )
            await self.pattern_discovery.agent_initialize()

            self.logger.info("Integration components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize integrations: {e}")
            raise

    async def setup_subscriptions(self) -> None:
        """
        Setup environmental mythology-specific message subscriptions.
        """
        if not self.messaging:
            return

        # Subscribe to environmental mythology analysis requests
        environmental_sub_id = await self.nats_client.subscribe(
            "agents.environmental_mythology.request",
            self._handle_environmental_request,
            queue_group="environmental-mythology-workers"
        )
        self.subscription_ids.append(environmental_sub_id)

        # Subscribe to environmental correlation analysis
        correlation_sub_id = await self.nats_client.subscribe(
            "agents.environmental.correlation.request",
            self._handle_correlation_request,
            queue_group="environmental-correlation"
        )
        self.subscription_ids.append(correlation_sub_id)

        # Subscribe to climatic mythology analysis
        climate_sub_id = await self.nats_client.subscribe(
            "agents.environmental.climate.request",
            self._handle_climate_request,
            queue_group="climate-mythology"
        )
        self.subscription_ids.append(climate_sub_id)

    async def handle_task(self, task: Task) -> None:
        """
        Handle environmental mythology analysis tasks.
        """
        self.logger.info(f"Processing environmental mythology task {task.task_id}: {task.task_type}")

        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)

            result = None

            if task.task_type == "environmental_mythological_correlation":
                result = await self.analyze_environmental_mythological_correlations(
                    task.input_data.get("environmental_data", {}),
                    task.input_data.get("mythological_texts", []),
                    task.parameters.get("correlation_method", "statistical")
                )
            elif task.task_type == "climatic_narrative_influence":
                result = await self.analyze_climatic_narrative_influence(
                    task.input_data.get("climate_data", {}),
                    task.input_data.get("narratives", []),
                    task.parameters.get("time_period", "annual")
                )
            elif task.task_type == "geological_mythological_associations":
                result = await self.analyze_geological_mythological_associations(
                    task.input_data.get("geological_features", []),
                    task.input_data.get("mythological_motifs", []),
                    task.parameters.get("association_type", "symbolic")
                )
            elif task.task_type == "ecological_mythological_symbols":
                result = await self.analyze_ecological_mythological_symbols(
                    task.input_data.get("ecological_data", {}),
                    task.input_data.get("mythological_symbols", []),
                    task.parameters.get("symbolic_analysis", "archetypal")
                )
            elif task.task_type == "seasonal_mythological_cycles":
                result = await self.analyze_seasonal_mythological_cycles(
                    task.input_data.get("seasonal_data", {}),
                    task.input_data.get("mythological_cycles", []),
                    task.parameters.get("cycle_analysis", "temporal")
                )
            else:
                raise ValueError(f"Unknown environmental mythology task type: {task.task_type}")

            # Store results in database
            if result:
                await self._store_environmental_results(task_id, result)

            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)

            self.processed_tasks += 1
            self.environmental_analyses_performed += 1

            # Update counters based on result type
            if result and "correlations" in result:
                self.mythological_correlations_found += len(result["correlations"])
            if result and "climatic_patterns" in result:
                self.climatic_patterns_identified += len(result["climatic_patterns"])
            if result and "geological_associations" in result:
                self.geological_associations_mapped += len(result["geological_associations"])
            if result and "ecological_narratives" in result:
                self.ecological_narratives_analyzed += len(result["ecological_narratives"])

            self.logger.info(f"Completed environmental mythology task {task_id}")

        except Exception as e:
            self.logger.error(f"Error processing environmental mythology task {task.task_id}: {e}")

            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)

            self.failed_tasks += 1

        finally:
            self.current_tasks.discard(task_id)

    async def analyze_environmental_mythological_correlations(
        self,
        environmental_data: Dict[str, Any],
        mythological_texts: List[Dict[str, Any]],
        correlation_method: str = "statistical"
    ) -> Dict[str, Any]:
        """
        Analyze correlations between environmental factors and mythological narratives.

        Args:
            environmental_data: Environmental data (climate, geology, ecology)
            mythological_texts: Mythological texts and narratives
            correlation_method: Method for correlation analysis

        Returns:
            Correlation analysis results
        """
        try:
            self.logger.info(f"Analyzing environmental-mythological correlations using {correlation_method} method")

            # Extract environmental features
            environmental_features = self._extract_environmental_features(environmental_data)

            # Extract mythological environmental references
            mythological_references = await self._extract_mythological_environmental_references(mythological_texts)

            # Calculate correlations based on method
            if correlation_method == "statistical":
                correlations = self._calculate_statistical_correlations(environmental_features, mythological_references)
            elif correlation_method == "symbolic":
                correlations = self._calculate_symbolic_correlations(environmental_features, mythological_references)
            elif correlation_method == "narrative":
                correlations = self._calculate_narrative_correlations(environmental_features, mythological_references)
            else:
                correlations = []

            # Identify significant correlations
            significant_correlations = self._identify_significant_correlations(correlations)

            # Analyze causal relationships
            causal_relationships = self._analyze_environmental_causal_relationships(significant_correlations)

            # Integrate with text processor for deeper analysis
            if self.text_processor and significant_correlations:
                await self._integrate_mythological_text_analysis(significant_correlations)

            # Validate correlations
            if self.validation_agent and significant_correlations:
                validation_results = await self._validate_environmental_correlations(significant_correlations)

            result = {
                "analysis_type": "environmental_mythological_correlation",
                "correlation_method": correlation_method,
                "environmental_features": len(environmental_features),
                "mythological_texts": len(mythological_texts),
                "correlations": correlations,
                "significant_correlations": significant_correlations,
                "causal_relationships": causal_relationships,
                "total_correlations": len(correlations),
                "significant_count": len(significant_correlations),
                "validation_results": validation_results if 'validation_results' in locals() else None,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing environmental-mythological correlations: {e}")
            return {
                "error": str(e),
                "correlation_method": correlation_method,
                "correlations": [],
                "significant_correlations": []
            }

    async def analyze_climatic_narrative_influence(
        self,
        climate_data: Dict[str, Any],
        narratives: List[Dict[str, Any]],
        time_period: str = "annual"
    ) -> Dict[str, Any]:
        """
        Analyze how climatic patterns influence mythological narratives.

        Args:
            climate_data: Climatic data and patterns
            narratives: Mythological narratives
            time_period: Time period for analysis

        Returns:
            Climatic influence analysis results
        """
        try:
            self.logger.info(f"Analyzing climatic influence on narratives for {time_period} period")

            # Extract climatic patterns
            climatic_patterns = self._extract_climatic_patterns(climate_data, time_period)

            # Analyze narrative climatic references
            narrative_climatic_references = await self._analyze_narrative_climatic_references(narratives)

            # Calculate climatic influence on narratives
            climatic_influences = self._calculate_climatic_narrative_influence(
                climatic_patterns, narrative_climatic_references
            )

            # Identify climatic mythological motifs
            climatic_motifs = self._identify_climatic_mythological_motifs(climatic_influences)

            # Analyze seasonal narrative patterns
            seasonal_patterns = self._analyze_seasonal_narrative_patterns(climatic_influences, time_period)

            result = {
                "analysis_type": "climatic_narrative_influence",
                "time_period": time_period,
                "climatic_patterns": climatic_patterns,
                "narrative_references": narrative_climatic_references,
                "climatic_influences": climatic_influences,
                "climatic_motifs": climatic_motifs,
                "seasonal_patterns": seasonal_patterns,
                "significant_influences": len([i for i in climatic_influences if i.get("influence_strength", 0) > self.climatic_influence_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing climatic narrative influence: {e}")
            return {
                "error": str(e),
                "time_period": time_period,
                "climatic_influences": [],
                "climatic_motifs": []
            }

    async def analyze_geological_mythological_associations(
        self,
        geological_features: List[Dict[str, Any]],
        mythological_motifs: List[Dict[str, Any]],
        association_type: str = "symbolic"
    ) -> Dict[str, Any]:
        """
        Analyze associations between geological features and mythological motifs.

        Args:
            geological_features: Geological features data
            mythological_motifs: Mythological motifs and symbols
            association_type: Type of association analysis

        Returns:
            Geological-mythological association analysis results
        """
        try:
            self.logger.info(f"Analyzing geological-mythological associations using {association_type} analysis")

            # Calculate feature-motif associations
            if association_type == "symbolic":
                associations = self._calculate_symbolic_geological_associations(geological_features, mythological_motifs)
            elif association_type == "functional":
                associations = self._calculate_functional_geological_associations(geological_features, mythological_motifs)
            elif association_type == "narrative":
                associations = self._calculate_narrative_geological_associations(geological_features, mythological_motifs)
            else:
                associations = []

            # Identify significant associations
            significant_associations = self._identify_significant_geological_associations(associations)

            # Analyze geological mythological symbolism
            geological_symbolism = self._analyze_geological_mythological_symbolism(significant_associations)

            # Map geological features to mythological landscapes
            mythological_landscapes = self._map_geological_mythological_landscapes(significant_associations)

            result = {
                "analysis_type": "geological_mythological_associations",
                "association_type": association_type,
                "geological_features": len(geological_features),
                "mythological_motifs": len(mythological_motifs),
                "associations": associations,
                "significant_associations": significant_associations,
                "geological_symbolism": geological_symbolism,
                "mythological_landscapes": mythological_landscapes,
                "total_associations": len(associations),
                "significant_count": len(significant_associations),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing geological-mythological associations: {e}")
            return {
                "error": str(e),
                "association_type": association_type,
                "associations": [],
                "significant_associations": []
            }

    async def analyze_ecological_mythological_symbols(
        self,
        ecological_data: Dict[str, Any],
        mythological_symbols: List[Dict[str, Any]],
        symbolic_analysis: str = "archetypal"
    ) -> Dict[str, Any]:
        """
        Analyze ecological influences on mythological symbols.

        Args:
            ecological_data: Ecological and biodiversity data
            mythological_symbols: Mythological symbols and archetypes
            symbolic_analysis: Type of symbolic analysis

        Returns:
            Ecological-mythological symbolic analysis results
        """
        try:
            self.logger.info(f"Analyzing ecological-mythological symbols using {symbolic_analysis} analysis")

            # Extract ecological features
            ecological_features = self._extract_ecological_features(ecological_data)

            # Analyze symbolic ecological representations
            if symbolic_analysis == "archetypal":
                symbolic_representations = self._analyze_archetypal_ecological_symbols(ecological_features, mythological_symbols)
            elif symbolic_analysis == "functional":
                symbolic_representations = self._analyze_functional_ecological_symbols(ecological_features, mythological_symbols)
            elif symbolic_analysis == "evolutionary":
                symbolic_representations = self._analyze_evolutionary_ecological_symbols(ecological_features, mythological_symbols)
            else:
                symbolic_representations = []

            # Identify ecological mythological archetypes
            ecological_archetypes = self._identify_ecological_mythological_archetypes(symbolic_representations)

            # Analyze biodiversity mythological symbolism
            biodiversity_symbolism = self._analyze_biodiversity_mythological_symbolism(ecological_features, symbolic_representations)

            # Map ecological cycles to mythological narratives
            ecological_cycles = self._map_ecological_cycles_mythological_narratives(ecological_features, symbolic_representations)

            result = {
                "analysis_type": "ecological_mythological_symbols",
                "symbolic_analysis": symbolic_analysis,
                "ecological_features": len(ecological_features),
                "mythological_symbols": len(mythological_symbols),
                "symbolic_representations": symbolic_representations,
                "ecological_archetypes": ecological_archetypes,
                "biodiversity_symbolism": biodiversity_symbolism,
                "ecological_cycles": ecological_cycles,
                "significant_representations": len([r for r in symbolic_representations if r.get("symbolic_strength", 0) > self.ecological_representation_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing ecological-mythological symbols: {e}")
            return {
                "error": str(e),
                "symbolic_analysis": symbolic_analysis,
                "symbolic_representations": [],
                "ecological_archetypes": []
            }

    async def analyze_seasonal_mythological_cycles(
        self,
        seasonal_data: Dict[str, Any],
        mythological_cycles: List[Dict[str, Any]],
        cycle_analysis: str = "temporal"
    ) -> Dict[str, Any]:
        """
        Analyze seasonal cycles in mythological narratives.

        Args:
            seasonal_data: Seasonal environmental data
            mythological_cycles: Mythological cycle representations
            cycle_analysis: Type of cycle analysis

        Returns:
            Seasonal mythological cycle analysis results
        """
        try:
            self.logger.info(f"Analyzing seasonal mythological cycles using {cycle_analysis} analysis")

            # Extract seasonal patterns
            seasonal_patterns = self._extract_seasonal_patterns(seasonal_data)

            # Analyze mythological seasonal representations
            mythological_seasonal_representations = self._analyze_mythological_seasonal_representations(mythological_cycles)

            # Correlate seasonal and mythological cycles
            if cycle_analysis == "temporal":
                cycle_correlations = self._correlate_temporal_seasonal_cycles(seasonal_patterns, mythological_seasonal_representations)
            elif cycle_analysis == "symbolic":
                cycle_correlations = self._correlate_symbolic_seasonal_cycles(seasonal_patterns, mythological_seasonal_representations)
            elif cycle_analysis == "ritual":
                cycle_correlations = self._correlate_ritual_seasonal_cycles(seasonal_patterns, mythological_seasonal_representations)
            else:
                cycle_correlations = []

            # Identify seasonal mythological archetypes
            seasonal_archetypes = self._identify_seasonal_mythological_archetypes(cycle_correlations)

            # Analyze agricultural mythological cycles
            agricultural_cycles = self._analyze_agricultural_mythological_cycles(cycle_correlations)

            result = {
                "analysis_type": "seasonal_mythological_cycles",
                "cycle_analysis": cycle_analysis,
                "seasonal_patterns": seasonal_patterns,
                "mythological_representations": mythological_seasonal_representations,
                "cycle_correlations": cycle_correlations,
                "seasonal_archetypes": seasonal_archetypes,
                "agricultural_cycles": agricultural_cycles,
                "significant_correlations": len([c for c in cycle_correlations if c.get("correlation_strength", 0) > self.correlation_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing seasonal mythological cycles: {e}")
            return {
                "error": str(e),
                "cycle_analysis": cycle_analysis,
                "cycle_correlations": [],
                "seasonal_archetypes": []
            }

    # Core environmental analysis methods

    def _extract_environmental_features(self, environmental_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract environmental features for analysis."""
        features = []

        for category, subcategories in self.environmental_categories.items():
            if category in environmental_data:
                category_data = environmental_data[category]
                for subcategory in subcategories:
                    if subcategory in category_data:
                        feature = {
                            "category": category,
                            "subcategory": subcategory,
                            "data": category_data[subcategory],
                            "feature_id": str(uuid.uuid4())
                        }
                        features.append(feature)

        return features

    async def _extract_mythological_environmental_references(self, mythological_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract environmental references from mythological texts."""
        references = []

        if not self.text_processor:
            return references

        for text in mythological_texts:
            content = text.get("content", "")
            if content:
                # Process text for environmental references
                result = await self.text_processor.process_text_string(content)

                if result.success:
                    # Extract environmental entities
                    environmental_entities = []
                    for entity in result.entities:
                        if self._is_environmental_entity(entity):
                            environmental_entities.append(entity)

                    if environmental_entities:
                        reference = {
                            "text_id": text.get("id", str(uuid.uuid4())),
                            "text_title": text.get("title", "Unknown"),
                            "environmental_entities": environmental_entities,
                            "sentiment": result.sentiment_analysis,
                            "reference_id": str(uuid.uuid4())
                        }
                        references.append(reference)

        return references

    def _calculate_statistical_correlations(self, environmental_features: List[Dict[str, Any]], mythological_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate statistical correlations between environmental features and mythological references."""
        correlations = []

        # Create feature-reference matrix
        feature_names = [f"{f['category']}_{f['subcategory']}" for f in environmental_features]
        reference_entities = []

        for ref in mythological_references:
            for entity in ref.get("environmental_entities", []):
                entity_name = entity.get("name", "").lower()
                if entity_name not in reference_entities:
                    reference_entities.append(entity_name)

        # Calculate co-occurrence correlations
        for feature in environmental_features:
            feature_name = f"{feature['category']}_{feature['subcategory']}"
            feature_correlations = []

            for entity_name in reference_entities:
                # Calculate correlation coefficient
                correlation = self._calculate_feature_entity_correlation(feature, entity_name, mythological_references)

                if abs(correlation) > self.correlation_threshold:
                    feature_correlations.append({
                        "entity": entity_name,
                        "correlation_coefficient": correlation,
                        "correlation_type": "statistical",
                        "significance": abs(correlation) > self.correlation_threshold
                    })

            if feature_correlations:
                correlation_entry = {
                    "feature": feature_name,
                    "feature_category": feature["category"],
                    "entity_correlations": feature_correlations,
                    "strongest_correlation": max(feature_correlations, key=lambda x: abs(x["correlation_coefficient"])),
                    "correlation_id": str(uuid.uuid4())
                }
                correlations.append(correlation_entry)

        return correlations

    def _calculate_symbolic_correlations(self, environmental_features: List[Dict[str, Any]], mythological_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate symbolic correlations between environmental features and mythological elements."""
        correlations = []

        # Use environmental-mythological lexicon for symbolic mapping
        for feature in environmental_features:
            feature_name = f"{feature['category']}_{feature['subcategory']}"

            symbolic_mappings = self.mythological_environmental_lexicon.get(feature_name, [])

            if symbolic_mappings:
                correlation_entry = {
                    "feature": feature_name,
                    "feature_category": feature["category"],
                    "symbolic_mappings": symbolic_mappings,
                    "correlation_type": "symbolic",
                    "symbolic_strength": len(symbolic_mappings) / 10.0,  # Normalize
                    "correlation_id": str(uuid.uuid4())
                }
                correlations.append(correlation_entry)

        return correlations

    def _calculate_narrative_correlations(self, environmental_features: List[Dict[str, Any]], mythological_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate narrative-based correlations."""
        correlations = []

        # Analyze how environmental features appear in narrative contexts
        for feature in environmental_features:
            feature_name = f"{feature['category']}_{feature['subcategory']}"

            narrative_contexts = []
            for ref in mythological_references:
                for entity in ref.get("environmental_entities", []):
                    if feature_name.lower() in entity.get("name", "").lower():
                        narrative_contexts.append({
                            "text_id": ref["text_id"],
                            "context": entity.get("context", ""),
                            "sentiment": ref.get("sentiment", {})
                        })

            if narrative_contexts:
                correlation_entry = {
                    "feature": feature_name,
                    "feature_category": feature["category"],
                    "narrative_contexts": narrative_contexts,
                    "correlation_type": "narrative",
                    "narrative_prevalence": len(narrative_contexts),
                    "correlation_id": str(uuid.uuid4())
                }
                correlations.append(correlation_entry)

        return correlations

    def _identify_significant_correlations(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify correlations that exceed significance thresholds."""
        significant = []

        for correlation in correlations:
            significance_score = 0.0

            if correlation["correlation_type"] == "statistical":
                max_corr = max([abs(c["correlation_coefficient"]) for c in correlation.get("entity_correlations", [])])
                significance_score = max_corr
            elif correlation["correlation_type"] == "symbolic":
                significance_score = correlation.get("symbolic_strength", 0.0)
            elif correlation["correlation_type"] == "narrative":
                significance_score = min(1.0, correlation.get("narrative_prevalence", 0) / 5.0)

            if significance_score > self.correlation_threshold:
                significant_correlation = correlation.copy()
                significant_correlation["significance_score"] = significance_score
                significant.append(significant_correlation)

        return significant

    def _analyze_environmental_causal_relationships(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze causal relationships between environmental factors and mythological development."""
        relationships = []

        # Group correlations by environmental category
        category_correlations = defaultdict(list)
        for correlation in correlations:
            category = correlation.get("feature_category")
            if category:
                category_correlations[category].append(correlation)

        # Analyze causal patterns within categories
        for category, category_corrs in category_correlations.items():
            if len(category_corrs) > 1:
                # Look for causal chains
                causal_chain = self._identify_causal_chain(category_corrs)
                if causal_chain:
                    relationships.append({
                        "category": category,
                        "causal_chain": causal_chain,
                        "relationship_type": "environmental_causal",
                        "strength": len(causal_chain) / len(category_corrs),
                        "relationship_id": str(uuid.uuid4())
                    })

        return relationships

    # Integration methods

    async def _integrate_mythological_text_analysis(self, correlations: List[Dict[str, Any]]) -> None:
        """Integrate deeper text analysis for correlations."""
        if not self.text_processor:
            return

        try:
            for correlation in correlations:
                # Get texts related to this correlation
                related_texts = self._get_correlation_related_texts(correlation)

                for text in related_texts:
                    # Perform deeper analysis
                    result = await self.text_processor.process_text_string(text)

                    if result.success:
                        correlation["deep_text_analysis"] = result.entities
                        correlation["text_sentiment"] = result.sentiment_analysis

        except Exception as e:
            self.logger.warning(f"Failed to integrate mythological text analysis: {e}")

    async def _validate_environmental_correlations(self, correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate environmental correlations using multi-layered validation."""
        if not self.validation_agent:
            return {"validation_status": "no_validator_available"}

        try:
            validation_data = {
                "pattern_data": {
                    "pattern_type": "environmental_mythological_correlation",
                    "correlations": correlations,
                    "metadata": {
                        "analysis_agent": self.agent_id,
                        "correlation_count": len(correlations)
                    }
                }
            }

            validation_result = await self.validation_agent.validate_pattern_multi_layered(
                pattern_id=str(uuid.uuid4()),
                pattern_data=validation_data,
                validation_layers=["statistical", "cultural"],
                store_results=True
            )

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate environmental correlations: {e}")
            return {"validation_error": str(e)}

    # Utility methods

    def _is_environmental_entity(self, entity: Dict[str, Any]) -> bool:
        """Check if an entity is environmental."""
        entity_name = entity.get("name", "").lower()
        entity_type = entity.get("entity_type", "")

        # Check against environmental categories
        for category, subcategories in self.environmental_categories.items():
            for subcategory in subcategories:
                if subcategory in entity_name or subcategory in entity_type:
                    return True

        return False

    def _calculate_feature_entity_correlation(self, feature: Dict[str, Any], entity_name: str, references: List[Dict[str, Any]]) -> float:
        """Calculate correlation between a feature and entity."""
        # Simple co-occurrence based correlation
        feature_present = 0
        entity_present = 0
        both_present = 0

        for ref in references:
            has_feature = feature["subcategory"].lower() in ref.get("text_title", "").lower()
            has_entity = any(entity_name in entity.get("name", "").lower() for entity in ref.get("environmental_entities", []))

            if has_feature:
                feature_present += 1
            if has_entity:
                entity_present += 1
            if has_feature and has_entity:
                both_present += 1

        if feature_present == 0 or entity_present == 0:
            return 0.0

        # Calculate Jaccard similarity as correlation measure
        union = feature_present + entity_present - both_present
        if union == 0:
            return 0.0

        return both_present / union

    def _identify_causal_chain(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify causal chains in correlations."""
        # Simple chain identification based on correlation strengths
        sorted_correlations = sorted(correlations, key=lambda x: x.get("significance_score", 0), reverse=True)

        if len(sorted_correlations) >= 2:
            return sorted_correlations[:min(3, len(sorted_correlations))]

        return []

    def _get_correlation_related_texts(self, correlation: Dict[str, Any]) -> List[str]:
        """Get texts related to a correlation."""
        # Extract text content from correlation data
        texts = []

        if "narrative_contexts" in correlation:
            for context in correlation["narrative_contexts"]:
                texts.append(context.get("context", ""))

        return texts

    # Placeholder methods for detailed analysis

    def _extract_climatic_patterns(self, climate_data: Dict[str, Any], time_period: str) -> List[Dict[str, Any]]:
        """Extract climatic patterns."""
        return []

    async def _analyze_narrative_climatic_references(self, narratives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze climatic references in narratives."""
        return []

    def _calculate_climatic_narrative_influence(self, patterns: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate climatic influence on narratives."""
        return []

    def _identify_climatic_mythological_motifs(self, influences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify climatic mythological motifs."""
        return []

    def _analyze_seasonal_narrative_patterns(self, influences: List[Dict[str, Any]], time_period: str) -> Dict[str, Any]:
        """Analyze seasonal narrative patterns."""
        return {}

    def _calculate_symbolic_geological_associations(self, features: List[Dict[str, Any]], motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate symbolic geological associations."""
        return []

    def _calculate_functional_geological_associations(self, features: List[Dict[str, Any]], motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate functional geological associations."""
        return []

    def _calculate_narrative_geological_associations(self, features: List[Dict[str, Any]], motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate narrative geological associations."""
        return []

    def _identify_significant_geological_associations(self, associations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify significant geological associations."""
        return []

    def _analyze_geological_mythological_symbolism(self, associations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze geological mythological symbolism."""
        return {}

    def _map_geological_mythological_landscapes(self, associations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map geological features to mythological landscapes."""
        return []

    def _extract_ecological_features(self, ecological_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ecological features."""
        return []

    def _analyze_archetypal_ecological_symbols(self, features: List[Dict[str, Any]], symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze archetypal ecological symbols."""
        return []

    def _analyze_functional_ecological_symbols(self, features: List[Dict[str, Any]], symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze functional ecological symbols."""
        return []

    def _analyze_evolutionary_ecological_symbols(self, features: List[Dict[str, Any]], symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze evolutionary ecological symbols."""
        return []

    def _identify_ecological_mythological_archetypes(self, representations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify ecological mythological archetypes."""
        return []

    def _analyze_biodiversity_mythological_symbolism(self, features: List[Dict[str, Any]], representations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze biodiversity mythological symbolism."""
        return {}

    def _map_ecological_cycles_mythological_narratives(self, features: List[Dict[str, Any]], representations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map ecological cycles to mythological narratives."""
        return []

    def _extract_seasonal_patterns(self, seasonal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract seasonal patterns."""
        return []

    def _analyze_mythological_seasonal_representations(self, cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze mythological seasonal representations."""
        return []

    def _correlate_temporal_seasonal_cycles(self, patterns: List[Dict[str, Any]], representations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate temporal seasonal cycles."""
        return []

    def _correlate_symbolic_seasonal_cycles(self, patterns: List[Dict[str, Any]], representations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate symbolic seasonal cycles."""
        return []

    def _correlate_ritual_seasonal_cycles(self, patterns: List[Dict[str, Any]], representations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate ritual seasonal cycles."""
        return []

    def _identify_seasonal_mythological_archetypes(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify seasonal mythological archetypes."""
        return []

    def _analyze_agricultural_mythological_cycles(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze agricultural mythological cycles."""
        return []

    # Data loading methods

    def _load_environmental_data(self) -> Dict[str, Any]:
        """Load environmental data and patterns."""
        return {
            "climatic_zones": {
                "tropical": {"temperature_range": [20, 35], "precipitation": "high"},
                "temperate": {"temperature_range": [5, 25], "precipitation": "moderate"},
                "arctic": {"temperature_range": [-30, 10], "precipitation": "low"}
            },
            "geological_features": {
                "mountains": ["sacred_peaks", "divine_abodes", "cosmic_axes"],
                "rivers": ["life_givers", "boundary_markers", "purification_sources"],
                "caves": ["underworld_entrances", "initiation_sites", "spiritual_realms"]
            }
        }

    def _load_mythological_environmental_lexicon(self) -> Dict[str, Any]:
        """Load lexicon mapping environmental features to mythological concepts."""
        return {
            "climate_temperature": ["fire_gods", "ice_giants", "seasonal_deities"],
            "geology_mountains": ["sacred_peaks", "divine_thrones", "cosmic_pillars"],
            "ecology_flora": ["world_trees", "sacred_groves", "magical_plants"],
            "hydrology_rivers": ["life_streams", "boundary_waters", "purification_rivers"]
        }

    async def _load_environmental_mythology_data(self) -> None:
        """Load environmental mythology analysis data."""
        # This would load environmental databases, mythological corpora, etc.
        pass

    # Database storage

    async def _store_environmental_results(self, task_id: str, results: Dict[str, Any]) -> None:
        """Store environmental mythology analysis results in database."""
        try:
            protocol_result = {
                "protocol_type": "environmental_mythology_determinants",
                "task_id": task_id,
                "agent_id": self.agent_id,
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "analysis_type": results.get("analysis_type"),
                    "significant_findings": results.get("significant_count", 0)
                }
            }

            result_id = await self.pattern_storage.store_protocol_result(protocol_result)

            self.logger.info(f"Stored environmental mythology results with ID: {result_id}")

        except Exception as e:
            self.logger.error(f"Failed to store environmental mythology results: {e}")

    # Message handlers

    async def _handle_environmental_request(self, message: AgentMessage) -> None:
        """Handle environmental mythology analysis requests."""
        try:
            request_data = message.payload
            analysis_type = request_data.get("analysis_type", "environmental_mythological_correlation")
            input_data = request_data.get("input_data", {})

            # Perform analysis based on type
            if analysis_type == "environmental_mythological_correlation":
                result = await self.analyze_environmental_mythological_correlations(
                    input_data.get("environmental_data", {}),
                    input_data.get("mythological_texts", [])
                )
            elif analysis_type == "climatic_narrative_influence":
                result = await self.analyze_climatic_narrative_influence(
                    input_data.get("climate_data", {}),
                    input_data.get("narratives", [])
                )
            elif analysis_type == "geological_mythological_associations":
                result = await self.analyze_geological_mythological_associations(
                    input_data.get("geological_features", []),
                    input_data.get("mythological_motifs", [])
                )
            elif analysis_type == "ecological_mythological_symbols":
                result = await self.analyze_ecological_mythological_symbols(
                    input_data.get("ecological_data", {}),
                    input_data.get("mythological_symbols", [])
                )
            elif analysis_type == "seasonal_mythological_cycles":
                result = await self.analyze_seasonal_mythological_cycles(
                    input_data.get("seasonal_data", {}),
                    input_data.get("mythological_cycles", [])
                )
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="environmental_mythology_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling environmental mythology request: {e}")

    async def _handle_correlation_request(self, message: AgentMessage) -> None:
        """Handle environmental correlation requests."""
        try:
            request_data = message.payload
            environmental_data = request_data.get("environmental_data", {})
            mythological_texts = request_data.get("mythological_texts", [])
            correlation_method = request_data.get("correlation_method", "statistical")

            result = await self.analyze_environmental_mythological_correlations(
                environmental_data, mythological_texts, correlation_method
            )

            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="environmental_correlation_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling correlation request: {e}")

    async def _handle_climate_request(self, message: AgentMessage) -> None:
        """Handle climatic mythology requests."""
        try:
            request_data = message.payload
            climate_data = request_data.get("climate_data", {})
            narratives = request_data.get("narratives", [])
            time_period = request_data.get("time_period", "annual")

            result = await self.analyze_climatic_narrative_influence(
                climate_data, narratives, time_period
            )

            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="climate_mythology_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling climate request: {e}")

    async def _process_environmental_queue(self) -> None:
        """Process queued environmental mythology requests."""
        # Implementation for processing queued requests
        pass

    async def _perform_periodic_environmental_analysis(self) -> None:
        """Perform periodic environmental-mythological analysis."""
        try:
            # Fetch environmental data and mythological texts
            # This would integrate with external data sources

            # For now, perform basic correlation analysis
            self.logger.info("Performing periodic environmental-mythological analysis")

        except Exception as e:
            self.logger.error(f"Error in periodic environmental analysis: {e}")

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect environmental mythology agent metrics."""
        base_metrics = await super().collect_metrics() or {}

        environmental_metrics = {
            "environmental_analyses_performed": self.environmental_analyses_performed,
            "mythological_correlations_found": self.mythological_correlations_found,
            "climatic_patterns_identified": self.climatic_patterns_identified,
            "geological_associations_mapped": self.geological_associations_mapped,
            "ecological_narratives_analyzed": self.ecological_narratives_analyzed,
            "environmental_categories": len(self.environmental_categories),
            "mythological_motifs": len(self.mythological_motifs),
            "integration_status": {
                "text_processor": self.text_processor is not None,
                "validation_agent": self.validation_agent is not None,
                "pattern_discovery": self.pattern_discovery is not None
            }
        }

        return {**base_metrics, **environmental_metrics}

    def _get_capabilities(self) -> List[str]:
        """Get environmental mythology agent capabilities."""
        return [
            "environmental_mythology_agent",
            "environmental_mythological_correlation",
            "climatic_narrative_influence",
            "geological_mythological_associations",
            "ecological_mythological_symbols",
            "seasonal_mythological_cycles",
            "statistical_correlation_analysis",
            "symbolic_correlation_analysis",
            "narrative_correlation_analysis",
            "causal_relationship_analysis",
            "archetypal_symbol_analysis",
            "functional_symbol_analysis",
            "evolutionary_symbol_analysis",
            "temporal_cycle_analysis",
            "ritual_cycle_analysis"
        ]

    async def shutdown(self) -> None:
        """Shutdown environmental mythology agent."""
        try:
            # Shutdown integration components
            if self.text_processor:
                await self.text_processor.cleanup()
            if self.validation_agent:
                await self.validation_agent.shutdown()
            if self.pattern_discovery:
                await self.pattern_discovery.shutdown()

            # Call parent shutdown
            await super().shutdown()

            self.logger.info("EnvironmentalMythologyAgent shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during environmental mythology agent shutdown: {e}")


# Factory function
def create_environmental_mythology_agent(agent_id: Optional[str] = None, **kwargs) -> EnvironmentalMythologyAgent:
    """
    Factory function to create environmental mythology agents.

    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured environmental mythology agent
    """
    return EnvironmentalMythologyAgent(agent_id=agent_id, **kwargs)


# Main entry point
async def main():
    """Main entry point for running the EnvironmentalMythologyAgent."""
    import signal
    import sys

    # Create and configure agent
    agent = EnvironmentalMythologyAgent()

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
        print(f"EnvironmentalMythologyAgent failed: {e}")
        sys.exit(1)

    print("EnvironmentalMythologyAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())