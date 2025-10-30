"""
A2A World Platform - Mythological Geography Agent

Specialized agent for analyzing mythological geography in classical literature.
Examines how geographical features, landscapes, and spatial relationships
are represented, mythologized, and interpreted in classical texts and narratives.
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
import re

from agents.core.base_agent import BaseAgent
from agents.core.config import DiscoveryAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.pattern_storage import PatternStorage

# Import integration components
from agents.parsers.data_processors.text_processor import TextProcessor
from agents.validation.multi_layered_validation_agent import MultiLayeredValidationAgent
from agents.discovery.pattern_discovery import PatternDiscoveryAgent


class MythologicalGeographyAgent(BaseAgent):
    """
    Agent specialized in mythological geography of classical literature.

    Capabilities:
    - Geographical feature analysis in classical texts
    - Mythological landscape interpretation
    - Spatial relationships in narratives
    - Cultural geography in literature
    - Sacred geography mapping
    - Literary cartography analysis
    - Topographical mythology
    - Geographical symbolism in texts
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DiscoveryAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="mythological_geography",
            config=config or DiscoveryAgentConfig(),
            config_file=config_file
        )

        # Geographical analysis parameters
        self.geographical_reference_threshold = 0.6  # Threshold for geographical references
        self.mythological_symbolism_threshold = 0.5  # Threshold for mythological symbolism
        self.spatial_relationship_threshold = 0.4  # Threshold for spatial relationships
        self.sacred_geography_threshold = 0.7  # Threshold for sacred geography identification

        # Geographical categories in classical literature
        self.geographical_categories = {
            "mountains": ["peaks", "ranges", "volcanoes", "sacred_mounts"],
            "rivers": ["streams", "rivers", "springs", "water_sources"],
            "seas": ["oceans", "seas", "gulfs", "straits"],
            "islands": ["islands", "archipelagos", "peninsulas"],
            "forests": ["woods", "groves", "thickets", "sacred_groves"],
            "caves": ["caves", "grottos", "underground_chambers"],
            "cities": ["cities", "towns", "settlements", "capitals"],
            "deserts": ["wastes", "deserts", "steppes", "barren_lands"]
        }

        # Mythological geographical motifs
        self.mythological_motifs = {
            "sacred_mountains": ["olympus", "sina", "zion", "meru"],
            "underworld_entrances": ["caves", "springs", "volcanoes", "lakes"],
            "divine_realms": ["heavens", "underworld", "otherworlds"],
            "hero_journeys": ["crossroads", "thresholds", "boundaries"],
            "cosmic_centers": ["omphalos", "axis_mundi", "world_navel"]
        }

        # Classical literature periods
        self.literary_periods = {
            "ancient_greek": {"start": -800, "end": -300, "key_works": ["iliad", "odyssey", "theogony"]},
            "ancient_roman": {"start": -300, "end": 500, "key_works": ["aeneid", "metamorphoses"]},
            "medieval": {"start": 500, "end": 1500, "key_works": ["divine_comedy", "canterbury_tales"]},
            "renaissance": {"start": 1300, "end": 1700, "key_works": ["paradise_lost", "faerie_queene"]}
        }

        # Performance tracking
        self.geographical_analyses_performed = 0
        self.mythological_references_identified = 0
        self.spatial_relationships_mapped = 0
        self.sacred_geographies_analyzed = 0
        self.literary_cartographies_created = 0

        # Integration components
        self.text_processor = None
        self.validation_agent = None
        self.pattern_discovery = None

        # Geographical knowledge base
        self.geographical_lexicon = self._load_geographical_lexicon()
        self.mythological_geography_database = self._load_mythological_geography_database()

        # Database integration
        self.pattern_storage = PatternStorage()

        self.logger.info(f"MythologicalGeographyAgent {self.agent_id} initialized")

    async def process(self) -> None:
        """
        Main processing loop for mythological geography analysis.
        """
        try:
            # Process any pending mythological geography requests
            await self._process_geography_queue()

            # Perform periodic geographical analysis in literature
            if self.processed_tasks % 140 == 0:
                await self._perform_periodic_geographical_analysis()

        except Exception as e:
            self.logger.error(f"Error in mythological geography process: {e}")

    async def agent_initialize(self) -> None:
        """
        Mythological geography agent specific initialization.
        """
        try:
            # Initialize integration components
            await self._initialize_integrations()

            # Load geographical and literary data
            await self._load_geographical_literary_data()

            self.logger.info("MythologicalGeographyAgent initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize MythologicalGeographyAgent: {e}")
            raise

    async def _initialize_integrations(self) -> None:
        """Initialize integration with other agents and processors."""
        try:
            # Initialize text processor for literary text analysis
            self.text_processor = TextProcessor()
            await self.text_processor.initialize()

            # Initialize validation agent for multi-layered validation
            self.validation_agent = MultiLayeredValidationAgent(
                agent_id=f"{self.agent_id}_validation"
            )
            await self.validation_agent.agent_initialize()

            # Initialize pattern discovery for geographical pattern analysis
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
        Setup mythological geography-specific message subscriptions.
        """
        if not self.messaging:
            return

        # Subscribe to mythological geography analysis requests
        geography_sub_id = await self.nats_client.subscribe(
            "agents.mythological_geography.request",
            self._handle_geography_request,
            queue_group="mythological-geography-workers"
        )
        self.subscription_ids.append(geography_sub_id)

        # Subscribe to geographical reference analysis
        reference_sub_id = await self.nats_client.subscribe(
            "agents.geography.reference.request",
            self._handle_reference_request,
            queue_group="geographical-reference"
        )
        self.subscription_ids.append(reference_sub_id)

        # Subscribe to sacred geography analysis
        sacred_sub_id = await self.nats_client.subscribe(
            "agents.geography.sacred.request",
            self._handle_sacred_request,
            queue_group="sacred-geography"
        )
        self.subscription_ids.append(sacred_sub_id)

    async def handle_task(self, task: Task) -> None:
        """
        Handle mythological geography analysis tasks.
        """
        self.logger.info(f"Processing mythological geography task {task.task_id}: {task.task_type}")

        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)

            result = None

            if task.task_type == "geographical_feature_analysis":
                result = await self.analyze_geographical_feature_analysis(
                    task.input_data.get("text_data", []),
                    task.parameters.get("feature_types", ["mountains", "rivers"])
                )
            elif task.task_type == "mythological_landscape_interpretation":
                result = await self.analyze_mythological_landscape_interpretation(
                    task.input_data.get("landscape_descriptions", []),
                    task.parameters.get("interpretation_framework", "symbolic")
                )
            elif task.task_type == "spatial_relationships_literature":
                result = await self.analyze_spatial_relationships_literature(
                    task.input_data.get("narrative_texts", []),
                    task.parameters.get("relationship_types", ["contiguity", "hierarchy"])
                )
            elif task.task_type == "sacred_geography_mapping":
                result = await self.analyze_sacred_geography_mapping(
                    task.input_data.get("sacred_sites", []),
                    task.input_data.get("literary_references", []),
                    task.parameters.get("mapping_method", "symbolic_correspondence")
                )
            elif task.task_type == "literary_cartography":
                result = await self.analyze_literary_cartography(
                    task.input_data.get("texts", []),
                    task.parameters.get("cartography_type", "cognitive_mapping")
                )
            else:
                raise ValueError(f"Unknown mythological geography task type: {task.task_type}")

            # Store results in database
            if result:
                await self._store_geography_results(task_id, result)

            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)

            self.processed_tasks += 1
            self.geographical_analyses_performed += 1

            # Update counters based on result type
            if result and "geographical_references" in result:
                self.mythological_references_identified += len(result["geographical_references"])
            if result and "spatial_relationships" in result:
                self.spatial_relationships_mapped += len(result["spatial_relationships"])
            if result and "sacred_geographies" in result:
                self.sacred_geographies_analyzed += len(result["sacred_geographies"])
            if result and "literary_cartographies" in result:
                self.literary_cartographies_created += len(result["literary_cartographies"])

            self.logger.info(f"Completed mythological geography task {task_id}")

        except Exception as e:
            self.logger.error(f"Error processing mythological geography task {task.task_id}: {e}")

            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)

            self.failed_tasks += 1

        finally:
            self.current_tasks.discard(task_id)

    async def analyze_geographical_feature_analysis(
        self,
        text_data: List[Dict[str, Any]],
        feature_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze geographical features in classical literature.

        Args:
            text_data: Literary texts to analyze
            feature_types: Types of geographical features to analyze

        Returns:
            Geographical feature analysis results
        """
        if feature_types is None:
            feature_types = ["mountains", "rivers", "seas"]

        try:
            self.logger.info(f"Analyzing geographical features in {len(text_data)} texts")

            # Extract geographical references from texts
            geographical_references = await self._extract_geographical_references(text_data, feature_types)

            # Classify references by feature type
            classified_references = self._classify_geographical_references(geographical_references)

            # Analyze geographical density and distribution
            geographical_density = self._analyze_geographical_density(classified_references)

            # Identify geographical motifs and patterns
            geographical_motifs = self._identify_geographical_motifs(classified_references)

            # Analyze temporal evolution of geographical references
            temporal_evolution = self._analyze_temporal_geographical_evolution(classified_references, text_data)

            # Integrate with text processor for deeper analysis
            if self.text_processor and geographical_references:
                await self._integrate_literary_context(geographical_references)

            # Validate geographical analysis
            if self.validation_agent and geographical_motifs:
                validation_results = await self._validate_geographical_analysis(geographical_motifs)

            result = {
                "analysis_type": "geographical_feature_analysis",
                "texts_analyzed": len(text_data),
                "feature_types": feature_types,
                "geographical_references": geographical_references,
                "classified_references": classified_references,
                "geographical_density": geographical_density,
                "geographical_motifs": geographical_motifs,
                "temporal_evolution": temporal_evolution,
                "total_references": len(geographical_references),
                "significant_motifs": len([m for m in geographical_motifs if m.get("significance_score", 0) > self.geographical_reference_threshold]),
                "validation_results": validation_results if 'validation_results' in locals() else None,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing geographical features: {e}")
            return {
                "error": str(e),
                "feature_types": feature_types,
                "geographical_references": [],
                "geographical_motifs": []
            }

    async def analyze_mythological_landscape_interpretation(
        self,
        landscape_descriptions: List[Dict[str, Any]],
        interpretation_framework: str = "symbolic"
    ) -> Dict[str, Any]:
        """
        Analyze mythological interpretations of landscapes in literature.

        Args:
            landscape_descriptions: Landscape descriptions from texts
            interpretation_framework: Framework for interpretation

        Returns:
            Mythological landscape interpretation results
        """
        try:
            self.logger.info(f"Analyzing mythological landscape interpretations using {interpretation_framework} framework")

            # Extract landscape elements
            landscape_elements = self._extract_landscape_elements(landscape_descriptions)

            # Apply interpretation framework
            if interpretation_framework == "symbolic":
                mythological_interpretations = self._apply_symbolic_interpretation(landscape_elements)
            elif interpretation_framework == "archetypal":
                mythological_interpretations = self._apply_archetypal_interpretation(landscape_elements)
            elif interpretation_framework == "structural":
                mythological_interpretations = self._apply_structural_interpretation(landscape_elements)
            else:
                mythological_interpretations = []

            # Identify mythological landscape archetypes
            landscape_archetypes = self._identify_landscape_archetypes(mythological_interpretations)

            # Analyze symbolic landscape functions
            symbolic_functions = self._analyze_symbolic_landscape_functions(mythological_interpretations)

            # Map mythological landscape transformations
            landscape_transformations = self._map_mythological_landscape_transformations(landscape_descriptions)

            result = {
                "analysis_type": "mythological_landscape_interpretation",
                "interpretation_framework": interpretation_framework,
                "descriptions_analyzed": len(landscape_descriptions),
                "landscape_elements": landscape_elements,
                "mythological_interpretations": mythological_interpretations,
                "landscape_archetypes": landscape_archetypes,
                "symbolic_functions": symbolic_functions,
                "landscape_transformations": landscape_transformations,
                "significant_interpretations": len([i for i in mythological_interpretations if i.get("mythological_strength", 0) > self.mythological_symbolism_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing mythological landscape interpretation: {e}")
            return {
                "error": str(e),
                "interpretation_framework": interpretation_framework,
                "mythological_interpretations": [],
                "landscape_archetypes": []
            }

    async def analyze_spatial_relationships_literature(
        self,
        narrative_texts: List[Dict[str, Any]],
        relationship_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze spatial relationships in literary narratives.

        Args:
            narrative_texts: Literary texts with spatial content
            relationship_types: Types of spatial relationships to analyze

        Returns:
            Spatial relationships analysis results
        """
        if relationship_types is None:
            relationship_types = ["contiguity", "hierarchy", "directionality"]

        try:
            self.logger.info(f"Analyzing spatial relationships in {len(narrative_texts)} narratives")

            # Extract spatial references from narratives
            spatial_references = await self._extract_spatial_references(narrative_texts)

            # Analyze spatial relationships
            spatial_relationships = self._analyze_spatial_relationships(spatial_references, relationship_types)

            # Map narrative spaces
            narrative_spaces = self._map_narrative_spaces(spatial_relationships)

            # Analyze spatial symbolism
            spatial_symbolism = self._analyze_spatial_symbolism(spatial_relationships)

            # Identify spatial narrative patterns
            spatial_patterns = self._identify_spatial_narrative_patterns(narrative_spaces)

            result = {
                "analysis_type": "spatial_relationships_literature",
                "relationship_types": relationship_types,
                "texts_analyzed": len(narrative_texts),
                "spatial_references": spatial_references,
                "spatial_relationships": spatial_relationships,
                "narrative_spaces": narrative_spaces,
                "spatial_symbolism": spatial_symbolism,
                "spatial_patterns": spatial_patterns,
                "significant_relationships": len([r for r in spatial_relationships if r.get("relationship_strength", 0) > self.spatial_relationship_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing spatial relationships: {e}")
            return {
                "error": str(e),
                "relationship_types": relationship_types,
                "spatial_relationships": [],
                "narrative_spaces": []
            }

    async def analyze_sacred_geography_mapping(
        self,
        sacred_sites: List[Dict[str, Any]],
        literary_references: List[Dict[str, Any]],
        mapping_method: str = "symbolic_correspondence"
    ) -> Dict[str, Any]:
        """
        Map sacred geography in classical literature.

        Args:
            sacred_sites: Sacred sites data
            literary_references: Literary references to sacred sites
            mapping_method: Method for mapping

        Returns:
            Sacred geography mapping results
        """
        try:
            self.logger.info(f"Mapping sacred geography using {mapping_method} method")

            # Correlate sacred sites with literary references
            site_references = self._correlate_sacred_sites_literary(sacred_sites, literary_references)

            # Apply mapping method
            if mapping_method == "symbolic_correspondence":
                sacred_mappings = self._apply_symbolic_correspondence_mapping(site_references)
            elif mapping_method == "ritual_landscape":
                sacred_mappings = self._apply_ritual_landscape_mapping(site_references)
            elif mapping_method == "cosmological_alignment":
                sacred_mappings = self._apply_cosmological_alignment_mapping(site_references)
            else:
                sacred_mappings = []

            # Identify sacred geographical patterns
            sacred_patterns = self._identify_sacred_geographical_patterns(sacred_mappings)

            # Analyze sacred geography symbolism
            sacred_symbolism = self._analyze_sacred_geography_symbolism(sacred_mappings)

            # Map sacred landscape networks
            sacred_networks = self._map_sacred_landscape_networks(sacred_mappings)

            result = {
                "analysis_type": "sacred_geography_mapping",
                "mapping_method": mapping_method,
                "sacred_sites": len(sacred_sites),
                "literary_references": len(literary_references),
                "site_references": site_references,
                "sacred_mappings": sacred_mappings,
                "sacred_patterns": sacred_patterns,
                "sacred_symbolism": sacred_symbolism,
                "sacred_networks": sacred_networks,
                "significant_mappings": len([m for m in sacred_mappings if m.get("sacred_significance", 0) > self.sacred_geography_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing sacred geography mapping: {e}")
            return {
                "error": str(e),
                "mapping_method": mapping_method,
                "sacred_mappings": [],
                "sacred_patterns": []
            }

    async def analyze_literary_cartography(
        self,
        texts: List[Dict[str, Any]],
        cartography_type: str = "cognitive_mapping"
    ) -> Dict[str, Any]:
        """
        Analyze literary cartography and spatial representations.

        Args:
            texts: Literary texts to analyze
            cartography_type: Type of cartography analysis

        Returns:
            Literary cartography analysis results
        """
        try:
            self.logger.info(f"Analyzing literary cartography using {cartography_type} method")

            # Extract spatial information from texts
            spatial_information = await self._extract_spatial_information(texts)

            # Create literary maps
            if cartography_type == "cognitive_mapping":
                literary_maps = self._create_cognitive_literary_maps(spatial_information)
            elif cartography_type == "narrative_cartography":
                literary_maps = self._create_narrative_cartography_maps(spatial_information)
            elif cartography_type == "symbolic_cartography":
                literary_maps = self._create_symbolic_cartography_maps(spatial_information)
            else:
                literary_maps = []

            # Analyze cartographic patterns
            cartographic_patterns = self._analyze_cartographic_patterns(literary_maps)

            # Identify literary spatial schemas
            spatial_schemas = self._identify_literary_spatial_schemas(cartographic_patterns)

            # Map narrative geographies
            narrative_geographies = self._map_narrative_geographies(literary_maps)

            result = {
                "analysis_type": "literary_cartography",
                "cartography_type": cartography_type,
                "texts_analyzed": len(texts),
                "spatial_information": spatial_information,
                "literary_maps": literary_maps,
                "cartographic_patterns": cartographic_patterns,
                "spatial_schemas": spatial_schemas,
                "narrative_geographies": narrative_geographies,
                "maps_created": len(literary_maps),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing literary cartography: {e}")
            return {
                "error": str(e),
                "cartography_type": cartography_type,
                "literary_maps": [],
                "cartographic_patterns": []
            }

    # Core geographical analysis methods

    async def _extract_geographical_references(self, text_data: List[Dict[str, Any]], feature_types: List[str]) -> List[Dict[str, Any]]:
        """Extract geographical references from literary texts."""
        references = []

        if not self.text_processor:
            return references

        for text in text_data:
            content = text.get("content", "")
            if content:
                # Process text for geographical references
                result = await self.text_processor.process_text_string(content)

                if result.success:
                    # Extract geographical entities
                    geographical_entities = []
                    for entity in result.entities:
                        if self._is_geographical_entity(entity, feature_types):
                            geographical_entities.append(entity)

                    if geographical_entities:
                        reference = {
                            "text_id": text.get("id", str(uuid.uuid4())),
                            "text_title": text.get("title", "Unknown"),
                            "period": text.get("period", "unknown"),
                            "geographical_entities": geographical_entities,
                            "entity_count": len(geographical_entities),
                            "reference_id": str(uuid.uuid4())
                        }
                        references.append(reference)

        return references

    def _classify_geographical_references(self, references: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify geographical references by feature type."""
        classified = defaultdict(list)

        for ref in references:
            for entity in ref.get("geographical_entities", []):
                feature_type = self._determine_feature_type(entity)
                if feature_type:
                    classified_entity = entity.copy()
                    classified_entity["text_id"] = ref["text_id"]
                    classified_entity["period"] = ref["period"]
                    classified[feature_type].append(classified_entity)

        return dict(classified)

    def _analyze_geographical_density(self, classified_references: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze geographical density and distribution."""
        density_analysis = {}

        for feature_type, references in classified_references.items():
            # Calculate density metrics
            total_references = len(references)
            unique_entities = len(set(ref.get("name", "").lower() for ref in references))

            periods = [ref.get("period") for ref in references if ref.get("period")]
            period_distribution = {}
            for period in periods:
                period_distribution[period] = period_distribution.get(period, 0) + 1

            density_analysis[feature_type] = {
                "total_references": total_references,
                "unique_entities": unique_entities,
                "period_distribution": period_distribution,
                "density_score": total_references / max(1, unique_entities)
            }

        return density_analysis

    def _identify_geographical_motifs(self, classified_references: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify geographical motifs and patterns."""
        motifs = []

        for feature_type, references in classified_references.items():
            # Group by entity name
            entity_groups = defaultdict(list)
            for ref in references:
                entity_name = ref.get("name", "").lower()
                entity_groups[entity_name].append(ref)

            # Identify recurring motifs
            for entity_name, entity_refs in entity_groups.items():
                if len(entity_refs) >= 2:  # Recurring motif
                    motif = {
                        "motif_name": entity_name,
                        "feature_type": feature_type,
                        "occurrences": len(entity_refs),
                        "periods": list(set(ref.get("period", "unknown") for ref in entity_refs)),
                        "significance_score": len(entity_refs) / 10.0,  # Normalize
                        "motif_id": str(uuid.uuid4())
                    }
                    motifs.append(motif)

        return motifs

    def _analyze_temporal_geographical_evolution(self, classified_references: Dict[str, List[Dict[str, Any]]], text_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal evolution of geographical references."""
        evolution = {}

        # Group by period
        period_references = defaultdict(lambda: defaultdict(list))

        for feature_type, references in classified_references.items():
            for ref in references:
                period = ref.get("period", "unknown")
                period_references[period][feature_type].append(ref)

        # Analyze evolution patterns
        for period, features in period_references.items():
            evolution[period] = {
                "feature_counts": {ft: len(refs) for ft, refs in features.items()},
                "total_references": sum(len(refs) for refs in features.values()),
                "dominant_features": sorted(features.keys(), key=lambda x: len(features[x]), reverse=True)[:3]
            }

        return evolution

    # Integration methods

    async def _integrate_literary_context(self, references: List[Dict[str, Any]]) -> None:
        """Integrate literary context from text processor."""
        if not self.text_processor:
            return

        try:
            for ref in references:
                # Get deeper analysis for geographical references
                text_content = ref.get("text_title", "")
                if text_content:
                    result = await self.text_processor.process_text_string(text_content)

                    if result.success:
                        ref["literary_context"] = result.entities
                        ref["context_sentiment"] = result.sentiment_analysis

        except Exception as e:
            self.logger.warning(f"Failed to integrate literary context: {e}")

    async def _validate_geographical_analysis(self, motifs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate geographical analysis using multi-layered validation."""
        if not self.validation_agent:
            return {"validation_status": "no_validator_available"}

        try:
            validation_data = {
                "pattern_data": {
                    "pattern_type": "mythological_geography",
                    "geographical_motifs": motifs,
                    "metadata": {
                        "analysis_agent": self.agent_id,
                        "motif_count": len(motifs)
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
            self.logger.error(f"Failed to validate geographical analysis: {e}")
            return {"validation_error": str(e)}

    # Utility methods

    def _is_geographical_entity(self, entity: Dict[str, Any], feature_types: List[str]) -> bool:
        """Check if an entity is geographical."""
        entity_name = entity.get("name", "").lower()
        entity_type = entity.get("entity_type", "")

        for feature_type in feature_types:
            if feature_type in self.geographical_categories:
                patterns = self.geographical_categories[feature_type]
                for pattern in patterns:
                    if pattern in entity_name or pattern in entity_type:
                        return True

        return False

    def _determine_feature_type(self, entity: Dict[str, Any]) -> Optional[str]:
        """Determine the geographical feature type of an entity."""
        entity_name = entity.get("name", "").lower()

        for feature_type, patterns in self.geographical_categories.items():
            for pattern in patterns:
                if pattern in entity_name:
                    return feature_type

        return None

    # Placeholder methods for detailed analysis

    def _extract_landscape_elements(self, descriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract landscape elements from descriptions."""
        return []

    def _apply_symbolic_interpretation(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply symbolic interpretation to landscape elements."""
        return []

    def _apply_archetypal_interpretation(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply archetypal interpretation to landscape elements."""
        return []

    def _apply_structural_interpretation(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply structural interpretation to landscape elements."""
        return []

    def _identify_landscape_archetypes(self, interpretations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify landscape archetypes."""
        return []

    def _analyze_symbolic_landscape_functions(self, interpretations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze symbolic landscape functions."""
        return {}

    def _map_mythological_landscape_transformations(self, descriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map mythological landscape transformations."""
        return []

    async def _extract_spatial_references(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract spatial references from texts."""
        return []

    def _analyze_spatial_relationships(self, references: List[Dict[str, Any]], relationship_types: List[str]) -> List[Dict[str, Any]]:
        """Analyze spatial relationships."""
        return []

    def _map_narrative_spaces(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map narrative spaces."""
        return []

    def _analyze_spatial_symbolism(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spatial symbolism."""
        return {}

    def _identify_spatial_narrative_patterns(self, spaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify spatial narrative patterns."""
        return []

    def _correlate_sacred_sites_literary(self, sites: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate sacred sites with literary references."""
        return []

    def _apply_symbolic_correspondence_mapping(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply symbolic correspondence mapping."""
        return []

    def _apply_ritual_landscape_mapping(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply ritual landscape mapping."""
        return []

    def _apply_cosmological_alignment_mapping(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply cosmological alignment mapping."""
        return []

    def _identify_sacred_geographical_patterns(self, mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify sacred geographical patterns."""
        return []

    def _analyze_sacred_geography_symbolism(self, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sacred geography symbolism."""
        return {}

    def _map_sacred_landscape_networks(self, mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map sacred landscape networks."""
        return []

    async def _extract_spatial_information(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract spatial information from texts."""
        return []

    def _create_cognitive_literary_maps(self, spatial_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create cognitive literary maps."""
        return []

    def _create_narrative_cartography_maps(self, spatial_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create narrative cartography maps."""
        return []

    def _create_symbolic_cartography_maps(self, spatial_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create symbolic cartography maps."""
        return []

    def _analyze_cartographic_patterns(self, maps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze cartographic patterns."""
        return []

    def _identify_literary_spatial_schemas(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify literary spatial schemas."""
        return []

    def _map_narrative_geographies(self, maps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map narrative geographies."""
        return []

    # Data loading methods

    def _load_geographical_lexicon(self) -> Dict[str, Any]:
        """Load geographical lexicon for classical literature."""
        return {
            "mountains": ["peak", "summit", "ridge", "cliff", "crag", "tor"],
            "rivers": ["stream", "brook", "torrent", "cascade", "spring"],
            "seas": ["ocean", "deep", "abyss", "gulf", "bay", "cove"],
            "forests": ["wood", "grove", "thicket", "wilderness", "copse"],
            "caves": ["cavern", "grotto", "den", "lair", "hollow"],
            "sacred_terms": ["holy", "divine", "sacred", "blessed", "consecrated"]
        }

    def _load_mythological_geography_database(self) -> Dict[str, Any]:
        """Load mythological geography database."""
        return {
            "sacred_mountains": {
                "olympus": {"culture": "greek", "deities": ["zeus", "hera"], "symbolism": "divine_realm"},
                "sina": {"culture": "judeo_christian", "events": ["moses_law"], "symbolism": "revelation"},
                "zion": {"culture": "judeo_christian", "symbolism": "salvation", "meaning": "fortress"}
            },
            "mythical_rivers": {
                "styx": {"culture": "greek", "function": "underworld_boundary", "guardian": "charon"},
                "lethe": {"culture": "greek", "function": "forgetfulness", "domain": "underworld"},
                "ganges": {"culture": "hindu", "symbolism": "purification", "sacredness": "highest"}
            },
            "otherworldly_places": {
                "atlantis": {"culture": "greek", "status": "lost_continent", "symbolism": "hubris_punishment"},
                "avalon": {"culture": "celtic_british", "function": "blessed_isle", "associations": ["morgan_le_fay"]},
                "shangri_la": {"culture": "tibetan", "symbolism": "utopia", "accessibility": "hidden"}
            }
        }

    async def _load_geographical_literary_data(self) -> None:
        """Load geographical and literary analysis data."""
        # This would load geographical databases, literary corpora, etc.
        pass

    # Database storage

    async def _store_geography_results(self, task_id: str, results: Dict[str, Any]) -> None:
        """Store mythological geography analysis results in database."""
        try:
            protocol_result = {
                "protocol_type": "mythological_geography_classical_literature",
                "task_id": task_id,
                "agent_id": self.agent_id,
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "analysis_type": results.get("analysis_type"),
                    "significant_findings": results.get("significant_motifs", 0)
                }
            }

            result_id = await self.pattern_storage.store_protocol_result(protocol_result)

            self.logger.info(f"Stored mythological geography results with ID: {result_id}")

        except Exception as e:
            self.logger.error(f"Failed to store mythological geography results: {e}")

    # Message handlers

    async def _handle_geography_request(self, message: AgentMessage) -> None:
        """Handle mythological geography analysis requests."""
        try:
            request_data = message.payload
            analysis_type = request_data.get("analysis_type", "geographical_feature_analysis")
            input_data = request_data.get("input_data", {})

            # Perform analysis based on type
            if analysis_type == "geographical_feature_analysis":
                result = await self.analyze_geographical_feature_analysis(
                    input_data.get("text_data", [])
                )
            elif analysis_type == "mythological_landscape_interpretation":
                result = await self.analyze_mythological_landscape_interpretation(
                    input_data.get("landscape_descriptions", [])
                )
            elif analysis_type == "spatial_relationships_literature":
                result = await self.analyze_spatial_relationships_literature(
                    input_data.get("narrative_texts", [])
                )
            elif analysis_type == "sacred_geography_mapping":
                result = await self.analyze_sacred_geography_mapping(
                    input_data.get("sacred_sites", []),
                    input_data.get("literary_references", [])
                )
            elif analysis_type == "literary_cartography":
                result = await self.analyze_literary_cartography(
                    input_data.get("texts", [])
                )
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="mythological_geography_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling mythological geography request: {e}")

    async def _handle_reference_request(self, message: AgentMessage) -> None:
        """Handle geographical reference requests."""
        try:
            request_data = message.payload
            text_data = request_data.get("text_data", [])
            feature_types = request_data.get("feature_types", ["mountains", "rivers"])

            result = await self.analyze_geographical_feature_analysis(text_data, feature_types)

            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="geographical_reference_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling reference request: {e}")

    async def _handle_sacred_request(self, message: AgentMessage) -> None:
        """Handle sacred geography requests."""
        try:
            request_data = message.payload
            sacred_sites = request_data.get("sacred_sites", [])
            literary_references = request_data.get("literary_references", [])
            mapping_method = request_data.get("mapping_method", "symbolic_correspondence")

            result = await self.analyze_sacred_geography_mapping(
                sacred_sites, literary_references, mapping_method
            )

            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="sacred_geography_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling sacred request: {e}")

    async def _process_geography_queue(self) -> None:
        """Process queued mythological geography requests."""
        # Implementation for processing queued requests
        pass

    async def _perform_periodic_geographical_analysis(self) -> None:
        """Perform periodic geographical analysis in literature."""
        try:
            # Fetch literary texts from database
            # This would integrate with external literary databases

            self.logger.info("Performing periodic mythological geographical analysis")

        except Exception as e:
            self.logger.error(f"Error in periodic geographical analysis: {e}")

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect mythological geography agent metrics."""
        base_metrics = await super().collect_metrics() or {}

        geography_metrics = {
            "geographical_analyses_performed": self.geographical_analyses_performed,
            "mythological_references_identified": self.mythological_references_identified,
            "spatial_relationships_mapped": self.spatial_relationships_mapped,
            "sacred_geographies_analyzed": self.sacred_geographies_analyzed,
            "literary_cartographies_created": self.literary_cartographies_created,
            "geographical_categories": len(self.geographical_categories),
            "mythological_motifs": len(self.mythological_motifs),
            "literary_periods": len(self.literary_periods),
            "integration_status": {
                "text_processor": self.text_processor is not None,
                "validation_agent": self.validation_agent is not None,
                "pattern_discovery": self.pattern_discovery is not None
            }
        }

        return {**base_metrics, **geography_metrics}

    def _get_capabilities(self) -> List[str]:
        """Get mythological geography agent capabilities."""
        return [
            "mythological_geography_agent",
            "geographical_feature_analysis",
            "mythological_landscape_interpretation",
            "spatial_relationships_literature",
            "sacred_geography_mapping",
            "literary_cartography",
            "classical_literature_analysis",
            "geographical_reference_extraction",
            "spatial_symbolism_analysis",
            "narrative_cartography",
            "sacred_landscape_networks",
            "literary_spatial_schemas",
            "mythological_geography_database",
            "temporal_geographical_evolution",
            "symbolic_landscape_interpretation"
        ]

    async def shutdown(self) -> None:
        """Shutdown mythological geography agent."""
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

            self.logger.info("MythologicalGeographyAgent shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during mythological geography agent shutdown: {e}")


# Factory function
def create_mythological_geography_agent(agent_id: Optional[str] = None, **kwargs) -> MythologicalGeographyAgent:
    """
    Factory function to create mythological geography agents.

    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured mythological geography agent
    """
    return MythologicalGeographyAgent(agent_id=agent_id, **kwargs)


# Main entry point
async def main():
    """Main entry point for running the MythologicalGeographyAgent."""
    import signal
    import sys

    # Create and configure agent
    agent = MythologicalGeographyAgent()

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
        print(f"MythologicalGeographyAgent failed: {e}")
        sys.exit(1)

    print("MythologicalGeographyAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())