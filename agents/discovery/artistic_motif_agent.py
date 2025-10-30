"""
A2A World Platform - Artistic Motif Agent

Specialized agent for analyzing artistic motif diffusion and cultural contact.
Examines how artistic motifs, symbols, and design elements spread across cultures,
identifying patterns of cultural exchange, trade routes, migration, and artistic influence.
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
from itertools import combinations

from agents.core.base_agent import BaseAgent
from agents.core.config import DiscoveryAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.pattern_storage import PatternStorage

# Import integration components
from agents.parsers.data_processors.text_processor import TextProcessor
from agents.validation.multi_layered_validation_agent import MultiLayeredValidationAgent
from agents.discovery.pattern_discovery import PatternDiscoveryAgent


class ArtisticMotifAgent(BaseAgent):
    """
    Agent specialized in artistic motif diffusion and cultural contact analysis.

    Capabilities:
    - Artistic motif identification and classification
    - Cross-cultural motif diffusion analysis
    - Cultural contact pattern recognition
    - Trade route motif transmission
    - Migration pattern artistic influence
    - Chronological motif evolution tracking
    - Stylistic convergence analysis
    - Iconographic exchange networks
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DiscoveryAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="artistic_motif",
            config=config or DiscoveryAgentConfig(),
            config_file=config_file
        )

        # Artistic analysis parameters
        self.motif_similarity_threshold = 0.7  # Threshold for motif similarity detection
        self.diffusion_confidence_threshold = 0.6  # Threshold for diffusion pattern confidence
        self.cultural_contact_threshold = 0.5  # Threshold for cultural contact detection
        self.temporal_precision_years = 50  # Temporal precision for dating motifs

        # Motif categories
        self.motif_categories = {
            "geometric": ["spirals", "meanders", "chevrons", "zigzags", "circles"],
            "figurative": ["animals", "humans", "deities", "mythical_beings"],
            "symbolic": ["crosses", "stars", "mandalas", "labyrinths", "tree_of_life"],
            "ornamental": ["palmettes", "lotus_flowers", "acanthus_leaves", "rosettes"],
            "abstract": ["waves", "flames", "clouds", "mountains", "rivers"]
        }

        # Cultural transmission models
        self.transmission_models = {
            "diffusion": ["contagion", "hierarchical", "wave"],
            "migration": ["demic", "elite_dominance", "cultural_exchange"],
            "trade": ["maritime_routes", "silk_road", "steppe_routes", "coastal_networks"],
            "conquest": ["military_expansion", "colonial_contact", "religious_spread"]
        }

        # Performance tracking
        self.motif_analyses_performed = 0
        self.diffusion_patterns_identified = 0
        self.cultural_contacts_detected = 0
        self.motif_evolutions_tracked = 0
        self.stylistic_convergences_mapped = 0

        # Integration components
        self.text_processor = None
        self.validation_agent = None
        self.pattern_discovery = None

        # Artistic knowledge base
        self.motif_database = self._load_motif_database()
        self.cultural_artistic_lexicon = self._load_cultural_artistic_lexicon()

        # Database integration
        self.pattern_storage = PatternStorage()

        self.logger.info(f"ArtisticMotifAgent {self.agent_id} initialized")

    async def process(self) -> None:
        """
        Main processing loop for artistic motif analysis.
        """
        try:
            # Process any pending artistic motif requests
            await self._process_artistic_queue()

            # Perform periodic motif diffusion analysis
            if self.processed_tasks % 130 == 0:
                await self._perform_periodic_motif_analysis()

        except Exception as e:
            self.logger.error(f"Error in artistic motif process: {e}")

    async def agent_initialize(self) -> None:
        """
        Artistic motif agent specific initialization.
        """
        try:
            # Initialize integration components
            await self._initialize_integrations()

            # Load artistic and cultural data
            await self._load_artistic_motif_data()

            self.logger.info("ArtisticMotifAgent initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize ArtisticMotifAgent: {e}")
            raise

    async def _initialize_integrations(self) -> None:
        """Initialize integration with other agents and processors."""
        try:
            # Initialize text processor for artistic text analysis
            self.text_processor = TextProcessor()
            await self.text_processor.initialize()

            # Initialize validation agent for multi-layered validation
            self.validation_agent = MultiLayeredValidationAgent(
                agent_id=f"{self.agent_id}_validation"
            )
            await self.validation_agent.agent_initialize()

            # Initialize pattern discovery for motif pattern analysis
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
        Setup artistic motif-specific message subscriptions.
        """
        if not self.messaging:
            return

        # Subscribe to artistic motif analysis requests
        artistic_sub_id = await self.nats_client.subscribe(
            "agents.artistic_motif.request",
            self._handle_artistic_request,
            queue_group="artistic-motif-workers"
        )
        self.subscription_ids.append(artistic_sub_id)

        # Subscribe to motif diffusion analysis
        diffusion_sub_id = await self.nats_client.subscribe(
            "agents.artistic.diffusion.request",
            self._handle_diffusion_request,
            queue_group="motif-diffusion"
        )
        self.subscription_ids.append(diffusion_sub_id)

        # Subscribe to cultural contact analysis
        contact_sub_id = await self.nats_client.subscribe(
            "agents.artistic.contact.request",
            self._handle_contact_request,
            queue_group="cultural-contact"
        )
        self.subscription_ids.append(contact_sub_id)

    async def handle_task(self, task: Task) -> None:
        """
        Handle artistic motif analysis tasks.
        """
        self.logger.info(f"Processing artistic motif task {task.task_id}: {task.task_type}")

        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)

            result = None

            if task.task_type == "motif_diffusion_analysis":
                result = await self.analyze_motif_diffusion(
                    task.input_data.get("motifs", []),
                    task.input_data.get("cultural_contexts", []),
                    task.parameters.get("diffusion_model", "diffusion")
                )
            elif task.task_type == "cultural_contact_artistic":
                result = await self.analyze_cultural_contact_artistic(
                    task.input_data.get("artifacts", []),
                    task.parameters.get("contact_type", "trade")
                )
            elif task.task_type == "stylistic_convergence":
                result = await self.analyze_stylistic_convergence(
                    task.input_data.get("artworks", []),
                    task.parameters.get("convergence_method", "similarity")
                )
            elif task.task_type == "iconographic_exchange":
                result = await self.analyze_iconographic_exchange(
                    task.input_data.get("iconography", []),
                    task.parameters.get("exchange_network", "regional")
                )
            elif task.task_type == "motif_evolution_tracking":
                result = await self.analyze_motif_evolution_tracking(
                    task.input_data.get("motif_series", []),
                    task.parameters.get("evolution_type", "chronological")
                )
            else:
                raise ValueError(f"Unknown artistic motif task type: {task.task_type}")

            # Store results in database
            if result:
                await self._store_artistic_results(task_id, result)

            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)

            self.processed_tasks += 1
            self.motif_analyses_performed += 1

            # Update counters based on result type
            if result and "diffusion_patterns" in result:
                self.diffusion_patterns_identified += len(result["diffusion_patterns"])
            if result and "cultural_contacts" in result:
                self.cultural_contacts_detected += len(result["cultural_contacts"])
            if result and "motif_evolutions" in result:
                self.motif_evolutions_tracked += len(result["motif_evolutions"])
            if result and "stylistic_convergences" in result:
                self.stylistic_convergences_mapped += len(result["stylistic_convergences"])

            self.logger.info(f"Completed artistic motif task {task_id}")

        except Exception as e:
            self.logger.error(f"Error processing artistic motif task {task.task_id}: {e}")

            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)

            self.failed_tasks += 1

        finally:
            self.current_tasks.discard(task_id)

    async def analyze_motif_diffusion(
        self,
        motifs: List[Dict[str, Any]],
        cultural_contexts: List[Dict[str, Any]],
        diffusion_model: str = "diffusion"
    ) -> Dict[str, Any]:
        """
        Analyze diffusion patterns of artistic motifs across cultures.

        Args:
            motifs: List of artistic motifs with metadata
            cultural_contexts: Cultural context information
            diffusion_model: Model for diffusion analysis

        Returns:
            Motif diffusion analysis results
        """
        try:
            self.logger.info(f"Analyzing motif diffusion using {diffusion_model} model")

            # Classify motifs by category and characteristics
            classified_motifs = self._classify_motifs(motifs)

            # Identify similar motifs across cultures
            motif_similarities = self._identify_motif_similarities(classified_motifs)

            # Apply diffusion model
            if diffusion_model == "diffusion":
                diffusion_patterns = self._apply_diffusion_model(motif_similarities, cultural_contexts)
            elif diffusion_model == "migration":
                diffusion_patterns = self._apply_migration_model(motif_similarities, cultural_contexts)
            elif diffusion_model == "trade":
                diffusion_patterns = self._apply_trade_model(motif_similarities, cultural_contexts)
            else:
                diffusion_patterns = []

            # Calculate diffusion confidence scores
            diffusion_confidence = self._calculate_diffusion_confidence(diffusion_patterns)

            # Identify significant diffusion patterns
            significant_patterns = self._identify_significant_diffusion_patterns(
                diffusion_patterns, diffusion_confidence
            )

            # Analyze transmission routes
            transmission_routes = self._analyze_transmission_routes(significant_patterns, cultural_contexts)

            # Integrate with text processor for cultural narratives
            if self.text_processor and significant_patterns:
                await self._integrate_cultural_narratives(significant_patterns)

            # Validate diffusion patterns
            if self.validation_agent and significant_patterns:
                validation_results = await self._validate_diffusion_patterns(significant_patterns)

            result = {
                "analysis_type": "motif_diffusion",
                "diffusion_model": diffusion_model,
                "motifs_analyzed": len(motifs),
                "cultural_contexts": len(cultural_contexts),
                "classified_motifs": classified_motifs,
                "motif_similarities": motif_similarities,
                "diffusion_patterns": diffusion_patterns,
                "diffusion_confidence": diffusion_confidence,
                "significant_patterns": significant_patterns,
                "transmission_routes": transmission_routes,
                "total_patterns": len(diffusion_patterns),
                "significant_count": len(significant_patterns),
                "validation_results": validation_results if 'validation_results' in locals() else None,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing motif diffusion: {e}")
            return {
                "error": str(e),
                "diffusion_model": diffusion_model,
                "diffusion_patterns": [],
                "significant_patterns": []
            }

    async def analyze_cultural_contact_artistic(
        self,
        artifacts: List[Dict[str, Any]],
        contact_type: str = "trade"
    ) -> Dict[str, Any]:
        """
        Analyze cultural contact patterns through artistic artifacts.

        Args:
            artifacts: List of artistic artifacts
            contact_type: Type of cultural contact

        Returns:
            Cultural contact analysis results
        """
        try:
            self.logger.info(f"Analyzing cultural contact through artistic artifacts using {contact_type} model")

            # Extract artistic features from artifacts
            artistic_features = self._extract_artistic_features(artifacts)

            # Identify contact indicators in art
            contact_indicators = self._identify_contact_indicators(artistic_features, contact_type)

            # Map cultural contact networks
            contact_networks = self._map_cultural_contact_networks(contact_indicators, artifacts)

            # Analyze contact chronology
            contact_chronology = self._analyze_contact_chronology(contact_networks)

            # Identify hybrid artistic styles
            hybrid_styles = self._identify_hybrid_artistic_styles(contact_networks)

            # Calculate contact intensity
            contact_intensity = self._calculate_contact_intensity(contact_networks)

            result = {
                "analysis_type": "cultural_contact_artistic",
                "contact_type": contact_type,
                "artifacts_analyzed": len(artifacts),
                "artistic_features": artistic_features,
                "contact_indicators": contact_indicators,
                "contact_networks": contact_networks,
                "contact_chronology": contact_chronology,
                "hybrid_styles": hybrid_styles,
                "contact_intensity": contact_intensity,
                "significant_contacts": len([c for c in contact_networks if c.get("contact_strength", 0) > self.cultural_contact_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing cultural contact: {e}")
            return {
                "error": str(e),
                "contact_type": contact_type,
                "contact_networks": [],
                "hybrid_styles": []
            }

    async def analyze_stylistic_convergence(
        self,
        artworks: List[Dict[str, Any]],
        convergence_method: str = "similarity"
    ) -> Dict[str, Any]:
        """
        Analyze stylistic convergence in artworks across cultures.

        Args:
            artworks: List of artworks with stylistic data
            convergence_method: Method for convergence analysis

        Returns:
            Stylistic convergence analysis results
        """
        try:
            self.logger.info(f"Analyzing stylistic convergence using {convergence_method} method")

            # Extract stylistic features
            stylistic_features = self._extract_stylistic_features(artworks)

            # Calculate stylistic similarities
            if convergence_method == "similarity":
                stylistic_similarities = self._calculate_stylistic_similarities(stylistic_features)
            elif convergence_method == "clustering":
                stylistic_similarities = self._calculate_stylistic_clustering(stylistic_features)
            elif convergence_method == "network":
                stylistic_similarities = self._calculate_stylistic_network(stylistic_features)
            else:
                stylistic_similarities = []

            # Identify convergence zones
            convergence_zones = self._identify_convergence_zones(stylistic_similarities, artworks)

            # Analyze convergence chronology
            convergence_chronology = self._analyze_convergence_chronology(convergence_zones)

            # Map stylistic influence flows
            influence_flows = self._map_stylistic_influence_flows(convergence_zones)

            # Calculate convergence strength
            convergence_strength = self._calculate_convergence_strength(convergence_zones)

            result = {
                "analysis_type": "stylistic_convergence",
                "convergence_method": convergence_method,
                "artworks_analyzed": len(artworks),
                "stylistic_features": stylistic_features,
                "stylistic_similarities": stylistic_similarities,
                "convergence_zones": convergence_zones,
                "convergence_chronology": convergence_chronology,
                "influence_flows": influence_flows,
                "convergence_strength": convergence_strength,
                "significant_convergences": len([c for c in convergence_zones if c.get("convergence_score", 0) > self.diffusion_confidence_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing stylistic convergence: {e}")
            return {
                "error": str(e),
                "convergence_method": convergence_method,
                "convergence_zones": [],
                "influence_flows": []
            }

    async def analyze_iconographic_exchange(
        self,
        iconography: List[Dict[str, Any]],
        exchange_network: str = "regional"
    ) -> Dict[str, Any]:
        """
        Analyze iconographic exchange networks.

        Args:
            iconography: List of iconographic elements
            exchange_network: Type of exchange network

        Returns:
            Iconographic exchange analysis results
        """
        try:
            self.logger.info(f"Analyzing iconographic exchange in {exchange_network} network")

            # Classify iconographic elements
            classified_iconography = self._classify_iconographic_elements(iconography)

            # Identify exchange patterns
            if exchange_network == "regional":
                exchange_patterns = self._identify_regional_exchange_patterns(classified_iconography)
            elif exchange_network == "continental":
                exchange_patterns = self._identify_continental_exchange_patterns(classified_iconography)
            elif exchange_network == "global":
                exchange_patterns = self._identify_global_exchange_patterns(classified_iconography)
            else:
                exchange_patterns = []

            # Map exchange routes
            exchange_routes = self._map_exchange_routes(exchange_patterns)

            # Analyze exchange chronology
            exchange_chronology = self._analyze_exchange_chronology(exchange_routes)

            # Identify key exchange hubs
            exchange_hubs = self._identify_exchange_hubs(exchange_routes)

            # Calculate exchange intensity
            exchange_intensity = self._calculate_exchange_intensity(exchange_routes)

            result = {
                "analysis_type": "iconographic_exchange",
                "exchange_network": exchange_network,
                "iconography_analyzed": len(iconography),
                "classified_iconography": classified_iconography,
                "exchange_patterns": exchange_patterns,
                "exchange_routes": exchange_routes,
                "exchange_chronology": exchange_chronology,
                "exchange_hubs": exchange_hubs,
                "exchange_intensity": exchange_intensity,
                "significant_exchanges": len([e for e in exchange_patterns if e.get("exchange_confidence", 0) > self.diffusion_confidence_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing iconographic exchange: {e}")
            return {
                "error": str(e),
                "exchange_network": exchange_network,
                "exchange_patterns": [],
                "exchange_routes": []
            }

    async def analyze_motif_evolution_tracking(
        self,
        motif_series: List[Dict[str, Any]],
        evolution_type: str = "chronological"
    ) -> Dict[str, Any]:
        """
        Track evolution of artistic motifs over time.

        Args:
            motif_series: Series of related motifs
            evolution_type: Type of evolution analysis

        Returns:
            Motif evolution tracking results
        """
        try:
            self.logger.info(f"Tracking motif evolution using {evolution_type} analysis")

            # Organize motifs chronologically
            chronological_motifs = self._organize_motifs_chronologically(motif_series)

            # Analyze evolutionary changes
            if evolution_type == "chronological":
                evolutionary_changes = self._analyze_chronological_evolution(chronological_motifs)
            elif evolution_type == "stylistic":
                evolutionary_changes = self._analyze_stylistic_evolution(chronological_motifs)
            elif evolution_type == "cultural":
                evolutionary_changes = self._analyze_cultural_evolution(chronological_motifs)
            else:
                evolutionary_changes = []

            # Identify evolution patterns
            evolution_patterns = self._identify_evolution_patterns(evolutionary_changes)

            # Map evolutionary trajectories
            evolutionary_trajectories = self._map_evolutionary_trajectories(evolution_patterns)

            # Calculate evolution rates
            evolution_rates = self._calculate_evolution_rates(evolutionary_trajectories)

            # Identify evolutionary milestones
            evolutionary_milestones = self._identify_evolutionary_milestones(evolutionary_trajectories)

            result = {
                "analysis_type": "motif_evolution_tracking",
                "evolution_type": evolution_type,
                "motif_series": len(motif_series),
                "chronological_motifs": chronological_motifs,
                "evolutionary_changes": evolutionary_changes,
                "evolution_patterns": evolution_patterns,
                "evolutionary_trajectories": evolutionary_trajectories,
                "evolution_rates": evolution_rates,
                "evolutionary_milestones": evolutionary_milestones,
                "significant_evolutions": len([e for e in evolution_patterns if e.get("evolution_significance", 0) > self.diffusion_confidence_threshold]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing motif evolution: {e}")
            return {
                "error": str(e),
                "evolution_type": evolution_type,
                "evolution_patterns": [],
                "evolutionary_trajectories": []
            }

    # Core artistic analysis methods

    def _classify_motifs(self, motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify motifs by category and characteristics."""
        classified = []

        for motif in motifs:
            motif_description = motif.get("description", "").lower()
            motif_category = "unknown"

            # Classify based on description
            for category, patterns in self.motif_categories.items():
                for pattern in patterns:
                    if pattern in motif_description:
                        motif_category = category
                        break
                if motif_category != "unknown":
                    break

            classified_motif = motif.copy()
            classified_motif["category"] = motif_category
            classified_motif["classification_confidence"] = 0.8 if motif_category != "unknown" else 0.0
            classified.append(classified_motif)

        return classified

    def _identify_motif_similarities(self, classified_motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify similarities between motifs across cultures."""
        similarities = []

        # Compare motifs pairwise
        for motif1, motif2 in combinations(classified_motifs, 2):
            if motif1["category"] == motif2["category"]:
                similarity_score = self._calculate_motif_similarity(motif1, motif2)

                if similarity_score > self.motif_similarity_threshold:
                    similarity = {
                        "motif1": motif1,
                        "motif2": motif2,
                        "similarity_score": similarity_score,
                        "shared_category": motif1["category"],
                        "similarity_type": "categorical",
                        "similarity_id": str(uuid.uuid4())
                    }
                    similarities.append(similarity)

        return similarities

    def _apply_diffusion_model(self, similarities: List[Dict[str, Any]], cultural_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diffusion model to motif similarities."""
        diffusion_patterns = []

        # Group similarities by motif pairs
        motif_pairs = defaultdict(list)
        for sim in similarities:
            pair_key = f"{sim['motif1']['id']}_{sim['motif2']['id']}"
            motif_pairs[pair_key].append(sim)

        # Analyze diffusion for each pair
        for pair_key, pair_sims in motif_pairs.items():
            avg_similarity = statistics.mean([s["similarity_score"] for s in pair_sims])

            # Find cultural contexts
            motif1_culture = self._find_motif_culture(pair_sims[0]["motif1"], cultural_contexts)
            motif2_culture = self._find_motif_culture(pair_sims[0]["motif2"], cultural_contexts)

            if motif1_culture and motif2_culture and motif1_culture != motif2_culture:
                diffusion_pattern = {
                    "motif_pair": pair_key,
                    "source_culture": motif1_culture,
                    "target_culture": motif2_culture,
                    "diffusion_strength": avg_similarity,
                    "diffusion_type": "cultural_exchange",
                    "diffusion_model": "diffusion",
                    "pattern_id": str(uuid.uuid4())
                }
                diffusion_patterns.append(diffusion_pattern)

        return diffusion_patterns

    def _apply_migration_model(self, similarities: List[Dict[str, Any]], cultural_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply migration model to motif similarities."""
        migration_patterns = []

        # Similar to diffusion but focused on migration routes
        diffusion_patterns = self._apply_diffusion_model(similarities, cultural_contexts)

        for pattern in diffusion_patterns:
            migration_pattern = pattern.copy()
            migration_pattern["diffusion_model"] = "migration"
            migration_pattern["migration_type"] = "cultural_carriage"
            migration_patterns.append(migration_pattern)

        return migration_patterns

    def _apply_trade_model(self, similarities: List[Dict[str, Any]], cultural_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply trade model to motif similarities."""
        trade_patterns = []

        # Focus on trade route patterns
        diffusion_patterns = self._apply_diffusion_model(similarities, cultural_contexts)

        for pattern in diffusion_patterns:
            trade_pattern = pattern.copy()
            trade_pattern["diffusion_model"] = "trade"
            trade_pattern["trade_route_type"] = "cultural_goods"
            trade_patterns.append(trade_pattern)

        return trade_patterns

    def _calculate_diffusion_confidence(self, diffusion_patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for diffusion patterns."""
        confidence_scores = {}

        for pattern in diffusion_patterns:
            pattern_id = pattern.get("pattern_id", str(uuid.uuid4()))
            strength = pattern.get("diffusion_strength", 0.0)

            # Base confidence on strength and supporting evidence
            confidence = min(1.0, strength * 1.2)  # Boost for strong patterns
            confidence_scores[pattern_id] = confidence

        return confidence_scores

    def _identify_significant_diffusion_patterns(self, patterns: List[Dict[str, Any]], confidence_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify diffusion patterns that exceed significance thresholds."""
        significant = []

        for pattern in patterns:
            pattern_id = pattern.get("pattern_id", str(uuid.uuid4()))
            confidence = confidence_scores.get(pattern_id, 0.0)

            if confidence > self.diffusion_confidence_threshold:
                significant_pattern = pattern.copy()
                significant_pattern["confidence_score"] = confidence
                significant_pattern["significance_factors"] = self._identify_significance_factors(pattern)
                significant.append(significant_pattern)

        return significant

    def _analyze_transmission_routes(self, patterns: List[Dict[str, Any]], cultural_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze transmission routes for diffusion patterns."""
        routes = []

        for pattern in patterns:
            source_culture = pattern.get("source_culture")
            target_culture = pattern.get("target_culture")

            if source_culture and target_culture:
                # Find geographical and cultural connections
                route = self._find_cultural_route(source_culture, target_culture, cultural_contexts)

                if route:
                    transmission_route = {
                        "pattern_id": pattern.get("pattern_id"),
                        "source_culture": source_culture,
                        "target_culture": target_culture,
                        "route_type": route.get("type", "unknown"),
                        "route_distance": route.get("distance", 0),
                        "transmission_mechanism": pattern.get("diffusion_model"),
                        "route_confidence": route.get("confidence", 0.0),
                        "route_id": str(uuid.uuid4())
                    }
                    routes.append(transmission_route)

        return routes

    # Integration methods

    async def _integrate_cultural_narratives(self, patterns: List[Dict[str, Any]]) -> None:
        """Integrate cultural narratives from text processor."""
        if not self.text_processor:
            return

        try:
            for pattern in patterns:
                # Search for narratives related to motif diffusion
                source_culture = pattern.get("source_culture", "")
                target_culture = pattern.get("target_culture", "")

                if source_culture and target_culture:
                    search_text = f"{source_culture} {target_culture} artistic motif diffusion cultural exchange"
                    result = await self.text_processor.process_text_string(search_text)

                    if result.success:
                        pattern["cultural_narratives"] = result.entities
                        pattern["narrative_sentiment"] = result.sentiment_analysis

        except Exception as e:
            self.logger.warning(f"Failed to integrate cultural narratives: {e}")

    async def _validate_diffusion_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate diffusion patterns using multi-layered validation."""
        if not self.validation_agent:
            return {"validation_status": "no_validator_available"}

        try:
            validation_data = {
                "pattern_data": {
                    "pattern_type": "artistic_motif_diffusion",
                    "diffusion_patterns": patterns,
                    "metadata": {
                        "analysis_agent": self.agent_id,
                        "pattern_count": len(patterns)
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
            self.logger.error(f"Failed to validate diffusion patterns: {e}")
            return {"validation_error": str(e)}

    # Utility methods

    def _calculate_motif_similarity(self, motif1: Dict[str, Any], motif2: Dict[str, Any]) -> float:
        """Calculate similarity between two motifs."""
        # Simple similarity based on category and description overlap
        if motif1.get("category") != motif2.get("category"):
            return 0.0

        desc1 = set(motif1.get("description", "").lower().split())
        desc2 = set(motif2.get("description", "").lower().split())

        if not desc1 or not desc2:
            return 0.0

        intersection = desc1.intersection(desc2)
        union = desc1.union(desc2)

        return len(intersection) / len(union) if union else 0.0

    def _find_motif_culture(self, motif: Dict[str, Any], cultural_contexts: List[Dict[str, Any]]) -> Optional[str]:
        """Find the cultural context for a motif."""
        motif_culture = motif.get("culture", "").lower()

        for context in cultural_contexts:
            if context.get("name", "").lower() in motif_culture or motif_culture in context.get("name", "").lower():
                return context.get("name")

        return None

    def _find_cultural_route(self, source: str, target: str, contexts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find a cultural route between two cultures."""
        # Placeholder for route finding logic
        return {
            "type": "cultural_exchange",
            "distance": 1000,  # Placeholder
            "confidence": 0.7
        }

    def _identify_significance_factors(self, pattern: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to pattern significance."""
        factors = []

        strength = pattern.get("diffusion_strength", 0.0)
        if strength > 0.8:
            factors.append("high_similarity")
        if pattern.get("source_culture") and pattern.get("target_culture"):
            factors.append("cross_cultural")

        return factors

    # Placeholder methods for detailed analysis

    def _extract_artistic_features(self, artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract artistic features from artifacts."""
        return []

    def _identify_contact_indicators(self, features: List[Dict[str, Any]], contact_type: str) -> List[Dict[str, Any]]:
        """Identify contact indicators in artistic features."""
        return []

    def _map_cultural_contact_networks(self, indicators: List[Dict[str, Any]], artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map cultural contact networks."""
        return []

    def _analyze_contact_chronology(self, networks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze contact chronology."""
        return {}

    def _identify_hybrid_artistic_styles(self, networks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify hybrid artistic styles."""
        return []

    def _calculate_contact_intensity(self, networks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate contact intensity."""
        return {}

    def _extract_stylistic_features(self, artworks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract stylistic features from artworks."""
        return []

    def _calculate_stylistic_similarities(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate stylistic similarities."""
        return []

    def _calculate_stylistic_clustering(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate stylistic clustering."""
        return []

    def _calculate_stylistic_network(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate stylistic network."""
        return []

    def _identify_convergence_zones(self, similarities: List[Dict[str, Any]], artworks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify convergence zones."""
        return []

    def _analyze_convergence_chronology(self, zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convergence chronology."""
        return {}

    def _map_stylistic_influence_flows(self, zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map stylistic influence flows."""
        return []

    def _calculate_convergence_strength(self, zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate convergence strength."""
        return {}

    def _classify_iconographic_elements(self, iconography: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify iconographic elements."""
        return []

    def _identify_regional_exchange_patterns(self, iconography: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify regional exchange patterns."""
        return []

    def _identify_continental_exchange_patterns(self, iconography: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify continental exchange patterns."""
        return []

    def _identify_global_exchange_patterns(self, iconography: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify global exchange patterns."""
        return []

    def _map_exchange_routes(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map exchange routes."""
        return []

    def _analyze_exchange_chronology(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze exchange chronology."""
        return {}

    def _identify_exchange_hubs(self, routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify exchange hubs."""
        return []

    def _calculate_exchange_intensity(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate exchange intensity."""
        return {}

    def _organize_motifs_chronologically(self, motif_series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize motifs chronologically."""
        return sorted(motif_series, key=lambda x: x.get("date", 0))

    def _analyze_chronological_evolution(self, motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze chronological evolution."""
        return []

    def _analyze_stylistic_evolution(self, motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze stylistic evolution."""
        return []

    def _analyze_cultural_evolution(self, motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze cultural evolution."""
        return []

    def _identify_evolution_patterns(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify evolution patterns."""
        return []

    def _map_evolutionary_trajectories(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map evolutionary trajectories."""
        return []

    def _calculate_evolution_rates(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evolution rates."""
        return {}

    def _identify_evolutionary_milestones(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify evolutionary milestones."""
        return []

    # Data loading methods

    def _load_motif_database(self) -> Dict[str, Any]:
        """Load artistic motif database."""
        return {
            "geometric_motifs": {
                "spiral": {"cultures": ["celtic", "minoan", "prehistoric"], "meanings": ["eternity", "growth"]},
                "meander": {"cultures": ["greek", "etruscan", "roman"], "meanings": ["infinity", "flow"]},
                "chevron": {"cultures": ["mesopotamian", "egyptian"], "meanings": ["power", "protection"]}
            },
            "symbolic_motifs": {
                "cross": {"cultures": ["christian", "pagan", "hindu"], "meanings": ["intersection", "balance"]},
                "star": {"cultures": ["babylonian", "chinese", "islamic"], "meanings": ["guidance", "divine"]}
            }
        }

    def _load_cultural_artistic_lexicon(self) -> Dict[str, Any]:
        """Load cultural artistic lexicon."""
        return {
            "diffusion_terms": ["influence", "borrowing", "adoption", "transmission"],
            "contact_terms": ["exchange", "trade", "migration", "conquest"],
            "stylistic_terms": ["convergence", "hybridization", "fusion", "syncretism"]
        }

    async def _load_artistic_motif_data(self) -> None:
        """Load artistic motif analysis data."""
        # This would load motif databases, cultural exchange records, etc.
        pass

    # Database storage

    async def _store_artistic_results(self, task_id: str, results: Dict[str, Any]) -> None:
        """Store artistic motif analysis results in database."""
        try:
            protocol_result = {
                "protocol_type": "artistic_motif_diffusion_cultural_contact",
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

            self.logger.info(f"Stored artistic motif results with ID: {result_id}")

        except Exception as e:
            self.logger.error(f"Failed to store artistic motif results: {e}")

    # Message handlers

    async def _handle_artistic_request(self, message: AgentMessage) -> None:
        """Handle artistic motif analysis requests."""
        try:
            request_data = message.payload
            analysis_type = request_data.get("analysis_type", "motif_diffusion")
            input_data = request_data.get("input_data", {})

            # Perform analysis based on type
            if analysis_type == "motif_diffusion":
                result = await self.analyze_motif_diffusion(
                    input_data.get("motifs", []),
                    input_data.get("cultural_contexts", [])
                )
            elif analysis_type == "cultural_contact":
                result = await self.analyze_cultural_contact_artistic(
                    input_data.get("artifacts", [])
                )
            elif analysis_type == "stylistic_convergence":
                result = await self.analyze_stylistic_convergence(
                    input_data.get("artworks", [])
                )
            elif analysis_type == "iconographic_exchange":
                result = await self.analyze_iconographic_exchange(
                    input_data.get("iconography", [])
                )
            elif analysis_type == "motif_evolution":
                result = await self.analyze_motif_evolution_tracking(
                    input_data.get("motif_series", [])
                )
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="artistic_motif_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling artistic motif request: {e}")

    async def _handle_diffusion_request(self, message: AgentMessage) -> None:
        """Handle motif diffusion requests."""
        try:
            request_data = message.payload
            motifs = request_data.get("motifs", [])
            cultural_contexts = request_data.get("cultural_contexts", [])
            diffusion_model = request_data.get("diffusion_model", "diffusion")

            result = await self.analyze_motif_diffusion(motifs, cultural_contexts, diffusion_model)

            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="motif_diffusion_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling diffusion request: {e}")

    async def _handle_contact_request(self, message: AgentMessage) -> None:
        """Handle cultural contact requests."""
        try:
            request_data = message.payload
            artifacts = request_data.get("artifacts", [])
            contact_type = request_data.get("contact_type", "trade")

            result = await self.analyze_cultural_contact_artistic(artifacts, contact_type)

            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="cultural_contact_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling contact request: {e}")

    async def _process_artistic_queue(self) -> None:
        """Process queued artistic motif requests."""
        # Implementation for processing queued requests
        pass

    async def _perform_periodic_motif_analysis(self) -> None:
        """Perform periodic motif diffusion analysis."""
        try:
            # Fetch artistic data from database
            # This would integrate with external artistic databases

            self.logger.info("Performing periodic artistic motif analysis")

        except Exception as e:
            self.logger.error(f"Error in periodic motif analysis: {e}")

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect artistic motif agent metrics."""
        base_metrics = await super().collect_metrics() or {}

        artistic_metrics = {
            "motif_analyses_performed": self.motif_analyses_performed,
            "diffusion_patterns_identified": self.diffusion_patterns_identified,
            "cultural_contacts_detected": self.cultural_contacts_detected,
            "motif_evolutions_tracked": self.motif_evolutions_tracked,
            "stylistic_convergences_mapped": self.stylistic_convergences_mapped,
            "motif_categories": len(self.motif_categories),
            "transmission_models": len(self.transmission_models),
            "integration_status": {
                "text_processor": self.text_processor is not None,
                "validation_agent": self.validation_agent is not None,
                "pattern_discovery": self.pattern_discovery is not None
            }
        }

        return {**base_metrics, **artistic_metrics}

    def _get_capabilities(self) -> List[str]:
        """Get artistic motif agent capabilities."""
        return [
            "artistic_motif_agent",
            "motif_diffusion_analysis",
            "cultural_contact_artistic",
            "stylistic_convergence",
            "iconographic_exchange",
            "motif_evolution_tracking",
            "diffusion_model_analysis",
            "migration_model_analysis",
            "trade_model_analysis",
            "similarity_detection",
            "cultural_exchange_mapping",
            "chronological_evolution",
            "stylistic_evolution",
            "cultural_evolution",
            "transmission_route_analysis"
        ]

    async def shutdown(self) -> None:
        """Shutdown artistic motif agent."""
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

            self.logger.info("ArtisticMotifAgent shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during artistic motif agent shutdown: {e}")


# Factory function
def create_artistic_motif_agent(agent_id: Optional[str] = None, **kwargs) -> ArtisticMotifAgent:
    """
    Factory function to create artistic motif agents.

    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured artistic motif agent
    """
    return ArtisticMotifAgent(agent_id=agent_id, **kwargs)


# Main entry point
async def main():
    """Main entry point for running the ArtisticMotifAgent."""
    import signal
    import sys

    # Create and configure agent
    agent = ArtisticMotifAgent()

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
        print(f"ArtisticMotifAgent failed: {e}")
        sys.exit(1)

    print("ArtisticMotifAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())