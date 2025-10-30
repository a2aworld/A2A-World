"""
A2A World Platform - Archaeoastronomy Agent

Specialized agent for archaeoastronomy and celestial alignment analysis.
Analyzes astronomical correlations, celestial alignments, and astronomical
significance of sacred sites across different cultures and time periods.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import json
from math import radians, degrees, sin, cos, atan2, sqrt, pi

from agents.core.base_agent import BaseAgent
from agents.core.config import DiscoveryAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.pattern_storage import PatternStorage

# Import integration components
from agents.parsers.data_processors.text_processor import TextProcessor
from agents.validation.multi_layered_validation_agent import MultiLayeredValidationAgent
from agents.discovery.pattern_discovery import PatternDiscoveryAgent


class ArchaeoastronomyAgent(BaseAgent):
    """
    Agent specialized in archaeoastronomy and celestial alignment analysis.

    Capabilities:
    - Celestial alignment detection (solstice, equinox, lunar alignments)
    - Astronomical correlation analysis
    - Horizon astronomy and astronomical sightlines
    - Cultural astronomical practices analysis
    - Temporal astronomical pattern recognition
    - Cross-cultural astronomical comparisons
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DiscoveryAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="archaeoastronomy",
            config=config or DiscoveryAgentConfig(),
            config_file=config_file
        )

        # Astronomical constants
        self.EARTH_RADIUS_KM = 6371.0
        self.SOLAR_YEAR_DAYS = 365.25
        self.LUNAR_MONTH_DAYS = 29.53

        # Celestial alignment thresholds
        self.alignment_tolerance_degrees = 1.0  # Tolerance for alignment detection
        self.horizon_alignment_threshold = 0.5  # Degrees from horizon
        self.sightline_precision_km = 0.1  # Precision for sightline calculations

        # Analysis parameters
        self.min_sites_for_alignment = 3
        self.confidence_threshold = self.config.confidence_threshold
        self.search_radius_km = self.config.search_radius_km

        # Performance tracking
        self.analyses_performed = 0
        self.alignments_discovered = 0
        self.correlations_found = 0
        self.cultural_patterns_identified = 0

        # Integration components
        self.text_processor = None
        self.validation_agent = None
        self.pattern_discovery = None

        # Astronomical knowledge base
        self.astronomical_sites = self._load_astronomical_sites()
        self.cultural_astronomy_practices = self._load_cultural_astronomy()

        # Database integration
        self.pattern_storage = PatternStorage()

        self.logger.info(f"ArchaeoastronomyAgent {self.agent_id} initialized")

    async def process(self) -> None:
        """
        Main processing loop for archaeoastronomy analysis.
        """
        try:
            # Process any pending archaeoastronomy requests
            await self._process_archaeoastronomy_queue()

            # Perform periodic astronomical correlation analysis
            if self.processed_tasks % 100 == 0:
                await self._perform_periodic_astronomical_analysis()

        except Exception as e:
            self.logger.error(f"Error in archaeoastronomy process: {e}")

    async def agent_initialize(self) -> None:
        """
        Archaeoastronomy agent specific initialization.
        """
        try:
            # Initialize integration components
            await self._initialize_integrations()

            # Load astronomical data
            await self._load_astronomical_data()

            self.logger.info("ArchaeoastronomyAgent initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize ArchaeoastronomyAgent: {e}")
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

            # Initialize pattern discovery for base pattern analysis
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
        Setup archaeoastronomy-specific message subscriptions.
        """
        if not self.messaging:
            return

        # Subscribe to archaeoastronomy analysis requests
        archaeo_sub_id = await self.nats_client.subscribe(
            "agents.archaeoastronomy.request",
            self._handle_archaeoastronomy_request,
            queue_group="archaeoastronomy-workers"
        )
        self.subscription_ids.append(archaeo_sub_id)

        # Subscribe to celestial alignment detection requests
        alignment_sub_id = await self.nats_client.subscribe(
            "agents.archaeoastronomy.alignment.request",
            self._handle_alignment_request,
            queue_group="alignment-analysis"
        )
        self.subscription_ids.append(alignment_sub_id)

        # Subscribe to astronomical correlation analysis
        correlation_sub_id = await self.nats_client.subscribe(
            "agents.archaeoastronomy.correlation.request",
            self._handle_correlation_request,
            queue_group="astronomical-correlation"
        )
        self.subscription_ids.append(correlation_sub_id)

    async def handle_task(self, task: Task) -> None:
        """
        Handle archaeoastronomy analysis tasks.
        """
        self.logger.info(f"Processing archaeoastronomy task {task.task_id}: {task.task_type}")

        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)

            result = None

            if task.task_type == "celestial_alignment_analysis":
                result = await self.analyze_celestial_alignments(
                    task.input_data.get("sites", []),
                    task.parameters.get("alignment_types", ["solstice", "equinox"])
                )
            elif task.task_type == "astronomical_correlation":
                result = await self.analyze_astronomical_correlations(
                    task.input_data.get("sites", []),
                    task.parameters.get("correlation_type", "solar")
                )
            elif task.task_type == "horizon_astronomy":
                result = await self.analyze_horizon_astronomy(
                    task.input_data.get("site", {}),
                    task.parameters.get("horizon_data", {})
                )
            elif task.task_type == "cultural_astronomy":
                result = await self.analyze_cultural_astronomy(
                    task.input_data.get("sites", []),
                    task.parameters.get("cultural_context", {})
                )
            else:
                raise ValueError(f"Unknown archaeoastronomy task type: {task.task_type}")

            # Store results in database
            if result:
                await self._store_archaeoastronomy_results(task_id, result)

            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)

            self.processed_tasks += 1
            self.analyses_performed += 1

            # Update counters
            if result and "alignments" in result:
                self.alignments_discovered += len(result["alignments"])
            if result and "correlations" in result:
                self.correlations_found += len(result["correlations"])

            self.logger.info(f"Completed archaeoastronomy task {task_id}")

        except Exception as e:
            self.logger.error(f"Error processing archaeoastronomy task {task.task_id}: {e}")

            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)

            self.failed_tasks += 1

        finally:
            self.current_tasks.discard(task.task_id)

    async def analyze_celestial_alignments(
        self,
        sites: List[Dict[str, Any]],
        alignment_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze celestial alignments for sacred sites.

        Args:
            sites: List of sacred sites with coordinates
            alignment_types: Types of alignments to analyze

        Returns:
            Analysis results with detected alignments
        """
        if alignment_types is None:
            alignment_types = ["solstice", "equinox", "lunar"]

        try:
            self.logger.info(f"Analyzing celestial alignments for {len(sites)} sites")

            alignments = []
            site_coordinates = []

            # Extract coordinates
            for site in sites:
                lat = site.get("latitude")
                lon = site.get("longitude")
                if lat is not None and lon is not None:
                    site_coordinates.append((lat, lon, site))

            # Analyze each alignment type
            for alignment_type in alignment_types:
                if alignment_type == "solstice":
                    type_alignments = await self._analyze_solstice_alignments(site_coordinates)
                elif alignment_type == "equinox":
                    type_alignments = await self._analyze_equinox_alignments(site_coordinates)
                elif alignment_type == "lunar":
                    type_alignments = await self._analyze_lunar_alignments(site_coordinates)
                else:
                    continue

                alignments.extend(type_alignments)

            # Calculate alignment significance
            significant_alignments = []
            for alignment in alignments:
                significance = self._calculate_alignment_significance(alignment, sites)
                alignment["significance_score"] = significance
                if significance > self.confidence_threshold:
                    significant_alignments.append(alignment)

            # Integrate with text processor for mythological context
            if self.text_processor and significant_alignments:
                await self._integrate_mythological_context(significant_alignments)

            # Validate results using multi-layered validation
            if self.validation_agent and significant_alignments:
                validation_results = await self._validate_alignments(significant_alignments)
            else:
                validation_results = {"validation_status": "not_validated"}

            result = {
                "analysis_type": "celestial_alignment",
                "sites_analyzed": len(sites),
                "alignment_types": alignment_types,
                "alignments": alignments,
                "significant_alignments": significant_alignments,
                "total_alignments": len(alignments),
                "significant_count": len(significant_alignments),
                "validation_results": validation_results,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing celestial alignments: {e}")
            return {
                "error": str(e),
                "sites_analyzed": len(sites),
                "alignments": [],
                "significant_alignments": []
            }

    async def analyze_astronomical_correlations(
        self,
        sites: List[Dict[str, Any]],
        correlation_type: str = "solar"
    ) -> Dict[str, Any]:
        """
        Analyze astronomical correlations between sites.

        Args:
            sites: List of sacred sites
            correlation_type: Type of astronomical correlation

        Returns:
            Correlation analysis results
        """
        try:
            self.logger.info(f"Analyzing astronomical correlations for {len(sites)} sites")

            correlations = []

            if correlation_type == "solar":
                correlations = await self._analyze_solar_correlations(sites)
            elif correlation_type == "lunar":
                correlations = await self._analyze_lunar_correlations(sites)
            elif correlation_type == "stellar":
                correlations = await self._analyze_stellar_correlations(sites)

            # Calculate correlation significance
            significant_correlations = []
            for correlation in correlations:
                significance = self._calculate_correlation_significance(correlation, sites)
                correlation["significance_score"] = significance
                if significance > self.confidence_threshold:
                    significant_correlations.append(correlation)

            # Cross-reference with pattern discovery
            if self.pattern_discovery and significant_correlations:
                pattern_results = await self._integrate_pattern_discovery(significant_correlations)

            result = {
                "analysis_type": "astronomical_correlation",
                "correlation_type": correlation_type,
                "sites_analyzed": len(sites),
                "correlations": correlations,
                "significant_correlations": significant_correlations,
                "total_correlations": len(correlations),
                "significant_count": len(significant_correlations),
                "pattern_integration": pattern_results if 'pattern_results' in locals() else None,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing astronomical correlations: {e}")
            return {
                "error": str(e),
                "correlation_type": correlation_type,
                "correlations": [],
                "significant_correlations": []
            }

    async def analyze_horizon_astronomy(
        self,
        site: Dict[str, Any],
        horizon_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze horizon astronomy for a specific site.

        Args:
            site: Sacred site data
            horizon_data: Horizon elevation data

        Returns:
            Horizon astronomy analysis results
        """
        try:
            self.logger.info(f"Analyzing horizon astronomy for site: {site.get('name', 'unknown')}")

            # Calculate astronomical sightlines
            sightlines = self._calculate_astronomical_sightlines(site, horizon_data)

            # Analyze astronomical events visible from horizon
            astronomical_events = self._analyze_horizon_events(site, horizon_data)

            # Determine astronomical significance
            significance = self._calculate_horizon_significance(sightlines, astronomical_events)

            result = {
                "analysis_type": "horizon_astronomy",
                "site": site,
                "sightlines": sightlines,
                "astronomical_events": astronomical_events,
                "significance_score": significance,
                "horizon_analysis_complete": True,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing horizon astronomy: {e}")
            return {
                "error": str(e),
                "site": site,
                "sightlines": [],
                "astronomical_events": []
            }

    async def analyze_cultural_astronomy(
        self,
        sites: List[Dict[str, Any]],
        cultural_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze cultural astronomy practices across sites.

        Args:
            sites: List of sacred sites
            cultural_context: Cultural context information

        Returns:
            Cultural astronomy analysis results
        """
        try:
            self.logger.info(f"Analyzing cultural astronomy for {len(sites)} sites")

            # Identify cultural astronomical patterns
            cultural_patterns = await self._identify_cultural_patterns(sites, cultural_context)

            # Cross-reference with mythological texts
            if self.text_processor:
                mythological_references = await self._cross_reference_mythology(sites, cultural_context)

            # Analyze astronomical knowledge transmission
            transmission_patterns = self._analyze_knowledge_transmission(sites, cultural_patterns)

            result = {
                "analysis_type": "cultural_astronomy",
                "sites_analyzed": len(sites),
                "cultural_patterns": cultural_patterns,
                "mythological_references": mythological_references if 'mythological_references' in locals() else None,
                "transmission_patterns": transmission_patterns,
                "cultural_context": cultural_context,
                "patterns_identified": len(cultural_patterns),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }

            self.cultural_patterns_identified += len(cultural_patterns)

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing cultural astronomy: {e}")
            return {
                "error": str(e),
                "cultural_patterns": [],
                "transmission_patterns": []
            }

    # Core astronomical analysis methods

    async def _analyze_solstice_alignments(self, site_coordinates: List[Tuple]) -> List[Dict[str, Any]]:
        """Analyze solstice alignments between sites."""
        alignments = []

        # Summer solstice sunrise/sunset azimuths (approximate)
        summer_solstice_azimuths = {
            "sunrise": 55.0,  # Northern hemisphere
            "sunset": 305.0
        }

        # Winter solstice azimuths
        winter_solstice_azimuths = {
            "sunrise": 125.0,
            "sunset": 235.0
        }

        for i, (lat1, lon1, site1) in enumerate(site_coordinates):
            for j, (lat2, lon2, site2) in enumerate(site_coordinates[i+1:], i+1):
                # Calculate bearing between sites
                bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
                distance = self._calculate_distance(lat1, lon1, lat2, lon2)

                # Check alignment with solstice azimuths
                for season, azimuths in [("summer", summer_solstice_azimuths), ("winter", winter_solstice_azimuths)]:
                    for event, azimuth in azimuths.items():
                        alignment_angle = abs(bearing - azimuth)
                        if alignment_angle > 180:
                            alignment_angle = 360 - alignment_angle

                        if alignment_angle <= self.alignment_tolerance_degrees:
                            alignment = {
                                "type": "solstice_alignment",
                                "season": season,
                                "event": event,
                                "azimuth": azimuth,
                                "site1": site1,
                                "site2": site2,
                                "bearing": bearing,
                                "distance_km": distance,
                                "alignment_precision": alignment_angle,
                                "alignment_id": str(uuid.uuid4())
                            }
                            alignments.append(alignment)

        return alignments

    async def _analyze_equinox_alignments(self, site_coordinates: List[Tuple]) -> List[Dict[str, Any]]:
        """Analyze equinox alignments."""
        alignments = []

        # Equinox sunrise/sunset azimuths (due east/west)
        equinox_azimuths = {
            "sunrise": 90.0,
            "sunset": 270.0
        }

        for i, (lat1, lon1, site1) in enumerate(site_coordinates):
            for j, (lat2, lon2, site2) in enumerate(site_coordinates[i+1:], i+1):
                bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
                distance = self._calculate_distance(lat1, lon1, lat2, lon2)

                for event, azimuth in equinox_azimuths.items():
                    alignment_angle = abs(bearing - azimuth)
                    if alignment_angle > 180:
                        alignment_angle = 360 - alignment_angle

                    if alignment_angle <= self.alignment_tolerance_degrees:
                        alignment = {
                            "type": "equinox_alignment",
                            "event": event,
                            "azimuth": azimuth,
                            "site1": site1,
                            "site2": site2,
                            "bearing": bearing,
                            "distance_km": distance,
                            "alignment_precision": alignment_angle,
                            "alignment_id": str(uuid.uuid4())
                        }
                        alignments.append(alignment)

        return alignments

    async def _analyze_lunar_alignments(self, site_coordinates: List[Tuple]) -> List[Dict[str, Any]]:
        """Analyze lunar alignments."""
        alignments = []

        # Major lunar standstill azimuths (approximate)
        lunar_standstill_azimuths = {
            "major_standstill_rise": 45.0,
            "major_standstill_set": 315.0,
            "minor_standstill_rise": 67.5,
            "minor_standstill_set": 292.5
        }

        for i, (lat1, lon1, site1) in enumerate(site_coordinates):
            for j, (lat2, lon2, site2) in enumerate(site_coordinates[i+1:], i+1):
                bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
                distance = self._calculate_distance(lat1, lon1, lat2, lon2)

                for standstill, azimuth in lunar_standstill_azimuths.items():
                    alignment_angle = abs(bearing - azimuth)
                    if alignment_angle > 180:
                        alignment_angle = 360 - alignment_angle

                    if alignment_angle <= self.alignment_tolerance_degrees:
                        alignment = {
                            "type": "lunar_standstill_alignment",
                            "standstill": standstill,
                            "azimuth": azimuth,
                            "site1": site1,
                            "site2": site2,
                            "bearing": bearing,
                            "distance_km": distance,
                            "alignment_precision": alignment_angle,
                            "alignment_id": str(uuid.uuid4())
                        }
                        alignments.append(alignment)

        return alignments

    async def _analyze_solar_correlations(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze solar astronomical correlations."""
        correlations = []

        # Analyze solar year correlations
        solar_cycles = self._identify_solar_cycles(sites)
        correlations.extend(solar_cycles)

        # Analyze solar event correlations
        solar_events = self._correlate_solar_events(sites)
        correlations.extend(solar_events)

        return correlations

    async def _analyze_lunar_correlations(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze lunar astronomical correlations."""
        correlations = []

        # Analyze lunar cycle correlations
        lunar_cycles = self._identify_lunar_cycles(sites)
        correlations.extend(lunar_cycles)

        # Analyze lunar event correlations
        lunar_events = self._correlate_lunar_events(sites)
        correlations.extend(lunar_events)

        return correlations

    async def _analyze_stellar_correlations(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze stellar astronomical correlations."""
        correlations = []

        # Analyze stellar alignment correlations
        stellar_alignments = self._identify_stellar_alignments(sites)
        correlations.extend(stellar_alignments)

        return correlations

    # Integration methods

    async def _integrate_mythological_context(self, alignments: List[Dict[str, Any]]) -> None:
        """Integrate mythological context from text processor."""
        if not self.text_processor:
            return

        try:
            for alignment in alignments:
                # Extract site names for text search
                site1_name = alignment["site1"].get("name", "")
                site2_name = alignment["site2"].get("name", "")

                if site1_name or site2_name:
                    search_text = f"{site1_name} {site2_name} astronomy alignment celestial"
                    result = await self.text_processor.process_text_string(
                        search_text, extract_entities=True, analyze_sentiment=True
                    )

                    if result.success and result.entities:
                        alignment["mythological_context"] = result.entities
                        alignment["sentiment_analysis"] = result.sentiment_analysis

        except Exception as e:
            self.logger.warning(f"Failed to integrate mythological context: {e}")

    async def _validate_alignments(self, alignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate alignments using multi-layered validation."""
        if not self.validation_agent:
            return {"validation_status": "no_validator_available"}

        try:
            # Prepare alignment data for validation
            validation_data = {
                "pattern_data": {
                    "pattern_type": "astronomical_alignment",
                    "alignments": alignments,
                    "metadata": {
                        "analysis_agent": self.agent_id,
                        "alignment_count": len(alignments)
                    }
                }
            }

            # Perform multi-layered validation
            validation_result = await self.validation_agent.validate_pattern_multi_layered(
                pattern_id=str(uuid.uuid4()),
                pattern_data=validation_data,
                validation_layers=["statistical", "cultural"],
                store_results=True
            )

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate alignments: {e}")
            return {"validation_error": str(e)}

    async def _integrate_pattern_discovery(self, correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate with pattern discovery agent."""
        if not self.pattern_discovery:
            return {"integration_status": "no_pattern_discovery_available"}

        try:
            # Convert correlations to pattern format
            pattern_data = {
                "features": correlations,
                "metadata": {
                    "source": "archaeoastronomy_correlations",
                    "correlation_count": len(correlations)
                }
            }

            # Discover patterns
            pattern_result = await self.pattern_discovery.discover_patterns(
                pattern_data, algorithm="hdbscan"
            )

            return pattern_result

        except Exception as e:
            self.logger.error(f"Failed to integrate pattern discovery: {e}")
            return {"integration_error": str(e)}

    # Utility methods

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points."""
        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)

        delta_lon = lon2_rad - lon1_rad

        x = sin(delta_lon) * cos(lat2_rad)
        y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(delta_lon)

        bearing_rad = atan2(x, y)
        bearing_deg = degrees(bearing_rad)

        return (bearing_deg + 360) % 360

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using haversine formula."""
        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)

        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return self.EARTH_RADIUS_KM * c

    def _calculate_alignment_significance(self, alignment: Dict[str, Any], all_sites: List[Dict[str, Any]]) -> float:
        """Calculate statistical significance of an alignment."""
        try:
            precision = alignment.get("alignment_precision", 1.0)
            distance = alignment.get("distance_km", 0)

            # Base significance on precision and distance
            precision_score = max(0, 1.0 - (precision / self.alignment_tolerance_degrees))
            distance_score = min(1.0, distance / 1000.0)  # Favor longer alignments

            # Consider number of sites
            site_ratio = len(all_sites) / max(1, len(all_sites))
            site_score = min(1.0, site_ratio)

            significance = (precision_score * 0.5) + (distance_score * 0.3) + (site_score * 0.2)

            return min(1.0, significance)

        except Exception:
            return 0.0

    def _calculate_correlation_significance(self, correlation: Dict[str, Any], all_sites: List[Dict[str, Any]]) -> float:
        """Calculate significance of astronomical correlation."""
        try:
            correlation_strength = correlation.get("correlation_coefficient", 0.0)
            sample_size = correlation.get("sample_size", 1)

            # Statistical significance based on correlation strength and sample size
            strength_score = abs(correlation_strength)
            size_score = min(1.0, sample_size / 10.0)

            significance = (strength_score * 0.7) + (size_score * 0.3)

            return min(1.0, significance)

        except Exception:
            return 0.0

    def _calculate_horizon_significance(self, sightlines: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> float:
        """Calculate significance of horizon astronomy."""
        try:
            sightline_score = min(1.0, len(sightlines) / 10.0)
            event_score = min(1.0, len(events) / 5.0)

            significance = (sightline_score * 0.6) + (event_score * 0.4)

            return significance

        except Exception:
            return 0.0

    # Data loading methods

    def _load_astronomical_sites(self) -> Dict[str, Any]:
        """Load known astronomical sites database."""
        return {
            "stonehenge": {"lat": 51.1789, "lon": -1.8262, "type": "stone_circle"},
            "newgrange": {"lat": 53.6947, "lon": -6.4751, "type": "passage_tomb"},
            "chichen_itza": {"lat": 20.6829, "lon": -88.5686, "type": "pyramid"},
            "carnac": {"lat": 47.5877, "lon": -3.0783, "type": "standing_stones"}
        }

    def _load_cultural_astronomy(self) -> Dict[str, Any]:
        """Load cultural astronomy practices."""
        return {
            "mayan": ["zenith_passage", "solstice_celebrations", "lunar_tables"],
            "egyptian": ["sirius_heliacal_rising", "nile_flood_correlation"],
            "mesopotamian": ["venus_tablets", "lunar_eclipses"],
            "indigenous_american": ["medicine_wheel", "solstice_markers"]
        }

    async def _load_astronomical_data(self) -> None:
        """Load astronomical calculation data."""
        # This would load astronomical constants, ephemeris data, etc.
        pass

    # Placeholder methods for astronomical analysis

    def _calculate_astronomical_sightlines(self, site: Dict[str, Any], horizon_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate astronomical sightlines from horizon data."""
        # Placeholder implementation
        return []

    def _analyze_horizon_events(self, site: Dict[str, Any], horizon_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze astronomical events visible from horizon."""
        # Placeholder implementation
        return []

    def _identify_solar_cycles(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify solar cycle correlations."""
        # Placeholder implementation
        return []

    def _correlate_solar_events(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate solar astronomical events."""
        # Placeholder implementation
        return []

    def _identify_lunar_cycles(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify lunar cycle correlations."""
        # Placeholder implementation
        return []

    def _correlate_lunar_events(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate lunar astronomical events."""
        # Placeholder implementation
        return []

    def _identify_stellar_alignments(self, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify stellar alignment correlations."""
        # Placeholder implementation
        return []

    async def _identify_cultural_patterns(self, sites: List[Dict[str, Any]], cultural_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify cultural astronomical patterns."""
        # Placeholder implementation
        return []

    async def _cross_reference_mythology(self, sites: List[Dict[str, Any]], cultural_context: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-reference with mythological texts."""
        # Placeholder implementation
        return {}

    def _analyze_knowledge_transmission(self, sites: List[Dict[str, Any]], cultural_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze astronomical knowledge transmission."""
        # Placeholder implementation
        return []

    # Database storage

    async def _store_archaeoastronomy_results(self, task_id: str, results: Dict[str, Any]) -> None:
        """Store archaeoastronomy analysis results in database."""
        try:
            # Create protocol result record
            protocol_result = {
                "protocol_type": "archaeoastronomy_celestial_alignment",
                "task_id": task_id,
                "agent_id": self.agent_id,
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "analysis_type": results.get("analysis_type"),
                    "sites_analyzed": results.get("sites_analyzed", 0),
                    "significant_findings": results.get("significant_count", 0)
                }
            }

            # Store using pattern storage
            result_id = await self.pattern_storage.store_protocol_result(protocol_result)

            self.logger.info(f"Stored archaeoastronomy results with ID: {result_id}")

        except Exception as e:
            self.logger.error(f"Failed to store archaeoastronomy results: {e}")

    # Message handlers

    async def _handle_archaeoastronomy_request(self, message: AgentMessage) -> None:
        """Handle archaeoastronomy analysis requests."""
        try:
            request_data = message.payload
            analysis_type = request_data.get("analysis_type", "celestial_alignment")
            sites = request_data.get("sites", [])

            # Perform analysis based on type
            if analysis_type == "celestial_alignment":
                result = await self.analyze_celestial_alignments(sites)
            elif analysis_type == "astronomical_correlation":
                result = await self.analyze_astronomical_correlations(sites)
            elif analysis_type == "cultural_astronomy":
                result = await self.analyze_cultural_astronomy(sites, request_data.get("cultural_context", {}))
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="archaeoastronomy_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling archaeoastronomy request: {e}")

    async def _handle_alignment_request(self, message: AgentMessage) -> None:
        """Handle celestial alignment requests."""
        try:
            request_data = message.payload
            sites = request_data.get("sites", [])
            alignment_types = request_data.get("alignment_types", ["solstice", "equinox"])

            result = await self.analyze_celestial_alignments(sites, alignment_types)

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="alignment_analysis_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling alignment request: {e}")

    async def _handle_correlation_request(self, message: AgentMessage) -> None:
        """Handle astronomical correlation requests."""
        try:
            request_data = message.payload
            sites = request_data.get("sites", [])
            correlation_type = request_data.get("correlation_type", "solar")

            result = await self.analyze_astronomical_correlations(sites, correlation_type)

            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="correlation_analysis_response",
                payload=result,
                correlation_id=message.correlation_id
            )

            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)

        except Exception as e:
            self.logger.error(f"Error handling correlation request: {e}")

    async def _process_archaeoastronomy_queue(self) -> None:
        """Process queued archaeoastronomy requests."""
        # Implementation for processing queued requests
        pass

    async def _perform_periodic_astronomical_analysis(self) -> None:
        """Perform periodic astronomical analysis on stored data."""
        try:
            # Fetch sacred sites from database
            sacred_sites = await self.pattern_storage.get_sacred_sites(limit=100)

            if sacred_sites:
                # Perform comprehensive astronomical analysis
                analysis_result = await self.analyze_celestial_alignments(sacred_sites)

                if analysis_result.get("significant_count", 0) > 0:
                    self.logger.info(f"Periodic analysis found {analysis_result['significant_count']} significant alignments")

                    # Publish results
                    await self.messaging.publish_discovery(analysis_result)

        except Exception as e:
            self.logger.error(f"Error in periodic astronomical analysis: {e}")

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect archaeoastronomy agent metrics."""
        base_metrics = await super().collect_metrics() or {}

        archaeoastronomy_metrics = {
            "analyses_performed": self.analyses_performed,
            "alignments_discovered": self.alignments_discovered,
            "correlations_found": self.correlations_found,
            "cultural_patterns_identified": self.cultural_patterns_identified,
            "astronomical_sites_known": len(self.astronomical_sites),
            "cultural_practices_loaded": len(self.cultural_astronomy_practices),
            "integration_status": {
                "text_processor": self.text_processor is not None,
                "validation_agent": self.validation_agent is not None,
                "pattern_discovery": self.pattern_discovery is not None
            }
        }

        return {**base_metrics, **archaeoastronomy_metrics}

    def _get_capabilities(self) -> List[str]:
        """Get archaeoastronomy agent capabilities."""
        return [
            "archaeoastronomy_agent",
            "celestial_alignment_analysis",
            "astronomical_correlation",
            "horizon_astronomy",
            "cultural_astronomy",
            "solstice_alignment",
            "equinox_alignment",
            "lunar_standstill",
            "solar_correlation",
            "lunar_correlation",
            "stellar_alignment",
            "mythological_astronomy_integration",
            "multi_layered_validation_integration"
        ]

    async def shutdown(self) -> None:
        """Shutdown archaeoastronomy agent."""
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

            self.logger.info("ArchaeoastronomyAgent shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during archaeoastronomy agent shutdown: {e}")


# Factory function
def create_archaeoastronomy_agent(agent_id: Optional[str] = None, **kwargs) -> ArchaeoastronomyAgent:
    """
    Factory function to create archaeoastronomy agents.

    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured archaeoastronomy agent
    """
    return ArchaeoastronomyAgent(agent_id=agent_id, **kwargs)


# Main entry point
async def main():
    """Main entry point for running the ArchaeoastronomyAgent."""
    import signal
    import sys

    # Create and configure agent
    agent = ArchaeoastronomyAgent()

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
        print(f"ArchaeoastronomyAgent failed: {e}")
        sys.exit(1)

    print("ArchaeoastronomyAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())