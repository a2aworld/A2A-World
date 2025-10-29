"""
A2A World Platform - Consensus Validation Agent

Enhanced validation agent with integrated peer-to-peer consensus capabilities.
Participates in distributed pattern validation processes using consensus protocols
and contributes to collaborative pattern significance assessment.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import json

from agents.core.messaging import AgentMessage, NATSClient
from agents.validation.enhanced_validation_agent import EnhancedValidationAgent
from agents.consensus.voting_mechanisms import Vote, VoteType
from agents.consensus.reputation_system import ValidationOutcome, get_reputation_system
from agents.consensus.consensus_coordinator import ConsensusValidationRequest
from agents.validation.statistical_validation import StatisticalResult


@dataclass
class ConsensusParticipation:
    """Record of participation in a consensus validation."""
    consensus_id: str
    pattern_id: str
    agent_vote: VoteType
    confidence: float
    statistical_evidence: Dict[str, Any]
    consensus_decision: Optional[VoteType] = None
    peer_votes: List[Dict[str, Any]] = None
    participation_timestamp: str = None
    completion_timestamp: Optional[str] = None
    reputation_impact: float = 0.0
    
    def __post_init__(self):
        if self.participation_timestamp is None:
            self.participation_timestamp = datetime.utcnow().isoformat()
        if self.peer_votes is None:
            self.peer_votes = []


class ConsensusValidationAgent(EnhancedValidationAgent):
    """
    Enhanced validation agent with consensus participation capabilities.
    
    Extends the statistical validation framework to participate in distributed
    consensus protocols for collaborative pattern validation and reputation management.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None
    ):
        """
        Initialize consensus validation agent.
        
        Args:
            agent_id: Unique agent identifier
            config: Configuration parameters
            config_file: Configuration file path
        """
        super().__init__(agent_id, config, config_file)
        
        # Override agent type
        self.agent_type = "consensus_validation_agent"
        
        # Consensus-specific state
        self.consensus_participations: Dict[str, ConsensusParticipation] = {}
        self.active_consensus_requests: Dict[str, Dict[str, Any]] = {}
        self.peer_agents: Set[str] = set()
        self.consensus_coordinators: Set[str] = set()
        
        # Reputation system integration
        self.reputation_system = get_reputation_system(self.nats_client)
        self.reputation_tracker = None
        
        # Consensus performance metrics
        self.consensus_metrics = {
            'total_consensus_participations': 0,
            'votes_cast': 0,
            'consensus_agreements': 0,
            'reputation_score': 0.5,
            'average_confidence': 0.0,
            'peer_agreement_rate': 0.0,
            'response_time_avg': 0.0
        }
        
        self.logger = logging.getLogger(f"consensus_validation_agent.{self.agent_id}")
        self.logger.info(f"Initialized consensus validation agent {self.agent_id}")
    
    async def agent_initialize(self) -> None:
        """Initialize consensus validation agent."""
        try:
            # Call parent initialization
            await super().agent_initialize()
            
            # Initialize reputation system
            await self.reputation_system.start()
            
            # Register with reputation system
            self.reputation_tracker = self.reputation_system.register_agent(self.agent_id)
            
            # Start consensus-specific background tasks
            self.background_tasks.extend([
                asyncio.create_task(self._peer_discovery_loop()),
                asyncio.create_task(self._reputation_monitoring_loop()),
                asyncio.create_task(self._consensus_health_monitoring())
            ])
            
            # Announce availability for consensus
            await self._announce_consensus_availability()
            
            self.logger.info("Consensus validation agent initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize consensus validation agent: {e}")
            raise
    
    async def agent_cleanup(self) -> None:
        """Cleanup consensus validation agent resources."""
        try:
            # Stop reputation system
            if self.reputation_system:
                await self.reputation_system.stop()
            
            # Call parent cleanup
            await super().agent_cleanup()
            
            self.logger.info("Consensus validation agent cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during consensus validation agent cleanup: {e}")
    
    async def setup_subscriptions(self) -> None:
        """Setup consensus-specific message subscriptions."""
        await super().setup_subscriptions()
        
        if not self.messaging:
            return
        
        try:
            # Subscribe to consensus vote requests
            vote_request_sub_id = await self.nats_client.subscribe(
                "consensus.vote_request",
                self._handle_consensus_vote_request,
                queue_group=f"consensus-validation-agents-{self.agent_id}"
            )
            self.subscription_ids.append(vote_request_sub_id)
            
            # Subscribe to peer agent announcements
            peer_announce_sub_id = await self.nats_client.subscribe(
                "consensus.agents.announce",
                self._handle_peer_agent_announcement
            )
            self.subscription_ids.append(peer_announce_sub_id)
            
            # Subscribe to consensus coordinator announcements
            coordinator_announce_sub_id = await self.nats_client.subscribe(
                "consensus.coordinator.announce",
                self._handle_coordinator_announcement
            )
            self.subscription_ids.append(coordinator_announce_sub_id)
            
            # Subscribe to consensus results for reputation updates
            result_sub_id = await self.nats_client.subscribe(
                "consensus.results.broadcast",
                self._handle_consensus_result_broadcast
            )
            self.subscription_ids.append(result_sub_id)
            
            self.logger.info("Set up consensus validation agent subscriptions")
            
        except Exception as e:
            self.logger.error(f"Failed to setup consensus subscriptions: {e}")
    
    async def validate_pattern_with_consensus(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any],
        consensus_enabled: bool = True,
        voting_mechanism: str = "adaptive",
        min_participants: int = 3,
        timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """
        Validate pattern using both statistical analysis and consensus.
        
        Args:
            pattern_id: Pattern identifier
            pattern_data: Pattern data for validation
            consensus_enabled: Whether to use consensus validation
            voting_mechanism: Voting mechanism to use
            min_participants: Minimum consensus participants
            timeout_seconds: Consensus timeout
            
        Returns:
            Combined statistical and consensus validation results
        """
        try:
            self.logger.info(f"Starting consensus-enhanced validation for pattern {pattern_id}")
            start_time = time.time()
            
            # First, perform statistical validation
            statistical_results = await self.validate_pattern_enhanced(
                pattern_id=pattern_id,
                pattern_data=pattern_data,
                validation_methods=["full_statistical_suite"],
                store_results=False  # We'll store combined results
            )
            
            if not consensus_enabled:
                return statistical_results
            
            # Request consensus validation if statistical analysis is complete
            consensus_results = None
            if statistical_results.get("statistical_results"):
                consensus_request = ConsensusValidationRequest(
                    request_id=str(uuid.uuid4()),
                    pattern_id=pattern_id,
                    pattern_data=pattern_data,
                    statistical_results=statistical_results["statistical_results"],
                    requester_id=self.agent_id,
                    voting_mechanism=voting_mechanism,
                    min_participants=min_participants,
                    timeout_seconds=timeout_seconds
                )
                
                consensus_results = await self._request_consensus_validation(consensus_request)
            
            # Combine statistical and consensus results
            combined_results = self._combine_statistical_and_consensus_results(
                statistical_results, consensus_results
            )
            
            combined_results["execution_time_seconds"] = time.time() - start_time
            combined_results["consensus_enabled"] = consensus_enabled
            
            # Store combined results
            if combined_results.get("statistical_results"):
                stored_validation_id = await self._store_enhanced_validation_results(
                    pattern_id, combined_results
                )
                combined_results["stored_validation_id"] = stored_validation_id
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Consensus-enhanced validation failed for pattern {pattern_id}: {e}")
            return {
                "pattern_id": pattern_id,
                "error": str(e),
                "validation_timestamp": datetime.utcnow().isoformat(),
                "consensus_enabled": consensus_enabled,
                "statistical_results": [],
                "consensus_results": None
            }
    
    async def participate_in_consensus(
        self,
        consensus_request: Dict[str, Any]
    ) -> Optional[Vote]:
        """
        Participate in a consensus validation request.
        
        Args:
            consensus_request: Consensus validation request data
            
        Returns:
            Vote cast by this agent or None if unable to participate
        """
        try:
            request_id = consensus_request["request_id"]
            pattern_id = consensus_request["pattern_id"]
            pattern_data = consensus_request["pattern_data"]
            timeout_seconds = consensus_request.get("timeout_seconds", 30)
            
            self.logger.info(f"Participating in consensus for pattern {pattern_id}")
            start_time = time.time()
            
            # Perform statistical validation for this pattern
            validation_results = await self.validate_pattern_enhanced(
                pattern_id=pattern_id,
                pattern_data=pattern_data,
                validation_methods=["comprehensive_morans_i", "csr_testing", "pattern_significance"],
                store_results=False
            )
            
            if not validation_results.get("statistical_results"):
                self.logger.warning(f"No statistical results for consensus participation: {pattern_id}")
                return None
            
            # Determine vote based on statistical significance
            vote_decision, confidence = self._determine_vote_from_statistics(validation_results)
            
            # Create vote with reputation weighting
            reputation_score = self.reputation_tracker.get_reputation_score() if self.reputation_tracker else None
            voting_weight = self.reputation_system.get_voting_weight(self.agent_id) if self.reputation_system else 1.0
            
            vote = Vote(
                agent_id=self.agent_id,
                pattern_id=pattern_id,
                vote=vote_decision,
                confidence=confidence,
                statistical_evidence=self._extract_statistical_evidence(validation_results),
                reasoning=self._generate_vote_reasoning(validation_results, vote_decision),
                timestamp=datetime.utcnow().isoformat(),
                weight=voting_weight,
                reputation_score=reputation_score.overall_score if reputation_score else 0.5
            )
            
            # Record consensus participation
            participation = ConsensusParticipation(
                consensus_id=request_id,
                pattern_id=pattern_id,
                agent_vote=vote_decision,
                confidence=confidence,
                statistical_evidence=vote.statistical_evidence
            )
            self.consensus_participations[request_id] = participation
            
            # Update metrics
            self.consensus_metrics["votes_cast"] += 1
            self.consensus_metrics["total_consensus_participations"] += 1
            
            response_time = time.time() - start_time
            self._update_response_time_metric(response_time)
            
            self.logger.info(f"Cast vote for pattern {pattern_id}: {vote_decision.value} (confidence: {confidence:.3f})")
            
            return vote
            
        except Exception as e:
            self.logger.error(f"Error participating in consensus: {e}")
            return None
    
    def _determine_vote_from_statistics(self, validation_results: Dict[str, Any]) -> Tuple[VoteType, float]:
        """
        Determine vote and confidence from statistical validation results.
        
        Args:
            validation_results: Statistical validation results
            
        Returns:
            Tuple of (vote_type, confidence)
        """
        try:
            statistical_results = validation_results.get("statistical_results", [])
            significance_classification = validation_results.get("significance_classification", {})
            
            if not statistical_results:
                return VoteType.UNCERTAIN, 0.0
            
            # Get overall classification
            overall_classification = significance_classification.get("overall_classification", "not_significant")
            reliability_score = significance_classification.get("reliability_score", 0.0)
            
            # Map classification to vote
            if overall_classification in ["very_high", "high"]:
                vote = VoteType.SIGNIFICANT
                confidence = min(0.95, 0.6 + reliability_score * 0.4)  # 0.6 to 0.95 range
            elif overall_classification == "moderate":
                # Use additional criteria for moderate cases
                significant_tests = sum(1 for result in statistical_results if hasattr(result, 'significant') and result.significant)
                total_tests = len(statistical_results)
                
                if significant_tests >= total_tests * 0.6:  # 60% of tests significant
                    vote = VoteType.SIGNIFICANT
                    confidence = min(0.8, 0.5 + reliability_score * 0.3)
                else:
                    vote = VoteType.NOT_SIGNIFICANT
                    confidence = min(0.7, 0.4 + (1 - reliability_score) * 0.3)
            elif overall_classification == "low":
                vote = VoteType.NOT_SIGNIFICANT
                confidence = min(0.8, 0.5 + (1 - reliability_score) * 0.3)
            else:  # not_significant
                vote = VoteType.NOT_SIGNIFICANT
                confidence = min(0.9, 0.6 + (1 - reliability_score) * 0.3)
            
            # Adjust confidence based on sample size and test quality
            enhanced_metrics = validation_results.get("enhanced_metrics", {})
            sample_size = enhanced_metrics.get("sample_size", 0)
            
            if sample_size < 10:
                confidence *= 0.7  # Reduce confidence for small samples
            elif sample_size > 100:
                confidence = min(1.0, confidence * 1.1)  # Boost confidence for large samples
            
            return vote, max(0.1, min(0.99, confidence))  # Ensure confidence is in valid range
            
        except Exception as e:
            self.logger.error(f"Error determining vote from statistics: {e}")
            return VoteType.UNCERTAIN, 0.0
    
    def _extract_statistical_evidence(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key statistical evidence for vote justification."""
        try:
            evidence = {}
            
            statistical_results = validation_results.get("statistical_results", [])
            significance_classification = validation_results.get("significance_classification", {})
            enhanced_metrics = validation_results.get("enhanced_metrics", {})
            
            # Extract key statistics
            evidence["total_tests"] = len(statistical_results)
            evidence["significant_tests"] = sum(1 for result in statistical_results if hasattr(result, 'significant') and result.significant)
            evidence["overall_classification"] = significance_classification.get("overall_classification", "unknown")
            evidence["reliability_score"] = significance_classification.get("reliability_score", 0.0)
            evidence["sample_size"] = enhanced_metrics.get("sample_size", 0)
            
            # Extract specific test results
            for result in statistical_results:
                if hasattr(result, 'statistic_name') and hasattr(result, 'p_value'):
                    evidence[f"{result.statistic_name}_p_value"] = result.p_value
                    evidence[f"{result.statistic_name}_significant"] = getattr(result, 'significant', False)
            
            # Extract p-value summary
            p_value_summary = significance_classification.get("p_value_summary", {})
            if p_value_summary:
                evidence["min_p_value"] = p_value_summary.get("minimum", 1.0)
                evidence["mean_p_value"] = p_value_summary.get("mean", 1.0)
            
            return evidence
            
        except Exception as e:
            self.logger.error(f"Error extracting statistical evidence: {e}")
            return {}
    
    def _generate_vote_reasoning(self, validation_results: Dict[str, Any], vote: VoteType) -> str:
        """Generate human-readable reasoning for the vote."""
        try:
            statistical_results = validation_results.get("statistical_results", [])
            significance_classification = validation_results.get("significance_classification", {})
            
            classification = significance_classification.get("overall_classification", "unknown")
            reliability = significance_classification.get("reliability_score", 0.0)
            total_tests = len(statistical_results)
            significant_tests = sum(1 for result in statistical_results if hasattr(result, 'significant') and result.significant)
            
            reasoning = f"Based on {total_tests} statistical tests with {significant_tests} showing significance. "
            reasoning += f"Overall classification: {classification} (reliability: {reliability:.3f}). "
            
            if vote == VoteType.SIGNIFICANT:
                reasoning += "Statistical evidence supports pattern significance."
            elif vote == VoteType.NOT_SIGNIFICANT:
                reasoning += "Statistical evidence does not support pattern significance."
            else:
                reasoning += "Statistical evidence is inconclusive."
            
            # Add specific test mentions
            significant_test_names = [
                result.statistic_name for result in statistical_results 
                if hasattr(result, 'statistic_name') and hasattr(result, 'significant') and result.significant
            ]
            if significant_test_names:
                reasoning += f" Significant tests: {', '.join(significant_test_names[:3])}."
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating vote reasoning: {e}")
            return f"Vote: {vote.value} based on statistical analysis (error in reasoning generation)"
    
    async def _request_consensus_validation(self, request: ConsensusValidationRequest) -> Optional[Dict[str, Any]]:
        """Request consensus validation from coordinators."""
        try:
            if not self.consensus_coordinators:
                self.logger.warning("No consensus coordinators available")
                return None
            
            # Send request to first available coordinator
            coordinator_id = next(iter(self.consensus_coordinators))
            
            message = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=coordinator_id,
                message_type="consensus_validation_request",
                payload=request.to_dict()
            )
            
            # Send request and wait for response
            response = await self.nats_client.request(
                "consensus.validation.request",
                message,
                timeout=request.timeout_seconds + 10  # Extra time for coordinator processing
            )
            
            return response.payload if response else None
            
        except Exception as e:
            self.logger.error(f"Error requesting consensus validation: {e}")
            return None
    
    def _combine_statistical_and_consensus_results(
        self,
        statistical_results: Dict[str, Any],
        consensus_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine statistical and consensus validation results."""
        try:
            # Start with statistical results as base
            combined = statistical_results.copy()
            
            # Add consensus information
            combined["consensus_results"] = consensus_results
            combined["validation_type"] = "consensus_enhanced_statistical"
            
            if consensus_results:
                # Extract consensus decision and confidence
                consensus_decision = consensus_results.get("decision")
                consensus_confidence = consensus_results.get("confidence", 0.0)
                consensus_achieved = consensus_results.get("status") == "completed"
                
                # Update significance classification with consensus input
                if "significance_classification" in combined:
                    sig_class = combined["significance_classification"]
                    
                    # Create consensus-enhanced classification
                    if consensus_achieved and consensus_decision:
                        # Weight consensus and statistical results
                        statistical_reliability = sig_class.get("reliability_score", 0.0)
                        
                        # Combine confidences (weighted average)
                        combined_confidence = (statistical_reliability + consensus_confidence) / 2
                        
                        # Update classification based on consensus agreement
                        statistical_classification = sig_class.get("overall_classification", "not_significant")
                        
                        if consensus_decision == "significant":
                            if statistical_classification in ["very_high", "high", "moderate"]:
                                enhanced_classification = "very_high"  # Consensus confirms statistics
                            else:
                                enhanced_classification = "moderate"  # Consensus overrides weak statistics
                        elif consensus_decision == "not_significant":
                            if statistical_classification in ["low", "not_significant"]:
                                enhanced_classification = "not_significant"  # Consensus confirms
                            else:
                                enhanced_classification = "low"  # Consensus questions statistics
                        else:  # uncertain or other
                            enhanced_classification = statistical_classification  # Keep statistical
                        
                        sig_class["consensus_enhanced_classification"] = enhanced_classification
                        sig_class["consensus_enhanced_confidence"] = combined_confidence
                        sig_class["consensus_agreement"] = consensus_decision == statistical_classification
                
                # Add consensus metadata
                combined["consensus_metadata"] = {
                    "consensus_achieved": consensus_achieved,
                    "participating_agents": consensus_results.get("participating_agents", []),
                    "voting_result": consensus_results.get("voting_result"),
                    "execution_time": consensus_results.get("execution_time_seconds", 0.0)
                }
            else:
                # No consensus results available
                combined["consensus_metadata"] = {
                    "consensus_achieved": False,
                    "consensus_attempted": True,
                    "consensus_error": "Failed to obtain consensus results"
                }
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining statistical and consensus results: {e}")
            return statistical_results
    
    def _update_response_time_metric(self, response_time: float) -> None:
        """Update average response time metric."""
        try:
            current_avg = self.consensus_metrics["response_time_avg"]
            total_participations = self.consensus_metrics["total_consensus_participations"]
            
            if total_participations > 1:
                self.consensus_metrics["response_time_avg"] = (
                    (current_avg * (total_participations - 1) + response_time) / total_participations
                )
            else:
                self.consensus_metrics["response_time_avg"] = response_time
            
            # Record response time with reputation system
            if self.reputation_system:
                self.reputation_system.record_response_time(self.agent_id, response_time)
                
        except Exception as e:
            self.logger.error(f"Error updating response time metric: {e}")
    
    async def _handle_consensus_vote_request(self, message: AgentMessage) -> None:
        """Handle consensus vote request messages."""
        try:
            if message.sender_id == self.agent_id:
                return  # Ignore own messages
            
            consensus_request = message.payload
            
            # Participate in consensus
            vote = await self.participate_in_consensus(consensus_request)
            
            if vote:
                # Send vote response
                response = AgentMessage.create(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type="consensus_vote_response",
                    payload={
                        "vote": vote.to_dict(),
                        "request_id": consensus_request.get("request_id"),
                        "agent_id": self.agent_id
                    },
                    correlation_id=message.correlation_id
                )
                
                # Publish vote to consensus system
                await self.nats_client.publish("consensus.votes", response)
                
                self.logger.debug(f"Responded to consensus vote request for pattern {consensus_request.get('pattern_id')}")
            
        except Exception as e:
            self.logger.error(f"Error handling consensus vote request: {e}")
    
    async def _handle_peer_agent_announcement(self, message: AgentMessage) -> None:
        """Handle peer agent announcements."""
        try:
            payload = message.payload
            agent_id = payload.get("agent_id", message.sender_id)
            capabilities = payload.get("capabilities", [])
            
            if agent_id and agent_id != self.agent_id:
                # Check if agent has validation capabilities
                if any(cap in capabilities for cap in ["validation", "consensus_validation", "statistical_validation"]):
                    self.peer_agents.add(agent_id)
                    self.logger.debug(f"Discovered peer validation agent: {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling peer agent announcement: {e}")
    
    async def _handle_coordinator_announcement(self, message: AgentMessage) -> None:
        """Handle consensus coordinator announcements."""
        try:
            payload = message.payload
            coordinator_id = payload.get("coordinator_id", message.sender_id)
            protocols = payload.get("protocols_supported", [])
            
            if coordinator_id and coordinator_id != self.agent_id:
                self.consensus_coordinators.add(coordinator_id)
                self.logger.debug(f"Discovered consensus coordinator: {coordinator_id} (protocols: {protocols})")
            
        except Exception as e:
            self.logger.error(f"Error handling coordinator announcement: {e}")
    
    async def _handle_consensus_result_broadcast(self, message: AgentMessage) -> None:
        """Handle consensus result broadcasts for reputation updates."""
        try:
            payload = message.payload
            request_id = payload.get("request_id")
            consensus_decision = payload.get("decision")
            participating_agents = payload.get("participating_agents", [])
            
            # Update participation record if we participated
            if request_id in self.consensus_participations and self.agent_id in participating_agents:
                participation = self.consensus_participations[request_id]
                participation.consensus_decision = VoteType(consensus_decision) if consensus_decision else None
                participation.completion_timestamp = datetime.utcnow().isoformat()
                
                # Check if our vote agreed with consensus
                if participation.consensus_decision == participation.agent_vote:
                    self.consensus_metrics["consensus_agreements"] += 1
                    participation.reputation_impact = 0.1  # Small positive impact
                else:
                    participation.reputation_impact = -0.05  # Small negative impact
                
                self.logger.debug(f"Updated consensus participation record for {request_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling consensus result broadcast: {e}")
    
    async def _announce_consensus_availability(self) -> None:
        """Announce availability for consensus participation."""
        try:
            announcement = AgentMessage.create(
                sender_id=self.agent_id,
                message_type="consensus_agent_announcement",
                payload={
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "capabilities": [
                        "statistical_validation",
                        "consensus_participation",
                        "pattern_validation",
                        "morans_i_analysis",
                        "monte_carlo_testing",
                        "bootstrap_validation"
                    ],
                    "consensus_protocols_supported": ["bft", "raft", "voting"],
                    "statistical_methods": [
                        "comprehensive_morans_i",
                        "monte_carlo_validation",
                        "csr_testing",
                        "hotspot_analysis",
                        "pattern_significance"
                    ],
                    "reputation_score": self.consensus_metrics["reputation_score"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.nats_client.publish("consensus.agents.announce", announcement)
            self.logger.info(f"Announced consensus availability for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error announcing consensus availability: {e}")
    
    async def _peer_discovery_loop(self) -> None:
        """Background task for peer discovery."""
        while not self.shutdown_event.is_set():
            try:
                # Periodically announce availability
                await self._announce_consensus_availability()
                
                await asyncio.sleep(60)  # Announce every minute
                
            except Exception as e:
                self.logger.error(f"Error in peer discovery loop: {e}")
                await asyncio.sleep(30)
    
    async def _reputation_monitoring_loop(self) -> None:
        """Background task for reputation monitoring."""
        while not self.shutdown_event.is_set():
            try:
                # Update reputation metrics
                if self.reputation_tracker:
                    reputation_score = self.reputation_tracker.get_reputation_score()
                    self.consensus_metrics["reputation_score"] = reputation_score.overall_score
                    
                    # Calculate consensus-specific metrics
                    total_participations = self.consensus_metrics["total_consensus_participations"]
                    agreements = self.consensus_metrics["consensus_agreements"]
                    
                    if total_participations > 0:
                        self.consensus_metrics["peer_agreement_rate"] = agreements / total_participations
                    
                    # Record availability
                    self.reputation_system.record_availability(self.agent_id, True)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in reputation monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _consensus_health_monitoring(self) -> None:
        """Background task for consensus health monitoring."""
        while not self.shutdown_event.is_set():
            try:
                # Monitor peer connectivity
                peer_count = len(self.peer_agents)
                coordinator_count = len(self.consensus_coordinators)
                
                if peer_count < 2:
                    self.logger.warning("Few peer agents available for consensus")
                
                if coordinator_count == 0:
                    self.logger.warning("No consensus coordinators available")
                
                # Clean up old participation records
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_participations = [
                    pid for pid, participation in self.consensus_participations.items()
                    if datetime.fromisoformat(participation.participation_timestamp) < cutoff_time
                ]
                
                for pid in old_participations:
                    del self.consensus_participations[pid]
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in consensus health monitoring: {e}")
                await asyncio.sleep(120)
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect agent metrics including consensus-specific metrics."""
        base_metrics = await super().collect_metrics() or {}
        
        # Add consensus-specific metrics
        consensus_metrics = self.consensus_metrics.copy()
        consensus_metrics.update({
            "peer_agents_discovered": len(self.peer_agents),
            "consensus_coordinators_available": len(self.consensus_coordinators),
            "active_consensus_participations": len(self.consensus_participations),
            "recent_participations": len([
                p for p in self.consensus_participations.values()
                if datetime.fromisoformat(p.participation_timestamp) > datetime.utcnow() - timedelta(hours=24)
            ])
        })
        
        return {**base_metrics, "consensus_metrics": consensus_metrics}
    
    def _get_capabilities(self) -> List[str]:
        """Get consensus validation agent capabilities."""
        base_capabilities = [
            "consensus_validation_agent",
            "statistical_validation",
            "consensus_participation",
            "peer_to_peer_validation",
            "reputation_tracking",
            "distributed_decision_making"
        ]
        
        # Add statistical capabilities from parent
        base_capabilities.extend([
            "morans_i_analysis",
            "monte_carlo_testing", 
            "bootstrap_validation",
            "csr_testing",
            "hotspot_analysis",
            "pattern_significance_classification"
        ])
        
        return base_capabilities


# Factory function for creating consensus validation agents
def create_consensus_validation_agent(agent_id: Optional[str] = None, **kwargs) -> ConsensusValidationAgent:
    """
    Factory function to create consensus validation agents.
    
    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured consensus validation agent
    """
    return ConsensusValidationAgent(agent_id=agent_id, **kwargs)


# Main entry point for running the consensus validation agent
async def main():
    """Main entry point for running the Consensus ValidationAgent."""
    import signal
    import sys
    
    # Create and configure agent
    agent = ConsensusValidationAgent()
    
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
        print(f"Consensus ValidationAgent failed: {e}")
        sys.exit(1)
    
    print("Consensus ValidationAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())