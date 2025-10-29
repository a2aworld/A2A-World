"""
A2A World Platform - Consensus Coordinator

Main orchestration component for peer-to-peer consensus-based pattern validation.
Coordinates between different consensus algorithms, voting mechanisms, and reputation systems
to provide unified consensus decision-making for pattern significance assessment.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json

from agents.core.messaging import AgentMessage, NATSClient
from agents.core.base_agent import BaseAgent
from .bft_consensus import ByzantineFaultTolerantConsensus, ConsensusRequest, ConsensusVote
from .raft_consensus import RaftConsensus, ValidationEntry
from .voting_mechanisms import (
    VotingMechanism, Vote, VoteType, VotingResult,
    create_voting_mechanism
)
from .reputation_system import (
    ReputationSystem, ValidationOutcome, get_reputation_system
)


class ConsensusProtocol(Enum):
    """Available consensus protocols."""
    BFT = "byzantine_fault_tolerant"
    RAFT = "raft"
    VOTING_ONLY = "voting_only"
    ADAPTIVE = "adaptive"


class ConsensusStatus(Enum):
    """Consensus request status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ConsensusValidationRequest:
    """Request for consensus-based pattern validation."""
    request_id: str
    pattern_id: str
    pattern_data: Dict[str, Any]
    statistical_results: List[Dict[str, Any]]
    requester_id: str
    validation_methods: List[str] = field(default_factory=list)
    consensus_protocol: ConsensusProtocol = ConsensusProtocol.ADAPTIVE
    voting_mechanism: str = "adaptive"
    timeout_seconds: int = 60
    min_participants: int = 3
    require_statistical_evidence: bool = True
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'pattern_id': self.pattern_id,
            'pattern_data': self.pattern_data,
            'statistical_results': self.statistical_results,
            'requester_id': self.requester_id,
            'validation_methods': self.validation_methods,
            'consensus_protocol': self.consensus_protocol.value,
            'voting_mechanism': self.voting_mechanism,
            'timeout_seconds': self.timeout_seconds,
            'min_participants': self.min_participants,
            'require_statistical_evidence': self.require_statistical_evidence,
            'timestamp': self.timestamp
        }


@dataclass
class ConsensusValidationResult:
    """Result of consensus-based pattern validation."""
    request_id: str
    pattern_id: str
    status: ConsensusStatus
    decision: Optional[VoteType] = None
    confidence: float = 0.0
    consensus_protocol_used: Optional[ConsensusProtocol] = None
    voting_result: Optional[VotingResult] = None
    participating_agents: List[str] = field(default_factory=list)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    reputation_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'pattern_id': self.pattern_id,
            'status': self.status.value,
            'decision': self.decision.value if self.decision else None,
            'confidence': self.confidence,
            'consensus_protocol_used': self.consensus_protocol_used.value if self.consensus_protocol_used else None,
            'voting_result': self.voting_result.to_dict() if self.voting_result else None,
            'participating_agents': self.participating_agents,
            'statistical_summary': self.statistical_summary,
            'reputation_adjustments': self.reputation_adjustments,
            'execution_time_seconds': self.execution_time_seconds,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class ConsensusCoordinator(BaseAgent):
    """
    Main consensus coordinator that orchestrates peer-to-peer pattern validation.
    
    Integrates multiple consensus protocols, voting mechanisms, and reputation systems
    to provide robust distributed decision-making for pattern significance assessment.
    """
    
    def __init__(
        self,
        coordinator_id: Optional[str] = None,
        nats_client: Optional[NATSClient] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize consensus coordinator.
        
        Args:
            coordinator_id: Unique coordinator identifier
            nats_client: NATS client for communication
            config: Configuration parameters
        """
        super().__init__(
            agent_id=coordinator_id or f"consensus_coordinator_{uuid.uuid4().hex[:8]}",
            agent_type="consensus_coordinator"
        )
        
        self.nats_client = nats_client or self.nats_client
        self.config_params = config or {}
        
        # Consensus protocols
        self.bft_consensus: Optional[ByzantineFaultTolerantConsensus] = None
        self.raft_consensus: Optional[RaftConsensus] = None
        
        # Voting mechanisms
        self.voting_mechanisms: Dict[str, VotingMechanism] = {}
        
        # Reputation system
        self.reputation_system: ReputationSystem = get_reputation_system(self.nats_client)
        
        # Active consensus requests
        self.active_requests: Dict[str, ConsensusValidationRequest] = {}
        self.request_status: Dict[str, ConsensusStatus] = {}
        self.request_results: Dict[str, ConsensusValidationResult] = {}
        self.request_timeouts: Dict[str, datetime] = {}
        
        # Agent network state
        self.available_agents: Set[str] = set()
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.network_topology: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.consensus_metrics: Dict[str, Any] = {
            'total_requests': 0,
            'successful_consensus': 0,
            'failed_consensus': 0,
            'average_response_time': 0.0,
            'protocol_usage': {},
            'agent_participation_rates': {}
        }
        
        self.logger = logging.getLogger(f"consensus_coordinator.{self.agent_id}")
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
    
    async def agent_initialize(self) -> None:
        """Initialize consensus coordinator components."""
        try:
            self.logger.info("Initializing consensus coordinator")
            
            # Initialize consensus protocols
            await self._initialize_consensus_protocols()
            
            # Initialize voting mechanisms
            self._initialize_voting_mechanisms()
            
            # Start reputation system
            await self.reputation_system.start()
            
            # Start background tasks
            self.background_tasks.extend([
                asyncio.create_task(self._agent_discovery_loop()),
                asyncio.create_task(self._request_timeout_monitor()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._health_monitoring_loop())
            ])
            
            self.logger.info("Consensus coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize consensus coordinator: {e}")
            raise
    
    async def agent_cleanup(self) -> None:
        """Cleanup consensus coordinator resources."""
        try:
            self.logger.info("Cleaning up consensus coordinator")
            
            # Set shutdown event
            self.shutdown_event.set()
            
            # Stop consensus protocols
            if self.bft_consensus:
                await self.bft_consensus.stop()
            if self.raft_consensus:
                await self.raft_consensus.stop()
            
            # Stop reputation system
            await self.reputation_system.stop()
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.logger.info("Consensus coordinator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during consensus coordinator cleanup: {e}")
    
    async def setup_subscriptions(self) -> None:
        """Setup NATS message subscriptions."""
        if not self.messaging:
            return
        
        try:
            # Subscribe to consensus validation requests
            await self.nats_client.subscribe(
                "consensus.validation.request",
                self._handle_validation_request,
                queue_group="consensus-coordinators"
            )
            
            # Subscribe to agent votes
            await self.nats_client.subscribe(
                "consensus.votes",
                self._handle_vote_message
            )
            
            # Subscribe to agent announcements
            await self.nats_client.subscribe(
                "consensus.agents.announce",
                self._handle_agent_announcement
            )
            
            # Subscribe to status requests
            await self.nats_client.subscribe(
                f"consensus.{self.agent_id}.status",
                self._handle_status_request
            )
            
            self.logger.info("Set up consensus coordinator subscriptions")
            
        except Exception as e:
            self.logger.error(f"Failed to setup subscriptions: {e}")
    
    async def process(self) -> None:
        """Main processing loop."""
        try:
            # Process any completed requests
            await self._process_completed_requests()
            
            # Update network topology
            await self._update_network_topology()
            
            # Clean up old requests (every 100 iterations)
            if self.processed_tasks % 100 == 0:
                await self._cleanup_old_requests()
            
        except Exception as e:
            self.logger.error(f"Error in consensus coordinator processing: {e}")
    
    async def request_consensus_validation(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any],
        statistical_results: List[Dict[str, Any]],
        **kwargs
    ) -> ConsensusValidationResult:
        """
        Request consensus-based pattern validation.
        
        Args:
            pattern_id: Pattern identifier
            pattern_data: Pattern data for validation
            statistical_results: Statistical validation results
            **kwargs: Additional validation parameters
            
        Returns:
            Consensus validation result
        """
        try:
            # Create validation request
            request = ConsensusValidationRequest(
                request_id=str(uuid.uuid4()),
                pattern_id=pattern_id,
                pattern_data=pattern_data,
                statistical_results=statistical_results,
                requester_id=self.agent_id,
                **kwargs
            )
            
            self.logger.info(f"Starting consensus validation for pattern {pattern_id}")
            
            # Execute consensus validation
            result = await self._execute_consensus_validation(request)
            
            # Update metrics
            self._update_consensus_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Consensus validation failed for pattern {pattern_id}: {e}")
            return ConsensusValidationResult(
                request_id=str(uuid.uuid4()),
                pattern_id=pattern_id,
                status=ConsensusStatus.FAILED,
                error_message=str(e)
            )
    
    async def _execute_consensus_validation(
        self,
        request: ConsensusValidationRequest
    ) -> ConsensusValidationResult:
        """Execute the consensus validation process."""
        start_time = time.time()
        
        try:
            # Store active request
            self.active_requests[request.request_id] = request
            self.request_status[request.request_id] = ConsensusStatus.PENDING
            self.request_timeouts[request.request_id] = datetime.utcnow() + timedelta(seconds=request.timeout_seconds)
            
            # Check if we have enough agents
            if len(self.available_agents) < request.min_participants:
                return ConsensusValidationResult(
                    request_id=request.request_id,
                    pattern_id=request.pattern_id,
                    status=ConsensusStatus.FAILED,
                    error_message=f"Insufficient agents: {len(self.available_agents)} < {request.min_participants}",
                    execution_time_seconds=time.time() - start_time
                )
            
            self.request_status[request.request_id] = ConsensusStatus.IN_PROGRESS
            
            # Select consensus protocol
            protocol = self._select_consensus_protocol(request)
            
            # Execute consensus based on protocol
            if protocol == ConsensusProtocol.BFT:
                result = await self._execute_bft_consensus(request)
            elif protocol == ConsensusProtocol.RAFT:
                result = await self._execute_raft_consensus(request)
            elif protocol == ConsensusProtocol.VOTING_ONLY:
                result = await self._execute_voting_only_consensus(request)
            else:
                result = await self._execute_adaptive_consensus(request)
            
            result.consensus_protocol_used = protocol
            result.execution_time_seconds = time.time() - start_time
            
            # Update reputation based on results
            if result.voting_result:
                await self._update_agent_reputations(request, result.voting_result)
            
            # Store result
            self.request_results[request.request_id] = result
            self.request_status[request.request_id] = result.status
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing consensus validation: {e}")
            return ConsensusValidationResult(
                request_id=request.request_id,
                pattern_id=request.pattern_id,
                status=ConsensusStatus.FAILED,
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
        
        finally:
            # Cleanup
            self.active_requests.pop(request.request_id, None)
            self.request_timeouts.pop(request.request_id, None)
    
    def _select_consensus_protocol(self, request: ConsensusValidationRequest) -> ConsensusProtocol:
        """Select appropriate consensus protocol based on request and network state."""
        if request.consensus_protocol != ConsensusProtocol.ADAPTIVE:
            return request.consensus_protocol
        
        # Adaptive protocol selection logic
        num_agents = len(self.available_agents)
        
        # Use BFT for critical patterns with enough agents
        if num_agents >= 7 and request.require_statistical_evidence:
            return ConsensusProtocol.BFT
        
        # Use RAFT for moderate-sized networks
        elif num_agents >= 5:
            return ConsensusProtocol.RAFT
        
        # Use voting-only for small networks
        else:
            return ConsensusProtocol.VOTING_ONLY
    
    async def _execute_bft_consensus(self, request: ConsensusValidationRequest) -> ConsensusValidationResult:
        """Execute BFT consensus protocol."""
        try:
            if not self.bft_consensus:
                raise RuntimeError("BFT consensus not initialized")
            
            # Create consensus request
            consensus_request = ConsensusRequest(
                request_id=request.request_id,
                pattern_id=request.pattern_id,
                pattern_data=request.pattern_data,
                statistical_results=request.statistical_results,
                requester_id=request.requester_id,
                timeout_seconds=request.timeout_seconds
            )
            
            # Execute BFT consensus
            bft_result = await self.bft_consensus.request_consensus(consensus_request)
            
            if bft_result and bft_result.get('consensus_achieved'):
                decision_str = bft_result.get('decision', 'uncertain')
                decision = VoteType(decision_str) if decision_str in [v.value for v in VoteType] else VoteType.UNCERTAIN
                
                return ConsensusValidationResult(
                    request_id=request.request_id,
                    pattern_id=request.pattern_id,
                    status=ConsensusStatus.COMPLETED,
                    decision=decision,
                    confidence=bft_result.get('confidence', 0.0),
                    participating_agents=bft_result.get('participating_nodes', []),
                    metadata={'bft_result': bft_result}
                )
            else:
                return ConsensusValidationResult(
                    request_id=request.request_id,
                    pattern_id=request.pattern_id,
                    status=ConsensusStatus.FAILED,
                    error_message="BFT consensus failed to reach agreement"
                )
            
        except Exception as e:
            self.logger.error(f"BFT consensus execution failed: {e}")
            return ConsensusValidationResult(
                request_id=request.request_id,
                pattern_id=request.pattern_id,
                status=ConsensusStatus.FAILED,
                error_message=f"BFT consensus error: {e}"
            )
    
    async def _execute_raft_consensus(self, request: ConsensusValidationRequest) -> ConsensusValidationResult:
        """Execute RAFT consensus protocol."""
        try:
            if not self.raft_consensus:
                raise RuntimeError("RAFT consensus not initialized")
            
            # Execute RAFT consensus
            raft_result = await self.raft_consensus.validate_pattern(
                pattern_id=request.pattern_id,
                pattern_data=request.pattern_data,
                statistical_results=request.statistical_results,
                timeout_seconds=request.timeout_seconds
            )
            
            if raft_result and raft_result.get('status') == 'completed':
                decision_str = raft_result.get('consensus_decision', 'uncertain')
                decision = VoteType(decision_str) if decision_str in [v.value for v in VoteType] else VoteType.UNCERTAIN
                
                return ConsensusValidationResult(
                    request_id=request.request_id,
                    pattern_id=request.pattern_id,
                    status=ConsensusStatus.COMPLETED,
                    decision=decision,
                    confidence=raft_result.get('confidence', 0.0),
                    participating_agents=raft_result.get('participating_nodes', []),
                    metadata={'raft_result': raft_result}
                )
            else:
                return ConsensusValidationResult(
                    request_id=request.request_id,
                    pattern_id=request.pattern_id,
                    status=ConsensusStatus.FAILED,
                    error_message="RAFT consensus failed"
                )
            
        except Exception as e:
            self.logger.error(f"RAFT consensus execution failed: {e}")
            return ConsensusValidationResult(
                request_id=request.request_id,
                pattern_id=request.pattern_id,
                status=ConsensusStatus.FAILED,
                error_message=f"RAFT consensus error: {e}"
            )
    
    async def _execute_voting_only_consensus(self, request: ConsensusValidationRequest) -> ConsensusValidationResult:
        """Execute voting-only consensus (no Byzantine fault tolerance)."""
        try:
            # Request votes from available agents
            votes = await self._collect_agent_votes(request)
            
            if not votes:
                return ConsensusValidationResult(
                    request_id=request.request_id,
                    pattern_id=request.pattern_id,
                    status=ConsensusStatus.FAILED,
                    error_message="No votes received from agents"
                )
            
            # Apply voting mechanism
            voting_mechanism = self.voting_mechanisms.get(
                request.voting_mechanism,
                self.voting_mechanisms['adaptive']
            )
            
            voting_result = voting_mechanism.compute_result(votes, request.pattern_id)
            
            return ConsensusValidationResult(
                request_id=request.request_id,
                pattern_id=request.pattern_id,
                status=ConsensusStatus.COMPLETED if voting_result.consensus_achieved else ConsensusStatus.FAILED,
                decision=voting_result.decision,
                confidence=voting_result.confidence,
                voting_result=voting_result,
                participating_agents=voting_result.participating_agents
            )
            
        except Exception as e:
            self.logger.error(f"Voting-only consensus execution failed: {e}")
            return ConsensusValidationResult(
                request_id=request.request_id,
                pattern_id=request.pattern_id,
                status=ConsensusStatus.FAILED,
                error_message=f"Voting consensus error: {e}"
            )
    
    async def _execute_adaptive_consensus(self, request: ConsensusValidationRequest) -> ConsensusValidationResult:
        """Execute adaptive consensus with fallback mechanisms."""
        try:
            # Try primary protocol first
            primary_protocol = self._select_consensus_protocol(request)
            
            if primary_protocol == ConsensusProtocol.BFT:
                result = await self._execute_bft_consensus(request)
            elif primary_protocol == ConsensusProtocol.RAFT:
                result = await self._execute_raft_consensus(request)
            else:
                result = await self._execute_voting_only_consensus(request)
            
            # If primary failed, try fallback
            if result.status == ConsensusStatus.FAILED:
                self.logger.info(f"Primary consensus protocol failed, trying fallback for {request.request_id}")
                
                # Try voting-only as fallback
                if primary_protocol != ConsensusProtocol.VOTING_ONLY:
                    fallback_result = await self._execute_voting_only_consensus(request)
                    if fallback_result.status == ConsensusStatus.COMPLETED:
                        fallback_result.metadata['fallback_used'] = True
                        fallback_result.metadata['primary_protocol_failed'] = primary_protocol.value
                        return fallback_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive consensus execution failed: {e}")
            return ConsensusValidationResult(
                request_id=request.request_id,
                pattern_id=request.pattern_id,
                status=ConsensusStatus.FAILED,
                error_message=f"Adaptive consensus error: {e}"
            )
    
    async def _collect_agent_votes(self, request: ConsensusValidationRequest) -> List[Vote]:
        """Collect votes from available validation agents."""
        try:
            # Send vote requests to available agents
            vote_request_message = AgentMessage.create(
                sender_id=self.agent_id,
                message_type="consensus_vote_request",
                payload={
                    'request_id': request.request_id,
                    'pattern_id': request.pattern_id,
                    'pattern_data': request.pattern_data,
                    'statistical_results': request.statistical_results,
                    'timeout_seconds': min(30, request.timeout_seconds // 2)
                }
            )
            
            await self.nats_client.publish("consensus.vote_request", vote_request_message)
            
            # Wait for votes with timeout
            votes = []
            end_time = time.time() + min(30, request.timeout_seconds // 2)
            
            # This is simplified - in practice would collect votes via message handlers
            # For now, simulate vote collection
            await asyncio.sleep(2)  # Wait for responses
            
            # Generate simulated votes for available agents
            for agent_id in list(self.available_agents)[:request.min_participants]:
                # Get agent reputation for weighting
                reputation_score = self.reputation_system.get_agent_reputation(agent_id)
                weight = self.reputation_system.get_voting_weight(agent_id)
                
                # Simulate vote based on statistical results (simplified)
                confidence = 0.7 + (hash(f"{agent_id}{request.pattern_id}") % 30) / 100
                vote_decision = VoteType.SIGNIFICANT if confidence > 0.8 else VoteType.NOT_SIGNIFICANT
                
                vote = Vote(
                    agent_id=agent_id,
                    pattern_id=request.pattern_id,
                    vote=vote_decision,
                    confidence=confidence,
                    statistical_evidence=request.statistical_results[0] if request.statistical_results else {},
                    reasoning=f"Statistical analysis by {agent_id}",
                    timestamp=datetime.utcnow().isoformat(),
                    weight=weight,
                    reputation_score=reputation_score.overall_score if reputation_score else 0.5
                )
                
                votes.append(vote)
            
            self.logger.info(f"Collected {len(votes)} votes for pattern {request.pattern_id}")
            return votes
            
        except Exception as e:
            self.logger.error(f"Error collecting agent votes: {e}")
            return []
    
    async def _update_agent_reputations(self, request: ConsensusValidationRequest, voting_result: VotingResult) -> None:
        """Update agent reputations based on consensus results."""
        try:
            if not voting_result.detailed_votes:
                return
            
            # Determine consensus decision for reputation updates
            consensus_decision = voting_result.decision.value if voting_result.consensus_achieved else None
            
            for vote in voting_result.detailed_votes:
                # Create validation outcome record
                outcome = ValidationOutcome(
                    validation_id=request.request_id,
                    agent_id=vote.agent_id,
                    pattern_id=request.pattern_id,
                    prediction=vote.vote.value,
                    confidence=vote.confidence,
                    peer_consensus=consensus_decision,
                    statistical_evidence_quality=0.8,  # Simplified
                    response_time_seconds=2.0  # Simplified
                )
                
                # Record outcome for reputation tracking
                self.reputation_system.record_validation_outcome(outcome)
            
            self.logger.info(f"Updated reputations for {len(voting_result.detailed_votes)} agents")
            
        except Exception as e:
            self.logger.error(f"Error updating agent reputations: {e}")
    
    async def _initialize_consensus_protocols(self) -> None:
        """Initialize consensus protocols."""
        try:
            if self.nats_client:
                # Initialize BFT consensus
                self.bft_consensus = ByzantineFaultTolerantConsensus(
                    node_id=f"bft_{self.agent_id}",
                    nats_client=self.nats_client
                )
                await self.bft_consensus.start()
                
                # Initialize RAFT consensus
                self.raft_consensus = RaftConsensus(
                    node_id=f"raft_{self.agent_id}",
                    nats_client=self.nats_client
                )
                await self.raft_consensus.start()
                
                self.logger.info("Consensus protocols initialized")
            else:
                self.logger.warning("NATS client not available, consensus protocols not initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize consensus protocols: {e}")
    
    def _initialize_voting_mechanisms(self) -> None:
        """Initialize voting mechanisms."""
        try:
            # Create various voting mechanisms
            self.voting_mechanisms = {
                'majority': create_voting_mechanism('majority', require_majority=True),
                'weighted': create_voting_mechanism('weighted', weight_threshold=0.6),
                'threshold': create_voting_mechanism('threshold', significance_threshold=0.67),
                'quorum': create_voting_mechanism('quorum', min_participants=3),
                'adaptive': create_voting_mechanism('adaptive')
            }
            
            self.logger.info(f"Initialized {len(self.voting_mechanisms)} voting mechanisms")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voting mechanisms: {e}")
    
    async def _handle_validation_request(self, message: AgentMessage) -> None:
        """Handle consensus validation request messages."""
        try:
            payload = message.payload
            
            # Create validation request from message
            request = ConsensusValidationRequest(
                request_id=payload.get('request_id', str(uuid.uuid4())),
                pattern_id=payload['pattern_id'],
                pattern_data=payload['pattern_data'],
                statistical_results=payload.get('statistical_results', []),
                requester_id=message.sender_id,
                **{k: v for k, v in payload.items() if k in [
                    'validation_methods', 'consensus_protocol', 'voting_mechanism',
                    'timeout_seconds', 'min_participants', 'require_statistical_evidence'
                ]}
            )
            
            # Execute consensus validation
            result = await self._execute_consensus_validation(request)
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="consensus_validation_response",
                payload=result.to_dict(),
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling validation request: {e}")
    
    async def _handle_vote_message(self, message: AgentMessage) -> None:
        """Handle vote messages from agents."""
        try:
            # This would handle actual vote messages in a real implementation
            # For now, just log the vote
            payload = message.payload
            agent_id = payload.get('agent_id', message.sender_id)
            pattern_id = payload.get('pattern_id')
            vote_data = payload.get('vote')
            
            self.logger.debug(f"Received vote from {agent_id} for pattern {pattern_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling vote message: {e}")
    
    async def _handle_agent_announcement(self, message: AgentMessage) -> None:
        """Handle agent availability announcements."""
        try:
            payload = message.payload
            agent_id = payload.get('agent_id', message.sender_id)
            capabilities = payload.get('capabilities', [])
            
            if agent_id and agent_id != self.agent_id:
                self.available_agents.add(agent_id)
                self.agent_capabilities[agent_id] = capabilities
                
                # Register with reputation system
                self.reputation_system.register_agent(agent_id)
                
                self.logger.debug(f"Registered agent {agent_id} with capabilities: {capabilities}")
            
        except Exception as e:
            self.logger.error(f"Error handling agent announcement: {e}")
    
    async def _handle_status_request(self, message: AgentMessage) -> None:
        """Handle status request messages."""
        try:
            status = self.get_consensus_status()
            
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="consensus_status_response",
                payload=status,
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling status request: {e}")
    
    def get_consensus_status(self) -> Dict[str, Any]:
        """Get current consensus coordinator status."""
        return {
            'coordinator_id': self.agent_id,
            'available_agents': len(self.available_agents),
            'active_requests': len(self.active_requests),
            'consensus_metrics': self.consensus_metrics,
            'protocols_available': {
                'bft': self.bft_consensus is not None,
                'raft': self.raft_consensus is not None
            },
            'voting_mechanisms': list(self.voting_mechanisms.keys()),
            'reputation_system_active': True,
            'network_topology': self.network_topology,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _update_consensus_metrics(self, result: ConsensusValidationResult) -> None:
        """Update consensus performance metrics."""
        try:
            self.consensus_metrics['total_requests'] += 1
            
            if result.status == ConsensusStatus.COMPLETED:
                self.consensus_metrics['successful_consensus'] += 1
            else:
                self.consensus_metrics['failed_consensus'] += 1
            
            # Update average response time
            if result.execution_time_seconds > 0:
                current_avg = self.consensus_metrics['average_response_time']
                total_requests = self.consensus_metrics['total_requests']
                self.consensus_metrics['average_response_time'] = (
                    (current_avg * (total_requests - 1) + result.execution_time_seconds) / total_requests
                )
            
            # Update protocol usage
            if result.consensus_protocol_used:
                protocol_name = result.consensus_protocol_used.value
                self.consensus_metrics['protocol_usage'][protocol_name] = (
                    self.consensus_metrics['protocol_usage'].get(protocol_name, 0) + 1
                )
            
            # Update agent participation rates
            for agent_id in result.participating_agents:
                self.consensus_metrics['agent_participation_rates'][agent_id] = (
                    self.consensus_metrics['agent_participation_rates'].get(agent_id, 0) + 1
                )
            
        except Exception as e:
            self.logger.error(f"Error updating consensus metrics: {e}")
    
    async def _agent_discovery_loop(self) -> None:
        """Background task for agent discovery."""
        while not self.shutdown_event.is_set():
            try:
                # Announce coordinator presence
                announcement = AgentMessage.create(
                    sender_id=self.agent_id,
                    message_type="consensus_coordinator_announcement",
                    payload={
                        'coordinator_id': self.agent_id,
                        'protocols_supported': ['bft', 'raft', 'voting_only'],
                        'voting_mechanisms': list(self.voting_mechanisms.keys()),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                await self.nats_client.publish("consensus.coordinator.announce", announcement)
                
                await asyncio.sleep(30)  # Announce every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in agent discovery loop: {e}")
                await asyncio.sleep(10)
    
    async def _request_timeout_monitor(self) -> None:
        """Monitor request timeouts."""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                expired_requests = []
                
                for request_id, timeout_time in self.request_timeouts.items():
                    if current_time > timeout_time:
                        expired_requests.append(request_id)
                
                # Handle expired requests
                for request_id in expired_requests:
                    await self._handle_request_timeout(request_id)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in timeout monitor: {e}")
                await asyncio.sleep(10)
    
    async def _handle_request_timeout(self, request_id: str) -> None:
        """Handle request timeout."""
        try:
            self.logger.warning(f"Request {request_id} timed out")
            
            request = self.active_requests.get(request_id)
            if request:
                result = ConsensusValidationResult(
                    request_id=request_id,
                    pattern_id=request.pattern_id,
                    status=ConsensusStatus.TIMEOUT,
                    error_message="Consensus request timed out"
                )
                
                self.request_results[request_id] = result
                self.request_status[request_id] = ConsensusStatus.TIMEOUT
            
            # Cleanup
            self.active_requests.pop(request_id, None)
            self.request_timeouts.pop(request_id, None)
            
        except Exception as e:
            self.logger.error(f"Error handling request timeout: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Background task for metrics collection."""
        while not self.shutdown_event.is_set():
            try:
                # Update network topology metrics
                self.network_topology = {
                    'total_agents': len(self.available_agents),
                    'agent_capabilities': self.agent_capabilities,
                    'reputation_stats': self.reputation_system.get_system_stats(),
                    'last_updated': datetime.utcnow().isoformat()
                }
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(30)
    
    async def _health_monitoring_loop(self) -> None:
        """Background task for health monitoring."""
        while not self.shutdown_event.is_set():
            try:
                # Check consensus protocol health
                if self.bft_consensus:
                    bft_status = self.bft_consensus.get_status()
                    if bft_status.get('current_node_count', 0) < 4:
                        self.logger.warning("BFT consensus has insufficient nodes")
                
                if self.raft_consensus:
                    raft_status = self.raft_consensus.get_status()
                    if raft_status.get('state') != 'leader' and len(self.available_agents) > 0:
                        self.logger.debug("RAFT consensus is not leader")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _process_completed_requests(self) -> None:
        """Process any completed requests."""
        # This would handle any additional processing needed for completed requests
        pass
    
    async def _update_network_topology(self) -> None:
        """Update network topology information."""
        # This would update network topology based on agent announcements
        pass
    
    async def _cleanup_old_requests(self) -> None:
        """Clean up old completed requests."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Clean up old results (keep for 24 hours)
            old_results = [
                request_id for request_id, result in self.request_results.items()
                if datetime.fromisoformat(result.timestamp) < cutoff_time
            ]
            
            for request_id in old_results:
                self.request_results.pop(request_id, None)
                self.request_status.pop(request_id, None)
            
            if old_results:
                self.logger.info(f"Cleaned up {len(old_results)} old consensus results")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old requests: {e}")