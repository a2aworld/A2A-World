"""
A2A World Platform - Byzantine Fault Tolerant Consensus

Implements Byzantine Fault Tolerant consensus algorithm for reliable
pattern validation in the presence of malicious or faulty agents.
Based on practical Byzantine Fault Tolerance (pBFT) protocol.
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import uuid
import json

from agents.core.messaging import AgentMessage, NATSClient


class ConsensusPhase(Enum):
    """Consensus phases in the pBFT protocol."""
    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    DECIDED = "decided"


class MessageType(Enum):
    """Byzantine consensus message types."""
    REQUEST = "request"
    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"


@dataclass
class ConsensusRequest:
    """Request for consensus on pattern validation."""
    request_id: str
    pattern_id: str
    pattern_data: Dict[str, Any]
    statistical_results: List[Dict[str, Any]]
    requester_id: str
    timestamp: str
    timeout_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'pattern_id': self.pattern_id,
            'pattern_data': self.pattern_data,
            'statistical_results': self.statistical_results,
            'requester_id': self.requester_id,
            'timestamp': self.timestamp,
            'timeout_seconds': self.timeout_seconds
        }
    
    def hash(self) -> str:
        """Generate hash of request for message integrity."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ConsensusVote:
    """Individual agent vote on pattern significance."""
    agent_id: str
    pattern_id: str
    vote: str  # "significant", "not_significant", "uncertain"
    confidence: float  # 0.0 to 1.0
    statistical_evidence: Dict[str, Any]
    reasoning: str
    timestamp: str
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'pattern_id': self.pattern_id,
            'vote': self.vote,
            'confidence': self.confidence,
            'statistical_evidence': self.statistical_evidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp,
            'signature': self.signature
        }


@dataclass
class ConsensusMessage:
    """Byzantine consensus protocol message."""
    message_id: str
    message_type: MessageType
    view_number: int
    sequence_number: int
    sender_id: str
    request_hash: str
    vote_data: Optional[ConsensusVote] = None
    signatures: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.signatures is None:
            self.signatures = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'view_number': self.view_number,
            'sequence_number': self.sequence_number,
            'sender_id': self.sender_id,
            'request_hash': self.request_hash,
            'vote_data': self.vote_data.to_dict() if self.vote_data else None,
            'signatures': self.signatures,
            'timestamp': self.timestamp
        }


class ByzantineFaultTolerantConsensus:
    """
    Byzantine Fault Tolerant consensus implementation for pattern validation.
    
    Implements practical Byzantine Fault Tolerance (pBFT) protocol adapted
    for distributed pattern validation scenarios. Handles up to f faulty
    nodes out of 3f+1 total nodes.
    """
    
    def __init__(
        self,
        node_id: str,
        nats_client: NATSClient,
        min_nodes: int = 4,
        timeout_seconds: int = 30
    ):
        """
        Initialize Byzantine Fault Tolerant consensus node.
        
        Args:
            node_id: Unique identifier for this consensus node
            nats_client: NATS client for inter-node communication
            min_nodes: Minimum number of nodes required (3f+1)
            timeout_seconds: Consensus timeout in seconds
        """
        self.node_id = node_id
        self.nats_client = nats_client
        self.min_nodes = max(min_nodes, 4)  # Ensure at least 4 nodes (f=1)
        self.timeout_seconds = timeout_seconds
        
        # Consensus state
        self.view_number = 0
        self.sequence_number = 0
        self.is_primary = False
        self.active_nodes: Set[str] = set()
        self.faulty_nodes: Set[str] = set()
        
        # Message storage
        self.pending_requests: Dict[str, ConsensusRequest] = {}
        self.prepare_messages: Dict[str, Dict[str, ConsensusMessage]] = {}
        self.pre_commit_messages: Dict[str, Dict[str, ConsensusMessage]] = {}
        self.commit_messages: Dict[str, Dict[str, ConsensusMessage]] = {}
        self.consensus_results: Dict[str, Dict[str, Any]] = {}
        
        # Request tracking
        self.request_phases: Dict[str, ConsensusPhase] = {}
        self.request_timeouts: Dict[str, datetime] = {}
        
        self.logger = logging.getLogger(f"bft_consensus.{node_id}")
        self.logger.info(f"Initialized BFT consensus node {node_id}")
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the BFT consensus node."""
        self.logger.info(f"Starting BFT consensus node {self.node_id}")
        
        # Subscribe to consensus messages
        await self._setup_subscriptions()
        
        # Start background tasks
        self.background_tasks.extend([
            asyncio.create_task(self._timeout_monitor()),
            asyncio.create_task(self._view_change_monitor()),
            asyncio.create_task(self._health_monitor())
        ])
        
        # Announce node availability
        await self._announce_node_availability()
        
        self.logger.info(f"BFT consensus node {self.node_id} started")
    
    async def stop(self) -> None:
        """Stop the BFT consensus node."""
        self.logger.info(f"Stopping BFT consensus node {self.node_id}")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info(f"BFT consensus node {self.node_id} stopped")
    
    async def request_consensus(self, request: ConsensusRequest) -> Optional[Dict[str, Any]]:
        """
        Request consensus on pattern validation.
        
        Args:
            request: Consensus request containing pattern data
            
        Returns:
            Consensus result or None if consensus fails
        """
        try:
            self.logger.info(f"Requesting consensus for pattern {request.pattern_id}")
            
            # Check if we have enough nodes
            if len(self.active_nodes) < self.min_nodes:
                raise ValueError(f"Insufficient active nodes: {len(self.active_nodes)} < {self.min_nodes}")
            
            # Store request
            self.pending_requests[request.request_id] = request
            self.request_phases[request.request_id] = ConsensusPhase.PREPARE
            self.request_timeouts[request.request_id] = datetime.utcnow() + timedelta(seconds=request.timeout_seconds)
            
            # If we're the primary, initiate consensus
            if self.is_primary:
                await self._initiate_consensus(request)
            else:
                # Send request to primary
                await self._forward_to_primary(request)
            
            # Wait for consensus result
            return await self._wait_for_consensus(request.request_id, request.timeout_seconds)
            
        except Exception as e:
            self.logger.error(f"Consensus request failed for pattern {request.pattern_id}: {e}")
            return None
    
    async def cast_vote(self, pattern_id: str, vote: ConsensusVote) -> bool:
        """
        Cast a vote on pattern significance.
        
        Args:
            pattern_id: Pattern identifier
            vote: Consensus vote with statistical evidence
            
        Returns:
            True if vote was cast successfully
        """
        try:
            # Validate vote
            if vote.confidence < 0.0 or vote.confidence > 1.0:
                raise ValueError(f"Invalid confidence value: {vote.confidence}")
            
            if vote.vote not in ["significant", "not_significant", "uncertain"]:
                raise ValueError(f"Invalid vote value: {vote.vote}")
            
            # Sign vote (simplified - in production would use proper cryptographic signatures)
            vote.signature = self._sign_vote(vote)
            
            # Send vote to all nodes
            message = AgentMessage.create(
                sender_id=self.node_id,
                message_type="consensus_vote",
                payload={
                    "pattern_id": pattern_id,
                    "vote": vote.to_dict()
                }
            )
            
            await self.nats_client.publish("consensus.votes", message)
            
            self.logger.info(f"Cast vote for pattern {pattern_id}: {vote.vote} (confidence: {vote.confidence:.3f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cast vote for pattern {pattern_id}: {e}")
            return False
    
    async def _initiate_consensus(self, request: ConsensusRequest) -> None:
        """Initiate consensus protocol as primary node."""
        try:
            # Increment sequence number
            self.sequence_number += 1
            
            # Create prepare message
            prepare_msg = ConsensusMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.PREPARE,
                view_number=self.view_number,
                sequence_number=self.sequence_number,
                sender_id=self.node_id,
                request_hash=request.hash()
            )
            
            # Initialize message tracking
            self.prepare_messages[request.request_id] = {self.node_id: prepare_msg}
            
            # Broadcast prepare message to all nodes
            await self._broadcast_consensus_message(prepare_msg, request)
            
            self.logger.info(f"Initiated consensus for request {request.request_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initiate consensus: {e}")
    
    async def _handle_prepare_message(self, message: ConsensusMessage, request: ConsensusRequest) -> None:
        """Handle prepare phase message."""
        try:
            request_id = request.request_id
            
            # Store prepare message
            if request_id not in self.prepare_messages:
                self.prepare_messages[request_id] = {}
            self.prepare_messages[request_id][message.sender_id] = message
            
            # Check if we have enough prepare messages (2f+1)
            required_prepares = (2 * self._max_faults()) + 1
            if len(self.prepare_messages[request_id]) >= required_prepares:
                # Move to pre-commit phase
                await self._send_pre_commit(request)
                self.request_phases[request_id] = ConsensusPhase.PRE_COMMIT
            
        except Exception as e:
            self.logger.error(f"Error handling prepare message: {e}")
    
    async def _send_pre_commit(self, request: ConsensusRequest) -> None:
        """Send pre-commit message."""
        try:
            pre_commit_msg = ConsensusMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.PRE_COMMIT,
                view_number=self.view_number,
                sequence_number=self.sequence_number,
                sender_id=self.node_id,
                request_hash=request.hash()
            )
            
            # Initialize pre-commit tracking
            if request.request_id not in self.pre_commit_messages:
                self.pre_commit_messages[request.request_id] = {}
            self.pre_commit_messages[request.request_id][self.node_id] = pre_commit_msg
            
            # Broadcast pre-commit message
            await self._broadcast_consensus_message(pre_commit_msg, request)
            
            self.logger.debug(f"Sent pre-commit for request {request.request_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send pre-commit: {e}")
    
    async def _handle_pre_commit_message(self, message: ConsensusMessage, request: ConsensusRequest) -> None:
        """Handle pre-commit phase message."""
        try:
            request_id = request.request_id
            
            # Store pre-commit message
            if request_id not in self.pre_commit_messages:
                self.pre_commit_messages[request_id] = {}
            self.pre_commit_messages[request_id][message.sender_id] = message
            
            # Check if we have enough pre-commit messages (2f+1)
            required_pre_commits = (2 * self._max_faults()) + 1
            if len(self.pre_commit_messages[request_id]) >= required_pre_commits:
                # Move to commit phase
                await self._send_commit(request)
                self.request_phases[request_id] = ConsensusPhase.COMMIT
            
        except Exception as e:
            self.logger.error(f"Error handling pre-commit message: {e}")
    
    async def _send_commit(self, request: ConsensusRequest) -> None:
        """Send commit message."""
        try:
            commit_msg = ConsensusMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.COMMIT,
                view_number=self.view_number,
                sequence_number=self.sequence_number,
                sender_id=self.node_id,
                request_hash=request.hash()
            )
            
            # Initialize commit tracking
            if request.request_id not in self.commit_messages:
                self.commit_messages[request.request_id] = {}
            self.commit_messages[request.request_id][self.node_id] = commit_msg
            
            # Broadcast commit message
            await self._broadcast_consensus_message(commit_msg, request)
            
            self.logger.debug(f"Sent commit for request {request.request_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send commit: {e}")
    
    async def _handle_commit_message(self, message: ConsensusMessage, request: ConsensusRequest) -> None:
        """Handle commit phase message."""
        try:
            request_id = request.request_id
            
            # Store commit message
            if request_id not in self.commit_messages:
                self.commit_messages[request_id] = {}
            self.commit_messages[request_id][message.sender_id] = message
            
            # Check if we have enough commit messages (2f+1)
            required_commits = (2 * self._max_faults()) + 1
            if len(self.commit_messages[request_id]) >= required_commits:
                # Finalize consensus
                await self._finalize_consensus(request)
                self.request_phases[request_id] = ConsensusPhase.DECIDED
            
        except Exception as e:
            self.logger.error(f"Error handling commit message: {e}")
    
    async def _finalize_consensus(self, request: ConsensusRequest) -> None:
        """Finalize consensus and compute result."""
        try:
            request_id = request.request_id
            
            # Collect all votes from statistical evidence
            votes = []
            confidences = []
            
            # Extract votes from commit messages
            for commit_msg in self.commit_messages[request_id].values():
                if commit_msg.vote_data:
                    votes.append(commit_msg.vote_data.vote)
                    confidences.append(commit_msg.vote_data.confidence)
            
            # Compute consensus result
            consensus_result = self._compute_consensus_result(votes, confidences, request)
            
            # Store result
            self.consensus_results[request_id] = consensus_result
            
            self.logger.info(f"Finalized consensus for request {request_id}: {consensus_result.get('decision', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to finalize consensus: {e}")
    
    def _compute_consensus_result(self, votes: List[str], confidences: List[float], request: ConsensusRequest) -> Dict[str, Any]:
        """Compute final consensus result from votes."""
        if not votes:
            return {
                'decision': 'uncertain',
                'confidence': 0.0,
                'vote_breakdown': {},
                'participating_nodes': 0,
                'consensus_achieved': False,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Count votes
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Find majority decision
        majority_vote = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_vote]
        
        # Calculate confidence (average of majority votes)
        majority_confidences = [conf for vote, conf in zip(votes, confidences) if vote == majority_vote]
        avg_confidence = sum(majority_confidences) / len(majority_confidences) if majority_confidences else 0.0
        
        # Determine if consensus achieved (simple majority for now)
        consensus_achieved = majority_count > len(votes) / 2
        
        return {
            'decision': majority_vote,
            'confidence': avg_confidence,
            'vote_breakdown': vote_counts,
            'participating_nodes': len(votes),
            'total_votes': len(votes),
            'majority_votes': majority_count,
            'consensus_achieved': consensus_achieved,
            'pattern_id': request.pattern_id,
            'request_id': request.request_id,
            'view_number': self.view_number,
            'sequence_number': self.sequence_number,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _wait_for_consensus(self, request_id: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
        """Wait for consensus result with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check if consensus reached
            if request_id in self.consensus_results:
                return self.consensus_results[request_id]
            
            # Check if request timed out
            if request_id in self.request_timeouts:
                if datetime.utcnow() > self.request_timeouts[request_id]:
                    self.logger.warning(f"Consensus request {request_id} timed out")
                    break
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def _broadcast_consensus_message(self, message: ConsensusMessage, request: ConsensusRequest) -> None:
        """Broadcast consensus message to all nodes."""
        try:
            agent_message = AgentMessage.create(
                sender_id=self.node_id,
                message_type=f"bft_{message.message_type.value}",
                payload={
                    'consensus_message': message.to_dict(),
                    'request_data': request.to_dict()
                }
            )
            
            await self.nats_client.publish("consensus.bft", agent_message)
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast consensus message: {e}")
    
    async def _setup_subscriptions(self) -> None:
        """Setup NATS subscriptions for consensus messages."""
        try:
            # Subscribe to BFT consensus messages
            await self.nats_client.subscribe(
                "consensus.bft",
                self._handle_consensus_message,
                queue_group=f"bft-consensus-{self.node_id}"
            )
            
            # Subscribe to node announcements
            await self.nats_client.subscribe(
                "consensus.nodes.announce",
                self._handle_node_announcement
            )
            
            # Subscribe to vote messages
            await self.nats_client.subscribe(
                "consensus.votes",
                self._handle_vote_message
            )
            
            self.logger.info("Set up BFT consensus subscriptions")
            
        except Exception as e:
            self.logger.error(f"Failed to setup subscriptions: {e}")
    
    async def _handle_consensus_message(self, message: AgentMessage) -> None:
        """Handle incoming consensus messages."""
        try:
            if message.sender_id == self.node_id:
                return  # Ignore own messages
            
            payload = message.payload
            consensus_msg_data = payload.get('consensus_message', {})
            request_data = payload.get('request_data', {})
            
            # Reconstruct objects
            consensus_msg = ConsensusMessage(**consensus_msg_data)
            request = ConsensusRequest(**request_data)
            
            # Handle based on message type
            if consensus_msg.message_type == MessageType.PREPARE:
                await self._handle_prepare_message(consensus_msg, request)
            elif consensus_msg.message_type == MessageType.PRE_COMMIT:
                await self._handle_pre_commit_message(consensus_msg, request)
            elif consensus_msg.message_type == MessageType.COMMIT:
                await self._handle_commit_message(consensus_msg, request)
            
        except Exception as e:
            self.logger.error(f"Error handling consensus message: {e}")
    
    async def _handle_node_announcement(self, message: AgentMessage) -> None:
        """Handle node availability announcements."""
        try:
            node_id = message.payload.get('node_id')
            if node_id and node_id != self.node_id:
                self.active_nodes.add(node_id)
                self.logger.info(f"Node {node_id} announced availability")
                
                # Check if we should be primary
                await self._check_primary_status()
            
        except Exception as e:
            self.logger.error(f"Error handling node announcement: {e}")
    
    async def _handle_vote_message(self, message: AgentMessage) -> None:
        """Handle vote messages."""
        try:
            payload = message.payload
            vote_data = payload.get('vote', {})
            pattern_id = payload.get('pattern_id')
            
            if vote_data and pattern_id:
                vote = ConsensusVote(**vote_data)
                
                # Verify vote signature (simplified)
                if self._verify_vote_signature(vote):
                    self.logger.debug(f"Received valid vote from {vote.agent_id} for pattern {pattern_id}")
                else:
                    self.logger.warning(f"Invalid vote signature from {vote.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling vote message: {e}")
    
    async def _announce_node_availability(self) -> None:
        """Announce this node's availability."""
        try:
            message = AgentMessage.create(
                sender_id=self.node_id,
                message_type="node_announcement",
                payload={
                    'node_id': self.node_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'capabilities': ['bft_consensus', 'pattern_validation']
                }
            )
            
            await self.nats_client.publish("consensus.nodes.announce", message)
            self.logger.info(f"Announced node {self.node_id} availability")
            
        except Exception as e:
            self.logger.error(f"Failed to announce node availability: {e}")
    
    async def _check_primary_status(self) -> None:
        """Check and update primary node status."""
        try:
            # Simple primary election: lowest node ID is primary
            if self.active_nodes:
                sorted_nodes = sorted(list(self.active_nodes) + [self.node_id])
                new_primary = sorted_nodes[0] == self.node_id
                
                if new_primary != self.is_primary:
                    self.is_primary = new_primary
                    if self.is_primary:
                        self.logger.info(f"Node {self.node_id} elected as primary")
                    else:
                        self.logger.info(f"Node {self.node_id} is backup")
            
        except Exception as e:
            self.logger.error(f"Error checking primary status: {e}")
    
    async def _forward_to_primary(self, request: ConsensusRequest) -> None:
        """Forward request to primary node."""
        # For now, just wait - in full implementation would forward to primary
        pass
    
    def _max_faults(self) -> int:
        """Calculate maximum number of Byzantine faults tolerated."""
        total_nodes = len(self.active_nodes) + 1  # +1 for self
        return (total_nodes - 1) // 3
    
    def _sign_vote(self, vote: ConsensusVote) -> str:
        """Sign a vote (simplified implementation)."""
        # In production, would use proper cryptographic signatures
        content = json.dumps(vote.to_dict(), sort_keys=True)
        return hashlib.sha256(f"{self.node_id}:{content}".encode()).hexdigest()
    
    def _verify_vote_signature(self, vote: ConsensusVote) -> bool:
        """Verify vote signature (simplified implementation)."""
        # In production, would verify cryptographic signature
        expected_sig = hashlib.sha256(f"{vote.agent_id}:{json.dumps(vote.to_dict(), sort_keys=True)}".encode()).hexdigest()
        return vote.signature == expected_sig
    
    async def _timeout_monitor(self) -> None:
        """Monitor request timeouts."""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                expired_requests = []
                
                for request_id, timeout_time in self.request_timeouts.items():
                    if current_time > timeout_time:
                        expired_requests.append(request_id)
                
                # Clean up expired requests
                for request_id in expired_requests:
                    await self._handle_request_timeout(request_id)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in timeout monitor: {e}")
                await asyncio.sleep(5)
    
    async def _handle_request_timeout(self, request_id: str) -> None:
        """Handle request timeout."""
        try:
            self.logger.warning(f"Request {request_id} timed out")
            
            # Clean up request state
            self.request_timeouts.pop(request_id, None)
            self.pending_requests.pop(request_id, None)
            self.request_phases.pop(request_id, None)
            self.prepare_messages.pop(request_id, None)
            self.pre_commit_messages.pop(request_id, None)
            self.commit_messages.pop(request_id, None)
            
            # Set timeout result
            self.consensus_results[request_id] = {
                'decision': 'timeout',
                'confidence': 0.0,
                'consensus_achieved': False,
                'error': 'Request timed out',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error handling request timeout: {e}")
    
    async def _view_change_monitor(self) -> None:
        """Monitor for view changes (primary failures)."""
        while not self.shutdown_event.is_set():
            try:
                # Check if primary is responsive
                # In full implementation would detect primary failures
                # and initiate view change protocol
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in view change monitor: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitor(self) -> None:
        """Monitor node health and connectivity."""
        while not self.shutdown_event.is_set():
            try:
                # Check NATS connectivity
                if not self.nats_client.is_connected:
                    self.logger.error("NATS connection lost")
                
                # Re-announce availability periodically
                if len(self.active_nodes) > 0:
                    await self._announce_node_availability()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current consensus node status."""
        return {
            'node_id': self.node_id,
            'is_primary': self.is_primary,
            'view_number': self.view_number,
            'sequence_number': self.sequence_number,
            'active_nodes': list(self.active_nodes),
            'faulty_nodes': list(self.faulty_nodes),
            'pending_requests': len(self.pending_requests),
            'max_faults_tolerated': self._max_faults(),
            'min_nodes_required': self.min_nodes,
            'current_node_count': len(self.active_nodes) + 1
        }