"""
A2A World Platform - RAFT Consensus Protocol

Implements RAFT consensus algorithm for distributed pattern validation.
RAFT provides leader election, log replication, and consistent state management
across validation agents in a more understandable approach than Byzantine protocols.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json

from agents.core.messaging import AgentMessage, NATSClient


class NodeState(Enum):
    """RAFT node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LogEntryType(Enum):
    """Types of log entries in RAFT."""
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESULT = "validation_result"
    CONFIGURATION_CHANGE = "configuration_change"
    NO_OP = "no_op"


@dataclass
class LogEntry:
    """RAFT log entry containing validation operations."""
    term: int
    index: int
    entry_type: LogEntryType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    committed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'index': self.index,
            'entry_type': self.entry_type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'committed': self.committed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        return cls(
            term=data['term'],
            index=data['index'],
            entry_type=LogEntryType(data['entry_type']),
            data=data['data'],
            timestamp=data.get('timestamp', datetime.utcnow().isoformat()),
            committed=data.get('committed', False)
        )


@dataclass
class ValidationEntry:
    """Validation request entry for RAFT log."""
    validation_id: str
    pattern_id: str
    pattern_data: Dict[str, Any]
    statistical_results: List[Dict[str, Any]]
    requester_id: str
    validation_method: str = "consensus"
    priority: int = 5
    timeout_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'validation_id': self.validation_id,
            'pattern_id': self.pattern_id,
            'pattern_data': self.pattern_data,
            'statistical_results': self.statistical_results,
            'requester_id': self.requester_id,
            'validation_method': self.validation_method,
            'priority': self.priority,
            'timeout_seconds': self.timeout_seconds
        }


@dataclass
class RaftMessage:
    """RAFT protocol message."""
    message_type: str  # "request_vote", "append_entries", "install_snapshot"
    term: int
    sender_id: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_type': self.message_type,
            'term': self.term,
            'sender_id': self.sender_id,
            'data': self.data,
            'timestamp': self.timestamp
        }


class RaftNode:
    """
    RAFT consensus node for distributed pattern validation.
    
    Implements the RAFT consensus algorithm with leader election,
    log replication, and state machine replication for validation coordination.
    """
    
    def __init__(
        self,
        node_id: str,
        nats_client: NATSClient,
        election_timeout_range: Tuple[int, int] = (150, 300),
        heartbeat_interval: int = 50
    ):
        """
        Initialize RAFT consensus node.
        
        Args:
            node_id: Unique identifier for this node
            nats_client: NATS client for communication
            election_timeout_range: Range for random election timeout (ms)
            heartbeat_interval: Heartbeat interval for leader (ms)
        """
        self.node_id = node_id
        self.nats_client = nats_client
        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval
        
        # RAFT state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Node state
        self.state = NodeState.FOLLOWER
        self.current_leader: Optional[str] = None
        self.cluster_nodes: Set[str] = set()
        
        # Leader state (only used when leader)
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timers and tasks
        self.election_timer: Optional[asyncio.Task] = None
        self.heartbeat_timer: Optional[asyncio.Task] = None
        self.last_heartbeat_received = time.time()
        
        # Validation state machine
        self.validation_requests: Dict[str, ValidationEntry] = {}
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self.pending_validations: Dict[str, asyncio.Event] = {}
        
        self.logger = logging.getLogger(f"raft.{node_id}")
        self.logger.info(f"Initialized RAFT node {node_id}")
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the RAFT node."""
        self.logger.info(f"Starting RAFT node {self.node_id}")
        
        # Setup message subscriptions
        await self._setup_subscriptions()
        
        # Start as follower
        await self._become_follower(0)
        
        # Start background tasks
        self.background_tasks.extend([
            asyncio.create_task(self._apply_committed_entries()),
            asyncio.create_task(self._cluster_discovery())
        ])
        
        self.logger.info(f"RAFT node {self.node_id} started as follower")
    
    async def stop(self) -> None:
        """Stop the RAFT node."""
        self.logger.info(f"Stopping RAFT node {self.node_id}")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Cancel timers
        if self.election_timer:
            self.election_timer.cancel()
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info(f"RAFT node {self.node_id} stopped")
    
    async def submit_validation(self, validation_entry: ValidationEntry) -> Optional[Dict[str, Any]]:
        """
        Submit validation request for consensus processing.
        
        Args:
            validation_entry: Validation request to process
            
        Returns:
            Validation result or None if failed
        """
        try:
            self.logger.info(f"Submitting validation for pattern {validation_entry.pattern_id}")
            
            # Only leader can accept new entries
            if self.state != NodeState.LEADER:
                if self.current_leader:
                    # Forward to leader
                    return await self._forward_to_leader(validation_entry)
                else:
                    raise ValueError("No leader available to process validation request")
            
            # Create log entry
            log_entry = LogEntry(
                term=self.current_term,
                index=len(self.log) + 1,
                entry_type=LogEntryType.VALIDATION_REQUEST,
                data=validation_entry.to_dict()
            )
            
            # Add to log
            self.log.append(log_entry)
            self.validation_requests[validation_entry.validation_id] = validation_entry
            
            # Create completion event
            completion_event = asyncio.Event()
            self.pending_validations[validation_entry.validation_id] = completion_event
            
            # Replicate to followers
            await self._replicate_log_entry(log_entry)
            
            # Wait for completion or timeout
            try:
                await asyncio.wait_for(
                    completion_event.wait(),
                    timeout=validation_entry.timeout_seconds
                )
                
                # Return result
                return self.validation_results.get(validation_entry.validation_id)
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Validation {validation_entry.validation_id} timed out")
                return {
                    'validation_id': validation_entry.validation_id,
                    'status': 'timeout',
                    'error': 'Validation request timed out'
                }
            
        except Exception as e:
            self.logger.error(f"Failed to submit validation: {e}")
            return None
    
    async def _become_follower(self, term: int) -> None:
        """Transition to follower state."""
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
        
        self.state = NodeState.FOLLOWER
        self.current_leader = None
        
        # Cancel heartbeat timer if running
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None
        
        # Start election timer
        await self._reset_election_timer()
        
        self.logger.info(f"Node {self.node_id} became follower for term {self.current_term}")
    
    async def _become_candidate(self) -> None:
        """Transition to candidate state and start election."""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.current_leader = None
        
        # Cancel election timer
        if self.election_timer:
            self.election_timer.cancel()
            self.election_timer = None
        
        self.logger.info(f"Node {self.node_id} became candidate for term {self.current_term}")
        
        # Start election
        await self._start_election()
    
    async def _become_leader(self) -> None:
        """Transition to leader state."""
        self.state = NodeState.LEADER
        self.current_leader = self.node_id
        
        # Initialize leader state
        self.next_index = {node: len(self.log) + 1 for node in self.cluster_nodes}
        self.match_index = {node: 0 for node in self.cluster_nodes}
        
        # Cancel election timer
        if self.election_timer:
            self.election_timer.cancel()
            self.election_timer = None
        
        # Start heartbeat timer
        self.heartbeat_timer = asyncio.create_task(self._heartbeat_loop())
        
        # Send initial heartbeat
        await self._send_heartbeat()
        
        self.logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        # Add no-op entry to commit previous entries
        await self._add_no_op_entry()
    
    async def _start_election(self) -> None:
        """Start leader election process."""
        try:
            self.logger.info(f"Starting election for term {self.current_term}")
            
            # Vote for self
            votes_received = 1
            votes_needed = (len(self.cluster_nodes) + 1) // 2 + 1  # Majority
            
            # Send RequestVote RPCs to all other nodes
            vote_tasks = []
            for node_id in self.cluster_nodes:
                if node_id != self.node_id:
                    task = asyncio.create_task(self._request_vote(node_id))
                    vote_tasks.append(task)
            
            # Wait for votes with timeout
            if vote_tasks:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*vote_tasks, return_exceptions=True),
                        timeout=5.0  # 5 second election timeout
                    )
                    
                    # Count votes
                    for result in results:
                        if isinstance(result, bool) and result:
                            votes_received += 1
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Election timed out for term {self.current_term}")
            
            # Check if won election
            if votes_received >= votes_needed:
                await self._become_leader()
            else:
                # Election failed, become follower
                await self._become_follower(self.current_term)
                self.logger.info(f"Election failed, received {votes_received}/{votes_needed} votes")
            
        except Exception as e:
            self.logger.error(f"Error during election: {e}")
            await self._become_follower(self.current_term)
    
    async def _request_vote(self, node_id: str) -> bool:
        """Request vote from a specific node."""
        try:
            last_log_index = len(self.log)
            last_log_term = self.log[-1].term if self.log else 0
            
            vote_request = RaftMessage(
                message_type="request_vote",
                term=self.current_term,
                sender_id=self.node_id,
                data={
                    'candidate_id': self.node_id,
                    'last_log_index': last_log_index,
                    'last_log_term': last_log_term
                }
            )
            
            # Send vote request
            agent_message = AgentMessage.create(
                sender_id=self.node_id,
                receiver_id=node_id,
                message_type="raft_request_vote",
                payload=vote_request.to_dict()
            )
            
            # Use request/response pattern with timeout
            response = await self.nats_client.request(
                f"raft.{node_id}.vote",
                agent_message,
                timeout=2.0
            )
            
            response_data = response.payload.get('vote_granted', False)
            response_term = response.payload.get('term', 0)
            
            # Check if we need to step down
            if response_term > self.current_term:
                await self._become_follower(response_term)
                return False
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error requesting vote from {node_id}: {e}")
            return False
    
    async def _handle_request_vote(self, message: AgentMessage) -> None:
        """Handle RequestVote RPC."""
        try:
            raft_msg = RaftMessage(**message.payload)
            candidate_id = raft_msg.data['candidate_id']
            candidate_term = raft_msg.term
            last_log_index = raft_msg.data['last_log_index']
            last_log_term = raft_msg.data['last_log_term']
            
            vote_granted = False
            
            # Check term
            if candidate_term > self.current_term:
                await self._become_follower(candidate_term)
            
            # Vote logic
            if (candidate_term >= self.current_term and
                (self.voted_for is None or self.voted_for == candidate_id) and
                self._is_log_up_to_date(last_log_index, last_log_term)):
                
                vote_granted = True
                self.voted_for = candidate_id
                self.last_heartbeat_received = time.time()
                await self._reset_election_timer()
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.node_id,
                receiver_id=message.sender_id,
                message_type="raft_vote_response",
                payload={
                    'term': self.current_term,
                    'vote_granted': vote_granted
                },
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
            self.logger.debug(f"Voted {vote_granted} for candidate {candidate_id} in term {candidate_term}")
            
        except Exception as e:
            self.logger.error(f"Error handling vote request: {e}")
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat (empty AppendEntries) to all followers."""
        if self.state != NodeState.LEADER:
            return
        
        try:
            for node_id in self.cluster_nodes:
                if node_id != self.node_id:
                    await self._send_append_entries(node_id)
            
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
    
    async def _send_append_entries(self, node_id: str, entries: List[LogEntry] = None) -> bool:
        """Send AppendEntries RPC to a specific node."""
        try:
            if entries is None:
                entries = []
            
            # Get previous log info
            next_index = self.next_index.get(node_id, len(self.log) + 1)
            prev_log_index = next_index - 1
            prev_log_term = 0
            
            if prev_log_index > 0 and prev_log_index <= len(self.log):
                prev_log_term = self.log[prev_log_index - 1].term
            
            # Prepare entries to send
            entries_to_send = []
            if not entries:  # Heartbeat or catch-up
                if next_index <= len(self.log):
                    entries_to_send = self.log[next_index - 1:]
            else:
                entries_to_send = entries
            
            append_entries_msg = RaftMessage(
                message_type="append_entries",
                term=self.current_term,
                sender_id=self.node_id,
                data={
                    'leader_id': self.node_id,
                    'prev_log_index': prev_log_index,
                    'prev_log_term': prev_log_term,
                    'entries': [entry.to_dict() for entry in entries_to_send],
                    'leader_commit': self.commit_index
                }
            )
            
            # Send message
            agent_message = AgentMessage.create(
                sender_id=self.node_id,
                receiver_id=node_id,
                message_type="raft_append_entries",
                payload=append_entries_msg.to_dict()
            )
            
            # Send with timeout
            response = await self.nats_client.request(
                f"raft.{node_id}.append",
                agent_message,
                timeout=1.0
            )
            
            success = response.payload.get('success', False)
            response_term = response.payload.get('term', 0)
            
            # Handle response
            if response_term > self.current_term:
                await self._become_follower(response_term)
                return False
            
            if success:
                # Update indices
                if entries_to_send:
                    self.next_index[node_id] = prev_log_index + len(entries_to_send) + 1
                    self.match_index[node_id] = prev_log_index + len(entries_to_send)
            else:
                # Decrement next_index and retry
                self.next_index[node_id] = max(1, self.next_index[node_id] - 1)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending append entries to {node_id}: {e}")
            return False
    
    async def _handle_append_entries(self, message: AgentMessage) -> None:
        """Handle AppendEntries RPC."""
        try:
            raft_msg = RaftMessage(**message.payload)
            leader_id = raft_msg.data['leader_id']
            leader_term = raft_msg.term
            prev_log_index = raft_msg.data['prev_log_index']
            prev_log_term = raft_msg.data['prev_log_term']
            entries_data = raft_msg.data['entries']
            leader_commit = raft_msg.data['leader_commit']
            
            success = False
            
            # Convert entries
            entries = [LogEntry.from_dict(entry_data) for entry_data in entries_data]
            
            # Update term if necessary
            if leader_term > self.current_term:
                await self._become_follower(leader_term)
            
            # Reset election timer (received heartbeat)
            self.last_heartbeat_received = time.time()
            await self._reset_election_timer()
            
            # Set current leader
            self.current_leader = leader_id
            
            # Log consistency check
            if (prev_log_index == 0 or 
                (prev_log_index <= len(self.log) and 
                 self.log[prev_log_index - 1].term == prev_log_term)):
                
                success = True
                
                # Append new entries
                if entries:
                    # Remove conflicting entries
                    if prev_log_index < len(self.log):
                        self.log = self.log[:prev_log_index]
                    
                    # Append new entries
                    self.log.extend(entries)
                
                # Update commit index
                if leader_commit > self.commit_index:
                    self.commit_index = min(leader_commit, len(self.log))
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.node_id,
                receiver_id=message.sender_id,
                message_type="raft_append_response",
                payload={
                    'term': self.current_term,
                    'success': success
                },
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
            self.logger.debug(f"Handled append entries from {leader_id}: success={success}")
            
        except Exception as e:
            self.logger.error(f"Error handling append entries: {e}")
    
    async def _replicate_log_entry(self, entry: LogEntry) -> None:
        """Replicate log entry to all followers."""
        if self.state != NodeState.LEADER:
            return
        
        try:
            # Send to all followers
            replication_tasks = []
            for node_id in self.cluster_nodes:
                if node_id != self.node_id:
                    task = asyncio.create_task(self._send_append_entries(node_id, [entry]))
                    replication_tasks.append(task)
            
            if replication_tasks:
                # Wait for majority to respond
                responses = await asyncio.gather(*replication_tasks, return_exceptions=True)
                successful_replications = sum(1 for r in responses if r is True)
                
                # Check if we have majority
                majority = (len(self.cluster_nodes) + 1) // 2
                if successful_replications >= majority:
                    # Entry is committed
                    if entry.index > self.commit_index:
                        self.commit_index = entry.index
                        entry.committed = True
                        
                        self.logger.info(f"Entry {entry.index} committed with {successful_replications + 1} replicas")
            
        except Exception as e:
            self.logger.error(f"Error replicating log entry: {e}")
    
    async def _apply_committed_entries(self) -> None:
        """Apply committed log entries to state machine."""
        while not self.shutdown_event.is_set():
            try:
                # Apply entries up to commit_index
                while self.last_applied < self.commit_index:
                    self.last_applied += 1
                    
                    if self.last_applied <= len(self.log):
                        entry = self.log[self.last_applied - 1]
                        await self._apply_entry(entry)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error applying committed entries: {e}")
                await asyncio.sleep(1)
    
    async def _apply_entry(self, entry: LogEntry) -> None:
        """Apply a committed log entry to the state machine."""
        try:
            if entry.entry_type == LogEntryType.VALIDATION_REQUEST:
                await self._process_validation_request(entry)
            elif entry.entry_type == LogEntryType.VALIDATION_RESULT:
                await self._process_validation_result(entry)
            elif entry.entry_type == LogEntryType.CONFIGURATION_CHANGE:
                await self._process_configuration_change(entry)
            # NO_OP entries don't need processing
            
        except Exception as e:
            self.logger.error(f"Error applying entry {entry.index}: {e}")
    
    async def _process_validation_request(self, entry: LogEntry) -> None:
        """Process a validation request from the log."""
        try:
            validation_data = entry.data
            validation_id = validation_data['validation_id']
            
            # Simulate validation processing (would integrate with actual validation agents)
            result = {
                'validation_id': validation_id,
                'pattern_id': validation_data['pattern_id'],
                'status': 'completed',
                'consensus_decision': 'significant',  # Placeholder
                'confidence': 0.85,  # Placeholder
                'participating_nodes': list(self.cluster_nodes),
                'timestamp': datetime.utcnow().isoformat(),
                'term': entry.term,
                'log_index': entry.index
            }
            
            # Store result
            self.validation_results[validation_id] = result
            
            # Signal completion
            if validation_id in self.pending_validations:
                self.pending_validations[validation_id].set()
            
            self.logger.info(f"Processed validation request {validation_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing validation request: {e}")
    
    async def _process_validation_result(self, entry: LogEntry) -> None:
        """Process a validation result from the log."""
        # Results are processed when they're added to the log
        pass
    
    async def _process_configuration_change(self, entry: LogEntry) -> None:
        """Process cluster configuration change."""
        try:
            config_data = entry.data
            if config_data.get('operation') == 'add_node':
                new_node = config_data.get('node_id')
                if new_node:
                    self.cluster_nodes.add(new_node)
                    self.logger.info(f"Added node {new_node} to cluster")
            elif config_data.get('operation') == 'remove_node':
                removed_node = config_data.get('node_id')
                if removed_node:
                    self.cluster_nodes.discard(removed_node)
                    self.logger.info(f"Removed node {removed_node} from cluster")
            
        except Exception as e:
            self.logger.error(f"Error processing configuration change: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Leader heartbeat loop."""
        while not self.shutdown_event.is_set() and self.state == NodeState.LEADER:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval / 1000.0)  # Convert to seconds
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(1)
    
    async def _reset_election_timer(self) -> None:
        """Reset the election timeout timer."""
        if self.election_timer:
            self.election_timer.cancel()
        
        # Random timeout to avoid split votes
        timeout_ms = random.randint(*self.election_timeout_range)
        self.election_timer = asyncio.create_task(self._election_timeout(timeout_ms / 1000.0))
    
    async def _election_timeout(self, timeout_seconds: float) -> None:
        """Handle election timeout."""
        try:
            await asyncio.sleep(timeout_seconds)
            
            # Check if we haven't heard from leader recently
            if (time.time() - self.last_heartbeat_received > timeout_seconds and 
                self.state == NodeState.FOLLOWER):
                await self._become_candidate()
            
        except asyncio.CancelledError:
            pass  # Timer was cancelled
        except Exception as e:
            self.logger.error(f"Error in election timeout: {e}")
    
    async def _add_no_op_entry(self) -> None:
        """Add a no-op entry to commit previous terms' entries."""
        no_op_entry = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            entry_type=LogEntryType.NO_OP,
            data={'operation': 'no_op', 'leader_id': self.node_id}
        )
        
        self.log.append(no_op_entry)
        await self._replicate_log_entry(no_op_entry)
    
    async def _forward_to_leader(self, validation_entry: ValidationEntry) -> Optional[Dict[str, Any]]:
        """Forward validation request to current leader."""
        if not self.current_leader:
            return None
        
        try:
            # Send validation request to leader
            message = AgentMessage.create(
                sender_id=self.node_id,
                receiver_id=self.current_leader,
                message_type="raft_forward_validation",
                payload=validation_entry.to_dict()
            )
            
            response = await self.nats_client.request(
                f"raft.{self.current_leader}.validation",
                message,
                timeout=validation_entry.timeout_seconds
            )
            
            return response.payload
            
        except Exception as e:
            self.logger.error(f"Error forwarding to leader: {e}")
            return None
    
    def _is_log_up_to_date(self, candidate_last_index: int, candidate_last_term: int) -> bool:
        """Check if candidate's log is at least as up-to-date as ours."""
        if not self.log:
            return True
        
        our_last_term = self.log[-1].term
        our_last_index = len(self.log)
        
        # Candidate is more up-to-date if:
        # 1. Last term is greater, OR
        # 2. Same last term but longer log
        return (candidate_last_term > our_last_term or 
                (candidate_last_term == our_last_term and candidate_last_index >= our_last_index))
    
    async def _setup_subscriptions(self) -> None:
        """Setup NATS subscriptions for RAFT messages."""
        try:
            # Subscribe to vote requests
            await self.nats_client.subscribe(
                f"raft.{self.node_id}.vote",
                self._handle_request_vote
            )
            
            # Subscribe to append entries
            await self.nats_client.subscribe(
                f"raft.{self.node_id}.append",
                self._handle_append_entries
            )
            
            # Subscribe to validation forwards
            await self.nats_client.subscribe(
                f"raft.{self.node_id}.validation",
                self._handle_validation_forward
            )
            
            # Subscribe to cluster announcements
            await self.nats_client.subscribe(
                "raft.cluster.announce",
                self._handle_cluster_announcement
            )
            
            self.logger.info("Set up RAFT subscriptions")
            
        except Exception as e:
            self.logger.error(f"Failed to setup subscriptions: {e}")
    
    async def _handle_validation_forward(self, message: AgentMessage) -> None:
        """Handle validation request forwarded from follower."""
        try:
            validation_data = message.payload
            validation_entry = ValidationEntry(**validation_data)
            
            # Process if we're the leader
            if self.state == NodeState.LEADER:
                result = await self.submit_validation(validation_entry)
                
                # Send response
                response = AgentMessage.create(
                    sender_id=self.node_id,
                    receiver_id=message.sender_id,
                    message_type="raft_validation_response",
                    payload=result or {'error': 'Validation failed'},
                    correlation_id=message.correlation_id
                )
                
                if message.reply_to:
                    await self.nats_client.publish(message.reply_to, response)
            else:
                # Not the leader, send error
                error_response = AgentMessage.create(
                    sender_id=self.node_id,
                    receiver_id=message.sender_id,
                    message_type="raft_validation_response",
                    payload={'error': 'Not the leader', 'leader': self.current_leader},
                    correlation_id=message.correlation_id
                )
                
                if message.reply_to:
                    await self.nats_client.publish(message.reply_to, error_response)
            
        except Exception as e:
            self.logger.error(f"Error handling validation forward: {e}")
    
    async def _cluster_discovery(self) -> None:
        """Discover other nodes in the cluster."""
        while not self.shutdown_event.is_set():
            try:
                # Announce our presence
                announcement = AgentMessage.create(
                    sender_id=self.node_id,
                    message_type="raft_node_announcement",
                    payload={
                        'node_id': self.node_id,
                        'term': self.current_term,
                        'state': self.state.value,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                await self.nats_client.publish("raft.cluster.announce", announcement)
                
                await asyncio.sleep(10)  # Announce every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in cluster discovery: {e}")
                await asyncio.sleep(5)
    
    async def _handle_cluster_announcement(self, message: AgentMessage) -> None:
        """Handle cluster node announcements."""
        try:
            if message.sender_id != self.node_id:
                node_id = message.payload.get('node_id')
                if node_id:
                    self.cluster_nodes.add(node_id)
                    self.logger.debug(f"Discovered node {node_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling cluster announcement: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current RAFT node status."""
        return {
            'node_id': self.node_id,
            'state': self.state.value,
            'current_term': self.current_term,
            'current_leader': self.current_leader,
            'voted_for': self.voted_for,
            'log_length': len(self.log),
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'cluster_size': len(self.cluster_nodes) + 1,
            'cluster_nodes': list(self.cluster_nodes),
            'pending_validations': len(self.pending_validations),
            'validation_results': len(self.validation_results)
        }


class RaftConsensus:
    """
    High-level RAFT consensus interface for pattern validation.
    """
    
    def __init__(self, node_id: str, nats_client: NATSClient):
        """Initialize RAFT consensus system."""
        self.node_id = node_id
        self.nats_client = nats_client
        self.raft_node = RaftNode(node_id, nats_client)
        self.logger = logging.getLogger(f"raft_consensus.{node_id}")
    
    async def start(self) -> None:
        """Start RAFT consensus."""
        await self.raft_node.start()
        self.logger.info(f"RAFT consensus started for node {self.node_id}")
    
    async def stop(self) -> None:
        """Stop RAFT consensus."""
        await self.raft_node.stop()
        self.logger.info(f"RAFT consensus stopped for node {self.node_id}")
    
    async def validate_pattern(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any],
        statistical_results: List[Dict[str, Any]],
        timeout_seconds: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Request consensus-based pattern validation.
        
        Args:
            pattern_id: Pattern identifier
            pattern_data: Pattern data for validation
            statistical_results: Existing statistical validation results
            timeout_seconds: Validation timeout
            
        Returns:
            Consensus validation result
        """
        validation_entry = ValidationEntry(
            validation_id=str(uuid.uuid4()),
            pattern_id=pattern_id,
            pattern_data=pattern_data,
            statistical_results=statistical_results,
            requester_id=self.node_id,
            timeout_seconds=timeout_seconds
        )
        
        return await self.raft_node.submit_validation(validation_entry)
    
    def get_status(self) -> Dict[str, Any]:
        """Get consensus system status."""
        return self.raft_node.get_status()