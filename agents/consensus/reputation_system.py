"""
A2A World Platform - Reputation System

Agent reputation tracking and trust management for consensus-based validation.
Tracks validation accuracy, reliability, and behavior to weight voting contributions.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

from agents.core.messaging import AgentMessage, NATSClient


class ReputationMetric(Enum):
    """Types of reputation metrics tracked."""
    VALIDATION_ACCURACY = "validation_accuracy"
    RESPONSE_TIME = "response_time"
    AVAILABILITY = "availability"
    CONSENSUS_PARTICIPATION = "consensus_participation"
    STATISTICAL_QUALITY = "statistical_quality"
    PEER_RATING = "peer_rating"


@dataclass
class ValidationOutcome:
    """Record of a validation outcome for reputation tracking."""
    validation_id: str
    agent_id: str
    pattern_id: str
    prediction: str  # "significant", "not_significant", "uncertain"
    confidence: float
    actual_outcome: Optional[str] = None  # Ground truth if available
    peer_consensus: Optional[str] = None  # What consensus decided
    statistical_evidence_quality: float = 0.0  # 0-1 score
    response_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def is_correct_prediction(self) -> Optional[bool]:
        """Check if prediction matched actual outcome."""
        if self.actual_outcome:
            return self.prediction == self.actual_outcome
        elif self.peer_consensus:
            return self.prediction == self.peer_consensus
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'validation_id': self.validation_id,
            'agent_id': self.agent_id,
            'pattern_id': self.pattern_id,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'actual_outcome': self.actual_outcome,
            'peer_consensus': self.peer_consensus,
            'statistical_evidence_quality': self.statistical_evidence_quality,
            'response_time_seconds': self.response_time_seconds,
            'timestamp': self.timestamp,
            'is_correct': self.is_correct_prediction()
        }


@dataclass
class ReputationScore:
    """Agent reputation score components."""
    agent_id: str
    overall_score: float  # 0.0 to 1.0
    accuracy_score: float
    reliability_score: float
    timeliness_score: float
    participation_score: float
    quality_score: float
    peer_score: float
    total_validations: int
    correct_predictions: int
    consensus_agreements: int
    average_response_time: float
    uptime_percentage: float
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'overall_score': self.overall_score,
            'accuracy_score': self.accuracy_score,
            'reliability_score': self.reliability_score,
            'timeliness_score': self.timeliness_score,
            'participation_score': self.participation_score,
            'quality_score': self.quality_score,
            'peer_score': self.peer_score,
            'total_validations': self.total_validations,
            'correct_predictions': self.correct_predictions,
            'consensus_agreements': self.consensus_agreements,
            'average_response_time': self.average_response_time,
            'uptime_percentage': self.uptime_percentage,
            'last_updated': self.last_updated
        }


class AgentReputationTracker:
    """
    Tracks and updates reputation for a single agent.
    """
    
    def __init__(self, agent_id: str, decay_rate: float = 0.95, min_validations: int = 5):
        """
        Initialize agent reputation tracker.
        
        Args:
            agent_id: Agent identifier
            decay_rate: Rate at which old reputation decays (0.0 to 1.0)
            min_validations: Minimum validations before reputation is considered stable
        """
        self.agent_id = agent_id
        self.decay_rate = decay_rate
        self.min_validations = min_validations
        
        # Validation history
        self.validation_outcomes: List[ValidationOutcome] = []
        self.response_times: List[float] = []
        self.availability_events: List[Tuple[datetime, bool]] = []  # (timestamp, available)
        
        # Peer ratings
        self.peer_ratings: Dict[str, float] = {}  # peer_id -> rating (0.0 to 1.0)
        
        # Current reputation score
        self.reputation_score = ReputationScore(
            agent_id=agent_id,
            overall_score=0.5,  # Start neutral
            accuracy_score=0.5,
            reliability_score=0.5,
            timeliness_score=0.5,
            participation_score=0.5,
            quality_score=0.5,
            peer_score=0.5,
            total_validations=0,
            correct_predictions=0,
            consensus_agreements=0,
            average_response_time=0.0,
            uptime_percentage=100.0
        )
        
        self.logger = logging.getLogger(f"reputation.{agent_id}")
    
    def add_validation_outcome(self, outcome: ValidationOutcome) -> None:
        """Add a validation outcome to the history."""
        self.validation_outcomes.append(outcome)
        
        # Limit history size
        if len(self.validation_outcomes) > 1000:
            self.validation_outcomes = self.validation_outcomes[-1000:]
        
        # Update scores
        self._update_reputation_scores()
    
    def add_response_time(self, response_time: float) -> None:
        """Record response time for an interaction."""
        self.response_times.append(response_time)
        
        # Limit history size
        if len(self.response_times) > 500:
            self.response_times = self.response_times[-500:]
        
        self._update_timeliness_score()
    
    def record_availability(self, available: bool) -> None:
        """Record availability status."""
        self.availability_events.append((datetime.utcnow(), available))
        
        # Clean old events (keep last 30 days)
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.availability_events = [
            (ts, avail) for ts, avail in self.availability_events if ts > cutoff
        ]
        
        self._update_reliability_score()
    
    def add_peer_rating(self, peer_id: str, rating: float) -> None:
        """Add peer rating for this agent."""
        if 0.0 <= rating <= 1.0:
            self.peer_ratings[peer_id] = rating
            self._update_peer_score()
        else:
            raise ValueError(f"Rating must be between 0.0 and 1.0, got {rating}")
    
    def get_reputation_score(self) -> ReputationScore:
        """Get current reputation score."""
        return self.reputation_score
    
    def get_voting_weight(self, base_weight: float = 1.0) -> float:
        """Calculate voting weight based on reputation."""
        if self.reputation_score.total_validations < self.min_validations:
            # New agents get reduced weight until they build reputation
            return base_weight * 0.5
        
        # Weight based on overall score with minimum threshold
        weight_multiplier = max(0.1, self.reputation_score.overall_score)
        return base_weight * weight_multiplier
    
    def _update_reputation_scores(self) -> None:
        """Update all reputation score components."""
        self._update_accuracy_score()
        self._update_participation_score()
        self._update_quality_score()
        self._update_overall_score()
        
        self.reputation_score.last_updated = datetime.utcnow().isoformat()
    
    def _update_accuracy_score(self) -> None:
        """Update accuracy score based on validation outcomes."""
        if not self.validation_outcomes:
            return
        
        # Calculate accuracy metrics
        total_validations = len(self.validation_outcomes)
        correct_predictions = 0
        consensus_agreements = 0
        
        for outcome in self.validation_outcomes:
            if outcome.is_correct_prediction() is True:
                correct_predictions += 1
            
            if outcome.peer_consensus and outcome.prediction == outcome.peer_consensus:
                consensus_agreements += 1
        
        # Update reputation score
        self.reputation_score.total_validations = total_validations
        self.reputation_score.correct_predictions = correct_predictions
        self.reputation_score.consensus_agreements = consensus_agreements
        
        if total_validations > 0:
            accuracy_rate = correct_predictions / total_validations
            consensus_rate = consensus_agreements / total_validations
            
            # Weighted combination of accuracy and consensus agreement
            self.reputation_score.accuracy_score = (0.7 * accuracy_rate + 0.3 * consensus_rate)
        else:
            self.reputation_score.accuracy_score = 0.5  # Neutral for new agents
    
    def _update_reliability_score(self) -> None:
        """Update reliability score based on availability."""
        if not self.availability_events:
            return
        
        # Calculate uptime percentage over last 30 days
        total_time = 0
        uptime = 0
        
        for i in range(len(self.availability_events) - 1):
            start_time, available = self.availability_events[i]
            end_time, _ = self.availability_events[i + 1]
            
            duration = (end_time - start_time).total_seconds()
            total_time += duration
            
            if available:
                uptime += duration
        
        if total_time > 0:
            uptime_percentage = (uptime / total_time) * 100
            self.reputation_score.uptime_percentage = uptime_percentage
            
            # Convert to 0-1 score (95% uptime = 1.0, 50% uptime = 0.0)
            self.reputation_score.reliability_score = max(0.0, (uptime_percentage - 50) / 45)
    
    def _update_timeliness_score(self) -> None:
        """Update timeliness score based on response times."""
        if not self.response_times:
            return
        
        avg_response_time = statistics.mean(self.response_times)
        self.reputation_score.average_response_time = avg_response_time
        
        # Good response time is < 5 seconds, poor is > 30 seconds
        if avg_response_time <= 5.0:
            self.reputation_score.timeliness_score = 1.0
        elif avg_response_time >= 30.0:
            self.reputation_score.timeliness_score = 0.0
        else:
            # Linear interpolation between 5 and 30 seconds
            self.reputation_score.timeliness_score = 1.0 - ((avg_response_time - 5.0) / 25.0)
    
    def _update_participation_score(self) -> None:
        """Update participation score based on validation activity."""
        if not self.validation_outcomes:
            self.reputation_score.participation_score = 0.0
            return
        
        # Calculate participation over recent time window
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_validations = [
            outcome for outcome in self.validation_outcomes
            if datetime.fromisoformat(outcome.timestamp) > recent_cutoff
        ]
        
        # Score based on recent activity (1+ per day is good)
        daily_rate = len(recent_validations) / 7.0
        self.reputation_score.participation_score = min(1.0, daily_rate / 2.0)  # 2 per day = 1.0
    
    def _update_quality_score(self) -> None:
        """Update quality score based on statistical evidence quality."""
        if not self.validation_outcomes:
            return
        
        quality_scores = [
            outcome.statistical_evidence_quality 
            for outcome in self.validation_outcomes 
            if outcome.statistical_evidence_quality > 0
        ]
        
        if quality_scores:
            self.reputation_score.quality_score = statistics.mean(quality_scores)
        else:
            self.reputation_score.quality_score = 0.5
    
    def _update_peer_score(self) -> None:
        """Update peer score based on peer ratings."""
        if not self.peer_ratings:
            self.reputation_score.peer_score = 0.5  # Neutral
            return
        
        # Average of all peer ratings
        self.reputation_score.peer_score = statistics.mean(self.peer_ratings.values())
    
    def _update_overall_score(self) -> None:
        """Update overall reputation score as weighted combination."""
        # Weights for different score components
        weights = {
            'accuracy': 0.30,
            'reliability': 0.20,
            'timeliness': 0.15,
            'participation': 0.15,
            'quality': 0.10,
            'peer': 0.10
        }
        
        # Calculate weighted average
        overall_score = (
            weights['accuracy'] * self.reputation_score.accuracy_score +
            weights['reliability'] * self.reputation_score.reliability_score +
            weights['timeliness'] * self.reputation_score.timeliness_score +
            weights['participation'] * self.reputation_score.participation_score +
            weights['quality'] * self.reputation_score.quality_score +
            weights['peer'] * self.reputation_score.peer_score
        )
        
        # Apply decay to previous score for stability
        if hasattr(self.reputation_score, 'overall_score'):
            overall_score = (self.decay_rate * self.reputation_score.overall_score + 
                           (1 - self.decay_rate) * overall_score)
        
        self.reputation_score.overall_score = max(0.0, min(1.0, overall_score))


class ReputationSystem:
    """
    Central reputation management system for all agents in the consensus network.
    """
    
    def __init__(self, nats_client: Optional[NATSClient] = None):
        """
        Initialize reputation system.
        
        Args:
            nats_client: NATS client for distributed updates
        """
        self.nats_client = nats_client
        self.agent_trackers: Dict[str, AgentReputationTracker] = {}
        self.system_metrics: Dict[str, Any] = {}
        
        self.logger = logging.getLogger("reputation_system")
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the reputation system."""
        self.logger.info("Starting reputation system")
        
        if self.nats_client:
            # Setup message subscriptions
            await self._setup_subscriptions()
        
        # Start background tasks
        self.background_tasks.extend([
            asyncio.create_task(self._periodic_updates()),
            asyncio.create_task(self._system_metrics_calculation())
        ])
        
        self.logger.info("Reputation system started")
    
    async def stop(self) -> None:
        """Stop the reputation system."""
        self.logger.info("Stopping reputation system")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Reputation system stopped")
    
    def register_agent(self, agent_id: str) -> AgentReputationTracker:
        """Register a new agent in the reputation system."""
        if agent_id not in self.agent_trackers:
            self.agent_trackers[agent_id] = AgentReputationTracker(agent_id)
            self.logger.info(f"Registered agent {agent_id} in reputation system")
        
        return self.agent_trackers[agent_id]
    
    def get_agent_reputation(self, agent_id: str) -> Optional[ReputationScore]:
        """Get reputation score for an agent."""
        tracker = self.agent_trackers.get(agent_id)
        return tracker.get_reputation_score() if tracker else None
    
    def get_voting_weight(self, agent_id: str, base_weight: float = 1.0) -> float:
        """Get voting weight for an agent based on reputation."""
        tracker = self.agent_trackers.get(agent_id)
        if tracker:
            return tracker.get_voting_weight(base_weight)
        else:
            # Unknown agents get minimal weight
            return base_weight * 0.1
    
    def record_validation_outcome(self, outcome: ValidationOutcome) -> None:
        """Record validation outcome for reputation tracking."""
        tracker = self.agent_trackers.get(outcome.agent_id)
        if not tracker:
            tracker = self.register_agent(outcome.agent_id)
        
        tracker.add_validation_outcome(outcome)
        
        # Broadcast update if connected
        if self.nats_client:
            asyncio.create_task(self._broadcast_reputation_update(outcome.agent_id))
    
    def record_response_time(self, agent_id: str, response_time: float) -> None:
        """Record response time for an agent."""
        tracker = self.agent_trackers.get(agent_id)
        if not tracker:
            tracker = self.register_agent(agent_id)
        
        tracker.add_response_time(response_time)
    
    def record_availability(self, agent_id: str, available: bool) -> None:
        """Record availability event for an agent."""
        tracker = self.agent_trackers.get(agent_id)
        if not tracker:
            tracker = self.register_agent(agent_id)
        
        tracker.record_availability(available)
    
    def add_peer_rating(self, agent_id: str, peer_id: str, rating: float) -> None:
        """Add peer rating for an agent."""
        tracker = self.agent_trackers.get(agent_id)
        if not tracker:
            tracker = self.register_agent(agent_id)
        
        tracker.add_peer_rating(peer_id, rating)
        
        self.logger.info(f"Added peer rating: {peer_id} rated {agent_id} as {rating}")
    
    def get_top_agents(self, limit: int = 10) -> List[Tuple[str, ReputationScore]]:
        """Get top agents by reputation score."""
        agent_scores = [
            (agent_id, tracker.get_reputation_score())
            for agent_id, tracker in self.agent_trackers.items()
        ]
        
        # Sort by overall score descending
        agent_scores.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return agent_scores[:limit]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide reputation statistics."""
        if not self.agent_trackers:
            return {'total_agents': 0}
        
        scores = [tracker.get_reputation_score() for tracker in self.agent_trackers.values()]
        overall_scores = [score.overall_score for score in scores]
        accuracy_scores = [score.accuracy_score for score in scores]
        
        return {
            'total_agents': len(self.agent_trackers),
            'average_reputation': statistics.mean(overall_scores),
            'median_reputation': statistics.median(overall_scores),
            'reputation_std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            'average_accuracy': statistics.mean(accuracy_scores),
            'high_reputation_agents': len([s for s in overall_scores if s > 0.8]),
            'low_reputation_agents': len([s for s in overall_scores if s < 0.3]),
            'total_validations': sum(score.total_validations for score in scores),
            'total_correct_predictions': sum(score.correct_predictions for score in scores),
            'system_accuracy': sum(score.correct_predictions for score in scores) / max(1, sum(score.total_validations for score in scores))
        }
    
    async def _setup_subscriptions(self) -> None:
        """Setup NATS subscriptions for reputation updates."""
        if not self.nats_client:
            return
        
        try:
            # Subscribe to validation outcomes
            await self.nats_client.subscribe(
                "reputation.validation_outcome",
                self._handle_validation_outcome_message
            )
            
            # Subscribe to peer ratings
            await self.nats_client.subscribe(
                "reputation.peer_rating",
                self._handle_peer_rating_message
            )
            
            # Subscribe to availability updates
            await self.nats_client.subscribe(
                "reputation.availability",
                self._handle_availability_message
            )
            
            self.logger.info("Set up reputation system subscriptions")
            
        except Exception as e:
            self.logger.error(f"Failed to setup subscriptions: {e}")
    
    async def _handle_validation_outcome_message(self, message: AgentMessage) -> None:
        """Handle validation outcome messages."""
        try:
            outcome_data = message.payload
            outcome = ValidationOutcome(**outcome_data)
            self.record_validation_outcome(outcome)
            
        except Exception as e:
            self.logger.error(f"Error handling validation outcome message: {e}")
    
    async def _handle_peer_rating_message(self, message: AgentMessage) -> None:
        """Handle peer rating messages."""
        try:
            payload = message.payload
            agent_id = payload['agent_id']
            peer_id = payload['peer_id']
            rating = payload['rating']
            
            self.add_peer_rating(agent_id, peer_id, rating)
            
        except Exception as e:
            self.logger.error(f"Error handling peer rating message: {e}")
    
    async def _handle_availability_message(self, message: AgentMessage) -> None:
        """Handle availability update messages."""
        try:
            payload = message.payload
            agent_id = payload['agent_id']
            available = payload['available']
            
            self.record_availability(agent_id, available)
            
        except Exception as e:
            self.logger.error(f"Error handling availability message: {e}")
    
    async def _broadcast_reputation_update(self, agent_id: str) -> None:
        """Broadcast reputation update to network."""
        if not self.nats_client:
            return
        
        try:
            reputation = self.get_agent_reputation(agent_id)
            if reputation:
                message = AgentMessage.create(
                    sender_id="reputation_system",
                    message_type="reputation_update",
                    payload={
                        'agent_id': agent_id,
                        'reputation_score': reputation.to_dict(),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                await self.nats_client.publish("reputation.updates", message)
                
        except Exception as e:
            self.logger.error(f"Error broadcasting reputation update: {e}")
    
    async def _periodic_updates(self) -> None:
        """Perform periodic reputation updates and cleanup."""
        while not self.shutdown_event.is_set():
            try:
                # Update all agent reputations
                for tracker in self.agent_trackers.values():
                    tracker._update_reputation_scores()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(60)
    
    async def _system_metrics_calculation(self) -> None:
        """Calculate system-wide metrics periodically."""
        while not self.shutdown_event.is_set():
            try:
                self.system_metrics = self.get_system_stats()
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error calculating system metrics: {e}")
                await asyncio.sleep(120)
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old reputation data."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            for tracker in self.agent_trackers.values():
                # Clean old validation outcomes
                tracker.validation_outcomes = [
                    outcome for outcome in tracker.validation_outcomes
                    if datetime.fromisoformat(outcome.timestamp) > cutoff_date
                ]
                
                # Clean old response times (keep last 500)
                if len(tracker.response_times) > 500:
                    tracker.response_times = tracker.response_times[-500:]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")


# Global reputation system instance
_global_reputation_system: Optional[ReputationSystem] = None


def get_reputation_system(nats_client: Optional[NATSClient] = None) -> ReputationSystem:
    """Get or create global reputation system instance."""
    global _global_reputation_system
    
    if _global_reputation_system is None:
        _global_reputation_system = ReputationSystem(nats_client)
    
    return _global_reputation_system


async def cleanup_reputation_system():
    """Cleanup global reputation system."""
    global _global_reputation_system
    
    if _global_reputation_system:
        await _global_reputation_system.stop()
        _global_reputation_system = None