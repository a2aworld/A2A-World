"""
A2A World Platform - Voting Mechanisms

Various voting algorithms and strategies for consensus-based pattern validation.
Implements different voting mechanisms including majority, weighted, threshold, and quorum voting.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import statistics


class VoteType(Enum):
    """Types of votes in pattern validation."""
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    UNCERTAIN = "uncertain"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """Individual vote cast by a validation agent."""
    agent_id: str
    pattern_id: str
    vote: VoteType
    confidence: float  # 0.0 to 1.0
    statistical_evidence: Dict[str, Any]
    reasoning: str
    timestamp: str
    weight: float = 1.0  # Agent's voting weight
    reputation_score: float = 1.0  # Agent's reputation
    
    def __post_init__(self):
        """Validate vote parameters."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.weight < 0.0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if self.reputation_score < 0.0:
            raise ValueError(f"Reputation score must be non-negative, got {self.reputation_score}")
    
    def effective_weight(self) -> float:
        """Calculate effective voting weight including reputation."""
        return self.weight * self.reputation_score * self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'pattern_id': self.pattern_id,
            'vote': self.vote.value,
            'confidence': self.confidence,
            'statistical_evidence': self.statistical_evidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp,
            'weight': self.weight,
            'reputation_score': self.reputation_score,
            'effective_weight': self.effective_weight()
        }


@dataclass
class VotingResult:
    """Result of a voting process."""
    pattern_id: str
    decision: VoteType
    confidence: float
    vote_breakdown: Dict[VoteType, int]
    weighted_breakdown: Dict[VoteType, float]
    participating_agents: List[str]
    total_votes: int
    total_weight: float
    consensus_achieved: bool
    voting_method: str
    timestamp: str
    detailed_votes: List[Vote]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'decision': self.decision.value,
            'confidence': self.confidence,
            'vote_breakdown': {k.value: v for k, v in self.vote_breakdown.items()},
            'weighted_breakdown': {k.value: v for k, v in self.weighted_breakdown.items()},
            'participating_agents': self.participating_agents,
            'total_votes': self.total_votes,
            'total_weight': self.total_weight,
            'consensus_achieved': self.consensus_achieved,
            'voting_method': self.voting_method,
            'timestamp': self.timestamp,
            'detailed_votes': [vote.to_dict() for vote in self.detailed_votes]
        }


class VotingMechanism(ABC):
    """
    Abstract base class for voting mechanisms.
    """
    
    def __init__(self, name: str):
        """
        Initialize voting mechanism.
        
        Args:
            name: Name of the voting mechanism
        """
        self.name = name
        self.logger = logging.getLogger(f"voting.{name}")
    
    @abstractmethod
    def compute_result(self, votes: List[Vote], pattern_id: str) -> VotingResult:
        """
        Compute voting result from individual votes.
        
        Args:
            votes: List of individual votes
            pattern_id: Pattern being voted on
            
        Returns:
            Voting result with decision and metadata
        """
        pass
    
    def _calculate_vote_breakdown(self, votes: List[Vote]) -> Tuple[Dict[VoteType, int], Dict[VoteType, float]]:
        """Calculate vote counts and weighted counts."""
        vote_counts = {vote_type: 0 for vote_type in VoteType}
        weighted_counts = {vote_type: 0.0 for vote_type in VoteType}
        
        for vote in votes:
            vote_counts[vote.vote] += 1
            weighted_counts[vote.vote] += vote.effective_weight()
        
        return vote_counts, weighted_counts
    
    def _calculate_confidence(self, votes: List[Vote], winning_vote: VoteType) -> float:
        """Calculate overall confidence for the winning vote."""
        winning_votes = [v for v in votes if v.vote == winning_vote]
        if not winning_votes:
            return 0.0
        
        # Weighted average of confidence scores
        total_weight = sum(v.effective_weight() for v in winning_votes)
        if total_weight == 0:
            return 0.0
        
        weighted_confidence = sum(v.confidence * v.effective_weight() for v in winning_votes)
        return weighted_confidence / total_weight


class MajorityVoting(VotingMechanism):
    """
    Simple majority voting mechanism.
    Decision is made by the vote type with the most votes.
    """
    
    def __init__(self, require_majority: bool = True):
        """
        Initialize majority voting.
        
        Args:
            require_majority: If True, requires >50% for decision; if False, plurality wins
        """
        super().__init__("majority_voting")
        self.require_majority = require_majority
    
    def compute_result(self, votes: List[Vote], pattern_id: str) -> VotingResult:
        """Compute majority voting result."""
        if not votes:
            return self._empty_result(pattern_id)
        
        # Calculate vote breakdown
        vote_counts, weighted_counts = self._calculate_vote_breakdown(votes)
        
        # Find winning vote type
        max_votes = max(vote_counts.values())
        winning_types = [vote_type for vote_type, count in vote_counts.items() if count == max_votes]
        
        # Handle ties
        if len(winning_types) > 1:
            # Use weighted voting to break ties
            winning_type = max(winning_types, key=lambda vt: weighted_counts[vt])
        else:
            winning_type = winning_types[0]
        
        # Check if majority/plurality requirement met
        total_votes = len(votes)
        if self.require_majority:
            consensus_achieved = vote_counts[winning_type] > total_votes / 2
        else:
            consensus_achieved = vote_counts[winning_type] > 0
        
        # If no consensus, default to uncertain
        if not consensus_achieved:
            winning_type = VoteType.UNCERTAIN
        
        # Calculate confidence
        confidence = self._calculate_confidence(votes, winning_type)
        
        return VotingResult(
            pattern_id=pattern_id,
            decision=winning_type,
            confidence=confidence,
            vote_breakdown=vote_counts,
            weighted_breakdown=weighted_counts,
            participating_agents=[v.agent_id for v in votes],
            total_votes=total_votes,
            total_weight=sum(v.effective_weight() for v in votes),
            consensus_achieved=consensus_achieved,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=votes
        )
    
    def _empty_result(self, pattern_id: str) -> VotingResult:
        """Create empty result when no votes."""
        return VotingResult(
            pattern_id=pattern_id,
            decision=VoteType.UNCERTAIN,
            confidence=0.0,
            vote_breakdown={vote_type: 0 for vote_type in VoteType},
            weighted_breakdown={vote_type: 0.0 for vote_type in VoteType},
            participating_agents=[],
            total_votes=0,
            total_weight=0.0,
            consensus_achieved=False,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=[]
        )


class WeightedVoting(VotingMechanism):
    """
    Weighted voting mechanism where agents have different voting weights.
    Decision is based on total weighted votes.
    """
    
    def __init__(self, weight_threshold: float = 0.5):
        """
        Initialize weighted voting.
        
        Args:
            weight_threshold: Minimum proportion of total weight needed for decision
        """
        super().__init__("weighted_voting")
        self.weight_threshold = weight_threshold
    
    def compute_result(self, votes: List[Vote], pattern_id: str) -> VotingResult:
        """Compute weighted voting result."""
        if not votes:
            return self._empty_result(pattern_id)
        
        # Calculate vote breakdown
        vote_counts, weighted_counts = self._calculate_vote_breakdown(votes)
        
        # Find winning vote type by weight
        total_weight = sum(weighted_counts.values())
        if total_weight == 0:
            return self._empty_result(pattern_id)
        
        # Calculate proportions
        weight_proportions = {vt: weight / total_weight for vt, weight in weighted_counts.items()}
        
        # Find winner
        winning_type = max(weight_proportions.keys(), key=lambda vt: weight_proportions[vt])
        
        # Check threshold
        consensus_achieved = weight_proportions[winning_type] >= self.weight_threshold
        
        # If no consensus, default to uncertain
        if not consensus_achieved:
            winning_type = VoteType.UNCERTAIN
        
        # Calculate confidence
        confidence = self._calculate_confidence(votes, winning_type)
        
        return VotingResult(
            pattern_id=pattern_id,
            decision=winning_type,
            confidence=confidence,
            vote_breakdown=vote_counts,
            weighted_breakdown=weighted_counts,
            participating_agents=[v.agent_id for v in votes],
            total_votes=len(votes),
            total_weight=total_weight,
            consensus_achieved=consensus_achieved,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=votes
        )
    
    def _empty_result(self, pattern_id: str) -> VotingResult:
        """Create empty result when no votes."""
        return VotingResult(
            pattern_id=pattern_id,
            decision=VoteType.UNCERTAIN,
            confidence=0.0,
            vote_breakdown={vote_type: 0 for vote_type in VoteType},
            weighted_breakdown={vote_type: 0.0 for vote_type in VoteType},
            participating_agents=[],
            total_votes=0,
            total_weight=0.0,
            consensus_achieved=False,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=[]
        )


class ThresholdVoting(VotingMechanism):
    """
    Threshold voting mechanism where a vote type must exceed a threshold.
    Useful for requiring strong consensus on pattern significance.
    """
    
    def __init__(
        self,
        significance_threshold: float = 0.67,
        confidence_threshold: float = 0.75,
        use_weighted: bool = True
    ):
        """
        Initialize threshold voting.
        
        Args:
            significance_threshold: Minimum proportion needed to declare significant
            confidence_threshold: Minimum average confidence needed
            use_weighted: Whether to use weighted votes or simple counts
        """
        super().__init__("threshold_voting")
        self.significance_threshold = significance_threshold
        self.confidence_threshold = confidence_threshold
        self.use_weighted = use_weighted
    
    def compute_result(self, votes: List[Vote], pattern_id: str) -> VotingResult:
        """Compute threshold voting result."""
        if not votes:
            return self._empty_result(pattern_id)
        
        # Calculate vote breakdown
        vote_counts, weighted_counts = self._calculate_vote_breakdown(votes)
        
        # Choose counting method
        if self.use_weighted:
            counts = weighted_counts
            total = sum(weighted_counts.values())
        else:
            counts = {vt: float(count) for vt, count in vote_counts.items()}
            total = float(len(votes))
        
        if total == 0:
            return self._empty_result(pattern_id)
        
        # Calculate proportions
        proportions = {vt: count / total for vt, count in counts.items()}
        
        # Apply threshold logic
        significant_proportion = proportions[VoteType.SIGNIFICANT]
        not_significant_proportion = proportions[VoteType.NOT_SIGNIFICANT]
        
        # Check confidence threshold
        significant_votes = [v for v in votes if v.vote == VoteType.SIGNIFICANT]
        not_significant_votes = [v for v in votes if v.vote == VoteType.NOT_SIGNIFICANT]
        
        significant_avg_conf = statistics.mean([v.confidence for v in significant_votes]) if significant_votes else 0.0
        not_significant_avg_conf = statistics.mean([v.confidence for v in not_significant_votes]) if not_significant_votes else 0.0
        
        # Decision logic
        if (significant_proportion >= self.significance_threshold and 
            significant_avg_conf >= self.confidence_threshold):
            decision = VoteType.SIGNIFICANT
            confidence = significant_avg_conf
            consensus_achieved = True
        elif (not_significant_proportion >= self.significance_threshold and
              not_significant_avg_conf >= self.confidence_threshold):
            decision = VoteType.NOT_SIGNIFICANT
            confidence = not_significant_avg_conf
            consensus_achieved = True
        else:
            decision = VoteType.UNCERTAIN
            confidence = max(significant_avg_conf, not_significant_avg_conf) if votes else 0.0
            consensus_achieved = False
        
        return VotingResult(
            pattern_id=pattern_id,
            decision=decision,
            confidence=confidence,
            vote_breakdown=vote_counts,
            weighted_breakdown=weighted_counts,
            participating_agents=[v.agent_id for v in votes],
            total_votes=len(votes),
            total_weight=sum(v.effective_weight() for v in votes),
            consensus_achieved=consensus_achieved,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=votes
        )
    
    def _empty_result(self, pattern_id: str) -> VotingResult:
        """Create empty result when no votes."""
        return VotingResult(
            pattern_id=pattern_id,
            decision=VoteType.UNCERTAIN,
            confidence=0.0,
            vote_breakdown={vote_type: 0 for vote_type in VoteType},
            weighted_breakdown={vote_type: 0.0 for vote_type in VoteType},
            participating_agents=[],
            total_votes=0,
            total_weight=0.0,
            consensus_achieved=False,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=[]
        )


class QuorumVoting(VotingMechanism):
    """
    Quorum-based voting mechanism requiring minimum participation.
    Ensures decisions are made with sufficient agent involvement.
    """
    
    def __init__(
        self,
        min_participants: int = 3,
        min_weight: float = 0.6,
        fallback_mechanism: Optional[VotingMechanism] = None
    ):
        """
        Initialize quorum voting.
        
        Args:
            min_participants: Minimum number of participating agents
            min_weight: Minimum total voting weight required
            fallback_mechanism: Voting mechanism to use if quorum not met
        """
        super().__init__("quorum_voting")
        self.min_participants = min_participants
        self.min_weight = min_weight
        self.fallback_mechanism = fallback_mechanism or MajorityVoting(require_majority=False)
    
    def compute_result(self, votes: List[Vote], pattern_id: str) -> VotingResult:
        """Compute quorum voting result."""
        if not votes:
            return self._empty_result(pattern_id)
        
        # Check quorum requirements
        total_participants = len(votes)
        total_weight = sum(v.effective_weight() for v in votes)
        
        quorum_met = (total_participants >= self.min_participants and 
                     total_weight >= self.min_weight)
        
        if not quorum_met:
            # Use fallback mechanism but mark as no consensus
            result = self.fallback_mechanism.compute_result(votes, pattern_id)
            result.consensus_achieved = False
            result.voting_method = f"{self.name}_no_quorum_{result.voting_method}"
            return result
        
        # Quorum met, proceed with weighted voting
        vote_counts, weighted_counts = self._calculate_vote_breakdown(votes)
        
        # Find winning vote type by weight
        if total_weight == 0:
            return self._empty_result(pattern_id)
        
        # Calculate proportions
        weight_proportions = {vt: weight / total_weight for vt, weight in weighted_counts.items()}
        
        # Find winner (simple plurality with quorum)
        winning_type = max(weight_proportions.keys(), key=lambda vt: weight_proportions[vt])
        
        # Calculate confidence
        confidence = self._calculate_confidence(votes, winning_type)
        
        return VotingResult(
            pattern_id=pattern_id,
            decision=winning_type,
            confidence=confidence,
            vote_breakdown=vote_counts,
            weighted_breakdown=weighted_counts,
            participating_agents=[v.agent_id for v in votes],
            total_votes=total_participants,
            total_weight=total_weight,
            consensus_achieved=True,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=votes
        )
    
    def _empty_result(self, pattern_id: str) -> VotingResult:
        """Create empty result when no votes."""
        return VotingResult(
            pattern_id=pattern_id,
            decision=VoteType.UNCERTAIN,
            confidence=0.0,
            vote_breakdown={vote_type: 0 for vote_type in VoteType},
            weighted_breakdown={vote_type: 0.0 for vote_type in VoteType},
            participating_agents=[],
            total_votes=0,
            total_weight=0.0,
            consensus_achieved=False,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=[]
        )


class AdaptiveVoting(VotingMechanism):
    """
    Adaptive voting mechanism that selects voting method based on context.
    Uses different strategies based on vote distribution and agent characteristics.
    """
    
    def __init__(self):
        """Initialize adaptive voting with multiple mechanisms."""
        super().__init__("adaptive_voting")
        
        # Available voting mechanisms
        self.mechanisms = {
            'majority': MajorityVoting(require_majority=True),
            'weighted': WeightedVoting(weight_threshold=0.6),
            'threshold': ThresholdVoting(significance_threshold=0.7, confidence_threshold=0.8),
            'quorum': QuorumVoting(min_participants=3, min_weight=0.5)
        }
    
    def compute_result(self, votes: List[Vote], pattern_id: str) -> VotingResult:
        """Compute adaptive voting result."""
        if not votes:
            return self._empty_result(pattern_id)
        
        # Analyze vote characteristics
        analysis = self._analyze_votes(votes)
        
        # Select appropriate mechanism
        mechanism_name = self._select_mechanism(analysis)
        mechanism = self.mechanisms[mechanism_name]
        
        # Compute result
        result = mechanism.compute_result(votes, pattern_id)
        result.voting_method = f"{self.name}_{mechanism_name}"
        
        return result
    
    def _analyze_votes(self, votes: List[Vote]) -> Dict[str, Any]:
        """Analyze vote characteristics."""
        total_votes = len(votes)
        vote_counts, weighted_counts = self._calculate_vote_breakdown(votes)
        
        # Calculate various metrics
        confidence_scores = [v.confidence for v in votes]
        reputation_scores = [v.reputation_score for v in votes]
        weights = [v.weight for v in votes]
        
        # Distribution analysis
        max_vote_count = max(vote_counts.values()) if vote_counts else 0
        vote_spread = max_vote_count / total_votes if total_votes > 0 else 0
        
        # Confidence analysis
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        confidence_std = statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
        
        # Reputation analysis
        avg_reputation = statistics.mean(reputation_scores) if reputation_scores else 1
        reputation_std = statistics.stdev(reputation_scores) if len(reputation_scores) > 1 else 0
        
        # Weight analysis
        total_weight = sum(weights)
        weight_inequality = statistics.stdev(weights) if len(weights) > 1 else 0
        
        return {
            'total_votes': total_votes,
            'vote_spread': vote_spread,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'avg_reputation': avg_reputation,
            'reputation_std': reputation_std,
            'total_weight': total_weight,
            'weight_inequality': weight_inequality,
            'vote_counts': vote_counts,
            'weighted_counts': weighted_counts
        }
    
    def _select_mechanism(self, analysis: Dict[str, Any]) -> str:
        """Select appropriate voting mechanism based on analysis."""
        total_votes = analysis['total_votes']
        vote_spread = analysis['vote_spread']
        avg_confidence = analysis['avg_confidence']
        confidence_std = analysis['confidence_std']
        avg_reputation = analysis['avg_reputation']
        reputation_std = analysis['reputation_std']
        weight_inequality = analysis['weight_inequality']
        
        # Decision logic
        if total_votes < 3:
            # Few votes, use simple majority
            return 'majority'
        
        elif weight_inequality > 0.5 or reputation_std > 0.3:
            # High weight/reputation inequality, use weighted voting
            return 'weighted'
        
        elif avg_confidence > 0.8 and confidence_std < 0.2:
            # High confidence with low spread, use threshold voting
            return 'threshold'
        
        elif total_votes >= 5 and avg_reputation > 0.7:
            # Good participation and reputation, use quorum voting
            return 'quorum'
        
        else:
            # Default to weighted voting
            return 'weighted'
    
    def _empty_result(self, pattern_id: str) -> VotingResult:
        """Create empty result when no votes."""
        return VotingResult(
            pattern_id=pattern_id,
            decision=VoteType.UNCERTAIN,
            confidence=0.0,
            vote_breakdown={vote_type: 0 for vote_type in VoteType},
            weighted_breakdown={vote_type: 0.0 for vote_type in VoteType},
            participating_agents=[],
            total_votes=0,
            total_weight=0.0,
            consensus_achieved=False,
            voting_method=self.name,
            timestamp=datetime.utcnow().isoformat(),
            detailed_votes=[]
        )


def create_voting_mechanism(mechanism_type: str, **kwargs) -> VotingMechanism:
    """
    Factory function to create voting mechanisms.
    
    Args:
        mechanism_type: Type of voting mechanism
        **kwargs: Mechanism-specific parameters
        
    Returns:
        Configured voting mechanism
    """
    mechanisms = {
        'majority': MajorityVoting,
        'weighted': WeightedVoting,
        'threshold': ThresholdVoting,
        'quorum': QuorumVoting,
        'adaptive': AdaptiveVoting
    }
    
    if mechanism_type not in mechanisms:
        raise ValueError(f"Unknown voting mechanism: {mechanism_type}")
    
    return mechanisms[mechanism_type](**kwargs)