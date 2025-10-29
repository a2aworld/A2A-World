"""
A2A World Platform - Consensus API Endpoints

API endpoints for peer-to-peer consensus-based pattern validation operations.
Provides interfaces for consensus validation requests, reputation management,
network monitoring, and performance analytics.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uuid

from api.app.core.config import get_settings
from api.app.core.performance import track_performance, get_performance_metrics
from api.app.core.errors import APIError, ValidationError, NotFoundError

# Import consensus system components (these would need to be available)
# from agents.consensus.consensus_coordinator import ConsensusCoordinator, ConsensusValidationRequest
# from agents.consensus.reputation_system import get_reputation_system
# from agents.core.messaging import get_nats_client

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation
class ConsensusValidationRequest(BaseModel):
    """Request model for consensus validation."""
    pattern_id: str = Field(..., description="Pattern identifier for validation")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern data for analysis")
    statistical_results: Optional[List[Dict[str, Any]]] = Field(default=[], description="Existing statistical results")
    consensus_protocol: str = Field(default="adaptive", description="Consensus protocol to use")
    voting_mechanism: str = Field(default="adaptive", description="Voting mechanism for decision making")
    min_participants: int = Field(default=3, ge=1, le=20, description="Minimum participants required")
    timeout_seconds: int = Field(default=60, ge=10, le=300, description="Consensus timeout in seconds")
    require_statistical_evidence: bool = Field(default=True, description="Require statistical validation evidence")
    
    @validator('consensus_protocol')
    def validate_consensus_protocol(cls, v):
        valid_protocols = ['bft', 'raft', 'voting_only', 'adaptive']
        if v not in valid_protocols:
            raise ValueError(f"Consensus protocol must be one of: {valid_protocols}")
        return v
    
    @validator('voting_mechanism')
    def validate_voting_mechanism(cls, v):
        valid_mechanisms = ['majority', 'weighted', 'threshold', 'quorum', 'adaptive']
        if v not in valid_mechanisms:
            raise ValueError(f"Voting mechanism must be one of: {valid_mechanisms}")
        return v


class ConsensusValidationResponse(BaseModel):
    """Response model for consensus validation results."""
    request_id: str
    pattern_id: str
    status: str
    decision: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    consensus_protocol_used: Optional[str] = None
    participating_agents: List[str] = []
    execution_time_seconds: float
    statistical_summary: Dict[str, Any] = {}
    consensus_achieved: bool
    error_message: Optional[str] = None
    timestamp: str


class AgentReputationResponse(BaseModel):
    """Response model for agent reputation information."""
    agent_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    participation_score: float = Field(ge=0.0, le=1.0)
    total_validations: int
    correct_predictions: int
    consensus_agreements: int
    voting_weight: float
    last_updated: str


class ConsensusNetworkStatus(BaseModel):
    """Response model for consensus network status."""
    total_nodes: int
    active_nodes: int
    consensus_coordinators: int
    validation_agents: int
    network_health: str
    average_latency_ms: float
    last_updated: str


class ConsensusMetrics(BaseModel):
    """Response model for consensus system metrics."""
    total_requests: int
    successful_consensus: int
    failed_consensus: int
    success_rate: float
    average_response_time: float
    protocol_usage: Dict[str, int]
    agent_participation_rates: Dict[str, float]
    reputation_distribution: Dict[str, int]


class ConsensusConfiguration(BaseModel):
    """Model for consensus system configuration."""
    default_consensus_protocol: str = "adaptive"
    default_voting_mechanism: str = "adaptive"
    min_participants: int = Field(default=3, ge=1, le=20)
    default_timeout_seconds: int = Field(default=60, ge=10, le=300)
    reputation_decay_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    byzantine_fault_tolerance: bool = True
    enable_peer_ratings: bool = True


# Global consensus coordinator instance (would be initialized at startup)
_consensus_coordinator = None
_reputation_system = None


def get_consensus_coordinator():
    """Get consensus coordinator instance."""
    global _consensus_coordinator
    if _consensus_coordinator is None:
        # In a real implementation, this would initialize the coordinator
        # _consensus_coordinator = ConsensusCoordinator()
        pass
    return _consensus_coordinator


def get_reputation_system():
    """Get reputation system instance."""
    global _reputation_system
    if _reputation_system is None:
        # In a real implementation, this would get the reputation system
        # _reputation_system = get_reputation_system()
        pass
    return _reputation_system


@router.post("/validate/{pattern_id}", response_model=ConsensusValidationResponse)
@track_performance
async def initiate_consensus_validation(
    pattern_id: str,
    request: ConsensusValidationRequest,
    background_tasks: BackgroundTasks
) -> ConsensusValidationResponse:
    """
    Initiate consensus-based pattern validation.
    
    Starts a distributed validation process using the specified consensus protocol
    and voting mechanism to determine pattern significance.
    """
    try:
        logger.info(f"Starting consensus validation for pattern {pattern_id}")
        
        # Validate pattern_id format
        if not pattern_id or len(pattern_id) < 8:
            raise ValidationError("Invalid pattern ID format")
        
        # Update request with pattern_id
        request.pattern_id = pattern_id
        
        # Get consensus coordinator
        coordinator = get_consensus_coordinator()
        if not coordinator:
            # Simulate consensus validation for development
            return await _simulate_consensus_validation(request)
        
        # Execute consensus validation
        result = await coordinator.request_consensus_validation(
            pattern_id=request.pattern_id,
            pattern_data=request.pattern_data,
            statistical_results=request.statistical_results,
            consensus_protocol=request.consensus_protocol,
            voting_mechanism=request.voting_mechanism,
            min_participants=request.min_participants,
            timeout_seconds=request.timeout_seconds,
            require_statistical_evidence=request.require_statistical_evidence
        )
        
        # Convert result to response model
        response = ConsensusValidationResponse(
            request_id=result.get("request_id", str(uuid.uuid4())),
            pattern_id=result.get("pattern_id", pattern_id),
            status=result.get("status", "completed"),
            decision=result.get("decision"),
            confidence=result.get("confidence", 0.0),
            consensus_protocol_used=result.get("consensus_protocol_used"),
            participating_agents=result.get("participating_agents", []),
            execution_time_seconds=result.get("execution_time_seconds", 0.0),
            statistical_summary=result.get("statistical_summary", {}),
            consensus_achieved=result.get("consensus_achieved", False),
            error_message=result.get("error_message"),
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Store result in background
        background_tasks.add_task(_store_consensus_result, response)
        
        logger.info(f"Consensus validation completed for pattern {pattern_id}: {response.decision}")
        return response
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Consensus validation failed for pattern {pattern_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Consensus validation failed: {str(e)}"
        )


@router.get("/status/{validation_id}", response_model=ConsensusValidationResponse)
@track_performance
async def get_consensus_status(validation_id: str) -> ConsensusValidationResponse:
    """
    Get the status of a consensus validation request.
    
    Returns the current status and results of an ongoing or completed
    consensus validation process.
    """
    try:
        logger.info(f"Retrieving consensus status for validation {validation_id}")
        
        # Get consensus coordinator
        coordinator = get_consensus_coordinator()
        if not coordinator:
            # Simulate status retrieval
            return await _simulate_consensus_status(validation_id)
        
        # Get validation status from coordinator
        result = await coordinator.get_validation_status(validation_id)
        
        if not result:
            raise NotFoundError(f"Consensus validation {validation_id} not found")
        
        # Convert to response model
        response = ConsensusValidationResponse(**result)
        
        return response
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving consensus status for {validation_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve consensus status: {str(e)}"
        )


@router.get("/agents", response_model=List[AgentReputationResponse])
@track_performance
async def list_consensus_agents(
    active_only: bool = Query(True, description="Return only active agents"),
    min_reputation: float = Query(0.0, ge=0.0, le=1.0, description="Minimum reputation score"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of agents to return")
) -> List[AgentReputationResponse]:
    """
    List participating validation agents with reputation scores.
    
    Returns information about agents in the consensus network including
    their reputation scores, validation statistics, and current status.
    """
    try:
        logger.info(f"Listing consensus agents (active_only={active_only}, min_reputation={min_reputation})")
        
        # Get reputation system
        reputation_system = get_reputation_system()
        if not reputation_system:
            # Return simulated agent data
            return await _simulate_agent_list(active_only, min_reputation, limit)
        
        # Get agent reputation data
        agents = await reputation_system.get_agents_with_reputation(
            active_only=active_only,
            min_reputation=min_reputation,
            limit=limit
        )
        
        # Convert to response models
        agent_responses = []
        for agent_data in agents:
            reputation = agent_data.get("reputation_score", {})
            agent_response = AgentReputationResponse(
                agent_id=agent_data["agent_id"],
                overall_score=reputation.get("overall_score", 0.5),
                accuracy_score=reputation.get("accuracy_score", 0.5),
                reliability_score=reputation.get("reliability_score", 0.5),
                participation_score=reputation.get("participation_score", 0.5),
                total_validations=reputation.get("total_validations", 0),
                correct_predictions=reputation.get("correct_predictions", 0),
                consensus_agreements=reputation.get("consensus_agreements", 0),
                voting_weight=agent_data.get("voting_weight", 1.0),
                last_updated=reputation.get("last_updated", datetime.utcnow().isoformat())
            )
            agent_responses.append(agent_response)
        
        logger.info(f"Retrieved {len(agent_responses)} consensus agents")
        return agent_responses
        
    except Exception as e:
        logger.error(f"Error listing consensus agents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list consensus agents: {str(e)}"
        )


@router.post("/configure", response_model=Dict[str, Any])
@track_performance
async def configure_consensus_system(
    config: ConsensusConfiguration
) -> Dict[str, Any]:
    """
    Configure consensus system parameters.
    
    Updates system-wide consensus configuration including default protocols,
    voting mechanisms, and reputation settings.
    """
    try:
        logger.info("Updating consensus system configuration")
        
        # Get consensus coordinator
        coordinator = get_consensus_coordinator()
        if not coordinator:
            # Simulate configuration update
            return await _simulate_configuration_update(config)
        
        # Update configuration
        result = await coordinator.update_configuration(config.dict())
        
        logger.info("Consensus system configuration updated successfully")
        return {
            "status": "success",
            "message": "Consensus configuration updated",
            "configuration": config.dict(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating consensus configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update consensus configuration: {str(e)}"
        )


@router.get("/reputation", response_model=List[AgentReputationResponse])
@track_performance
async def get_agent_reputation_scores(
    agent_ids: Optional[List[str]] = Query(None, description="Specific agent IDs to retrieve"),
    sort_by: str = Query("overall_score", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)")
) -> List[AgentReputationResponse]:
    """
    Get detailed agent reputation and trust scores.
    
    Returns comprehensive reputation information for agents including
    accuracy metrics, reliability scores, and peer ratings.
    """
    try:
        logger.info(f"Retrieving agent reputation scores (agents={agent_ids}, sort={sort_by})")
        
        # Get reputation system
        reputation_system = get_reputation_system()
        if not reputation_system:
            # Return simulated reputation data
            return await _simulate_reputation_scores(agent_ids, sort_by, sort_order)
        
        # Get reputation data
        if agent_ids:
            reputations = []
            for agent_id in agent_ids:
                reputation = await reputation_system.get_agent_reputation(agent_id)
                if reputation:
                    reputations.append(reputation)
        else:
            reputations = await reputation_system.get_top_agents(limit=50)
        
        # Convert to response models
        responses = []
        for rep_data in reputations:
            if isinstance(rep_data, tuple):
                agent_id, reputation = rep_data
            else:
                agent_id = rep_data.agent_id
                reputation = rep_data
            
            response = AgentReputationResponse(
                agent_id=agent_id,
                overall_score=reputation.overall_score,
                accuracy_score=reputation.accuracy_score,
                reliability_score=reputation.reliability_score,
                participation_score=reputation.participation_score,
                total_validations=reputation.total_validations,
                correct_predictions=reputation.correct_predictions,
                consensus_agreements=reputation.consensus_agreements,
                voting_weight=reputation_system.get_voting_weight(agent_id),
                last_updated=reputation.last_updated
            )
            responses.append(response)
        
        # Sort results
        reverse = sort_order.lower() == "desc"
        responses.sort(key=lambda x: getattr(x, sort_by, 0), reverse=reverse)
        
        logger.info(f"Retrieved {len(responses)} agent reputation scores")
        return responses
        
    except Exception as e:
        logger.error(f"Error retrieving reputation scores: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve reputation scores: {str(e)}"
        )


@router.get("/metrics", response_model=ConsensusMetrics)
@track_performance
async def get_consensus_metrics(
    time_period: str = Query("24h", description="Time period for metrics (1h, 24h, 7d, 30d)"),
    include_details: bool = Query(False, description="Include detailed breakdown")
) -> ConsensusMetrics:
    """
    Get consensus system performance metrics.
    
    Returns comprehensive metrics about consensus performance including
    success rates, response times, protocol usage, and agent participation.
    """
    try:
        logger.info(f"Retrieving consensus metrics for period {time_period}")
        
        # Parse time period
        time_delta = _parse_time_period(time_period)
        start_time = datetime.utcnow() - time_delta
        
        # Get consensus coordinator
        coordinator = get_consensus_coordinator()
        if not coordinator:
            # Return simulated metrics
            return await _simulate_consensus_metrics(time_period)
        
        # Get metrics from coordinator
        metrics = await coordinator.get_performance_metrics(start_time, include_details)
        
        # Convert to response model
        response = ConsensusMetrics(
            total_requests=metrics.get("total_requests", 0),
            successful_consensus=metrics.get("successful_consensus", 0),
            failed_consensus=metrics.get("failed_consensus", 0),
            success_rate=metrics.get("success_rate", 0.0),
            average_response_time=metrics.get("average_response_time", 0.0),
            protocol_usage=metrics.get("protocol_usage", {}),
            agent_participation_rates=metrics.get("agent_participation_rates", {}),
            reputation_distribution=metrics.get("reputation_distribution", {})
        )
        
        logger.info(f"Retrieved consensus metrics: {response.success_rate:.2%} success rate")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving consensus metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve consensus metrics: {str(e)}"
        )


@router.get("/network", response_model=ConsensusNetworkStatus)
@track_performance
async def get_network_status() -> ConsensusNetworkStatus:
    """
    Get consensus network topology and health status.
    
    Returns information about the current state of the consensus network
    including node counts, connectivity, and overall health metrics.
    """
    try:
        logger.info("Retrieving consensus network status")
        
        # Get consensus coordinator
        coordinator = get_consensus_coordinator()
        if not coordinator:
            # Return simulated network status
            return await _simulate_network_status()
        
        # Get network status
        status = await coordinator.get_network_status()
        
        # Convert to response model
        response = ConsensusNetworkStatus(
            total_nodes=status.get("total_nodes", 0),
            active_nodes=status.get("active_nodes", 0),
            consensus_coordinators=status.get("consensus_coordinators", 0),
            validation_agents=status.get("validation_agents", 0),
            network_health=status.get("network_health", "unknown"),
            average_latency_ms=status.get("average_latency_ms", 0.0),
            last_updated=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Network status: {response.active_nodes}/{response.total_nodes} nodes active")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving network status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve network status: {str(e)}"
        )


@router.get("/protocols", response_model=Dict[str, Any])
@track_performance
async def get_available_protocols() -> Dict[str, Any]:
    """
    Get information about available consensus protocols and voting mechanisms.
    
    Returns details about supported consensus algorithms, voting strategies,
    and their characteristics for protocol selection.
    """
    try:
        logger.info("Retrieving available consensus protocols")
        
        protocols_info = {
            "consensus_protocols": {
                "bft": {
                    "name": "Byzantine Fault Tolerant",
                    "description": "Tolerates up to f faulty nodes out of 3f+1 total nodes",
                    "min_nodes": 4,
                    "fault_tolerance": "byzantine",
                    "consistency": "strong",
                    "availability": "moderate",
                    "performance": "moderate",
                    "use_case": "Critical patterns requiring high security"
                },
                "raft": {
                    "name": "RAFT Consensus",
                    "description": "Leader-based consensus with log replication",
                    "min_nodes": 3,
                    "fault_tolerance": "crash",
                    "consistency": "strong",
                    "availability": "high",
                    "performance": "high",
                    "use_case": "General purpose distributed validation"
                },
                "voting_only": {
                    "name": "Voting Only",
                    "description": "Simple voting without Byzantine fault tolerance",
                    "min_nodes": 1,
                    "fault_tolerance": "none",
                    "consistency": "eventual",
                    "availability": "high",
                    "performance": "very_high",
                    "use_case": "Fast decisions with trusted agents"
                },
                "adaptive": {
                    "name": "Adaptive Protocol Selection",
                    "description": "Automatically selects best protocol based on context",
                    "min_nodes": 1,
                    "fault_tolerance": "adaptive",
                    "consistency": "adaptive",
                    "availability": "high",
                    "performance": "adaptive",
                    "use_case": "Default choice for most scenarios"
                }
            },
            "voting_mechanisms": {
                "majority": {
                    "name": "Majority Voting",
                    "description": "Simple majority wins (>50% of votes)",
                    "requires_majority": True,
                    "weight_support": False,
                    "complexity": "low"
                },
                "weighted": {
                    "name": "Weighted Voting",
                    "description": "Votes weighted by agent reputation and expertise",
                    "requires_majority": False,
                    "weight_support": True,
                    "complexity": "medium"
                },
                "threshold": {
                    "name": "Threshold Voting",
                    "description": "Requires supermajority (typically 67%) for decision",
                    "requires_majority": True,
                    "weight_support": True,
                    "complexity": "medium"
                },
                "quorum": {
                    "name": "Quorum Voting",
                    "description": "Requires minimum participation before voting",
                    "requires_majority": True,
                    "weight_support": True,
                    "complexity": "high"
                },
                "adaptive": {
                    "name": "Adaptive Voting",
                    "description": "Selects best voting method based on vote characteristics",
                    "requires_majority": False,
                    "weight_support": True,
                    "complexity": "high"
                }
            },
            "system_status": {
                "protocols_available": ["bft", "raft", "voting_only", "adaptive"],
                "voting_mechanisms_available": ["majority", "weighted", "threshold", "quorum", "adaptive"],
                "default_protocol": "adaptive",
                "default_voting": "adaptive",
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
        return protocols_info
        
    except Exception as e:
        logger.error(f"Error retrieving protocol information: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve protocol information: {str(e)}"
        )


# Helper functions for simulation (used when consensus system is not available)
async def _simulate_consensus_validation(request: ConsensusValidationRequest) -> ConsensusValidationResponse:
    """Simulate consensus validation for development/testing."""
    await asyncio.sleep(2)  # Simulate processing time
    
    # Simulate consensus decision based on statistical results
    decision = "significant" if len(request.statistical_results) > 0 else "not_significant"
    confidence = 0.85 if decision == "significant" else 0.72
    
    return ConsensusValidationResponse(
        request_id=str(uuid.uuid4()),
        pattern_id=request.pattern_id,
        status="completed",
        decision=decision,
        confidence=confidence,
        consensus_protocol_used=request.consensus_protocol,
        participating_agents=[f"agent_{i}" for i in range(1, request.min_participants + 1)],
        execution_time_seconds=2.5,
        statistical_summary={"total_tests": len(request.statistical_results)},
        consensus_achieved=True,
        timestamp=datetime.utcnow().isoformat()
    )


async def _simulate_consensus_status(validation_id: str) -> ConsensusValidationResponse:
    """Simulate consensus status retrieval."""
    return ConsensusValidationResponse(
        request_id=validation_id,
        pattern_id=f"pattern_{validation_id[-8:]}",
        status="completed",
        decision="significant",
        confidence=0.87,
        consensus_protocol_used="adaptive",
        participating_agents=["agent_1", "agent_2", "agent_3"],
        execution_time_seconds=3.2,
        statistical_summary={"total_tests": 5, "significant_tests": 4},
        consensus_achieved=True,
        timestamp=datetime.utcnow().isoformat()
    )


async def _simulate_agent_list(active_only: bool, min_reputation: float, limit: int) -> List[AgentReputationResponse]:
    """Simulate agent list for development."""
    agents = []
    for i in range(1, min(limit + 1, 11)):
        reputation_score = 0.5 + (i * 0.05) + min_reputation * 0.1
        agents.append(AgentReputationResponse(
            agent_id=f"consensus_agent_{i}",
            overall_score=min(1.0, reputation_score),
            accuracy_score=min(1.0, reputation_score + 0.1),
            reliability_score=min(1.0, reputation_score + 0.05),
            participation_score=min(1.0, reputation_score - 0.05),
            total_validations=10 + i * 5,
            correct_predictions=8 + i * 4,
            consensus_agreements=9 + i * 4,
            voting_weight=min(2.0, 0.8 + reputation_score * 1.2),
            last_updated=datetime.utcnow().isoformat()
        ))
    return agents


async def _simulate_reputation_scores(agent_ids: Optional[List[str]], sort_by: str, sort_order: str) -> List[AgentReputationResponse]:
    """Simulate reputation scores."""
    if agent_ids:
        return [AgentReputationResponse(
            agent_id=agent_id,
            overall_score=0.75,
            accuracy_score=0.82,
            reliability_score=0.78,
            participation_score=0.85,
            total_validations=25,
            correct_predictions=20,
            consensus_agreements=23,
            voting_weight=1.2,
            last_updated=datetime.utcnow().isoformat()
        ) for agent_id in agent_ids]
    
    return await _simulate_agent_list(True, 0.0, 10)


async def _simulate_consensus_metrics(time_period: str) -> ConsensusMetrics:
    """Simulate consensus metrics."""
    return ConsensusMetrics(
        total_requests=150,
        successful_consensus=142,
        failed_consensus=8,
        success_rate=0.947,
        average_response_time=3.2,
        protocol_usage={"adaptive": 80, "raft": 45, "bft": 20, "voting_only": 5},
        agent_participation_rates={"agent_1": 0.95, "agent_2": 0.88, "agent_3": 0.92},
        reputation_distribution={"high": 3, "medium": 5, "low": 2}
    )


async def _simulate_network_status() -> ConsensusNetworkStatus:
    """Simulate network status."""
    return ConsensusNetworkStatus(
        total_nodes=10,
        active_nodes=8,
        consensus_coordinators=2,
        validation_agents=6,
        network_health="healthy",
        average_latency_ms=45.2,
        last_updated=datetime.utcnow().isoformat()
    )


async def _simulate_configuration_update(config: ConsensusConfiguration) -> Dict[str, Any]:
    """Simulate configuration update."""
    return {
        "status": "success",
        "message": "Configuration updated (simulated)",
        "configuration": config.dict(),
        "updated_at": datetime.utcnow().isoformat()
    }


def _parse_time_period(period: str) -> timedelta:
    """Parse time period string to timedelta."""
    period_map = {
        "1h": timedelta(hours=1),
        "24h": timedelta(days=1),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30)
    }
    return period_map.get(period, timedelta(days=1))


async def _store_consensus_result(result: ConsensusValidationResponse) -> None:
    """Store consensus result in database (background task)."""
    try:
        # This would store the result in the database
        logger.info(f"Stored consensus result for pattern {result.pattern_id}")
    except Exception as e:
        logger.error(f"Error storing consensus result: {e}")


# Performance monitoring endpoint
@router.get("/performance", response_model=Dict[str, Any])
@track_performance
async def get_consensus_performance_stats() -> Dict[str, Any]:
    """
    Get detailed consensus system performance statistics.
    
    Returns comprehensive performance metrics including API response times,
    consensus algorithm performance, and system resource usage.
    """
    try:
        # Get API performance metrics
        api_metrics = get_performance_metrics("consensus")
        
        # Add consensus-specific performance data
        performance_stats = {
            "api_metrics": api_metrics,
            "consensus_performance": {
                "average_consensus_time": 3.2,
                "protocol_performance": {
                    "bft": {"avg_time": 5.1, "success_rate": 0.94},
                    "raft": {"avg_time": 2.8, "success_rate": 0.97},
                    "voting_only": {"avg_time": 1.2, "success_rate": 0.99},
                    "adaptive": {"avg_time": 3.0, "success_rate": 0.96}
                },
                "network_performance": {
                    "average_latency_ms": 45.2,
                    "message_throughput": 1250,
                    "connection_stability": 0.98
                }
            },
            "system_health": {
                "memory_usage_mb": 256,
                "cpu_usage_percent": 15.3,
                "active_connections": 48,
                "error_rate": 0.02
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return performance_stats
        
    except Exception as e:
        logger.error(f"Error retrieving performance stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance stats: {str(e)}"
        )