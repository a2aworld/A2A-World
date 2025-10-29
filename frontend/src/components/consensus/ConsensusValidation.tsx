/**
 * A2A World Platform - Consensus Validation Component
 * 
 * Displays consensus-based pattern validation results including voting breakdown,
 * agent participation, confidence scores, and decision reasoning.
 */

import React, { useState, useEffect } from 'react';
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Clock, 
  Users, 
  TrendingUp,
  Shield,
  Network,
  ChevronDown,
  ChevronUp,
  Info
} from 'lucide-react';
import { BrandedCard } from '../branding/BrandedCard';
import { BrandedChart } from '../branding/BrandedChart';
import { LoadingSpinner } from '../branding/LoadingSpinner';

// Types for consensus validation data
interface ConsensusVote {
  agent_id: string;
  vote: 'significant' | 'not_significant' | 'uncertain' | 'abstain';
  confidence: number;
  statistical_evidence: Record<string, any>;
  reasoning: string;
  voting_weight: number;
  reputation_score: number;
}

interface ConsensusValidationResult {
  request_id: string;
  pattern_id: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'timeout';
  decision?: string;
  confidence: number;
  consensus_protocol_used?: string;
  voting_mechanism_used?: string;
  participating_agents: string[];
  vote_breakdown: Record<string, number>;
  weighted_breakdown: Record<string, number>;
  execution_time_seconds: number;
  consensus_achieved: boolean;
  statistical_summary: Record<string, any>;
  detailed_votes?: ConsensusVote[];
  error_message?: string;
  timestamp: string;
}

interface ConsensusValidationProps {
  patternId: string;
  validationResult?: ConsensusValidationResult;
  onRefresh?: () => void;
  showDetailedBreakdown?: boolean;
}

export const ConsensusValidation: React.FC<ConsensusValidationProps> = ({
  patternId,
  validationResult,
  onRefresh,
  showDetailedBreakdown = false
}) => {
  const [isLoading, setIsLoading] = useState(!validationResult);
  const [result, setResult] = useState<ConsensusValidationResult | null>(validationResult || null);
  const [showDetails, setShowDetails] = useState(showDetailedBreakdown);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!validationResult && patternId) {
      fetchValidationResult();
    }
  }, [patternId, validationResult]);

  const fetchValidationResult = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/v1/consensus/validate/${patternId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Error fetching consensus validation result:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusIcon = (status: string, decision?: string) => {
    switch (status) {
      case 'completed':
        return decision === 'significant' ? 
          <CheckCircle className="h-6 w-6 text-green-500" /> :
          <XCircle className="h-6 w-6 text-red-500" />;
      case 'pending':
      case 'in_progress':
        return <Clock className="h-6 w-6 text-yellow-500" />;
      case 'failed':
      case 'timeout':
        return <AlertCircle className="h-6 w-6 text-red-500" />;
      default:
        return <AlertCircle className="h-6 w-6 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string, decision?: string) => {
    switch (status) {
      case 'completed':
        return decision === 'significant' ? 'green' : 'red';
      case 'pending':
      case 'in_progress':
        return 'yellow';
      case 'failed':
      case 'timeout':
        return 'red';
      default:
        return 'gray';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

  const createVoteBreakdownChart = () => {
    if (!result?.vote_breakdown) return null;

    const data = Object.entries(result.vote_breakdown).map(([vote, count]) => ({
      name: vote.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: count,
      percentage: (count / result.participating_agents.length) * 100
    }));

    return (
      <BrandedChart
        data={data}
        type="pie"
        title="Vote Distribution"
        className="h-64"
      />
    );
  };

  const createConfidenceChart = () => {
    if (!result?.detailed_votes) return null;

    const confidenceRanges = {
      'Very High (≥90%)': 0,
      'High (70-89%)': 0,
      'Medium (50-69%)': 0,
      'Low (<50%)': 0
    };

    result.detailed_votes.forEach(vote => {
      if (vote.confidence >= 0.9) confidenceRanges['Very High (≥90%)']++;
      else if (vote.confidence >= 0.7) confidenceRanges['High (70-89%)']++;
      else if (vote.confidence >= 0.5) confidenceRanges['Medium (50-69%)']++;
      else confidenceRanges['Low (<50%)']++;
    });

    const data = Object.entries(confidenceRanges).map(([range, count]) => ({
      name: range,
      value: count
    }));

    return (
      <BrandedChart
        data={data}
        type="bar"
        title="Confidence Distribution"
        className="h-64"
      />
    );
  };

  if (isLoading) {
    return (
      <BrandedCard className="p-6">
        <div className="flex items-center justify-center">
          <LoadingSpinner size="lg" />
          <span className="ml-3 text-lg">Processing consensus validation...</span>
        </div>
      </BrandedCard>
    );
  }

  if (error) {
    return (
      <BrandedCard className="p-6 border-red-200 bg-red-50">
        <div className="flex items-center mb-4">
          <AlertCircle className="h-6 w-6 text-red-500 mr-3" />
          <h3 className="text-lg font-semibold text-red-700">Consensus Validation Error</h3>
        </div>
        <p className="text-red-600 mb-4">{error}</p>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
          >
            Retry Validation
          </button>
        )}
      </BrandedCard>
    );
  }

  if (!result) {
    return (
      <BrandedCard className="p-6">
        <div className="text-center text-gray-500">
          No consensus validation result available
        </div>
      </BrandedCard>
    );
  }

  return (
    <div className="space-y-6">
      {/* Main Results Card */}
      <BrandedCard className="p-6">
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center">
            {getStatusIcon(result.status, result.decision)}
            <div className="ml-4">
              <h2 className="text-xl font-bold text-gray-900">
                Consensus Validation Result
              </h2>
              <p className="text-sm text-gray-600">Pattern ID: {result.pattern_id}</p>
            </div>
          </div>
          
          <div className="text-right">
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              getStatusColor(result.status, result.decision) === 'green' 
                ? 'bg-green-100 text-green-800'
                : getStatusColor(result.status, result.decision) === 'red'
                ? 'bg-red-100 text-red-800'
                : getStatusColor(result.status, result.decision) === 'yellow'
                ? 'bg-yellow-100 text-yellow-800'
                : 'bg-gray-100 text-gray-800'
            }`}>
              {result.status === 'completed' ? 
                `${result.decision?.replace('_', ' ').toUpperCase()}` : 
                result.status.replace('_', ' ').toUpperCase()
              }
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center">
              <TrendingUp className="h-5 w-5 text-blue-500 mr-2" />
              <span className="text-sm font-medium text-gray-600">Confidence</span>
            </div>
            <div className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
              {formatPercentage(result.confidence)}
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center">
              <Users className="h-5 w-5 text-green-500 mr-2" />
              <span className="text-sm font-medium text-gray-600">Participants</span>
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {result.participating_agents.length}
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center">
              <Shield className="h-5 w-5 text-purple-500 mr-2" />
              <span className="text-sm font-medium text-gray-600">Protocol</span>
            </div>
            <div className="text-sm font-bold text-gray-900 uppercase">
              {result.consensus_protocol_used || 'Unknown'}
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center">
              <Clock className="h-5 w-5 text-orange-500 mr-2" />
              <span className="text-sm font-medium text-gray-600">Duration</span>
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {result.execution_time_seconds.toFixed(1)}s
            </div>
          </div>
        </div>

        {/* Consensus Achievement Status */}
        <div className={`p-4 rounded-lg mb-6 ${
          result.consensus_achieved 
            ? 'bg-green-50 border border-green-200'
            : 'bg-red-50 border border-red-200'
        }`}>
          <div className="flex items-center">
            <Network className={`h-5 w-5 mr-2 ${
              result.consensus_achieved ? 'text-green-500' : 'text-red-500'
            }`} />
            <span className={`font-medium ${
              result.consensus_achieved ? 'text-green-800' : 'text-red-800'
            }`}>
              {result.consensus_achieved 
                ? 'Consensus Successfully Achieved'
                : 'Consensus Not Achieved'
              }
            </span>
          </div>
          {result.error_message && (
            <p className="mt-2 text-sm text-red-600">{result.error_message}</p>
          )}
        </div>

        {/* Toggle Details Button */}
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center justify-center w-full py-2 text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
        >
          {showDetails ? (
            <>Hide Detailed Breakdown <ChevronUp className="ml-1 h-4 w-4" /></>
          ) : (
            <>Show Detailed Breakdown <ChevronDown className="ml-1 h-4 w-4" /></>
          )}
        </button>
      </BrandedCard>

      {/* Detailed Breakdown */}
      {showDetails && (
        <>
          {/* Vote Breakdown Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <BrandedCard className="p-6">
              {createVoteBreakdownChart()}
            </BrandedCard>
            
            {result.detailed_votes && (
              <BrandedCard className="p-6">
                {createConfidenceChart()}
              </BrandedCard>
            )}
          </div>

          {/* Vote Details Table */}
          {result.detailed_votes && (
            <BrandedCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Individual Agent Votes
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Agent
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Vote
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Weight
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Reputation
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Reasoning
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {result.detailed_votes.map((vote, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {vote.agent_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            vote.vote === 'significant'
                              ? 'bg-green-100 text-green-800'
                              : vote.vote === 'not_significant'
                              ? 'bg-red-100 text-red-800'
                              : vote.vote === 'uncertain'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            {vote.vote.replace('_', ' ')}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`text-sm font-medium ${getConfidenceColor(vote.confidence)}`}>
                            {formatPercentage(vote.confidence)}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {vote.voting_weight.toFixed(2)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="text-sm text-gray-900">
                              {formatPercentage(vote.reputation_score)}
                            </div>
                            <div className={`ml-2 w-16 bg-gray-200 rounded-full h-2`}>
                              <div
                                className={`h-2 rounded-full ${
                                  vote.reputation_score >= 0.8
                                    ? 'bg-green-500'
                                    : vote.reputation_score >= 0.6
                                    ? 'bg-yellow-500'
                                    : 'bg-red-500'
                                }`}
                                style={{ width: `${vote.reputation_score * 100}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-500 max-w-xs">
                          <div className="truncate" title={vote.reasoning}>
                            {vote.reasoning}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </BrandedCard>
          )}

          {/* Statistical Summary */}
          {Object.keys(result.statistical_summary).length > 0 && (
            <BrandedCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Statistical Evidence Summary
              </h3>
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="flex items-start">
                  <Info className="h-5 w-5 text-blue-500 mt-0.5 mr-3 flex-shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-800 mb-2">
                      Statistical Analysis Results
                    </h4>
                    <div className="space-y-1">
                      {Object.entries(result.statistical_summary).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-blue-700 font-medium">
                            {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                          </span>
                          <span className="text-blue-600">
                            {typeof value === 'number' 
                              ? value % 1 === 0 ? value : value.toFixed(3)
                              : String(value)
                            }
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </BrandedCard>
          )}
        </>
      )}
    </div>
  );
};

export default ConsensusValidation;