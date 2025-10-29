/**
 * A2A World Platform - Agent Reputation Component
 * 
 * Displays agent reputation and trust scores for consensus participants,
 * including performance metrics, validation accuracy, and peer ratings.
 */

import React, { useState, useEffect } from 'react';
import { 
  Star, 
  TrendingUp, 
  Users, 
  Shield, 
  Clock, 
  CheckCircle,
  AlertCircle,
  Activity,
  Award,
  Eye,
  EyeOff
} from 'lucide-react';
import { BrandedCard } from '../branding/BrandedCard';
import { BrandedChart } from '../branding/BrandedChart';
import { LoadingSpinner } from '../branding/LoadingSpinner';

// Types for agent reputation data
interface AgentReputation {
  agent_id: string;
  overall_score: number;
  accuracy_score: number;
  reliability_score: number;
  timeliness_score: number;
  participation_score: number;
  quality_score: number;
  peer_score: number;
  total_validations: number;
  correct_predictions: number;
  consensus_agreements: number;
  voting_weight: number;
  last_updated: string;
}

interface AgentReputationProps {
  agentId?: string;
  showComparison?: boolean;
  limit?: number;
  minReputation?: number;
  onAgentSelect?: (agentId: string) => void;
}

export const AgentReputation: React.FC<AgentReputationProps> = ({
  agentId,
  showComparison = false,
  limit = 10,
  minReputation = 0.0,
  onAgentSelect
}) => {
  const [agents, setAgents] = useState<AgentReputation[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentReputation | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<keyof AgentReputation>('overall_score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showInactive, setShowInactive] = useState(false);

  useEffect(() => {
    fetchAgentReputations();
  }, [agentId, limit, minReputation, showInactive]);

  const fetchAgentReputations = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        active_only: (!showInactive).toString(),
        min_reputation: minReputation.toString(),
        limit: limit.toString(),
        sort_by: sortBy,
        sort_order: sortOrder
      });

      if (agentId) {
        params.append('agent_ids', agentId);
      }

      const response = await fetch(`/api/v1/consensus/reputation?${params}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setAgents(data);
      
      if (agentId && data.length > 0) {
        setSelectedAgent(data[0]);
      }
    } catch (err) {
      console.error('Error fetching agent reputations:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const getReputationColor = (score: number): string => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    if (score >= 0.4) return 'text-orange-600';
    return 'text-red-600';
  };

  const getReputationBadge = (score: number): { label: string; color: string } => {
    if (score >= 0.9) return { label: 'Excellent', color: 'bg-green-100 text-green-800' };
    if (score >= 0.8) return { label: 'Very Good', color: 'bg-green-100 text-green-700' };
    if (score >= 0.7) return { label: 'Good', color: 'bg-yellow-100 text-yellow-800' };
    if (score >= 0.6) return { label: 'Average', color: 'bg-yellow-100 text-yellow-700' };
    if (score >= 0.4) return { label: 'Below Average', color: 'bg-orange-100 text-orange-800' };
    return { label: 'Poor', color: 'bg-red-100 text-red-800' };
  };

  const formatPercentage = (value: number): string => `${(value * 100).toFixed(1)}%`;

  const createReputationRadarChart = (agent: AgentReputation) => {
    const radarData = [
      { subject: 'Accuracy', A: agent.accuracy_score * 100, fullMark: 100 },
      { subject: 'Reliability', A: agent.reliability_score * 100, fullMark: 100 },
      { subject: 'Timeliness', A: agent.timeliness_score * 100, fullMark: 100 },
      { subject: 'Participation', A: agent.participation_score * 100, fullMark: 100 },
      { subject: 'Quality', A: agent.quality_score * 100, fullMark: 100 },
      { subject: 'Peer Rating', A: agent.peer_score * 100, fullMark: 100 }
    ];

    return (
      <BrandedChart
        data={radarData}
        type="radar"
        title="Reputation Breakdown"
        className="h-64"
      />
    );
  };

  const createComparisonChart = () => {
    const topAgents = agents.slice(0, 5);
    const comparisonData = topAgents.map(agent => ({
      name: agent.agent_id.replace('_', ' '),
      overall: agent.overall_score * 100,
      accuracy: agent.accuracy_score * 100,
      reliability: agent.reliability_score * 100
    }));

    return (
      <BrandedChart
        data={comparisonData}
        type="bar"
        title="Top Agents Comparison"
        className="h-64"
      />
    );
  };

  const handleSort = (field: keyof AgentReputation) => {
    const newOrder = field === sortBy && sortOrder === 'desc' ? 'asc' : 'desc';
    setSortBy(field);
    setSortOrder(newOrder);
    
    const sorted = [...agents].sort((a, b) => {
      const aVal = a[field];
      const bVal = b[field];
      const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return newOrder === 'desc' ? -comparison : comparison;
    });
    
    setAgents(sorted);
  };

  if (isLoading) {
    return (
      <BrandedCard className="p-6">
        <div className="flex items-center justify-center">
          <LoadingSpinner size="lg" />
          <span className="ml-3 text-lg">Loading agent reputations...</span>
        </div>
      </BrandedCard>
    );
  }

  if (error) {
    return (
      <BrandedCard className="p-6 border-red-200 bg-red-50">
        <div className="flex items-center mb-4">
          <AlertCircle className="h-6 w-6 text-red-500 mr-3" />
          <h3 className="text-lg font-semibold text-red-700">Error Loading Reputations</h3>
        </div>
        <p className="text-red-600">{error}</p>
      </BrandedCard>
    );
  }

  if (agents.length === 0) {
    return (
      <BrandedCard className="p-6">
        <div className="text-center text-gray-500">
          No agents found matching the specified criteria
        </div>
      </BrandedCard>
    );
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <BrandedCard className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showInactive}
                onChange={(e) => setShowInactive(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm">Show inactive agents</span>
            </label>
            
            <button
              onClick={() => fetchAgentReputations()}
              className="flex items-center px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              <Activity className="h-4 w-4 mr-1" />
              Refresh
            </button>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => handleSort(e.target.value as keyof AgentReputation)}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="overall_score">Overall Score</option>
              <option value="accuracy_score">Accuracy</option>
              <option value="reliability_score">Reliability</option>
              <option value="total_validations">Total Validations</option>
              <option value="voting_weight">Voting Weight</option>
            </select>
          </div>
        </div>
      </BrandedCard>

      {/* Selected Agent Details */}
      {selectedAgent && (
        <BrandedCard className="p-6">
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center">
              <div className="bg-blue-100 p-3 rounded-full">
                <Shield className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <h2 className="text-xl font-bold text-gray-900">{selectedAgent.agent_id}</h2>
                <div className="flex items-center mt-1">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    getReputationBadge(selectedAgent.overall_score).color
                  }`}>
                    {getReputationBadge(selectedAgent.overall_score).label}
                  </span>
                  <span className="ml-2 text-sm text-gray-600">
                    Updated: {new Date(selectedAgent.last_updated).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <div className={`text-3xl font-bold ${getReputationColor(selectedAgent.overall_score)}`}>
                {formatPercentage(selectedAgent.overall_score)}
              </div>
              <div className="text-sm text-gray-600">Overall Score</div>
            </div>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <CheckCircle className="h-6 w-6 text-green-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {selectedAgent.total_validations}
              </div>
              <div className="text-sm text-gray-600">Total Validations</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <TrendingUp className="h-6 w-6 text-blue-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {formatPercentage(selectedAgent.correct_predictions / Math.max(selectedAgent.total_validations, 1))}
              </div>
              <div className="text-sm text-gray-600">Accuracy Rate</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <Users className="h-6 w-6 text-purple-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {selectedAgent.consensus_agreements}
              </div>
              <div className="text-sm text-gray-600">Consensus Agreements</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <Star className="h-6 w-6 text-yellow-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {selectedAgent.voting_weight.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Voting Weight</div>
            </div>
          </div>

          {/* Reputation Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              {createReputationRadarChart(selectedAgent)}
            </div>
            
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-900">Score Breakdown</h4>
              
              {[
                { label: 'Accuracy', score: selectedAgent.accuracy_score, icon: CheckCircle },
                { label: 'Reliability', score: selectedAgent.reliability_score, icon: Shield },
                { label: 'Timeliness', score: selectedAgent.timeliness_score, icon: Clock },
                { label: 'Participation', score: selectedAgent.participation_score, icon: Users },
                { label: 'Quality', score: selectedAgent.quality_score, icon: Award },
                { label: 'Peer Rating', score: selectedAgent.peer_score, icon: Star }
              ].map(({ label, score, icon: Icon }) => (
                <div key={label} className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Icon className="h-4 w-4 text-gray-500 mr-2" />
                    <span className="text-sm font-medium text-gray-700">{label}</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-24 bg-gray-200 rounded-full h-2 mr-3">
                      <div
                        className={`h-2 rounded-full ${
                          score >= 0.8 ? 'bg-green-500' :
                          score >= 0.6 ? 'bg-yellow-500' :
                          score >= 0.4 ? 'bg-orange-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <span className={`text-sm font-medium ${getReputationColor(score)}`}>
                      {formatPercentage(score)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </BrandedCard>
      )}

      {/* Agent Ranking Table */}
      <BrandedCard className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Agent Reputation Ranking</h3>
          <span className="text-sm text-gray-600">{agents.length} agents</span>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Rank
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('agent_id')}
                >
                  Agent
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('overall_score')}
                >
                  Overall Score
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('accuracy_score')}
                >
                  Accuracy
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('total_validations')}
                >
                  Validations
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('voting_weight')}
                >
                  Weight
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {agents.map((agent, index) => (
                <tr 
                  key={agent.agent_id} 
                  className={`hover:bg-gray-50 cursor-pointer ${
                    selectedAgent?.agent_id === agent.agent_id ? 'bg-blue-50' : ''
                  }`}
                  onClick={() => setSelectedAgent(agent)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {index < 3 && (
                        <Award className={`h-4 w-4 mr-2 ${
                          index === 0 ? 'text-yellow-500' :
                          index === 1 ? 'text-gray-400' :
                          'text-yellow-600'
                        }`} />
                      )}
                      <span className="text-sm font-medium text-gray-900">
                        #{index + 1}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="text-sm font-medium text-gray-900">
                        {agent.agent_id}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className={`text-sm font-medium ${getReputationColor(agent.overall_score)}`}>
                        {formatPercentage(agent.overall_score)}
                      </div>
                      <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            agent.overall_score >= 0.8 ? 'bg-green-500' :
                            agent.overall_score >= 0.6 ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${agent.overall_score * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatPercentage(agent.accuracy_score)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {agent.total_validations}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {agent.voting_weight.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (onAgentSelect) onAgentSelect(agent.agent_id);
                      }}
                      className="text-blue-600 hover:text-blue-900"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </BrandedCard>

      {/* Comparison Chart */}
      {showComparison && agents.length > 1 && (
        <BrandedCard className="p-6">
          {createComparisonChart()}
        </BrandedCard>
      )}
    </div>
  );
};

export default AgentReputation;