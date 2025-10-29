/**
 * A2A World Platform - Consensus History Component
 * 
 * Displays historical consensus decisions and patterns with trend analysis.
 * Shows consensus performance over time and decision patterns.
 */

import React, { useState, useEffect } from 'react';
import { 
  History, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  PieChart, 
  Filter,
  Calendar,
  Download,
  RefreshCw
} from 'lucide-react';
import { BrandedCard } from '../branding/BrandedCard';
import { BrandedChart } from '../branding/BrandedChart';
import { LoadingSpinner } from '../branding/LoadingSpinner';

// Types for consensus history data
interface ConsensusHistoryEntry {
  id: string;
  pattern_id: string;
  decision: 'significant' | 'not_significant' | 'uncertain';
  confidence: number;
  participating_agents: string[];
  consensus_protocol: string;
  voting_mechanism: string;
  execution_time_seconds: number;
  decision_timestamp: string;
  vote_breakdown: Record<string, number>;
}

interface ConsensusTrend {
  date: string;
  total_decisions: number;
  significant_count: number;
  not_significant_count: number;
  uncertain_count: number;
  average_confidence: number;
  average_execution_time: number;
  protocol_usage: Record<string, number>;
}

interface ConsensusHistoryProps {
  patternId?: string;
  timeRange?: '1d' | '7d' | '30d' | '90d';
  limit?: number;
}

export const ConsensusHistory: React.FC<ConsensusHistoryProps> = ({
  patternId,
  timeRange = '30d',
  limit = 100
}) => {
  const [historyData, setHistoryData] = useState<ConsensusHistoryEntry[]>([]);
  const [trends, setTrends] = useState<ConsensusTrend[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState(timeRange);
  const [filterProtocol, setFilterProtocol] = useState<string>('all');
  const [filterDecision, setFilterDecision] = useState<string>('all');

  useEffect(() => {
    fetchConsensusHistory();
  }, [patternId, selectedTimeRange, filterProtocol, filterDecision]);

  const fetchConsensusHistory = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        time_range: selectedTimeRange,
        limit: limit.toString()
      });

      if (patternId) {
        params.append('pattern_id', patternId);
      }
      if (filterProtocol !== 'all') {
        params.append('protocol', filterProtocol);
      }
      if (filterDecision !== 'all') {
        params.append('decision', filterDecision);
      }

      // Simulate API call - in real implementation would call actual endpoint
      const simulatedHistory = generateSimulatedHistory(selectedTimeRange, limit);
      setHistoryData(simulatedHistory.entries);
      setTrends(simulatedHistory.trends);

    } catch (err) {
      console.error('Error fetching consensus history:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const generateSimulatedHistory = (timeRange: string, limit: number) => {
    const entries: ConsensusHistoryEntry[] = [];
    const trends: ConsensusTrend[] = [];
    
    const days = timeRange === '1d' ? 1 : timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
    const entriesPerDay = Math.ceil(limit / days);

    for (let d = 0; d < days; d++) {
      const date = new Date();
      date.setDate(date.getDate() - d);
      
      let dailyEntries = 0;
      let significantCount = 0;
      let notSignificantCount = 0;
      let uncertainCount = 0;
      let totalConfidence = 0;
      let totalExecutionTime = 0;
      const protocolUsage: Record<string, number> = {};

      for (let i = 0; i < entriesPerDay && entries.length < limit; i++) {
        const entryDate = new Date(date);
        entryDate.setHours(Math.random() * 24, Math.random() * 60);
        
        const decisions = ['significant', 'not_significant', 'uncertain'] as const;
        const protocols = ['adaptive', 'raft', 'bft', 'voting_only'];
        const votingMechanisms = ['adaptive', 'weighted', 'threshold', 'majority'];
        
        const decision = decisions[Math.floor(Math.random() * decisions.length)];
        const protocol = protocols[Math.floor(Math.random() * protocols.length)];
        const votingMech = votingMechanisms[Math.floor(Math.random() * votingMechanisms.length)];
        
        const confidence = 0.5 + Math.random() * 0.5;
        const executionTime = 1 + Math.random() * 10;
        
        // Count for trends
        dailyEntries++;
        totalConfidence += confidence;
        totalExecutionTime += executionTime;
        
        if (decision === 'significant') significantCount++;
        else if (decision === 'not_significant') notSignificantCount++;
        else uncertainCount++;
        
        protocolUsage[protocol] = (protocolUsage[protocol] || 0) + 1;

        const entry: ConsensusHistoryEntry = {
          id: `consensus_${Date.now()}_${i}`,
          pattern_id: patternId || `pattern_${Math.random().toString(36).substr(2, 9)}`,
          decision,
          confidence,
          participating_agents: Array.from({ length: 3 + Math.floor(Math.random() * 5) }, 
            (_, idx) => `agent_${idx + 1}`),
          consensus_protocol: protocol,
          voting_mechanism: votingMech,
          execution_time_seconds: executionTime,
          decision_timestamp: entryDate.toISOString(),
          vote_breakdown: {
            [decision]: 3 + Math.floor(Math.random() * 3),
            'other': Math.floor(Math.random() * 2)
          }
        };
        
        entries.push(entry);
      }

      if (dailyEntries > 0) {
        trends.push({
          date: date.toISOString().split('T')[0],
          total_decisions: dailyEntries,
          significant_count: significantCount,
          not_significant_count: notSignificantCount,
          uncertain_count: uncertainCount,
          average_confidence: totalConfidence / dailyEntries,
          average_execution_time: totalExecutionTime / dailyEntries,
          protocol_usage: protocolUsage
        });
      }
    }

    return { entries: entries.reverse(), trends: trends.reverse() };
  };

  const createDecisionTrendChart = () => {
    const trendData = trends.map(trend => ({
      date: new Date(trend.date).toLocaleDateString(),
      Significant: trend.significant_count,
      'Not Significant': trend.not_significant_count,
      Uncertain: trend.uncertain_count
    }));

    return (
      <BrandedChart
        data={trendData}
        type="line"
        title="Decision Trends Over Time"
        className="h-64"
      />
    );
  };

  const createProtocolUsageChart = () => {
    const protocolData: Record<string, number> = {};
    
    historyData.forEach(entry => {
      protocolData[entry.consensus_protocol] = (protocolData[entry.consensus_protocol] || 0) + 1;
    });

    const data = Object.entries(protocolData).map(([protocol, count]) => ({
      name: protocol.toUpperCase(),
      value: count
    }));

    return (
      <BrandedChart
        data={data}
        type="pie"
        title="Consensus Protocol Usage"
        className="h-64"
      />
    );
  };

  const createConfidenceTrendChart = () => {
    const confidenceData = trends.map(trend => ({
      date: new Date(trend.date).toLocaleDateString(),
      confidence: (trend.average_confidence * 100).toFixed(1)
    }));

    return (
      <BrandedChart
        data={confidenceData}
        type="line"
        title="Average Confidence Over Time"
        className="h-64"
      />
    );
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    return `${seconds.toFixed(1)}s`;
  };

  const getDecisionIcon = (decision: string) => {
    switch (decision) {
      case 'significant':
        return '✓';
      case 'not_significant':
        return '✗';
      case 'uncertain':
        return '?';
      default:
        return '•';
    }
  };

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'significant':
        return 'text-green-600';
      case 'not_significant':
        return 'text-red-600';
      case 'uncertain':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };

  if (isLoading) {
    return (
      <BrandedCard className="p-6">
        <div className="flex items-center justify-center">
          <LoadingSpinner size="lg" />
          <span className="ml-3 text-lg">Loading consensus history...</span>
        </div>
      </BrandedCard>
    );
  }

  if (error) {
    return (
      <BrandedCard className="p-6 border-red-200 bg-red-50">
        <div className="flex items-center mb-4">
          <History className="h-6 w-6 text-red-500 mr-3" />
          <h3 className="text-lg font-semibold text-red-700">History Error</h3>
        </div>
        <p className="text-red-600">{error}</p>
      </BrandedCard>
    );
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <BrandedCard className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Calendar className="h-4 w-4 text-gray-500" />
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value as any)}
                className="text-sm border border-gray-300 rounded px-2 py-1"
              >
                <option value="1d">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last 90 Days</option>
              </select>
            </div>
            
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <select
                value={filterProtocol}
                onChange={(e) => setFilterProtocol(e.target.value)}
                className="text-sm border border-gray-300 rounded px-2 py-1"
              >
                <option value="all">All Protocols</option>
                <option value="bft">BFT</option>
                <option value="raft">RAFT</option>
                <option value="adaptive">Adaptive</option>
                <option value="voting_only">Voting Only</option>
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <select
                value={filterDecision}
                onChange={(e) => setFilterDecision(e.target.value)}
                className="text-sm border border-gray-300 rounded px-2 py-1"
              >
                <option value="all">All Decisions</option>
                <option value="significant">Significant</option>
                <option value="not_significant">Not Significant</option>
                <option value="uncertain">Uncertain</option>
              </select>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={fetchConsensusHistory}
              className="flex items-center px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </button>
            
            <button
              onClick={() => {/* Export functionality */}}
              className="flex items-center px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </button>
          </div>
        </div>
      </BrandedCard>

      {/* Summary Statistics */}
      {historyData.length > 0 && (
        <BrandedCard className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Summary Statistics</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <BarChart3 className="h-6 w-6 text-blue-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {historyData.length}
              </div>
              <div className="text-sm text-gray-600">Total Decisions</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <TrendingUp className="h-6 w-6 text-green-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-green-600">
                {historyData.filter(h => h.decision === 'significant').length}
              </div>
              <div className="text-sm text-gray-600">Significant</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <TrendingDown className="h-6 w-6 text-red-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-red-600">
                {historyData.filter(h => h.decision === 'not_significant').length}
              </div>
              <div className="text-sm text-gray-600">Not Significant</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <PieChart className="h-6 w-6 text-yellow-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-yellow-600">
                {historyData.filter(h => h.decision === 'uncertain').length}
              </div>
              <div className="text-sm text-gray-600">Uncertain</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Average Confidence</div>
              <div className="text-xl font-bold text-blue-600">
                {((historyData.reduce((sum, h) => sum + h.confidence, 0) / historyData.length) * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Avg Execution Time</div>
              <div className="text-xl font-bold text-green-600">
                {formatDuration(historyData.reduce((sum, h) => sum + h.execution_time_seconds, 0) / historyData.length)}
              </div>
            </div>
            
            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Avg Participants</div>
              <div className="text-xl font-bold text-purple-600">
                {(historyData.reduce((sum, h) => sum + h.participating_agents.length, 0) / historyData.length).toFixed(1)}
              </div>
            </div>
          </div>
        </BrandedCard>
      )}

      {/* Trend Charts */}
      {trends.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <BrandedCard className="p-6">
            {createDecisionTrendChart()}
          </BrandedCard>
          
          <BrandedCard className="p-6">
            {createProtocolUsageChart()}
          </BrandedCard>
        </div>
      )}

      {/* Confidence Trend */}
      {trends.length > 0 && (
        <BrandedCard className="p-6">
          {createConfidenceTrendChart()}
        </BrandedCard>
      )}

      {/* History Table */}
      <BrandedCard className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Recent Consensus Decisions</h3>
          <span className="text-sm text-gray-600">{historyData.length} entries</span>
        </div>
        
        {historyData.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            No consensus history found for the selected criteria
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Pattern
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Decision
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Protocol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Participants
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Duration
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Timestamp
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {historyData.map((entry) => (
                  <tr key={entry.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {entry.pattern_id.substring(0, 12)}...
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <span className={`mr-2 ${getDecisionColor(entry.decision)}`}>
                          {getDecisionIcon(entry.decision)}
                        </span>
                        <span className={`text-sm font-medium ${getDecisionColor(entry.decision)}`}>
                          {entry.decision.replace('_', ' ')}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="text-sm text-gray-900">
                          {(entry.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              entry.confidence >= 0.8 ? 'bg-green-500' :
                              entry.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${entry.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 uppercase">
                      {entry.consensus_protocol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {entry.participating_agents.length}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatDuration(entry.execution_time_seconds)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(entry.decision_timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </BrandedCard>
    </div>
  );
};

export default ConsensusHistory;