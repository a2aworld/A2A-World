/**
 * A2A World Platform - Enhanced Dashboard Component
 * 
 * Comprehensive dashboard integrating all visualization components for Phase 2 completion.
 * Features data visualization, pattern analysis, agent monitoring, and system metrics.
 */

import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import { 
  Globe, 
  Activity, 
  Brain, 
  Database, 
  Users, 
  TrendingUp,
  MapPin,
  FileText,
  Settings,
  RefreshCw
} from 'lucide-react';

// Import our visualization components
import { BarChart, LineChart, PieChart } from '../charts';
import { StatCard } from '../widgets/StatCard';
import { ProgressWidget } from '../widgets/ProgressWidget';
import { ConfidenceIndicator } from '../patterns/ConfidenceIndicator';

export interface DashboardData {
  systemMetrics: {
    totalPatterns: number;
    activeAgents: number;
    datasetCount: number;
    processingStatus: string;
    uptime: string;
    systemHealth: 'healthy' | 'warning' | 'critical';
  };
  patternStats: {
    discovered: number;
    validated: number;
    confidence: number;
    trends: Array<{
      timestamp: string;
      discovered: number;
      validated: number;
    }>;
  };
  dataQuality: {
    overall: number;
    byType: Array<{
      name: string;
      value: number;
      color?: string;
    }>;
  };
  agentPerformance: Array<{
    name: string;
    processed: number;
    success_rate: number;
    status: 'active' | 'idle' | 'error';
  }>;
  recentActivity: Array<{
    id: string;
    type: 'pattern_discovered' | 'data_uploaded' | 'agent_started' | 'validation_completed';
    description: string;
    timestamp: string;
    status: 'success' | 'warning' | 'error';
  }>;
}

export interface EnhancedDashboardProps {
  data?: DashboardData;
  loading?: boolean;
  error?: string;
  onRefresh?: () => void;
  className?: string;
}

// Mock data for demonstration
const mockData: DashboardData = {
  systemMetrics: {
    totalPatterns: 156,
    activeAgents: 4,
    datasetCount: 23,
    processingStatus: 'active',
    uptime: '2d 14h 32m',
    systemHealth: 'healthy'
  },
  patternStats: {
    discovered: 156,
    validated: 89,
    confidence: 0.847,
    trends: [
      { timestamp: '2024-01-25T00:00:00Z', discovered: 12, validated: 8 },
      { timestamp: '2024-01-26T00:00:00Z', discovered: 18, validated: 12 },
      { timestamp: '2024-01-27T00:00:00Z', discovered: 25, validated: 18 },
      { timestamp: '2024-01-28T00:00:00Z', discovered: 31, validated: 22 },
      { timestamp: '2024-01-29T00:00:00Z', discovered: 28, validated: 15 },
      { timestamp: '2024-01-30T00:00:00Z', discovered: 42, validated: 14 }
    ]
  },
  dataQuality: {
    overall: 0.92,
    byType: [
      { name: 'KML Files', value: 8, color: '#3B82F6' },
      { name: 'GeoJSON', value: 6, color: '#10B981' },
      { name: 'CSV Data', value: 5, color: '#F59E0B' },
      { name: 'Sacred Sites', value: 4, color: '#8B5CF6' }
    ]
  },
  agentPerformance: [
    { name: 'Pattern Discovery', processed: 1250, success_rate: 0.94, status: 'active' },
    { name: 'Data Validation', processed: 890, success_rate: 0.87, status: 'active' },
    { name: 'KML Parser', processed: 456, success_rate: 0.96, status: 'idle' },
    { name: 'Cultural Analysis', processed: 234, success_rate: 0.78, status: 'error' }
  ],
  recentActivity: [
    {
      id: '1',
      type: 'pattern_discovered',
      description: 'New geometric pattern discovered in Southwest region',
      timestamp: '2024-01-30T10:30:00Z',
      status: 'success'
    },
    {
      id: '2', 
      type: 'data_uploaded',
      description: 'Sacred sites dataset uploaded and processed',
      timestamp: '2024-01-30T10:15:00Z',
      status: 'success'
    },
    {
      id: '3',
      type: 'validation_completed',
      description: 'Pattern validation completed with 89% confidence',
      timestamp: '2024-01-30T09:45:00Z', 
      status: 'success'
    }
  ]
};

const getActivityIcon = (type: string) => {
  switch (type) {
    case 'pattern_discovered':
      return Brain;
    case 'data_uploaded':
      return Database;
    case 'agent_started':
      return Users;
    case 'validation_completed':
      return TrendingUp;
    default:
      return Activity;
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'success':
      return 'text-green-600 bg-green-50';
    case 'warning':
      return 'text-yellow-600 bg-yellow-50';
    case 'error':
      return 'text-red-600 bg-red-50';
    default:
      return 'text-gray-600 bg-gray-50';
  }
};

export function EnhancedDashboard({
  data = mockData,
  loading = false,
  error,
  onRefresh,
  className
}: EnhancedDashboardProps) {
  const [selectedTimeRange, setSelectedTimeRange] = useState<'24h' | '7d' | '30d'>('7d');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (autoRefresh && onRefresh) {
      const interval = setInterval(onRefresh, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh, onRefresh]);

  if (loading) {
    return (
      <div className={clsx('p-6 space-y-6', className)}>
        <div className="animate-pulse">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded-lg" />
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-96 bg-gray-200 rounded-lg" />
            <div className="h-96 bg-gray-200 rounded-lg" />
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={clsx('p-6 flex items-center justify-center', className)}>
        <div className="text-center">
          <div className="text-red-500 mb-4">
            <Activity className="h-12 w-12 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Dashboard Error</h3>
          <p className="text-gray-600 mb-4">{error}</p>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('p-6 space-y-6', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">A2A World Dashboard</h1>
          <p className="text-sm text-gray-600 mt-1">
            Pattern Discovery & Cultural Analysis Platform
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Time Range Selector */}
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value as any)}
            className="text-sm border border-gray-300 rounded-md px-3 py-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          
          {/* Auto Refresh Toggle */}
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={clsx(
              'text-sm px-3 py-2 rounded-md border',
              autoRefresh
                ? 'bg-green-50 text-green-700 border-green-200'
                : 'bg-gray-50 text-gray-700 border-gray-200'
            )}
          >
            Auto Refresh {autoRefresh ? 'On' : 'Off'}
          </button>
          
          {/* Manual Refresh */}
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="p-2 text-gray-400 hover:text-gray-600 border border-gray-300 rounded-md"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Patterns"
          value={data.systemMetrics.totalPatterns}
          icon={Brain}
          color="purple"
          trend={{
            value: 12.5,
            label: 'this week',
            direction: 'up'
          }}
        />
        <StatCard
          title="Active Agents"
          value={data.systemMetrics.activeAgents}
          subtitle="4 total agents"
          icon={Users}
          color="blue"
          trend={{
            value: 0,
            label: 'unchanged',
            direction: 'neutral'
          }}
        />
        <StatCard
          title="Datasets"
          value={data.systemMetrics.datasetCount}
          icon={Database}
          color="green"
          trend={{
            value: 8.3,
            label: 'this month',
            direction: 'up'
          }}
        />
        <StatCard
          title="System Health"
          value={data.systemMetrics.systemHealth.toUpperCase()}
          subtitle={`Uptime: ${data.systemMetrics.uptime}`}
          icon={Activity}
          color="green"
        />
      </div>

      {/* Main Visualization Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pattern Discovery Trends */}
        <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Pattern Discovery Trends</h3>
            <div className="text-sm text-gray-500">Last 7 days</div>
          </div>
          <LineChart
            data={data.patternStats.trends}
            lines={[
              { key: 'discovered', name: 'Discovered', color: '#3B82F6' },
              { key: 'validated', name: 'Validated', color: '#10B981' }
            ]}
            height={300}
            xAxisFormat="date"
            showGrid={true}
            showLegend={true}
          />
        </div>

        {/* Pattern Confidence */}
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Pattern Confidence</h3>
          <div className="flex items-center justify-center">
            <ConfidenceIndicator
              value={data.patternStats.confidence}
              size="lg"
              showThreshold={true}
              animated={true}
            />
          </div>
          <div className="mt-4 text-center">
            <div className="text-sm text-gray-600">
              Average confidence across {data.patternStats.validated} validated patterns
            </div>
          </div>
        </div>
      </div>

      {/* Secondary Visualization Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Quality Distribution */}
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Quality by Type</h3>
          <PieChart
            data={data.dataQuality.byType}
            height={300}
            showLegend={true}
            showTooltip={true}
            donut={true}
          />
        </div>

        {/* Agent Performance */}
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Performance</h3>
          <BarChart
            data={data.agentPerformance.map(agent => ({
              name: agent.name,
              value: Math.round(agent.success_rate * 100),
              metadata: {
                processed: agent.processed,
                status: agent.status
              }
            }))}
            height={300}
            yAxisKey="value"
            showDataLabels={true}
            colors={['#3B82F6', '#10B981', '#F59E0B', '#EF4444']}
          />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow border">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {data.recentActivity.map((activity) => {
              const Icon = getActivityIcon(activity.type);
              return (
                <div key={activity.id} className="flex items-start space-x-3">
                  <div className={clsx('p-2 rounded-full', getStatusColor(activity.status))}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">
                      {activity.description}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(activity.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

export default EnhancedDashboard;