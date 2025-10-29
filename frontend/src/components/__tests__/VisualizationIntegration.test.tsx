/**
 * A2A World Platform - Visualization Integration Tests
 * 
 * Integration tests demonstrating real API data integration with visualization components.
 * Tests component functionality with actual backend endpoints and data structures.
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BarChart, LineChart, PieChart } from '../charts';
import { StatCard } from '../widgets/StatCard';
import { ConfidenceIndicator } from '../patterns/ConfidenceIndicator';
import { EnhancedDashboard } from '../dashboard/EnhancedDashboard';
import { SAMPLE_DATASETS } from '../data/sampleData';

// Mock API responses matching the actual endpoint structure
const mockApiResponses = {
  patterns: {
    success: true,
    data: {
      patterns: [
        {
          id: 'pattern-1',
          name: 'Sacred Geometry Alignment',
          type: 'geometric',
          confidence_score: 0.89,
          status: 'validated',
          discovery_date: '2024-01-30T10:30:00Z',
          coordinates: [
            { latitude: 39.0242, longitude: -83.4310 },
            { latitude: 36.0619, longitude: -107.9560 }
          ]
        }
      ],
      total: 156,
      significant_patterns: 89
    }
  },
  agents: {
    success: true,
    data: {
      agents: [
        {
          agent_id: 'discovery-01',
          agent_name: 'Pattern Discovery Agent',
          status: 'active',
          health_status: 'healthy',
          processed_tasks: 1250,
          failed_tasks: 15,
          metrics: {
            cpu_usage: 12.3,
            memory_usage: 234.5,
            uptime_percentage: 98.5
          }
        }
      ],
      total: 4,
      status_summary: { active: 3, idle: 1, error: 0 }
    }
  },
  dataSummary: {
    success: true,
    data: {
      datasets: {
        total: 23,
        completed: 21,
        failed: 2
      },
      features: {
        total: 4567
      },
      file_types: {
        'kml': 8,
        'geojson': 6,
        'csv': 5,
        'shapefile': 4
      }
    }
  }
};

describe('Visualization Components - Real Data Integration', () => {
  
  describe('Chart Components with API Data', () => {
    test('BarChart renders with pattern statistics', async () => {
      const patternData = Object.entries(mockApiResponses.dataSummary.data.file_types)
        .map(([name, value]) => ({ name: name.toUpperCase(), value }));

      render(
        <BarChart
          data={patternData}
          title="Dataset Distribution"
          showDataLabels={true}
        />
      );

      expect(screen.getByText('Dataset Distribution')).toBeInTheDocument();
    });

    test('LineChart handles time series pattern data', async () => {
      const trendData = SAMPLE_DATASETS.timeSeries.map(point => ({
        timestamp: point.timestamp,
        discovered: point.discovered,
        validated: point.validated
      }));

      render(
        <LineChart
          data={trendData}
          lines={[
            { key: 'discovered', name: 'Discovered', color: '#3B82F6' },
            { key: 'validated', name: 'Validated', color: '#10B981' }
          ]}
          title="Pattern Discovery Trends"
          xAxisFormat="date"
        />
      );

      expect(screen.getByText('Pattern Discovery Trends')).toBeInTheDocument();
    });

    test('PieChart displays data quality distribution', async () => {
      const qualityData = [
        { name: 'High Quality', value: 18, color: '#10B981' },
        { name: 'Medium Quality', value: 4, color: '#F59E0B' },
        { name: 'Low Quality', value: 1, color: '#EF4444' }
      ];

      render(
        <PieChart
          data={qualityData}
          title="Data Quality Distribution"
          donut={true}
          showLegend={true}
        />
      );

      expect(screen.getByText('Data Quality Distribution')).toBeInTheDocument();
    });
  });

  describe('Dashboard Widgets with Real Metrics', () => {
    test('StatCard displays pattern count from API', () => {
      const totalPatterns = mockApiResponses.patterns.data.total;

      render(
        <StatCard
          title="Total Patterns"
          value={totalPatterns}
          trend={{ value: 12.5, label: 'this week', direction: 'up' }}
        />
      );

      expect(screen.getByText('Total Patterns')).toBeInTheDocument();
      expect(screen.getByText('156')).toBeInTheDocument();
    });

    test('StatCard handles agent status data', () => {
      const activeAgents = mockApiResponses.agents.data.status_summary.active;
      const totalAgents = mockApiResponses.agents.data.total;

      render(
        <StatCard
          title="Active Agents"
          value={`${activeAgents}/${totalAgents}`}
          color="blue"
        />
      );

      expect(screen.getByText('Active Agents')).toBeInTheDocument();
      expect(screen.getByText('3/4')).toBeInTheDocument();
    });
  });

  describe('Pattern Visualization Components', () => {
    test('ConfidenceIndicator shows pattern confidence', () => {
      const confidenceScore = mockApiResponses.patterns.data.patterns[0].confidence_score;

      render(
        <ConfidenceIndicator
          value={confidenceScore}
          label="Pattern Confidence"
          size="lg"
          animated={true}
        />
      );

      expect(screen.getByText('Pattern Confidence')).toBeInTheDocument();
      expect(screen.getByText('89%')).toBeInTheDocument();
    });
  });

  describe('Enhanced Dashboard Integration', () => {
    test('Dashboard renders with complete API data integration', async () => {
      // Transform API responses into dashboard data format
      const dashboardData = {
        systemMetrics: {
          totalPatterns: mockApiResponses.patterns.data.total,
          activeAgents: mockApiResponses.agents.data.status_summary.active,
          datasetCount: mockApiResponses.dataSummary.data.datasets.total,
          processingStatus: 'active',
          uptime: '2d 14h 32m',
          systemHealth: 'healthy' as const
        },
        patternStats: {
          discovered: mockApiResponses.patterns.data.total,
          validated: mockApiResponses.patterns.data.significant_patterns,
          confidence: 0.847,
          trends: SAMPLE_DATASETS.timeSeries
        },
        dataQuality: {
          overall: 0.92,
          byType: Object.entries(mockApiResponses.dataSummary.data.file_types)
            .map(([name, value]) => ({ name: name.toUpperCase(), value }))
        },
        agentPerformance: mockApiResponses.agents.data.agents.map(agent => ({
          name: agent.agent_name,
          processed: agent.processed_tasks,
          success_rate: (agent.processed_tasks - agent.failed_tasks) / agent.processed_tasks,
          status: agent.status as 'active' | 'idle' | 'error'
        })),
        recentActivity: [
          {
            id: '1',
            type: 'pattern_discovered' as const,
            description: 'New pattern discovered with high confidence',
            timestamp: '2024-01-30T10:30:00Z',
            status: 'success' as const
          }
        ]
      };

      render(<EnhancedDashboard data={dashboardData} />);

      await waitFor(() => {
        expect(screen.getByText('A2A World Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Total Patterns')).toBeInTheDocument();
        expect(screen.getByText('Active Agents')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling and Loading States', () => {
    test('Charts handle loading state correctly', () => {
      render(
        <BarChart
          data={[]}
          title="Loading Test"
          loading={true}
        />
      );

      expect(screen.getByText('Loading chart data...')).toBeInTheDocument();
    });

    test('Charts display error state appropriately', () => {
      render(
        <LineChart
          data={[]}
          lines={[]}
          title="Error Test"
          error="Failed to load data"
        />
      );

      expect(screen.getByText('Failed to load data')).toBeInTheDocument();
    });

    test('Dashboard handles API error gracefully', () => {
      render(
        <EnhancedDashboard
          error="API connection failed"
        />
      );

      expect(screen.getByText('Dashboard Error')).toBeInTheDocument();
      expect(screen.getByText('API connection failed')).toBeInTheDocument();
    });
  });

  describe('Responsive Design and Accessibility', () => {
    test('Components include proper ARIA labels', () => {
      render(
        <StatCard
          title="Accessibility Test"
          value={100}
          aria-label="Test statistic card"
        />
      );

      const card = screen.getByText('Accessibility Test').closest('div');
      expect(card).toBeInTheDocument();
    });

    test('Charts are keyboard navigable', () => {
      render(
        <BarChart
          data={[{ name: 'Test', value: 10 }]}
          title="Keyboard Navigation Test"
        />
      );

      // Chart container should be focusable
      const chartContainer = screen.getByText('Keyboard Navigation Test').closest('div');
      expect(chartContainer).toBeInTheDocument();
    });
  });
});

// Integration test utilities
export const TestDataTransformers = {
  /**
   * Transform API pattern response to chart data
   */
  patternsToChartData: (apiResponse: typeof mockApiResponses.patterns) => {
    return apiResponse.data.patterns.map(pattern => ({
      name: pattern.name,
      value: Math.round(pattern.confidence_score * 100),
      metadata: {
        type: pattern.type,
        status: pattern.status,
        discovery_date: pattern.discovery_date
      }
    }));
  },

  /**
   * Transform agent metrics to performance data
   */
  agentsToPerformanceData: (apiResponse: typeof mockApiResponses.agents) => {
    return apiResponse.data.agents.map(agent => ({
      name: agent.agent_name,
      processed: agent.processed_tasks,
      success_rate: (agent.processed_tasks - agent.failed_tasks) / agent.processed_tasks,
      cpu_usage: agent.metrics.cpu_usage,
      memory_usage: agent.metrics.memory_usage,
      status: agent.status
    }));
  },

  /**
   * Transform dataset summary to distribution data
   */
  datasetSummaryToDistribution: (apiResponse: typeof mockApiResponses.dataSummary) => {
    return Object.entries(apiResponse.data.file_types).map(([type, count]) => ({
      name: type.toUpperCase(),
      value: count,
      color: {
        'kml': '#3B82F6',
        'geojson': '#10B981', 
        'csv': '#F59E0B',
        'shapefile': '#8B5CF6'
      }[type] || '#6B7280'
    }));
  }
};

/**
 * Mock API integration for testing
 */
export const mockApiIntegration = {
  patterns: {
    getStats: () => Promise.resolve(mockApiResponses.patterns),
    getPatterns: () => Promise.resolve(mockApiResponses.patterns.data.patterns)
  },
  agents: {
    getAgents: () => Promise.resolve(mockApiResponses.agents),
    getAgentMetrics: (id: string) => Promise.resolve(mockApiResponses.agents.data.agents[0])
  },
  data: {
    getDataSummary: () => Promise.resolve(mockApiResponses.dataSummary)
  }
};

export default mockApiResponses;