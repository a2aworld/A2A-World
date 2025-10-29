/**
 * A2A World Platform - Validation Dashboard Component
 * 
 * Comprehensive dashboard for monitoring statistical validation results,
 * performance metrics, and validation analytics.
 */

import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import { BrandedCard, MetricCard } from '../branding/BrandedCard';
import { BrandedChart } from '../branding/BrandedChart';
import { StatisticalValidation } from './StatisticalValidation';
import { ValidationReports } from './ValidationReports';

export interface ValidationDashboardProps {
  /**
   * Show detailed view
   */
  showDetails?: boolean;
  /**
   * Auto-refresh interval in seconds
   */
  refreshInterval?: number;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export interface DashboardData {
  overview_metrics: {
    total_validations: number;
    highly_significant_patterns: number;
    avg_reliability_score: number;
    validation_success_rate: number;
    active_validation_agents: number;
  };
  significance_indicators: Array<{
    test_name: string;
    significance_level: string;
    p_value: number;
    color: string;
    significant: boolean;
    effect_size?: number;
  }>;
  recent_validations: Array<{
    pattern_id: string;
    pattern_name: string;
    overall_significance: string;
    reliability_score: number;
    validation_timestamp: string;
    processing_time_ms: number;
  }>;
  performance_trends: {
    validation_rate_trend: string;
    significance_rate_trend: string;
    processing_time_trend: string;
  };
  alerts: Array<{
    type: string;
    message: string;
    priority: string;
  }>;
  charts?: {
    significance_distribution: {
      type: string;
      data: { labels: string[]; values: number[] };
      title: string;
    };
    validation_timeline: {
      type: string;
      data: { dates: string[]; validations: number[] };
      title: string;
    };
    reliability_scores: {
      type: string;
      data: { bins: string[]; counts: number[] };
      title: string;
    };
  };
}

export function ValidationDashboard({
  showDetails = true,
  refreshInterval = 300,
  className
}: ValidationDashboardProps) {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // Mock data for demonstration
  const mockDashboardData: DashboardData = {
    overview_metrics: {
      total_validations: 0,
      highly_significant_patterns: 0,
      avg_reliability_score: 0.0,
      validation_success_rate: 0.0,
      active_validation_agents: 1
    },
    significance_indicators: [],
    recent_validations: [],
    performance_trends: {
      validation_rate_trend: "stable",
      significance_rate_trend: "stable", 
      processing_time_trend: "stable"
    },
    alerts: [
      {
        type: "info",
        message: "Enhanced statistical validation framework is ready for use",
        priority: "low"
      },
      {
        type: "success",
        message: "Phase 3 statistical validation implementation completed",
        priority: "medium"
      }
    ],
    charts: {
      significance_distribution: {
        type: "pie",
        data: { labels: [], values: [] },
        title: "Pattern Significance Distribution"
      },
      validation_timeline: {
        type: "line",
        data: { dates: [], validations: [] },
        title: "Validation Activity Over Time"
      },
      reliability_scores: {
        type: "histogram",
        data: { bins: [], counts: [] },
        title: "Reliability Score Distribution"
      }
    }
  };

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // In a real implementation, this would call the API
      // const response = await fetch('/api/v1/validation/dashboard/data');
      // const data = await response.json();
      
      // For now, use mock data
      setDashboardData(mockDashboardData);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchDashboardData, refreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [refreshInterval]);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return '↗️';
      case 'down':
        return '↘️';
      default:
        return '➡️';
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  if (loading && !dashboardData) {
    return (
      <div className={clsx('space-y-6', className)}>
        <BrandedCard
          title="Validation Dashboard"
          subtitle="Loading dashboard data..."
          showBranding={true}
          brandingPosition="corner"
        >
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-blue-600 border-t-transparent" />
              <span className="text-lg text-gray-600">Loading validation analytics...</span>
            </div>
          </div>
        </BrandedCard>
      </div>
    );
  }

  if (error) {
    return (
      <div className={clsx('space-y-6', className)}>
        <BrandedCard
          title="Validation Dashboard"
          subtitle="Dashboard Error"
          showBranding={true}
          brandingPosition="corner"
        >
          <div className="text-center py-12">
            <div className="text-red-500 mb-4">
              <svg className="h-12 w-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-lg font-medium text-gray-900">Dashboard Load Failed</p>
            <p className="text-sm text-gray-600 mt-2">{error}</p>
            <button
              onClick={fetchDashboardData}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Retry
            </button>
          </div>
        </BrandedCard>
      </div>
    );
  }

  const data = dashboardData!;

  return (
    <div className={clsx('space-y-6', className)}>
      {/* Header */}
      <BrandedCard
        title="Statistical Validation Dashboard"
        subtitle={`Last updated: ${lastRefresh.toLocaleTimeString()}`}
        action={
          <button
            onClick={fetchDashboardData}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100 transition-colors"
            title="Refresh dashboard"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        }
        showBranding={true}
        brandingPosition="corner"
        variant="bordered"
      >
        {/* Overview Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <MetricCard
            title="Total Validations"
            value={data.overview_metrics.total_validations.toLocaleString()}
            color="blue"
            icon={ValidationsIcon}
          />
          
          <MetricCard
            title="Highly Significant"
            value={data.overview_metrics.highly_significant_patterns.toLocaleString()}
            subtitle={`${data.overview_metrics.total_validations > 0 ? 
              Math.round((data.overview_metrics.highly_significant_patterns / data.overview_metrics.total_validations) * 100) : 0}% of total`}
            color="green"
            icon={SignificantIcon}
          />
          
          <MetricCard
            title="Avg Reliability"
            value={`${Math.round(data.overview_metrics.avg_reliability_score * 100)}%`}
            color="purple"
            icon={ReliabilityIcon}
          />
          
          <MetricCard
            title="Success Rate"
            value={`${Math.round(data.overview_metrics.validation_success_rate * 100)}%`}
            color="yellow"
            icon={SuccessIcon}
          />
          
          <MetricCard
            title="Active Agents"
            value={data.overview_metrics.active_validation_agents.toString()}
            color="gray"
            icon={AgentIcon}
          />
        </div>
      </BrandedCard>

      {/* Performance Trends */}
      <BrandedCard
        title="Performance Trends"
        variant="bordered"
        showBranding={true}
        brandingPosition="corner"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl mb-2">
              {getTrendIcon(data.performance_trends.validation_rate_trend)}
            </div>
            <div className="font-semibold text-gray-900">Validation Rate</div>
            <div className="text-sm text-gray-600 capitalize">
              {data.performance_trends.validation_rate_trend}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl mb-2">
              {getTrendIcon(data.performance_trends.significance_rate_trend)}
            </div>
            <div className="font-semibold text-gray-900">Significance Rate</div>
            <div className="text-sm text-gray-600 capitalize">
              {data.performance_trends.significance_rate_trend}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl mb-2">
              {getTrendIcon(data.performance_trends.processing_time_trend)}
            </div>
            <div className="font-semibold text-gray-900">Processing Time</div>
            <div className="text-sm text-gray-600 capitalize">
              {data.performance_trends.processing_time_trend}
            </div>
          </div>
        </div>
      </BrandedCard>

      {/* Charts */}
      {showDetails && data.charts && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <BrandedChart
            title={data.charts.significance_distribution.title}
            showBranding={true}
            dataSource="Statistical Validation Framework"
          >
            <div className="flex items-center justify-center h-64">
              <div className="text-center text-gray-500">
                <svg className="h-16 w-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <p>No validation data available</p>
                <p className="text-sm">Run statistical validations to see distribution</p>
              </div>
            </div>
          </BrandedChart>
          
          <BrandedChart
            title={data.charts.validation_timeline.title}
            showBranding={true}
            dataSource="Statistical Validation Framework"
          >
            <div className="flex items-center justify-center h-64">
              <div className="text-center text-gray-500">
                <svg className="h-16 w-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                <p>No validation history available</p>
                <p className="text-sm">Validation activity will appear here</p>
              </div>
            </div>
          </BrandedChart>
        </div>
      )}

      {/* Recent Validations */}
      {data.recent_validations.length > 0 && (
        <BrandedCard
          title="Recent Validations"
          variant="bordered"
          showBranding={true}
          brandingPosition="corner"
        >
          <div className="space-y-3">
            {data.recent_validations.map((validation, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <div className="font-semibold text-gray-900">{validation.pattern_name}</div>
                  <div className="text-sm text-gray-600">Pattern ID: {validation.pattern_id}</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold capitalize">
                    {validation.overall_significance.replace('_', ' ')}
                  </div>
                  <div className="text-sm text-gray-600">
                    {Math.round(validation.reliability_score * 100)}% reliable
                  </div>
                </div>
                <div className="text-right text-sm text-gray-600">
                  <div>{new Date(validation.validation_timestamp).toLocaleDateString()}</div>
                  <div>{validation.processing_time_ms}ms</div>
                </div>
              </div>
            ))}
          </div>
        </BrandedCard>
      )}

      {/* Alerts */}
      {data.alerts.length > 0 && (
        <BrandedCard
          title="System Alerts"
          variant="bordered"
          showBranding={true}
          brandingPosition="corner"
        >
          <div className="space-y-3">
            {data.alerts.map((alert, index) => (
              <div key={index} className={clsx(
                'p-3 border rounded-lg',
                getAlertColor(alert.type)
              )}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="font-medium">{alert.message}</div>
                  </div>
                  <div className="text-xs uppercase font-semibold">
                    {alert.priority}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </BrandedCard>
      )}
    </div>
  );
}

// Icons for metrics
function ValidationsIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
    </svg>
  );
}

function SignificantIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  );
}

function ReliabilityIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  );
}

function SuccessIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function AgentIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );
}

export default ValidationDashboard;