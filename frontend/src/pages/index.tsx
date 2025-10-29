/**
 * A2A World Platform - Dashboard Page
 *
 * Main dashboard showing system overview, agent status, recent discoveries,
 * and quick actions for the A2A World platform with comprehensive A2A branding.
 */

import Link from 'next/link';
import { useState, useEffect } from 'react';
import {
  Activity,
  Brain,
  MapPin,
  Database,
  Users,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Play,
  RefreshCw
} from 'lucide-react';
import Layout from '../components/layout/Layout';
import { MetricCard, BrandedCard } from '../components/branding';
import { LoadingSpinner, InlineLoading } from '../components/branding';

// Mock data interfaces
interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime: string;
  version: string;
  services: {
    api: 'online' | 'offline';
    database: 'online' | 'offline';
    messaging: 'online' | 'offline';
    agents: 'online' | 'offline';
  };
}

interface AgentSummary {
  total: number;
  active: number;
  idle: number;
  error: number;
}

interface RecentPattern {
  id: string;
  name: string;
  type: string;
  confidence: number;
  discoveredAt: string;
}

export default function Dashboard() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [agentSummary, setAgentSummary] = useState<AgentSummary | null>(null);
  const [recentPatterns, setRecentPatterns] = useState<RecentPattern[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Mock data - In production this would fetch from the API
    setTimeout(() => {
      setSystemHealth({
        status: 'healthy',
        uptime: '2d 14h 32m',
        version: '0.1.0',
        services: {
          api: 'online',
          database: 'online',
          messaging: 'online',
          agents: 'online'
        }
      });

      setAgentSummary({
        total: 4,
        active: 3,
        idle: 1,
        error: 0
      });

      setRecentPatterns([
        {
          id: '1',
          name: 'Sacred Geometry Alignment',
          type: 'geometric',
          confidence: 0.89,
          discoveredAt: '2 hours ago'
        },
        {
          id: '2',
          name: 'Ceremonial Site Cluster',
          type: 'cultural',
          confidence: 0.76,
          discoveredAt: '4 hours ago'
        },
        {
          id: '3',
          name: 'Astronomical Correlation',
          type: 'temporal',
          confidence: 0.93,
          discoveredAt: '6 hours ago'
        }
      ]);

      setIsLoading(false);
    }, 1000);
  }, []);

  const StatusIndicator = ({ status }: { status: 'healthy' | 'degraded' | 'unhealthy' }) => {
    const colors = {
      healthy: 'bg-green-500',
      degraded: 'bg-yellow-500',
      unhealthy: 'bg-red-500'
    };
    
    return <div className={`w-3 h-3 rounded-full ${colors[status]} animate-pulse`} />;
  };

  return (
    <Layout
      title="Dashboard"
      description="A2A World Platform Dashboard - Monitor system status, agents, and pattern discoveries"
      footerVariant="default"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <BrandedCard
            title="Welcome to A2A World"
            subtitle="Advanced Pattern Discovery Platform"
            showBranding={true}
            brandingPosition="header"
            variant="gradient"
            padding="lg"
          >
            <p className="text-gray-600 text-sm">
              Discover hidden connections and patterns across ancient sites and sacred locations
              using cutting-edge AI and geospatial analysis.
            </p>
          </BrandedCard>
        </div>

        {/* System Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="System Status"
            value={systemHealth?.status || 'Loading...'}
            subtitle={systemHealth ? `Uptime: ${systemHealth.uptime}` : 'Checking status...'}
            icon={Activity}
            color="green"
            trend={{
              value: 0,
              label: 'stable',
              direction: 'neutral'
            }}
          />

          <MetricCard
            title="Active Agents"
            value={agentSummary ? `${agentSummary.active}/${agentSummary.total}` : 'Loading...'}
            subtitle="AI processing agents"
            icon={Users}
            color="blue"
            trend={agentSummary ? {
              value: 0,
              label: 'agents running',
              direction: 'neutral'
            } : undefined}
          />

          <MetricCard
            title="Patterns Found"
            value={recentPatterns.length.toString()}
            subtitle="Recent discoveries"
            icon={Brain}
            color="purple"
            trend={{
              value: 23.5,
              label: 'this week',
              direction: 'up'
            }}
          />

          <MetricCard
            title="System Health"
            value="Optimal"
            subtitle={systemHealth?.uptime || 'Loading...'}
            icon={CheckCircle}
            color="green"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Pattern Discoveries */}
          <div className="lg:col-span-2">
            <BrandedCard
              title="Recent Discoveries"
              showBranding={true}
              brandingPosition="corner"
              action={
                <Link
                  href="/patterns"
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                >
                  View All
                </Link>
              }
            >
              {isLoading ? (
                <LoadingSpinner
                  variant="dots"
                  text="Loading recent discoveries..."
                  size="md"
                />
              ) : recentPatterns.length > 0 ? (
                <div className="space-y-4">
                  {recentPatterns.map((pattern) => (
                    <div key={pattern.id} className="flex items-center space-x-4 p-3 hover:bg-gray-50 rounded-lg transition-colors">
                      <div className="flex-shrink-0">
                        <Brain className="h-8 w-8 text-purple-500" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {pattern.name}
                        </p>
                        <div className="flex items-center space-x-2 text-xs text-gray-500">
                          <span className="capitalize">{pattern.type}</span>
                          <span>•</span>
                          <span className={`font-medium ${
                            pattern.confidence >= 0.8 ? 'text-green-600' :
                            pattern.confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                          }`}>
                            {Math.round(pattern.confidence * 100)}% confidence
                          </span>
                          <span>•</span>
                          <span>{pattern.discoveredAt}</span>
                        </div>
                      </div>
                      <div className="flex-shrink-0">
                        <TrendingUp className="h-4 w-4 text-green-500" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Brain className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No patterns discovered</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Start agents to begin pattern discovery
                  </p>
                </div>
              )}
            </BrandedCard>
          </div>

          {/* System Services & Quick Actions */}
          <div className="space-y-6">
            {/* Services Status */}
            <BrandedCard
              title="Services"
              showBranding={true}
              brandingPosition="corner"
            >
              {systemHealth ? (
                <div className="space-y-3">
                  {Object.entries(systemHealth.services).map(([service, status]) => (
                    <div key={service} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
                      <span className="text-sm text-gray-700 capitalize font-medium">{service}</span>
                      <div className="flex items-center">
                        {status === 'online' ? (
                          <CheckCircle className="h-4 w-4 text-green-500 mr-1" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-red-500 mr-1" />
                        )}
                        <span className={`text-xs font-medium ${
                          status === 'online' ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <InlineLoading text="Loading services..." />
              )}
            </BrandedCard>

            {/* Quick Actions */}
            <BrandedCard
              title="Quick Actions"
              showBranding={true}
              brandingPosition="corner"
            >
              <div className="space-y-2">
                <Link
                  href="/maps"
                  className="flex items-center p-3 rounded-lg hover:bg-blue-50 hover:border-blue-200 border border-transparent transition-all"
                >
                  <MapPin className="h-5 w-5 text-blue-600 mr-3" />
                  <div>
                    <span className="text-sm font-medium text-gray-900">View Interactive Map</span>
                    <div className="text-xs text-gray-500">Explore geospatial patterns</div>
                  </div>
                </Link>
                
                <Link
                  href="/data"
                  className="flex items-center p-3 rounded-lg hover:bg-green-50 hover:border-green-200 border border-transparent transition-all"
                >
                  <Database className="h-5 w-5 text-green-600 mr-3" />
                  <div>
                    <span className="text-sm font-medium text-gray-900">Upload Dataset</span>
                    <div className="text-xs text-gray-500">Add new data sources</div>
                  </div>
                </Link>
                
                <Link
                  href="/agents"
                  className="flex items-center p-3 rounded-lg hover:bg-purple-50 hover:border-purple-200 border border-transparent transition-all"
                >
                  <Play className="h-5 w-5 text-purple-600 mr-3" />
                  <div>
                    <span className="text-sm font-medium text-gray-900">Manage Agents</span>
                    <div className="text-xs text-gray-500">Control AI processing</div>
                  </div>
                </Link>
                
                <button className="w-full flex items-center p-3 rounded-lg hover:bg-gray-50 hover:border-gray-200 border border-transparent transition-all">
                  <RefreshCw className="h-5 w-5 text-gray-600 mr-3" />
                  <div className="text-left">
                    <span className="text-sm font-medium text-gray-900">Refresh Data</span>
                    <div className="text-xs text-gray-500">Update all metrics</div>
                  </div>
                </button>
              </div>
            </BrandedCard>
          </div>
        </div>
      </div>
    </Layout>
  );
}