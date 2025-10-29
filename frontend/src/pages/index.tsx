/**
 * A2A World Platform - Dashboard Page
 *
 * Main dashboard showing system overview, agent status, recent discoveries,
 * and quick actions for the A2A World platform.
 */

import Head from 'next/head';
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
  Zap,
  Globe,
  Settings,
  Play,
  RefreshCw
} from 'lucide-react';

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
    <>
      <Head>
        <title>Dashboard - A2A World Platform</title>
        <meta name="description" content="A2A World Platform Dashboard - Monitor system status, agents, and pattern discoveries" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <Globe className="h-8 w-8 text-primary-600 mr-3" />
                <div>
                  <h1 className="text-xl font-bold text-gray-900">A2A World</h1>
                  <p className="text-sm text-gray-500">Pattern Discovery Platform</p>
                </div>
              </div>
              
              <nav className="hidden md:flex space-x-6">
                <Link href="/" className="text-primary-600 font-medium">
                  Dashboard
                </Link>
                <Link href="/maps" className="text-gray-600 hover:text-gray-900">
                  Maps
                </Link>
                <Link href="/patterns" className="text-gray-600 hover:text-gray-900">
                  Patterns
                </Link>
                <Link href="/agents" className="text-gray-600 hover:text-gray-900">
                  Agents
                </Link>
                <Link href="/data" className="text-gray-600 hover:text-gray-900">
                  Data
                </Link>
              </nav>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* System Status Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  {systemHealth ? (
                    <StatusIndicator status={systemHealth.status} />
                  ) : (
                    <div className="w-3 h-3 bg-gray-300 rounded-full animate-pulse" />
                  )}
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">System Status</p>
                  <p className="text-lg font-bold text-gray-700 capitalize">
                    {systemHealth?.status || 'Loading...'}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <Users className="h-8 w-8 text-blue-500" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">Active Agents</p>
                  <p className="text-lg font-bold text-gray-700">
                    {agentSummary ? `${agentSummary.active}/${agentSummary.total}` : 'Loading...'}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <Brain className="h-8 w-8 text-purple-500" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">Patterns Found</p>
                  <p className="text-lg font-bold text-gray-700">
                    {recentPatterns.length}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-green-500" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">Uptime</p>
                  <p className="text-lg font-bold text-gray-700">
                    {systemHealth?.uptime || 'Loading...'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Recent Pattern Discoveries */}
            <div className="lg:col-span-2 bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-medium text-gray-900">Recent Discoveries</h3>
                  <Link
                    href="/patterns"
                    className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                  >
                    View All
                  </Link>
                </div>
              </div>
              <div className="p-6">
                {isLoading ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map(i => (
                      <div key={i} className="animate-pulse flex space-x-4">
                        <div className="rounded-full bg-gray-200 h-10 w-10"></div>
                        <div className="flex-1 space-y-2 py-1">
                          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                          <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : recentPatterns.length > 0 ? (
                  <div className="space-y-4">
                    {recentPatterns.map((pattern) => (
                      <div key={pattern.id} className="flex items-center space-x-4">
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
                            <span>{Math.round(pattern.confidence * 100)}% confidence</span>
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
              </div>
            </div>

            {/* System Services & Quick Actions */}
            <div className="space-y-6">
              {/* Services Status */}
              <div className="bg-white rounded-lg shadow">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-medium text-gray-900">Services</h3>
                </div>
                <div className="p-6">
                  {systemHealth ? (
                    <div className="space-y-3">
                      {Object.entries(systemHealth.services).map(([service, status]) => (
                        <div key={service} className="flex items-center justify-between">
                          <span className="text-sm text-gray-700 capitalize">{service}</span>
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
                    <div className="animate-pulse space-y-3">
                      {[1, 2, 3, 4].map(i => (
                        <div key={i} className="flex items-center justify-between">
                          <div className="h-4 bg-gray-200 rounded w-20"></div>
                          <div className="h-4 bg-gray-200 rounded w-16"></div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Quick Actions */}
              <div className="bg-white rounded-lg shadow">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-medium text-gray-900">Quick Actions</h3>
                </div>
                <div className="p-6 space-y-3">
                  <Link
                    href="/maps"
                    className="flex items-center p-3 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <MapPin className="h-5 w-5 text-primary-600 mr-3" />
                    <span className="text-sm font-medium text-gray-700">View Map</span>
                  </Link>
                  
                  <Link
                    href="/data"
                    className="flex items-center p-3 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <Database className="h-5 w-5 text-primary-600 mr-3" />
                    <span className="text-sm font-medium text-gray-700">Upload Data</span>
                  </Link>
                  
                  <Link
                    href="/agents"
                    className="flex items-center p-3 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <Play className="h-5 w-5 text-primary-600 mr-3" />
                    <span className="text-sm font-medium text-gray-700">Manage Agents</span>
                  </Link>
                  
                  <button className="w-full flex items-center p-3 rounded-lg hover:bg-gray-50 transition-colors">
                    <RefreshCw className="h-5 w-5 text-primary-600 mr-3" />
                    <span className="text-sm font-medium text-gray-700">Refresh Data</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}