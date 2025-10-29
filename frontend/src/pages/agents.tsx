/**
 * A2A World Platform - Agent Management Page
 * 
 * Monitor agent status, manage agent lifecycle, view logs,
 * and display performance metrics with real-time updates.
 */

import Head from 'next/head';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { 
  Users, 
  Play, 
  Square, 
  RefreshCw, 
  Settings, 
  Activity, 
  AlertCircle,
  CheckCircle,
  Clock,
  Home,
  FileText,
  BarChart3,
  Cpu,
  Memory,
  Zap,
  Eye,
  Download,
  Filter,
  Search
} from 'lucide-react';

// Mock data interfaces
interface Agent {
  id: string;
  name: string;
  type: 'discovery' | 'validation' | 'monitoring' | 'narrative';
  status: 'active' | 'idle' | 'error' | 'maintenance' | 'stopped';
  health: 'healthy' | 'warning' | 'critical';
  version: string;
  uptime: string;
  configuration: {
    discovery_radius?: number;
    confidence_threshold?: number;
    processing_interval?: number;
    max_concurrent_tasks?: number;
  };
  metrics: {
    patterns_discovered: number;
    patterns_validated: number;
    tasks_completed: number;
    tasks_failed: number;
    avg_processing_time: number;
    uptime_percentage: number;
    memory_usage: number;
    cpu_usage: number;
  };
  last_heartbeat: string;
  created_at: string;
}

interface AgentTask {
  id: string;
  agent_id: string;
  type: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  progress: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

interface AgentLog {
  id: string;
  agent_id: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  timestamp: string;
  metadata?: any;
}

export default function Agents() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [agentTasks, setAgentTasks] = useState<AgentTask[]>([]);
  const [agentLogs, setAgentLogs] = useState<AgentLog[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'tasks' | 'logs' | 'config'>('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Mock data - In production this would fetch from the API
    setTimeout(() => {
      const mockAgents: Agent[] = [
        {
          id: 'agent-discovery-01',
          name: 'Pattern Discovery Agent',
          type: 'discovery',
          status: 'active',
          health: 'healthy',
          version: '1.2.3',
          uptime: '2d 14h 32m',
          configuration: {
            discovery_radius: 50000,
            confidence_threshold: 0.7,
            processing_interval: 300,
            max_concurrent_tasks: 5
          },
          metrics: {
            patterns_discovered: 24,
            patterns_validated: 0,
            tasks_completed: 156,
            tasks_failed: 3,
            avg_processing_time: 45.2,
            uptime_percentage: 98.5,
            memory_usage: 234.5,
            cpu_usage: 12.3
          },
          last_heartbeat: '2024-01-30T10:30:00Z',
          created_at: '2024-01-28T09:15:00Z'
        },
        {
          id: 'agent-validation-01',
          name: 'Pattern Validation Agent',
          type: 'validation',
          status: 'active',
          health: 'warning',
          version: '1.1.8',
          uptime: '1d 8h 15m',
          configuration: {
            confidence_threshold: 0.8,
            processing_interval: 600,
            max_concurrent_tasks: 3
          },
          metrics: {
            patterns_discovered: 0,
            patterns_validated: 18,
            tasks_completed: 87,
            tasks_failed: 8,
            avg_processing_time: 122.7,
            uptime_percentage: 94.2,
            memory_usage: 445.8,
            cpu_usage: 28.7
          },
          last_heartbeat: '2024-01-30T10:29:00Z',
          created_at: '2024-01-29T14:20:00Z'
        },
        {
          id: 'agent-monitoring-01',
          name: 'System Monitor Agent',
          type: 'monitoring',
          status: 'idle',
          health: 'healthy',
          version: '0.9.12',
          uptime: '12h 45m',
          configuration: {
            processing_interval: 30,
            max_concurrent_tasks: 10
          },
          metrics: {
            patterns_discovered: 0,
            patterns_validated: 0,
            tasks_completed: 1205,
            tasks_failed: 12,
            avg_processing_time: 2.1,
            uptime_percentage: 99.8,
            memory_usage: 128.3,
            cpu_usage: 5.4
          },
          last_heartbeat: '2024-01-30T10:30:00Z',
          created_at: '2024-01-30T22:00:00Z'
        },
        {
          id: 'agent-narrative-01',
          name: 'Cultural Narrative Agent',
          type: 'narrative',
          status: 'error',
          health: 'critical',
          version: '0.8.5',
          uptime: '0h 0m',
          configuration: {
            confidence_threshold: 0.6,
            processing_interval: 1800,
            max_concurrent_tasks: 2
          },
          metrics: {
            patterns_discovered: 0,
            patterns_validated: 0,
            tasks_completed: 42,
            tasks_failed: 15,
            avg_processing_time: 285.4,
            uptime_percentage: 67.3,
            memory_usage: 0,
            cpu_usage: 0
          },
          last_heartbeat: '2024-01-30T08:15:00Z',
          created_at: '2024-01-28T16:30:00Z'
        }
      ];

      setAgents(mockAgents);
      setSelectedAgent(mockAgents[0]);
      setIsLoading(false);

      // Mock tasks for selected agent
      setAgentTasks([
        {
          id: 'task-1',
          agent_id: 'agent-discovery-01',
          type: 'pattern_analysis',
          status: 'processing',
          priority: 'medium',
          progress: 65,
          started_at: '2024-01-30T10:15:00Z'
        },
        {
          id: 'task-2',
          agent_id: 'agent-discovery-01',
          type: 'data_validation',
          status: 'queued',
          priority: 'low',
          progress: 0
        },
        {
          id: 'task-3',
          agent_id: 'agent-discovery-01',
          type: 'pattern_discovery',
          status: 'completed',
          priority: 'high',
          progress: 100,
          started_at: '2024-01-30T09:30:00Z',
          completed_at: '2024-01-30T09:42:00Z'
        }
      ]);

      // Mock logs for selected agent
      setAgentLogs([
        {
          id: 'log-1',
          agent_id: 'agent-discovery-01',
          level: 'info',
          message: 'Starting pattern analysis task',
          timestamp: '2024-01-30T10:15:00Z'
        },
        {
          id: 'log-2',
          agent_id: 'agent-discovery-01',
          level: 'debug',
          message: 'Processing geospatial data batch 15/20',
          timestamp: '2024-01-30T10:12:00Z'
        },
        {
          id: 'log-3',
          agent_id: 'agent-discovery-01',
          level: 'warn',
          message: 'Low confidence pattern detected, flagging for review',
          timestamp: '2024-01-30T10:08:00Z'
        },
        {
          id: 'log-4',
          agent_id: 'agent-discovery-01',
          level: 'info',
          message: 'Pattern discovery task completed successfully',
          timestamp: '2024-01-30T09:42:00Z'
        }
      ]);
    }, 1000);
  }, []);

  const getStatusIcon = (status: string, health: string) => {
    if (status === 'error') return <AlertCircle className="h-4 w-4 text-red-500" />;
    if (health === 'critical') return <AlertCircle className="h-4 w-4 text-red-500" />;
    if (health === 'warning') return <AlertCircle className="h-4 w-4 text-yellow-500" />;
    if (status === 'active') return <CheckCircle className="h-4 w-4 text-green-500" />;
    if (status === 'idle') return <Clock className="h-4 w-4 text-blue-500" />;
    return <RefreshCw className="h-4 w-4 text-gray-500" />;
  };

  const getStatusColor = (status: string, health: string) => {
    if (status === 'error' || health === 'critical') return 'text-red-600 bg-red-50 border-red-200';
    if (health === 'warning') return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    if (status === 'active') return 'text-green-600 bg-green-50 border-green-200';
    if (status === 'idle') return 'text-blue-600 bg-blue-50 border-blue-200';
    return 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const getTypeColor = (type: string) => {
    const colors = {
      discovery: 'bg-purple-100 text-purple-800',
      validation: 'bg-green-100 text-green-800',
      monitoring: 'bg-blue-100 text-blue-800',
      narrative: 'bg-orange-100 text-orange-800'
    };
    return colors[type as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const getTaskStatusColor = (status: string) => {
    const colors = {
      queued: 'bg-gray-100 text-gray-800',
      processing: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800'
    };
    return colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const getLogLevelColor = (level: string) => {
    const colors = {
      debug: 'text-gray-600',
      info: 'text-blue-600',
      warn: 'text-yellow-600',
      error: 'text-red-600'
    };
    return colors[level as keyof typeof colors] || 'text-gray-600';
  };

  const controlAgent = async (agentId: string, action: 'start' | 'stop' | 'restart') => {
    // In production, this would call the API
    console.log(`${action} agent ${agentId}`);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const filteredAgents = agents.filter(agent => {
    const matchesSearch = searchQuery === '' || 
      agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      agent.type.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || agent.status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

  return (
    <>
      <Head>
        <title>Agent Management - A2A World Platform</title>
        <meta name="description" content="Monitor and manage A2A World platform agents" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                <Link href="/" className="flex items-center text-gray-600 hover:text-gray-900">
                  <Home className="h-5 w-5 mr-2" />
                  <span className="text-sm font-medium">Dashboard</span>
                </Link>
                <div className="flex items-center">
                  <Users className="h-6 w-6 text-primary-600 mr-2" />
                  <h1 className="text-lg font-semibold text-gray-900">Agent Management</h1>
                </div>
              </div>

              <nav className="hidden md:flex space-x-6">
                <Link href="/maps" className="text-gray-600 hover:text-gray-900">
                  Maps
                </Link>
                <Link href="/patterns" className="text-gray-600 hover:text-gray-900">
                  Patterns
                </Link>
                <Link href="/data" className="text-gray-600 hover:text-gray-900">
                  Data
                </Link>
              </nav>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Agent List */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow">
                <div className="p-4 border-b border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-medium text-gray-900">Agents</h2>
                    <span className="text-sm text-gray-500">
                      {agents.filter(a => a.status === 'active').length}/{agents.length} Active
                    </span>
                  </div>

                  {/* Search and Filter */}
                  <div className="space-y-3">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Search agents..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9 pr-4 py-2 w-full border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
                      />
                    </div>
                    
                    <select
                      value={statusFilter}
                      onChange={(e) => setStatusFilter(e.target.value)}
                      className="w-full border border-gray-300 rounded-md text-sm px-3 py-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                      <option value="all">All Status</option>
                      <option value="active">Active</option>
                      <option value="idle">Idle</option>
                      <option value="error">Error</option>
                      <option value="stopped">Stopped</option>
                    </select>
                  </div>
                </div>

                <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                  {isLoading ? (
                    <div className="p-4 text-center">
                      <RefreshCw className="h-6 w-6 animate-spin text-gray-400 mx-auto" />
                      <p className="mt-2 text-sm text-gray-500">Loading agents...</p>
                    </div>
                  ) : filteredAgents.length === 0 ? (
                    <div className="p-4 text-center">
                      <Users className="h-8 w-8 text-gray-400 mx-auto" />
                      <p className="mt-2 text-sm text-gray-500">No agents found</p>
                    </div>
                  ) : (
                    filteredAgents.map((agent) => (
                      <button
                        key={agent.id}
                        onClick={() => setSelectedAgent(agent)}
                        className={`w-full p-4 text-left hover:bg-gray-50 ${
                          selectedAgent?.id === agent.id ? 'bg-primary-50 border-r-2 border-primary-600' : ''
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-sm font-medium text-gray-900 truncate">
                            {agent.name}
                          </h3>
                          {getStatusIcon(agent.status, agent.health)}
                        </div>
                        
                        <div className="flex items-center space-x-2 mb-1">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getTypeColor(agent.type)}`}>
                            {agent.type}
                          </span>
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(agent.status, agent.health)}`}>
                            {agent.status}
                          </span>
                        </div>
                        
                        <div className="text-xs text-gray-500">
                          <div>Uptime: {agent.uptime}</div>
                          <div>CPU: {agent.metrics.cpu_usage}% | Memory: {agent.metrics.memory_usage}MB</div>
                        </div>
                      </button>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Agent Details */}
            <div className="lg:col-span-2">
              {selectedAgent ? (
                <div className="space-y-6">
                  {/* Agent Header */}
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div>
                          <h2 className="text-xl font-semibold text-gray-900">
                            {selectedAgent.name}
                          </h2>
                          <div className="flex items-center space-x-2 mt-1">
                            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getTypeColor(selectedAgent.type)}`}>
                              {selectedAgent.type}
                            </span>
                            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(selectedAgent.status, selectedAgent.health)}`}>
                              {selectedAgent.status}
                            </span>
                            <span className="text-xs text-gray-500">v{selectedAgent.version}</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => controlAgent(selectedAgent.id, 'start')}
                          disabled={selectedAgent.status === 'active'}
                          className="flex items-center px-3 py-2 text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-400 rounded-md"
                        >
                          <Play className="h-4 w-4 mr-1" />
                          Start
                        </button>
                        <button
                          onClick={() => controlAgent(selectedAgent.id, 'stop')}
                          disabled={selectedAgent.status === 'stopped' || selectedAgent.status === 'error'}
                          className="flex items-center px-3 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 disabled:bg-gray-400 rounded-md"
                        >
                          <Square className="h-4 w-4 mr-1" />
                          Stop
                        </button>
                        <button
                          onClick={() => controlAgent(selectedAgent.id, 'restart')}
                          className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 hover:bg-gray-50 rounded-md"
                        >
                          <RefreshCw className="h-4 w-4 mr-1" />
                          Restart
                        </button>
                      </div>
                    </div>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900">
                          {selectedAgent.metrics.patterns_discovered + selectedAgent.metrics.patterns_validated}
                        </div>
                        <div className="text-sm text-gray-600">Patterns Processed</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900">
                          {selectedAgent.metrics.tasks_completed}
                        </div>
                        <div className="text-sm text-gray-600">Tasks Completed</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900">
                          {selectedAgent.metrics.uptime_percentage}%
                        </div>
                        <div className="text-sm text-gray-600">Uptime</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900">
                          {selectedAgent.metrics.avg_processing_time}s
                        </div>
                        <div className="text-sm text-gray-600">Avg Process Time</div>
                      </div>
                    </div>
                  </div>

                  {/* Tab Navigation */}
                  <div className="bg-white rounded-lg shadow">
                    <div className="border-b border-gray-200">
                      <nav className="-mb-px flex space-x-8 px-6">
                        {[
                          { id: 'overview', label: 'Overview', icon: Activity },
                          { id: 'tasks', label: 'Tasks', icon: BarChart3 },
                          { id: 'logs', label: 'Logs', icon: FileText },
                          { id: 'config', label: 'Config', icon: Settings }
                        ].map(tab => (
                          <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                              activeTab === tab.id
                                ? 'border-primary-500 text-primary-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                          >
                            <tab.icon className="h-4 w-4 mr-2" />
                            {tab.label}
                          </button>
                        ))}
                      </nav>
                    </div>

                    {/* Tab Content */}
                    <div className="p-6">
                      {activeTab === 'overview' && (
                        <div className="space-y-6">
                          {/* Resource Usage */}
                          <div>
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Resource Usage</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="p-4 border rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center">
                                    <Cpu className="h-4 w-4 text-blue-500 mr-2" />
                                    <span className="text-sm font-medium">CPU Usage</span>
                                  </div>
                                  <span className="text-sm text-gray-600">
                                    {selectedAgent.metrics.cpu_usage}%
                                  </span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div 
                                    className="bg-blue-600 h-2 rounded-full" 
                                    style={{ width: `${selectedAgent.metrics.cpu_usage}%` }}
                                  />
                                </div>
                              </div>

                              <div className="p-4 border rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center">
                                    <Memory className="h-4 w-4 text-green-500 mr-2" />
                                    <span className="text-sm font-medium">Memory Usage</span>
                                  </div>
                                  <span className="text-sm text-gray-600">
                                    {selectedAgent.metrics.memory_usage} MB
                                  </span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div 
                                    className="bg-green-600 h-2 rounded-full" 
                                    style={{ width: `${Math.min((selectedAgent.metrics.memory_usage / 1024) * 100, 100)}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Performance Stats */}
                          <div>
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Performance</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                              <div>
                                <div className="flex justify-between py-2">
                                  <span className="text-gray-600">Last Heartbeat:</span>
                                  <span className="font-medium">{formatTimestamp(selectedAgent.last_heartbeat)}</span>
                                </div>
                                <div className="flex justify-between py-2">
                                  <span className="text-gray-600">Created:</span>
                                  <span className="font-medium">{formatTimestamp(selectedAgent.created_at)}</span>
                                </div>
                                <div className="flex justify-between py-2">
                                  <span className="text-gray-600">Failed Tasks:</span>
                                  <span className="font-medium">{selectedAgent.metrics.tasks_failed}</span>
                                </div>
                              </div>
                              <div>
                                <div className="flex justify-between py-2">
                                  <span className="text-gray-600">Success Rate:</span>
                                  <span className="font-medium">
                                    {Math.round((selectedAgent.metrics.tasks_completed / (selectedAgent.metrics.tasks_completed + selectedAgent.metrics.tasks_failed)) * 100)}%
                                  </span>
                                </div>
                                <div className="flex justify-between py-2">
                                  <span className="text-gray-600">Avg Processing:</span>
                                  <span className="font-medium">{selectedAgent.metrics.avg_processing_time}s</span>
                                </div>
                                <div className="flex justify-between py-2">
                                  <span className="text-gray-600">Uptime:</span>
                                  <span className="font-medium">{selectedAgent.uptime}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {activeTab === 'tasks' && (
                        <div>
                          <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-medium text-gray-900">Recent Tasks</h3>
                            <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                              View All
                            </button>
                          </div>
                          
                          <div className="space-y-3">
                            {agentTasks.map((task) => (
                              <div key={task.id} className="border rounded-lg p-4">
                                <div className="flex items-center justify-between mb-2">
                                  <div>
                                    <h4 className="text-sm font-medium text-gray-900">
                                      {task.type.replace('_', ' ').toUpperCase()}
                                    </h4>
                                    <p className="text-xs text-gray-500">Task ID: {task.id}</p>
                                  </div>
                                  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getTaskStatusColor(task.status)}`}>
                                    {task.status}
                                  </span>
                                </div>
                                
                                {task.status === 'processing' && (
                                  <div className="mb-2">
                                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                                      <span>Progress</span>
                                      <span>{task.progress}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-1">
                                      <div 
                                        className="bg-blue-600 h-1 rounded-full transition-all duration-300" 
                                        style={{ width: `${task.progress}%` }}
                                      />
                                    </div>
                                  </div>
                                )}
                                
                                <div className="flex items-center justify-between text-xs text-gray-500">
                                  <span className="capitalize">Priority: {task.priority}</span>
                                  <span>
                                    {task.started_at && `Started: ${formatTimestamp(task.started_at)}`}
                                    {task.completed_at && ` | Completed: ${formatTimestamp(task.completed_at)}`}
                                  </span>
                                </div>
                                
                                {task.error_message && (
                                  <div className="mt-2 p-2 bg-red-50 text-red-700 text-xs rounded">
                                    {task.error_message}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {activeTab === 'logs' && (
                        <div>
                          <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-medium text-gray-900">Recent Logs</h3>
                            <div className="flex items-center space-x-2">
                              <select className="text-sm border border-gray-300 rounded px-2 py-1">
                                <option value="all">All Levels</option>
                                <option value="error">Error</option>
                                <option value="warn">Warning</option>
                                <option value="info">Info</option>
                                <option value="debug">Debug</option>
                              </select>
                              <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                                <Download className="h-4 w-4" />
                              </button>
                            </div>
                          </div>
                          
                          <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm max-h-96 overflow-y-auto">
                            {agentLogs.map((log) => (
                              <div key={log.id} className="flex items-start space-x-3 mb-2">
                                <span className="text-gray-400 text-xs whitespace-nowrap">
                                  {formatTimestamp(log.timestamp)}
                                </span>
                                <span className={`uppercase text-xs font-medium ${getLogLevelColor(log.level)} whitespace-nowrap`}>
                                  {log.level}
                                </span>
                                <span className="text-gray-300 flex-1">
                                  {log.message}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {activeTab === 'config' && (
                        <div>
                          <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-medium text-gray-900">Configuration</h3>
                            <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                              Edit Config
                            </button>
                          </div>
                          
                          <div className="space-y-4">
                            {Object.entries(selectedAgent.configuration).map(([key, value]) => (
                              <div key={key} className="flex items-center justify-between py-2 border-b border-gray-200">
                                <span className="text-gray-600 capitalize">
                                  {key.replace('_', ' ')}
                                </span>
                                <span className="font-medium">
                                  {typeof value === 'number' ? value.toLocaleString() : value}
                                  {key.includes('threshold') && typeof value === 'number' && value < 1 && ' (%)'}
                                  {key.includes('interval') && ' sec'}
                                  {key.includes('radius') && ' m'}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow p-8 text-center">
                  <Users className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No Agent Selected</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Select an agent from the list to view details
                  </p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </>
  );
}