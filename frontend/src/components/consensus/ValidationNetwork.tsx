/**
 * A2A World Platform - Validation Network Component
 * 
 * Visualizes the consensus network topology, agent connections, and real-time
 * consensus flow between validation agents and coordinators.
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Network, 
  Users, 
  Shield, 
  Activity, 
  Wifi, 
  WifiOff,
  Zap,
  AlertCircle,
  CheckCircle,
  Clock,
  Settings,
  Maximize2,
  Minimize2
} from 'lucide-react';
import { BrandedCard } from '../branding/BrandedCard';
import { LoadingSpinner } from '../branding/LoadingSpinner';

// Types for network data
interface NetworkNode {
  id: string;
  type: 'coordinator' | 'validation_agent' | 'observer';
  status: 'online' | 'offline' | 'degraded';
  position?: { x: number; y: number };
  reputation_score?: number;
  capabilities: string[];
  last_seen: string;
  metrics: {
    response_time_ms: number;
    success_rate: number;
    active_validations: number;
  };
}

interface NetworkConnection {
  source: string;
  target: string;
  type: 'peer' | 'coordinator' | 'backup';
  status: 'active' | 'inactive' | 'failed';
  latency_ms: number;
  bandwidth_mbps: number;
  message_count: number;
}

interface NetworkStatus {
  total_nodes: number;
  active_nodes: number;
  consensus_coordinators: number;
  validation_agents: number;
  network_health: 'healthy' | 'degraded' | 'critical';
  average_latency_ms: number;
}

interface ValidationNetworkProps {
  autoRefresh?: boolean;
  refreshInterval?: number;
  showMetrics?: boolean;
  interactive?: boolean;
}

export const ValidationNetwork: React.FC<ValidationNetworkProps> = ({
  autoRefresh = true,
  refreshInterval = 30000,
  showMetrics = true,
  interactive = true
}) => {
  const [networkData, setNetworkData] = useState<{
    nodes: NetworkNode[];
    connections: NetworkConnection[];
    status: NetworkStatus;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const networkCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchNetworkData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchNetworkData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  useEffect(() => {
    if (networkData && networkCanvasRef.current) {
      renderNetworkVisualization();
    }
  }, [networkData, isFullscreen]);

  const fetchNetworkData = async () => {
    try {
      setError(null);
      
      const response = await fetch('/api/v1/consensus/network');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const status = await response.json();
      
      // Simulate network topology data (in real implementation, this would come from the API)
      const simulatedData = generateSimulatedNetworkData(status);
      setNetworkData(simulatedData);
    } catch (err) {
      console.error('Error fetching network data:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const generateSimulatedNetworkData = (status: NetworkStatus) => {
    const nodes: NetworkNode[] = [
      {
        id: 'coordinator_1',
        type: 'coordinator',
        status: 'online',
        position: { x: 400, y: 200 },
        capabilities: ['bft_consensus', 'raft_consensus', 'voting'],
        last_seen: new Date().toISOString(),
        metrics: {
          response_time_ms: 25,
          success_rate: 0.98,
          active_validations: 5
        }
      },
      {
        id: 'coordinator_2',
        type: 'coordinator',
        status: 'online',
        position: { x: 600, y: 200 },
        capabilities: ['bft_consensus', 'raft_consensus'],
        last_seen: new Date().toISOString(),
        metrics: {
          response_time_ms: 32,
          success_rate: 0.95,
          active_validations: 3
        }
      }
    ];

    // Add validation agents
    const agentPositions = [
      { x: 200, y: 100 }, { x: 300, y: 350 }, { x: 500, y: 350 },
      { x: 700, y: 350 }, { x: 800, y: 100 }, { x: 150, y: 250 }
    ];

    agentPositions.forEach((pos, index) => {
      nodes.push({
        id: `agent_${index + 1}`,
        type: 'validation_agent',
        status: index === 5 ? 'degraded' : 'online',
        position: pos,
        reputation_score: 0.7 + (Math.random() * 0.3),
        capabilities: ['statistical_validation', 'consensus_participation'],
        last_seen: new Date().toISOString(),
        metrics: {
          response_time_ms: 15 + Math.random() * 50,
          success_rate: 0.85 + Math.random() * 0.15,
          active_validations: Math.floor(Math.random() * 3)
        }
      });
    });

    // Generate connections
    const connections: NetworkConnection[] = [];
    
    // Connect agents to coordinators
    nodes.filter(n => n.type === 'validation_agent').forEach(agent => {
      nodes.filter(n => n.type === 'coordinator').forEach(coordinator => {
        connections.push({
          source: agent.id,
          target: coordinator.id,
          type: 'coordinator',
          status: agent.status === 'online' && coordinator.status === 'online' ? 'active' : 'inactive',
          latency_ms: 20 + Math.random() * 80,
          bandwidth_mbps: 10 + Math.random() * 90,
          message_count: Math.floor(Math.random() * 100)
        });
      });
    });

    // Add peer connections between agents
    const agents = nodes.filter(n => n.type === 'validation_agent');
    for (let i = 0; i < agents.length - 1; i++) {
      connections.push({
        source: agents[i].id,
        target: agents[i + 1].id,
        type: 'peer',
        status: 'active',
        latency_ms: 15 + Math.random() * 30,
        bandwidth_mbps: 50 + Math.random() * 50,
        message_count: Math.floor(Math.random() * 50)
      });
    }

    return {
      nodes,
      connections,
      status: {
        ...status,
        total_nodes: nodes.length,
        active_nodes: nodes.filter(n => n.status === 'online').length,
        consensus_coordinators: nodes.filter(n => n.type === 'coordinator').length,
        validation_agents: nodes.filter(n => n.type === 'validation_agent').length
      }
    };
  };

  const renderNetworkVisualization = () => {
    const canvas = networkCanvasRef.current;
    if (!canvas || !networkData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections first (so they appear behind nodes)
    networkData.connections.forEach(connection => {
      const sourceNode = networkData.nodes.find(n => n.id === connection.source);
      const targetNode = networkData.nodes.find(n => n.id === connection.target);
      
      if (sourceNode?.position && targetNode?.position) {
        ctx.beginPath();
        ctx.moveTo(sourceNode.position.x, sourceNode.position.y);
        ctx.lineTo(targetNode.position.x, targetNode.position.y);
        
        // Style based on connection status
        if (connection.status === 'active') {
          ctx.strokeStyle = connection.type === 'coordinator' ? '#3b82f6' : '#10b981';
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 5]);
        }
        
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Draw nodes
    networkData.nodes.forEach(node => {
      if (!node.position) return;

      ctx.beginPath();
      ctx.arc(node.position.x, node.position.y, 20, 0, 2 * Math.PI);
      
      // Fill based on node type and status
      if (node.status === 'online') {
        ctx.fillStyle = node.type === 'coordinator' ? '#8b5cf6' : '#10b981';
      } else if (node.status === 'degraded') {
        ctx.fillStyle = '#f59e0b';
      } else {
        ctx.fillStyle = '#ef4444';
      }
      
      ctx.fill();
      
      // Border
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Node label
      ctx.fillStyle = '#1f2937';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        node.id.replace('_', ' '), 
        node.position.x, 
        node.position.y + 35
      );
    });
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!interactive || !networkData) return;

    const canvas = networkCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked node
    const clickedNode = networkData.nodes.find(node => {
      if (!node.position) return false;
      const distance = Math.sqrt(
        Math.pow(x - node.position.x, 2) + Math.pow(y - node.position.y, 2)
      );
      return distance <= 20;
    });

    setSelectedNode(clickedNode || null);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'degraded':
        return <AlertCircle className="h-5 w-5 text-yellow-500" />;
      case 'offline':
        return <WifiOff className="h-5 w-5 text-red-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'text-green-600';
      case 'degraded':
        return 'text-yellow-600';
      case 'critical':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  if (isLoading && !networkData) {
    return (
      <BrandedCard className="p-6">
        <div className="flex items-center justify-center">
          <LoadingSpinner size="lg" />
          <span className="ml-3 text-lg">Loading network topology...</span>
        </div>
      </BrandedCard>
    );
  }

  if (error) {
    return (
      <BrandedCard className="p-6 border-red-200 bg-red-50">
        <div className="flex items-center mb-4">
          <AlertCircle className="h-6 w-6 text-red-500 mr-3" />
          <h3 className="text-lg font-semibold text-red-700">Network Error</h3>
        </div>
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={fetchNetworkData}
          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
        >
          Retry
        </button>
      </BrandedCard>
    );
  }

  return (
    <div className={`space-y-6 ${isFullscreen ? 'fixed inset-0 z-50 bg-white p-6 overflow-auto' : ''}`}>
      {/* Network Status Overview */}
      {networkData && (
        <BrandedCard className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Network className="h-6 w-6 text-blue-600 mr-3" />
              <h2 className="text-xl font-bold text-gray-900">
                Validation Network Status
              </h2>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 text-gray-600 hover:text-gray-900"
              >
                <Settings className="h-5 w-5" />
              </button>
              <button
                onClick={() => setIsFullscreen(!isFullscreen)}
                className="p-2 text-gray-600 hover:text-gray-900"
              >
                {isFullscreen ? <Minimize2 className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <Users className="h-6 w-6 text-blue-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {networkData.status.total_nodes}
              </div>
              <div className="text-sm text-gray-600">Total Nodes</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <Wifi className="h-6 w-6 text-green-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {networkData.status.active_nodes}
              </div>
              <div className="text-sm text-gray-600">Active Nodes</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <Shield className="h-6 w-6 text-purple-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {networkData.status.consensus_coordinators}
              </div>
              <div className="text-sm text-gray-600">Coordinators</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <Activity className="h-6 w-6 text-orange-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {networkData.status.average_latency_ms.toFixed(0)}ms
              </div>
              <div className="text-sm text-gray-600">Avg Latency</div>
            </div>
          </div>

          <div className={`p-4 rounded-lg ${
            networkData.status.network_health === 'healthy'
              ? 'bg-green-50 border border-green-200'
              : networkData.status.network_health === 'degraded'
              ? 'bg-yellow-50 border border-yellow-200'
              : 'bg-red-50 border border-red-200'
          }`}>
            <div className="flex items-center">
              <Zap className={`h-5 w-5 mr-2 ${
                networkData.status.network_health === 'healthy'
                  ? 'text-green-500'
                  : networkData.status.network_health === 'degraded'
                  ? 'text-yellow-500'
                  : 'text-red-500'
              }`} />
              <span className={`font-medium ${getHealthColor(networkData.status.network_health)}`}>
                Network Health: {networkData.status.network_health.toUpperCase()}
              </span>
            </div>
          </div>
        </BrandedCard>
      )}

      {/* Network Visualization */}
      <BrandedCard className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Network Topology</h3>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
              <span>Coordinators</span>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>Agents</span>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <span>Degraded</span>
            </div>
          </div>
        </div>
        
        <div className="relative">
          <canvas
            ref={networkCanvasRef}
            width={800}
            height={400}
            className="border border-gray-200 rounded-lg cursor-pointer"
            onClick={handleCanvasClick}
          />
          
          {interactive && (
            <div className="absolute top-2 right-2 text-xs text-gray-500 bg-white p-2 rounded shadow">
              Click nodes for details
            </div>
          )}
        </div>
      </BrandedCard>

      {/* Selected Node Details */}
      {selectedNode && (
        <BrandedCard className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center">
              {getStatusIcon(selectedNode.status)}
              <div className="ml-3">
                <h3 className="text-lg font-semibold text-gray-900">
                  {selectedNode.id.replace('_', ' ').toUpperCase()}
                </h3>
                <p className="text-sm text-gray-600">
                  {selectedNode.type.replace('_', ' ').toUpperCase()}
                </p>
              </div>
            </div>
            
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Response Time</div>
              <div className="text-xl font-bold text-gray-900">
                {selectedNode.metrics.response_time_ms.toFixed(0)}ms
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Success Rate</div>
              <div className="text-xl font-bold text-gray-900">
                {(selectedNode.metrics.success_rate * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Active Validations</div>
              <div className="text-xl font-bold text-gray-900">
                {selectedNode.metrics.active_validations}
              </div>
            </div>
          </div>

          <div className="mt-4">
            <div className="text-sm text-gray-600 mb-2">Capabilities</div>
            <div className="flex flex-wrap gap-2">
              {selectedNode.capabilities.map(capability => (
                <span
                  key={capability}
                  className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded"
                >
                  {capability.replace('_', ' ')}
                </span>
              ))}
            </div>
          </div>

          {selectedNode.reputation_score && (
            <div className="mt-4">
              <div className="text-sm text-gray-600 mb-2">Reputation Score</div>
              <div className="flex items-center">
                <div className="text-lg font-bold text-gray-900 mr-3">
                  {(selectedNode.reputation_score * 100).toFixed(1)}%
                </div>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      selectedNode.reputation_score >= 0.8 ? 'bg-green-500' :
                      selectedNode.reputation_score >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${selectedNode.reputation_score * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </BrandedCard>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <BrandedCard className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Network Settings</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">Auto Refresh</span>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => {
                  // In real implementation, this would update props
                  console.log('Auto refresh:', e.target.checked);
                }}
                className="h-4 w-4"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">Show Metrics</span>
              <input
                type="checkbox"
                checked={showMetrics}
                onChange={(e) => {
                  console.log('Show metrics:', e.target.checked);
                }}
                className="h-4 w-4"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">Interactive Mode</span>
              <input
                type="checkbox"
                checked={interactive}
                onChange={(e) => {
                  console.log('Interactive mode:', e.target.checked);
                }}
                className="h-4 w-4"
              />
            </div>
          </div>
        </BrandedCard>
      )}
    </div>
  );
};

export default ValidationNetwork;