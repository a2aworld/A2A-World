/**
 * A2A World Platform - WebSocket Hook
 * 
 * React hook for managing WebSocket connections and real-time updates.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { WebSocketMessage, SystemStatus, Agent, Pattern } from '@/types';

interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface WebSocketState {
  socket: Socket | null;
  connected: boolean;
  error: string | null;
  reconnectAttempts: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
  } = options;

  const [state, setState] = useState<WebSocketState>({
    socket: null,
    connected: false,
    error: null,
    reconnectAttempts: 0,
  });

  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(() => {
    try {
      const socket = io(url, {
        transports: ['websocket', 'polling'],
      });

      socket.on('connect', () => {
        setState(prev => ({
          ...prev,
          socket,
          connected: true,
          error: null,
        }));
        reconnectAttemptsRef.current = 0;
      });

      socket.on('disconnect', () => {
        setState(prev => ({
          ...prev,
          connected: false,
        }));
      });

      socket.on('connect_error', (error) => {
        setState(prev => ({
          ...prev,
          error: error.message,
          connected: false,
        }));

        // Auto-reconnect logic
        if (reconnectAttemptsRef.current < reconnectAttempts) {
          reconnectAttemptsRef.current++;
          reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
        }
      });

      setState(prev => ({ ...prev, socket }));

    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Connection failed',
      }));
    }
  }, [url, reconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (state.socket) {
      state.socket.disconnect();
      setState(prev => ({
        ...prev,
        socket: null,
        connected: false,
      }));
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
  }, [state.socket]);

  const emit = useCallback((event: string, data: any) => {
    if (state.socket && state.connected) {
      state.socket.emit(event, data);
    }
  }, [state.socket, state.connected]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    ...state,
    connect,
    disconnect,
    emit,
  };
}

// Specialized hooks for different message types

export function useSystemStatus() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const { socket, connected } = useWebSocket();

  useEffect(() => {
    if (!socket || !connected) return;

    const handleSystemHealth = (message: WebSocketMessage) => {
      if (message.type === 'system_health') {
        setSystemStatus(message.data);
      }
    };

    socket.on('system_health', handleSystemHealth);
    
    // Request initial status
    socket.emit('get_system_status');

    return () => {
      socket.off('system_health', handleSystemHealth);
    };
  }, [socket, connected]);

  return systemStatus;
}

export function useAgentUpdates() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const { socket, connected } = useWebSocket();

  useEffect(() => {
    if (!socket || !connected) return;

    const handleAgentStatus = (message: WebSocketMessage) => {
      if (message.type === 'agent_status') {
        const { agent_id, status, health, metrics } = message.data;
        
        setAgents(prev => prev.map(agent => 
          agent.id === agent_id 
            ? { ...agent, status, health, metrics: { ...agent.metrics, ...metrics } }
            : agent
        ));
      }
    };

    const handleAgentList = (message: WebSocketMessage) => {
      if (message.type === 'agent_list') {
        setAgents(message.data);
      }
    };

    socket.on('agent_status', handleAgentStatus);
    socket.on('agent_list', handleAgentList);
    
    // Request initial agent list
    socket.emit('get_agents');

    return () => {
      socket.off('agent_status', handleAgentStatus);
      socket.off('agent_list', handleAgentList);
    };
  }, [socket, connected]);

  return agents;
}

export function usePatternUpdates() {
  const [newPatterns, setNewPatterns] = useState<Pattern[]>([]);
  const { socket, connected } = useWebSocket();

  useEffect(() => {
    if (!socket || !connected) return;

    const handlePatternDiscovered = (message: WebSocketMessage) => {
      if (message.type === 'pattern_discovered') {
        const pattern = message.data.pattern;
        setNewPatterns(prev => [pattern, ...prev.slice(0, 9)]); // Keep last 10 patterns
      }
    };

    socket.on('pattern_discovered', handlePatternDiscovered);

    return () => {
      socket.off('pattern_discovered', handlePatternDiscovered);
    };
  }, [socket, connected]);

  return newPatterns;
}

export default useWebSocket;