/**
 * A2A World Platform - Visualization Data Hook
 *
 * React hook for managing real-time visualization data updates
 * including terrain, multidisciplinary, and XAI visualization data.
 */

import { useEffect, useState, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { api } from '@/utils/api';
import { TerrainData, MultidisciplinaryData, XAIDecisionTree, ValidationResult } from '@/components';

interface UseTerrainDataOptions {
  autoFetch?: boolean;
  bounds?: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  elevationRange?: {
    min: number;
    max: number;
  };
}

interface UseMultidisciplinaryDataOptions {
  patternId?: string;
  includeAgents?: boolean;
  includeValidation?: boolean;
}

interface UseXaiDataOptions {
  patternId?: string;
  decisionTreeId?: string;
}

interface UseValidationDataOptions {
  patternIds?: string[];
  validationTypes?: ('cultural' | 'ethical' | 'statistical')[];
}

export function useTerrainData(options: UseTerrainDataOptions = {}) {
  const { autoFetch = true, bounds, elevationRange } = options;
  const [terrainData, setTerrainData] = useState<TerrainData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { socket, connected } = useWebSocket();

  const fetchTerrainData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const params: any = {};
      if (bounds) params.bounds = bounds;
      if (elevationRange) params.elevation_range = elevationRange;

      const response = await api.get('/visualization/terrain', { params });
      setTerrainData(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch terrain data');
    } finally {
      setLoading(false);
    }
  }, [bounds, elevationRange]);

  useEffect(() => {
    if (autoFetch) {
      fetchTerrainData();
    }
  }, [autoFetch, fetchTerrainData]);

  useEffect(() => {
    if (!socket || !connected) return;

    const handleTerrainUpdate = (message: any) => {
      if (message.type === 'terrain_update') {
        setTerrainData(message.data);
      }
    };

    socket.on('terrain_update', handleTerrainUpdate);

    return () => {
      socket.off('terrain_update', handleTerrainUpdate);
    };
  }, [socket, connected]);

  return {
    data: terrainData,
    loading,
    error,
    refetch: fetchTerrainData
  };
}

export function useMultidisciplinaryData(options: UseMultidisciplinaryDataOptions = {}) {
  const { patternId, includeAgents = true, includeValidation = true } = options;
  const [data, setData] = useState<MultidisciplinaryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { socket, connected } = useWebSocket();

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const params: any = {
        include_agents: includeAgents,
        include_validation: includeValidation
      };
      if (patternId) params.pattern_id = patternId;

      const response = await api.get('/visualization/multidisciplinary', { params });
      setData(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch multidisciplinary data');
    } finally {
      setLoading(false);
    }
  }, [patternId, includeAgents, includeValidation]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (!socket || !connected) return;

    const handleMultidisciplinaryUpdate = (message: any) => {
      if (message.type === 'multidisciplinary_update') {
        setData(message.data);
      }
    };

    socket.on('multidisciplinary_update', handleMultidisciplinaryUpdate);

    return () => {
      socket.off('multidisciplinary_update', handleMultidisciplinaryUpdate);
    };
  }, [socket, connected]);

  return {
    data,
    loading,
    error,
    refetch: fetchData
  };
}

export function useXaiData(options: UseXaiDataOptions = {}) {
  const { patternId, decisionTreeId } = options;
  const [data, setData] = useState<XAIDecisionTree | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { socket, connected } = useWebSocket();

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const params: any = {};
      if (patternId) params.pattern_id = patternId;
      if (decisionTreeId) params.decision_tree_id = decisionTreeId;

      const response = await api.get('/visualization/xai', { params });
      setData(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch XAI data');
    } finally {
      setLoading(false);
    }
  }, [patternId, decisionTreeId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (!socket || !connected) return;

    const handleXaiUpdate = (message: any) => {
      if (message.type === 'xai_update') {
        setData(message.data);
      }
    };

    socket.on('xai_update', handleXaiUpdate);

    return () => {
      socket.off('xai_update', handleXaiUpdate);
    };
  }, [socket, connected]);

  return {
    data,
    loading,
    error,
    refetch: fetchData
  };
}

export function useValidationData(options: UseValidationDataOptions = {}) {
  const { patternIds, validationTypes } = options;
  const [data, setData] = useState<ValidationResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { socket, connected } = useWebSocket();

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const params: any = {};
      if (patternIds) params.pattern_ids = patternIds.join(',');
      if (validationTypes) params.validation_types = validationTypes.join(',');

      const response = await api.get('/visualization/validation', { params });
      setData(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch validation data');
    } finally {
      setLoading(false);
    }
  }, [patternIds, validationTypes]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (!socket || !connected) return;

    const handleValidationUpdate = (message: any) => {
      if (message.type === 'validation_update') {
        setData(message.data);
      }
    };

    socket.on('validation_update', handleValidationUpdate);

    return () => {
      socket.off('validation_update', handleValidationUpdate);
    };
  }, [socket, connected]);

  return {
    data,
    loading,
    error,
    refetch: fetchData
  };
}

// Combined hook for all visualization data
export function useVisualizationData() {
  const terrain = useTerrainData();
  const multidisciplinary = useMultidisciplinaryData();
  const xai = useXaiData();
  const validation = useValidationData();

  return {
    terrain,
    multidisciplinary,
    xai,
    validation,
    // Combined loading and error states
    loading: terrain.loading || multidisciplinary.loading || xai.loading || validation.loading,
    error: terrain.error || multidisciplinary.error || xai.error || validation.error,
  };
}