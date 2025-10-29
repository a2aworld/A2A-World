/**
 * A2A World Platform - API Client
 * 
 * HTTP client configuration and API endpoints for the A2A World platform.
 * Handles authentication, error handling, and request/response transformation.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ApiResponse, PaginatedResponse } from '@/types';

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = 'v1';

// Create axios instance with default configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/${API_VERSION}`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// =============================================================================
// Request/Response Interceptors
// =============================================================================

// Request interceptor - add auth token if available
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token from localStorage if available
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - handle errors and transform responses
apiClient.interceptors.response.use(
  (response: AxiosResponse<ApiResponse>) => {
    return response;
  },
  (error) => {
    // Handle different error types
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const message = error.response.data?.message || error.message;

      switch (status) {
        case 401:
          // Unauthorized - clear token and redirect to login
          if (typeof window !== 'undefined') {
            localStorage.removeItem('auth_token');
            // Optionally redirect to login page
          }
          break;
        case 403:
          console.error('Access forbidden:', message);
          break;
        case 404:
          console.error('Resource not found:', message);
          break;
        case 500:
          console.error('Server error:', message);
          break;
        default:
          console.error('API error:', message);
      }

      return Promise.reject({
        status,
        message,
        data: error.response.data,
      });
    } else if (error.request) {
      // Network error
      console.error('Network error:', error.message);
      return Promise.reject({
        status: 0,
        message: 'Network error - please check your connection',
        data: null,
      });
    } else {
      // Other error
      console.error('Request error:', error.message);
      return Promise.reject({
        status: -1,
        message: error.message,
        data: null,
      });
    }
  }
);

// =============================================================================
// API Client Class
// =============================================================================

export class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = apiClient;
  }

  // Generic GET request
  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.get<ApiResponse<T>>(url, config);
    return response.data;
  }

  // Generic POST request
  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.post<ApiResponse<T>>(url, data, config);
    return response.data;
  }

  // Generic PUT request
  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.put<ApiResponse<T>>(url, data, config);
    return response.data;
  }

  // Generic DELETE request
  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.delete<ApiResponse<T>>(url, config);
    return response.data;
  }

  // File upload with progress tracking
  async uploadFile<T = any>(
    url: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<T>> {
    const formData = new FormData();
    formData.append('file', file);

    const config: AxiosRequestConfig = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    };

    const response = await this.client.post<ApiResponse<T>>(url, formData, config);
    return response.data;
  }
}

// =============================================================================
// API Endpoints
// =============================================================================

export class HealthApi extends ApiClient {
  async getHealth() {
    return this.get('/health');
  }

  async getSystemStatus() {
    return this.get('/health/system');
  }
}

export class AgentsApi extends ApiClient {
  async getAgents() {
    return this.get('/agents');
  }

  async getAgent(agentId: string) {
    return this.get(`/agents/${agentId}`);
  }

  async getAgentStatus(agentId: string) {
    return this.get(`/agents/${agentId}/status`);
  }

  async getAgentTasks(agentId: string, params?: { status?: string; limit?: number }) {
    const query = new URLSearchParams(params as any).toString();
    return this.get(`/agents/${agentId}/tasks${query ? `?${query}` : ''}`);
  }

  async getAgentLogs(agentId: string, params?: { level?: string; limit?: number }) {
    const query = new URLSearchParams(params as any).toString();
    return this.get(`/agents/${agentId}/logs${query ? `?${query}` : ''}`);
  }

  async startAgent(agentId: string) {
    return this.post(`/agents/${agentId}/start`);
  }

  async stopAgent(agentId: string) {
    return this.post(`/agents/${agentId}/stop`);
  }

  async restartAgent(agentId: string) {
    return this.post(`/agents/${agentId}/restart`);
  }
}

export class PatternsApi extends ApiClient {
  async getPatterns(params?: {
    page?: number;
    limit?: number;
    type?: string;
    status?: string;
    confidence_min?: number;
  }) {
    const query = new URLSearchParams(params as any).toString();
    return this.get<PaginatedResponse<any>>(`/patterns${query ? `?${query}` : ''}`);
  }

  async getPattern(patternId: string) {
    return this.get(`/patterns/${patternId}`);
  }

  async validatePattern(patternId: string, validation: { score: number; notes: string }) {
    return this.post(`/patterns/${patternId}/validate`, validation);
  }

  async exportPattern(patternId: string, format: 'json' | 'kml' | 'geojson' = 'json') {
    return this.get(`/patterns/${patternId}/export?format=${format}`);
  }

  async searchPatterns(query: {
    text?: string;
    bounds?: { north: number; south: number; east: number; west: number };
    filters?: Record<string, any>;
  }) {
    return this.post('/patterns/search', query);
  }
}

export class DataApi extends ApiClient {
  // Enhanced dataset management
  async getDatasets(params?: {
    limit?: number;
    offset?: number;
    file_type?: string;
  }) {
    const query = new URLSearchParams(params as any).toString();
    return this.get(`/data/${query ? `?${query}` : ''}`);
  }

  async getDataset(datasetId: string, includeFeatures?: boolean, featureLimit?: number) {
    const params = new URLSearchParams();
    if (includeFeatures) params.append('include_features', 'true');
    if (featureLimit) params.append('feature_limit', featureLimit.toString());
    
    const query = params.toString();
    return this.get(`/data/${datasetId}${query ? `?${query}` : ''}`);
  }

  // Enhanced file upload with real-time progress
  async uploadDataFile(file: File, onProgress?: (progress: number) => void) {
    return this.uploadFile('/data/upload', file, onProgress);
  }

  // Get upload status and progress
  async getUploadStatus(uploadId: string) {
    return this.get(`/data/upload/${uploadId}/status`);
  }

  // Validate file before upload
  async validateFile(filePath: string, fileType?: string) {
    return this.post('/data/validate', { file_path: filePath, file_type: fileType });
  }

  // Delete dataset
  async deleteDataset(datasetId: string) {
    return this.delete(`/data/${datasetId}`);
  }

  // Get summary statistics
  async getDataSummary() {
    return this.get('/data/stats/summary');
  }

  // Legacy method for backward compatibility
  async uploadDataset(file: File, metadata: any, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    const config: AxiosRequestConfig = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent: any) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    };

    return this.post('/data/upload', formData, config);
  }

  async getDatasetPoints(datasetId: string, params?: { bounds?: any; limit?: number }) {
    const query = new URLSearchParams(params as any).toString();
    return this.get(`/data/${datasetId}/points${query ? `?${query}` : ''}`);
  }
}

// =============================================================================
// Export API instances
// =============================================================================

export const healthApi = new HealthApi();
export const agentsApi = new AgentsApi();
export const patternsApi = new PatternsApi();
export const dataApi = new DataApi();

// Export the main client instance for custom requests
export const api = apiClient;
export default apiClient;