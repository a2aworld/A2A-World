/**
 * A2A World Platform - Type Definitions
 *
 * Core TypeScript types for the A2A World platform including
 * patterns, agents, geospatial data, and system interfaces.
 */

// React types - will be available when React is imported in consuming files
type ReactNode = any;

// =============================================================================
// Core System Types
// =============================================================================

export interface SystemStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  service: string;
  version: string;
  uptime: number;
  database: {
    status: 'connected' | 'disconnected';
    responseTime?: number;
  };
  messaging: {
    status: 'connected' | 'disconnected';
    queueDepth?: number;
  };
}

export interface ApiResponse<T = any> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// =============================================================================
// Geospatial Types
// =============================================================================

export interface Coordinates {
  latitude: number;
  longitude: number;
  elevation?: number;
}

export interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface GeospatialPoint {
  id: string;
  coordinates: Coordinates;
  properties: Record<string, any>;
  type: 'sacred_site' | 'cultural_landmark' | 'natural_feature' | 'pattern_point';
}

export interface GeospatialLayer {
  id: string;
  name: string;
  description: string;
  type: 'kml' | 'geojson' | 'pattern_overlay';
  visible: boolean;
  style: LayerStyle;
  data: GeospatialPoint[];
  bounds?: BoundingBox;
}

export interface LayerStyle {
  color: string;
  fillColor?: string;
  opacity: number;
  fillOpacity?: number;
  weight: number;
  radius?: number;
}

// =============================================================================
// Pattern Discovery Types
// =============================================================================

export interface Pattern {
  id: string;
  name: string;
  description: string;
  type: 'geometric' | 'cultural' | 'temporal' | 'environmental';
  status: 'discovered' | 'validating' | 'validated' | 'rejected';
  confidence: number;
  cultural_relevance: number;
  discovery_date: string;
  coordinates: Coordinates[];
  properties: PatternProperties;
  validation: PatternValidation;
  tags: string[];
  created_by: string;
  updated_at: string;
}

export interface PatternProperties {
  shape?: string;
  area?: number;
  perimeter?: number;
  orientation?: number;
  symmetry?: number;
  complexity?: number;
  historical_period?: string;
  cultural_context?: string;
  mythological_connections?: string[];
  environmental_factors?: string[];
}

export interface PatternValidation {
  consensus_score: number;
  validation_count: number;
  validators: string[];
  validation_notes: ValidationNote[];
  status: 'pending' | 'in_progress' | 'completed';
}

export interface ValidationNote {
  id: string;
  validator: string;
  score: number;
  notes: string;
  timestamp: string;
}

export interface PatternSearchFilters {
  type?: string[];
  status?: string[];
  confidence_min?: number;
  cultural_relevance_min?: number;
  date_range?: {
    start: string;
    end: string;
  };
  bounds?: BoundingBox;
  tags?: string[];
}

// =============================================================================
// Agent System Types
// =============================================================================

export interface Agent {
  id: string;
  name: string;
  type: 'discovery' | 'validation' | 'monitoring' | 'narrative';
  status: 'active' | 'idle' | 'error' | 'maintenance';
  health: 'healthy' | 'warning' | 'critical';
  configuration: AgentConfiguration;
  metrics: AgentMetrics;
  tasks: AgentTask[];
  last_heartbeat: string;
  created_at: string;
  updated_at: string;
}

export interface AgentConfiguration {
  discovery_radius?: number;
  confidence_threshold?: number;
  processing_interval?: number;
  max_concurrent_tasks?: number;
  data_sources?: string[];
  cultural_filters?: string[];
  validation_requirements?: string[];
}

export interface AgentMetrics {
  patterns_discovered: number;
  patterns_validated: number;
  tasks_completed: number;
  tasks_failed: number;
  avg_processing_time: number;
  uptime_percentage: number;
  memory_usage: number;
  cpu_usage: number;
}

export interface AgentTask {
  id: string;
  agent_id: string;
  type: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  data: Record<string, any>;
  progress: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface AgentLog {
  id: string;
  agent_id: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  metadata?: Record<string, any>;
  timestamp: string;
}

// =============================================================================
// Data Management Types
// =============================================================================

export interface Dataset {
  id: string;
  name: string;
  description: string;
  type: 'kml' | 'geojson' | 'csv' | 'shapefile';
  format: string;
  size: number;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  quality_score?: number;
  points_count: number;
  bounds?: BoundingBox;
  metadata: DatasetMetadata;
  upload_date: string;
  processed_date?: string;
  uploaded_by: string;
}

export interface DatasetMetadata {
  source?: string;
  license?: string;
  attribution?: string;
  coordinate_system?: string;
  encoding?: string;
  categories?: string[];
  tags?: string[];
  quality_report?: DataQualityReport;
}

export interface DataQualityReport {
  total_points: number;
  valid_points: number;
  invalid_points: number;
  duplicate_points: number;
  missing_coordinates: number;
  out_of_bounds: number;
  quality_issues: QualityIssue[];
  suggestions: string[];
}

export interface QualityIssue {
  type: 'warning' | 'error';
  message: string;
  affected_points: number;
  field?: string;
}

export interface FileUpload {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  dataset_id?: string;
}

// =============================================================================
// UI Component Types
// =============================================================================

export interface TableColumn<T = any> {
  key: keyof T | string;
  label: string;
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, record: T) => ReactNode;
  width?: string | number;
}

export interface TableProps<T = any> {
  data: T[];
  columns: TableColumn<T>[];
  loading?: boolean;
  pagination?: {
    page: number;
    limit: number;
    total: number;
    onPageChange: (page: number) => void;
    onLimitChange: (limit: number) => void;
  };
  sorting?: {
    field: keyof T | string;
    direction: 'asc' | 'desc';
    onSort: (field: keyof T | string, direction: 'asc' | 'desc') => void;
  };
  selection?: {
    selectedRows: string[];
    onSelectionChange: (selectedRows: string[]) => void;
  };
}

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  children: ReactNode;
}

export interface NotificationProps {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  actions?: {
    label: string;
    onClick: () => void;
  }[];
}

// =============================================================================
// WebSocket & Real-time Types
// =============================================================================

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
  source: string;
}

export interface AgentStatusUpdate extends WebSocketMessage {
  type: 'agent_status';
  data: {
    agent_id: string;
    status: Agent['status'];
    health: Agent['health'];
    metrics: Partial<AgentMetrics>;
  };
}

export interface PatternDiscoveryUpdate extends WebSocketMessage {
  type: 'pattern_discovered';
  data: {
    pattern: Pattern;
    agent_id: string;
  };
}

export interface SystemHealthUpdate extends WebSocketMessage {
  type: 'system_health';
  data: SystemStatus;
}

// =============================================================================
// All types are exported via individual export statements above
// =============================================================================