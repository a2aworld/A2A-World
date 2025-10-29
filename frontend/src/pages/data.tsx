/**
 * A2A World Platform - Data Management Page
 * 
 * Upload KML/GeoJSON files, manage datasets, view data quality reports,
 * and monitor import/processing status.
 */

import Head from 'next/head';
import Link from 'next/link';
import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Database, 
  Upload, 
  File, 
  CheckCircle, 
  AlertTriangle,
  Clock,
  Home,
  Download,
  Eye,
  Trash2,
  Search,
  Filter,
  FileText,
  MapPin,
  BarChart3,
  AlertCircle,
  X
} from 'lucide-react';

// Mock data interfaces
interface Dataset {
  id: string;
  name: string;
  description: string;
  type: 'kml' | 'geojson' | 'csv' | 'shapefile';
  format: string;
  size: number;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  quality_score?: number;
  points_count: number;
  bounds?: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  upload_date: string;
  processed_date?: string;
  uploaded_by: string;
  metadata: {
    source?: string;
    license?: string;
    attribution?: string;
    coordinate_system?: string;
    encoding?: string;
    categories?: string[];
    tags?: string[];
  };
  quality_report?: {
    total_points: number;
    valid_points: number;
    invalid_points: number;
    duplicate_points: number;
    missing_coordinates: number;
    out_of_bounds: number;
    quality_issues: Array<{
      type: 'warning' | 'error';
      message: string;
      affected_points: number;
      field?: string;
    }>;
    suggestions: string[];
  };
}

interface FileUpload {
  file: File;
  id: string;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  dataset_id?: string;
}

export default function DataManagement() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [uploads, setUploads] = useState<FileUpload[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Mock data - In production this would fetch from the API
    setTimeout(() => {
      const mockDatasets: Dataset[] = [
        {
          id: 'dataset-1',
          name: 'Sacred Sites of North America',
          description: 'Comprehensive collection of Native American sacred and ceremonial sites',
          type: 'kml',
          format: 'application/vnd.google-earth.kml+xml',
          size: 2.4 * 1024 * 1024,
          status: 'ready',
          quality_score: 0.92,
          points_count: 1247,
          bounds: {
            north: 49.0,
            south: 25.0,
            east: -66.0,
            west: -125.0
          },
          upload_date: '2024-01-25T14:30:00Z',
          processed_date: '2024-01-25T14:32:15Z',
          uploaded_by: 'admin',
          metadata: {
            source: 'National Park Service',
            license: 'Public Domain',
            attribution: 'U.S. National Park Service',
            coordinate_system: 'WGS84',
            encoding: 'UTF-8',
            categories: ['sacred_sites', 'native_american', 'cultural_heritage'],
            tags: ['historical', 'ceremonial', 'spiritual']
          },
          quality_report: {
            total_points: 1247,
            valid_points: 1147,
            invalid_points: 100,
            duplicate_points: 23,
            missing_coordinates: 45,
            out_of_bounds: 32,
            quality_issues: [
              {
                type: 'warning',
                message: 'Some coordinates appear to be duplicated',
                affected_points: 23,
                field: 'coordinates'
              },
              {
                type: 'error',
                message: 'Missing coordinate data for some points',
                affected_points: 45,
                field: 'coordinates'
              }
            ],
            suggestions: [
              'Review duplicate coordinates and merge if appropriate',
              'Add missing coordinate data where possible',
              'Validate coordinates against known geographical bounds'
            ]
          }
        },
        {
          id: 'dataset-2',
          name: 'Ancient Astronomical Alignments',
          description: 'Sites with confirmed astronomical alignments and celestial observations',
          type: 'geojson',
          format: 'application/geo+json',
          size: 856 * 1024,
          status: 'processing',
          points_count: 342,
          upload_date: '2024-01-30T09:15:00Z',
          uploaded_by: 'researcher',
          metadata: {
            source: 'Archaeological Survey Database',
            license: 'CC BY-SA 4.0',
            coordinate_system: 'WGS84',
            encoding: 'UTF-8',
            categories: ['astronomical', 'alignments', 'ancient_sites'],
            tags: ['astronomy', 'solstice', 'equinox']
          }
        },
        {
          id: 'dataset-3',
          name: 'Cultural Landmarks Database',
          description: 'Miscellaneous cultural and historical landmarks',
          type: 'csv',
          format: 'text/csv',
          size: 125 * 1024,
          status: 'error',
          points_count: 0,
          upload_date: '2024-01-29T16:45:00Z',
          uploaded_by: 'admin',
          metadata: {
            source: 'Local Historical Society',
            coordinate_system: 'Unknown',
            encoding: 'UTF-8',
            categories: ['landmarks', 'historical'],
            tags: ['monuments', 'buildings']
          }
        }
      ];

      setDatasets(mockDatasets);
      setSelectedDataset(mockDatasets[0]);
      setIsLoading(false);
    }, 1000);
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newUploads = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      progress: 0,
      status: 'pending' as const
    }));

    setUploads(prev => [...prev, ...newUploads]);

    // Simulate upload process
    newUploads.forEach(upload => {
      setTimeout(() => {
        setUploads(prev => prev.map(u => 
          u.id === upload.id ? { ...u, status: 'uploading' } : u
        ));

        // Simulate progress
        const interval = setInterval(() => {
          setUploads(prev => prev.map(u => {
            if (u.id === upload.id && u.progress < 100) {
              const newProgress = Math.min(u.progress + Math.random() * 20, 100);
              if (newProgress >= 100) {
                clearInterval(interval);
                setTimeout(() => {
                  setUploads(prev => prev.map(up => 
                    up.id === upload.id ? { ...up, status: 'completed' } : up
                  ));
                }, 500);
              }
              return { ...u, progress: newProgress };
            }
            return u;
          }));
        }, 500);
      }, 1000);
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.google-earth.kml+xml': ['.kml'],
      'application/geo+json': ['.geojson'],
      'text/csv': ['.csv'],
      'application/zip': ['.zip']
    },
    multiple: true
  });

  const removeUpload = (uploadId: string) => {
    setUploads(prev => prev.filter(u => u.id !== uploadId));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default:
        return <File className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    const colors = {
      ready: 'text-green-600 bg-green-50 border-green-200',
      processing: 'text-blue-600 bg-blue-50 border-blue-200',
      error: 'text-red-600 bg-red-50 border-red-200',
      uploading: 'text-yellow-600 bg-yellow-50 border-yellow-200'
    };
    return colors[status as keyof typeof colors] || 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'kml':
        return '🗺️';
      case 'geojson':
        return '🌍';
      case 'csv':
        return '📊';
      case 'shapefile':
        return '📐';
      default:
        return '📄';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + ' KB';
    return Math.round(bytes / (1024 * 1024)) + ' MB';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const filteredDatasets = datasets.filter(dataset => {
    const matchesSearch = searchQuery === '' || 
      dataset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      dataset.description.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || dataset.status === statusFilter;
    const matchesType = typeFilter === 'all' || dataset.type === typeFilter;
    
    return matchesSearch && matchesStatus && matchesType;
  });

  return (
    <>
      <Head>
        <title>Data Management - A2A World Platform</title>
        <meta name="description" content="Upload and manage geospatial datasets for the A2A World platform" />
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
                  <Database className="h-6 w-6 text-primary-600 mr-2" />
                  <h1 className="text-lg font-semibold text-gray-900">Data Management</h1>
                </div>
              </div>

              <nav className="hidden md:flex space-x-6">
                <Link href="/maps" className="text-gray-600 hover:text-gray-900">
                  Maps
                </Link>
                <Link href="/patterns" className="text-gray-600 hover:text-gray-900">
                  Patterns
                </Link>
                <Link href="/agents" className="text-gray-600 hover:text-gray-900">
                  Agents
                </Link>
              </nav>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Upload Section */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">Upload New Data</h2>
              <button
                onClick={() => setShowUploadModal(true)}
                className="flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md"
              >
                <Upload className="h-4 w-4 mr-2" />
                Upload Files
              </button>
            </div>

            {/* Active Uploads */}
            {uploads.length > 0 && (
              <div className="bg-white rounded-lg shadow mb-6">
                <div className="p-4 border-b border-gray-200">
                  <h3 className="text-sm font-medium text-gray-900">Active Uploads</h3>
                </div>
                <div className="p-4 space-y-3">
                  {uploads.map((upload) => (
                    <div key={upload.id} className="flex items-center space-x-4 p-3 border rounded-lg">
                      <div className="flex-shrink-0">
                        <File className="h-8 w-8 text-gray-400" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {upload.file.name}
                        </p>
                        <div className="flex items-center space-x-2 mt-1">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(upload.status)}`}>
                            {upload.status}
                          </span>
                          <span className="text-xs text-gray-500">
                            {formatFileSize(upload.file.size)}
                          </span>
                        </div>
                        {upload.status === 'uploading' && (
                          <div className="mt-2">
                            <div className="flex justify-between text-xs text-gray-600 mb-1">
                              <span>Progress</span>
                              <span>{Math.round(upload.progress)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1">
                              <div 
                                className="bg-primary-600 h-1 rounded-full transition-all duration-300" 
                                style={{ width: `${upload.progress}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="flex-shrink-0">
                        <button
                          onClick={() => removeUpload(upload.id)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Datasets Section */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Dataset List */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow">
                <div className="p-4 border-b border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-medium text-gray-900">Datasets</h2>
                    <span className="text-sm text-gray-500">
                      {datasets.length} total
                    </span>
                  </div>

                  {/* Search and Filter */}
                  <div className="space-y-3">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Search datasets..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9 pr-4 py-2 w-full border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
                      />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2">
                      <select
                        value={statusFilter}
                        onChange={(e) => setStatusFilter(e.target.value)}
                        className="border border-gray-300 rounded-md text-sm px-2 py-1 focus:ring-primary-500 focus:border-primary-500"
                      >
                        <option value="all">All Status</option>
                        <option value="ready">Ready</option>
                        <option value="processing">Processing</option>
                        <option value="error">Error</option>
                      </select>
                      
                      <select
                        value={typeFilter}
                        onChange={(e) => setTypeFilter(e.target.value)}
                        className="border border-gray-300 rounded-md text-sm px-2 py-1 focus:ring-primary-500 focus:border-primary-500"
                      >
                        <option value="all">All Types</option>
                        <option value="kml">KML</option>
                        <option value="geojson">GeoJSON</option>
                        <option value="csv">CSV</option>
                        <option value="shapefile">Shapefile</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                  {isLoading ? (
                    <div className="p-4 text-center">
                      <Database className="h-6 w-6 animate-pulse text-gray-400 mx-auto" />
                      <p className="mt-2 text-sm text-gray-500">Loading datasets...</p>
                    </div>
                  ) : filteredDatasets.length === 0 ? (
                    <div className="p-4 text-center">
                      <Database className="h-8 w-8 text-gray-400 mx-auto" />
                      <p className="mt-2 text-sm text-gray-500">No datasets found</p>
                    </div>
                  ) : (
                    filteredDatasets.map((dataset) => (
                      <button
                        key={dataset.id}
                        onClick={() => setSelectedDataset(dataset)}
                        className={`w-full p-4 text-left hover:bg-gray-50 ${
                          selectedDataset?.id === dataset.id ? 'bg-primary-50 border-r-2 border-primary-600' : ''
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-sm font-medium text-gray-900 truncate">
                            {dataset.name}
                          </h3>
                          {getStatusIcon(dataset.status)}
                        </div>
                        
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="text-lg">{getTypeIcon(dataset.type)}</span>
                          <span className="text-xs text-gray-500 uppercase">{dataset.type}</span>
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(dataset.status)}`}>
                            {dataset.status}
                          </span>
                        </div>
                        
                        <div className="text-xs text-gray-500">
                          <div>{dataset.points_count.toLocaleString()} points</div>
                          <div>{formatFileSize(dataset.size)}</div>
                          {dataset.quality_score && (
                            <div>Quality: {Math.round(dataset.quality_score * 100)}%</div>
                          )}
                        </div>
                      </button>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Dataset Details */}
            <div className="lg:col-span-2">
              {selectedDataset ? (
                <div className="space-y-6">
                  {/* Dataset Header */}
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h2 className="text-xl font-semibold text-gray-900 mb-2">
                          {selectedDataset.name}
                        </h2>
                        <p className="text-gray-600">{selectedDataset.description}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button className="text-gray-600 hover:text-gray-700">
                          <Eye className="h-5 w-5" />
                        </button>
                        <button className="text-gray-600 hover:text-gray-700">
                          <Download className="h-5 w-5" />
                        </button>
                        <button className="text-red-600 hover:text-red-700">
                          <Trash2 className="h-5 w-5" />
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900">
                          {selectedDataset.points_count.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-600">Data Points</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900">
                          {formatFileSize(selectedDataset.size)}
                        </div>
                        <div className="text-sm text-gray-600">File Size</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900">
                          {selectedDataset.quality_score ? Math.round(selectedDataset.quality_score * 100) + '%' : 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Quality Score</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900 uppercase">
                          {selectedDataset.type}
                        </div>
                        <div className="text-sm text-gray-600">Format</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="flex justify-between py-2">
                          <span className="text-gray-600">Uploaded:</span>
                          <span className="font-medium">{formatDate(selectedDataset.upload_date)}</span>
                        </div>
                        <div className="flex justify-between py-2">
                          <span className="text-gray-600">Uploaded by:</span>
                          <span className="font-medium">{selectedDataset.uploaded_by}</span>
                        </div>
                        <div className="flex justify-between py-2">
                          <span className="text-gray-600">Status:</span>
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(selectedDataset.status)}`}>
                            {selectedDataset.status}
                          </span>
                        </div>
                      </div>
                      <div>
                        {selectedDataset.processed_date && (
                          <div className="flex justify-between py-2">
                            <span className="text-gray-600">Processed:</span>
                            <span className="font-medium">{formatDate(selectedDataset.processed_date)}</span>
                          </div>
                        )}
                        <div className="flex justify-between py-2">
                          <span className="text-gray-600">Coordinate System:</span>
                          <span className="font-medium">{selectedDataset.metadata.coordinate_system || 'Unknown'}</span>
                        </div>
                        <div className="flex justify-between py-2">
                          <span className="text-gray-600">Encoding:</span>
                          <span className="font-medium">{selectedDataset.metadata.encoding || 'Unknown'}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Quality Report */}
                  {selectedDataset.quality_report && (
                    <div className="bg-white rounded-lg shadow p-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4">Data Quality Report</h3>
                      
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                        <div className="text-center p-3 bg-green-50 rounded-lg">
                          <div className="text-xl font-bold text-green-900">
                            {selectedDataset.quality_report.valid_points.toLocaleString()}
                          </div>
                          <div className="text-sm text-green-600">Valid Points</div>
                        </div>
                        <div className="text-center p-3 bg-red-50 rounded-lg">
                          <div className="text-xl font-bold text-red-900">
                            {selectedDataset.quality_report.invalid_points.toLocaleString()}
                          </div>
                          <div className="text-sm text-red-600">Invalid Points</div>
                        </div>
                        <div className="text-center p-3 bg-yellow-50 rounded-lg">
                          <div className="text-xl font-bold text-yellow-900">
                            {selectedDataset.quality_report.duplicate_points.toLocaleString()}
                          </div>
                          <div className="text-sm text-yellow-600">Duplicates</div>
                        </div>
                      </div>

                      {selectedDataset.quality_report.quality_issues.length > 0 && (
                        <div className="mb-4">
                          <h4 className="text-sm font-medium text-gray-900 mb-2">Issues Found</h4>
                          <div className="space-y-2">
                            {selectedDataset.quality_report.quality_issues.map((issue, index) => (
                              <div key={index} className={`p-3 rounded-lg border ${
                                issue.type === 'error' ? 'bg-red-50 border-red-200' : 'bg-yellow-50 border-yellow-200'
                              }`}>
                                <div className="flex items-center">
                                  {issue.type === 'error' ? (
                                    <AlertCircle className="h-4 w-4 text-red-500 mr-2" />
                                  ) : (
                                    <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2" />
                                  )}
                                  <span className={`text-sm font-medium ${
                                    issue.type === 'error' ? 'text-red-900' : 'text-yellow-900'
                                  }`}>
                                    {issue.message}
                                  </span>
                                </div>
                                <p className={`text-xs mt-1 ${
                                  issue.type === 'error' ? 'text-red-600' : 'text-yellow-600'
                                }`}>
                                  Affects {issue.affected_points} points
                                  {issue.field && ` in field: ${issue.field}`}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {selectedDataset.quality_report.suggestions.length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium text-gray-900 mb-2">Suggestions</h4>
                          <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                            {selectedDataset.quality_report.suggestions.map((suggestion, index) => (
                              <li key={index}>{suggestion}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Metadata */}
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Metadata</h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-sm font-medium text-gray-900 mb-2">General Information</h4>
                        <div className="space-y-2 text-sm">
                          {selectedDataset.metadata.source && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">Source:</span>
                              <span>{selectedDataset.metadata.source}</span>
                            </div>
                          )}
                          {selectedDataset.metadata.license && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">License:</span>
                              <span>{selectedDataset.metadata.license}</span>
                            </div>
                          )}
                          {selectedDataset.metadata.attribution && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">Attribution:</span>
                              <span>{selectedDataset.metadata.attribution}</span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-900 mb-2">Categories & Tags</h4>
                        {selectedDataset.metadata.categories && selectedDataset.metadata.categories.length > 0 && (
                          <div className="mb-3">
                            <span className="text-xs text-gray-600 mb-1 block">Categories:</span>
                            <div className="flex flex-wrap gap-1">
                              {selectedDataset.metadata.categories.map(category => (
                                <span key={category} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                                  {category}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {selectedDataset.metadata.tags && selectedDataset.metadata.tags.length > 0 && (
                          <div>
                            <span className="text-xs text-gray-600 mb-1 block">Tags:</span>
                            <div className="flex flex-wrap gap-1">
                              {selectedDataset.metadata.tags.map(tag => (
                                <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded">
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow p-8 text-center">
                  <Database className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No Dataset Selected</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Select a dataset from the list to view details
                  </p>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* Upload Modal */}
        {showUploadModal && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
            <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">Upload Files</h3>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
              
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-primary-400 bg-primary-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                {isDragActive ? (
                  <p>Drop the files here ...</p>
                ) : (
                  <div>
                    <p>Drag & drop files here, or click to select files</p>
                    <p className="text-sm text-gray-500 mt-2">
                      Supports: KML, GeoJSON, CSV, ZIP files
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}