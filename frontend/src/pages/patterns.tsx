/**
 * A2A World Platform - Pattern Explorer Page
 * 
 * Browse and analyze discovered patterns with detailed views,
 * search, filtering, and validation capabilities.
 */

import Head from 'next/head';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { 
  Brain, 
  Search, 
  Filter, 
  Eye, 
  Download, 
  CheckCircle, 
  Clock, 
  AlertTriangle,
  Home,
  ChevronDown,
  Star,
  Calendar,
  MapPin,
  TrendingUp,
  FileDown,
  ExternalLink
} from 'lucide-react';

// Mock data interfaces
interface Pattern {
  id: string;
  name: string;
  description: string;
  type: 'geometric' | 'cultural' | 'temporal' | 'environmental';
  status: 'discovered' | 'validating' | 'validated' | 'rejected';
  confidence: number;
  cultural_relevance: number;
  discovery_date: string;
  coordinates: { lat: number; lng: number }[];
  properties: {
    shape?: string;
    area?: number;
    complexity?: number;
    historical_period?: string;
    cultural_context?: string;
  };
  validation: {
    consensus_score: number;
    validation_count: number;
    status: 'pending' | 'in_progress' | 'completed';
  };
  tags: string[];
  discovered_by: string;
}

interface FilterState {
  type: string[];
  status: string[];
  confidenceMin: number;
  culturalRelevanceMin: number;
  dateRange: {
    start: string;
    end: string;
  };
  tags: string[];
}

export default function Patterns() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [filteredPatterns, setFilteredPatterns] = useState<Pattern[]>([]);
  const [selectedPattern, setSelectedPattern] = useState<Pattern | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [sortBy, setSortBy] = useState<'date' | 'confidence' | 'relevance'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [isLoading, setIsLoading] = useState(true);

  const [filters, setFilters] = useState<FilterState>({
    type: [],
    status: [],
    confidenceMin: 0,
    culturalRelevanceMin: 0,
    dateRange: { start: '', end: '' },
    tags: []
  });

  useEffect(() => {
    // Mock data - In production this would fetch from the API
    setTimeout(() => {
      const mockPatterns: Pattern[] = [
        {
          id: '1',
          name: 'Sacred Geometry Alignment',
          description: 'A geometric pattern discovered across multiple ancient sites showing precise alignment with celestial bodies',
          type: 'geometric',
          status: 'validated',
          confidence: 0.89,
          cultural_relevance: 0.94,
          discovery_date: '2024-01-15T10:30:00Z',
          coordinates: [
            { lat: 39.0242, lng: -83.4310 },
            { lat: 36.0619, lng: -107.9560 },
            { lat: 38.6581, lng: -90.0661 }
          ],
          properties: {
            shape: 'triangle',
            area: 125000,
            complexity: 0.78,
            historical_period: 'Pre-Columbian',
            cultural_context: 'Native American ceremonial sites'
          },
          validation: {
            consensus_score: 0.91,
            validation_count: 12,
            status: 'completed'
          },
          tags: ['sacred sites', 'astronomy', 'geometry'],
          discovered_by: 'Agent-Discovery-01'
        },
        {
          id: '2',
          name: 'Ceremonial Site Cluster',
          description: 'Cluster of ceremonial sites showing consistent architectural features and orientation patterns',
          type: 'cultural',
          status: 'validating',
          confidence: 0.76,
          cultural_relevance: 0.88,
          discovery_date: '2024-01-20T14:15:00Z',
          coordinates: [
            { lat: 35.2137, lng: -101.8313 },
            { lat: 35.0844, lng: -106.6504 },
            { lat: 34.7465, lng: -112.4068 }
          ],
          properties: {
            shape: 'cluster',
            area: 89000,
            complexity: 0.65,
            historical_period: 'Pueblo Period',
            cultural_context: 'Southwestern ceremonial complex'
          },
          validation: {
            consensus_score: 0.73,
            validation_count: 8,
            status: 'in_progress'
          },
          tags: ['ceremonial', 'pueblo', 'architecture'],
          discovered_by: 'Agent-Discovery-02'
        },
        {
          id: '3',
          name: 'Astronomical Correlation',
          description: 'Sites aligned with significant astronomical events including solstices and lunar cycles',
          type: 'temporal',
          status: 'validated',
          confidence: 0.93,
          cultural_relevance: 0.96,
          discovery_date: '2024-01-25T09:45:00Z',
          coordinates: [
            { lat: 43.8791, lng: -103.4591 },
            { lat: 44.5588, lng: -110.4089 },
            { lat: 45.7772, lng: -108.5069 }
          ],
          properties: {
            shape: 'linear',
            area: 156000,
            complexity: 0.85,
            historical_period: 'Various',
            cultural_context: 'Multi-cultural astronomical observations'
          },
          validation: {
            consensus_score: 0.95,
            validation_count: 15,
            status: 'completed'
          },
          tags: ['astronomy', 'calendar', 'solstice'],
          discovered_by: 'Agent-Discovery-03'
        }
      ];

      setPatterns(mockPatterns);
      setFilteredPatterns(mockPatterns);
      setIsLoading(false);
    }, 1000);
  }, []);

  // Apply filters and search
  useEffect(() => {
    let filtered = patterns.filter(pattern => {
      // Search query filter
      if (searchQuery && !pattern.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !pattern.description.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }

      // Type filter
      if (filters.type.length > 0 && !filters.type.includes(pattern.type)) {
        return false;
      }

      // Status filter
      if (filters.status.length > 0 && !filters.status.includes(pattern.status)) {
        return false;
      }

      // Confidence filter
      if (pattern.confidence < filters.confidenceMin / 100) {
        return false;
      }

      // Cultural relevance filter
      if (pattern.cultural_relevance < filters.culturalRelevanceMin / 100) {
        return false;
      }

      return true;
    });

    // Apply sorting
    filtered.sort((a, b) => {
      let aVal, bVal;
      switch (sortBy) {
        case 'confidence':
          aVal = a.confidence;
          bVal = b.confidence;
          break;
        case 'relevance':
          aVal = a.cultural_relevance;
          bVal = b.cultural_relevance;
          break;
        case 'date':
        default:
          aVal = new Date(a.discovery_date).getTime();
          bVal = new Date(b.discovery_date).getTime();
          break;
      }

      if (sortOrder === 'desc') {
        return bVal > aVal ? 1 : -1;
      } else {
        return aVal > bVal ? 1 : -1;
      }
    });

    setFilteredPatterns(filtered);
    setCurrentPage(1);
  }, [patterns, searchQuery, filters, sortBy, sortOrder]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'validated':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'validating':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'rejected':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default:
        return <Brain className="h-4 w-4 text-blue-500" />;
    }
  };

  const getTypeColor = (type: string) => {
    const colors = {
      geometric: 'bg-blue-100 text-blue-800',
      cultural: 'bg-green-100 text-green-800',
      temporal: 'bg-purple-100 text-purple-800',
      environmental: 'bg-yellow-100 text-yellow-800'
    };
    return colors[type as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const exportPattern = (pattern: Pattern, format: 'json' | 'kml' | 'csv' = 'json') => {
    // In production, this would call the API
    console.log(`Exporting pattern ${pattern.id} as ${format}`);
  };

  // Pagination
  const totalPages = Math.ceil(filteredPatterns.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedPatterns = filteredPatterns.slice(startIndex, startIndex + itemsPerPage);

  return (
    <>
      <Head>
        <title>Pattern Explorer - A2A World Platform</title>
        <meta name="description" content="Explore and analyze discovered patterns from the A2A World platform" />
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
                  <Brain className="h-6 w-6 text-primary-600 mr-2" />
                  <h1 className="text-lg font-semibold text-gray-900">Pattern Explorer</h1>
                </div>
              </div>

              <nav className="hidden md:flex space-x-6">
                <Link href="/maps" className="text-gray-600 hover:text-gray-900">
                  Maps
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
          {/* Search and Filter Bar */}
          <div className="bg-white rounded-lg shadow mb-6 p-4">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
              <div className="flex-1 flex items-center space-x-4">
                <div className="relative flex-1 max-w-md">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search patterns..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 pr-4 py-2 w-full border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
                  />
                </div>

                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                >
                  <Filter className="h-4 w-4 mr-2" />
                  Filters
                  <ChevronDown className="h-4 w-4 ml-2" />
                </button>
              </div>

              <div className="flex items-center space-x-4">
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="border border-gray-300 rounded-md text-sm px-3 py-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="date">Sort by Date</option>
                  <option value="confidence">Sort by Confidence</option>
                  <option value="relevance">Sort by Relevance</option>
                </select>

                <button
                  onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                  className="text-gray-400 hover:text-gray-600"
                >
                  {sortOrder === 'desc' ? '↓' : '↑'}
                </button>
              </div>
            </div>

            {/* Filter Panel */}
            {showFilters && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* Type Filter */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Type</label>
                    <div className="space-y-2">
                      {['geometric', 'cultural', 'temporal', 'environmental'].map(type => (
                        <label key={type} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={filters.type.includes(type)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setFilters(prev => ({ ...prev, type: [...prev.type, type] }));
                              } else {
                                setFilters(prev => ({ ...prev, type: prev.type.filter(t => t !== type) }));
                              }
                            }}
                            className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                          />
                          <span className="ml-2 text-sm text-gray-700 capitalize">{type}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Status Filter */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
                    <div className="space-y-2">
                      {['discovered', 'validating', 'validated', 'rejected'].map(status => (
                        <label key={status} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={filters.status.includes(status)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setFilters(prev => ({ ...prev, status: [...prev.status, status] }));
                              } else {
                                setFilters(prev => ({ ...prev, status: prev.status.filter(s => s !== status) }));
                              }
                            }}
                            className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                          />
                          <span className="ml-2 text-sm text-gray-700 capitalize">{status}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Confidence Filter */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Min Confidence: {filters.confidenceMin}%
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={filters.confidenceMin}
                      onChange={(e) => setFilters(prev => ({ ...prev, confidenceMin: parseInt(e.target.value) }))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Cultural Relevance Filter */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Min Cultural Relevance: {filters.culturalRelevanceMin}%
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={filters.culturalRelevanceMin}
                      onChange={(e) => setFilters(prev => ({ ...prev, culturalRelevanceMin: parseInt(e.target.value) }))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Results Summary */}
          <div className="mb-6">
            <p className="text-sm text-gray-600">
              Showing {paginatedPatterns.length} of {filteredPatterns.length} patterns
              {searchQuery && ` for "${searchQuery}"`}
            </p>
          </div>

          {/* Pattern List */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            {isLoading ? (
              <div className="p-8 text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
                <p className="mt-2 text-gray-500">Loading patterns...</p>
              </div>
            ) : paginatedPatterns.length === 0 ? (
              <div className="p-8 text-center">
                <Brain className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No patterns found</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Try adjusting your search criteria or filters.
                </p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {paginatedPatterns.map((pattern) => (
                  <div key={pattern.id} className="p-6 hover:bg-gray-50">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h3 className="text-lg font-medium text-gray-900">
                            {pattern.name}
                          </h3>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getTypeColor(pattern.type)}`}>
                            {pattern.type}
                          </span>
                          <div className="flex items-center">
                            {getStatusIcon(pattern.status)}
                            <span className="ml-1 text-sm text-gray-600 capitalize">
                              {pattern.status}
                            </span>
                          </div>
                        </div>

                        <p className="text-gray-700 mb-3">{pattern.description}</p>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                          <div>
                            <span className="text-sm text-gray-500">Confidence</span>
                            <div className="flex items-center">
                              <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                                <div 
                                  className="bg-primary-600 h-2 rounded-full" 
                                  style={{ width: `${pattern.confidence * 100}%` }}
                                />
                              </div>
                              <span className="text-sm font-medium">
                                {Math.round(pattern.confidence * 100)}%
                              </span>
                            </div>
                          </div>

                          <div>
                            <span className="text-sm text-gray-500">Cultural Relevance</span>
                            <div className="flex items-center">
                              <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                                <div 
                                  className="bg-green-600 h-2 rounded-full" 
                                  style={{ width: `${pattern.cultural_relevance * 100}%` }}
                                />
                              </div>
                              <span className="text-sm font-medium">
                                {Math.round(pattern.cultural_relevance * 100)}%
                              </span>
                            </div>
                          </div>

                          <div>
                            <span className="text-sm text-gray-500">Validation</span>
                            <p className="text-sm font-medium">
                              {pattern.validation.validation_count} validators
                            </p>
                            <p className="text-xs text-gray-500">
                              {Math.round(pattern.validation.consensus_score * 100)}% consensus
                            </p>
                          </div>

                          <div>
                            <span className="text-sm text-gray-500">Discovered</span>
                            <p className="text-sm font-medium flex items-center">
                              <Calendar className="h-3 w-3 mr-1" />
                              {formatDate(pattern.discovery_date)}
                            </p>
                            <p className="text-xs text-gray-500">{pattern.discovered_by}</p>
                          </div>
                        </div>

                        <div className="flex items-center space-x-4 text-sm text-gray-500">
                          <span className="flex items-center">
                            <MapPin className="h-3 w-3 mr-1" />
                            {pattern.coordinates.length} points
                          </span>
                          {pattern.properties.area && (
                            <span>{(pattern.properties.area / 1000).toFixed(1)}k km²</span>
                          )}
                          <span className="flex items-center space-x-1">
                            {pattern.tags.map(tag => (
                              <span key={tag} className="px-2 py-1 bg-gray-100 rounded text-xs">
                                {tag}
                              </span>
                            ))}
                          </span>
                        </div>
                      </div>

                      <div className="ml-4 flex-shrink-0 flex items-center space-x-2">
                        <button
                          onClick={() => setSelectedPattern(pattern)}
                          className="text-primary-600 hover:text-primary-700"
                        >
                          <Eye className="h-5 w-5" />
                        </button>
                        <button
                          onClick={() => exportPattern(pattern)}
                          className="text-gray-600 hover:text-gray-700"
                        >
                          <Download className="h-5 w-5" />
                        </button>
                        <Link href={`/maps?pattern=${pattern.id}`} className="text-gray-600 hover:text-gray-700">
                          <ExternalLink className="h-5 w-5" />
                        </Link>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="bg-white px-4 py-3 border-t border-gray-200 sm:px-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <p className="text-sm text-gray-700">
                      Showing {startIndex + 1} to {Math.min(startIndex + itemsPerPage, filteredPatterns.length)} of {filteredPatterns.length} results
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                      disabled={currentPage === 1}
                      className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <span className="text-sm text-gray-700">
                      Page {currentPage} of {totalPages}
                    </span>
                    <button
                      onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                      disabled={currentPage === totalPages}
                      className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}