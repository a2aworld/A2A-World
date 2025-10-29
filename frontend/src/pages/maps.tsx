/**
 * A2A World Platform - Interactive Maps Page
 * 
 * Full-screen interactive map with geospatial data layers,
 * pattern overlays, sacred sites, and clustering.
 */

import Head from 'next/head';
import Link from 'next/link';
import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { 
  Globe, 
  Layers, 
  Search, 
  Filter, 
  MapPin, 
  Brain, 
  Settings,
  Eye,
  EyeOff,
  ChevronDown,
  Home
} from 'lucide-react';

// Dynamic import for Leaflet components to avoid SSR issues
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
);

const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);

const Marker = dynamic(
  () => import('react-leaflet').then((mod) => mod.Marker),
  { ssr: false }
);

const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
);

const Circle = dynamic(
  () => import('react-leaflet').then((mod) => mod.Circle),
  { ssr: false }
);

// Mock data interfaces
interface MapLayer {
  id: string;
  name: string;
  type: 'sacred_sites' | 'patterns' | 'cultural_landmarks' | 'natural_features';
  visible: boolean;
  color: string;
  points: MapPoint[];
}

interface MapPoint {
  id: string;
  name: string;
  type: string;
  coordinates: [number, number];
  properties: {
    description?: string;
    confidence?: number;
    cultural_relevance?: number;
    discovered_date?: string;
    [key: string]: any;
  };
}

export default function Maps() {
  const [layers, setLayers] = useState<MapLayer[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPoint, setSelectedPoint] = useState<MapPoint | null>(null);
  const [mapCenter, setMapCenter] = useState<[number, number]>([39.8283, -98.5795]); // Center of USA
  const [mapZoom, setMapZoom] = useState(4);
  const [showLayerPanel, setShowLayerPanel] = useState(false);
  const [activeFilters, setActiveFilters] = useState<string[]>([]);

  useEffect(() => {
    // Mock data - In production this would fetch from the API
    const mockLayers: MapLayer[] = [
      {
        id: 'sacred_sites',
        name: 'Sacred Sites',
        type: 'sacred_sites',
        visible: true,
        color: '#10B981',
        points: [
          {
            id: '1',
            name: 'Serpent Mound',
            type: 'ancient_earthwork',
            coordinates: [39.0242, -83.4310],
            properties: {
              description: 'Ancient serpent-shaped earthwork in Ohio',
              cultural_relevance: 0.95,
              discovered_date: '1800s'
            }
          },
          {
            id: '2',
            name: 'Chaco Canyon',
            type: 'archaeological_site',
            coordinates: [36.0619, -107.9560],
            properties: {
              description: 'Ancient Puebloan cultural site with astronomical alignments',
              cultural_relevance: 0.98,
              discovered_date: '1849'
            }
          },
          {
            id: '3',
            name: 'Cahokia',
            type: 'ancient_city',
            coordinates: [38.6581, -90.0661],
            properties: {
              description: 'Pre-Columbian Native American city near St. Louis',
              cultural_relevance: 0.92,
              discovered_date: '1960s'
            }
          }
        ]
      },
      {
        id: 'patterns',
        name: 'Pattern Overlays',
        type: 'patterns',
        visible: true,
        color: '#8B5CF6',
        points: [
          {
            id: 'p1',
            name: 'Sacred Geometry Alignment',
            type: 'geometric_pattern',
            coordinates: [38.8951, -77.0364],
            properties: {
              description: 'Discovered geometric alignment between monuments',
              confidence: 0.89,
              discovered_date: '2024-01-15'
            }
          },
          {
            id: 'p2',
            name: 'Astronomical Correlation',
            type: 'celestial_pattern',
            coordinates: [40.7484, -73.9857],
            properties: {
              description: 'Sites aligned with celestial events',
              confidence: 0.93,
              discovered_date: '2024-01-20'
            }
          }
        ]
      },
      {
        id: 'cultural_landmarks',
        name: 'Cultural Landmarks',
        type: 'cultural_landmarks',
        visible: false,
        color: '#F59E0B',
        points: [
          {
            id: 'c1',
            name: 'Mount Rushmore',
            type: 'monument',
            coordinates: [43.8791, -103.4591],
            properties: {
              description: 'National monument carved into granite',
              cultural_relevance: 0.85
            }
          }
        ]
      }
    ];

    setLayers(mockLayers);
  }, []);

  const filteredLayers = useMemo(() => {
    return layers.filter(layer => {
      if (activeFilters.length === 0) return true;
      return activeFilters.includes(layer.type);
    });
  }, [layers, activeFilters]);

  const toggleLayer = (layerId: string) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
    ));
  };

  const toggleFilter = (filterType: string) => {
    setActiveFilters(prev =>
      prev.includes(filterType)
        ? prev.filter(f => f !== filterType)
        : [...prev, filterType]
    );
  };

  const getMarkerIcon = (point: MapPoint) => {
    const colors = {
      sacred_sites: '#10B981',
      patterns: '#8B5CF6',
      cultural_landmarks: '#F59E0B',
      natural_features: '#06B6D4'
    };

    const layer = layers.find(l => l.points.some(p => p.id === point.id));
    const color = layer ? layer.color : '#6B7280';

    return {
      iconUrl: `data:image/svg+xml,${encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}" width="24" height="24">
          <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
        </svg>
      `)}`,
      iconSize: [24, 24],
      iconAnchor: [12, 24],
      popupAnchor: [0, -24]
    };
  };

  return (
    <>
      <Head>
        <title>Interactive Maps - A2A World Platform</title>
        <meta name="description" content="Explore geospatial data, sacred sites, and pattern discoveries on an interactive map" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="h-screen flex flex-col bg-gray-100">
        {/* Top Navigation Bar */}
        <header className="bg-white shadow-sm border-b border-gray-200 z-50">
          <div className="px-4 h-14 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/" className="flex items-center text-gray-600 hover:text-gray-900">
                <Home className="h-5 w-5 mr-2" />
                <span className="text-sm font-medium">Dashboard</span>
              </Link>
              <div className="flex items-center">
                <Globe className="h-6 w-6 text-primary-600 mr-2" />
                <h1 className="text-lg font-semibold text-gray-900">Interactive Map</h1>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search locations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              {/* Layer Toggle */}
              <button
                onClick={() => setShowLayerPanel(!showLayerPanel)}
                className="flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <Layers className="h-4 w-4 mr-2" />
                Layers
                <ChevronDown className="h-4 w-4 ml-2" />
              </button>
            </div>
          </div>
        </header>

        <div className="flex-1 relative">
          {/* Layer Control Panel */}
          {showLayerPanel && (
            <div className="absolute top-4 right-4 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-40 max-h-96 overflow-y-auto">
              <div className="p-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">Map Layers</h3>
              </div>
              
              <div className="p-4 space-y-4">
                {/* Filters */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Filters</h4>
                  <div className="space-y-2">
                    {['sacred_sites', 'patterns', 'cultural_landmarks', 'natural_features'].map(filter => (
                      <label key={filter} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={activeFilters.includes(filter)}
                          onChange={() => toggleFilter(filter)}
                          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                        />
                        <span className="ml-2 text-sm text-gray-700 capitalize">
                          {filter.replace('_', ' ')}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Layer Visibility */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Layer Visibility</h4>
                  <div className="space-y-2">
                    {filteredLayers.map(layer => (
                      <div key={layer.id} className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div 
                            className="w-4 h-4 rounded-full mr-2" 
                            style={{ backgroundColor: layer.color }}
                          />
                          <span className="text-sm text-gray-700">{layer.name}</span>
                        </div>
                        <button
                          onClick={() => toggleLayer(layer.id)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          {layer.visible ? (
                            <Eye className="h-4 w-4" />
                          ) : (
                            <EyeOff className="h-4 w-4" />
                          )}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Map Container */}
          <div className="h-full">
            {typeof window !== 'undefined' && (
              <MapContainer
                center={mapCenter}
                zoom={mapZoom}
                style={{ height: '100%', width: '100%' }}
                className="z-10"
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* Render points from visible layers */}
                {filteredLayers
                  .filter(layer => layer.visible)
                  .forEach(layer => 
                    layer.points.map(point => (
                      <Marker
                        key={point.id}
                        position={point.coordinates}
                        icon={getMarkerIcon(point)}
                        eventHandlers={{
                          click: () => setSelectedPoint(point)
                        }}
                      >
                        <Popup>
                          <div className="p-2 min-w-64">
                            <h4 className="font-semibold text-gray-900 mb-2">
                              {point.name}
                            </h4>
                            <p className="text-sm text-gray-600 mb-2 capitalize">
                              {point.type.replace('_', ' ')}
                            </p>
                            {point.properties.description && (
                              <p className="text-sm text-gray-700 mb-2">
                                {point.properties.description}
                              </p>
                            )}
                            {point.properties.confidence && (
                              <div className="text-xs text-gray-500">
                                Confidence: {Math.round(point.properties.confidence * 100)}%
                              </div>
                            )}
                            {point.properties.cultural_relevance && (
                              <div className="text-xs text-gray-500">
                                Cultural Relevance: {Math.round(point.properties.cultural_relevance * 100)}%
                              </div>
                            )}
                          </div>
                        </Popup>
                      </Marker>
                    ))
                  )}

                {/* Pattern overlays as circles */}
                {filteredLayers
                  .filter(layer => layer.type === 'patterns' && layer.visible)
                  .forEach(layer =>
                    layer.points.map(point => (
                      <Circle
                        key={`circle-${point.id}`}
                        center={point.coordinates}
                        radius={5000}
                        pathOptions={{
                          color: layer.color,
                          fillColor: layer.color,
                          fillOpacity: 0.2,
                          opacity: 0.6
                        }}
                      />
                    ))
                  )}
              </MapContainer>
            )}
          </div>

          {/* Stats Panel */}
          <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg border border-gray-200 p-4 z-40">
            <h4 className="text-sm font-medium text-gray-900 mb-2">Map Statistics</h4>
            <div className="space-y-1 text-xs text-gray-600">
              <div>Total Points: {layers.reduce((sum, layer) => sum + layer.points.length, 0)}</div>
              <div>Visible Layers: {layers.filter(l => l.visible).length}/{layers.length}</div>
              <div>Sacred Sites: {layers.find(l => l.type === 'sacred_sites')?.points.length || 0}</div>
              <div>Patterns: {layers.find(l => l.type === 'patterns')?.points.length || 0}</div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}