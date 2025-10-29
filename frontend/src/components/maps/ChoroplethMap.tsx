/**
 * A2A World Platform - Choropleth Map Component
 * 
 * Pattern density visualization using colored regions/polygons.
 * Used for visualizing data density, statistical significance, and regional patterns.
 */

import React, { useMemo, useRef, useEffect } from 'react';
import { MapContainer, GeoJSON, TileLayer, useMap } from 'react-leaflet';
import { LatLngBounds, GeoJSON as LeafletGeoJSON } from 'leaflet';
import { clsx } from 'clsx';
import 'leaflet/dist/leaflet.css';

export interface ChoroplethData {
  id: string;
  value: number;
  properties: {
    name: string;
    [key: string]: any;
  };
  geometry: any; // GeoJSON geometry
}

export interface ColorScale {
  min: number;
  max: number;
  colors: string[];
}

export interface ChoroplethMapProps {
  data: ChoroplethData[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: string;
  center?: [number, number];
  zoom?: number;
  colorScale?: ColorScale;
  valueProperty?: string;
  showLegend?: boolean;
  showTooltip?: boolean;
  className?: string;
  loading?: boolean;
  error?: string;
  onFeatureClick?: (feature: ChoroplethData) => void;
  onFeatureHover?: (feature: ChoroplethData | null) => void;
}

const DEFAULT_COLOR_SCALE: ColorScale = {
  min: 0,
  max: 100,
  colors: ['#FEF0D9', '#FDCC8A', '#FC8D59', '#E34A33', '#B30000']
};

// Function to interpolate between colors
const interpolateColor = (color1: string, color2: string, factor: number): string => {
  const hex1 = color1.replace('#', '');
  const hex2 = color2.replace('#', '');
  
  const r1 = parseInt(hex1.substr(0, 2), 16);
  const g1 = parseInt(hex1.substr(2, 2), 16);
  const b1 = parseInt(hex1.substr(4, 2), 16);
  
  const r2 = parseInt(hex2.substr(0, 2), 16);
  const g2 = parseInt(hex2.substr(2, 2), 16);
  const b2 = parseInt(hex2.substr(4, 2), 16);
  
  const r = Math.round(r1 + (r2 - r1) * factor);
  const g = Math.round(g1 + (g2 - g1) * factor);
  const b = Math.round(b1 + (b2 - b1) * factor);
  
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
};

// Function to get color based on value and scale
const getColor = (value: number, colorScale: ColorScale): string => {
  const { min, max, colors } = colorScale;
  
  if (value <= min) return colors[0];
  if (value >= max) return colors[colors.length - 1];
  
  const normalizedValue = (value - min) / (max - min);
  const colorIndex = normalizedValue * (colors.length - 1);
  const lowerIndex = Math.floor(colorIndex);
  const upperIndex = Math.ceil(colorIndex);
  
  if (lowerIndex === upperIndex) return colors[lowerIndex];
  
  const factor = colorIndex - lowerIndex;
  return interpolateColor(colors[lowerIndex], colors[upperIndex], factor);
};

// Custom hook to handle map bounds fitting
const FitBoundsControl: React.FC<{ data: ChoroplethData[] }> = ({ data }) => {
  const map = useMap();
  
  useEffect(() => {
    if (data && data.length > 0) {
      try {
        const bounds = new LatLngBounds([]);
        data.forEach(feature => {
          if (feature.geometry && feature.geometry.coordinates) {
            // Extract coordinates from GeoJSON geometry
            const extractCoords = (coords: any): [number, number][] => {
              if (Array.isArray(coords[0])) {
                return coords.flatMap(extractCoords);
              } else {
                return [[coords[1], coords[0]]]; // GeoJSON is [lng, lat], Leaflet expects [lat, lng]
              }
            };
            
            const coordPairs = extractCoords(feature.geometry.coordinates);
            coordPairs.forEach(coord => bounds.extend(coord));
          }
        });
        
        if (bounds.isValid()) {
          map.fitBounds(bounds, { padding: [20, 20] });
        }
      } catch (error) {
        console.warn('Error fitting bounds:', error);
      }
    }
  }, [data, map]);
  
  return null;
};

// Legend component
const Legend: React.FC<{ colorScale: ColorScale }> = ({ colorScale }) => {
  const { min, max, colors } = colorScale;
  const steps = colors.length;
  
  return (
    <div className="absolute bottom-4 left-4 bg-white p-3 rounded-lg shadow-lg z-[1000]">
      <h4 className="text-sm font-medium text-gray-900 mb-2">Value Scale</h4>
      <div className="flex flex-col space-y-1">
        {colors.map((color, index) => {
          const value = min + (max - min) * (index / (steps - 1));
          return (
            <div key={index} className="flex items-center space-x-2">
              <div 
                className="w-4 h-4 border border-gray-300 rounded"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs text-gray-600">
                {value.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export function ChoroplethMap({
  data,
  title,
  subtitle,
  height = 500,
  width = '100%',
  center = [39.8283, -98.5795], // Center of US
  zoom = 4,
  colorScale = DEFAULT_COLOR_SCALE,
  valueProperty = 'value',
  showLegend = true,
  showTooltip = true,
  className,
  loading = false,
  error,
  onFeatureClick,
  onFeatureHover
}: ChoroplethMapProps) {
  const geoJsonRef = useRef<LeafletGeoJSON>(null);

  const processedColorScale = useMemo(() => {
    if (!data || data.length === 0) return colorScale;
    
    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return {
      ...colorScale,
      min: colorScale.min !== undefined ? colorScale.min : min,
      max: colorScale.max !== undefined ? colorScale.max : max
    };
  }, [data, colorScale]);

  const geoJsonData = useMemo(() => {
    if (!data || data.length === 0) return null;
    
    return {
      type: 'FeatureCollection',
      features: data.map(item => ({
        type: 'Feature',
        id: item.id,
        properties: {
          ...item.properties,
          value: item.value
        },
        geometry: item.geometry
      }))
    };
  }, [data]);

  const onEachFeature = (feature: any, layer: any) => {
    const value = feature.properties.value;
    const color = getColor(value, processedColorScale);
    
    layer.setStyle({
      fillColor: color,
      weight: 2,
      opacity: 1,
      color: 'white',
      dashArray: '3',
      fillOpacity: 0.7
    });

    if (showTooltip) {
      layer.bindTooltip(`
        <div class="p-2">
          <strong>${feature.properties.name}</strong><br/>
          Value: ${value.toLocaleString()}
        </div>
      `, {
        permanent: false,
        direction: 'center',
        className: 'custom-tooltip'
      });
    }

    layer.on({
      mouseover: (e: any) => {
        const layer = e.target;
        layer.setStyle({
          weight: 3,
          color: '#666',
          dashArray: '',
          fillOpacity: 0.8
        });
        
        if (onFeatureHover) {
          const choroplethData = data.find(d => d.id === feature.id);
          if (choroplethData) {
            onFeatureHover(choroplethData);
          }
        }
      },
      mouseout: (e: any) => {
        if (geoJsonRef.current) {
          geoJsonRef.current.resetStyle(e.target);
        }
        
        if (onFeatureHover) {
          onFeatureHover(null);
        }
      },
      click: (e: any) => {
        if (onFeatureClick) {
          const choroplethData = data.find(d => d.id === feature.id);
          if (choroplethData) {
            onFeatureClick(choroplethData);
          }
        }
      }
    });
  };

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading map data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="text-red-500 mb-2">
            <svg className="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="text-gray-400 mb-2">
            <svg className="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">No map data to display</p>
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('w-full', className)}>
      {(title || subtitle) && (
        <div className="mb-4">
          {title && (
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          )}
          {subtitle && (
            <p className="text-sm text-gray-600 mt-1">{subtitle}</p>
          )}
        </div>
      )}
      
      <div className="relative border rounded-lg overflow-hidden" style={{ height, width }}>
        <MapContainer
          center={center}
          zoom={zoom}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {geoJsonData && (
            <GeoJSON
              ref={geoJsonRef}
              data={geoJsonData}
              onEachFeature={onEachFeature}
            />
          )}
          
          <FitBoundsControl data={data} />
        </MapContainer>
        
        {showLegend && (
          <Legend colorScale={processedColorScale} />
        )}
      </div>
    </div>
  );
}

export default ChoroplethMap;