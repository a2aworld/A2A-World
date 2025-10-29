/**
 * A2A World Platform - Scatter Plot Component
 * 
 * Coordinate and relationship visualization with clustering and regression support.
 * Used for spatial data analysis, pattern correlation, and multi-dimensional data exploration.
 */

import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';
import { clsx } from 'clsx';

export interface ScatterDataPoint {
  x: number;
  y: number;
  z?: number;
  name?: string;
  category?: string;
  color?: string;
  size?: number;
  metadata?: any;
}

export interface ScatterPlotProps {
  data: ScatterDataPoint[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: number | string;
  xLabel?: string;
  yLabel?: string;
  showGrid?: boolean;
  showTooltip?: boolean;
  showTrendLine?: boolean;
  colorBy?: string;
  sizeBy?: string;
  colors?: string[];
  dotSize?: number;
  minDotSize?: number;
  maxDotSize?: number;
  xAxisFormat?: 'number' | 'percentage' | 'currency';
  yAxisFormat?: 'number' | 'percentage' | 'currency';
  referenceLines?: Array<{
    type: 'x' | 'y';
    value: number;
    label?: string;
    color?: string;
  }>;
  className?: string;
  loading?: boolean;
  error?: string;
  onPointClick?: (point: ScatterDataPoint, index: number) => void;
}

const DEFAULT_COLORS = [
  '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
  '#EC4899', '#14B8A6', '#F97316', '#84CC16', '#6366F1'
];

const calculateTrendLine = (data: ScatterDataPoint[]) => {
  if (data.length < 2) return null;

  const n = data.length;
  const sumX = data.reduce((sum, point) => sum + point.x, 0);
  const sumY = data.reduce((sum, point) => sum + point.y, 0);
  const sumXY = data.reduce((sum, point) => sum + (point.x * point.y), 0);
  const sumXX = data.reduce((sum, point) => sum + (point.x * point.x), 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  const minX = Math.min(...data.map(p => p.x));
  const maxX = Math.max(...data.map(p => p.x));

  return {
    slope,
    intercept,
    points: [
      { x: minX, y: slope * minX + intercept },
      { x: maxX, y: slope * maxX + intercept }
    ]
  };
};

const CustomTooltip = ({ 
  active, 
  payload, 
  xAxisFormat,
  yAxisFormat 
}: any) => {
  if (active && payload && payload.length) {
    const point: ScatterDataPoint = payload[0].payload;
    
    const formatValue = (value: number, format?: string) => {
      switch (format) {
        case 'percentage':
          return `${(value * 100).toFixed(1)}%`;
        case 'currency':
          return new Intl.NumberFormat('en-US', { 
            style: 'currency', 
            currency: 'USD' 
          }).format(value);
        default:
          return value.toLocaleString();
      }
    };

    return (
      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
        {point.name && (
          <p className="font-medium text-gray-900 mb-2">{point.name}</p>
        )}
        <div className="text-sm space-y-1">
          <p className="text-gray-600">
            X: <span className="font-medium text-blue-600">{formatValue(point.x, xAxisFormat)}</span>
          </p>
          <p className="text-gray-600">
            Y: <span className="font-medium text-blue-600">{formatValue(point.y, yAxisFormat)}</span>
          </p>
          {point.z !== undefined && (
            <p className="text-gray-600">
              Z: <span className="font-medium text-blue-600">{point.z.toLocaleString()}</span>
            </p>
          )}
          {point.category && (
            <p className="text-gray-600">
              Category: <span className="font-medium text-green-600">{point.category}</span>
            </p>
          )}
        </div>
        {point.metadata && (
          <div className="mt-2 text-xs text-gray-500">
            {Object.entries(point.metadata).map(([key, value]) => (
              <div key={key}>
                {key}: {String(value)}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }
  return null;
};

export function ScatterPlot({
  data,
  title,
  subtitle,
  height = 400,
  width = '100%',
  xLabel,
  yLabel,
  showGrid = true,
  showTooltip = true,
  showTrendLine = false,
  colorBy,
  sizeBy,
  colors = DEFAULT_COLORS,
  dotSize = 6,
  minDotSize = 4,
  maxDotSize = 12,
  xAxisFormat = 'number',
  yAxisFormat = 'number',
  referenceLines = [],
  className,
  loading = false,
  error,
  onPointClick
}: ScatterPlotProps) {
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return [];

    // Calculate size scaling if sizeBy is specified
    let sizeRange = { min: minDotSize, max: maxDotSize };
    if (sizeBy && data.some(point => point.size !== undefined)) {
      const sizes = data.map(point => point.size || 0);
      const minSize = Math.min(...sizes);
      const maxSize = Math.max(...sizes);
      const sizeScale = (size: number) => 
        minDotSize + ((size - minSize) / (maxSize - minSize)) * (maxDotSize - minDotSize);
      
      return data.map((point, index) => ({
        ...point,
        color: point.color || (colorBy && point.category ? 
          colors[data.findIndex(p => p.category === point.category) % colors.length] : 
          colors[index % colors.length]),
        scaledSize: point.size !== undefined ? sizeScale(point.size) : dotSize
      }));
    }

    return data.map((point, index) => ({
      ...point,
      color: point.color || (colorBy && point.category ? 
        colors[data.findIndex(p => p.category === point.category) % colors.length] : 
        colors[index % colors.length]),
      scaledSize: dotSize
    }));
  }, [data, colorBy, sizeBy, colors, dotSize, minDotSize, maxDotSize]);

  const trendLine = useMemo(() => {
    if (showTrendLine && data && data.length > 1) {
      return calculateTrendLine(data);
    }
    return null;
  }, [data, showTrendLine]);

  const formatAxisValue = (value: number, format: string) => {
    switch (format) {
      case 'percentage':
        return `${value}%`;
      case 'currency':
        return `$${value}`;
      default:
        return value.toLocaleString();
    }
  };

  const handlePointClick = (point: any, index: number) => {
    if (onPointClick) {
      onPointClick(point, index);
    }
  };

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center', className)} style={{ height }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading scatter plot data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={clsx('flex items-center justify-center', className)} style={{ height }}>
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
      <div className={clsx('flex items-center justify-center', className)} style={{ height }}>
        <div className="text-center">
          <div className="text-gray-400 mb-2">
            <svg className="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">No data to display</p>
        </div>
      </div>
    );
  }

  // Group data by category for multiple scatter series
  const groupedData = useMemo(() => {
    if (!colorBy) return [{ data: processedData, name: 'Data' }];
    
    const groups = processedData.reduce((acc, point) => {
      const category = point.category || 'Unknown';
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(point);
      return acc;
    }, {} as Record<string, typeof processedData>);

    return Object.entries(groups).map(([name, data]) => ({ name, data }));
  }, [processedData, colorBy]);

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
      
      <ResponsiveContainer width={width} height={height}>
        <ScatterChart
          margin={{
            top: 20,
            right: 30,
            bottom: 20,
            left: 20,
          }}
        >
          {showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" strokeOpacity={0.5} />
          )}
          <XAxis
            type="number"
            dataKey="x"
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => formatAxisValue(value, xAxisFormat)}
            label={xLabel ? { value: xLabel, position: 'bottom', offset: 0 } : undefined}
          />
          <YAxis
            type="number"
            dataKey="y"
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => formatAxisValue(value, yAxisFormat)}
            label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft' } : undefined}
          />
          {showTooltip && (
            <Tooltip 
              content={(props) => (
                <CustomTooltip 
                  {...props}
                  xAxisFormat={xAxisFormat}
                  yAxisFormat={yAxisFormat}
                />
              )}
            />
          )}

          {/* Reference lines */}
          {referenceLines.map((refLine, index) => (
            <ReferenceLine
              key={index}
              {...(refLine.type === 'x' ? { x: refLine.value } : { y: refLine.value })}
              stroke={refLine.color || '#EF4444'}
              strokeDasharray="5 5"
              strokeOpacity={0.7}
              label={refLine.label ? { value: refLine.label, position: 'topRight' } : undefined}
            />
          ))}

          {/* Trend line */}
          {trendLine && (
            <ReferenceLine
              segment={trendLine.points.map(p => ({ x: p.x, y: p.y }))}
              stroke="#10B981"
              strokeWidth={2}
              strokeDasharray="3 3"
            />
          )}

          {/* Scatter plot series */}
          {groupedData.map((group, groupIndex) => (
            <Scatter
              key={group.name}
              name={group.name}
              data={group.data}
              fill={colors[groupIndex % colors.length]}
              onClick={handlePointClick}
            >
              {group.data.map((point: any, index: number) => (
                <Cell
                  key={`cell-${groupIndex}-${index}`}
                  fill={point.color}
                  r={point.scaledSize}
                  style={{ cursor: 'pointer' }}
                />
              ))}
            </Scatter>
          ))}
        </ScatterChart>
      </ResponsiveContainer>

      {/* Legend for categories */}
      {colorBy && groupedData.length > 1 && (
        <div className="mt-4 flex flex-wrap justify-center gap-4">
          {groupedData.map((group, index) => (
            <div key={group.name} className="flex items-center space-x-2">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: colors[index % colors.length] }}
              />
              <span className="text-sm text-gray-700">{group.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ScatterPlot;