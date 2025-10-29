/**
 * A2A World Platform - Line Chart Component
 * 
 * Time series and trend visualization with multiple lines support.
 * Used for pattern discovery trends, temporal analysis, and performance metrics.
 */

import React, { useMemo } from 'react';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart
} from 'recharts';
import { clsx } from 'clsx';
import { format, parseISO } from 'date-fns';

export interface LineChartDataPoint {
  timestamp: string | number | Date;
  [key: string]: any;
}

export interface LineConfig {
  key: string;
  name: string;
  color: string;
  strokeWidth?: number;
  strokeDashArray?: string;
  dot?: boolean;
  area?: boolean;
}

export interface LineChartProps {
  data: LineChartDataPoint[];
  lines: LineConfig[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: number | string;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  showDots?: boolean;
  smooth?: boolean;
  fillArea?: boolean;
  xAxisFormat?: 'date' | 'time' | 'datetime' | 'number' | 'string';
  yAxisFormat?: 'number' | 'percentage' | 'currency';
  referenceLines?: Array<{
    value: number;
    label: string;
    color?: string;
  }>;
  className?: string;
  loading?: boolean;
  error?: string;
  onPointClick?: (data: any, lineKey: string) => void;
}

const DEFAULT_COLORS = [
  '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
  '#EC4899', '#14B8A6', '#F97316', '#84CC16', '#6366F1'
];

const CustomTooltip = ({ 
  active, 
  payload, 
  label, 
  xAxisFormat,
  yAxisFormat 
}: any) => {
  if (active && payload && payload.length) {
    const formatXValue = (value: any) => {
      if (!xAxisFormat || xAxisFormat === 'string') return value;
      
      try {
        const date = typeof value === 'string' ? parseISO(value) : new Date(value);
        
        switch (xAxisFormat) {
          case 'date':
            return format(date, 'MMM dd, yyyy');
          case 'time':
            return format(date, 'HH:mm:ss');
          case 'datetime':
            return format(date, 'MMM dd, yyyy HH:mm');
          default:
            return value;
        }
      } catch {
        return value;
      }
    };

    const formatYValue = (value: number) => {
      if (!yAxisFormat || yAxisFormat === 'number') {
        return value.toLocaleString();
      }
      
      switch (yAxisFormat) {
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
        <p className="font-medium text-gray-900 mb-2">{formatXValue(label)}</p>
        <div className="space-y-1">
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center space-x-2">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm text-gray-600">{entry.name}:</span>
              <span className="text-sm font-medium" style={{ color: entry.color }}>
                {formatYValue(entry.value)}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

export function LineChart({
  data,
  lines,
  title,
  subtitle,
  height = 400,
  width = '100%',
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  showDots = true,
  smooth = true,
  fillArea = false,
  xAxisFormat = 'string',
  yAxisFormat = 'number',
  referenceLines = [],
  className,
  loading = false,
  error,
  onPointClick
}: LineChartProps) {
  const processedLines = useMemo(() => {
    return lines.map((line, index) => ({
      ...line,
      color: line.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
      strokeWidth: line.strokeWidth || 2,
      dot: line.dot !== undefined ? line.dot : showDots,
      area: line.area !== undefined ? line.area : fillArea
    }));
  }, [lines, showDots, fillArea]);

  const formatXAxis = (value: any) => {
    if (!xAxisFormat || xAxisFormat === 'string') return value;
    
    try {
      const date = typeof value === 'string' ? parseISO(value) : new Date(value);
      
      switch (xAxisFormat) {
        case 'date':
          return format(date, 'MMM dd');
        case 'time':
          return format(date, 'HH:mm');
        case 'datetime':
          return format(date, 'MM/dd HH:mm');
        default:
          return value;
      }
    } catch {
      return value;
    }
  };

  const formatYAxis = (value: number) => {
    if (!yAxisFormat || yAxisFormat === 'number') {
      return value.toLocaleString();
    }
    
    switch (yAxisFormat) {
      case 'percentage':
        return `${value}%`;
      case 'currency':
        return `$${value}`;
      default:
        return value.toLocaleString();
    }
  };

  const handlePointClick = (data: any, lineKey: string) => {
    if (onPointClick) {
      onPointClick(data, lineKey);
    }
  };

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center', className)} style={{ height }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading chart data...</p>
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">No data to display</p>
        </div>
      </div>
    );
  }

  const ChartComponent = fillArea ? AreaChart : RechartsLineChart;

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
        <ChartComponent
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          {showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" strokeOpacity={0.5} />
          )}
          <XAxis
            dataKey="timestamp"
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatXAxis}
          />
          <YAxis
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatYAxis}
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
          {showLegend && <Legend />}
          
          {referenceLines.map((refLine, index) => (
            <ReferenceLine
              key={index}
              y={refLine.value}
              stroke={refLine.color || '#EF4444'}
              strokeDasharray="5 5"
              strokeOpacity={0.7}
              label={{ value: refLine.label, position: 'topRight' }}
            />
          ))}
          
          {processedLines.map((line) => (
            fillArea ? (
              <Area
                key={line.key}
                type={smooth ? "monotone" : "linear"}
                dataKey={line.key}
                name={line.name}
                stroke={line.color}
                fill={line.color}
                fillOpacity={0.3}
                strokeWidth={line.strokeWidth}
                strokeDasharray={line.strokeDashArray}
                dot={line.dot}
                onClick={(data) => handlePointClick(data, line.key)}
              />
            ) : (
              <Line
                key={line.key}
                type={smooth ? "monotone" : "linear"}
                dataKey={line.key}
                name={line.name}
                stroke={line.color}
                strokeWidth={line.strokeWidth}
                strokeDasharray={line.strokeDashArray}
                dot={line.dot}
                onClick={(data) => handlePointClick(data, line.key)}
              />
            )
          ))}
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
}

export default LineChart;