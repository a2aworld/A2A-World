/**
 * A2A World Platform - Histogram Chart Component
 * 
 * Data distribution analysis with customizable bins and statistical overlays.
 * Used for coordinate distribution, pattern density, and data quality metrics.
 */

import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';
import { clsx } from 'clsx';

export interface HistogramBin {
  binStart: number;
  binEnd: number;
  binCenter: number;
  count: number;
  frequency: number;
  label: string;
}

export interface HistogramProps {
  data: number[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: number | string;
  bins?: number;
  binSize?: number;
  showGrid?: boolean;
  showTooltip?: boolean;
  showMean?: boolean;
  showMedian?: boolean;
  showStdDev?: boolean;
  color?: string;
  className?: string;
  loading?: boolean;
  error?: string;
  onBinClick?: (bin: HistogramBin) => void;
}

const calculateStatistics = (data: number[]) => {
  if (data.length === 0) return { mean: 0, median: 0, stdDev: 0, min: 0, max: 0 };

  const sorted = [...data].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
  
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)];
  
  const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
  const stdDev = Math.sqrt(variance);

  return { mean, median, stdDev, min, max };
};

const createBins = (data: number[], binCount: number, binSize?: number): HistogramBin[] => {
  if (data.length === 0) return [];

  const stats = calculateStatistics(data);
  const { min, max } = stats;
  
  let actualBinCount = binCount;
  let actualBinSize = binSize;

  if (binSize) {
    actualBinCount = Math.ceil((max - min) / binSize);
  } else {
    actualBinSize = (max - min) / binCount;
  }

  const bins: HistogramBin[] = [];
  
  for (let i = 0; i < actualBinCount; i++) {
    const binStart = min + i * actualBinSize!;
    const binEnd = min + (i + 1) * actualBinSize!;
    const binCenter = (binStart + binEnd) / 2;
    
    const count = data.filter(val => {
      if (i === actualBinCount - 1) {
        // Include max value in the last bin
        return val >= binStart && val <= binEnd;
      }
      return val >= binStart && val < binEnd;
    }).length;
    
    const frequency = count / data.length;
    
    bins.push({
      binStart,
      binEnd,
      binCenter,
      count,
      frequency,
      label: `${binStart.toFixed(2)} - ${binEnd.toFixed(2)}`
    });
  }

  return bins;
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const bin: HistogramBin = payload[0].payload;
    return (
      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
        <p className="font-medium text-gray-900">Range: {bin.label}</p>
        <div className="mt-1 text-sm">
          <p className="text-gray-600">
            Count: <span className="font-medium text-blue-600">{bin.count}</span>
          </p>
          <p className="text-gray-600">
            Frequency: <span className="font-medium text-blue-600">{(bin.frequency * 100).toFixed(1)}%</span>
          </p>
          <p className="text-gray-600">
            Bin Center: <span className="font-medium text-blue-600">{bin.binCenter.toFixed(2)}</span>
          </p>
        </div>
      </div>
    );
  }
  return null;
};

export function HistogramChart({
  data,
  title,
  subtitle,
  height = 400,
  width = '100%',
  bins = 10,
  binSize,
  showGrid = true,
  showTooltip = true,
  showMean = true,
  showMedian = false,
  showStdDev = false,
  color = '#3B82F6',
  className,
  loading = false,
  error,
  onBinClick
}: HistogramProps) {
  const { histogramData, stats } = useMemo(() => {
    if (!data || data.length === 0) {
      return { histogramData: [], stats: { mean: 0, median: 0, stdDev: 0, min: 0, max: 0 } };
    }
    
    const histogramData = createBins(data, bins, binSize);
    const stats = calculateStatistics(data);
    return { histogramData, stats };
  }, [data, bins, binSize]);

  const handleBinClick = (bin: any) => {
    if (onBinClick) {
      onBinClick(bin);
    }
  };

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center', className)} style={{ height }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading histogram data...</p>
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

      {/* Statistics Summary */}
      <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="font-medium text-gray-900">{data.length.toLocaleString()}</div>
          <div className="text-gray-600">Count</div>
        </div>
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="font-medium text-gray-900">{stats.mean.toFixed(2)}</div>
          <div className="text-gray-600">Mean</div>
        </div>
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="font-medium text-gray-900">{stats.median.toFixed(2)}</div>
          <div className="text-gray-600">Median</div>
        </div>
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="font-medium text-gray-900">{stats.stdDev.toFixed(2)}</div>
          <div className="text-gray-600">Std Dev</div>
        </div>
      </div>

      <ResponsiveContainer width={width} height={height}>
        <BarChart
          data={histogramData}
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
            dataKey="binCenter"
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value.toFixed(1)}
          />
          <YAxis
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value.toLocaleString()}
          />
          {showTooltip && <Tooltip content={<CustomTooltip />} />}
          
          {/* Statistical reference lines */}
          {showMean && (
            <ReferenceLine
              x={stats.mean}
              stroke="#EF4444"
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{ value: `Mean: ${stats.mean.toFixed(2)}`, position: 'top' }}
            />
          )}
          {showMedian && (
            <ReferenceLine
              x={stats.median}
              stroke="#10B981"
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{ value: `Median: ${stats.median.toFixed(2)}`, position: 'top' }}
            />
          )}
          {showStdDev && (
            <>
              <ReferenceLine
                x={stats.mean - stats.stdDev}
                stroke="#8B5CF6"
                strokeDasharray="3 3"
                strokeOpacity={0.7}
                label={{ value: `-1σ`, position: 'top' }}
              />
              <ReferenceLine
                x={stats.mean + stats.stdDev}
                stroke="#8B5CF6"
                strokeDasharray="3 3"
                strokeOpacity={0.7}
                label={{ value: `+1σ`, position: 'top' }}
              />
            </>
          )}
          
          <Bar 
            dataKey="count" 
            fill={color}
            radius={[2, 2, 0, 0]}
            cursor="pointer"
            onClick={handleBinClick}
          >
            {histogramData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default HistogramChart;