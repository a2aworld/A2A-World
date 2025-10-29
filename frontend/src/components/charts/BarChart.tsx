/**
 * A2A World Platform - Bar Chart Component
 * 
 * Configurable bar chart with data labels, tooltips, and responsive design.
 * Used for dataset statistics, pattern counts, and distribution analysis.
 */

import React, { useMemo } from 'react';
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { clsx } from 'clsx';

export interface BarChartData {
  name: string;
  value: number;
  color?: string;
  metadata?: any;
}

export interface BarChartProps {
  data: BarChartData[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: number | string;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  showDataLabels?: boolean;
  colors?: string[];
  orientation?: 'vertical' | 'horizontal';
  xAxisKey?: string;
  yAxisKey?: string;
  className?: string;
  loading?: boolean;
  error?: string;
  onBarClick?: (data: BarChartData, index: number) => void;
}

const DEFAULT_COLORS = [
  '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
  '#EC4899', '#14B8A6', '#F97316', '#84CC16', '#6366F1'
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
        <p className="font-medium text-gray-900">{label}</p>
        <p className="text-sm text-gray-600">
          Value: <span className="font-medium text-blue-600">{payload[0].value.toLocaleString()}</span>
        </p>
        {data.metadata && (
          <div className="mt-2 text-xs text-gray-500">
            {Object.entries(data.metadata).map(([key, value]) => (
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

const DataLabel = (props: any) => {
  const { x, y, width, height, value } = props;
  const centerX = x + width / 2;
  const centerY = y - 5;

  if (height < 20) return null; // Don't show labels on very small bars

  return (
    <text
      x={centerX}
      y={centerY}
      fill="#374151"
      textAnchor="middle"
      dominantBaseline="middle"
      fontSize={12}
      fontWeight="500"
    >
      {value.toLocaleString()}
    </text>
  );
};

export function BarChart({
  data,
  title,
  subtitle,
  height = 400,
  width = '100%',
  showGrid = true,
  showLegend = false,
  showTooltip = true,
  showDataLabels = false,
  colors = DEFAULT_COLORS,
  orientation = 'vertical',
  xAxisKey = 'name',
  yAxisKey = 'value',
  className,
  loading = false,
  error,
  onBarClick
}: BarChartProps) {
  const chartData = useMemo(() => {
    return data.map((item, index) => ({
      ...item,
      color: item.color || colors[index % colors.length]
    }));
  }, [data, colors]);

  const handleBarClick = (data: any, index: number) => {
    if (onBarClick) {
      onBarClick(data, index);
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
        <RechartsBarChart
          data={chartData}
          margin={{
            top: showDataLabels ? 30 : 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          {showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" strokeOpacity={0.5} />
          )}
          <XAxis
            dataKey={xAxisKey}
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value.toLocaleString()}
          />
          {showTooltip && <Tooltip content={<CustomTooltip />} />}
          {showLegend && <Legend />}
          <Bar
            dataKey={yAxisKey}
            radius={[4, 4, 0, 0]}
            cursor="pointer"
            onClick={handleBarClick}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
            {showDataLabels && <DataLabel />}
          </Bar>
        </RechartsBarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default BarChart;