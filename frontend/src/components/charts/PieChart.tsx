/**
 * A2A World Platform - Pie Chart Component
 * 
 * Category distribution visualization with donut chart option.
 * Used for data quality distribution, file type statistics, and pattern type analysis.
 */

import React, { useMemo } from 'react';
import {
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
  LabelList
} from 'recharts';
import { clsx } from 'clsx';

export interface PieChartData {
  name: string;
  value: number;
  color?: string;
  metadata?: any;
}

export interface PieChartProps {
  data: PieChartData[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: number | string;
  showLegend?: boolean;
  showTooltip?: boolean;
  showLabels?: boolean;
  showValues?: boolean;
  showPercentages?: boolean;
  donut?: boolean;
  innerRadius?: number;
  outerRadius?: number;
  colors?: string[];
  className?: string;
  loading?: boolean;
  error?: string;
  onSliceClick?: (data: PieChartData, index: number) => void;
}

const DEFAULT_COLORS = [
  '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
  '#EC4899', '#14B8A6', '#F97316', '#84CC16', '#6366F1'
];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const total = payload[0].payload.total || 0;
    const percentage = total > 0 ? ((data.value / total) * 100).toFixed(1) : '0.0';
    
    return (
      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
        <p className="font-medium text-gray-900">{data.name}</p>
        <div className="mt-1 text-sm">
          <p className="text-gray-600">
            Value: <span className="font-medium text-blue-600">{data.value.toLocaleString()}</span>
          </p>
          <p className="text-gray-600">
            Percentage: <span className="font-medium text-blue-600">{percentage}%</span>
          </p>
        </div>
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

const CustomLabel = ({ 
  cx, 
  cy, 
  midAngle, 
  innerRadius, 
  outerRadius, 
  value, 
  name, 
  showLabels, 
  showValues, 
  showPercentages,
  total 
}: any) => {
  if (!showLabels && !showValues && !showPercentages) return null;

  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  
  const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : '0.0';
  
  let labelText = '';
  if (showLabels) labelText += name;
  if (showValues) labelText += (labelText ? '\n' : '') + value.toLocaleString();
  if (showPercentages) labelText += (labelText ? '\n' : '') + `${percentage}%`;

  // Don't show label if slice is too small
  if (parseFloat(percentage) < 2) return null;

  return (
    <text 
      x={x} 
      y={y} 
      fill="#374151" 
      textAnchor={x > cx ? 'start' : 'end'} 
      dominantBaseline="central"
      fontSize={12}
      fontWeight="500"
    >
      {labelText.split('\n').map((line, index) => (
        <tspan x={x} dy={index === 0 ? 0 : 14} key={index}>
          {line}
        </tspan>
      ))}
    </text>
  );
};

const CustomLegend = ({ payload }: any) => {
  return (
    <ul className="flex flex-wrap justify-center space-x-4 mt-4">
      {payload.map((entry: any, index: number) => (
        <li key={`legend-${index}`} className="flex items-center space-x-2 mb-2">
          <div 
            className="w-3 h-3 rounded-full" 
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-sm text-gray-700">{entry.value}</span>
        </li>
      ))}
    </ul>
  );
};

export function PieChart({
  data,
  title,
  subtitle,
  height = 400,
  width = '100%',
  showLegend = true,
  showTooltip = true,
  showLabels = false,
  showValues = false,
  showPercentages = true,
  donut = false,
  innerRadius,
  outerRadius,
  colors = DEFAULT_COLORS,
  className,
  loading = false,
  error,
  onSliceClick
}: PieChartProps) {
  const chartData = useMemo(() => {
    const total = data.reduce((sum, item) => sum + item.value, 0);
    return data.map((item, index) => ({
      ...item,
      color: item.color || colors[index % colors.length],
      total
    }));
  }, [data, colors]);

  const calculatedInnerRadius = useMemo(() => {
    if (innerRadius !== undefined) return innerRadius;
    return donut ? 60 : 0;
  }, [innerRadius, donut]);

  const calculatedOuterRadius = useMemo(() => {
    if (outerRadius !== undefined) return outerRadius;
    return Math.min(height / 2 - 20, 120);
  }, [outerRadius, height]);

  const handleSliceClick = (data: any, index: number) => {
    if (onSliceClick) {
      onSliceClick(data, index);
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">No data to display</p>
        </div>
      </div>
    );
  }

  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className={clsx('w-full', className)}>
      {(title || subtitle) && (
        <div className="mb-4 text-center">
          {title && (
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          )}
          {subtitle && (
            <p className="text-sm text-gray-600 mt-1">{subtitle}</p>
          )}
        </div>
      )}

      <ResponsiveContainer width={width} height={showLegend ? height - 60 : height}>
        <RechartsPieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={calculatedInnerRadius}
            outerRadius={calculatedOuterRadius}
            paddingAngle={2}
            dataKey="value"
            onClick={handleSliceClick}
            cursor="pointer"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
            {(showLabels || showValues || showPercentages) && (
              <LabelList 
                content={(props) => (
                  <CustomLabel 
                    {...props} 
                    showLabels={showLabels}
                    showValues={showValues}
                    showPercentages={showPercentages}
                    total={total}
                  />
                )}
              />
            )}
          </Pie>
          {showTooltip && <Tooltip content={<CustomTooltip />} />}
        </RechartsPieChart>
      </ResponsiveContainer>

      {donut && total > 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{total.toLocaleString()}</div>
            <div className="text-sm text-gray-600">Total</div>
          </div>
        </div>
      )}

      {showLegend && <CustomLegend payload={chartData} />}
    </div>
  );
}

export default PieChart;