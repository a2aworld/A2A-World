/**
 * A2A World Platform - Validation Charts Component
 *
 * Specialized charts for visualizing multi-layered validation results
 * including cultural, ethical, and statistical validation metrics.
 */

import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, PieChart, Pie, Cell } from 'recharts';
import { clsx } from 'clsx';

export interface ValidationResult {
  id: string;
  name: string;
  cultural_score: number;
  ethical_score: number;
  statistical_score: number;
  overall_score: number;
  issues: {
    cultural: string[];
    ethical: string[];
    statistical: string[];
  };
  recommendations: {
    cultural: string[];
    ethical: string[];
    statistical: string[];
  };
}

export interface ValidationChartsProps {
  data: ValidationResult[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: string;
  showCultural?: boolean;
  showEthical?: boolean;
  showStatistical?: boolean;
  chartType?: 'bar' | 'radar' | 'pie' | 'combined';
  className?: string;
  loading?: boolean;
  error?: string;
  onResultClick?: (result: ValidationResult) => void;
}

const COLORS = {
  cultural: '#10B981', // green
  ethical: '#EF4444', // red
  statistical: '#3B82F6', // blue
  overall: '#8B5CF6' // purple
};

const PIE_COLORS = ['#10B981', '#EF4444', '#3B82F6', '#8B5CF6', '#F59E0B', '#EC4899'];

export function ValidationCharts({
  data,
  title,
  subtitle,
  height = 400,
  width = '100%',
  showCultural = true,
  showEthical = true,
  showStatistical = true,
  chartType = 'combined',
  className,
  loading = false,
  error,
  onResultClick
}: ValidationChartsProps) {
  // Process data for different chart types
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return { barData: [], radarData: [], pieData: [] };

    // Bar chart data
    const barData = data.map(item => ({
      name: item.name.length > 15 ? item.name.substring(0, 12) + '...' : item.name,
      fullName: item.name,
      cultural: showCultural ? item.cultural_score * 100 : 0,
      ethical: showEthical ? item.ethical_score * 100 : 0,
      statistical: showStatistical ? item.statistical_score * 100 : 0,
      overall: item.overall_score * 100,
      item
    }));

    // Radar chart data (showing averages)
    const avgCultural = data.reduce((sum, item) => sum + item.cultural_score, 0) / data.length;
    const avgEthical = data.reduce((sum, item) => sum + item.ethical_score, 0) / data.length;
    const avgStatistical = data.reduce((sum, item) => sum + item.statistical_score, 0) / data.length;

    const radarData = [
      { subject: 'Cultural', A: avgCultural * 100, B: 100 },
      { subject: 'Ethical', A: avgEthical * 100, B: 100 },
      { subject: 'Statistical', A: avgStatistical * 100, B: 100 }
    ];

    // Pie chart data (distribution of validation types)
    const totalIssues = data.reduce((acc, item) => ({
      cultural: acc.cultural + item.issues.cultural.length,
      ethical: acc.ethical + item.issues.ethical.length,
      statistical: acc.statistical + item.issues.statistical.length
    }), { cultural: 0, ethical: 0, statistical: 0 });

    const pieData = [
      { name: 'Cultural Issues', value: totalIssues.cultural, color: COLORS.cultural },
      { name: 'Ethical Issues', value: totalIssues.ethical, color: COLORS.ethical },
      { name: 'Statistical Issues', value: totalIssues.statistical, color: COLORS.statistical }
    ].filter(item => item.value > 0);

    return { barData, radarData, pieData };
  }, [data, showCultural, showEthical, showStatistical]);

  const handleBarClick = (data: any) => {
    if (onResultClick && data?.item) {
      onResultClick(data.item);
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-medium">{data.fullName || label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.dataKey}: {entry.value.toFixed(1)}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading validation data...</p>
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">No validation data to display</p>
        </div>
      </div>
    );
  }

  const renderChart = () => {
    switch (chartType) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <BarChart data={processedData.barData} onClick={handleBarClick}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {showCultural && <Bar dataKey="cultural" fill={COLORS.cultural} name="Cultural" />}
              {showEthical && <Bar dataKey="ethical" fill={COLORS.ethical} name="Ethical" />}
              {showStatistical && <Bar dataKey="statistical" fill={COLORS.statistical} name="Statistical" />}
              <Bar dataKey="overall" fill={COLORS.overall} name="Overall" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'radar':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <RadarChart data={processedData.radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis domain={[0, 100]} />
              <Radar name="Average Score" dataKey="A" stroke={COLORS.overall} fill={COLORS.overall} fillOpacity={0.3} />
              <Tooltip />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <PieChart>
              <Pie
                data={processedData.pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {processedData.pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'combined':
      default:
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Bar Chart */}
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Validation Scores by Item</h4>
              <ResponsiveContainer width="100%" height={height / 2}>
                <BarChart data={processedData.barData.slice(0, 5)} onClick={handleBarClick}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  {showCultural && <Bar dataKey="cultural" fill={COLORS.cultural} name="Cultural" />}
                  {showEthical && <Bar dataKey="ethical" fill={COLORS.ethical} name="Ethical" />}
                  {showStatistical && <Bar dataKey="statistical" fill={COLORS.statistical} name="Statistical" />}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Radar Chart */}
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Average Validation Scores</h4>
              <ResponsiveContainer width="100%" height={height / 2}>
                <RadarChart data={processedData.radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis domain={[0, 100]} />
                  <Radar name="Average Score" dataKey="A" stroke={COLORS.overall} fill={COLORS.overall} fillOpacity={0.3} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Issues Distribution */}
            {processedData.pieData.length > 0 && (
              <div className="lg:col-span-2">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Issues Distribution</h4>
                <ResponsiveContainer width="100%" height={height / 3}>
                  <PieChart>
                    <Pie
                      data={processedData.pieData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={60}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {processedData.pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        );
    }
  };

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

      <div className="border rounded-lg p-4 bg-white">
        {renderChart()}
      </div>

      {/* Summary Statistics */}
      <div className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-green-50 p-3 rounded-lg">
          <div className="text-sm text-green-600 font-medium">Cultural Avg</div>
          <div className="text-lg font-bold text-green-900">
            {(data.reduce((sum, item) => sum + item.cultural_score, 0) / data.length * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-red-50 p-3 rounded-lg">
          <div className="text-sm text-red-600 font-medium">Ethical Avg</div>
          <div className="text-lg font-bold text-red-900">
            {(data.reduce((sum, item) => sum + item.ethical_score, 0) / data.length * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-blue-50 p-3 rounded-lg">
          <div className="text-sm text-blue-600 font-medium">Statistical Avg</div>
          <div className="text-lg font-bold text-blue-900">
            {(data.reduce((sum, item) => sum + item.statistical_score, 0) / data.length * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-purple-50 p-3 rounded-lg">
          <div className="text-sm text-purple-600 font-medium">Overall Avg</div>
          <div className="text-lg font-bold text-purple-900">
            {(data.reduce((sum, item) => sum + item.overall_score, 0) / data.length * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}

export default ValidationCharts;