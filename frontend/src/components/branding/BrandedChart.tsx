/**
 * A2A World Platform - Branded Chart Components
 * 
 * Chart wrapper components with consistent A2A World branding.
 */

import React from 'react';
import { clsx } from 'clsx';
import { Logo } from './Logo';
import { BrandedCard } from './BrandedCard';

export interface BrandedChartProps {
  /**
   * Chart content
   */
  children: React.ReactNode;
  /**
   * Chart title
   */
  title?: string;
  /**
   * Chart subtitle
   */
  subtitle?: string;
  /**
   * Whether to show A2A branding
   */
  showBranding?: boolean;
  /**
   * Chart data source attribution
   */
  dataSource?: string;
  /**
   * Export action
   */
  onExport?: () => void;
  /**
   * Refresh action
   */
  onRefresh?: () => void;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Chart loading state
   */
  loading?: boolean;
  /**
   * Chart error state
   */
  error?: string;
}

export function BrandedChart({
  children,
  title,
  subtitle,
  showBranding = true,
  dataSource,
  onExport,
  onRefresh,
  className,
  loading = false,
  error
}: BrandedChartProps) {
  const chartActions = (
    <div className="flex items-center space-x-2">
      {onRefresh && (
        <button
          onClick={onRefresh}
          className="p-1.5 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100 transition-colors"
          title="Refresh chart data"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      )}
      
      {onExport && (
        <button
          onClick={onExport}
          className="p-1.5 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100 transition-colors"
          title="Export chart"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </button>
      )}
    </div>
  );

  const footerContent = (dataSource || showBranding) && (
    <div className="flex justify-between items-center text-xs text-gray-500">
      <div>
        {dataSource && (
          <span>Data source: {dataSource}</span>
        )}
      </div>
      {showBranding && (
        <div className="flex items-center">
          <Logo variant="icon" size="xs" className="mr-1" />
          <span>A2A World Analytics</span>
        </div>
      )}
    </div>
  );

  return (
    <BrandedCard
      title={title}
      subtitle={subtitle}
      action={chartActions}
      footer={footerContent}
      variant="bordered"
      className={className}
    >
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center z-10">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent" />
              <span className="text-sm text-gray-600">Loading chart data...</span>
            </div>
          </div>
        )}
        
        {error && (
          <div className="p-8 text-center">
            <div className="text-red-500 mb-2">
              <svg className="h-8 w-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-900">Chart Error</p>
            <p className="text-xs text-gray-600 mt-1">{error}</p>
          </div>
        )}
        
        {!loading && !error && (
          <div className="min-h-64">
            {children}
          </div>
        )}
      </div>
    </BrandedCard>
  );
}

export interface ChartWatermarkProps {
  /**
   * Watermark text (defaults to "A2A World")
   */
  text?: string;
  /**
   * Watermark position
   */
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  /**
   * Watermark opacity
   */
  opacity?: number;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function ChartWatermark({
  text = 'A2A World',
  position = 'bottom-right',
  opacity = 0.1,
  className
}: ChartWatermarkProps) {
  const positionClasses = {
    'top-left': 'top-2 left-2',
    'top-right': 'top-2 right-2',
    'bottom-left': 'bottom-2 left-2',
    'bottom-right': 'bottom-2 right-2'
  };

  return (
    <div 
      className={clsx(
        'absolute pointer-events-none flex items-center text-xs font-medium text-gray-900',
        positionClasses[position],
        className
      )}
      style={{ opacity }}
    >
      <Logo variant="icon" size="xs" className="mr-1" />
      <span>{text}</span>
    </div>
  );
}

export interface PatternVisualizationProps {
  /**
   * Pattern data
   */
  pattern: {
    id: string;
    name: string;
    type: string;
    confidence: number;
    description?: string;
  };
  /**
   * Visualization content
   */
  children: React.ReactNode;
  /**
   * Additional actions
   */
  actions?: React.ReactNode;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function PatternVisualization({
  pattern,
  children,
  actions,
  className
}: PatternVisualizationProps) {
  const confidenceColor = pattern.confidence >= 0.8 
    ? 'text-green-600 bg-green-50' 
    : pattern.confidence >= 0.6 
    ? 'text-yellow-600 bg-yellow-50' 
    : 'text-red-600 bg-red-50';

  const headerAction = (
    <div className="flex items-center space-x-3">
      <div className={clsx(
        'px-2 py-1 rounded-md text-xs font-medium',
        confidenceColor
      )}>
        {Math.round(pattern.confidence * 100)}% Confidence
      </div>
      <div className="px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800 capitalize">
        {pattern.type}
      </div>
      {actions}
    </div>
  );

  return (
    <BrandedChart
      title={pattern.name}
      subtitle={pattern.description}
      showBranding={true}
      dataSource={`Pattern ID: ${pattern.id}`}
      className={className}
    >
      <div className="relative">
        <ChartWatermark position="bottom-right" />
        {children}
      </div>
    </BrandedChart>
  );
}

export default BrandedChart;