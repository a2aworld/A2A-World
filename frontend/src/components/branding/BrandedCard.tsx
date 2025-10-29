/**
 * A2A World Platform - Branded Card Components
 * 
 * Enhanced card components with consistent A2A World branding.
 */

import React from 'react';
import { clsx } from 'clsx';
import { Logo } from './Logo';
import { Card, CardHeader } from '../ui/Card';

export interface BrandedCardProps {
  /**
   * Card content
   */
  children: React.ReactNode;
  /**
   * Card title
   */
  title?: string;
  /**
   * Card subtitle
   */
  subtitle?: string;
  /**
   * Whether to show A2A branding
   */
  showBranding?: boolean;
  /**
   * Branding position
   */
  brandingPosition?: 'header' | 'footer' | 'corner';
  /**
   * Card variant
   */
  variant?: 'default' | 'bordered' | 'elevated' | 'gradient';
  /**
   * Card padding
   */
  padding?: 'sm' | 'md' | 'lg';
  /**
   * Header action element
   */
  action?: React.ReactNode;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Footer content
   */
  footer?: React.ReactNode;
}

export function BrandedCard({
  children,
  title,
  subtitle,
  showBranding = false,
  brandingPosition = 'header',
  variant = 'default',
  padding = 'md',
  action,
  className,
  footer
}: BrandedCardProps) {
  const cardClasses = clsx(
    'bg-white rounded-lg relative',
    {
      'shadow-sm border border-gray-200': variant === 'bordered',
      'shadow-md': variant === 'default',
      'shadow-lg': variant === 'elevated',
      'bg-gradient-to-br from-blue-50 via-white to-purple-50 border border-blue-100': variant === 'gradient'
    },
    className
  );

  const headerContent = (title || subtitle) && (
    <CardHeader
      title={title || ''}
      subtitle={subtitle}
      action={
        <div className="flex items-center space-x-3">
          {showBranding && brandingPosition === 'header' && (
            <Logo variant="icon" size="sm" />
          )}
          {action}
        </div>
      }
    />
  );

  const footerContent = (footer || (showBranding && brandingPosition === 'footer')) && (
    <div className="border-t border-gray-200 pt-4 mt-4">
      <div className="flex justify-between items-center">
        <div>{footer}</div>
        {showBranding && brandingPosition === 'footer' && (
          <div className="flex items-center text-xs text-gray-500">
            <Logo variant="icon" size="xs" className="mr-1" />
            <span>A2A World</span>
          </div>
        )}
      </div>
    </div>
  );

  const cornerBranding = showBranding && brandingPosition === 'corner' && (
    <div className="absolute top-3 right-3 opacity-10 hover:opacity-30 transition-opacity">
      <Logo variant="icon" size="sm" />
    </div>
  );

  return (
    <div className={cardClasses}>
      {cornerBranding}
      <div className={clsx(
        {
          'p-3': padding === 'sm',
          'p-6': padding === 'md',
          'p-8': padding === 'lg',
        }
      )}>
        {headerContent}
        {children}
        {footerContent}
      </div>
    </div>
  );
}

export interface MetricCardProps {
  /**
   * Metric title
   */
  title: string;
  /**
   * Metric value
   */
  value: string | number;
  /**
   * Metric subtitle/description
   */
  subtitle?: string;
  /**
   * Icon component
   */
  icon?: React.ComponentType<{ className?: string }>;
  /**
   * Trend information
   */
  trend?: {
    value: number;
    label: string;
    direction: 'up' | 'down' | 'neutral';
  };
  /**
   * Color theme
   */
  color?: 'blue' | 'green' | 'purple' | 'yellow' | 'red' | 'gray';
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color = 'blue',
  className
}: MetricCardProps) {
  const colorMap = {
    blue: 'text-blue-600 bg-blue-50',
    green: 'text-green-600 bg-green-50',
    purple: 'text-purple-600 bg-purple-50',
    yellow: 'text-yellow-600 bg-yellow-50',
    red: 'text-red-600 bg-red-50',
    gray: 'text-gray-600 bg-gray-50'
  };

  const trendColorMap = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-gray-600'
  };

  const trendSymbol = {
    up: '↗',
    down: '↘',
    neutral: '→'
  };

  return (
    <BrandedCard
      variant="bordered"
      padding="md"
      showBranding={true}
      brandingPosition="corner"
      className={className}
    >
      <div className="flex items-center">
        {Icon && (
          <div className={clsx('p-2 rounded-lg mr-4', colorMap[color])}>
            <Icon className="h-6 w-6" />
          </div>
        )}
        
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 truncate">
            {title}
          </p>
          <div className="flex items-baseline space-x-2">
            <p className="text-2xl font-bold text-gray-900">
              {value}
            </p>
            {trend && (
              <div className={clsx(
                'flex items-center text-sm font-medium',
                trendColorMap[trend.direction]
              )}>
                <span className="mr-1">{trendSymbol[trend.direction]}</span>
                <span>{trend.value}% {trend.label}</span>
              </div>
            )}
          </div>
          {subtitle && (
            <p className="text-sm text-gray-600 mt-1">
              {subtitle}
            </p>
          )}
        </div>
      </div>
    </BrandedCard>
  );
}

export interface FeatureCardProps {
  /**
   * Feature title
   */
  title: string;
  /**
   * Feature description
   */
  description: string;
  /**
   * Feature icon
   */
  icon?: React.ComponentType<{ className?: string }>;
  /**
   * Feature status
   */
  status?: 'active' | 'inactive' | 'beta' | 'coming-soon';
  /**
   * Action button
   */
  action?: {
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary';
  };
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function FeatureCard({
  title,
  description,
  icon: Icon,
  status = 'active',
  action,
  className
}: FeatureCardProps) {
  const statusColors = {
    active: 'bg-green-100 text-green-800',
    inactive: 'bg-gray-100 text-gray-800',
    beta: 'bg-blue-100 text-blue-800',
    'coming-soon': 'bg-yellow-100 text-yellow-800'
  };

  const statusLabels = {
    active: 'Active',
    inactive: 'Inactive',
    beta: 'Beta',
    'coming-soon': 'Coming Soon'
  };

  return (
    <BrandedCard
      variant="elevated"
      showBranding={true}
      brandingPosition="footer"
      className={clsx('h-full', className)}
    >
      <div className="flex flex-col h-full">
        <div className="flex items-start justify-between mb-4">
          {Icon && (
            <div className="p-2 bg-blue-50 rounded-lg">
              <Icon className="h-6 w-6 text-blue-600" />
            </div>
          )}
          <span className={clsx(
            'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
            statusColors[status]
          )}>
            {statusLabels[status]}
          </span>
        </div>
        
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {title}
          </h3>
          <p className="text-gray-600 text-sm">
            {description}
          </p>
        </div>
        
        {action && (
          <div className="mt-4">
            <button
              onClick={action.onClick}
              className={clsx(
                'w-full px-4 py-2 rounded-md font-medium transition-colors',
                action.variant === 'primary'
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-200 text-gray-900 hover:bg-gray-300'
              )}
            >
              {action.label}
            </button>
          </div>
        )}
      </div>
    </BrandedCard>
  );
}

export default BrandedCard;