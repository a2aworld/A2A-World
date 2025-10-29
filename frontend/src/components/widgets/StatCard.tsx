/**
 * A2A World Platform - Statistics Card Component
 * 
 * Summary statistics with icons, trends, and visual indicators.
 * Used for displaying key metrics in dashboard layouts.
 */

import React from 'react';
import { clsx } from 'clsx';
import { TrendingUp, TrendingDown, Minus, LucideIcon } from 'lucide-react';

export interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: {
    value: number;
    label: string;
    direction: 'up' | 'down' | 'neutral';
  };
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'gray';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  error?: string;
  className?: string;
  onClick?: () => void;
}

const colorVariants = {
  blue: {
    bg: 'bg-blue-50',
    icon: 'text-blue-600',
    accent: 'border-blue-200'
  },
  green: {
    bg: 'bg-green-50',
    icon: 'text-green-600',
    accent: 'border-green-200'
  },
  yellow: {
    bg: 'bg-yellow-50',
    icon: 'text-yellow-600',
    accent: 'border-yellow-200'
  },
  red: {
    bg: 'bg-red-50',
    icon: 'text-red-600',
    accent: 'border-red-200'
  },
  purple: {
    bg: 'bg-purple-50',
    icon: 'text-purple-600',
    accent: 'border-purple-200'
  },
  gray: {
    bg: 'bg-gray-50',
    icon: 'text-gray-600',
    accent: 'border-gray-200'
  }
};

const sizeVariants = {
  sm: {
    padding: 'p-4',
    iconSize: 'h-5 w-5',
    valueText: 'text-xl',
    titleText: 'text-sm',
    subtitleText: 'text-xs'
  },
  md: {
    padding: 'p-6',
    iconSize: 'h-6 w-6',
    valueText: 'text-2xl',
    titleText: 'text-sm',
    subtitleText: 'text-xs'
  },
  lg: {
    padding: 'p-8',
    iconSize: 'h-8 w-8',
    valueText: 'text-3xl',
    titleText: 'text-base',
    subtitleText: 'text-sm'
  }
};

const getTrendIcon = (direction: 'up' | 'down' | 'neutral') => {
  switch (direction) {
    case 'up':
      return TrendingUp;
    case 'down':
      return TrendingDown;
    case 'neutral':
    default:
      return Minus;
  }
};

const getTrendColor = (direction: 'up' | 'down' | 'neutral') => {
  switch (direction) {
    case 'up':
      return 'text-green-600';
    case 'down':
      return 'text-red-600';
    case 'neutral':
    default:
      return 'text-gray-600';
  }
};

const formatValue = (value: string | number): string => {
  if (typeof value === 'number') {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toLocaleString();
  }
  return String(value);
};

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color = 'gray',
  size = 'md',
  loading = false,
  error,
  className,
  onClick
}: StatCardProps) {
  const colors = colorVariants[color];
  const sizes = sizeVariants[size];
  const TrendIcon = trend ? getTrendIcon(trend.direction) : null;
  const trendColor = trend ? getTrendColor(trend.direction) : '';

  if (loading) {
    return (
      <div className={clsx(
        'bg-white rounded-lg shadow border animate-pulse',
        sizes.padding,
        colors.accent,
        className
      )}>
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
            <div className="h-3 bg-gray-200 rounded w-2/3"></div>
          </div>
          {Icon && (
            <div className={clsx('rounded-lg p-2', colors.bg)}>
              <div className={clsx('bg-gray-200 rounded', sizes.iconSize)}></div>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={clsx(
        'bg-white rounded-lg shadow border border-red-200',
        sizes.padding,
        className
      )}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="text-red-500 mb-1">
              <svg className="h-5 w-5 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-xs text-gray-600">Error loading data</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={clsx(
        'bg-white rounded-lg shadow border transition-all duration-200',
        sizes.padding,
        colors.accent,
        onClick && 'cursor-pointer hover:shadow-md',
        className
      )}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1 min-w-0">
          {/* Title */}
          <p className={clsx(
            'font-medium text-gray-700 mb-1 truncate',
            sizes.titleText
          )}>
            {title}
          </p>

          {/* Value */}
          <p className={clsx(
            'font-bold text-gray-900 mb-1',
            sizes.valueText
          )}>
            {formatValue(value)}
          </p>

          {/* Subtitle or Trend */}
          {(subtitle || trend) && (
            <div className="flex items-center space-x-2">
              {trend && TrendIcon && (
                <div className="flex items-center space-x-1">
                  <TrendIcon className={clsx('h-3 w-3', trendColor)} />
                  <span className={clsx('font-medium', sizes.subtitleText, trendColor)}>
                    {trend.value > 0 ? '+' : ''}{trend.value}%
                  </span>
                  <span className={clsx('text-gray-500', sizes.subtitleText)}>
                    {trend.label}
                  </span>
                </div>
              )}
              {subtitle && !trend && (
                <span className={clsx('text-gray-500', sizes.subtitleText)}>
                  {subtitle}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Icon */}
        {Icon && (
          <div className={clsx('rounded-lg p-2 ml-4 flex-shrink-0', colors.bg)}>
            <Icon className={clsx(colors.icon, sizes.iconSize)} />
          </div>
        )}
      </div>
    </div>
  );
}

export default StatCard;