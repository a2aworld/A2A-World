/**
 * A2A World Platform - Confidence Indicator Component
 * 
 * Pattern confidence visualization with visual indicators and thresholds.
 * Used for displaying pattern discovery confidence scores and validation levels.
 */

import React from 'react';
import { clsx } from 'clsx';
import { TrendingUp, AlertTriangle, CheckCircle, Info } from 'lucide-react';

export interface ConfidenceIndicatorProps {
  value: number;
  threshold?: number;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showValue?: boolean;
  showLabel?: boolean;
  showThreshold?: boolean;
  animated?: boolean;
  className?: string;
}

const getConfidenceLevel = (value: number, threshold = 0.7) => {
  if (value >= 0.9) return 'excellent';
  if (value >= threshold) return 'good';
  if (value >= 0.5) return 'moderate';
  return 'low';
};

const getConfidenceColor = (level: string) => {
  switch (level) {
    case 'excellent':
      return {
        bg: 'bg-green-500',
        text: 'text-green-600',
        icon: 'text-green-500',
        border: 'border-green-200'
      };
    case 'good':
      return {
        bg: 'bg-blue-500',
        text: 'text-blue-600',
        icon: 'text-blue-500',
        border: 'border-blue-200'
      };
    case 'moderate':
      return {
        bg: 'bg-yellow-500',
        text: 'text-yellow-600',
        icon: 'text-yellow-500',
        border: 'border-yellow-200'
      };
    case 'low':
      return {
        bg: 'bg-red-500',
        text: 'text-red-600',
        icon: 'text-red-500',
        border: 'border-red-200'
      };
    default:
      return {
        bg: 'bg-gray-500',
        text: 'text-gray-600',
        icon: 'text-gray-500',
        border: 'border-gray-200'
      };
  }
};

const getConfidenceIcon = (level: string) => {
  switch (level) {
    case 'excellent':
      return CheckCircle;
    case 'good':
      return TrendingUp;
    case 'moderate':
      return Info;
    case 'low':
      return AlertTriangle;
    default:
      return Info;
  }
};

const getSizeClasses = (size: string) => {
  switch (size) {
    case 'sm':
      return {
        container: 'w-16 h-16',
        progress: 'w-16 h-16',
        icon: 'h-4 w-4',
        value: 'text-xs',
        label: 'text-xs'
      };
    case 'lg':
      return {
        container: 'w-32 h-32',
        progress: 'w-32 h-32',
        icon: 'h-6 w-6',
        value: 'text-lg',
        label: 'text-sm'
      };
    case 'md':
    default:
      return {
        container: 'w-24 h-24',
        progress: 'w-24 h-24',
        icon: 'h-5 w-5',
        value: 'text-sm',
        label: 'text-sm'
      };
  }
};

export function ConfidenceIndicator({
  value,
  threshold = 0.7,
  label = 'Confidence',
  size = 'md',
  showValue = true,
  showLabel = true,
  showThreshold = false,
  animated = true,
  className
}: ConfidenceIndicatorProps) {
  const level = getConfidenceLevel(value, threshold);
  const colors = getConfidenceColor(level);
  const sizes = getSizeClasses(size);
  const Icon = getConfidenceIcon(level);
  
  const percentage = Math.round(value * 100);
  const circumference = 2 * Math.PI * 45; // radius of 45 for the circle
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (value * circumference);

  return (
    <div className={clsx('flex flex-col items-center space-y-2', className)}>
      {/* Circular Progress */}
      <div className={clsx('relative', sizes.container)}>
        <svg
          className={clsx('transform -rotate-90', sizes.progress)}
          width="100%"
          height="100%"
          viewBox="0 0 100 100"
        >
          {/* Background Circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            stroke="currentColor"
            strokeWidth="6"
            fill="transparent"
            className="text-gray-200"
          />
          
          {/* Progress Circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            stroke="currentColor"
            strokeWidth="6"
            fill="transparent"
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className={clsx(
              colors.bg.replace('bg-', 'text-'),
              animated && 'transition-all duration-1000 ease-out'
            )}
          />
          
          {/* Threshold Line */}
          {showThreshold && (
            <line
              x1="50"
              y1="5"
              x2="50"
              y2="15"
              stroke="currentColor"
              strokeWidth="2"
              className="text-gray-600"
              transform={`rotate(${threshold * 360 - 90} 50 50)`}
            />
          )}
        </svg>
        
        {/* Center Content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <Icon className={clsx(sizes.icon, colors.icon)} />
          {showValue && (
            <span className={clsx('font-bold mt-1', sizes.value, colors.text)}>
              {percentage}%
            </span>
          )}
        </div>
      </div>
      
      {/* Label and Level */}
      {(showLabel || level) && (
        <div className="text-center space-y-1">
          {showLabel && (
            <p className={clsx('font-medium text-gray-700', sizes.label)}>
              {label}
            </p>
          )}
          <div className={clsx(
            'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border',
            colors.text,
            colors.border
          )}>
            {level.charAt(0).toUpperCase() + level.slice(1)}
          </div>
        </div>
      )}
      
      {/* Threshold Indicator */}
      {showThreshold && (
        <div className="text-center">
          <p className="text-xs text-gray-500">
            Threshold: {Math.round(threshold * 100)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default ConfidenceIndicator;