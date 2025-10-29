/**
 * A2A World Platform - Progress Widget Component
 * 
 * Processing status visualization with progress bars and status indicators.
 * Used for data upload progress, agent task status, and system operations.
 */

import React from 'react';
import { clsx } from 'clsx';
import { CheckCircle, AlertCircle, Clock, Play, Pause, X } from 'lucide-react';

export interface ProgressStep {
  id: string;
  label: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  progress?: number;
  description?: string;
  timestamp?: string;
}

export interface ProgressWidgetProps {
  title: string;
  subtitle?: string;
  steps?: ProgressStep[];
  overallProgress?: number;
  status: 'idle' | 'running' | 'completed' | 'error' | 'paused';
  showSteps?: boolean;
  showTimestamps?: boolean;
  animated?: boolean;
  className?: string;
  onCancel?: () => void;
  onPause?: () => void;
  onResume?: () => void;
  onRetry?: () => void;
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'error':
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    case 'in_progress':
    case 'running':
      return <div className="h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
    case 'paused':
      return <Pause className="h-4 w-4 text-yellow-500" />;
    default:
      return <Clock className="h-4 w-4 text-gray-400" />;
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'text-green-600 bg-green-50 border-green-200';
    case 'error':
      return 'text-red-600 bg-red-50 border-red-200';
    case 'running':
    case 'in_progress':
      return 'text-blue-600 bg-blue-50 border-blue-200';
    case 'paused':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

const getProgressBarColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'bg-green-500';
    case 'error':
      return 'bg-red-500';
    case 'running':
    case 'in_progress':
      return 'bg-blue-500';
    case 'paused':
      return 'bg-yellow-500';
    default:
      return 'bg-gray-300';
  }
};

export function ProgressWidget({
  title,
  subtitle,
  steps = [],
  overallProgress,
  status,
  showSteps = true,
  showTimestamps = false,
  animated = true,
  className,
  onCancel,
  onPause,
  onResume,
  onRetry
}: ProgressWidgetProps) {
  const calculateOverallProgress = () => {
    if (overallProgress !== undefined) return overallProgress;
    if (steps.length === 0) return 0;
    
    const completedSteps = steps.filter(step => step.status === 'completed').length;
    return (completedSteps / steps.length) * 100;
  };

  const currentProgress = calculateOverallProgress();
  const statusColor = getStatusColor(status);
  const progressBarColor = getProgressBarColor(status);

  return (
    <div className={clsx('bg-white rounded-lg shadow border p-6', className)}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-1">
            {getStatusIcon(status)}
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          </div>
          {subtitle && (
            <p className="text-sm text-gray-600">{subtitle}</p>
          )}
        </div>
        
        {/* Action Buttons */}
        <div className="flex items-center space-x-2 ml-4">
          {status === 'running' && onPause && (
            <button
              onClick={onPause}
              className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
              title="Pause"
            >
              <Pause className="h-4 w-4" />
            </button>
          )}
          {status === 'paused' && onResume && (
            <button
              onClick={onResume}
              className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
              title="Resume"
            >
              <Play className="h-4 w-4" />
            </button>
          )}
          {status === 'error' && onRetry && (
            <button
              onClick={onRetry}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              Retry
            </button>
          )}
          {(status === 'running' || status === 'paused') && onCancel && (
            <button
              onClick={onCancel}
              className="p-1 text-gray-400 hover:text-red-600 transition-colors"
              title="Cancel"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Overall Progress Bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
          <span>Overall Progress</span>
          <span>{Math.round(currentProgress)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div
            className={clsx(
              'h-full transition-all duration-500 ease-out',
              progressBarColor,
              animated && 'transition-all'
            )}
            style={{ width: `${currentProgress}%` }}
          />
        </div>
      </div>

      {/* Status Badge */}
      <div className="mb-4">
        <span className={clsx(
          'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
          statusColor
        )}>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>

      {/* Steps */}
      {showSteps && steps.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700">Steps</h4>
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-start space-x-3">
              {/* Step Icon */}
              <div className="flex-shrink-0 mt-0.5">
                {getStatusIcon(step.status)}
              </div>
              
              {/* Step Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-gray-900">
                    {step.label}
                  </p>
                  {showTimestamps && step.timestamp && (
                    <span className="text-xs text-gray-500">
                      {new Date(step.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
                
                {step.description && (
                  <p className="text-xs text-gray-600 mt-1">
                    {step.description}
                  </p>
                )}
                
                {/* Individual Step Progress */}
                {step.progress !== undefined && step.status === 'in_progress' && (
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-1">
                      <div
                        className={clsx(
                          'h-full rounded-full transition-all duration-300',
                          getProgressBarColor(step.status)
                        )}
                        style={{ width: `${step.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Error Message */}
      {status === 'error' && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-700">
            An error occurred during processing. Please try again or contact support if the issue persists.
          </p>
        </div>
      )}
    </div>
  );
}

export default ProgressWidget;