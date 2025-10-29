/**
 * A2A World Platform - Branded Loading Components
 * 
 * Loading spinners and components with A2A World branding.
 */

import React from 'react';
import { clsx } from 'clsx';
import { LoadingLogo, Logo } from './Logo';

export interface LoadingSpinnerProps {
  /**
   * Loading spinner size
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  /**
   * Loading text to display
   */
  text?: string;
  /**
   * Whether to show the logo
   */
  showLogo?: boolean;
  /**
   * Loading spinner variant
   */
  variant?: 'spinner' | 'dots' | 'pulse' | 'logo';
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Center the loading component
   */
  center?: boolean;
}

const sizeMap = {
  xs: 'h-4 w-4',
  sm: 'h-6 w-6',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
  xl: 'h-16 w-16'
};

const textSizeMap = {
  xs: 'text-xs',
  sm: 'text-sm',
  md: 'text-base',
  lg: 'text-lg',
  xl: 'text-xl'
};

export function LoadingSpinner({
  size = 'md',
  text = 'Loading...',
  showLogo = false,
  variant = 'spinner',
  className,
  center = true
}: LoadingSpinnerProps) {
  const containerClasses = clsx(
    'flex flex-col items-center justify-center space-y-3',
    center && 'min-h-32',
    className
  );

  if (variant === 'logo') {
    return (
      <div className={containerClasses}>
        <LoadingLogo size={size} />
      </div>
    );
  }

  const spinner = (
    <div className={clsx(
      'animate-spin rounded-full border-2 border-gray-300 border-t-blue-600',
      sizeMap[size]
    )} />
  );

  const dots = (
    <div className="flex space-x-1">
      <div className={clsx(
        'rounded-full bg-blue-600 animate-bounce',
        size === 'xs' ? 'h-2 w-2' : size === 'sm' ? 'h-3 w-3' : 'h-4 w-4'
      )} style={{ animationDelay: '0ms' }} />
      <div className={clsx(
        'rounded-full bg-blue-600 animate-bounce',
        size === 'xs' ? 'h-2 w-2' : size === 'sm' ? 'h-3 w-3' : 'h-4 w-4'
      )} style={{ animationDelay: '150ms' }} />
      <div className={clsx(
        'rounded-full bg-blue-600 animate-bounce',
        size === 'xs' ? 'h-2 w-2' : size === 'sm' ? 'h-3 w-3' : 'h-4 w-4'
      )} style={{ animationDelay: '300ms' }} />
    </div>
  );

  const pulse = (
    <div className={clsx(
      'rounded-full bg-blue-200 animate-pulse',
      sizeMap[size]
    )} />
  );

  return (
    <div className={containerClasses}>
      {showLogo && (
        <Logo variant="icon" size={size === 'xs' ? 'sm' : size} />
      )}
      
      {variant === 'spinner' && spinner}
      {variant === 'dots' && dots}
      {variant === 'pulse' && pulse}
      
      {text && (
        <div className={clsx(
          'text-gray-600 font-medium',
          textSizeMap[size]
        )}>
          {text}
        </div>
      )}
    </div>
  );
}

export interface PageLoadingProps {
  /**
   * Loading message
   */
  message?: string;
  /**
   * Show detailed loading steps
   */
  showSteps?: boolean;
  /**
   * Current loading step
   */
  currentStep?: string;
  /**
   * Progress percentage (0-100)
   */
  progress?: number;
}

export function PageLoading({
  message = 'Loading A2A World...',
  showSteps = false,
  currentStep,
  progress
}: PageLoadingProps) {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="mb-8">
          <Logo variant="full" size="xl" showText={true} />
        </div>
        
        <div className="space-y-6">
          <LoadingSpinner
            variant="logo"
            size="lg"
            text=""
            center={false}
          />
          
          <div className="space-y-2">
            <div className="text-lg font-medium text-gray-900">
              {message}
            </div>
            
            {currentStep && (
              <div className="text-sm text-gray-600">
                {currentStep}
              </div>
            )}
          </div>
          
          {progress !== undefined && (
            <div className="w-64 mx-auto">
              <div className="bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {progress}%
              </div>
            </div>
          )}
          
          {showSteps && (
            <div className="text-xs text-gray-500 space-y-1">
              <div>• Initializing platform components</div>
              <div>• Loading geospatial data</div>
              <div>• Connecting to pattern discovery agents</div>
              <div>• Preparing visualization engine</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export interface InlineLoadingProps {
  /**
   * Loading text
   */
  text?: string;
  /**
   * Loading size
   */
  size?: 'xs' | 'sm' | 'md';
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function InlineLoading({
  text = 'Loading...',
  size = 'sm',
  className
}: InlineLoadingProps) {
  return (
    <div className={clsx('flex items-center space-x-2', className)}>
      <LoadingSpinner
        size={size}
        variant="spinner"
        text=""
        center={false}
      />
      {text && (
        <span className={clsx(
          'text-gray-600',
          textSizeMap[size]
        )}>
          {text}
        </span>
      )}
    </div>
  );
}

export interface SkeletonProps {
  /**
   * Skeleton variant
   */
  variant?: 'text' | 'rectangular' | 'circular' | 'card';
  /**
   * Width (CSS value)
   */
  width?: string | number;
  /**
   * Height (CSS value)
   */
  height?: string | number;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Number of lines (for text variant)
   */
  lines?: number;
}

export function Skeleton({
  variant = 'rectangular',
  width,
  height,
  className,
  lines = 1
}: SkeletonProps) {
  const baseClasses = 'animate-pulse bg-gray-200';
  
  if (variant === 'text') {
    return (
      <div className={clsx('space-y-2', className)}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={clsx(baseClasses, 'h-4 rounded')}
            style={{
              width: i === lines - 1 && lines > 1 ? '75%' : width || '100%'
            }}
          />
        ))}
      </div>
    );
  }
  
  if (variant === 'circular') {
    return (
      <div
        className={clsx(baseClasses, 'rounded-full', className)}
        style={{
          width: width || height || '2rem',
          height: height || width || '2rem'
        }}
      />
    );
  }
  
  if (variant === 'card') {
    return (
      <div className={clsx('p-6 bg-white rounded-lg shadow', className)}>
        <div className="space-y-4">
          <Skeleton variant="rectangular" height="1rem" width="60%" />
          <Skeleton variant="text" lines={3} />
          <Skeleton variant="rectangular" height="2rem" width="40%" />
        </div>
      </div>
    );
  }
  
  return (
    <div
      className={clsx(baseClasses, 'rounded', className)}
      style={{
        width: width || '100%',
        height: height || '1rem'
      }}
    />
  );
}

export default LoadingSpinner;