/**
 * A2A World Platform - Significance Indicators Component
 * 
 * Visual indicators for statistical significance levels and validation results.
 */

import React from 'react';
import { clsx } from 'clsx';

export interface SignificanceIndicatorsProps {
  /**
   * Overall significance classification
   */
  significanceClassification: string;
  /**
   * Reliability score (0-1)
   */
  reliabilityScore: number;
  /**
   * Test results summary
   */
  testResults: {
    total: number;
    significant: number;
    highlySignificant: number;
  };
  /**
   * Show labels
   */
  showLabels?: boolean;
  /**
   * Component size
   */
  size?: 'sm' | 'md' | 'lg';
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function SignificanceIndicators({
  significanceClassification,
  reliabilityScore,
  testResults,
  showLabels = true,
  size = 'md',
  className
}: SignificanceIndicatorsProps) {
  const sizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  };

  const getSignificanceColor = (classification: string): string => {
    switch (classification.toLowerCase()) {
      case 'very_high':
        return 'bg-green-500';
      case 'high':
        return 'bg-blue-500';
      case 'moderate':
        return 'bg-yellow-500';
      case 'low':
        return 'bg-orange-500';
      case 'not_significant':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getReliabilityColor = (score: number): string => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-blue-500';
    if (score >= 0.4) return 'bg-yellow-500';
    if (score >= 0.2) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getSignificanceLabel = (classification: string): string => {
    switch (classification.toLowerCase()) {
      case 'very_high':
        return 'Very High Significance';
      case 'high':
        return 'High Significance';
      case 'moderate':
        return 'Moderate Significance';
      case 'low':
        return 'Low Significance';
      case 'not_significant':
        return 'Not Significant';
      default:
        return 'Unknown Significance';
    }
  };

  const significanceRate = testResults.total > 0 ? testResults.significant / testResults.total : 0;
  const highSignificanceRate = testResults.total > 0 ? testResults.highlySignificant / testResults.total : 0;

  return (
    <div className={clsx('space-y-4', sizeClasses[size], className)}>
      {/* Overall Significance Indicator */}
      <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-3">
          <div className={clsx(
            'w-4 h-4 rounded-full flex-shrink-0',
            getSignificanceColor(significanceClassification)
          )} />
          {showLabels && (
            <div>
              <div className="font-semibold text-gray-900">
                {getSignificanceLabel(significanceClassification)}
              </div>
              <div className="text-gray-600">
                Overall Pattern Classification
              </div>
            </div>
          )}
        </div>
        <div className="text-right">
          <div className="font-bold text-gray-900 capitalize">
            {significanceClassification.replace('_', ' ')}
          </div>
        </div>
      </div>

      {/* Reliability Score Indicator */}
      <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-3">
          <div className={clsx(
            'w-4 h-4 rounded-full flex-shrink-0',
            getReliabilityColor(reliabilityScore)
          )} />
          {showLabels && (
            <div>
              <div className="font-semibold text-gray-900">
                Reliability Score
              </div>
              <div className="text-gray-600">
                Statistical Framework Confidence
              </div>
            </div>
          )}
        </div>
        <div className="text-right">
          <div className="font-bold text-gray-900">
            {Math.round(reliabilityScore * 100)}%
          </div>
        </div>
      </div>

      {/* Test Results Breakdown */}
      <div className="p-4 bg-gray-50 rounded-lg">
        <div className="flex justify-between items-center mb-3">
          <h4 className="font-semibold text-gray-900">Statistical Tests</h4>
          <span className="text-gray-600">
            {testResults.significant}/{testResults.total} Significant
          </span>
        </div>
        
        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-3 mb-3">
          <div className="flex h-full rounded-full overflow-hidden">
            {/* Highly Significant */}
            <div 
              className="bg-green-500 transition-all duration-300"
              style={{ width: `${highSignificanceRate * 100}%` }}
            />
            {/* Significant */}
            <div 
              className="bg-blue-500 transition-all duration-300"
              style={{ width: `${(significanceRate - highSignificanceRate) * 100}%` }}
            />
          </div>
        </div>

        {/* Legend */}
        <div className="flex justify-between text-xs text-gray-600">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span>Highly Significant ({testResults.highlySignificant})</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-blue-500 rounded-full" />
              <span>Significant ({testResults.significant - testResults.highlySignificant})</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-gray-300 rounded-full" />
              <span>Not Significant ({testResults.total - testResults.significant})</span>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Interpretation */}
      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-blue-800">
          <strong>Interpretation:</strong> {getInterpretation(significanceClassification, reliabilityScore, significanceRate)}
        </div>
      </div>
    </div>
  );
}

function getInterpretation(classification: string, reliability: number, significanceRate: number): string {
  if (classification === 'very_high' && reliability >= 0.8) {
    return 'Pattern shows very strong statistical evidence with high reliability. Suitable for research publication.';
  }
  
  if (classification === 'high' && reliability >= 0.6) {
    return 'Pattern shows strong statistical evidence with good reliability. Additional validation recommended.';
  }
  
  if (classification === 'moderate' && significanceRate >= 0.5) {
    return 'Pattern shows moderate statistical evidence. Results should be interpreted with caution.';
  }
  
  if (classification === 'low' || significanceRate < 0.3) {
    return 'Pattern shows weak statistical evidence. Consider alternative methods or additional data.';
  }
  
  if (classification === 'not_significant') {
    return 'Pattern does not show statistical significance. Review methodology and data quality.';
  }
  
  return 'Statistical validation results require careful interpretation based on context and methodology.';
}

export default SignificanceIndicators;