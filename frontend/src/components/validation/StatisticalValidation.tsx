/**
 * A2A World Platform - Statistical Validation Component
 * 
 * Component for displaying comprehensive statistical validation results including
 * Moran's I analysis, significance classification, and validation metrics.
 */

import React from 'react';
import { clsx } from 'clsx';
import { BrandedCard, MetricCard } from '../branding/BrandedCard';
import { BrandedChart } from '../branding/BrandedChart';
import { SignificanceIndicators } from './SignificanceIndicators';

export interface StatisticalValidationResult {
  pattern_id: string;
  pattern_name: string;
  overall_significance: string;
  reliability_score: number;
  statistical_tests_performed: number;
  significant_tests: number;
  highly_significant_tests: number;
  validation_timestamp: string;
  processing_time_ms?: number;
  detailed_results?: {
    morans_i_analysis?: MoransIResult;
    null_hypothesis_tests?: NullHypothesisResult[];
    spatial_statistics?: SpatialStatisticsResult;
    significance_classification?: SignificanceClassification;
  };
  recommendations?: string[];
}

export interface MoransIResult {
  statistic_name: string;
  statistic_value: number;
  p_value: number;
  z_score: number;
  confidence_interval: [number, number];
  significant: boolean;
  interpretation: string;
  metadata?: {
    expected_value: number;
    variance: number;
    sample_size: number;
    weights_method: string;
    monte_carlo_p_value?: number;
  };
}

export interface NullHypothesisResult {
  test_type: string;
  observed_statistic: number;
  p_value: number;
  significant: boolean;
  effect_size?: number;
  confidence_interval?: [number, number];
  interpretation: string;
  metadata?: any;
}

export interface SpatialStatisticsResult {
  hotspots?: number;
  coldspots?: number;
  gini_coefficient?: number;
  silhouette_score?: number;
  cluster_quality?: string;
}

export interface SignificanceClassification {
  overall_classification: string;
  reliability_score: number;
  min_p_value: number;
  mean_p_value: number;
  very_high_significant: number;
  high_significant: number;
  moderate_significant: number;
  interpretation: string;
}

export interface StatisticalValidationProps {
  /**
   * Statistical validation results
   */
  validationResult: StatisticalValidationResult;
  /**
   * Loading state
   */
  loading?: boolean;
  /**
   * Error state
   */
  error?: string;
  /**
   * Show detailed results
   */
  showDetails?: boolean;
  /**
   * Callback for refreshing validation
   */
  onRefresh?: () => void;
  /**
   * Callback for generating report
   */
  onGenerateReport?: () => void;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function StatisticalValidation({
  validationResult,
  loading = false,
  error,
  showDetails = true,
  onRefresh,
  onGenerateReport,
  className
}: StatisticalValidationProps) {
  if (loading) {
    return (
      <BrandedCard
        title="Statistical Validation"
        subtitle="Processing statistical validation..."
        showBranding={true}
        brandingPosition="corner"
        className={className}
      >
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-blue-600 border-t-transparent" />
            <span className="text-lg text-gray-600">Analyzing statistical significance...</span>
          </div>
        </div>
      </BrandedCard>
    );
  }

  if (error) {
    return (
      <BrandedCard
        title="Statistical Validation"
        subtitle="Validation Error"
        showBranding={true}
        brandingPosition="corner"
        className={className}
      >
        <div className="text-center py-12">
          <div className="text-red-500 mb-4">
            <svg className="h-12 w-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-lg font-medium text-gray-900">Statistical Validation Failed</p>
          <p className="text-sm text-gray-600 mt-2">{error}</p>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Retry Validation
            </button>
          )}
        </div>
      </BrandedCard>
    );
  }

  const headerAction = (
    <div className="flex items-center space-x-2">
      {onRefresh && (
        <button
          onClick={onRefresh}
          className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100 transition-colors"
          title="Refresh validation"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      )}
      {onGenerateReport && (
        <button
          onClick={onGenerateReport}
          className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100 transition-colors"
          title="Generate validation report"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </button>
      )}
    </div>
  );

  return (
    <div className={clsx('space-y-6', className)}>
      {/* Overview Card */}
      <BrandedCard
        title={`Statistical Validation: ${validationResult.pattern_name}`}
        subtitle={`Pattern ID: ${validationResult.pattern_id}`}
        action={headerAction}
        showBranding={true}
        brandingPosition="corner"
        variant="bordered"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Overall Significance"
            value={validationResult.overall_significance.replace('_', ' ').toUpperCase()}
            color={getSignificanceColor(validationResult.overall_significance)}
            icon={StatisticalIcon}
          />
          
          <MetricCard
            title="Reliability Score"
            value={`${Math.round(validationResult.reliability_score * 100)}%`}
            subtitle={getReliabilityLabel(validationResult.reliability_score)}
            color={getReliabilityColor(validationResult.reliability_score)}
            icon={ReliabilityIcon}
          />
          
          <MetricCard
            title="Statistical Tests"
            value={`${validationResult.significant_tests}/${validationResult.statistical_tests_performed}`}
            subtitle={`${validationResult.highly_significant_tests} highly significant`}
            color="blue"
            icon={TestsIcon}
          />
          
          <MetricCard
            title="Processing Time"
            value={validationResult.processing_time_ms ? `${validationResult.processing_time_ms}ms` : 'N/A'}
            subtitle="Analysis duration"
            color="gray"
            icon={ClockIcon}
          />
        </div>

        {/* Significance Indicators */}
        <div className="mt-6">
          <SignificanceIndicators
            significanceClassification={validationResult.overall_significance}
            reliabilityScore={validationResult.reliability_score}
            testResults={{
              total: validationResult.statistical_tests_performed,
              significant: validationResult.significant_tests,
              highlySignificant: validationResult.highly_significant_tests
            }}
          />
        </div>
      </BrandedCard>

      {/* Detailed Results */}
      {showDetails && validationResult.detailed_results && (
        <>
          {/* Moran's I Analysis */}
          {validationResult.detailed_results.morans_i_analysis && (
            <MoransIAnalysisCard 
              result={validationResult.detailed_results.morans_i_analysis} 
            />
          )}

          {/* Null Hypothesis Tests */}
          {validationResult.detailed_results.null_hypothesis_tests && (
            <NullHypothesisTestsCard 
              results={validationResult.detailed_results.null_hypothesis_tests} 
            />
          )}

          {/* Spatial Statistics */}
          {validationResult.detailed_results.spatial_statistics && (
            <SpatialStatisticsCard 
              result={validationResult.detailed_results.spatial_statistics} 
            />
          )}

          {/* Significance Classification Details */}
          {validationResult.detailed_results.significance_classification && (
            <SignificanceClassificationCard 
              classification={validationResult.detailed_results.significance_classification} 
            />
          )}
        </>
      )}

      {/* Recommendations */}
      {validationResult.recommendations && validationResult.recommendations.length > 0 && (
        <BrandedCard
          title="Validation Recommendations"
          variant="bordered"
          showBranding={true}
          brandingPosition="corner"
        >
          <div className="space-y-3">
            {validationResult.recommendations.map((recommendation, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                  {index + 1}
                </div>
                <p className="text-sm text-gray-700">{recommendation}</p>
              </div>
            ))}
          </div>
        </BrandedCard>
      )}
    </div>
  );
}

// Sub-components for detailed results

function MoransIAnalysisCard({ result }: { result: MoransIResult }) {
  return (
    <BrandedChart
      title="Moran's I Spatial Autocorrelation Analysis"
      subtitle={result.interpretation}
      showBranding={true}
      dataSource="Statistical Validation Framework"
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Moran's I Statistic</label>
            <div className="text-2xl font-bold text-gray-900">{result.statistic_value.toFixed(6)}</div>
            <div className="text-sm text-gray-600">
              Expected: {result.metadata?.expected_value?.toFixed(6) || 'N/A'}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">P-Value</label>
            <div className={clsx(
              'text-2xl font-bold',
              result.p_value < 0.001 ? 'text-green-600' :
              result.p_value < 0.01 ? 'text-blue-600' :
              result.p_value < 0.05 ? 'text-yellow-600' : 'text-red-600'
            )}>
              {result.p_value < 0.001 ? '< 0.001' : result.p_value.toFixed(6)}
            </div>
            <div className="text-sm text-gray-600">
              Monte Carlo: {result.metadata?.monte_carlo_p_value?.toFixed(6) || 'N/A'}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Z-Score</label>
            <div className="text-2xl font-bold text-gray-900">{result.z_score.toFixed(3)}</div>
            <div className="text-sm text-gray-600">Standard deviations from expected</div>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Confidence Interval (95%)</label>
            <div className="text-lg font-semibold text-gray-900">
              [{result.confidence_interval[0].toFixed(6)}, {result.confidence_interval[1].toFixed(6)}]
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Sample Size</label>
            <div className="text-lg font-semibold text-gray-900">
              {result.metadata?.sample_size || 'N/A'}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Weights Method</label>
            <div className="text-lg font-semibold text-gray-900 capitalize">
              {result.metadata?.weights_method || 'N/A'}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-center">
          <div className={clsx(
            'px-6 py-4 rounded-lg text-center',
            result.significant 
              ? 'bg-green-50 text-green-800 border border-green-200'
              : 'bg-red-50 text-red-800 border border-red-200'
          )}>
            <div className="text-2xl font-bold">
              {result.significant ? '✓' : '✗'}
            </div>
            <div className="text-sm font-medium mt-1">
              {result.significant ? 'Significant' : 'Not Significant'}
            </div>
            <div className="text-xs mt-1">
              Spatial Autocorrelation
            </div>
          </div>
        </div>
      </div>
    </BrandedChart>
  );
}

function NullHypothesisTestsCard({ results }: { results: NullHypothesisResult[] }) {
  return (
    <BrandedChart
      title="Null Hypothesis Testing Results"
      subtitle={`${results.length} statistical tests performed`}
      showBranding={true}
      dataSource="Statistical Validation Framework"
    >
      <div className="space-y-4">
        {results.map((result, index) => (
          <div key={index} className="p-4 border border-gray-200 rounded-lg">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h4 className="font-semibold text-gray-900 capitalize">
                  {result.test_type.replace('_', ' ')}
                </h4>
                <p className="text-sm text-gray-600 mt-1">{result.interpretation}</p>
              </div>
              <div className={clsx(
                'px-3 py-1 rounded-full text-sm font-medium',
                result.significant 
                  ? 'bg-green-100 text-green-800'
                  : 'bg-red-100 text-red-800'
              )}>
                {result.significant ? 'Significant' : 'Not Significant'}
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
              <div>
                <span className="text-xs text-gray-500">Test Statistic</span>
                <div className="font-semibold">{result.observed_statistic.toFixed(4)}</div>
              </div>
              <div>
                <span className="text-xs text-gray-500">P-Value</span>
                <div className="font-semibold">
                  {result.p_value < 0.001 ? '< 0.001' : result.p_value.toFixed(6)}
                </div>
              </div>
              {result.effect_size && (
                <div>
                  <span className="text-xs text-gray-500">Effect Size</span>
                  <div className="font-semibold">{result.effect_size.toFixed(3)}</div>
                </div>
              )}
              {result.confidence_interval && (
                <div>
                  <span className="text-xs text-gray-500">95% CI</span>
                  <div className="font-semibold text-xs">
                    [{result.confidence_interval[0].toFixed(3)}, {result.confidence_interval[1].toFixed(3)}]
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </BrandedChart>
  );
}

function SpatialStatisticsCard({ result }: { result: SpatialStatisticsResult }) {
  return (
    <BrandedChart
      title="Spatial Statistics Summary"
      subtitle="Advanced spatial analysis results"
      showBranding={true}
      dataSource="Statistical Validation Framework"
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {result.hotspots !== undefined && (
          <div className="text-center">
            <div className="text-3xl font-bold text-red-600">{result.hotspots}</div>
            <div className="text-sm text-gray-600">Significant Hotspots</div>
          </div>
        )}
        
        {result.coldspots !== undefined && (
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">{result.coldspots}</div>
            <div className="text-sm text-gray-600">Significant Coldspots</div>
          </div>
        )}
        
        {result.gini_coefficient !== undefined && (
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600">{result.gini_coefficient.toFixed(3)}</div>
            <div className="text-sm text-gray-600">Gini Coefficient</div>
          </div>
        )}
        
        {result.silhouette_score !== undefined && (
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">{result.silhouette_score.toFixed(3)}</div>
            <div className="text-sm text-gray-600">Silhouette Score</div>
            {result.cluster_quality && (
              <div className="text-xs text-gray-500 capitalize mt-1">{result.cluster_quality}</div>
            )}
          </div>
        )}
      </div>
    </BrandedChart>
  );
}

function SignificanceClassificationCard({ classification }: { classification: SignificanceClassification }) {
  const significanceCounts = [
    { level: 'Very High (p < 0.001)', count: classification.very_high_significant, color: 'bg-green-500' },
    { level: 'High (p < 0.01)', count: classification.high_significant, color: 'bg-blue-500' },
    { level: 'Moderate (p < 0.05)', count: classification.moderate_significant, color: 'bg-yellow-500' }
  ];

  const total = classification.very_high_significant + classification.high_significant + classification.moderate_significant;

  return (
    <BrandedChart
      title="Significance Classification Details"
      subtitle={classification.interpretation}
      showBranding={true}
      dataSource="Statistical Validation Framework"
    >
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">Overall Classification</label>
            <div className={clsx(
              'text-2xl font-bold capitalize',
              getSignificanceColor(classification.overall_classification, 'text')
            )}>
              {classification.overall_classification.replace('_', ' ')}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Reliability Score</label>
            <div className="text-2xl font-bold text-gray-900">
              {Math.round(classification.reliability_score * 100)}%
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Minimum P-Value</label>
            <div className="text-2xl font-bold text-gray-900">
              {classification.min_p_value < 0.001 ? '< 0.001' : classification.min_p_value.toFixed(6)}
            </div>
          </div>
        </div>

        {/* Significance Distribution */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">Significance Distribution</label>
          <div className="space-y-3">
            {significanceCounts.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={clsx('w-4 h-4 rounded', item.color)} />
                  <span className="text-sm font-medium text-gray-900">{item.level}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-semibold text-gray-900">{item.count}</span>
                  {total > 0 && (
                    <span className="text-xs text-gray-500">
                      ({Math.round((item.count / total) * 100)}%)
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </BrandedChart>
  );
}

// Helper functions and icons

function getSignificanceColor(significance: string, type: 'bg' | 'text' = 'bg'): string {
  const colorMap = {
    'very_high': type === 'bg' ? 'green' : 'text-green-600',
    'high': type === 'bg' ? 'blue' : 'text-blue-600', 
    'moderate': type === 'bg' ? 'yellow' : 'text-yellow-600',
    'low': type === 'bg' ? 'red' : 'text-red-600',
    'not_significant': type === 'bg' ? 'gray' : 'text-gray-600'
  };
  
  return colorMap[significance as keyof typeof colorMap] || (type === 'bg' ? 'gray' : 'text-gray-600');
}

function getReliabilityColor(score: number): 'green' | 'blue' | 'yellow' | 'red' | 'gray' {
  if (score >= 0.8) return 'green';
  if (score >= 0.6) return 'blue';
  if (score >= 0.4) return 'yellow';
  if (score >= 0.2) return 'red';
  return 'gray';
}

function getReliabilityLabel(score: number): string {
  if (score >= 0.8) return 'Highly Reliable';
  if (score >= 0.6) return 'Reliable';
  if (score >= 0.4) return 'Moderate';
  if (score >= 0.2) return 'Low Reliability';
  return 'Very Low';
}

// Icons
function StatisticalIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  );
}

function ReliabilityIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function TestsIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
    </svg>
  );
}

function ClockIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

export default StatisticalValidation;