/**
 * A2A World Platform - Validation Reports Component
 * 
 * Component for generating and displaying statistical validation reports
 * with export capabilities and comprehensive analysis summaries.
 */

import React, { useState } from 'react';
import { clsx } from 'clsx';
import { BrandedCard } from '../branding/BrandedCard';
import { BrandedChart } from '../branding/BrandedChart';

export interface ValidationReport {
  report_id: string;
  validation_id: string;
  report_type: string;
  format: string;
  generated_at: string;
  report_content: {
    executive_summary: string;
    detailed_findings: any;
    statistical_tables: any;
    conclusions: string[];
    recommendations: string[];
    visualizations: any;
  };
  metadata: {
    report_version: string;
    generated_by: string;
    includes_visualizations: boolean;
  };
}

export interface ValidationReportsProps {
  /**
   * Pattern ID for validation reports
   */
  patternId?: string;
  /**
   * Validation ID for specific validation reports
   */
  validationId?: string;
  /**
   * Available validation reports
   */
  reports?: ValidationReport[];
  /**
   * Loading state
   */
  loading?: boolean;
  /**
   * Error state
   */
  error?: string;
  /**
   * Callback for generating new report
   */
  onGenerateReport?: (reportType: string, format: string) => void;
  /**
   * Callback for downloading report
   */
  onDownloadReport?: (reportId: string) => void;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function ValidationReports({
  patternId,
  validationId,
  reports = [],
  loading = false,
  error,
  onGenerateReport,
  onDownloadReport,
  className
}: ValidationReportsProps) {
  const [selectedReportType, setSelectedReportType] = useState<string>('comprehensive');
  const [selectedFormat, setSelectedFormat] = useState<string>('json');

  const reportTypes = [
    { value: 'comprehensive', label: 'Comprehensive Analysis', description: 'Complete statistical validation report' },
    { value: 'summary', label: 'Executive Summary', description: 'High-level findings and recommendations' },
    { value: 'technical', label: 'Technical Details', description: 'Detailed statistical methodology and results' },
    { value: 'dashboard', label: 'Dashboard Export', description: 'Dashboard data for external visualization' }
  ];

  const formats = [
    { value: 'json', label: 'JSON', extension: '.json', description: 'Machine-readable format' },
    { value: 'html', label: 'HTML', extension: '.html', description: 'Web-ready report' },
    { value: 'pdf', label: 'PDF', extension: '.pdf', description: 'Print-ready document' },
    { value: 'csv', label: 'CSV', extension: '.csv', description: 'Spreadsheet format' }
  ];

  const handleGenerateReport = () => {
    if (onGenerateReport) {
      onGenerateReport(selectedReportType, selectedFormat);
    }
  };

  const getReportStatusColor = (reportType: string) => {
    switch (reportType) {
      case 'comprehensive':
        return 'bg-blue-100 text-blue-800';
      case 'summary':
        return 'bg-green-100 text-green-800';
      case 'technical':
        return 'bg-purple-100 text-purple-800';
      case 'dashboard':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'json':
        return '{ }';
      case 'html':
        return '<>';
      case 'pdf':
        return '📄';
      case 'csv':
        return '📊';
      default:
        return '📄';
    }
  };

  if (loading) {
    return (
      <BrandedCard
        title="Validation Reports"
        subtitle="Loading reports..."
        showBranding={true}
        brandingPosition="corner"
        className={className}
      >
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent" />
            <span className="text-gray-600">Loading validation reports...</span>
          </div>
        </div>
      </BrandedCard>
    );
  }

  return (
    <div className={clsx('space-y-6', className)}>
      {/* Report Generation */}
      <BrandedCard
        title="Generate Validation Report"
        subtitle={`${patternId ? `Pattern: ${patternId}` : ''} ${validationId ? `Validation: ${validationId}` : ''}`}
        showBranding={true}
        brandingPosition="corner"
        variant="bordered"
      >
        <div className="space-y-6">
          {/* Report Type Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Report Type
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {reportTypes.map((type) => (
                <div
                  key={type.value}
                  className={clsx(
                    'p-4 border-2 rounded-lg cursor-pointer transition-colors',
                    selectedReportType === type.value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  )}
                  onClick={() => setSelectedReportType(type.value)}
                >
                  <div className="flex items-center space-x-3">
                    <input
                      type="radio"
                      name="reportType"
                      value={type.value}
                      checked={selectedReportType === type.value}
                      onChange={() => setSelectedReportType(type.value)}
                      className="text-blue-600"
                    />
                    <div>
                      <div className="font-semibold text-gray-900">{type.label}</div>
                      <div className="text-sm text-gray-600">{type.description}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Format Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Output Format
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {formats.map((format) => (
                <div
                  key={format.value}
                  className={clsx(
                    'p-3 border-2 rounded-lg cursor-pointer transition-colors text-center',
                    selectedFormat === format.value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  )}
                  onClick={() => setSelectedFormat(format.value)}
                >
                  <div className="text-2xl mb-2">{getFormatIcon(format.value)}</div>
                  <div className="font-semibold text-gray-900">{format.label}</div>
                  <div className="text-xs text-gray-600">{format.description}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <div className="flex justify-center">
            <button
              onClick={handleGenerateReport}
              disabled={!selectedReportType || !selectedFormat}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>Generate Report</span>
            </button>
          </div>
        </div>
      </BrandedCard>

      {/* Existing Reports */}
      {reports.length > 0 && (
        <BrandedCard
          title="Generated Reports"
          subtitle={`${reports.length} report(s) available`}
          showBranding={true}
          brandingPosition="corner"
          variant="bordered"
        >
          <div className="space-y-4">
            {reports.map((report) => (
              <div key={report.report_id} className="p-4 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <span className={clsx(
                        'px-2 py-1 rounded-md text-xs font-medium',
                        getReportStatusColor(report.report_type)
                      )}>
                        {report.report_type.charAt(0).toUpperCase() + report.report_type.slice(1)}
                      </span>
                      <span className="text-sm text-gray-600">
                        {getFormatIcon(report.format)} {report.format.toUpperCase()}
                      </span>
                      <span className="text-sm text-gray-600">
                        Generated: {new Date(report.generated_at).toLocaleDateString()}
                      </span>
                    </div>
                    
                    <div className="mt-2">
                      <div className="font-medium text-gray-900">Report ID: {report.report_id}</div>
                      <div className="text-sm text-gray-600">
                        Version: {report.metadata.report_version} | 
                        Generated by: {report.metadata.generated_by}
                      </div>
                    </div>

                    {/* Report Summary */}
                    {report.report_content.executive_summary && (
                      <div className="mt-3 p-3 bg-gray-50 rounded-md">
                        <div className="text-sm text-gray-700">
                          <strong>Summary:</strong> {report.report_content.executive_summary}
                        </div>
                      </div>
                    )}

                    {/* Conclusions */}
                    {report.report_content.conclusions && report.report_content.conclusions.length > 0 && (
                      <div className="mt-3">
                        <div className="text-sm font-medium text-gray-700 mb-2">Key Conclusions:</div>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {report.report_content.conclusions.slice(0, 3).map((conclusion, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <span className="text-blue-500">•</span>
                              <span>{conclusion}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex items-center space-x-2 ml-4">
                    <button
                      onClick={() => onDownloadReport && onDownloadReport(report.report_id)}
                      className="p-2 text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
                      title="Download report"
                    >
                      <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </BrandedCard>
      )}

      {/* No Reports Message */}
      {reports.length === 0 && !loading && !error && (
        <BrandedCard
          title="No Reports Available"
          showBranding={true}
          brandingPosition="corner"
          variant="bordered"
        >
          <div className="text-center py-8">
            <div className="text-gray-400 mb-4">
              <svg className="h-16 w-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <p className="text-lg font-medium text-gray-900">No validation reports generated yet</p>
            <p className="text-sm text-gray-600 mt-1">
              Generate your first statistical validation report using the form above
            </p>
          </div>
        </BrandedCard>
      )}

      {/* Error State */}
      {error && (
        <BrandedCard
          title="Reports Error"
          showBranding={true}
          brandingPosition="corner"
          variant="bordered"
        >
          <div className="text-center py-8">
            <div className="text-red-500 mb-4">
              <svg className="h-16 w-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-lg font-medium text-gray-900">Failed to load reports</p>
            <p className="text-sm text-gray-600 mt-1">{error}</p>
          </div>
        </BrandedCard>
      )}
    </div>
  );
}

export default ValidationReports;