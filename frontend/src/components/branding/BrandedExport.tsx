/**
 * A2A World Platform - Branded Export Components
 * 
 * Export functionality components with consistent A2A World branding and watermarks.
 */

import React, { useState } from 'react';
import { clsx } from 'clsx';
import { Download, FileText, Image, FileSpreadsheet, X } from 'lucide-react';
import { Logo } from './Logo';

export interface BrandedExportProps {
  /**
   * Export data
   */
  data: any;
  /**
   * Export filename (without extension)
   */
  filename?: string;
  /**
   * Available export formats
   */
  formats?: ('csv' | 'json' | 'pdf' | 'png' | 'svg')[];
  /**
   * Export title for reports
   */
  title?: string;
  /**
   * Export description
   */
  description?: string;
  /**
   * Whether to include A2A branding in exports
   */
  includeBranding?: boolean;
  /**
   * Additional metadata to include
   */
  metadata?: Record<string, string>;
  /**
   * Export callback
   */
  onExport?: (format: string, data: any) => void;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function BrandedExport({
  data,
  filename = 'a2a-world-export',
  formats = ['csv', 'json', 'pdf'],
  title = 'A2A World Data Export',
  description,
  includeBranding = true,
  metadata = {},
  onExport,
  className
}: BrandedExportProps) {
  const [isExporting, setIsExporting] = useState<string | null>(null);

  const formatIcons = {
    csv: FileSpreadsheet,
    json: FileText,
    pdf: FileText,
    png: Image,
    svg: Image
  };

  const formatLabels = {
    csv: 'CSV Spreadsheet',
    json: 'JSON Data',
    pdf: 'PDF Report',
    png: 'PNG Image',
    svg: 'SVG Vector'
  };

  const handleExport = async (format: string) => {
    setIsExporting(format);
    try {
      const exportData = {
        ...data,
        ...(includeBranding && {
          _metadata: {
            exportedBy: 'A2A World Platform',
            exportedAt: new Date().toISOString(),
            title,
            description,
            ...metadata
          }
        })
      };

      if (onExport) {
        await onExport(format, exportData);
      } else {
        // Default export functionality
        await defaultExport(format, exportData);
      }
    } finally {
      setIsExporting(null);
    }
  };

  const defaultExport = async (format: string, exportData: any) => {
    const timestamp = new Date().toISOString().split('T')[0];
    const fullFilename = `${filename}-${timestamp}.${format}`;

    switch (format) {
      case 'csv':
        exportCSV(exportData, fullFilename);
        break;
      case 'json':
        exportJSON(exportData, fullFilename);
        break;
      case 'pdf':
        // Would integrate with PDF library
        console.log('PDF export not implemented yet');
        break;
      default:
        console.warn(`Export format ${format} not supported`);
    }
  };

  const exportCSV = (data: any, filename: string) => {
    // Simple CSV export implementation
    const csvContent = convertToCSV(data);
    downloadFile(csvContent, filename, 'text/csv');
  };

  const exportJSON = (data: any, filename: string) => {
    const jsonContent = JSON.stringify(data, null, 2);
    downloadFile(jsonContent, filename, 'application/json');
  };

  const convertToCSV = (data: any): string => {
    if (Array.isArray(data)) {
      if (data.length === 0) return '';
      
      const headers = Object.keys(data[0]);
      const csvRows = [
        headers.join(','),
        ...data.map(row => 
          headers.map(header => 
            JSON.stringify(row[header] || '')
          ).join(',')
        )
      ];
      
      return csvRows.join('\n');
    }
    
    return JSON.stringify(data);
  };

  const downloadFile = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={clsx('space-y-4', className)}>
      {includeBranding && (
        <div className="flex items-center space-x-2 text-sm text-gray-600">
          <Logo variant="icon" size="xs" />
          <span>A2A World Data Export</span>
        </div>
      )}
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {formats.map((format) => {
          const Icon = formatIcons[format];
          const isCurrentlyExporting = isExporting === format;
          
          return (
            <button
              key={format}
              onClick={() => handleExport(format)}
              disabled={isCurrentlyExporting}
              className={clsx(
                'flex items-center justify-center space-x-2 p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors',
                isCurrentlyExporting && 'bg-gray-100 cursor-not-allowed'
              )}
            >
              {isCurrentlyExporting ? (
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent" />
              ) : (
                <Icon className="h-4 w-4 text-gray-600" />
              )}
              <span className="text-sm font-medium text-gray-700">
                {formatLabels[format]}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export interface ExportButtonProps {
  /**
   * Export data
   */
  data: any;
  /**
   * Export format
   */
  format: 'csv' | 'json' | 'pdf' | 'png' | 'svg';
  /**
   * Button text
   */
  children?: React.ReactNode;
  /**
   * Export filename
   */
  filename?: string;
  /**
   * Button variant
   */
  variant?: 'primary' | 'secondary' | 'outline';
  /**
   * Button size
   */
  size?: 'sm' | 'md' | 'lg';
  /**
   * Export callback
   */
  onExport?: (format: string, data: any) => void;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export function ExportButton({
  data,
  format,
  children,
  filename = 'a2a-world-export',
  variant = 'outline',
  size = 'md',
  onExport,
  className
}: ExportButtonProps) {
  const [isExporting, setIsExporting] = useState(false);

  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
    outline: 'border border-gray-300 text-gray-700 hover:bg-gray-50'
  };

  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-2 text-sm',
    lg: 'px-4 py-2 text-base'
  };

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const exportData = {
        ...data,
        _metadata: {
          exportedBy: 'A2A World Platform',
          exportedAt: new Date().toISOString()
        }
      };

      if (onExport) {
        await onExport(format, exportData);
      }
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={isExporting}
      className={clsx(
        'inline-flex items-center space-x-2 font-medium rounded-md transition-colors',
        variantClasses[variant],
        sizeClasses[size],
        isExporting && 'opacity-50 cursor-not-allowed',
        className
      )}
    >
      {isExporting ? (
        <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent" />
      ) : (
        <Download className="h-4 w-4" />
      )}
      <span>{children || `Export ${format.toUpperCase()}`}</span>
    </button>
  );
}

export interface ExportModalProps {
  /**
   * Whether modal is open
   */
  isOpen: boolean;
  /**
   * Close modal callback
   */
  onClose: () => void;
  /**
   * Export data
   */
  data: any;
  /**
   * Export title
   */
  title?: string;
  /**
   * Export description
   */
  description?: string;
  /**
   * Available formats
   */
  formats?: ('csv' | 'json' | 'pdf' | 'png' | 'svg')[];
  /**
   * Export callback
   */
  onExport?: (format: string, data: any) => void;
}

export function ExportModal({
  isOpen,
  onClose,
  data,
  title = 'Export Data',
  description,
  formats = ['csv', 'json', 'pdf'],
  onExport
}: ExportModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />

        <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
          <div className="sm:flex sm:items-start">
            <div className="flex-1">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Logo variant="icon" size="sm" />
                  <h3 className="text-lg font-medium text-gray-900">
                    {title}
                  </h3>
                </div>
                <button
                  onClick={onClose}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              
              {description && (
                <p className="text-sm text-gray-600 mb-4">
                  {description}
                </p>
              )}

              <BrandedExport
                data={data}
                formats={formats}
                title={title}
                description={description}
                onExport={onExport}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BrandedExport;