/**
 * A2A World Platform - Enhanced File Upload Component
 * 
 * Advanced file upload with drag-and-drop, validation, progress tracking, and quality reporting.
 */

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  File, 
  CheckCircle, 
  AlertTriangle, 
  X, 
  FileText,
  AlertCircle,
  Clock,
  Info
} from 'lucide-react';
import { Button } from './Button';
import { dataApi } from '@/utils/api';

interface FileUploadProps {
  onUploadComplete?: (result: any) => void;
  onUploadError?: (error: any) => void;
  acceptedFormats?: string[];
  maxFileSize?: number;
  className?: string;
}

interface UploadFile {
  file: File;
  id: string;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  result?: any;
  uploadId?: string;
}

export function FileUpload({
  onUploadComplete,
  onUploadError,
  acceptedFormats = ['.kml', '.geojson', '.csv', '.zip'],
  maxFileSize = 100 * 1024 * 1024, // 100MB
  className = ''
}: FileUploadProps) {
  const [uploads, setUploads] = useState<UploadFile[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);

  const updateUpload = useCallback((id: string, updates: Partial<UploadFile>) => {
    setUploads(prev => prev.map(upload => 
      upload.id === id ? { ...upload, ...updates } : upload
    ));
  }, []);

  const removeUpload = useCallback((id: string) => {
    setUploads(prev => prev.filter(upload => upload.id !== id));
  }, []);

  const processFile = useCallback(async (uploadFile: UploadFile) => {
    try {
      updateUpload(uploadFile.id, { status: 'uploading' });

      // Upload file with progress tracking
      const uploadResult = await dataApi.uploadDataFile(
        uploadFile.file,
        (progress) => {
          updateUpload(uploadFile.id, { progress });
        }
      );

      if (uploadResult.upload_id) {
        updateUpload(uploadFile.id, {
          uploadId: uploadResult.upload_id,
          status: 'processing',
          progress: 100
        });

        // Poll for processing status
        const pollInterval = setInterval(async () => {
          try {
            const statusResult = await dataApi.getUploadStatus(uploadResult.upload_id);
            
            updateUpload(uploadFile.id, {
              progress: statusResult.progress?.progress || 100
            });

            if (statusResult.status === 'completed') {
              clearInterval(pollInterval);
              updateUpload(uploadFile.id, {
                status: 'completed',
                result: statusResult.result
              });
              
              if (onUploadComplete) {
                onUploadComplete(statusResult.result);
              }
            } else if (statusResult.status === 'error') {
              clearInterval(pollInterval);
              updateUpload(uploadFile.id, {
                status: 'error',
                error: statusResult.error || 'Processing failed'
              });
              
              if (onUploadError) {
                onUploadError(statusResult.error);
              }
            }
          } catch (error) {
            clearInterval(pollInterval);
            updateUpload(uploadFile.id, {
              status: 'error',
              error: 'Failed to check processing status'
            });
          }
        }, 2000); // Poll every 2 seconds

      } else {
        throw new Error('No upload ID received');
      }

    } catch (error: any) {
      updateUpload(uploadFile.id, {
        status: 'error',
        error: error.message || 'Upload failed'
      });
      
      if (onUploadError) {
        onUploadError(error);
      }
    }
  }, [updateUpload, onUploadComplete, onUploadError]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newUploads = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      progress: 0,
      status: 'pending' as const
    }));

    setUploads(prev => [...prev, ...newUploads]);

    // Start processing each file
    newUploads.forEach(upload => {
      processFile(upload);
    });
  }, [processFile]);

  const { getRootProps, getInputProps, isDragActive: dropzoneActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.google-earth.kml+xml': ['.kml'],
      'application/geo+json': ['.geojson'],
      'text/csv': ['.csv'],
      'application/zip': ['.zip']
    },
    multiple: true,
    maxSize: maxFileSize,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
    onDropAccepted: () => setIsDragActive(false),
    onDropRejected: () => setIsDragActive(false)
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'processing':
        return <Clock className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'uploading':
        return <Upload className="h-5 w-5 text-blue-500" />;
      case 'error':
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
      default:
        return <File className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    const colors = {
      completed: 'text-green-600 bg-green-50 border-green-200',
      processing: 'text-blue-600 bg-blue-50 border-blue-200',
      uploading: 'text-blue-600 bg-blue-50 border-blue-200',
      error: 'text-red-600 bg-red-50 border-red-200',
      pending: 'text-gray-600 bg-gray-50 border-gray-200'
    };
    return colors[status as keyof typeof colors] || 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + ' KB';
    return Math.round(bytes / (1024 * 1024)) + ' MB';
  };

  const getFileTypeEmoji = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'kml': return '🗺️';
      case 'geojson': return '🌍';
      case 'csv': return '📊';
      case 'zip': return '📦';
      default: return '📄';
    }
  };

  return (
    <div className={className}>
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
          isDragActive || dropzoneActive
            ? 'border-primary-400 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        {isDragActive || dropzoneActive ? (
          <p className="text-lg text-primary-600">Drop the files here...</p>
        ) : (
          <div>
            <p className="text-lg text-gray-700 mb-2">
              Drag & drop files here, or click to select
            </p>
            <p className="text-sm text-gray-500 mb-4">
              Supported formats: {acceptedFormats.join(', ')}
            </p>
            <p className="text-xs text-gray-400">
              Maximum file size: {formatFileSize(maxFileSize)}
            </p>
          </div>
        )}
      </div>

      {/* Upload Progress */}
      {uploads.length > 0 && (
        <div className="mt-6 space-y-4">
          <h3 className="text-lg font-medium text-gray-900">Upload Progress</h3>
          
          {uploads.map((upload) => (
            <div key={upload.id} className="bg-white border rounded-lg p-4 shadow-sm">
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 text-2xl">
                  {getFileTypeEmoji(upload.file.name)}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-gray-900 truncate">
                      {upload.file.name}
                    </h4>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(upload.status)}
                      <button
                        onClick={() => removeUpload(upload.id)}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 mb-2">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(upload.status)}`}>
                      {upload.status}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatFileSize(upload.file.size)}
                    </span>
                  </div>

                  {/* Progress Bar */}
                  {(upload.status === 'uploading' || upload.status === 'processing') && (
                    <div className="mb-2">
                      <div className="flex justify-between text-xs text-gray-600 mb-1">
                        <span>
                          {upload.status === 'uploading' ? 'Uploading' : 'Processing'}
                        </span>
                        <span>{Math.round(upload.progress)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-primary-600 h-2 rounded-full transition-all duration-300" 
                          style={{ width: `${upload.progress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {upload.status === 'error' && upload.error && (
                    <div className="flex items-center space-x-2 mt-2 p-2 bg-red-50 border border-red-200 rounded">
                      <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
                      <span className="text-xs text-red-700">{upload.error}</span>
                    </div>
                  )}

                  {/* Success Result */}
                  {upload.status === 'completed' && upload.result && (
                    <div className="mt-2 p-3 bg-green-50 border border-green-200 rounded">
                      <div className="flex items-center space-x-2 mb-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-medium text-green-900">
                          Processing Complete
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                        <div>
                          <span className="text-gray-600">Features:</span>
                          <span className="font-medium ml-1">
                            {upload.result.features_count?.toLocaleString() || 'N/A'}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">Format:</span>
                          <span className="font-medium ml-1 uppercase">
                            {upload.result.file_format || 'Unknown'}
                          </span>
                        </div>
                        {upload.result.quality_score && (
                          <div>
                            <span className="text-gray-600">Quality:</span>
                            <span className="font-medium ml-1">
                              {Math.round(upload.result.quality_score)}%
                            </span>
                          </div>
                        )}
                        {upload.result.database_stored && (
                          <div>
                            <span className="text-gray-600">Stored:</span>
                            <span className="font-medium ml-1 text-green-600">Yes</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default FileUpload;