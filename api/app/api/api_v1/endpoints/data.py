"""
A2A World Platform - Enhanced Data Management Endpoints

Comprehensive endpoints for data ingestion, processing, and retrieval with
file upload, progress tracking, and quality reporting capabilities.
"""

import os
import uuid
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, validator, Field
from typing import Union
import aiofiles
import io
import csv

try:
    from sqlalchemy.orm import Session
    from database.models.datasets import Dataset
    from database.models.geospatial import GeospatialFeature
    from database.connection import get_database_session
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Import our processors for file validation
try:
    from agents.parsers.data_processors import (
        KMLProcessor, GeoJSONProcessor, CSVProcessor,
        GeometryValidator, QualityChecker
    )
    PROCESSORS_AVAILABLE = True
except ImportError:
    PROCESSORS_AVAILABLE = False

router = APIRouter()

# In-memory storage for upload progress (in production, use Redis or database)
upload_progress = {}
processing_tasks = {}

# Pydantic models for request/response
class FileValidationRequest(BaseModel):
    file_path: str
    file_type: Optional[str] = None

class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    file_size: int
    file_type: str
    status: str
    message: str

class ProcessingStatus(BaseModel):
    upload_id: str
    status: str
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DataSearchRequest(BaseModel):
    """Advanced data search parameters"""
    query: Optional[str] = Field(None, description="Text search in name and description")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [min_lon, min_lat, max_lon, max_lat]")
    center_point: Optional[List[float]] = Field(None, description="Center point [lon, lat]")
    radius_km: Optional[float] = Field(None, description="Search radius in kilometers")
    feature_types: Optional[List[str]] = Field(None, description="Filter by feature types")
    date_from: Optional[str] = Field(None, description="Filter by creation date (ISO format)")
    date_to: Optional[str] = Field(None, description="Filter by creation date (ISO format)")
    dataset_ids: Optional[List[str]] = Field(None, description="Filter by specific dataset IDs")
    properties_filter: Optional[Dict[str, Any]] = Field(None, description="Filter by feature properties")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")
    sort_by: Optional[str] = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")
    
    @validator('bbox')
    def validate_bbox(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError('bbox must contain exactly 4 values: [min_lon, min_lat, max_lon, max_lat]')
        return v
    
    @validator('center_point')
    def validate_center_point(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError('center_point must contain exactly 2 values: [lon, lat]')
        return v

class DataExportRequest(BaseModel):
    """Data export configuration"""
    format: str = Field("geojson", description="Export format (geojson, kml, csv)")
    dataset_ids: Optional[List[str]] = Field(None, description="Specific datasets to export")
    search_params: Optional[DataSearchRequest] = Field(None, description="Search parameters for filtering")
    include_properties: bool = Field(True, description="Include feature properties")
    include_style: bool = Field(False, description="Include styling information")
    coordinate_system: str = Field("EPSG:4326", description="Output coordinate system")

class DataStatisticsResponse(BaseModel):
    """Dataset statistics response"""
    total_datasets: int
    total_features: int
    file_type_distribution: Dict[str, int]
    feature_type_distribution: Dict[str, int]
    processing_status_distribution: Dict[str, int]
    spatial_extent: Optional[Dict[str, float]] = None
    recent_uploads: int
    quality_metrics: Dict[str, float]
    storage_size_mb: float

# Helper functions
def get_db() -> Session:
    """Get database session dependency."""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    return get_database_session()

def detect_file_format(filename: str) -> str:
    """Detect file format from filename."""
    ext = Path(filename).suffix.lower()
    format_map = {
        '.kml': 'kml',
        '.kmz': 'kmz',
        '.geojson': 'geojson',
        '.json': 'geojson',
        '.csv': 'csv',
        '.zip': 'zip'
    }
    return format_map.get(ext, 'unknown')

async def process_uploaded_file(upload_id: str, file_path: str, filename: str):
    """Background task to process uploaded file."""
    if not PROCESSORS_AVAILABLE:
        upload_progress[upload_id] = {
            'status': 'error',
            'error': 'File processors not available',
            'progress': {'stage': 'error', 'progress': 0}
        }
        return

    try:
        # Update status to processing
        upload_progress[upload_id] = {
            'status': 'processing',
            'progress': {'stage': 'starting', 'progress': 0}
        }

        # Detect file format
        file_format = detect_file_format(filename)
        
        # Create appropriate processor
        processor = None
        if file_format in ['kml', 'kmz']:
            processor = KMLProcessor()
        elif file_format == 'geojson':
            processor = GeoJSONProcessor()
        elif file_format == 'csv':
            processor = CSVProcessor()
        else:
            upload_progress[upload_id] = {
                'status': 'error',
                'error': f'Unsupported file format: {file_format}',
                'progress': {'stage': 'error', 'progress': 0}
            }
            return

        # Process file with progress callback
        def progress_callback(progress_data):
            upload_progress[upload_id]['progress'] = progress_data

        # Process the file
        if file_format in ['kml', 'kmz']:
            result = processor.process_file(
                file_path,
                validate_geometry=True,
                generate_quality_report=True
            )
        elif file_format == 'geojson':
            result = processor.process_file(
                file_path,
                validate_geometry=True,
                generate_quality_report=True
            )
        elif file_format == 'csv':
            result = processor.process_file(
                file_path,
                validate_geometry=True,
                generate_quality_report=True
            )

        if result.success:
            # Store in database if available
            db_result = None
            if DATABASE_AVAILABLE:
                try:
                    db_result = await store_processing_result(upload_id, result, filename)
                except Exception as e:
                    print(f"Database storage failed: {e}")

            # Update final status
            upload_progress[upload_id] = {
                'status': 'completed',
                'progress': {'stage': 'completed', 'progress': 100},
                'result': {
                    'success': True,
                    'features_count': len(result.features),
                    'file_format': file_format,
                    'quality_score': result.quality_report.metrics.overall_quality_score if result.quality_report else None,
                    'database_stored': db_result is not None,
                    'dataset_id': db_result.get('dataset_id') if db_result else None
                }
            }
        else:
            upload_progress[upload_id] = {
                'status': 'error',
                'error': '; '.join(result.errors),
                'progress': {'stage': 'error', 'progress': 0}
            }

    except Exception as e:
        upload_progress[upload_id] = {
            'status': 'error',
            'error': str(e),
            'progress': {'stage': 'error', 'progress': 0}
        }

async def store_processing_result(upload_id: str, result, filename: str) -> Dict[str, Any]:
    """Store processing result in database."""
    with get_database_session() as session:
        # Create dataset record
        dataset = Dataset(
            name=f"Uploaded: {filename}",
            description=f"Dataset processed from uploaded file {filename}",
            file_type=detect_file_format(filename),
            status="completed",
            metadata={
                'upload_id': upload_id,
                'features_count': len(result.features),
                'processed_at': datetime.utcnow().isoformat(),
                'quality_score': result.quality_report.metrics.overall_quality_score if result.quality_report else None
            }
        )
        session.add(dataset)
        session.flush()

        # Store features
        stored_count = 0
        for feature in result.features:
            try:
                geospatial_feature = GeospatialFeature(
                    dataset_id=dataset.id,
                    name=feature.get("name"),
                    description=feature.get("description"),
                    geometry=f"SRID=4326;POINT({feature.get('longitude', 0)} {feature.get('latitude', 0)})",
                    properties=feature.get("properties", {}),
                    feature_type=feature.get("feature_type", "point")
                )
                session.add(geospatial_feature)
                stored_count += 1
            except Exception as e:
                print(f"Error storing feature: {e}")

        session.commit()
        return {
            'dataset_id': str(dataset.id),
            'features_stored': stored_count
        }

# API Endpoints

@router.post("/upload", response_model=UploadResponse)
async def upload_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> UploadResponse:
    """
    Upload and process geospatial data file.
    
    Supports KML, KMZ, GeoJSON, CSV, and ZIP formats.
    Returns upload ID for tracking processing status.
    """
    try:
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Detect and validate file format
        file_format = detect_file_format(file.filename)
        if file_format == 'unknown':
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: KML, KMZ, GeoJSON, CSV, ZIP"
            )
        
        # Check file size (limit to 100MB)
        content = await file.read()
        file_size = len(content)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")
        
        # Save file to temporary location
        temp_dir = Path(tempfile.gettempdir()) / "a2a_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{upload_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Initialize progress tracking
        upload_progress[upload_id] = {
            'status': 'queued',
            'progress': {'stage': 'queued', 'progress': 0},
            'filename': file.filename,
            'file_size': file_size,
            'file_format': file_format,
            'upload_time': datetime.utcnow().isoformat()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_uploaded_file,
            upload_id,
            str(file_path),
            file.filename
        )
        
        return UploadResponse(
            upload_id=upload_id,
            filename=file.filename,
            file_size=file_size,
            file_type=file_format,
            status="queued",
            message="File uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/upload/{upload_id}/status", response_model=ProcessingStatus)
async def get_upload_status(upload_id: str) -> ProcessingStatus:
    """
    Get processing status and progress for uploaded file.
    
    Returns current processing stage, progress percentage, and results when complete.
    """
    if upload_id not in upload_progress:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    progress_data = upload_progress[upload_id]
    
    return ProcessingStatus(
        upload_id=upload_id,
        status=progress_data['status'],
        progress=progress_data['progress'],
        result=progress_data.get('result'),
        error=progress_data.get('error')
    )

@router.get("/")
async def list_datasets(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    file_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    List available datasets with metadata and statistics.
    
    Supports pagination and filtering by file type.
    """
    try:
        query = db.query(Dataset)
        
        # Apply file type filter
        if file_type:
            query = query.filter(Dataset.file_type == file_type)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        datasets = query.offset(offset).limit(limit).all()
        
        # Format dataset information
        dataset_list = []
        for dataset in datasets:
            # Get feature count
            feature_count = db.query(GeospatialFeature).filter(
                GeospatialFeature.dataset_id == dataset.id
            ).count()
            
            dataset_info = {
                "id": str(dataset.id),
                "name": dataset.name,
                "description": dataset.description,
                "file_type": dataset.file_type,
                "status": dataset.status,
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
                "feature_count": feature_count,
                "metadata": dataset.metadata or {}
            }
            dataset_list.append(dataset_info)
        
        return {
            "datasets": dataset_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
            "supported_types": ["kml", "kmz", "geojson", "csv", "zip"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve datasets: {str(e)}")

@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    include_features: bool = Query(False),
    feature_limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific dataset.
    
    Optionally include sample features data.
    """
    try:
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get feature statistics
        features_query = db.query(GeospatialFeature).filter(
            GeospatialFeature.dataset_id == dataset_id
        )
        feature_count = features_query.count()
        
        dataset_info = {
            "id": str(dataset.id),
            "name": dataset.name,
            "description": dataset.description,
            "file_type": dataset.file_type,
            "file_size": dataset.file_size,
            "status": dataset.status,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
            "feature_count": feature_count,
            "metadata": dataset.metadata or {},
            "processing_log": dataset.processing_log
        }
        
        # Include sample features if requested
        if include_features and feature_count > 0:
            features = features_query.limit(feature_limit).all()
            dataset_info["features"] = [
                {
                    "id": str(feature.id),
                    "name": feature.name,
                    "description": feature.description,
                    "feature_type": feature.feature_type,
                    "properties": feature.properties or {},
                    "created_at": feature.created_at.isoformat() if feature.created_at else None
                }
                for feature in features
            ]
            dataset_info["features_included"] = len(features)
            dataset_info["features_truncated"] = feature_count > feature_limit
        
        return dataset_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset: {str(e)}")

@router.post("/validate")
async def validate_file(request: FileValidationRequest) -> Dict[str, Any]:
    """
    Validate a geospatial file without full processing.
    
    Returns validation results including structure check, geometry validation,
    and quality assessment preview.
    """
    if not PROCESSORS_AVAILABLE:
        raise HTTPException(status_code=503, detail="File processors not available")
    
    try:
        file_path = Path(request.file_path)
        
        if not file_path.exists():
            raise HTTPException(status_code=400, detail="File not found")
        
        # Detect file format
        file_format = request.file_type or detect_file_format(str(file_path))
        
        if file_format == 'unknown':
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create validator
        geometry_validator = GeometryValidator()
        
        # Quick validation based on file type
        if file_format == 'geojson':
            processor = GeoJSONProcessor()
            
            # Read and validate GeoJSON structure
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                
                structure_validation = processor.validate_geojson_structure(data)
                
                return {
                    "valid": structure_validation['valid'],
                    "file_format": file_format,
                    "file_size": file_path.stat().st_size,
                    "structure_validation": structure_validation,
                    "recommendation": "File can be processed" if structure_validation['valid'] else "Fix structure issues before processing"
                }
            except json.JSONDecodeError as e:
                return {
                    "valid": False,
                    "file_format": file_format,
                    "error": f"Invalid JSON: {str(e)}"
                }
        
        elif file_format == 'csv':
            processor = CSVProcessor()
            
            # Preview CSV structure
            preview = processor.preview_file(str(file_path))
            
            if 'error' in preview:
                return {
                    "valid": False,
                    "file_format": file_format,
                    "error": preview['error']
                }
            
            # Check if required columns were detected
            mapping = preview['detected_mapping']
            has_coords = mapping['latitude'] and mapping['longitude']
            
            return {
                "valid": has_coords,
                "file_format": file_format,
                "file_size": preview['file_size'],
                "preview": {
                    "headers": preview['headers'],
                    "sample_data": preview['sample_data'][:3],  # First 3 rows
                    "detected_columns": mapping,
                    "total_rows": preview['total_rows']
                },
                "recommendation": "File can be processed" if has_coords else "Could not detect coordinate columns"
            }
        
        elif file_format in ['kml', 'kmz']:
            # Basic KML validation
            try:
                processor = KMLProcessor()
                # For validation, we'll do a quick parse without full processing
                import xml.etree.ElementTree as ET
                
                if file_format == 'kmz':
                    import zipfile
                    with zipfile.ZipFile(file_path, 'r') as kmz:
                        kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
                        if not kml_files:
                            return {
                                "valid": False,
                                "file_format": file_format,
                                "error": "No KML files found in KMZ archive"
                            }
                        content = kmz.read(kml_files[0]).decode('utf-8')
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Try to parse XML
                root = ET.fromstring(content)
                
                # Count placemarks
                namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
                placemarks = root.findall('.//kml:Placemark', namespaces)
                
                return {
                    "valid": True,
                    "file_format": file_format,
                    "file_size": file_path.stat().st_size,
                    "preview": {
                        "placemark_count": len(placemarks),
                        "has_folders": len(root.findall('.//kml:Folder', namespaces)) > 0
                    },
                    "recommendation": f"File contains {len(placemarks)} placemarks and can be processed"
                }
                
            except ET.ParseError as e:
                return {
                    "valid": False,
                    "file_format": file_format,
                    "error": f"Invalid XML: {str(e)}"
                }
        
        return {
            "valid": True,
            "file_format": file_format,
            "file_size": file_path.stat().st_size,
            "recommendation": "File format supported but detailed validation not implemented"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete a dataset and all associated features.
    """
    try:
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Delete associated features (cascade should handle this, but let's be explicit)
        feature_count = db.query(GeospatialFeature).filter(
            GeospatialFeature.dataset_id == dataset_id
        ).count()
        
        db.query(GeospatialFeature).filter(
            GeospatialFeature.dataset_id == dataset_id
        ).delete()
        
        # Delete dataset
        db.delete(dataset)
        db.commit()
        
        return {
            "success": True,
            "message": f"Deleted dataset '{dataset.name}' and {feature_count} associated features",
            "dataset_id": dataset_id,
            "features_deleted": feature_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

@router.get("/stats/summary")
async def get_data_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get summary statistics about all datasets and processing activity.
    """
    try:
        # Dataset statistics
        total_datasets = db.query(Dataset).count()
        completed_datasets = db.query(Dataset).filter(Dataset.status == "completed").count()
        failed_datasets = db.query(Dataset).filter(Dataset.status == "failed").count()
        
        # Feature statistics
        total_features = db.query(GeospatialFeature).count()
        
        # File type distribution
        from sqlalchemy import func
        file_type_stats = db.query(
            Dataset.file_type,
            func.count(Dataset.id)
        ).group_by(Dataset.file_type).all()
        
        file_type_distribution = {file_type: count for file_type, count in file_type_stats}
        
        # Recent activity (last 7 days)
        from datetime import timedelta
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_datasets = db.query(Dataset).filter(
            Dataset.created_at >= recent_date
        ).count()
        
        return {
            "datasets": {
                "total": total_datasets,
                "completed": completed_datasets,
                "failed": failed_datasets,
                "success_rate": (completed_datasets / total_datasets * 100) if total_datasets > 0 else 0
            },
            "features": {
                "total": total_features,
                "average_per_dataset": (total_features / completed_datasets) if completed_datasets > 0 else 0
            },
            "file_types": file_type_distribution,
            "recent_activity": {
                "datasets_last_7_days": recent_datasets
            },
            "processing_capabilities": {
                "processors_available": PROCESSORS_AVAILABLE,
                "database_available": DATABASE_AVAILABLE,
                "supported_formats": ["kml", "kmz", "geojson", "csv", "zip"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@router.post("/search", response_model=Dict[str, Any])
async def search_data(
    search_request: DataSearchRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Advanced data search with geospatial filtering and complex criteria.
    Supports text search, bounding box, radius search, and property filtering.
    """
    try:
        from sqlalchemy import and_, or_, func, text
        from geoalchemy2.functions import ST_DWithin, ST_MakePoint, ST_Contains, ST_MakeEnvelope
        
        # Build base query for features
        query = db.query(GeospatialFeature)
        
        # Apply text search
        if search_request.query:
            query = query.filter(
                or_(
                    GeospatialFeature.name.ilike(f"%{search_request.query}%"),
                    GeospatialFeature.description.ilike(f"%{search_request.query}%")
                )
            )
        
        # Apply bounding box filter
        if search_request.bbox:
            min_lon, min_lat, max_lon, max_lat = search_request.bbox
            bbox_geom = ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
            query = query.filter(ST_Contains(bbox_geom, GeospatialFeature.geometry))
        
        # Apply radius search
        elif search_request.center_point and search_request.radius_km:
            lon, lat = search_request.center_point
            center_point = ST_MakePoint(lon, lat)
            radius_meters = search_request.radius_km * 1000
            query = query.filter(
                ST_DWithin(GeospatialFeature.geometry, center_point, radius_meters)
            )
        
        # Apply feature type filter
        if search_request.feature_types:
            query = query.filter(GeospatialFeature.feature_type.in_(search_request.feature_types))
        
        # Apply dataset filter
        if search_request.dataset_ids:
            query = query.filter(GeospatialFeature.dataset_id.in_(search_request.dataset_ids))
        
        # Apply date range filter
        if search_request.date_from or search_request.date_to:
            if search_request.date_from:
                from datetime import datetime
                date_from = datetime.fromisoformat(search_request.date_from.replace('Z', '+00:00'))
                query = query.filter(GeospatialFeature.created_at >= date_from)
            if search_request.date_to:
                date_to = datetime.fromisoformat(search_request.date_to.replace('Z', '+00:00'))
                query = query.filter(GeospatialFeature.created_at <= date_to)
        
        # Apply properties filter
        if search_request.properties_filter:
            for key, value in search_request.properties_filter.items():
                if isinstance(value, list):
                    query = query.filter(GeospatialFeature.properties[key].astext.in_(value))
                else:
                    query = query.filter(GeospatialFeature.properties[key].astext == str(value))
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply sorting
        if search_request.sort_by == "distance" and search_request.center_point:
            lon, lat = search_request.center_point
            center_point = ST_MakePoint(lon, lat)
            if search_request.sort_order.lower() == "asc":
                query = query.order_by(
                    func.ST_Distance(GeospatialFeature.geometry, center_point)
                )
            else:
                query = query.order_by(
                    func.ST_Distance(GeospatialFeature.geometry, center_point).desc()
                )
        else:
            sort_field = getattr(GeospatialFeature, search_request.sort_by, GeospatialFeature.created_at)
            if search_request.sort_order.lower() == "desc":
                query = query.order_by(sort_field.desc())
            else:
                query = query.order_by(sort_field.asc())
        
        # Apply pagination
        features = query.offset(search_request.offset).limit(search_request.limit).all()
        
        # Format results
        results = []
        for feature in features:
            feature_data = {
                "id": str(feature.id),
                "name": feature.name,
                "description": feature.description,
                "feature_type": feature.feature_type,
                "dataset_id": str(feature.dataset_id),
                "properties": feature.properties or {},
                "created_at": feature.created_at.isoformat() if feature.created_at else None
            }
            
            # Add distance if center point provided
            if search_request.center_point:
                try:
                    lon, lat = search_request.center_point
                    distance_query = db.execute(
                        text("SELECT ST_Distance(ST_Transform(:geom, 3857), ST_Transform(ST_MakePoint(:lon, :lat), 3857))"),
                        {"geom": feature.geometry, "lon": lon, "lat": lat}
                    ).scalar()
                    feature_data["distance_meters"] = round(distance_query or 0, 2)
                except Exception:
                    pass
            
            results.append(feature_data)
        
        return {
            "features": results,
            "total": total_count,
            "limit": search_request.limit,
            "offset": search_request.offset,
            "has_more": search_request.offset + search_request.limit < total_count,
            "search_parameters": search_request.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/export")
async def export_data(
    format: str = Query("geojson", description="Export format"),
    dataset_ids: Optional[str] = Query(None, description="Comma-separated dataset IDs"),
    bbox: Optional[str] = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    feature_types: Optional[str] = Query(None, description="Comma-separated feature types"),
    limit: int = Query(10000, ge=1, le=50000, description="Maximum features to export"),
    db: Session = Depends(get_db)
):
    """
    Export data in various formats (GeoJSON, KML, CSV).
    Supports filtering and format conversion.
    """
    try:
        from sqlalchemy import and_, or_
        from geoalchemy2.functions import ST_Contains, ST_MakeEnvelope
        import json
        
        # Validate format
        if format.lower() not in ["geojson", "kml", "csv"]:
            raise HTTPException(status_code=400, detail="Unsupported format. Use: geojson, kml, csv")
        
        # Build query
        query = db.query(GeospatialFeature)
        
        # Apply filters
        if dataset_ids:
            dataset_id_list = [id.strip() for id in dataset_ids.split(",")]
            query = query.filter(GeospatialFeature.dataset_id.in_(dataset_id_list))
        
        if bbox:
            try:
                min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
                bbox_geom = ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
                query = query.filter(ST_Contains(bbox_geom, GeospatialFeature.geometry))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        if feature_types:
            type_list = [t.strip() for t in feature_types.split(",")]
            query = query.filter(GeospatialFeature.feature_type.in_(type_list))
        
        # Get features
        features = query.limit(limit).all()
        
        if not features:
            raise HTTPException(status_code=404, detail="No features found matching criteria")
        
        # Generate export based on format
        if format.lower() == "geojson":
            return await _export_geojson(features, db)
        elif format.lower() == "kml":
            return await _export_kml(features, db)
        elif format.lower() == "csv":
            return await _export_csv(features, db)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

async def _export_geojson(features: List, db: Session) -> StreamingResponse:
    """Export features as GeoJSON."""
    try:
        from sqlalchemy import text
        import json
        
        geojson_features = []
        for feature in features:
            # Get geometry as GeoJSON
            geom_result = db.execute(
                text("SELECT ST_AsGeoJSON(:geom)"),
                {"geom": feature.geometry}
            ).scalar()
            
            if geom_result:
                geometry = json.loads(geom_result)
            else:
                continue
            
            geojson_feature = {
                "type": "Feature",
                "id": str(feature.id),
                "geometry": geometry,
                "properties": {
                    "name": feature.name,
                    "description": feature.description,
                    "feature_type": feature.feature_type,
                    "dataset_id": str(feature.dataset_id),
                    "created_at": feature.created_at.isoformat() if feature.created_at else None,
                    **(feature.properties or {})
                }
            }
            geojson_features.append(geojson_feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": geojson_features
        }
        
        output = io.StringIO()
        json.dump(geojson_data, output, indent=2)
        output.seek(0)
        
        response = StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="application/geo+json",
            headers={"Content-Disposition": "attachment; filename=export.geojson"}
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GeoJSON export failed: {str(e)}")

async def _export_kml(features: List, db: Session) -> StreamingResponse:
    """Export features as KML."""
    try:
        from sqlalchemy import text
        import xml.etree.ElementTree as ET
        
        # Create KML structure
        kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(kml, "Document")
        ET.SubElement(document, "name").text = "A2A World Export"
        ET.SubElement(document, "description").text = f"Export of {len(features)} features"
        
        for feature in features:
            placemark = ET.SubElement(document, "Placemark")
            ET.SubElement(placemark, "name").text = feature.name or f"Feature {feature.id}"
            if feature.description:
                ET.SubElement(placemark, "description").text = feature.description
            
            # Get geometry as KML
            geom_result = db.execute(
                text("SELECT ST_AsKML(:geom)"),
                {"geom": feature.geometry}
            ).scalar()
            
            if geom_result:
                # Parse and add geometry
                try:
                    geom_element = ET.fromstring(geom_result)
                    placemark.append(geom_element)
                except ET.ParseError:
                    pass
            
            # Add extended data
            if feature.properties:
                extended_data = ET.SubElement(placemark, "ExtendedData")
                for key, value in feature.properties.items():
                    data = ET.SubElement(extended_data, "Data", name=str(key))
                    ET.SubElement(data, "value").text = str(value)
        
        # Convert to string
        ET.register_namespace("", "http://www.opengis.net/kml/2.2")
        kml_string = ET.tostring(kml, encoding='unicode', method='xml')
        
        response = StreamingResponse(
            io.BytesIO(kml_string.encode('utf-8')),
            media_type="application/vnd.google-earth.kml+xml",
            headers={"Content-Disposition": "attachment; filename=export.kml"}
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KML export failed: {str(e)}")

async def _export_csv(features: List, db: Session) -> StreamingResponse:
    """Export features as CSV."""
    try:
        from sqlalchemy import text
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = ["id", "name", "description", "feature_type", "dataset_id", "longitude", "latitude", "created_at"]
        
        # Get all unique property keys
        all_properties = set()
        for feature in features:
            if feature.properties:
                all_properties.update(feature.properties.keys())
        
        headers.extend(sorted(all_properties))
        writer.writerow(headers)
        
        # Write data rows
        for feature in features:
            # Get coordinates
            coord_result = db.execute(
                text("SELECT ST_X(ST_Centroid(:geom)), ST_Y(ST_Centroid(:geom))"),
                {"geom": feature.geometry}
            ).fetchone()
            
            longitude = coord_result[0] if coord_result else None
            latitude = coord_result[1] if coord_result else None
            
            row = [
                str(feature.id),
                feature.name or "",
                feature.description or "",
                feature.feature_type or "",
                str(feature.dataset_id),
                longitude,
                latitude,
                feature.created_at.isoformat() if feature.created_at else ""
            ]
            
            # Add properties
            for prop_key in sorted(all_properties):
                value = ""
                if feature.properties and prop_key in feature.properties:
                    value = str(feature.properties[prop_key])
                row.append(value)
            
            writer.writerow(row)
        
        output.seek(0)
        
        response = StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=export.csv"}
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

@router.get("/statistics", response_model=DataStatisticsResponse)
async def get_data_statistics(db: Session = Depends(get_db)) -> DataStatisticsResponse:
    """
    Get comprehensive dataset statistics and analytics.
    Includes spatial extent, quality metrics, and distribution analysis.
    """
    try:
        from sqlalchemy import func, text
        from datetime import datetime, timedelta
        
        # Basic counts
        total_datasets = db.query(Dataset).count()
        total_features = db.query(GeospatialFeature).count()
        
        # File type distribution
        file_type_stats = db.query(
            Dataset.file_type,
            func.count(Dataset.id)
        ).group_by(Dataset.file_type).all()
        file_type_distribution = {file_type: count for file_type, count in file_type_stats}
        
        # Feature type distribution
        feature_type_stats = db.query(
            GeospatialFeature.feature_type,
            func.count(GeospatialFeature.id)
        ).group_by(GeospatialFeature.feature_type).all()
        feature_type_distribution = {feature_type or "unknown": count for feature_type, count in feature_type_stats}
        
        # Processing status distribution
        status_stats = db.query(
            Dataset.status,
            func.count(Dataset.id)
        ).group_by(Dataset.status).all()
        processing_status_distribution = {status: count for status, count in status_stats}
        
        # Calculate spatial extent
        spatial_extent = None
        if total_features > 0:
            try:
                extent_result = db.execute(
                    text("""
                        SELECT 
                            ST_XMin(extent) as min_lon,
                            ST_YMin(extent) as min_lat,
                            ST_XMax(extent) as max_lon,
                            ST_YMax(extent) as max_lat
                        FROM (
                            SELECT ST_Extent(geometry) as extent 
                            FROM geospatial_features
                        ) as bounds
                    """)
                ).fetchone()
                
                if extent_result and all(x is not None for x in extent_result):
                    spatial_extent = {
                        "min_longitude": float(extent_result[0]),
                        "min_latitude": float(extent_result[1]),
                        "max_longitude": float(extent_result[2]),
                        "max_latitude": float(extent_result[3])
                    }
            except Exception as e:
                pass  # Skip spatial extent if calculation fails
        
        # Recent uploads (last 7 days)
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_uploads = db.query(Dataset).filter(
            Dataset.created_at >= recent_date
        ).count()
        
        # Quality metrics
        completed_datasets = db.query(Dataset).filter(Dataset.status == "completed").count()
        failed_datasets = db.query(Dataset).filter(Dataset.status == "failed").count()
        
        success_rate = (completed_datasets / total_datasets * 100) if total_datasets > 0 else 0
        failure_rate = (failed_datasets / total_datasets * 100) if total_datasets > 0 else 0
        
        quality_metrics = {
            "success_rate_percent": round(success_rate, 2),
            "failure_rate_percent": round(failure_rate, 2),
            "average_features_per_dataset": round(total_features / completed_datasets, 2) if completed_datasets > 0 else 0
        }
        
        # Estimate storage size
        storage_size_result = db.execute(
            text("SELECT COALESCE(SUM(file_size), 0) FROM datasets WHERE file_size IS NOT NULL")
        ).scalar()
        storage_size_mb = (storage_size_result or 0) / 1024 / 1024
        
        return DataStatisticsResponse(
            total_datasets=total_datasets,
            total_features=total_features,
            file_type_distribution=file_type_distribution,
            feature_type_distribution=feature_type_distribution,
            processing_status_distribution=processing_status_distribution,
            spatial_extent=spatial_extent,
            recent_uploads=recent_uploads,
            quality_metrics=quality_metrics,
            storage_size_mb=round(storage_size_mb, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate statistics: {str(e)}")