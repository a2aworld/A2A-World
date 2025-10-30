"""
A2A World Platform - Enhanced KML Parser Agent

Agent responsible for parsing KML, GeoJSON, and CSV files with advanced
data validation, database integration, and quality reporting.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import zipfile
import tempfile
import os
from dataclasses import asdict

try:
    import sqlalchemy as sa
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.exc import IntegrityError
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

from agents.core.base_agent import BaseAgent
from agents.core.config import ParserAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task

# Import our enhanced processors
from .data_processors import (
    KMLProcessor, GeoJSONProcessor, CSVProcessor, TextProcessor,
    GeometryValidator, QualityChecker
)

# Import database models if available
if DATABASE_AVAILABLE:
    from database.models.geospatial import GeospatialFeature, SacredSite
    from database.models.datasets import Dataset
    from database.connection import get_database_session


class KMLParserAgent(BaseAgent):
    """
    Enhanced agent that parses various geospatial file formats and stores data in database.
    
    Capabilities:
    - Parse KML, KMZ, GeoJSON, and CSV files
    - Advanced geometry validation and correction
    - Database integration with transaction management
    - Quality assessment and reporting
    - Progress tracking for large files
    - Batch processing with memory optimization
    - Support for ZIP archives containing multiple files
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[ParserAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="parser",
            config=config or ParserAgentConfig(),
            config_file=config_file
        )
        
        # Initialize processors
        self.kml_processor = KMLProcessor()
        self.geojson_processor = GeoJSONProcessor()
        self.csv_processor = CSVProcessor()
        self.text_processor = TextProcessor()
        self.geometry_validator = GeometryValidator()
        self.quality_checker = QualityChecker()
        
        # Processing statistics
        self.files_processed = 0
        self.features_extracted = 0
        self.database_inserts = 0
        self.processing_errors = 0
        self.quality_reports_generated = 0
        
        # Supported file formats
        self.supported_formats = {
            '.kml': 'kml',
            '.kmz': 'kmz',
            '.geojson': 'geojson',
            '.json': 'geojson',
            '.csv': 'csv',
            '.txt': 'text',
            '.md': 'text'
        }
        
        # Database session factory
        self.db_session_factory = None
        if DATABASE_AVAILABLE:
            try:
                self.db_session_factory = get_database_session
            except Exception as e:
                self.logger.warning(f"Database not available: {e}")
        
        self.logger.info(f"Enhanced KMLParserAgent {self.agent_id} initialized")
    
    async def process(self) -> None:
        """Main processing loop - handle parsing requests and file monitoring."""
        try:
            # Process any pending parsing requests
            await self._process_parsing_queue()
            
        except Exception as e:
            self.logger.error(f"Error in parsing process: {e}")
    
    async def agent_initialize(self) -> None:
        """Parser agent specific initialization."""
        try:
            # Verify parsing dependencies
            self._verify_parsing_dependencies()

            # Create data directories if needed
            self.config.data_path.mkdir(parents=True, exist_ok=True)
            self.config.temp_path.mkdir(parents=True, exist_ok=True)

            # Initialize text processor
            await self.text_processor.initialize()

            # Test database connection if available
            if self.db_session_factory:
                try:
                    with self.db_session_factory() as session:
                        session.execute(sa.text("SELECT 1"))
                    self.logger.info("Database connection verified")
                except Exception as e:
                    self.logger.warning(f"Database connection failed: {e}")
                    self.db_session_factory = None

            self.logger.info("Enhanced KMLParserAgent initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced KMLParserAgent: {e}")
            raise
    
    async def setup_subscriptions(self) -> None:
        """Setup parser-specific message subscriptions."""
        if not self.messaging:
            return
        
        # Subscribe to file processing requests
        parse_sub_id = await self.nats_client.subscribe(
            "agents.parsers.request",
            self._handle_parse_request,
            queue_group="enhanced-parser-workers"
        )
        self.subscription_ids.append(parse_sub_id)
        
        # Subscribe to file upload notifications
        upload_sub_id = await self.nats_client.subscribe(
            "agents.files.uploaded",
            self._handle_file_upload,
            queue_group="enhanced-parser-uploads"
        )
        self.subscription_ids.append(upload_sub_id)
        
        # Subscribe to batch processing requests
        batch_sub_id = await self.nats_client.subscribe(
            "agents.parsers.batch",
            self._handle_batch_request,
            queue_group="enhanced-parser-batch"
        )
        self.subscription_ids.append(batch_sub_id)
    
    async def handle_task(self, task: Task) -> None:
        """Handle parsing task processing."""
        self.logger.info(f"Processing task {task.task_id}: {task.task_type}")
        
        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)
            
            result = None
            
            # Dispatch to appropriate handler
            if task.task_type == "parse_file":
                result = await self._parse_file_task(task)
            elif task.task_type == "parse_batch":
                result = await self._parse_batch_task(task)
            elif task.task_type == "validate_dataset":
                result = await self._validate_dataset_task(task)
            elif task.task_type == "store_features":
                result = await self._store_features_task(task)
            else:
                raise ValueError(f"Unknown parsing task type: {task.task_type}")
            
            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)
            
            self.processed_tasks += 1
            self._update_statistics_from_result(result)
            
            self.logger.info(f"Completed task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            
            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)
            
            self.failed_tasks += 1
            self.processing_errors += 1
        
        finally:
            self.current_tasks.discard(task.task_id)
    
    async def _parse_file_task(self, task: Task) -> Dict[str, Any]:
        """Handle single file parsing task."""
        file_path = task.parameters.get("file_path")
        if not file_path:
            raise ValueError("file_path parameter is required")
        
        # Optional parameters
        store_in_db = task.parameters.get("store_in_db", True)
        generate_report = task.parameters.get("generate_report", True)
        dataset_name = task.parameters.get("dataset_name")
        
        return await self.parse_file(
            file_path=file_path,
            store_in_database=store_in_db,
            generate_quality_report=generate_report,
            dataset_name=dataset_name,
            progress_callback=lambda p: self._send_progress_update(task.task_id, p)
        )
    
    async def _parse_batch_task(self, task: Task) -> Dict[str, Any]:
        """Handle batch file parsing task."""
        file_paths = task.parameters.get("file_paths", [])
        if not file_paths:
            raise ValueError("file_paths parameter is required")
        
        return await self.parse_batch(
            file_paths=file_paths,
            store_in_database=task.parameters.get("store_in_db", True),
            progress_callback=lambda p: self._send_progress_update(task.task_id, p)
        )
    
    async def _validate_dataset_task(self, task: Task) -> Dict[str, Any]:
        """Handle dataset validation task."""
        dataset_id = task.parameters.get("dataset_id")
        if not dataset_id:
            raise ValueError("dataset_id parameter is required")
        
        return await self.validate_dataset(dataset_id)
    
    async def _store_features_task(self, task: Task) -> Dict[str, Any]:
        """Handle feature storage task."""
        features = task.input_data.get("features", [])
        dataset_name = task.parameters.get("dataset_name", "Unknown Dataset")
        
        if not features:
            raise ValueError("features data is required")
        
        return await self.store_features_in_database(features, dataset_name)
    
    async def parse_file(
        self,
        file_path: str,
        store_in_database: bool = True,
        generate_quality_report: bool = True,
        dataset_name: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Parse a single geospatial file.
        
        Args:
            file_path: Path to file to parse
            store_in_database: Whether to store results in database
            generate_quality_report: Whether to generate quality report
            dataset_name: Name for the dataset
            progress_callback: Callback for progress updates
            
        Returns:
            Parsing result with features, metadata, and quality report
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.logger.info(f"Parsing file: {file_path}")
            
            if progress_callback:
                await progress_callback({"stage": "starting", "progress": 0})
            
            # Detect file format
            file_format = self._detect_file_format(file_path_obj)
            if not file_format:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
            
            if progress_callback:
                await progress_callback({"stage": "parsing", "progress": 10})
            
            # Parse based on format
            if file_format in ['kml', 'kmz']:
                result = self.kml_processor.process_file(
                    file_path,
                    validate_geometry=True,
                    generate_quality_report=generate_quality_report
                )
            elif file_format == 'geojson':
                result = self.geojson_processor.process_file(
                    file_path,
                    validate_geometry=True,
                    generate_quality_report=generate_quality_report
                )
            elif file_format == 'csv':
                result = self.csv_processor.process_file(
                    file_path,
                    validate_geometry=True,
                    generate_quality_report=generate_quality_report
                )
            elif file_format == 'text':
                result = await self.text_processor.process_file(
                    file_path,
                    extract_entities=True,
                    analyze_sentiment=True,
                    cross_reference_geo=True,
                    generate_quality_report=generate_quality_report
                )
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            if not result.success:
                raise Exception(f"Parsing failed: {'; '.join(result.errors)}")
            
            if progress_callback:
                await progress_callback({"stage": "processed", "progress": 60})
            
            # Store in database if requested
            if store_in_database and result.features:
                dataset_name = dataset_name or f"{file_format.upper()}: {file_path_obj.name}"
                db_result = await self.store_features_in_database(result.features, dataset_name)
                result.metadata.update(db_result)
            
            if progress_callback:
                await progress_callback({"stage": "completed", "progress": 100})
            
            # Compile final result
            if file_format == 'text':
                # Text processing returns different structure
                final_result = {
                    "success": True,
                    "file_path": str(file_path_obj),
                    "file_format": file_format,
                    "text_data_count": len(result.text_data),
                    "text_data": result.text_data,
                    "entities_count": len(result.entities),
                    "entities": result.entities,
                    "sentiment_analysis": result.sentiment_analysis,
                    "cross_references": result.cross_references,
                    "metadata": result.metadata,
                    "warnings": result.warnings,
                    "errors": result.errors
                }
            else:
                # Geospatial processing
                final_result = {
                    "success": True,
                    "file_path": str(file_path_obj),
                    "file_format": file_format,
                    "features_count": len(result.features),
                    "features": result.features,
                    "metadata": result.metadata,
                    "warnings": result.warnings,
                    "errors": result.errors
                }
            
            if result.quality_report:
                final_result["quality_report"] = result.quality_report.to_dict()
                self.quality_reports_generated += 1

            self.files_processed += 1
            if file_format == 'text':
                # For text files, track entities instead of features
                self.features_extracted += len(result.entities)
            else:
                self.features_extracted += len(result.features)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {e}")
            self.processing_errors += 1
            
            if progress_callback:
                await progress_callback({"stage": "error", "progress": 0, "error": str(e)})
            
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "features_count": 0,
                "features": [],
                "metadata": {},
                "warnings": [],
                "errors": [str(e)]
            }
    
    async def parse_batch(
        self,
        file_paths: List[str],
        store_in_database: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Parse multiple files in batch."""
        results = []
        total_features = 0
        total_errors = 0
        
        try:
            for i, file_path in enumerate(file_paths):
                if progress_callback:
                    overall_progress = int((i / len(file_paths)) * 100)
                    await progress_callback({
                        "stage": "batch_processing",
                        "progress": overall_progress,
                        "current_file": file_path,
                        "file_index": i + 1,
                        "total_files": len(file_paths)
                    })
                
                # Parse individual file
                result = await self.parse_file(
                    file_path,
                    store_in_database=store_in_database,
                    generate_quality_report=True
                )
                
                results.append(result)
                total_features += result.get("features_count", 0)
                
                if not result.get("success", False):
                    total_errors += 1
            
            if progress_callback:
                await progress_callback({
                    "stage": "batch_completed",
                    "progress": 100,
                    "total_features": total_features,
                    "total_errors": total_errors
                })
            
            return {
                "success": True,
                "batch_size": len(file_paths),
                "total_features": total_features,
                "total_errors": total_errors,
                "files_processed": len(file_paths) - total_errors,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_size": len(file_paths),
                "results": results
            }
    
    async def parse_zip_archive(
        self,
        zip_path: str,
        store_in_database: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Parse a ZIP archive containing multiple geospatial files."""
        try:
            zip_path_obj = Path(zip_path)
            
            if not zip_path_obj.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")
            
            self.logger.info(f"Processing ZIP archive: {zip_path}")
            
            # Extract ZIP contents to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_path_obj, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
                    
                    # Find supported files
                    extracted_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = Path(root) / file
                            if self._detect_file_format(file_path):
                                extracted_files.append(str(file_path))
                    
                    if not extracted_files:
                        return {
                            "success": False,
                            "error": "No supported geospatial files found in ZIP archive",
                            "zip_path": str(zip_path_obj)
                        }
                    
                    # Process files in batch
                    return await self.parse_batch(
                        extracted_files,
                        store_in_database=store_in_database,
                        progress_callback=progress_callback
                    )
        
        except Exception as e:
            self.logger.error(f"Error processing ZIP archive {zip_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "zip_path": zip_path
            }
    
    async def store_features_in_database(
        self,
        features: List[Dict[str, Any]],
        dataset_name: str
    ) -> Dict[str, Any]:
        """Store parsed features in database."""
        if not self.db_session_factory:
            return {
                "database_stored": False,
                "error": "Database not available"
            }
        
        try:
            with self.db_session_factory() as session:
                # Create dataset record
                dataset = Dataset(
                    name=dataset_name,
                    description=f"Processed dataset with {len(features)} features",
                    file_type="parsed",
                    status="processing"
                )
                session.add(dataset)
                session.flush()  # Get the dataset ID
                
                stored_features = 0
                stored_sites = 0
                
                # Store features
                for feature in features:
                    try:
                        # Create GeospatialFeature record
                        geospatial_feature = GeospatialFeature(
                            dataset_id=dataset.id,
                            name=feature.get("name"),
                            description=feature.get("description"),
                            geometry=f"SRID=4326;{self._geometry_to_wkt(feature.get('geometry'))}",
                            properties=feature.get("properties", {}),
                            feature_type=feature.get("feature_type", "point"),
                            source_layer=feature.get("folder_path", [""])[0] if feature.get("folder_path") else None
                        )
                        session.add(geospatial_feature)
                        stored_features += 1
                        
                        # If it's a cultural/sacred site, also create SacredSite record
                        if self._is_sacred_site(feature):
                            sacred_site = SacredSite(
                                name=feature.get("name"),
                                description=feature.get("description"),
                                site_type=self._determine_site_type(feature),
                                location=f"SRID=4326;{self._geometry_to_wkt(feature.get('geometry'))}",
                                metadata=feature.get("properties", {}),
                                verified=False
                            )
                            session.add(sacred_site)
                            stored_sites += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error storing feature: {e}")
                
                # Update dataset status
                dataset.status = "completed"
                dataset.metadata = {
                    "features_stored": stored_features,
                    "sacred_sites_created": stored_sites,
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                session.commit()
                
                self.database_inserts += stored_features + stored_sites
                
                return {
                    "database_stored": True,
                    "dataset_id": str(dataset.id),
                    "features_stored": stored_features,
                    "sacred_sites_created": stored_sites
                }
        
        except Exception as e:
            self.logger.error(f"Error storing features in database: {e}")
            return {
                "database_stored": False,
                "error": str(e)
            }
    
    async def validate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Validate an existing dataset."""
        if not self.db_session_factory:
            return {"error": "Database not available"}
        
        try:
            with self.db_session_factory() as session:
                # Get dataset and its features
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()
                if not dataset:
                    return {"error": f"Dataset {dataset_id} not found"}
                
                features = session.query(GeospatialFeature).filter_by(dataset_id=dataset_id).all()
                
                # Convert to standard format for quality checking
                feature_data = []
                for feature in features:
                    feature_dict = {
                        "name": feature.name,
                        "description": feature.description,
                        "geometry": self._wkt_to_geojson(feature.geometry),
                        "properties": feature.properties or {}
                    }
                    feature_data.append(feature_dict)
                
                # Generate quality report
                quality_report = self.quality_checker.check_dataset_quality(
                    feature_data,
                    dataset.name
                )
                
                # Update dataset with quality info
                if not dataset.metadata:
                    dataset.metadata = {}
                dataset.metadata.update({
                    "last_validated": datetime.utcnow().isoformat(),
                    "quality_score": quality_report.metrics.overall_quality_score,
                    "total_issues": len(quality_report.issues)
                })
                session.commit()
                
                return {
                    "success": True,
                    "dataset_id": dataset_id,
                    "quality_report": quality_report.to_dict()
                }
        
        except Exception as e:
            self.logger.error(f"Error validating dataset {dataset_id}: {e}")
            return {"error": str(e)}
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect parser-specific metrics."""
        avg_features_per_file = 0.0
        if self.files_processed > 0:
            avg_features_per_file = self.features_extracted / self.files_processed
        
        error_rate = 0.0
        total_operations = self.files_processed + self.processing_errors
        if total_operations > 0:
            error_rate = self.processing_errors / total_operations
        
        return {
            "files_processed": self.files_processed,
            "features_extracted": self.features_extracted,
            "database_inserts": self.database_inserts,
            "processing_errors": self.processing_errors,
            "quality_reports_generated": self.quality_reports_generated,
            "avg_features_per_file": avg_features_per_file,
            "error_rate": error_rate,
            "supported_formats": list(self.supported_formats.keys()),
            "database_available": self.db_session_factory is not None
        }
    
    def _get_capabilities(self) -> List[str]:
        """Get parser agent capabilities."""
        capabilities = [
            "enhanced_parser",
            "kml_parser",
            "geojson_parser",
            "csv_parser",
            "text_processor",
            "nlp_analysis",
            "mythological_entity_extraction",
            "sentiment_analysis",
            "batch_processing",
            "zip_archive_processing",
            "database_integration",
            "quality_assessment",
            "geometry_validation"
        ]
        
        for ext in self.supported_formats.keys():
            capabilities.append(f"parse{ext}")
        
        return capabilities
    
    def _detect_file_format(self, file_path: Path) -> Optional[str]:
        """Detect file format from extension."""
        return self.supported_formats.get(file_path.suffix.lower())
    
    def _is_sacred_site(self, feature: Dict[str, Any]) -> bool:
        """Determine if feature represents a sacred site."""
        # Simple heuristic - could be enhanced with ML classification
        props = feature.get("properties", {})
        name = feature.get("name", "").lower()
        description = feature.get("description", "").lower()
        
        sacred_keywords = [
            "temple", "shrine", "church", "cathedral", "mosque", "synagogue",
            "sacred", "holy", "spiritual", "religious", "monastery", "abbey",
            "monument", "memorial", "burial", "cemetery", "tomb", "pyramid",
            "stone circle", "henge", "dolmen", "menhir", "petroglyphs"
        ]
        
        text_to_check = f"{name} {description} {' '.join(str(v) for v in props.values())}"
        
        return any(keyword in text_to_check for keyword in sacred_keywords)
    
    def _determine_site_type(self, feature: Dict[str, Any]) -> str:
        """Determine site type from feature data."""
        # Simple classification - could be enhanced
        text = f"{feature.get('name', '')} {feature.get('description', '')}".lower()
        
        if any(word in text for word in ["temple", "shrine"]):
            return "temple"
        elif any(word in text for word in ["church", "cathedral", "chapel"]):
            return "temple"
        elif any(word in text for word in ["monument", "memorial"]):
            return "monument"
        elif any(word in text for word in ["burial", "cemetery", "tomb"]):
            return "burial_ground"
        elif any(word in text for word in ["stone", "circle", "henge"]):
            return "ceremonial_site"
        else:
            return "historical"
    
    def _geometry_to_wkt(self, geometry: Dict[str, Any]) -> str:
        """Convert GeoJSON geometry to WKT format."""
        if not geometry:
            return "POINT EMPTY"
        
        geom_type = geometry.get("type", "").upper()
        coordinates = geometry.get("coordinates", [])
        
        if geom_type == "POINT" and len(coordinates) >= 2:
            return f"POINT({coordinates[0]} {coordinates[1]})"
        elif geom_type == "LINESTRING":
            coords_str = ", ".join(f"{c[0]} {c[1]}" for c in coordinates)
            return f"LINESTRING({coords_str})"
        elif geom_type == "POLYGON":
            rings = []
            for ring in coordinates:
                ring_str = ", ".join(f"{c[0]} {c[1]}" for c in ring)
                rings.append(f"({ring_str})")
            return f"POLYGON({', '.join(rings)})"
        
        return "POINT EMPTY"
    
    def _wkt_to_geojson(self, wkt_geometry) -> Dict[str, Any]:
        """Convert WKT geometry to GeoJSON format (simplified)."""
        # This is a placeholder - would need proper WKT parsing
        return {"type": "Point", "coordinates": [0, 0]}
    
    def _update_statistics_from_result(self, result: Dict[str, Any]) -> None:
        """Update statistics from task result."""
        if result and result.get("success"):
            features_count = result.get("features_count", 0)
            self.features_extracted += features_count
            
            if "database_stored" in result and result["database_stored"]:
                self.database_inserts += result.get("features_stored", 0)
    
    async def _send_progress_update(self, task_id: str, progress: Dict[str, Any]) -> None:
        """Send progress update via NATS."""
        if not self.messaging:
            return
        
        try:
            progress_message = AgentMessage.create(
                sender_id=self.agent_id,
                message_type="task_progress",
                payload={
                    "task_id": task_id,
                    "progress": progress,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.nats_client.publish("agents.progress", progress_message)
            
        except Exception as e:
            self.logger.warning(f"Failed to send progress update: {e}")
    
    def _verify_parsing_dependencies(self) -> None:
        """Verify that parsing dependencies are available."""
        issues = []
        
        if not DATABASE_AVAILABLE:
            issues.append("Database libraries not available - no data persistence")
        
        if issues:
            self.logger.warning("Dependency issues: " + "; ".join(issues))
        else:
            self.logger.info("All parsing dependencies verified")
    
    async def _process_parsing_queue(self) -> None:
        """Process any queued parsing requests."""
        # This would integrate with the task queue system
        pass
    
    # Message handlers
    
    async def _handle_parse_request(self, message: AgentMessage) -> None:
        """Handle parsing requests via NATS."""
        try:
            request_data = message.payload
            file_path = request_data.get("file_path")
            
            if not file_path:
                raise ValueError("file_path is required")
            
            # Perform parsing
            result = await self.parse_file(
                file_path=file_path,
                store_in_database=request_data.get("store_in_db", True),
                generate_quality_report=request_data.get("generate_report", True)
            )
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="parse_response",
                payload=result,
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling parse request: {e}")
    
    async def _handle_file_upload(self, message: AgentMessage) -> None:
        """Handle file upload notifications for automatic parsing."""
        try:
            upload_data = message.payload
            file_path = upload_data.get("file_path")
            file_type = upload_data.get("file_type")
            
            if file_path and self._detect_file_format(Path(file_path)):
                self.logger.info(f"Auto-parsing uploaded file: {file_path}")
                result = await self.parse_file(file_path)
                
                # Publish parsing result
                result_message = AgentMessage.create(
                    sender_id=self.agent_id,
                    message_type="file_parsed",
                    payload={
                        "file_path": file_path,
                        "parsing_result": result,
                        "auto_parsed": True
                    }
                )
                
                await self.nats_client.publish("agents.parsers.results", result_message)
                
        except Exception as e:
            self.logger.error(f"Error handling file upload: {e}")
    
    async def _handle_batch_request(self, message: AgentMessage) -> None:
        """Handle batch processing requests."""
        try:
            request_data = message.payload
            file_paths = request_data.get("file_paths", [])
            
            if not file_paths:
                raise ValueError("file_paths is required")
            
            # Perform batch parsing
            result = await self.parse_batch(
                file_paths=file_paths,
                store_in_database=request_data.get("store_in_db", True)
            )
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="batch_response",
                payload=result,
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling batch request: {e}")


# Main entry point for running the agent
async def main():
    """Main entry point for running the Enhanced KMLParserAgent."""
    import signal
    import sys
    
    # Create and configure agent
    agent = KMLParserAgent()
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...")
        asyncio.create_task(agent.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the agent
        await agent.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Agent failed: {e}")
        sys.exit(1)
    
    print("Enhanced KMLParserAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())