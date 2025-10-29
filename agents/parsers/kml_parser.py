"""
A2A World Platform - KML Parser Agent

Agent responsible for parsing KML files and extracting geospatial data
for pattern discovery and analysis.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    from fastkml import kml
    import geojson
    from shapely.geometry import Point, LineString, Polygon
    GEOSPATIAL_LIBS_AVAILABLE = True
except ImportError:
    GEOSPATIAL_LIBS_AVAILABLE = False

from agents.core.base_agent import BaseAgent
from agents.core.config import ParserAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task


class KMLParserAgent(BaseAgent):
    """
    Agent that parses KML files and extracts geospatial features.
    
    Capabilities:
    - Parse KML and KMZ files
    - Extract placemarks, folders, and network links
    - Convert to standardized GeoJSON format
    - Validate geometry and metadata
    - Handle large files with streaming processing
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
        
        self.supported_formats = self.config.supported_formats
        self.max_file_size_mb = self.config.max_file_size_mb
        self.batch_size = self.config.batch_size
        
        # Parsing statistics
        self.files_parsed = 0
        self.features_extracted = 0
        self.parsing_errors = 0
        
        self.logger.info(f"KMLParserAgent {self.agent_id} initialized with formats: {self.supported_formats}")
    
    async def process(self) -> None:
        """
        Main processing loop - handle parsing requests and file monitoring.
        """
        try:
            # Process any pending parsing requests
            await self._process_parsing_queue()
            
        except Exception as e:
            self.logger.error(f"Error in parsing process: {e}")
    
    async def agent_initialize(self) -> None:
        """
        Parser agent specific initialization.
        """
        try:
            if not GEOSPATIAL_LIBS_AVAILABLE:
                self.logger.warning("Geospatial libraries not available - parsing will be limited")
            
            # Verify parsing capabilities
            self._verify_parsing_dependencies()
            
            # Create data directories if needed
            self.config.data_path.mkdir(parents=True, exist_ok=True)
            self.config.temp_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("KMLParserAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize KMLParserAgent: {e}")
            raise
    
    async def setup_subscriptions(self) -> None:
        """
        Setup parser-specific message subscriptions.
        """
        if not self.messaging:
            return
        
        # Subscribe to parsing requests
        parse_sub_id = await self.nats_client.subscribe(
            "agents.parsers.request",
            self._handle_parse_request,
            queue_group="parser-workers"
        )
        self.subscription_ids.append(parse_sub_id)
        
        # Subscribe to file upload notifications
        upload_sub_id = await self.nats_client.subscribe(
            "agents.files.uploaded",
            self._handle_file_upload,
            queue_group="parser-uploads"
        )
        self.subscription_ids.append(upload_sub_id)
    
    async def handle_task(self, task: Task) -> None:
        """
        Handle parsing task processing.
        """
        self.logger.info(f"Processing parsing task {task.task_id}: {task.task_type}")
        
        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)
            
            result = None
            
            if task.task_type == "parse_kml_file":
                result = await self._parse_kml_task(task)
            elif task.task_type == "parse_file":
                result = await self._parse_file_task(task)
            elif task.task_type == "validate_geometry":
                result = await self._validate_geometry_task(task)
            else:
                raise ValueError(f"Unknown parsing task type: {task.task_type}")
            
            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)
            
            self.processed_tasks += 1
            self.files_parsed += 1
            
            # Update features count
            if result and "features" in result:
                self.features_extracted += len(result["features"])
            
            self.logger.info(f"Completed parsing task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing parsing task {task.task_id}: {e}")
            
            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)
            
            self.failed_tasks += 1
            self.parsing_errors += 1
        
        finally:
            self.current_tasks.discard(task.task_id)
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect parser-specific metrics.
        """
        avg_features_per_file = 0.0
        if self.files_parsed > 0:
            avg_features_per_file = self.features_extracted / self.files_parsed
        
        error_rate = 0.0
        total_files = self.files_parsed + self.parsing_errors
        if total_files > 0:
            error_rate = self.parsing_errors / total_files
        
        return {
            "files_parsed": self.files_parsed,
            "features_extracted": self.features_extracted,
            "parsing_errors": self.parsing_errors,
            "avg_features_per_file": avg_features_per_file,
            "error_rate": error_rate,
            "supported_formats": len(self.supported_formats),
            "max_file_size_mb": self.max_file_size_mb
        }
    
    def _get_capabilities(self) -> List[str]:
        """
        Get parser agent capabilities.
        """
        capabilities = [
            "parser",
            "kml_parser",
            "parse_kml_file",
            "parse_file",
            "geojson_conversion",
            "geometry_validation"
        ]
        
        # Add format-specific capabilities
        for fmt in self.supported_formats:
            capabilities.append(f"parse_{fmt}")
        
        return capabilities
    
    async def _parse_kml_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle KML file parsing task.
        """
        file_path = task.parameters.get("file_path")
        if not file_path:
            raise ValueError("file_path parameter is required")
        
        return await self.parse_kml_file(file_path)
    
    async def _parse_file_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle generic file parsing task.
        """
        file_path = task.parameters.get("file_path")
        file_type = task.parameters.get("file_type", "kml")
        
        if not file_path:
            raise ValueError("file_path parameter is required")
        
        if file_type.lower() in ["kml", "kmz"]:
            return await self.parse_kml_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    async def _validate_geometry_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle geometry validation task.
        """
        geometry_data = task.input_data.get("geometry")
        if not geometry_data:
            raise ValueError("geometry data is required")
        
        return await self._validate_geometry(geometry_data)
    
    async def parse_kml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a KML file and extract geospatial features.
        """
        try:
            file_path_obj = Path(file_path)
            
            # Validate file
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")
            
            self.logger.info(f"Parsing KML file: {file_path} ({file_size_mb:.1f}MB)")
            
            # Parse based on availability of libraries
            if GEOSPATIAL_LIBS_AVAILABLE:
                result = await self._parse_kml_with_fastkml(file_path_obj)
            else:
                result = await self._parse_kml_basic(file_path_obj)
            
            # Add metadata
            result.update({
                "file_path": str(file_path_obj),
                "file_size_mb": file_size_mb,
                "parsed_at": datetime.utcnow().isoformat(),
                "parser_agent": self.agent_id
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing KML file {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "status": "failed",
                "parsed_at": datetime.utcnow().isoformat()
            }
    
    async def _parse_kml_with_fastkml(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse KML using fastkml library (full functionality).
        """
        features = []
        folders = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse KML
            k = kml.KML()
            k.from_string(content)
            
            # Extract features
            for feature in k.features():
                if hasattr(feature, 'features'):
                    # Document or Folder
                    folder_data = {
                        "name": getattr(feature, 'name', 'Unnamed'),
                        "description": getattr(feature, 'description', ''),
                        "type": "folder"
                    }
                    folders.append(folder_data)
                    
                    # Process placemarks within folder
                    for placemark in feature.features():
                        feature_data = await self._extract_placemark_data(placemark)
                        if feature_data:
                            features.append(feature_data)
                else:
                    # Individual placemark
                    feature_data = await self._extract_placemark_data(feature)
                    if feature_data:
                        features.append(feature_data)
            
            # Validate geometries if enabled
            if self.config.validate_geometry:
                features = await self._validate_features(features)
            
            return {
                "status": "success",
                "format": "kml",
                "features": features,
                "folders": folders,
                "feature_count": len(features),
                "library": "fastkml"
            }
            
        except Exception as e:
            raise Exception(f"FastKML parsing failed: {e}")
    
    async def _parse_kml_basic(self, file_path: Path) -> Dict[str, Any]:
        """
        Basic KML parsing without external libraries (limited functionality).
        """
        import xml.etree.ElementTree as ET
        
        features = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Handle namespaces
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            # Find all placemarks
            placemarks = root.findall('.//kml:Placemark', ns)
            
            for placemark in placemarks:
                feature_data = await self._extract_placemark_basic(placemark, ns)
                if feature_data:
                    features.append(feature_data)
            
            return {
                "status": "success",
                "format": "kml",
                "features": features,
                "feature_count": len(features),
                "library": "xml.etree (basic)"
            }
            
        except Exception as e:
            raise Exception(f"Basic XML parsing failed: {e}")
    
    async def _extract_placemark_data(self, placemark) -> Optional[Dict[str, Any]]:
        """
        Extract data from a placemark using fastkml.
        """
        try:
            feature_data = {
                "name": getattr(placemark, 'name', 'Unnamed'),
                "description": getattr(placemark, 'description', ''),
                "properties": {}
            }
            
            # Extract geometry
            if hasattr(placemark, 'geometry') and placemark.geometry:
                geometry = placemark.geometry
                
                if hasattr(geometry, 'geom_type'):
                    feature_data["geometry_type"] = geometry.geom_type
                    
                    if geometry.geom_type == 'Point':
                        coords = list(geometry.coords)[0]
                        feature_data["geometry"] = {
                            "type": "Point",
                            "coordinates": [coords[0], coords[1]]
                        }
                        feature_data["latitude"] = coords[1]
                        feature_data["longitude"] = coords[0]
                        
                    elif geometry.geom_type in ['LineString', 'Polygon']:
                        # Convert to GeoJSON format
                        feature_data["geometry"] = json.loads(geojson.dumps(geometry))
            
            # Extract extended data
            if hasattr(placemark, 'extended_data') and placemark.extended_data:
                for data in placemark.extended_data.elements:
                    if hasattr(data, 'name') and hasattr(data, 'value'):
                        feature_data["properties"][data.name] = data.value
            
            return feature_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting placemark data: {e}")
            return None
    
    async def _extract_placemark_basic(self, placemark, ns) -> Optional[Dict[str, Any]]:
        """
        Extract data from a placemark using basic XML parsing.
        """
        try:
            name_elem = placemark.find('kml:name', ns)
            name = name_elem.text if name_elem is not None else 'Unnamed'
            
            desc_elem = placemark.find('kml:description', ns)
            description = desc_elem.text if desc_elem is not None else ''
            
            feature_data = {
                "name": name,
                "description": description,
                "properties": {}
            }
            
            # Extract Point geometry (simplified)
            point_elem = placemark.find('.//kml:Point/kml:coordinates', ns)
            if point_elem is not None:
                coords_text = point_elem.text.strip()
                try:
                    coords = coords_text.split(',')
                    lon, lat = float(coords[0]), float(coords[1])
                    feature_data["geometry_type"] = "Point"
                    feature_data["longitude"] = lon
                    feature_data["latitude"] = lat
                    feature_data["geometry"] = {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                except (ValueError, IndexError):
                    self.logger.warning(f"Invalid coordinates: {coords_text}")
            
            return feature_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting placemark (basic): {e}")
            return None
    
    async def _validate_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate geometries in extracted features.
        """
        validated_features = []
        
        for feature in features:
            try:
                # Basic validation
                if "geometry" in feature:
                    geometry = feature["geometry"]
                    
                    # Validate coordinates are within valid ranges
                    if geometry.get("type") == "Point":
                        coords = geometry.get("coordinates", [])
                        if len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                feature["geometry_valid"] = True
                            else:
                                feature["geometry_valid"] = False
                                feature["validation_error"] = f"Invalid coordinates: {lon}, {lat}"
                        else:
                            feature["geometry_valid"] = False
                            feature["validation_error"] = "Missing coordinates"
                    else:
                        # For non-point geometries, assume valid for now
                        feature["geometry_valid"] = True
                else:
                    feature["geometry_valid"] = False
                    feature["validation_error"] = "No geometry found"
                
                validated_features.append(feature)
                
            except Exception as e:
                feature["geometry_valid"] = False
                feature["validation_error"] = str(e)
                validated_features.append(feature)
        
        return validated_features
    
    async def _validate_geometry(self, geometry_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single geometry object.
        """
        try:
            geometry_type = geometry_data.get("type")
            coordinates = geometry_data.get("coordinates")
            
            if not geometry_type or not coordinates:
                return {"valid": False, "error": "Missing geometry type or coordinates"}
            
            if geometry_type == "Point":
                if len(coordinates) >= 2:
                    lon, lat = coordinates[0], coordinates[1]
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        return {"valid": True, "geometry_type": geometry_type}
                    else:
                        return {"valid": False, "error": f"Invalid coordinates: {lon}, {lat}"}
                else:
                    return {"valid": False, "error": "Point must have at least 2 coordinates"}
            
            # Add validation for other geometry types as needed
            return {"valid": True, "geometry_type": geometry_type}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _verify_parsing_dependencies(self) -> None:
        """
        Verify that parsing dependencies are available.
        """
        if GEOSPATIAL_LIBS_AVAILABLE:
            self.logger.info("Geospatial libraries available: fastkml, geojson, shapely")
        else:
            self.logger.warning("Geospatial libraries not available - using basic XML parsing")
    
    async def _process_parsing_queue(self) -> None:
        """
        Process any queued parsing requests.
        """
        # This would integrate with the task queue system
        # For now, this is a placeholder
        pass
    
    # Message handlers
    
    async def _handle_parse_request(self, message: AgentMessage) -> None:
        """
        Handle parsing requests via NATS.
        """
        try:
            request_data = message.payload
            file_path = request_data.get("file_path")
            file_type = request_data.get("file_type", "kml")
            
            if not file_path:
                raise ValueError("file_path is required")
            
            # Perform parsing
            if file_type.lower() in ["kml", "kmz"]:
                result = await self.parse_kml_file(file_path)
            else:
                result = {"error": f"Unsupported file type: {file_type}"}
            
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
        """
        Handle file upload notifications for automatic parsing.
        """
        try:
            upload_data = message.payload
            file_path = upload_data.get("file_path")
            file_type = upload_data.get("file_type")
            
            if file_path and file_type in self.supported_formats:
                self.logger.info(f"Auto-parsing uploaded file: {file_path}")
                result = await self.parse_kml_file(file_path)
                
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


# Main entry point for running the agent
async def main():
    """
    Main entry point for running the KMLParserAgent.
    """
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
    
    print("KMLParserAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())