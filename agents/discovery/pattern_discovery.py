"""
A2A World Platform - Pattern Discovery Agent

Agent responsible for discovering patterns in geospatial data using
clustering algorithms like HDBSCAN and statistical analysis.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import json

try:
    import hdbscan
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import scipy.spatial.distance as distance
    CLUSTERING_LIBS_AVAILABLE = True
except ImportError:
    CLUSTERING_LIBS_AVAILABLE = False

from agents.core.base_agent import BaseAgent
from agents.core.config import DiscoveryAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task


class PatternDiscoveryAgent(BaseAgent):
    """
    Agent that discovers patterns in geospatial and cultural data.
    
    Capabilities:
    - HDBSCAN clustering for density-based pattern discovery
    - DBSCAN and K-Means clustering for comparison
    - Spatial pattern analysis and hotspot detection
    - Multi-dimensional feature clustering
    - Pattern significance assessment
    - Temporal pattern analysis
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DiscoveryAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="discovery",
            config=config or DiscoveryAgentConfig(),
            config_file=config_file
        )
        
        self.clustering_algorithms = ["hdbscan", "dbscan", "kmeans"]
        self.min_cluster_size = self.config.min_cluster_size
        self.min_samples = self.config.min_samples
        self.confidence_threshold = self.config.confidence_threshold
        self.search_radius_km = self.config.search_radius_km
        self.max_distance_km = self.config.max_distance_km
        
        # Discovery statistics
        self.patterns_discovered = 0
        self.datasets_processed = 0
        self.discovery_errors = 0
        self.significant_patterns = 0
        
        # Pattern cache for avoiding duplicate analysis
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"PatternDiscoveryAgent {self.agent_id} initialized with algorithms: {self.clustering_algorithms}")
    
    async def process(self) -> None:
        """
        Main processing loop - handle discovery requests and pattern analysis.
        """
        try:
            # Process any pending discovery requests
            await self._process_discovery_queue()
            
            # Clean up old cache entries periodically
            if self.processed_tasks % 50 == 0:
                await self._cleanup_pattern_cache()
                
        except Exception as e:
            self.logger.error(f"Error in discovery process: {e}")
    
    async def agent_initialize(self) -> None:
        """
        Discovery agent specific initialization.
        """
        try:
            if not CLUSTERING_LIBS_AVAILABLE:
                self.logger.warning("Clustering libraries not available - discovery will be limited")
            
            # Verify clustering capabilities
            self._verify_clustering_dependencies()
            
            self.logger.info("PatternDiscoveryAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PatternDiscoveryAgent: {e}")
            raise
    
    async def setup_subscriptions(self) -> None:
        """
        Setup discovery-specific message subscriptions.
        """
        if not self.messaging:
            return
        
        # Subscribe to discovery requests
        discovery_sub_id = await self.nats_client.subscribe(
            "agents.discovery.request",
            self._handle_discovery_request,
            queue_group="discovery-workers"
        )
        self.subscription_ids.append(discovery_sub_id)
        
        # Subscribe to parsed data for automatic pattern discovery
        parsed_sub_id = await self.nats_client.subscribe(
            "agents.parsers.results",
            self._handle_parsed_data,
            queue_group="discovery-parsed"
        )
        self.subscription_ids.append(parsed_sub_id)
    
    async def handle_task(self, task: Task) -> None:
        """
        Handle pattern discovery task processing.
        """
        self.logger.info(f"Processing discovery task {task.task_id}: {task.task_type}")
        
        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)
            
            result = None
            
            if task.task_type == "discover_patterns":
                result = await self._discover_patterns_task(task)
            elif task.task_type == "spatial_analysis":
                result = await self._spatial_analysis_task(task)
            elif task.task_type == "cluster_analysis":
                result = await self._cluster_analysis_task(task)
            elif task.task_type == "hotspot_detection":
                result = await self._hotspot_detection_task(task)
            else:
                raise ValueError(f"Unknown discovery task type: {task.task_type}")
            
            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)
            
            self.processed_tasks += 1
            self.datasets_processed += 1
            
            # Update pattern counts
            if result and "patterns" in result:
                patterns_found = len(result["patterns"])
                self.patterns_discovered += patterns_found
                
                # Count significant patterns
                significant = sum(1 for p in result["patterns"]
                                if p.get("significance_score", 0) > self.confidence_threshold)
                self.significant_patterns += significant
            
            self.logger.info(f"Completed discovery task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing discovery task {task.task_id}: {e}")
            
            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)
            
            self.failed_tasks += 1
            self.discovery_errors += 1
        
        finally:
            self.current_tasks.discard(task.task_id)
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect discovery-specific metrics.
        """
        avg_patterns_per_dataset = 0.0
        if self.datasets_processed > 0:
            avg_patterns_per_dataset = self.patterns_discovered / self.datasets_processed
        
        significance_rate = 0.0
        if self.patterns_discovered > 0:
            significance_rate = self.significant_patterns / self.patterns_discovered
        
        error_rate = 0.0
        total_processed = self.datasets_processed + self.discovery_errors
        if total_processed > 0:
            error_rate = self.discovery_errors / total_processed
        
        return {
            "patterns_discovered": self.patterns_discovered,
            "significant_patterns": self.significant_patterns,
            "datasets_processed": self.datasets_processed,
            "discovery_errors": self.discovery_errors,
            "avg_patterns_per_dataset": avg_patterns_per_dataset,
            "significance_rate": significance_rate,
            "error_rate": error_rate,
            "cache_size": len(self.pattern_cache),
            "available_algorithms": len(self.clustering_algorithms)
        }
    
    def _get_capabilities(self) -> List[str]:
        """
        Get discovery agent capabilities.
        """
        capabilities = [
            "discovery",
            "pattern_discovery",
            "discover_patterns",
            "spatial_analysis",
            "cluster_analysis",
            "hotspot_detection",
            "density_clustering",
            "spatial_clustering"
        ]
        
        # Add algorithm-specific capabilities
        for algo in self.clustering_algorithms:
            capabilities.append(f"clustering_{algo}")
        
        return capabilities
    
    async def _discover_patterns_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle pattern discovery task.
        """
        dataset = task.input_data.get("dataset", {})
        algorithm = task.parameters.get("algorithm", self.config.default_algorithm)
        
        if not dataset:
            raise ValueError("Dataset is required for pattern discovery")
        
        return await self.discover_patterns(dataset, algorithm)
    
    async def _spatial_analysis_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle spatial analysis task.
        """
        points = task.input_data.get("points", [])
        analysis_type = task.parameters.get("analysis_type", "hotspot")
        
        return await self._perform_spatial_analysis(points, analysis_type)
    
    async def _cluster_analysis_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle cluster analysis task.
        """
        data = task.input_data.get("data", [])
        algorithm = task.parameters.get("algorithm", "hdbscan")
        
        return await self._perform_clustering(data, algorithm)
    
    async def _hotspot_detection_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle hotspot detection task.
        """
        points = task.input_data.get("points", [])
        intensity_field = task.parameters.get("intensity_field", "value")
        
        return await self._detect_hotspots(points, intensity_field)
    
    async def discover_patterns(self, dataset: Dict[str, Any], algorithm: str = "hdbscan") -> Dict[str, Any]:
        """
        Discover patterns in the given dataset using specified algorithm.
        """
        try:
            dataset_id = dataset.get("id", str(uuid.uuid4()))
            
            # Check cache first
            cache_key = f"{dataset_id}:{algorithm}"
            if cache_key in self.pattern_cache:
                self.logger.debug(f"Returning cached patterns for dataset {dataset_id}")
                return self.pattern_cache[cache_key]
            
            self.logger.info(f"Discovering patterns in dataset {dataset_id} using {algorithm}")
            
            # Extract spatial data
            spatial_data = self._extract_spatial_features(dataset)
            if not spatial_data:
                raise ValueError("No spatial data found in dataset")
            
            # Perform clustering
            clustering_result = await self._perform_clustering(spatial_data, algorithm)
            
            # Analyze clusters for patterns
            patterns = await self._analyze_clusters(clustering_result, spatial_data)
            
            # Assess pattern significance
            for pattern in patterns:
                significance = await self._assess_pattern_significance(pattern, spatial_data)
                pattern.update(significance)
            
            result = {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "patterns": patterns,
                "pattern_count": len(patterns),
                "significant_patterns": len([p for p in patterns if p.get("significant", False)]),
                "clustering_result": clustering_result,
                "discovered_at": datetime.utcnow().isoformat(),
                "discovery_agent": self.agent_id
            }
            
            # Cache result
            self.pattern_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error discovering patterns: {e}")
            return {
                "dataset_id": dataset.get("id", "unknown"),
                "algorithm": algorithm,
                "error": str(e),
                "patterns": [],
                "pattern_count": 0
            }
    
    async def _perform_clustering(self, data: List[Dict[str, Any]], algorithm: str) -> Dict[str, Any]:
        """
        Perform clustering analysis on the data.
        """
        try:
            if not data:
                return {"error": "No data provided", "clusters": []}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Extract features for clustering
            features = []
            if "latitude" in df.columns and "longitude" in df.columns:
                features.extend(["latitude", "longitude"])
            
            # Add other numerical features
            for col in df.columns:
                if col not in features and pd.api.types.is_numeric_dtype(df[col]):
                    features.append(col)
            
            if not features:
                return {"error": "No suitable features for clustering", "clusters": []}
            
            X = df[features].values
            
            # Handle missing values
            if np.isnan(X).any():
                X = np.nan_to_num(X)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering based on algorithm
            if algorithm == "hdbscan" and CLUSTERING_LIBS_AVAILABLE:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples
                )
                cluster_labels = clusterer.fit_predict(X_scaled)
                
                # Extract additional HDBSCAN info
                cluster_info = {
                    "probabilities": clusterer.probabilities_.tolist() if hasattr(clusterer, 'probabilities_') else [],
                    "outlier_scores": clusterer.outlier_scores_.tolist() if hasattr(clusterer, 'outlier_scores_') else []
                }
                
            elif algorithm == "dbscan" and CLUSTERING_LIBS_AVAILABLE:
                clusterer = DBSCAN(eps=0.5, min_samples=self.min_samples)
                cluster_labels = clusterer.fit_predict(X_scaled)
                cluster_info = {}
                
            elif algorithm == "kmeans" and CLUSTERING_LIBS_AVAILABLE:
                # Estimate number of clusters
                n_clusters = min(8, max(2, len(data) // 10))
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(X_scaled)
                cluster_info = {
                    "centroids": clusterer.cluster_centers_.tolist(),
                    "inertia": clusterer.inertia_
                }
                
            else:
                # Simple distance-based clustering fallback
                cluster_labels = self._simple_clustering(X_scaled)
                cluster_info = {"algorithm": "simple_distance"}
            
            # Organize results
            clusters = self._organize_clusters(data, cluster_labels, features)
            
            return {
                "algorithm": algorithm,
                "features_used": features,
                "cluster_labels": cluster_labels.tolist(),
                "clusters": clusters,
                "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "n_noise": np.sum(cluster_labels == -1),
                "cluster_info": cluster_info
            }
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {"error": str(e), "clusters": []}
    
    def _simple_clustering(self, X: np.ndarray) -> np.ndarray:
        """
        Simple distance-based clustering fallback when libraries are not available.
        """
        if len(X) < 2:
            return np.array([0] * len(X))
        
        # Use simple threshold-based clustering
        distances = distance.pdist(X)
        threshold = np.percentile(distances, 25)  # Use 25th percentile as threshold
        
        labels = np.arange(len(X))
        
        # Simple agglomerative approach
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if np.linalg.norm(X[i] - X[j]) < threshold:
                    # Merge clusters
                    labels[labels == labels[j]] = labels[i]
        
        # Renumber labels to be consecutive
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[label] for label in labels])
    
    def _organize_clusters(self, data: List[Dict[str, Any]], labels: np.ndarray, features: List[str]) -> List[Dict[str, Any]]:
        """
        Organize clustering results into structured clusters.
        """
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise/outliers
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_points = [data[i] for i in cluster_indices]
            
            # Calculate cluster statistics
            cluster_data = pd.DataFrame(cluster_points)
            
            cluster_info = {
                "cluster_id": int(label),
                "size": len(cluster_points),
                "points": cluster_points,
                "indices": cluster_indices.tolist()
            }
            
            # Calculate centroid if spatial data available
            if "latitude" in cluster_data.columns and "longitude" in cluster_data.columns:
                cluster_info["centroid"] = {
                    "latitude": float(cluster_data["latitude"].mean()),
                    "longitude": float(cluster_data["longitude"].mean())
                }
                
                # Calculate spatial extent
                cluster_info["bounds"] = {
                    "north": float(cluster_data["latitude"].max()),
                    "south": float(cluster_data["latitude"].min()),
                    "east": float(cluster_data["longitude"].max()),
                    "west": float(cluster_data["longitude"].min())
                }
            
            # Calculate feature statistics
            for feature in features:
                if feature in cluster_data.columns:
                    values = cluster_data[feature]
                    cluster_info[f"{feature}_stats"] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max())
                    }
            
            clusters.append(cluster_info)
        
        return clusters
    
    async def _analyze_clusters(self, clustering_result: Dict[str, Any], spatial_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze clusters to identify meaningful patterns.
        """
        patterns = []
        clusters = clustering_result.get("clusters", [])
        
        for cluster in clusters:
            try:
                pattern = {
                    "pattern_id": str(uuid.uuid4()),
                    "pattern_type": "spatial_cluster",
                    "cluster_info": cluster,
                    "description": f"Spatial cluster with {cluster['size']} points"
                }
                
                # Add spatial pattern analysis
                if "centroid" in cluster:
                    pattern["spatial_properties"] = {
                        "centroid": cluster["centroid"],
                        "bounds": cluster.get("bounds", {}),
                        "density": cluster["size"] / max(1, self._calculate_cluster_area(cluster)),
                        "compactness": self._calculate_cluster_compactness(cluster)
                    }
                
                # Analyze temporal patterns if timestamps available
                temporal_analysis = self._analyze_temporal_patterns(cluster["points"])
                if temporal_analysis:
                    pattern["temporal_properties"] = temporal_analysis
                
                # Add cultural/thematic analysis if relevant fields present
                thematic_analysis = self._analyze_thematic_patterns(cluster["points"])
                if thematic_analysis:
                    pattern["thematic_properties"] = thematic_analysis
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing cluster {cluster.get('cluster_id', 'unknown')}: {e}")
        
        return patterns
    
    async def _assess_pattern_significance(self, pattern: Dict[str, Any], spatial_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the statistical significance of a discovered pattern.
        """
        try:
            cluster_info = pattern.get("cluster_info", {})
            cluster_size = cluster_info.get("size", 0)
            
            # Basic significance assessment
            significance_score = 0.0
            
            # Size-based significance
            if cluster_size >= self.min_cluster_size:
                significance_score += 0.3
            
            # Density-based significance
            if "spatial_properties" in pattern:
                density = pattern["spatial_properties"].get("density", 0)
                if density > 1.0:  # Above average density
                    significance_score += 0.3
                
                compactness = pattern["spatial_properties"].get("compactness", 0)
                if compactness > 0.5:  # Reasonably compact
                    significance_score += 0.2
            
            # Temporal significance
            if "temporal_properties" in pattern:
                temporal_clustering = pattern["temporal_properties"].get("temporal_clustering", 0)
                significance_score += min(0.2, temporal_clustering)
            
            # Determine if significant
            significant = significance_score >= self.confidence_threshold
            
            return {
                "significance_score": significance_score,
                "significant": significant,
                "confidence_level": significance_score,
                "assessment_criteria": {
                    "size_threshold": cluster_size >= self.min_cluster_size,
                    "density_above_average": pattern.get("spatial_properties", {}).get("density", 0) > 1.0,
                    "reasonable_compactness": pattern.get("spatial_properties", {}).get("compactness", 0) > 0.5
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Error assessing pattern significance: {e}")
            return {
                "significance_score": 0.0,
                "significant": False,
                "error": str(e)
            }
    
    def _extract_spatial_features(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract spatial features from dataset for clustering.
        """
        try:
            # Try different possible data structures
            if "features" in dataset:
                return dataset["features"]
            elif "points" in dataset:
                return dataset["points"]
            elif "data" in dataset:
                return dataset["data"]
            elif isinstance(dataset.get("parsing_result"), dict):
                parsing_result = dataset["parsing_result"]
                return parsing_result.get("features", [])
            else:
                # Assume dataset itself contains the features
                if isinstance(dataset, list):
                    return dataset
                else:
                    return []
        except Exception as e:
            self.logger.error(f"Error extracting spatial features: {e}")
            return []
    
    def _calculate_cluster_area(self, cluster: Dict[str, Any]) -> float:
        """
        Calculate approximate area of cluster in square km.
        """
        try:
            bounds = cluster.get("bounds", {})
            if not bounds:
                return 1.0
            
            # Simple rectangular area calculation
            lat_diff = bounds.get("north", 0) - bounds.get("south", 0)
            lon_diff = bounds.get("east", 0) - bounds.get("west", 0)
            
            # Convert to approximate km (rough calculation)
            area_sq_km = abs(lat_diff * lon_diff) * 111.32 * 111.32  # 1 degree ≈ 111.32 km
            
            return max(1.0, area_sq_km)
            
        except Exception:
            return 1.0
    
    def _calculate_cluster_compactness(self, cluster: Dict[str, Any]) -> float:
        """
        Calculate cluster compactness (0-1 scale).
        """
        try:
            points = cluster.get("points", [])
            if len(points) < 2:
                return 1.0
            
            # Calculate average distance from centroid
            centroid = cluster.get("centroid")
            if not centroid:
                return 0.5
            
            distances = []
            for point in points:
                if "latitude" in point and "longitude" in point:
                    dist = np.sqrt(
                        (point["latitude"] - centroid["latitude"])**2 +
                        (point["longitude"] - centroid["longitude"])**2
                    )
                    distances.append(dist)
            
            if not distances:
                return 0.5
            
            # Compactness inverse to average distance (normalized)
            avg_distance = np.mean(distances)
            max_distance = max(distances)
            
            compactness = 1.0 - (avg_distance / max_distance) if max_distance > 0 else 1.0
            return max(0.0, min(1.0, compactness))
            
        except Exception:
            return 0.5
    
    def _analyze_temporal_patterns(self, points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze temporal patterns in cluster points.
        """
        try:
            # Look for timestamp fields
            timestamp_fields = ["timestamp", "created_at", "date", "time"]
            timestamps = []
            
            for point in points:
                for field in timestamp_fields:
                    if field in point and point[field]:
                        try:
                            # Try to parse timestamp
                            if isinstance(point[field], str):
                                timestamp = datetime.fromisoformat(point[field].replace('Z', '+00:00'))
                            else:
                                timestamp = point[field]
                            timestamps.append(timestamp)
                            break
                        except Exception:
                            continue
            
            if len(timestamps) < 2:
                return None
            
            # Analyze temporal clustering
            timestamps.sort()
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds()
                         for i in range(len(timestamps)-1)]
            
            avg_time_diff = np.mean(time_diffs)
            std_time_diff = np.std(time_diffs)
            
            # Temporal clustering score (lower variance = higher clustering)
            temporal_clustering = 1.0 / (1.0 + std_time_diff / max(1, avg_time_diff))
            
            return {
                "temporal_clustering": temporal_clustering,
                "time_span_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
                "avg_interval_seconds": avg_time_diff,
                "timestamp_count": len(timestamps)
            }
            
        except Exception as e:
            self.logger.debug(f"Temporal analysis failed: {e}")
            return None
    
    def _analyze_thematic_patterns(self, points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze thematic/categorical patterns in cluster points.
        """
        try:
            # Look for categorical fields
            categorical_fields = ["type", "category", "class", "name", "description"]
            thematic_analysis = {}
            
            for field in categorical_fields:
                values = [point.get(field) for point in points if point.get(field)]
                if values:
                    unique_values = list(set(values))
                    value_counts = {val: values.count(val) for val in unique_values}
                    
                    thematic_analysis[field] = {
                        "unique_values": len(unique_values),
                        "most_common": max(value_counts, key=value_counts.get),
                        "diversity": len(unique_values) / len(values) if values else 0,
                        "distribution": value_counts
                    }
            
            return thematic_analysis if thematic_analysis else None
            
        except Exception as e:
            self.logger.debug(f"Thematic analysis failed: {e}")
            return None
    
    def _verify_clustering_dependencies(self) -> None:
        """
        Verify that clustering dependencies are available.
        """
        if CLUSTERING_LIBS_AVAILABLE:
            self.logger.info("Clustering libraries available: hdbscan, scikit-learn, scipy")
        else:
            self.logger.warning("Clustering libraries not available - using simple fallback methods")
    
    async def _process_discovery_queue(self) -> None:
        """
        Process any queued discovery requests.
        """
        # This would integrate with the task queue system
        # For now, this is a placeholder
        pass
    
    async def _cleanup_pattern_cache(self) -> None:
        """
        Clean up old pattern cache entries.
        """
        # Remove cache entries older than 1 hour
        # This is a simplified cleanup - in production you'd check timestamps
        if len(self.pattern_cache) > 100:
            # Keep only the 50 most recent entries
            cache_items = list(self.pattern_cache.items())
            self.pattern_cache = dict(cache_items[-50:])
            self.logger.info("Pattern cache cleaned up")
    
    # Message handlers
    
    async def _handle_discovery_request(self, message: AgentMessage) -> None:
        """
        Handle pattern discovery requests via NATS.
        """
        try:
            request_data = message.payload
            dataset = request_data.get("dataset", {})
            algorithm = request_data.get("algorithm", "hdbscan")
            
            # Perform pattern discovery
            result = await self.discover_patterns(dataset, algorithm)
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="discovery_response",
                payload=result,
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling discovery request: {e}")
    
    async def _handle_parsed_data(self, message: AgentMessage) -> None:
        """
        Handle parsed data for automatic pattern discovery.
        """
        try:
            if message.message_type == "file_parsed":
                parsed_data = message.payload
                parsing_result = parsed_data.get("parsing_result", {})
                
                if parsing_result.get("status") == "success":
                    self.logger.info("Auto-discovering patterns in parsed data")
                    
                    # Create dataset structure
                    dataset = {
                        "id": f"auto-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        "source": parsed_data.get("file_path", "unknown"),
                        "parsing_result": parsing_result
                    }
                    
                    # Discover patterns
                    discovery_result = await self.discover_patterns(dataset)
                    
                    # Publish discovery results
                    if discovery_result.get("pattern_count", 0) > 0:
                        await self.messaging.publish_discovery(discovery_result)
                        
                        self.logger.info(f"Discovered {discovery_result['pattern_count']} patterns in auto-analysis")
            
        except Exception as e:
            self.logger.error(f"Error handling parsed data: {e}")
    
    # Additional analysis methods
    
    async def _perform_spatial_analysis(self, points: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        """
        Perform spatial analysis on points.
        """
        try:
            if not points:
                return {"error": "No points provided"}
            
            if analysis_type == "hotspot":
                return await self._detect_hotspots(points)
            elif analysis_type == "clustering":
                return await self._perform_clustering(points, "hdbscan")
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _detect_hotspots(self, points: List[Dict[str, Any]], intensity_field: str = "value") -> Dict[str, Any]:
        """
        Detect hotspots in spatial point data.
        """
        try:
            if not points:
                return {"hotspots": [], "method": "none"}
            
            # Simple hotspot detection using spatial clustering
            clustering_result = await self._perform_clustering(points, "hdbscan")
            
            hotspots = []
            for cluster in clustering_result.get("clusters", []):
                if cluster["size"] >= self.min_cluster_size:
                    # Calculate hotspot intensity
                    cluster_points = cluster["points"]
                    intensity_values = [p.get(intensity_field, 1) for p in cluster_points]
                    
                    hotspot = {
                        "hotspot_id": f"hotspot_{cluster['cluster_id']}",
                        "centroid": cluster.get("centroid", {}),
                        "point_count": cluster["size"],
                        "intensity_sum": sum(intensity_values),
                        "intensity_avg": np.mean(intensity_values),
                        "bounds": cluster.get("bounds", {}),
                        "points": cluster_points
                    }
                    
                    hotspots.append(hotspot)
            
            return {
                "hotspots": hotspots,
                "hotspot_count": len(hotspots),
                "method": "clustering_based",
                "intensity_field": intensity_field
            }
            
        except Exception as e:
            return {"error": str(e), "hotspots": []}


# Main entry point for running the agent
async def main():
    """
    Main entry point for running the PatternDiscoveryAgent.
    """
    import signal
    import sys
    
    # Create and configure agent
    agent = PatternDiscoveryAgent()
    
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
    
    print("PatternDiscoveryAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())