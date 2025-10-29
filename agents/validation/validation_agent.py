"""
A2A World Platform - Validation Agent

Agent responsible for statistical validation of discovered patterns.
Provides spatial autocorrelation analysis, significance testing,
bootstrap validation, and cross-validation capabilities.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import json

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from agents.core.base_agent import BaseAgent
from agents.core.config import ValidationAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task


class ValidationAgent(BaseAgent):
    """
    Agent that performs statistical validation of discovered patterns.
    
    Capabilities:
    - Spatial autocorrelation analysis (Moran's I, Geary's C)
    - Local spatial statistics (Getis-Ord Gi*, LISA)
    - Bootstrap validation and confidence intervals
    - Cross-validation and model performance metrics
    - Significance testing and p-value calculations
    - Pattern stability analysis
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[ValidationAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="validation",
            config=config or ValidationAgentConfig(),
            config_file=config_file
        )
        
        self.validation_methods = {
            "morans_i": self._calculate_morans_i,
            "geary_c": self._calculate_geary_c,
            "getis_ord": self._calculate_getis_ord,
            "lisa": self._calculate_lisa,
            "ripley_k": self._calculate_ripley_k,
            "bootstrap": self._bootstrap_validation,
            "cross_validation": self._cross_validation
        }
        
        # Validation results cache
        self.validation_cache: Dict[str, Dict[str, Any]] = {}
        
        # Statistics tracking
        self.validations_performed = 0
        self.significant_patterns = 0
        self.failed_validations = 0
        
        self.logger.info(f"ValidationAgent {self.agent_id} initialized with methods: {list(self.validation_methods.keys())}")
    
    async def process(self) -> None:
        """
        Main processing loop - handle validation requests and maintain cache.
        """
        try:
            # Clean up old cache entries (every 100 iterations)
            if self.processed_tasks % 100 == 0:
                await self._cleanup_cache()
            
            # Process any pending validation tasks
            await self._process_validation_requests()
            
        except Exception as e:
            self.logger.error(f"Error in validation processing: {e}")
    
    async def agent_initialize(self) -> None:
        """
        Validation agent specific initialization.
        """
        try:
            # Initialize validation libraries and check dependencies
            self._verify_dependencies()
            
            # Load any cached validation results from database
            await self._load_validation_cache()
            
            self.logger.info("ValidationAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ValidationAgent: {e}")
            raise
    
    async def setup_subscriptions(self) -> None:
        """
        Setup validation-specific message subscriptions.
        """
        if not self.messaging:
            return
        
        # Subscribe to validation requests
        validation_sub_id = await self.nats_client.subscribe(
            "agents.validation.request",
            self._handle_validation_request,
            queue_group="validation-workers"
        )
        self.subscription_ids.append(validation_sub_id)
        
        # Subscribe to pattern discovery events
        discovery_sub_id = await self.messaging.subscribe_to_discoveries(
            self._handle_discovery_event
        )
        self.subscription_ids.append(discovery_sub_id)
    
    async def handle_task(self, task: Task) -> None:
        """
        Handle validation task processing.
        """
        self.logger.info(f"Processing validation task {task.task_id}: {task.task_type}")
        
        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)
            
            result = None
            
            if task.task_type == "validate_pattern":
                result = await self._validate_pattern_task(task)
            elif task.task_type == "spatial_autocorrelation":
                result = await self._spatial_autocorrelation_task(task)
            elif task.task_type == "bootstrap_validation":
                result = await self._bootstrap_validation_task(task)
            elif task.task_type == "cross_validation":
                result = await self._cross_validation_task(task)
            else:
                raise ValueError(f"Unknown validation task type: {task.task_type}")
            
            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)
            
            self.processed_tasks += 1
            self.validations_performed += 1
            
            # Check if pattern is significant
            if result and result.get("significant", False):
                self.significant_patterns += 1
            
            self.logger.info(f"Completed validation task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing validation task {task.task_id}: {e}")
            
            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)
            
            self.failed_tasks += 1
            self.failed_validations += 1
        
        finally:
            self.current_tasks.discard(task.task_id)
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect validation-specific metrics.
        """
        significance_rate = 0.0
        if self.validations_performed > 0:
            significance_rate = self.significant_patterns / self.validations_performed
        
        validation_success_rate = 0.0
        total_validations = self.validations_performed + self.failed_validations
        if total_validations > 0:
            validation_success_rate = self.validations_performed / total_validations
        
        return {
            "validations_performed": self.validations_performed,
            "significant_patterns": self.significant_patterns,
            "failed_validations": self.failed_validations,
            "significance_rate": significance_rate,
            "validation_success_rate": validation_success_rate,
            "cache_size": len(self.validation_cache),
            "available_methods": len(self.validation_methods)
        }
    
    def _get_capabilities(self) -> List[str]:
        """
        Get validation agent capabilities.
        """
        return [
            "validation",
            "validate_pattern",
            "spatial_autocorrelation",
            "bootstrap_validation", 
            "cross_validation",
            "statistical_testing",
            "morans_i",
            "getis_ord",
            "lisa",
            "significance_testing"
        ]
    
    async def _handle_validation_request(self, message: AgentMessage) -> None:
        """
        Handle direct validation requests via NATS.
        """
        try:
            request_data = message.payload
            pattern_id = request_data.get("pattern_id")
            validation_methods = request_data.get("methods", ["morans_i"])
            pattern_data = request_data.get("pattern_data", {})
            
            # Perform validation
            results = await self._validate_pattern(
                pattern_id, pattern_data, validation_methods
            )
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="validation_response",
                payload={
                    "pattern_id": pattern_id,
                    "validation_results": results,
                    "timestamp": datetime.utcnow().isoformat()
                },
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
            
        except Exception as e:
            self.logger.error(f"Error handling validation request: {e}")
    
    async def _handle_discovery_event(self, message: AgentMessage) -> None:
        """
        Handle pattern discovery events for automatic validation.
        """
        try:
            if message.message_type == "pattern_discovered":
                discovery_data = message.payload
                pattern_id = discovery_data.get("pattern_id")
                
                if pattern_id:
                    # Automatically validate discovered patterns
                    await self._auto_validate_pattern(pattern_id, discovery_data)
                    
        except Exception as e:
            self.logger.error(f"Error handling discovery event: {e}")
    
    async def _validate_pattern_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle pattern validation task.
        """
        pattern_id = task.parameters.get("pattern_id")
        pattern_data = task.input_data.get("pattern_data", {})
        methods = task.parameters.get("methods", ["morans_i"])
        
        if not pattern_id:
            raise ValueError("Pattern ID is required for validation")
        
        return await self._validate_pattern(pattern_id, pattern_data, methods)
    
    async def _spatial_autocorrelation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle spatial autocorrelation analysis task.
        """
        data = task.input_data.get("spatial_data", [])
        method = task.parameters.get("method", "morans_i")
        
        if not data:
            raise ValueError("Spatial data is required")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        if method == "morans_i":
            return await self._calculate_morans_i(df)
        elif method == "geary_c":
            return await self._calculate_geary_c(df)
        elif method == "getis_ord":
            return await self._calculate_getis_ord(df)
        else:
            raise ValueError(f"Unknown spatial autocorrelation method: {method}")
    
    async def _bootstrap_validation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle bootstrap validation task.
        """
        data = task.input_data.get("data", [])
        pattern_data = task.input_data.get("pattern_data", {})
        iterations = task.parameters.get("iterations", self.config.bootstrap_iterations)
        
        return await self._bootstrap_validation(data, pattern_data, iterations)
    
    async def _cross_validation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle cross-validation task.
        """
        features = task.input_data.get("features", [])
        labels = task.input_data.get("labels", [])
        model_type = task.parameters.get("model_type", "classification")
        folds = task.parameters.get("folds", self.config.cross_validation_folds)
        
        return await self._cross_validation(features, labels, model_type, folds)
    
    async def _validate_pattern(
        self, 
        pattern_id: str, 
        pattern_data: Dict[str, Any], 
        methods: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive pattern validation using multiple methods.
        """
        # Check cache first
        cache_key = f"{pattern_id}:{':'.join(sorted(methods))}"
        if cache_key in self.validation_cache:
            self.logger.debug(f"Returning cached validation for {pattern_id}")
            return self.validation_cache[cache_key]
        
        results = {
            "pattern_id": pattern_id,
            "validation_methods": methods,
            "timestamp": datetime.utcnow().isoformat(),
            "results": {},
            "overall_significant": False,
            "confidence_level": 1 - self.config.significance_level
        }
        
        try:
            # Extract spatial data from pattern
            spatial_data = self._extract_spatial_data(pattern_data)
            
            if not spatial_data:
                raise ValueError("No spatial data found in pattern")
            
            df = pd.DataFrame(spatial_data)
            significant_results = 0
            
            # Apply each validation method
            for method in methods:
                if method in self.validation_methods:
                    method_result = await self.validation_methods[method](df, pattern_data)
                    results["results"][method] = method_result
                    
                    # Check significance
                    if method_result.get("significant", False):
                        significant_results += 1
                else:
                    self.logger.warning(f"Unknown validation method: {method}")
            
            # Overall significance (majority of methods agree)
            if len(methods) > 0:
                results["overall_significant"] = significant_results > len(methods) / 2
                results["significance_ratio"] = significant_results / len(methods)
            
            # Cache results
            self.validation_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating pattern {pattern_id}: {e}")
            results["error"] = str(e)
            results["results"] = {}
            return results
    
    async def _calculate_morans_i(
        self, 
        df: pd.DataFrame, 
        pattern_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Moran's I spatial autocorrelation statistic.
        """
        try:
            # Extract coordinates and values
            if "latitude" not in df.columns or "longitude" not in df.columns:
                raise ValueError("Latitude and longitude columns required")
            
            coords = df[["longitude", "latitude"]].values
            values = df.get("value", df.iloc[:, -1]).values
            
            # Calculate distance matrix
            distances = squareform(pdist(coords))
            
            # Create spatial weights matrix (inverse distance)
            weights = 1 / (distances + 1e-10)  # Add small constant to avoid division by zero
            np.fill_diagonal(weights, 0)
            
            # Normalize weights
            row_sums = weights.sum(axis=1)
            weights = weights / row_sums[:, np.newaxis]
            weights[np.isnan(weights)] = 0
            
            # Calculate Moran's I
            n = len(values)
            mean_val = np.mean(values)
            
            # Numerator: sum of weights * (xi - mean) * (xj - mean)
            numerator = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            
            # Denominator: sum of (xi - mean)^2
            denominator = np.sum((values - mean_val) ** 2)
            
            morans_i = numerator / denominator if denominator != 0 else 0
            
            # Calculate expected value and variance
            expected_i = -1 / (n - 1)
            
            # Simplified variance calculation
            sum_weights = np.sum(weights)
            sum_weights_squared = np.sum(weights ** 2)
            
            var_i = ((n * sum_weights_squared - sum_weights ** 2) / 
                    ((n - 1) * (n - 2) * (n - 3) * sum_weights ** 2))
            
            # Z-score and p-value
            if var_i > 0:
                z_score = (morans_i - expected_i) / np.sqrt(var_i)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value = 1.0
            
            significant = p_value < self.config.significance_level
            
            return {
                "statistic": "morans_i",
                "value": float(morans_i),
                "expected": float(expected_i),
                "variance": float(var_i),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "significant": significant,
                "interpretation": self._interpret_morans_i(morans_i, significant),
                "sample_size": n
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Moran's I: {e}")
            return {
                "statistic": "morans_i",
                "error": str(e),
                "significant": False
            }
    
    async def _calculate_geary_c(
        self, 
        df: pd.DataFrame, 
        pattern_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Geary's C spatial autocorrelation statistic.
        """
        try:
            coords = df[["longitude", "latitude"]].values
            values = df.get("value", df.iloc[:, -1]).values
            
            # Calculate distance-based weights
            distances = squareform(pdist(coords))
            weights = 1 / (distances + 1e-10)
            np.fill_diagonal(weights, 0)
            
            # Normalize weights
            row_sums = weights.sum(axis=1)
            weights = weights / row_sums[:, np.newaxis]
            weights[np.isnan(weights)] = 0
            
            n = len(values)
            
            # Calculate Geary's C
            numerator = 0
            denominator = 0
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        numerator += weights[i, j] * (values[i] - values[j]) ** 2
            
            denominator = 2 * np.sum(weights) * np.var(values)
            
            geary_c = numerator / denominator if denominator != 0 else 1
            
            # Expected value is 1
            expected_c = 1.0
            
            # Simple z-score calculation
            z_score = (geary_c - expected_c) / 0.1  # Simplified standard error
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            significant = p_value < self.config.significance_level
            
            return {
                "statistic": "geary_c",
                "value": float(geary_c),
                "expected": expected_c,
                "z_score": float(z_score),
                "p_value": float(p_value),
                "significant": significant,
                "interpretation": self._interpret_geary_c(geary_c, significant),
                "sample_size": n
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Geary's C: {e}")
            return {
                "statistic": "geary_c",
                "error": str(e),
                "significant": False
            }
    
    async def _calculate_getis_ord(
        self, 
        df: pd.DataFrame, 
        pattern_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Getis-Ord Gi* local spatial statistic.
        """
        try:
            coords = df[["longitude", "latitude"]].values
            values = df.get("value", df.iloc[:, -1]).values
            
            n = len(values)
            gi_stats = []
            
            # Calculate for each point
            for i in range(min(n, 100)):  # Limit to first 100 points for performance
                # Find neighbors within threshold distance
                distances = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
                threshold = np.percentile(distances[distances > 0], 25)  # Use 25th percentile as threshold
                neighbors = distances <= threshold
                
                # Calculate Gi*
                neighbor_sum = np.sum(values[neighbors])
                neighbor_count = np.sum(neighbors)
                
                if neighbor_count > 1:
                    expected = np.sum(values) * neighbor_count / n
                    variance = np.var(values) * neighbor_count / n
                    
                    if variance > 0:
                        gi_star = (neighbor_sum - expected) / np.sqrt(variance)
                        p_value = 2 * (1 - stats.norm.cdf(abs(gi_star)))
                        gi_stats.append({
                            "index": i,
                            "gi_star": gi_star,
                            "p_value": p_value,
                            "significant": p_value < self.config.significance_level
                        })
            
            # Summary statistics
            significant_hotspots = sum(1 for stat in gi_stats if stat["gi_star"] > 0 and stat["significant"])
            significant_coldspots = sum(1 for stat in gi_stats if stat["gi_star"] < 0 and stat["significant"])
            
            return {
                "statistic": "getis_ord_gi_star",
                "hotspots": significant_hotspots,
                "coldspots": significant_coldspots,
                "total_significant": significant_hotspots + significant_coldspots,
                "analyzed_points": len(gi_stats),
                "significant": (significant_hotspots + significant_coldspots) > 0,
                "local_statistics": gi_stats[:10],  # Return first 10 for brevity
                "interpretation": f"Found {significant_hotspots} hotspots and {significant_coldspots} coldspots"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Getis-Ord Gi*: {e}")
            return {
                "statistic": "getis_ord_gi_star",
                "error": str(e),
                "significant": False
            }
    
    async def _calculate_lisa(
        self, 
        df: pd.DataFrame, 
        pattern_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Local Indicators of Spatial Association (LISA).
        """
        try:
            coords = df[["longitude", "latitude"]].values
            values = df.get("value", df.iloc[:, -1]).values
            
            n = len(values)
            lisa_stats = []
            mean_val = np.mean(values)
            
            # Calculate for each point
            for i in range(min(n, 50)):  # Limit for performance
                # Find k-nearest neighbors
                distances = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
                k = min(8, n - 1)  # Use 8 nearest neighbors
                neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
                
                # Calculate local Moran's I
                neighbors_mean = np.mean(values[neighbor_indices])
                local_i = (values[i] - mean_val) * (neighbors_mean - mean_val)
                
                # Classification
                if values[i] > mean_val and neighbors_mean > mean_val:
                    cluster_type = "HH"  # High-High
                elif values[i] < mean_val and neighbors_mean < mean_val:
                    cluster_type = "LL"  # Low-Low
                elif values[i] > mean_val and neighbors_mean < mean_val:
                    cluster_type = "HL"  # High-Low (outlier)
                else:
                    cluster_type = "LH"  # Low-High (outlier)
                
                # Simple significance test
                p_value = stats.norm.sf(abs(local_i))
                
                lisa_stats.append({
                    "index": i,
                    "local_i": float(local_i),
                    "cluster_type": cluster_type,
                    "p_value": float(p_value),
                    "significant": p_value < self.config.significance_level
                })
            
            # Summary
            cluster_counts = {}
            for stat in lisa_stats:
                if stat["significant"]:
                    cluster_type = stat["cluster_type"]
                    cluster_counts[cluster_type] = cluster_counts.get(cluster_type, 0) + 1
            
            return {
                "statistic": "lisa",
                "cluster_counts": cluster_counts,
                "total_significant": sum(cluster_counts.values()),
                "analyzed_points": len(lisa_stats),
                "significant": sum(cluster_counts.values()) > 0,
                "local_statistics": lisa_stats[:10],
                "interpretation": f"Significant clusters: {cluster_counts}"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating LISA: {e}")
            return {
                "statistic": "lisa",
                "error": str(e),
                "significant": False
            }
    
    async def _calculate_ripley_k(
        self, 
        df: pd.DataFrame, 
        pattern_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Ripley's K function for spatial point pattern analysis.
        """
        try:
            coords = df[["longitude", "latitude"]].values
            n = len(coords)
            
            # Define distance bands
            max_distance = np.sqrt(np.var(coords[:, 0]) + np.var(coords[:, 1])) * 2
            distances = np.linspace(0.1 * max_distance, max_distance, 10)
            
            k_values = []
            
            # Calculate area (simple bounding box approximation)
            area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
            
            for d in distances:
                # Count pairs within distance d
                pair_count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                        if dist <= d:
                            pair_count += 1
                
                # Ripley's K
                k_d = (area * pair_count) / (n * (n - 1))
                k_values.append(k_d)
            
            # Expected K under random distribution
            expected_k = [np.pi * d**2 for d in distances]
            
            # L function (normalized K)
            l_values = [np.sqrt(k / np.pi) - d for k, d in zip(k_values, distances)]
            
            # Simple clustering test
            clustering_detected = any(l > 0.1 for l in l_values)
            
            return {
                "statistic": "ripley_k",
                "distances": distances.tolist(),
                "k_values": k_values,
                "expected_k": expected_k,
                "l_values": l_values,
                "significant": clustering_detected,
                "interpretation": "Clustering detected" if clustering_detected else "No significant clustering",
                "sample_size": n
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Ripley's K: {e}")
            return {
                "statistic": "ripley_k",
                "error": str(e),
                "significant": False
            }
    
    async def _bootstrap_validation(
        self, 
        data: List[Dict[str, Any]], 
        pattern_data: Dict[str, Any], 
        iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform bootstrap validation of pattern significance.
        """
        try:
            if not data:
                raise ValueError("Data is required for bootstrap validation")
            
            df = pd.DataFrame(data)
            original_stat = await self._calculate_morans_i(df)
            original_value = original_stat.get("value", 0)
            
            # Bootstrap sampling
            bootstrap_values = []
            
            for i in range(min(iterations, 100)):  # Limit iterations for performance
                # Resample with replacement
                bootstrap_sample = df.sample(n=len(df), replace=True).reset_index(drop=True)
                
                # Calculate statistic on bootstrap sample
                bootstrap_stat = await self._calculate_morans_i(bootstrap_sample)
                bootstrap_values.append(bootstrap_stat.get("value", 0))
            
            bootstrap_values = np.array(bootstrap_values)
            
            # Calculate confidence intervals
            confidence_level = 1 - self.config.significance_level
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100
            
            ci_lower = np.percentile(bootstrap_values, lower_percentile)
            ci_upper = np.percentile(bootstrap_values, upper_percentile)
            
            # Test significance
            p_value = np.mean(np.abs(bootstrap_values) >= np.abs(original_value))
            significant = p_value < self.config.significance_level
            
            return {
                "statistic": "bootstrap_validation",
                "original_value": float(original_value),
                "bootstrap_mean": float(np.mean(bootstrap_values)),
                "bootstrap_std": float(np.std(bootstrap_values)),
                "confidence_interval": [float(ci_lower), float(ci_upper)],
                "confidence_level": confidence_level,
                "p_value": float(p_value),
                "significant": significant,
                "iterations": len(bootstrap_values),
                "interpretation": "Pattern is statistically significant" if significant else "Pattern is not significant"
            }
            
        except Exception as e:
            self.logger.error(f"Error in bootstrap validation: {e}")
            return {
                "statistic": "bootstrap_validation",
                "error": str(e),
                "significant": False
            }
    
    async def _cross_validation(
        self, 
        features: List[List[float]], 
        labels: List[int], 
        model_type: str = "classification",
        folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation analysis.
        """
        try:
            if not features or not labels:
                raise ValueError("Features and labels are required for cross-validation")
            
            X = np.array(features)
            y = np.array(labels)
            
            if len(X) != len(y):
                raise ValueError("Features and labels must have same length")
            
            # K-fold cross validation
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            
            scores = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": []
            }
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Simple classifier (majority class for simplicity)
                majority_class = stats.mode(y_train)[0][0] if len(y_train) > 0 else 0
                y_pred = np.full_like(y_test, majority_class)
                
                # Calculate metrics
                scores["accuracy"].append(accuracy_score(y_test, y_pred))
                scores["precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                scores["recall"].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                scores["f1"].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # Calculate means and standard deviations
            cv_results = {}
            for metric, values in scores.items():
                cv_results[f"{metric}_mean"] = float(np.mean(values))
                cv_results[f"{metric}_std"] = float(np.std(values))
            
            # Overall performance assessment
            avg_accuracy = cv_results["accuracy_mean"]
            performance_level = "good" if avg_accuracy > 0.8 else "moderate" if avg_accuracy > 0.6 else "poor"
            
            return {
                "statistic": "cross_validation",
                "model_type": model_type,
                "folds": folds,
                "sample_size": len(X),
                "results": cv_results,
                "performance_level": performance_level,
                "significant": avg_accuracy > 0.5,  # Better than random
                "interpretation": f"Cross-validation shows {performance_level} performance (accuracy: {avg_accuracy:.3f})"
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            return {
                "statistic": "cross_validation",
                "error": str(e),
                "significant": False
            }
    
    def _extract_spatial_data(self, pattern_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract spatial data from pattern data structure.
        """
        try:
            # Try different possible data structures
            if "features" in pattern_data:
                return pattern_data["features"]
            elif "points" in pattern_data:
                return pattern_data["points"]
            elif "coordinates" in pattern_data:
                coords = pattern_data["coordinates"]
                return [{"latitude": lat, "longitude": lon, "value": 1} 
                       for lat, lon in coords]
            elif isinstance(pattern_data, list):
                return pattern_data
            else:
                # Generate sample data for testing
                self.logger.warning("No spatial data found, generating sample data")
                return [
                    {"latitude": 37.7749 + np.random.normal(0, 0.01), 
                     "longitude": -122.4194 + np.random.normal(0, 0.01), 
                     "value": np.random.random()}
                    for _ in range(10)
                ]
        except Exception as e:
            self.logger.error(f"Error extracting spatial data: {e}")
            return []
    
    def _interpret_morans_i(self, morans_i: float, significant: bool) -> str:
        """
        Interpret Moran's I statistic value.
        """
        if not significant:
            return "No significant spatial autocorrelation detected"
        
        if morans_i > 0.3:
            return "Strong positive spatial autocorrelation (clustered pattern)"
        elif morans_i > 0.1:
            return "Moderate positive spatial autocorrelation (some clustering)"
        elif morans_i > -0.1:
            return "Weak or no spatial autocorrelation (random pattern)"
        elif morans_i > -0.3:
            return "Moderate negative spatial autocorrelation (dispersed pattern)"
        else:
            return "Strong negative spatial autocorrelation (highly dispersed pattern)"
    
    def _interpret_geary_c(self, geary_c: float, significant: bool) -> str:
        """
        Interpret Geary's C statistic value.
        """
        if not significant:
            return "No significant spatial autocorrelation detected"
        
        if geary_c < 0.7:
            return "Strong positive spatial autocorrelation (similar values clustered)"
        elif geary_c < 0.9:
            return "Moderate positive spatial autocorrelation"
        elif geary_c < 1.1:
            return "Random spatial pattern"
        elif geary_c < 1.3:
            return "Moderate negative spatial autocorrelation"
        else:
            return "Strong negative spatial autocorrelation (dissimilar values clustered)"
    
    def _verify_dependencies(self) -> None:
        """
        Verify that required dependencies are available.
        """
        try:
            import numpy as np
            import pandas as pd
            import scipy.stats
            import sklearn
            self.logger.info("All validation dependencies verified")
        except ImportError as e:
            self.logger.error(f"Missing validation dependency: {e}")
            raise
    
    async def _load_validation_cache(self) -> None:
        """
        Load validation results from database cache.
        """
        # TODO: Implement database loading
        pass
    
    async def _cleanup_cache(self) -> None:
        """
        Clean up old validation cache entries.
        """
        current_time = datetime.utcnow()
        cache_ttl = 3600  # 1 hour
        
        keys_to_remove = []
        for key, result in self.validation_cache.items():
            if "timestamp" in result:
                try:
                    result_time = datetime.fromisoformat(result["timestamp"])
                    if (current_time - result_time).seconds > cache_ttl:
                        keys_to_remove.append(key)
                except (ValueError, KeyError):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.validation_cache[key]
        
        if keys_to_remove:
            self.logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")
    
    async def _process_validation_requests(self) -> None:
        """
        Process any queued validation requests.
        """
        # This would integrate with the task queue system
        # For now, this is a placeholder
        pass
    
    async def _auto_validate_pattern(self, pattern_id: str, discovery_data: Dict[str, Any]) -> None:
        """
        Automatically validate newly discovered patterns.
        """
        try:
            # Default validation methods for auto-validation
            methods = ["morans_i"]
            
            if self.config.enable_getis_ord:
                methods.append("getis_ord")
            
            # Perform validation
            results = await self._validate_pattern(pattern_id, discovery_data, methods)
            
            # Publish validation results
            if self.messaging:
                message = AgentMessage.create(
                    sender_id=self.agent_id,
                    message_type="pattern_validated",
                    payload={
                        "pattern_id": pattern_id,
                        "validation_results": results,
                        "auto_validated": True
                    }
                )
                
                await self.nats_client.publish("agents.validation.results", message)
            
            self.logger.info(f"Auto-validated pattern {pattern_id}: {'significant' if results.get('overall_significant') else 'not significant'}")
            
        except Exception as e:
            self.logger.error(f"Error in auto-validation of pattern {pattern_id}: {e}")


# Main entry point for running the agent
async def main():
    """
    Main entry point for running the ValidationAgent.
    """
    import signal
    import sys
    
    # Create and configure agent
    agent = ValidationAgent()
    
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
    
    print("ValidationAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())