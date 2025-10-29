"""
A2A World Platform - Clustering Utilities

Optimized clustering algorithms for geospatial pattern discovery with HDBSCAN,
spatial statistics, and pattern significance testing.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import warnings

try:
    import hdbscan
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors
    import scipy.spatial.distance as distance
    from scipy import stats
    CLUSTERING_LIBS_AVAILABLE = True
except ImportError as e:
    CLUSTERING_LIBS_AVAILABLE = False
    warnings.warn(f"Clustering libraries not available: {e}")

try:
    from geopy.distance import geodesic
    import shapely.geometry as geom
    GEOSPATIAL_LIBS_AVAILABLE = True
except ImportError:
    GEOSPATIAL_LIBS_AVAILABLE = False


class GeospatialHDBSCAN:
    """
    Geospatially-optimized HDBSCAN clustering for sacred sites and cultural landmarks.
    """
    
    def __init__(self, 
                 min_cluster_size: int = 5,
                 min_samples: int = 3,
                 metric: str = 'euclidean',
                 cluster_selection_epsilon: float = 0.0,
                 alpha: float = 1.0):
        """
        Initialize GeospatialHDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood for core points
            metric: Distance metric for clustering
            cluster_selection_epsilon: Distance threshold for cluster extraction
            alpha: Alpha parameter for cluster selection
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.alpha = alpha
        
        self.logger = logging.getLogger(__name__)
        
        # Clustering results
        self.labels_ = None
        self.probabilities_ = None
        self.outlier_scores_ = None
        self.cluster_persistence_ = None
        self.condensed_tree_ = None
        
    def fit_predict(self, X: np.ndarray, 
                    coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform HDBSCAN clustering on the data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            coordinates: Optional lat/lon coordinates for geospatial weighting
            
        Returns:
            Cluster labels array
        """
        try:
            if not CLUSTERING_LIBS_AVAILABLE:
                return self._fallback_clustering(X)
            
            # Prepare data for clustering
            X_processed = self._preprocess_features(X, coordinates)
            
            # Create and fit HDBSCAN clusterer
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                alpha=self.alpha,
                cluster_selection_method='eom'  # Excess of Mass
            )
            
            # Perform clustering
            cluster_labels = clusterer.fit_predict(X_processed)
            
            # Store results
            self.labels_ = cluster_labels
            self.probabilities_ = getattr(clusterer, 'probabilities_', None)
            self.outlier_scores_ = getattr(clusterer, 'outlier_scores_', None)
            self.cluster_persistence_ = getattr(clusterer, 'cluster_persistence_', None)
            self.condensed_tree_ = getattr(clusterer, 'condensed_tree_', None)
            
            self.logger.info(f"HDBSCAN clustering completed: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters, "
                           f"{np.sum(cluster_labels == -1)} noise points")
            
            return cluster_labels
            
        except Exception as e:
            self.logger.error(f"HDBSCAN clustering failed: {e}")
            return self._fallback_clustering(X)
    
    def _preprocess_features(self, X: np.ndarray, 
                           coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess features for geospatial clustering.
        
        Args:
            X: Feature matrix
            coordinates: Optional lat/lon coordinates
            
        Returns:
            Processed feature matrix
        """
        try:
            # Handle missing values
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            if X.shape[1] > 1:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.copy()
            
            # Add geospatial weighting if coordinates provided
            if coordinates is not None and GEOSPATIAL_LIBS_AVAILABLE:
                X_scaled = self._add_geospatial_features(X_scaled, coordinates)
            
            return X_scaled
            
        except Exception as e:
            self.logger.warning(f"Feature preprocessing failed: {e}")
            return X
    
    def _add_geospatial_features(self, X: np.ndarray, 
                               coordinates: np.ndarray) -> np.ndarray:
        """
        Add geospatial distance features to the feature matrix.
        
        Args:
            X: Existing feature matrix
            coordinates: Lat/lon coordinates (n_samples, 2)
            
        Returns:
            Enhanced feature matrix with geospatial features
        """
        try:
            # Calculate pairwise geodesic distances
            n_samples = coordinates.shape[0]
            distance_features = []
            
            # Calculate distance to geographic center
            center_lat = np.mean(coordinates[:, 0])
            center_lon = np.mean(coordinates[:, 1])
            center = (center_lat, center_lon)
            
            for i in range(n_samples):
                point = (coordinates[i, 0], coordinates[i, 1])
                dist = geodesic(point, center).kilometers
                distance_features.append(dist)
            
            # Normalize distance features
            distance_features = np.array(distance_features).reshape(-1, 1)
            scaler = MinMaxScaler()
            distance_scaled = scaler.fit_transform(distance_features)
            
            # Combine with existing features
            X_enhanced = np.hstack([X, distance_scaled])
            
            return X_enhanced
            
        except Exception as e:
            self.logger.warning(f"Geospatial feature addition failed: {e}")
            return X
    
    def _fallback_clustering(self, X: np.ndarray) -> np.ndarray:
        """
        Fallback clustering when HDBSCAN is not available.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        try:
            if X.shape[0] < self.min_cluster_size:
                return np.zeros(X.shape[0], dtype=int)
            
            # Use simple distance-based clustering
            distances = distance.pdist(X)
            threshold = np.percentile(distances, 25)
            
            labels = np.arange(X.shape[0])
            
            # Simple agglomerative clustering
            for i in range(X.shape[0]):
                for j in range(i + 1, X.shape[0]):
                    if np.linalg.norm(X[i] - X[j]) < threshold:
                        labels[labels == labels[j]] = labels[i]
            
            # Renumber labels
            unique_labels = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            
            return np.array([label_map[label] for label in labels])
            
        except Exception as e:
            self.logger.error(f"Fallback clustering failed: {e}")
            return np.zeros(X.shape[0], dtype=int)


class SpatialStatistics:
    """
    Spatial statistics calculations for pattern significance assessment.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_spatial_autocorrelation(self, points: List[Dict[str, Any]], 
                                        values: Optional[List[float]] = None,
                                        k_neighbors: int = 8) -> Dict[str, float]:
        """
        Calculate Moran's I spatial autocorrelation statistic.
        
        Args:
            points: List of point dictionaries with lat/lon
            values: Optional values for each point (uses significance_level if None)
            k_neighbors: Number of neighbors for spatial weights
            
        Returns:
            Dictionary with Moran's I statistics
        """
        try:
            if len(points) < 3:
                return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0}
            
            # Extract coordinates
            coordinates = []
            for point in points:
                if "latitude" in point and "longitude" in point:
                    coordinates.append([point["latitude"], point["longitude"]])
            
            if len(coordinates) < 3:
                return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0}
            
            coordinates = np.array(coordinates)
            
            # Use values or default to significance levels
            if values is None:
                values = [point.get("significance_level", 1.0) for point in points]
            
            values = np.array(values)
            
            # Calculate spatial weights matrix using k-nearest neighbors
            weights_matrix = self._calculate_spatial_weights(coordinates, k_neighbors)
            
            # Calculate Moran's I
            morans_i = self._calculate_morans_i(values, weights_matrix)
            
            # Calculate expected value and variance
            n = len(values)
            expected_i = -1.0 / (n - 1)
            
            # Simplified variance calculation
            variance_i = (2.0 / ((n - 1) * (n - 2))) * (1.0 / n)
            
            # Z-score and p-value
            z_score = (morans_i - expected_i) / np.sqrt(variance_i) if variance_i > 0 else 0.0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if z_score != 0 else 1.0
            
            return {
                "morans_i": float(morans_i),
                "expected_i": float(expected_i),
                "variance_i": float(variance_i),
                "z_score": float(z_score),
                "p_value": float(p_value)
            }
            
        except Exception as e:
            self.logger.error(f"Spatial autocorrelation calculation failed: {e}")
            return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0}
    
    def calculate_nearest_neighbor_statistic(self, points: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate nearest neighbor analysis statistics.
        
        Args:
            points: List of point dictionaries with lat/lon
            
        Returns:
            Dictionary with nearest neighbor statistics
        """
        try:
            if len(points) < 2:
                return {"nn_ratio": 1.0, "z_score": 0.0, "p_value": 1.0}
            
            # Extract coordinates
            coordinates = []
            for point in points:
                if "latitude" in point and "longitude" in point:
                    coordinates.append([point["latitude"], point["longitude"]])
            
            if len(coordinates) < 2:
                return {"nn_ratio": 1.0, "z_score": 0.0, "p_value": 1.0}
            
            coordinates = np.array(coordinates)
            
            # Calculate nearest neighbor distances
            nn_distances = []
            for i, point in enumerate(coordinates):
                distances = [np.linalg.norm(point - other) for j, other in enumerate(coordinates) if i != j]
                nn_distances.append(min(distances))
            
            # Observed mean nearest neighbor distance
            observed_mean = np.mean(nn_distances)
            
            # Expected mean for random distribution
            n = len(coordinates)
            # Approximate area (bounding box)
            min_lat, max_lat = coordinates[:, 0].min(), coordinates[:, 0].max()
            min_lon, max_lon = coordinates[:, 1].min(), coordinates[:, 1].max()
            area = (max_lat - min_lat) * (max_lon - min_lon) * 111.32 * 111.32  # Rough km²
            
            expected_mean = 0.5 * np.sqrt(area / n) if area > 0 and n > 0 else 1.0
            
            # Nearest neighbor ratio
            nn_ratio = observed_mean / expected_mean if expected_mean > 0 else 1.0
            
            # Z-score (simplified)
            std_error = 0.26136 / np.sqrt(n * n / area) if area > 0 and n > 0 else 1.0
            z_score = (observed_mean - expected_mean) / std_error if std_error > 0 else 0.0
            
            # P-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if z_score != 0 else 1.0
            
            return {
                "nn_ratio": float(nn_ratio),
                "observed_mean": float(observed_mean),
                "expected_mean": float(expected_mean),
                "z_score": float(z_score),
                "p_value": float(p_value)
            }
            
        except Exception as e:
            self.logger.error(f"Nearest neighbor calculation failed: {e}")
            return {"nn_ratio": 1.0, "z_score": 0.0, "p_value": 1.0}
    
    def _calculate_spatial_weights(self, coordinates: np.ndarray, 
                                 k_neighbors: int = 8) -> np.ndarray:
        """
        Calculate spatial weights matrix using k-nearest neighbors.
        
        Args:
            coordinates: Array of coordinate pairs
            k_neighbors: Number of neighbors
            
        Returns:
            Spatial weights matrix
        """
        try:
            n = coordinates.shape[0]
            k = min(k_neighbors, n - 1)
            
            # Use nearest neighbors to find spatial relationships
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates)
            
            # Create weights matrix
            weights = np.zeros((n, n))
            
            for i in range(n):
                for j in range(1, k + 1):  # Skip self (index 0)
                    neighbor_idx = indices[i, j]
                    # Use inverse distance weighting
                    weight = 1.0 / (distances[i, j] + 1e-10)
                    weights[i, neighbor_idx] = weight
            
            # Row standardize weights
            row_sums = weights.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            weights = weights / row_sums[:, np.newaxis]
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Spatial weights calculation failed: {e}")
            return np.eye(coordinates.shape[0])
    
    def _calculate_morans_i(self, values: np.ndarray, 
                          weights: np.ndarray) -> float:
        """
        Calculate Moran's I statistic.
        
        Args:
            values: Array of values
            weights: Spatial weights matrix
            
        Returns:
            Moran's I statistic
        """
        try:
            n = len(values)
            if n == 0:
                return 0.0
            
            # Center the values
            mean_val = np.mean(values)
            centered = values - mean_val
            
            # Calculate numerator and denominator
            numerator = 0.0
            denominator = np.sum(centered**2)
            
            W = np.sum(weights)  # Sum of all weights
            
            if W == 0 or denominator == 0:
                return 0.0
            
            for i in range(n):
                for j in range(n):
                    numerator += weights[i, j] * centered[i] * centered[j]
            
            morans_i = (n / W) * (numerator / denominator)
            
            return morans_i
            
        except Exception as e:
            self.logger.error(f"Moran's I calculation failed: {e}")
            return 0.0


class PatternSignificanceTest:
    """
    Statistical significance testing for discovered patterns.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize pattern significance tester.
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
    
    def assess_cluster_significance(self, cluster_data: Dict[str, Any], 
                                  all_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the statistical significance of a discovered cluster.
        
        Args:
            cluster_data: Cluster information from clustering
            all_points: All points in the dataset for comparison
            
        Returns:
            Significance assessment results
        """
        try:
            cluster_points = cluster_data.get("points", [])
            cluster_size = len(cluster_points)
            total_points = len(all_points)
            
            if cluster_size < 3 or total_points < 10:
                return {
                    "significant": False,
                    "significance_score": 0.0,
                    "p_value": 1.0,
                    "test_results": {"error": "Insufficient data for significance testing"}
                }
            
            # Multiple significance tests
            test_results = {}
            significance_scores = []
            
            # 1. Size-based significance (cluster larger than expected by chance)
            expected_cluster_size = total_points * 0.1  # 10% baseline
            size_z_score = (cluster_size - expected_cluster_size) / np.sqrt(expected_cluster_size) if expected_cluster_size > 0 else 0
            size_p_value = 1 - stats.norm.cdf(size_z_score) if size_z_score > 0 else 1.0
            
            test_results["size_test"] = {
                "z_score": float(size_z_score),
                "p_value": float(size_p_value),
                "significant": size_p_value < self.significance_level
            }
            
            if size_p_value < self.significance_level:
                significance_scores.append(0.3)
            
            # 2. Spatial concentration test
            spatial_stats = SpatialStatistics()
            spatial_test = spatial_stats.calculate_nearest_neighbor_statistic(cluster_points)
            
            test_results["spatial_concentration"] = spatial_test
            if spatial_test.get("p_value", 1.0) < self.significance_level and spatial_test.get("nn_ratio", 1.0) < 1.0:
                significance_scores.append(0.4)
            
            # 3. Compactness test
            compactness = cluster_data.get("compactness", 0.5)
            if compactness > 0.7:  # High compactness threshold
                significance_scores.append(0.3)
                test_results["compactness_test"] = {
                    "compactness": compactness,
                    "significant": True
                }
            else:
                test_results["compactness_test"] = {
                    "compactness": compactness,
                    "significant": False
                }
            
            # Overall significance assessment
            overall_score = sum(significance_scores)
            is_significant = overall_score >= 0.5  # At least moderate significance
            
            # Combined p-value (Fisher's method approximation)
            p_values = [test_results.get("size_test", {}).get("p_value", 1.0),
                       test_results.get("spatial_concentration", {}).get("p_value", 1.0)]
            combined_p = stats.combine_pvalues(p_values, method='fisher')[1] if len(p_values) > 1 else min(p_values)
            
            return {
                "significant": is_significant,
                "significance_score": overall_score,
                "p_value": float(combined_p),
                "confidence_level": 1.0 - combined_p,
                "test_results": test_results,
                "assessment_method": "multi_test_combination"
            }
            
        except Exception as e:
            self.logger.error(f"Cluster significance assessment failed: {e}")
            return {
                "significant": False,
                "significance_score": 0.0,
                "p_value": 1.0,
                "test_results": {"error": str(e)}
            }
    
    def bootstrap_cluster_stability(self, points: List[Dict[str, Any]], 
                                  cluster_labels: np.ndarray,
                                  n_bootstrap: int = 100) -> Dict[str, float]:
        """
        Test cluster stability using bootstrap resampling.
        
        Args:
            points: Original data points
            cluster_labels: Original cluster labels
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Stability metrics
        """
        try:
            if len(points) < 10 or len(set(cluster_labels)) < 2:
                return {"stability_score": 0.0, "consistency_ratio": 0.0}
            
            # Prepare data matrix
            features = []
            for point in points:
                if "latitude" in point and "longitude" in point:
                    features.append([point["latitude"], point["longitude"]])
            
            if len(features) < len(points):
                return {"stability_score": 0.0, "consistency_ratio": 0.0}
            
            X = np.array(features)
            n_samples = X.shape[0]
            
            # Bootstrap resampling
            stability_scores = []
            
            clusterer = GeospatialHDBSCAN(
                min_cluster_size=max(3, len(points) // 20),
                min_samples=3
            )
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[bootstrap_indices]
                
                # Cluster bootstrap sample
                bootstrap_labels = clusterer.fit_predict(X_bootstrap)
                
                # Calculate similarity with original clustering
                # (This is simplified - in practice would use adjusted rand index)
                n_clusters_orig = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_clusters_boot = len(set(bootstrap_labels)) - (1 if -1 in bootstrap_labels else 0)
                
                cluster_similarity = 1.0 - abs(n_clusters_orig - n_clusters_boot) / max(n_clusters_orig, n_clusters_boot, 1)
                stability_scores.append(cluster_similarity)
            
            # Calculate stability metrics
            mean_stability = np.mean(stability_scores)
            consistency_ratio = np.sum(np.array(stability_scores) > 0.8) / n_bootstrap
            
            return {
                "stability_score": float(mean_stability),
                "consistency_ratio": float(consistency_ratio),
                "bootstrap_samples": n_bootstrap
            }
            
        except Exception as e:
            self.logger.error(f"Bootstrap stability test failed: {e}")
            return {"stability_score": 0.0, "consistency_ratio": 0.0}