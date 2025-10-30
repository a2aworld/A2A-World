"""
A2A World Platform - MARL Environment for HDBSCAN Parameter Optimization

Reinforcement Learning environment for optimizing HDBSCAN clustering parameters
using multi-agent reinforcement learning (MARL) techniques.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass

try:
    import hdbscan
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

from agents.discovery.clustering import GeospatialHDBSCAN, SpatialStatistics, PatternSignificanceTest


@dataclass
class ClusteringMetrics:
    """Container for clustering performance metrics."""
    silhouette_score: float = -1.0
    calinski_harabasz_score: float = 0.0
    davies_bouldin_score: float = float('inf')
    num_clusters: int = 0
    noise_ratio: float = 1.0
    pattern_quality_score: float = 0.0
    spatial_coherence: float = 0.0
    significance_score: float = 0.0


class HDBSCANOptimizationEnv(gym.Env):
    """
    Multi-Agent Reinforcement Learning environment for HDBSCAN parameter optimization.

    This environment allows MARL agents to learn optimal HDBSCAN parameters:
    - min_samples: Controls how conservative clustering is
    - min_cluster_size: Minimum size of clusters
    - cluster_selection_epsilon: Distance threshold for cluster extraction

    State space includes:
    - Current parameter values
    - Dataset characteristics (size, dimensionality, spatial distribution)
    - Previous clustering performance metrics

    Action space:
    - Discrete actions for parameter adjustments
    - Continuous actions for fine-tuning parameters

    Reward function based on:
    - Clustering quality metrics (silhouette, CH, DB scores)
    - Pattern significance and spatial coherence
    - Stability and reproducibility
    """

    def __init__(self,
                 dataset: List[Dict[str, Any]],
                 max_steps: int = 100,
                 parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize the HDBSCAN optimization environment.

        Args:
            dataset: List of data points for clustering
            max_steps: Maximum steps per episode
            parameter_ranges: Custom parameter ranges (min, max) for each parameter
        """
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.dataset = dataset
        self.max_steps = max_steps
        self.current_step = 0

        # Default parameter ranges
        self.parameter_ranges = parameter_ranges or {
            'min_samples': (1, 50),
            'min_cluster_size': (2, 100),
            'cluster_selection_epsilon': (0.0, 2.0)
        }

        # Extract features from dataset
        self.features, self.coordinates = self._extract_features(dataset)

        # Initialize clustering utilities
        self.spatial_stats = SpatialStatistics()
        self.significance_tester = PatternSignificanceTest()

        # Define action and observation spaces
        self._define_spaces()

        # Current state
        self.current_params = self._get_default_params()
        self.previous_metrics = ClusteringMetrics()
        self.best_metrics = ClusteringMetrics()
        self.best_params = self.current_params.copy()

        # Episode tracking
        self.episode_rewards = []
        self.parameter_history = []

        self.logger.info(f"HDBSCAN Optimization Environment initialized with {len(dataset)} data points")

    def _define_spaces(self):
        """Define Gymnasium action and observation spaces."""

        # Action space: Discrete parameter adjustments
        # Actions: 0=decrease_min_samples, 1=increase_min_samples,
        #          2=decrease_min_cluster_size, 3=increase_min_cluster_size,
        #          4=decrease_epsilon, 5=increase_epsilon
        self.action_space = spaces.Discrete(6)

        # Observation space: Current parameters + performance metrics + dataset features
        # Parameters: min_samples, min_cluster_size, cluster_selection_epsilon (normalized)
        # Metrics: silhouette, ch_score, db_score, num_clusters, noise_ratio, pattern_quality
        # Dataset: size, dimensionality, spatial_extent
        obs_low = np.array([
            0.0, 0.0, 0.0,  # normalized parameters
            -1.0, 0.0, 0.0,  # metrics (silhouette can be -1)
            0.0, 0.0, 0.0,   # more metrics
            0.0, 0.0, 0.0    # dataset features
        ])

        obs_high = np.array([
            1.0, 1.0, 1.0,   # normalized parameters
            1.0, 1000.0, 10.0,  # metrics
            1.0, 1.0, 1.0,    # more metrics
            10000.0, 100.0, 1000.0  # dataset features
        ])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _extract_features(self, dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract numerical features and coordinates from dataset."""

        if not dataset:
            return np.array([]), None

        # Extract coordinates
        coordinates = []
        for point in dataset:
            if 'latitude' in point and 'longitude' in point:
                coordinates.append([point['latitude'], point['longitude']])

        coordinates = np.array(coordinates) if coordinates else None

        # Extract numerical features
        features = []
        for point in dataset:
            feature_vector = []
            # Always include coordinates if available
            if coordinates is not None:
                feature_vector.extend([point.get('latitude', 0), point.get('longitude', 0)])

            # Add other numerical features
            for key, value in point.items():
                if key not in ['latitude', 'longitude', 'id', 'name', 'description'] and isinstance(value, (int, float)):
                    feature_vector.append(float(value))

            features.append(feature_vector)

        # Ensure consistent feature dimensions
        if features:
            max_features = max(len(f) for f in features)
            for i, f in enumerate(features):
                while len(f) < max_features:
                    f.append(0.0)
                features[i] = f[:max_features]  # Truncate if too long

        return np.array(features), coordinates

    def _get_default_params(self) -> Dict[str, float]:
        """Get default HDBSCAN parameters."""
        n_samples = len(self.dataset)
        return {
            'min_samples': max(1, n_samples // 100),  # 1% of dataset size
            'min_cluster_size': max(2, n_samples // 50),  # 2% of dataset size
            'cluster_selection_epsilon': 0.0
        }

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            value = params[param_name]
            normalized_val = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            normalized.append(np.clip(normalized_val, 0.0, 1.0))
        return np.array(normalized)

    def _evaluate_clustering(self, params: Dict[str, float]) -> ClusteringMetrics:
        """Evaluate clustering performance with given parameters."""

        if not CLUSTERING_AVAILABLE or self.features.size == 0:
            return ClusteringMetrics()

        try:
            # Perform clustering
            clusterer = GeospatialHDBSCAN(
                min_cluster_size=int(params['min_cluster_size']),
                min_samples=int(params['min_samples']),
                cluster_selection_epsilon=params['cluster_selection_epsilon']
            )

            cluster_labels = clusterer.fit_predict(self.features, self.coordinates)

            # Calculate basic metrics
            metrics = ClusteringMetrics()

            # Filter out noise points for quality metrics
            valid_labels = cluster_labels[cluster_labels != -1]
            valid_features = self.features[cluster_labels != -1]

            if len(valid_labels) > 1 and len(set(valid_labels)) > 1:
                try:
                    metrics.silhouette_score = silhouette_score(valid_features, valid_labels)
                except:
                    metrics.silhouette_score = -1.0

                try:
                    metrics.calinski_harabasz_score = calinski_harabasz_score(valid_features, valid_labels)
                except:
                    metrics.calinski_harabasz_score = 0.0

                try:
                    metrics.davies_bouldin_score = davies_bouldin_score(valid_features, valid_labels)
                except:
                    metrics.davies_bouldin_score = float('inf')

            # Basic cluster statistics
            unique_labels = set(cluster_labels)
            metrics.num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            metrics.noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels) if len(cluster_labels) > 0 else 1.0

            # Pattern quality assessment
            if metrics.num_clusters > 0:
                # Analyze clusters for patterns
                clusters_data = self._organize_clusters(cluster_labels)
                pattern_scores = []

                for cluster in clusters_data:
                    if cluster['size'] >= params['min_cluster_size']:
                        # Assess cluster significance
                        significance = self.significance_tester.assess_cluster_significance(
                            cluster, self.dataset
                        )
                        pattern_scores.append(significance.get('significance_score', 0.0))

                        # Spatial coherence
                        if 'spatial_metrics' in cluster:
                            metrics.spatial_coherence += cluster['spatial_metrics'].get('compactness', 0.0)

                if pattern_scores:
                    metrics.pattern_quality_score = np.mean(pattern_scores)
                    metrics.significance_score = np.mean(pattern_scores)

                if metrics.num_clusters > 0:
                    metrics.spatial_coherence /= metrics.num_clusters

            return metrics

        except Exception as e:
            self.logger.warning(f"Clustering evaluation failed: {e}")
            return ClusteringMetrics()

    def _organize_clusters(self, labels: np.ndarray) -> List[Dict[str, Any]]:
        """Organize clustering results into cluster data structures."""
        clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_points = [self.dataset[i] for i in cluster_indices]

            cluster_info = {
                "cluster_id": int(label),
                "size": len(cluster_points),
                "points": cluster_points,
                "indices": cluster_indices.tolist()
            }

            # Add spatial metrics if coordinates available
            if self.coordinates is not None:
                cluster_coords = self.coordinates[cluster_indices]
                if len(cluster_coords) >= 3:
                    # Calculate centroid and bounds
                    centroid_lat = float(np.mean(cluster_coords[:, 0]))
                    centroid_lon = float(np.mean(cluster_coords[:, 1]))

                    cluster_info["centroid"] = {"latitude": centroid_lat, "longitude": centroid_lon}
                    cluster_info["bounds"] = {
                        "north": float(np.max(cluster_coords[:, 0])),
                        "south": float(np.min(cluster_coords[:, 0])),
                        "east": float(np.max(cluster_coords[:, 1])),
                        "west": float(np.min(cluster_coords[:, 1]))
                    }

                    # Spatial metrics
                    area = self._calculate_cluster_area(cluster_info)
                    density = len(cluster_points) / max(1, area)
                    compactness = self._calculate_cluster_compactness(cluster_info)

                    cluster_info["spatial_metrics"] = {
                        "area_km2": area,
                        "density": density,
                        "compactness": compactness
                    }

            clusters.append(cluster_info)

        return clusters

    def _calculate_cluster_area(self, cluster: Dict[str, Any]) -> float:
        """Calculate approximate cluster area in km²."""
        bounds = cluster.get("bounds", {})
        if not bounds:
            return 1.0

        lat_diff = bounds.get("north", 0) - bounds.get("south", 0)
        lon_diff = bounds.get("east", 0) - bounds.get("west", 0)

        # Rough conversion to km²
        return abs(lat_diff * lon_diff) * 111.32 * 111.32

    def _calculate_cluster_compactness(self, cluster: Dict[str, Any]) -> float:
        """Calculate cluster compactness (0-1 scale)."""
        points = cluster.get("points", [])
        if len(points) < 2:
            return 1.0

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

        avg_distance = np.mean(distances)
        max_distance = max(distances)

        return 1.0 - (avg_distance / max_distance) if max_distance > 0 else 1.0

    def _calculate_reward(self, current_metrics: ClusteringMetrics,
                         previous_metrics: ClusteringMetrics) -> float:
        """Calculate reward based on clustering performance improvement."""

        reward = 0.0

        # Reward for better silhouette score
        if current_metrics.silhouette_score > previous_metrics.silhouette_score:
            reward += (current_metrics.silhouette_score - previous_metrics.silhouette_score) * 10.0

        # Reward for better Calinski-Harabasz score (higher is better)
        if current_metrics.calinski_harabasz_score > previous_metrics.calinski_harabasz_score:
            improvement = current_metrics.calinski_harabasz_score - previous_metrics.calinski_harabasz_score
            reward += min(improvement / 100.0, 2.0)  # Cap at 2.0

        # Penalize for worse Davies-Bouldin score (lower is better)
        if current_metrics.davies_bouldin_score < previous_metrics.davies_bouldin_score:
            improvement = previous_metrics.davies_bouldin_score - current_metrics.davies_bouldin_score
            reward += min(improvement, 2.0)

        # Reward for better pattern quality
        if current_metrics.pattern_quality_score > previous_metrics.pattern_quality_score:
            reward += (current_metrics.pattern_quality_score - previous_metrics.pattern_quality_score) * 5.0

        # Reward for better spatial coherence
        if current_metrics.spatial_coherence > previous_metrics.spatial_coherence:
            reward += (current_metrics.spatial_coherence - previous_metrics.spatial_coherence) * 3.0

        # Small penalty for too many clusters (prefer parsimonious solutions)
        if current_metrics.num_clusters > 0:
            optimal_clusters = max(2, len(self.dataset) // 20)  # Rough heuristic
            cluster_penalty = abs(current_metrics.num_clusters - optimal_clusters) / optimal_clusters
            reward -= min(cluster_penalty, 1.0)

        # Penalty for too much noise
        if current_metrics.noise_ratio > 0.8:  # Too much noise
            reward -= 1.0
        elif current_metrics.noise_ratio < 0.2:  # Good balance
            reward += 0.5

        # Bonus for significant improvement over best
        if current_metrics.pattern_quality_score > self.best_metrics.pattern_quality_score:
            reward += 2.0

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""

        # Normalized parameters
        param_obs = self._normalize_params(self.current_params)

        # Performance metrics
        metrics_obs = np.array([
            self.previous_metrics.silhouette_score,
            self.previous_metrics.calinski_harabasz_score,
            self.previous_metrics.davies_bouldin_score,
            self.previous_metrics.num_clusters,
            self.previous_metrics.noise_ratio,
            self.previous_metrics.pattern_quality_score
        ])

        # Dataset characteristics
        dataset_obs = np.array([
            len(self.dataset),  # dataset size
            self.features.shape[1] if self.features.size > 0 else 0,  # dimensionality
            self._calculate_spatial_extent()  # spatial extent
        ])

        return np.concatenate([param_obs, metrics_obs, dataset_obs]).astype(np.float32)

    def _calculate_spatial_extent(self) -> float:
        """Calculate spatial extent of dataset."""
        if self.coordinates is None or len(self.coordinates) == 0:
            return 0.0

        lat_range = np.max(self.coordinates[:, 0]) - np.min(self.coordinates[:, 0])
        lon_range = np.max(self.coordinates[:, 1]) - np.min(self.coordinates[:, 1])

        return np.sqrt(lat_range**2 + lon_range**2)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0-5 for parameter adjustments)

        Returns:
            observation, reward, terminated, truncated, info
        """

        # Adjust parameters based on action
        param_names = ['min_samples', 'min_cluster_size', 'cluster_selection_epsilon']
        adjustments = [-1, 1, -1, 1, -0.1, 0.1]  # Decrease/increase for each param

        param_idx = action // 2
        adjustment = adjustments[action]

        if param_idx < len(param_names):
            param_name = param_names[param_idx]
            current_value = self.current_params[param_name]
            min_val, max_val = self.parameter_ranges[param_name]

            # Apply adjustment with bounds checking
            if param_name in ['min_samples', 'min_cluster_size']:
                new_value = int(np.clip(current_value + adjustment, min_val, max_val))
            else:  # cluster_selection_epsilon
                new_value = np.clip(current_value + adjustment, min_val, max_val)

            self.current_params[param_name] = new_value

        # Evaluate new parameters
        current_metrics = self._evaluate_clustering(self.current_params)

        # Calculate reward
        reward = self._calculate_reward(current_metrics, self.previous_metrics)

        # Update best parameters if improved
        if current_metrics.pattern_quality_score > self.best_metrics.pattern_quality_score:
            self.best_metrics = current_metrics
            self.best_params = self.current_params.copy()

        # Update state
        self.previous_metrics = current_metrics
        self.parameter_history.append(self.current_params.copy())
        self.episode_rewards.append(reward)

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get new observation
        observation = self._get_observation()

        info = {
            "current_params": self.current_params.copy(),
            "current_metrics": {
                "silhouette_score": current_metrics.silhouette_score,
                "num_clusters": current_metrics.num_clusters,
                "noise_ratio": current_metrics.noise_ratio,
                "pattern_quality_score": current_metrics.pattern_quality_score
            },
            "best_params": self.best_params.copy(),
            "best_score": self.best_metrics.pattern_quality_score,
            "step": self.current_step
        }

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""

        super().reset(seed=seed)

        self.current_step = 0
        self.current_params = self._get_default_params()
        self.previous_metrics = ClusteringMetrics()
        self.episode_rewards = []
        self.parameter_history = []

        # Keep track of best across episodes
        if not hasattr(self, 'best_metrics') or self.best_metrics.pattern_quality_score == 0.0:
            self.best_metrics = ClusteringMetrics()
            self.best_params = self.current_params.copy()

        observation = self._get_observation()

        info = {
            "initial_params": self.current_params.copy(),
            "dataset_size": len(self.dataset),
            "feature_dims": self.features.shape[1] if self.features.size > 0 else 0
        }

        return observation, info

    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found during training."""
        return {
            "parameters": self.best_params.copy(),
            "metrics": {
                "silhouette_score": self.best_metrics.silhouette_score,
                "calinski_harabasz_score": self.best_metrics.calinski_harabasz_score,
                "davies_bouldin_score": self.best_metrics.davies_bouldin_score,
                "num_clusters": self.best_metrics.num_clusters,
                "noise_ratio": self.best_metrics.noise_ratio,
                "pattern_quality_score": self.best_metrics.pattern_quality_score,
                "spatial_coherence": self.best_metrics.spatial_coherence,
                "significance_score": self.best_metrics.significance_score
            },
            "training_history": {
                "total_steps": self.current_step,
                "parameter_history": self.parameter_history,
                "episode_rewards": self.episode_rewards
            }
        }


class MultiAgentHDBSCANEnv:
    """
    Multi-Agent wrapper for HDBSCAN optimization environment.

    This allows multiple agents to collaborate on parameter optimization,
    where each agent specializes in different aspects of the clustering problem.
    """

    def __init__(self, dataset: List[Dict[str, Any]], num_agents: int = 3):
        """
        Initialize multi-agent environment.

        Args:
            dataset: Dataset for clustering
            num_agents: Number of collaborating agents
        """
        self.dataset = dataset
        self.num_agents = num_agents

        # Create individual agent environments
        self.agent_envs = [HDBSCANOptimizationEnv(dataset) for _ in range(num_agents)]

        # Shared best parameters across agents
        self.global_best_params = None
        self.global_best_score = 0.0

        self.logger = logging.getLogger(__name__)

    async def step_agents(self, actions: List[int]) -> List[Tuple]:
        """Execute steps for all agents asynchronously."""
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        # Execute steps concurrently
        tasks = []
        for i, action in enumerate(actions):
            task = asyncio.get_event_loop().run_in_executor(None, self.agent_envs[i].step, action)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Update global best
        for result in results:
            obs, reward, terminated, truncated, info = result
            agent_best_score = info.get("best_score", 0.0)
            if agent_best_score > self.global_best_score:
                self.global_best_score = agent_best_score
                self.global_best_params = info.get("best_params")

        return results

    def get_global_best(self) -> Dict[str, Any]:
        """Get globally best parameters across all agents."""
        return {
            "best_params": self.global_best_params,
            "best_score": self.global_best_score
        }

    def reset_all(self):
        """Reset all agent environments."""
        for env in self.agent_envs:
            env.reset()