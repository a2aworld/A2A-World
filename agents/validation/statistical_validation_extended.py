"""
A2A World Platform - Extended Statistical Validation Framework (Part 2)

Contains SpatialStatistics, SignificanceClassifier, and StatisticalReports classes
for comprehensive statistical validation of discovered patterns.
"""

import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
import warnings

# Import from the main statistical_validation module
from .statistical_validation import StatisticalResult, SignificanceLevel, PatternSignificance


class SpatialStatistics:
    """
    Comprehensive spatial analysis tools including Getis-Ord Gi* statistic,
    spatial concentration indices, and advanced spatial metrics.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize spatial statistics analyzer.
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
    
    def getis_ord_gi_star(self, coordinates: np.ndarray,
                         values: np.ndarray,
                         weights_method: str = "distance",
                         distance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate Getis-Ord Gi* statistic for hot spot analysis.
        
        Args:
            coordinates: Array of coordinate pairs
            values: Array of values for analysis
            weights_method: Method for spatial weights ('distance', 'knn')
            distance_threshold: Distance threshold for spatial weights
            
        Returns:
            Dictionary with Gi* statistics and hot spot analysis
        """
        try:
            n = len(coordinates)
            if n < 3:
                return {"error": "Insufficient data for Gi* analysis"}
            
            # Calculate spatial weights matrix
            W = self._calculate_spatial_weights(coordinates, weights_method, distance_threshold)
            
            # Calculate global statistics
            mean_x = np.mean(values)
            std_x = np.std(values)
            S = np.sum(W, axis=1)  # Sum of weights for each location
            
            # Calculate Gi* for each location
            gi_stats = []
            hotspots = []
            coldspots = []
            
            for i in range(n):
                # Include focal location in calculation (Gi* vs Gi)
                W_star = W[i, :].copy()
                W_star[i] = 1.0  # Include self
                
                # Sum of weighted values
                numerator = np.sum(W_star * values)
                
                # Expected value
                wi_sum = np.sum(W_star)
                expected = mean_x * wi_sum
                
                # Variance
                wi_sum_sq = np.sum(W_star**2)
                variance = (wi_sum_sq * (n - 1) * std_x**2 - expected**2) / (n - 1)
                
                # Gi* statistic
                if variance > 0:
                    gi_star = (numerator - expected) / np.sqrt(variance)
                    p_value = 2 * (1 - stats.norm.cdf(abs(gi_star)))
                else:
                    gi_star = 0.0
                    p_value = 1.0
                
                significant = p_value < self.significance_level
                
                # Classify as hotspot or coldspot
                if significant:
                    if gi_star > 0:
                        hotspots.append({
                            "index": i,
                            "gi_star": gi_star,
                            "p_value": p_value,
                            "coordinates": coordinates[i].tolist(),
                            "value": values[i]
                        })
                    else:
                        coldspots.append({
                            "index": i,
                            "gi_star": gi_star,
                            "p_value": p_value,
                            "coordinates": coordinates[i].tolist(),
                            "value": values[i]
                        })
                
                gi_stats.append({
                    "index": i,
                    "gi_star": float(gi_star),
                    "p_value": float(p_value),
                    "significant": significant,
                    "type": "hotspot" if gi_star > 0 and significant else 
                           "coldspot" if gi_star < 0 and significant else "not_significant"
                })
            
            # Apply multiple comparison correction
            p_values = [stat["p_value"] for stat in gi_stats]
            corrected_results = self._apply_bonferroni_correction(p_values)
            
            # Update significance with correction
            for i, stat in enumerate(gi_stats):
                stat["bonferroni_significant"] = corrected_results["significant"][i]
                stat["corrected_p_value"] = corrected_results["corrected_p_values"][i]
            
            return {
                "gi_star_statistics": gi_stats,
                "hotspots": hotspots,
                "coldspots": coldspots,
                "summary": {
                    "total_locations": n,
                    "significant_hotspots": len(hotspots),
                    "significant_coldspots": len(coldspots),
                    "bonferroni_significant": corrected_results["n_significant_corrected"],
                    "hotspot_locations": [h["coordinates"] for h in hotspots],
                    "coldspot_locations": [c["coordinates"] for c in coldspots]
                },
                "multiple_testing": corrected_results
            }
            
        except Exception as e:
            self.logger.error(f"Getis-Ord Gi* calculation failed: {e}")
            return {"error": str(e)}
    
    def gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for spatial concentration measurement.
        
        Args:
            values: Array of values to analyze
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        try:
            if len(values) == 0:
                return 0.0
            
            # Sort values
            sorted_values = np.sort(values)
            n = len(sorted_values)
            
            # Calculate cumulative sums
            cumulative_values = np.cumsum(sorted_values)
            
            # Gini coefficient formula
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
            
            return max(0.0, min(1.0, gini))  # Ensure valid range
            
        except Exception as e:
            self.logger.error(f"Gini coefficient calculation failed: {e}")
            return 0.0
    
    def location_quotient(self, local_values: np.ndarray, 
                         global_values: np.ndarray) -> np.ndarray:
        """
        Calculate Location Quotient for spatial concentration analysis.
        
        Args:
            local_values: Values for local areas
            global_values: Values for the entire study area
            
        Returns:
            Array of location quotients
        """
        try:
            local_total = np.sum(local_values)
            global_total = np.sum(global_values)
            
            if local_total == 0 or global_total == 0:
                return np.zeros_like(local_values)
            
            # LQ = (local_i / local_total) / (global_i / global_total)
            local_proportions = local_values / local_total
            global_proportions = global_values / global_total
            
            # Avoid division by zero
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                location_quotients = np.where(global_proportions > 0,
                                           local_proportions / global_proportions,
                                           0.0)
            
            return location_quotients
            
        except Exception as e:
            self.logger.error(f"Location quotient calculation failed: {e}")
            return np.zeros_like(local_values)
    
    def silhouette_analysis(self, coordinates: np.ndarray,
                          cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Perform silhouette analysis for cluster quality assessment.
        
        Args:
            coordinates: Array of coordinate pairs
            cluster_labels: Cluster assignments for each point
            
        Returns:
            Dictionary with silhouette analysis results
        """
        try:
            if len(coordinates) < 2:
                return {"error": "Insufficient data for silhouette analysis"}
            
            # Filter out noise points (-1 labels)
            mask = cluster_labels != -1
            if np.sum(mask) < 2:
                return {"error": "No valid clusters for silhouette analysis"}
            
            filtered_coords = coordinates[mask]
            filtered_labels = cluster_labels[mask]
            
            # Calculate silhouette scores
            if len(np.unique(filtered_labels)) > 1:
                overall_score = silhouette_score(filtered_coords, filtered_labels)
                
                # Individual silhouette scores
                from sklearn.metrics import silhouette_samples
                individual_scores = silhouette_samples(filtered_coords, filtered_labels)
                
                # Analyze by cluster
                cluster_analysis = {}
                unique_labels = np.unique(filtered_labels)
                
                for label in unique_labels:
                    cluster_mask = filtered_labels == label
                    cluster_scores = individual_scores[cluster_mask]
                    
                    cluster_analysis[int(label)] = {
                        "mean_silhouette": float(np.mean(cluster_scores)),
                        "std_silhouette": float(np.std(cluster_scores)),
                        "min_silhouette": float(np.min(cluster_scores)),
                        "max_silhouette": float(np.max(cluster_scores)),
                        "cluster_size": int(np.sum(cluster_mask)),
                        "below_average": int(np.sum(cluster_scores < overall_score))
                    }
                
                return {
                    "overall_silhouette_score": float(overall_score),
                    "individual_scores": individual_scores.tolist(),
                    "cluster_analysis": cluster_analysis,
                    "quality_assessment": self._assess_silhouette_quality(overall_score),
                    "n_clusters": len(unique_labels),
                    "n_points_analyzed": len(filtered_coords)
                }
            else:
                return {
                    "overall_silhouette_score": 0.0,
                    "error": "Only one cluster found",
                    "n_clusters": 1
                }
            
        except Exception as e:
            self.logger.error(f"Silhouette analysis failed: {e}")
            return {"error": str(e)}
    
    def spatial_association_matrix(self, coordinates: np.ndarray,
                                 categorical_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate spatial association matrix for categorical spatial data.
        
        Args:
            coordinates: Array of coordinate pairs
            categorical_values: Array of categorical values
            
        Returns:
            Dictionary with spatial association analysis
        """
        try:
            n = len(coordinates)
            if n < 3:
                return {"error": "Insufficient data for spatial association analysis"}
            
            # Get unique categories
            unique_categories = np.unique(categorical_values)
            n_categories = len(unique_categories)
            
            if n_categories < 2:
                return {"error": "Need at least 2 categories for association analysis"}
            
            # Calculate spatial weights (k-nearest neighbors)
            k = min(8, n - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates)
            
            # Create spatial association matrix
            association_matrix = np.zeros((n_categories, n_categories))
            category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
            
            for i in range(n):
                focal_category = categorical_values[i]
                focal_idx = category_to_idx[focal_category]
                
                # Check neighbors (excluding self)
                for j in range(1, k + 1):
                    neighbor_idx = indices[i, j]
                    neighbor_category = categorical_values[neighbor_idx]
                    neighbor_cat_idx = category_to_idx[neighbor_category]
                    
                    # Weight by inverse distance
                    weight = 1.0 / (distances[i, j] + 1e-10)
                    association_matrix[focal_idx, neighbor_cat_idx] += weight
            
            # Normalize by row sums
            row_sums = association_matrix.sum(axis=1)
            normalized_matrix = association_matrix / row_sums[:, np.newaxis]
            
            # Calculate spatial association indices
            association_indices = {}
            for i, cat1 in enumerate(unique_categories):
                for j, cat2 in enumerate(unique_categories):
                    if i != j:  # Skip diagonal
                        observed = normalized_matrix[i, j]
                        expected = 1.0 / (n_categories - 1)  # Equal probability
                        
                        if expected > 0:
                            association_index = (observed - expected) / expected
                        else:
                            association_index = 0.0
                        
                        association_indices[f"{cat1}_{cat2}"] = float(association_index)
            
            return {
                "association_matrix": association_matrix.tolist(),
                "normalized_matrix": normalized_matrix.tolist(),
                "association_indices": association_indices,
                "categories": unique_categories.tolist(),
                "interpretation": self._interpret_spatial_association(association_indices),
                "n_categories": n_categories,
                "sample_size": n
            }
            
        except Exception as e:
            self.logger.error(f"Spatial association analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_spatial_weights(self, coordinates: np.ndarray,
                                 method: str = "distance",
                                 threshold: Optional[float] = None) -> np.ndarray:
        """Calculate spatial weights matrix."""
        n = coordinates.shape[0]
        
        if method == "distance":
            # Distance-based weights
            distances = squareform(pdist(coordinates))
            
            if threshold is None:
                threshold = np.percentile(distances[distances > 0], 25)
            
            W = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j and distances[i, j] <= threshold:
                        W[i, j] = 1.0
            
            return W
        
        elif method == "knn":
            # k-nearest neighbors
            k = min(8, n - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates)
            
            W = np.zeros((n, n))
            for i in range(n):
                for j in range(1, k + 1):  # Skip self
                    neighbor_idx = indices[i, j]
                    W[i, neighbor_idx] = 1.0
            
            return W
        
        else:
            # Default: identity matrix
            return np.eye(n)
    
    def _apply_bonferroni_correction(self, p_values: List[float]) -> Dict[str, Any]:
        """Apply Bonferroni correction for multiple testing."""
        p_array = np.array(p_values)
        n = len(p_array)
        
        corrected_p = np.minimum(p_array * n, 1.0)
        corrected_alpha = self.significance_level / n
        significant = corrected_p < corrected_alpha
        
        return {
            "original_p_values": p_values,
            "corrected_p_values": corrected_p.tolist(),
            "corrected_alpha": corrected_alpha,
            "significant": significant.tolist(),
            "n_significant_original": np.sum(p_array < self.significance_level),
            "n_significant_corrected": np.sum(significant)
        }
    
    def _assess_silhouette_quality(self, score: float) -> str:
        """Assess clustering quality based on silhouette score."""
        if score > 0.7:
            return "excellent"
        elif score > 0.5:
            return "good"
        elif score > 0.25:
            return "fair"
        elif score > 0:
            return "poor"
        else:
            return "very_poor"
    
    def _interpret_spatial_association(self, association_indices: Dict[str, float]) -> str:
        """Interpret spatial association results."""
        strong_associations = []
        
        for pair, index in association_indices.items():
            if abs(index) > 0.5:
                direction = "attraction" if index > 0 else "repulsion"
                strong_associations.append(f"{pair}: {direction} (index: {index:.3f})")
        
        if strong_associations:
            return f"Strong spatial associations found: {'; '.join(strong_associations)}"
        else:
            return "No strong spatial associations detected"


class SignificanceClassifier:
    """
    Multi-tier significance classification system for pattern reliability scoring.
    """
    
    def __init__(self, significance_levels: Optional[Dict[str, float]] = None):
        """
        Initialize significance classifier.
        
        Args:
            significance_levels: Custom significance level thresholds
        """
        self.significance_levels = significance_levels or {
            "very_high": 0.001,
            "high": 0.01,
            "moderate": 0.05,
            "low": 0.10
        }
        self.logger = logging.getLogger(__name__)
    
    def classify_pattern_significance(self, statistical_results: List[StatisticalResult]) -> Dict[str, Any]:
        """
        Classify pattern significance based on multiple statistical tests.
        
        Args:
            statistical_results: List of statistical test results
            
        Returns:
            Dictionary with significance classification
        """
        try:
            if not statistical_results:
                return {"error": "No statistical results provided"}
            
            # Extract p-values and effect sizes
            p_values = []
            effect_sizes = []
            test_names = []
            
            for result in statistical_results:
                if result.p_value is not None:
                    p_values.append(result.p_value)
                    effect_sizes.append(result.effect_size or 0.0)
                    test_names.append(result.statistic_name)
            
            if not p_values:
                return {"error": "No valid p-values found"}
            
            # Calculate overall significance metrics
            min_p_value = min(p_values)
            mean_p_value = np.mean(p_values)
            median_p_value = np.median(p_values)
            
            # Count significant tests at different levels
            significance_counts = {}
            for level_name, threshold in self.significance_levels.items():
                count = sum(1 for p in p_values if p < threshold)
                significance_counts[level_name] = {
                    "count": count,
                    "proportion": count / len(p_values),
                    "threshold": threshold
                }
            
            # Overall pattern significance classification
            overall_classification = self._determine_overall_significance(
                min_p_value, mean_p_value, significance_counts, len(p_values)
            )
            
            # Calculate reliability score (0-1)
            reliability_score = self._calculate_reliability_score(
                p_values, effect_sizes, statistical_results
            )
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(statistical_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_classification, reliability_score, significance_counts
            )
            
            return {
                "overall_classification": overall_classification,
                "reliability_score": reliability_score,
                "confidence_metrics": confidence_metrics,
                "p_value_summary": {
                    "minimum": min_p_value,
                    "mean": mean_p_value,
                    "median": median_p_value,
                    "values": p_values
                },
                "significance_breakdown": significance_counts,
                "test_summary": {
                    "total_tests": len(p_values),
                    "test_names": test_names,
                    "effect_sizes": effect_sizes
                },
                "recommendations": recommendations,
                "interpretation": self._create_interpretation(
                    overall_classification, reliability_score, len(p_values)
                )
            }
            
        except Exception as e:
            self.logger.error(f"Pattern significance classification failed: {e}")
            return {"error": str(e)}
    
    def calculate_pattern_reliability_score(self, statistical_results: List[StatisticalResult],
                                          pattern_metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate comprehensive pattern reliability score (0-1 scale).
        
        Args:
            statistical_results: List of statistical test results
            pattern_metadata: Additional pattern metadata for scoring
            
        Returns:
            Reliability score between 0 and 1
        """
        try:
            if not statistical_results:
                return 0.0
            
            # Component scores (each 0-1)
            scores = {
                "statistical_significance": 0.0,
                "effect_size": 0.0,
                "consistency": 0.0,
                "sample_size": 0.0,
                "multiple_testing": 0.0
            }
            
            # 1. Statistical significance score
            p_values = [r.p_value for r in statistical_results if r.p_value is not None]
            if p_values:
                # Use geometric mean of p-values (more conservative)
                geometric_mean_p = np.exp(np.mean(np.log([max(p, 1e-10) for p in p_values])))
                scores["statistical_significance"] = max(0, 1 - geometric_mean_p / 0.05)
            
            # 2. Effect size score
            effect_sizes = [r.effect_size for r in statistical_results if r.effect_size is not None]
            if effect_sizes:
                mean_effect_size = np.mean([abs(es) for es in effect_sizes])
                scores["effect_size"] = min(1.0, mean_effect_size / 0.8)  # Normalize by large effect threshold
            
            # 3. Consistency across tests
            significant_tests = sum(1 for r in statistical_results if r.significant)
            if len(statistical_results) > 0:
                scores["consistency"] = significant_tests / len(statistical_results)
            
            # 4. Sample size adequacy
            sample_sizes = []
            for result in statistical_results:
                if result.metadata and "sample_size" in result.metadata:
                    sample_sizes.append(result.metadata["sample_size"])
            
            if sample_sizes:
                min_sample_size = min(sample_sizes)
                scores["sample_size"] = min(1.0, min_sample_size / 100)  # Normalize by adequate sample size
            else:
                scores["sample_size"] = 0.5  # Default for unknown sample size
            
            # 5. Multiple testing correction handling
            corrected_tests = sum(1 for r in statistical_results 
                                if r.metadata and "corrected_p_value" in r.metadata)
            if len(statistical_results) > 1:
                scores["multiple_testing"] = corrected_tests / len(statistical_results)
            else:
                scores["multiple_testing"] = 1.0  # Single test doesn't need correction
            
            # Weighted combination of scores
            weights = {
                "statistical_significance": 0.3,
                "effect_size": 0.25,
                "consistency": 0.2,
                "sample_size": 0.15,
                "multiple_testing": 0.1
            }
            
            reliability_score = sum(scores[component] * weights[component] 
                                  for component in scores)
            
            return min(1.0, max(0.0, reliability_score))
            
        except Exception as e:
            self.logger.error(f"Reliability score calculation failed: {e}")
            return 0.0
    
    def create_significance_thresholds(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Create automated significance thresholds based on base alpha level.
        
        Args:
            alpha: Base significance level
            
        Returns:
            Dictionary with significance thresholds
        """
        return {
            "very_high": alpha / 50,   # 0.001 for α=0.05
            "high": alpha / 5,         # 0.01 for α=0.05
            "moderate": alpha,         # 0.05 for α=0.05
            "low": alpha * 2,         # 0.10 for α=0.05
            "marginal": alpha * 4     # 0.20 for α=0.05
        }
    
    def _determine_overall_significance(self, min_p: float, mean_p: float,
                                      significance_counts: Dict[str, Dict],
                                      n_tests: int) -> str:
        """Determine overall pattern significance classification."""
        
        # Very high significance: multiple tests highly significant
        if (significance_counts["very_high"]["count"] >= max(1, n_tests // 2) or
            (min_p < 0.001 and significance_counts["high"]["proportion"] > 0.5)):
            return "very_high"
        
        # High significance: consistent significant results
        elif (significance_counts["high"]["count"] >= max(1, n_tests // 2) or
              (min_p < 0.01 and significance_counts["moderate"]["proportion"] > 0.7)):
            return "high"
        
        # Moderate significance: some consistent results
        elif (significance_counts["moderate"]["count"] >= max(1, n_tests // 3) or
              mean_p < 0.05):
            return "moderate"
        
        # Low significance: few significant results
        elif significance_counts["low"]["count"] > 0 or mean_p < 0.10:
            return "low"
        
        # Not significant
        else:
            return "not_significant"
    
    def _calculate_reliability_score(self, p_values: List[float],
                                   effect_sizes: List[float],
                                   statistical_results: List[StatisticalResult]) -> float:
        """Calculate reliability score from statistical results."""
        return self.calculate_pattern_reliability_score(statistical_results)
    
    def _calculate_confidence_metrics(self, statistical_results: List[StatisticalResult]) -> Dict[str, float]:
        """Calculate confidence metrics from statistical results."""
        try:
            confidence_intervals = []
            z_scores = []
            
            for result in statistical_results:
                if result.confidence_interval:
                    ci_width = result.confidence_interval[1] - result.confidence_interval[0]
                    confidence_intervals.append(ci_width)
                
                if result.z_score is not None:
                    z_scores.append(abs(result.z_score))
            
            metrics = {}
            
            if confidence_intervals:
                metrics["mean_ci_width"] = np.mean(confidence_intervals)
                metrics["median_ci_width"] = np.median(confidence_intervals)
            
            if z_scores:
                metrics["mean_z_score"] = np.mean(z_scores)
                metrics["max_z_score"] = np.max(z_scores)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Confidence metrics calculation failed: {e}")
            return {}
    
    def _generate_recommendations(self, classification: str,
                                reliability_score: float,
                                significance_counts: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on significance analysis."""
        recommendations = []
        
        if classification == "very_high":
            recommendations.append("Pattern shows very high statistical significance - suitable for publication")
            recommendations.append("Consider replication with independent datasets")
        
        elif classification == "high":
            recommendations.append("Pattern shows high statistical significance")
            recommendations.append("Additional validation with larger sample sizes recommended")
        
        elif classification == "moderate":
            recommendations.append("Pattern shows moderate statistical significance")
            recommendations.append("Consider additional statistical tests or larger sample sizes")
        
        elif classification == "low":
            recommendations.append("Pattern shows low statistical significance")
            recommendations.append("Results should be interpreted with caution")
            recommendations.append("Consider alternative analytical approaches")
        
        else:
            recommendations.append("Pattern does not show statistical significance")
            recommendations.append("Consider methodological review or alternative hypotheses")
        
        # Reliability-based recommendations
        if reliability_score < 0.3:
            recommendations.append("Low reliability score - recommend methodological review")
        elif reliability_score < 0.5:
            recommendations.append("Moderate reliability - additional validation recommended")
        
        return recommendations
    
    def _create_interpretation(self, classification: str,
                             reliability_score: float,
                             n_tests: int) -> str:
        """Create human-readable interpretation."""
        interpretation = f"Pattern classified as '{classification}' significance "
        interpretation += f"with reliability score of {reliability_score:.3f} "
        interpretation += f"based on {n_tests} statistical tests."
        
        if classification in ["very_high", "high"]:
            interpretation += " Strong evidence for non-random pattern."
        elif classification == "moderate":
            interpretation += " Moderate evidence for pattern, requires careful interpretation."
        elif classification == "low":
            interpretation += " Weak evidence for pattern, interpret with caution."
        else:
            interpretation += " No strong evidence for non-random pattern."
        
        return interpretation


class StatisticalReports:
    """
    Statistical report generation and visualization for pattern validation results.
    """
    
    def __init__(self):
        """Initialize statistical reports generator."""
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self, validation_results: Dict[str, Any],
                                    pattern_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistical validation report.
        
        Args:
            validation_results: Results from statistical validation
            pattern_metadata: Additional pattern metadata
            
        Returns:
            Dictionary containing comprehensive report
        """
        try:
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "report_type": "comprehensive_statistical_validation",
                "summary": {},
                "detailed_results": {},
                "visualizations": {},
                "conclusions": {},
                "recommendations": []
            }
            
            # Extract key information
            if "statistical_results" in validation_results:
                stats_results = validation_results["statistical_results"]
                
                # Generate summary statistics
                report["summary"] = self._generate_summary_statistics(stats_results)
                
                # Detailed statistical analysis
                report["detailed_results"] = self._format_detailed_results(stats_results)
                
                # Generate conclusions
                report["conclusions"] = self._generate_conclusions(stats_results)
                
                # Generate recommendations
                report["recommendations"] = self._generate_recommendations(stats_results)
            
            if pattern_metadata:
                report["pattern_metadata"] = pattern_metadata
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {"error": str(e)}
    
    def create_validation_dashboard_data(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create data structure for validation dashboard display.
        
        Args:
            validation_results: Statistical validation results
            
        Returns:
            Dashboard-ready data structure
        """
        try:
            dashboard_data = {
                "overview_metrics": {},
                "significance_indicators": {},
                "statistical_charts": {},
                "detailed_tables": {},
                "alerts": []
            }
            
            # Overview metrics for cards/widgets
            if "significance_classification" in validation_results:
                sig_class = validation_results["significance_classification"]
                
                dashboard_data["overview_metrics"] = {
                    "overall_significance": sig_class.get("overall_classification", "unknown"),
                    "reliability_score": sig_class.get("reliability_score", 0.0),
                    "total_tests": sig_class.get("test_summary", {}).get("total_tests", 0),
                    "significant_tests": sum(1 for p in sig_class.get("p_value_summary", {}).get("values", []) if p < 0.05),
                    "min_p_value": sig_class.get("p_value_summary", {}).get("minimum", 1.0)
                }
            
            # Significance indicators for visual display
            if "statistical_results" in validation_results:
                dashboard_data["significance_indicators"] = self._create_significance_indicators(
                    validation_results["statistical_results"]
                )
            
            # Chart data for visualizations
            dashboard_data["statistical_charts"] = self._create_chart_data(validation_results)
            
            # Detailed tables for drill-down
            dashboard_data["detailed_tables"] = self._create_table_data(validation_results)
            
            # Generate alerts for important findings
            dashboard_data["alerts"] = self._generate_alerts(validation_results)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Dashboard data creation failed: {e}")
            return {"error": str(e)}
    
    def _generate_summary_statistics(self, statistical_results: List[StatisticalResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        try:
            p_values = [r.p_value for r in statistical_results if r.p_value is not None]
            effect_sizes = [r.effect_size for r in statistical_results if r.effect_size is not None]
            
            summary = {
                "total_tests": len(statistical_results),
                "significant_tests": sum(1 for r in statistical_results if r.significant),
                "significance_rate": sum(1 for r in statistical_results if r.significant) / len(statistical_results) if statistical_results else 0
            }
            
            if p_values:
                summary.update({
                    "min_p_value": min(p_values),
                    "mean_p_value": np.mean(p_values),
                    "median_p_value": np.median(p_values)
                })
            
            if effect_sizes:
                summary.update({
                    "mean_effect_size": np.mean([abs(es) for es in effect_sizes]),
                    "max_effect_size": max([abs(es) for es in effect_sizes])
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary statistics generation failed: {e}")
            return {}
    
    def _format_detailed_results(self, statistical_results: List[StatisticalResult]) -> Dict[str, Any]:
        """Format detailed statistical results for reporting."""
        detailed = {}
        
        for result in statistical_results:
            detailed[result.statistic_name] = {
                "statistic_value": result.statistic_value,
                "p_value": result.p_value,
                "significant": result.significant,
                "z_score": result.z_score,
                "effect_size": result.effect_size,
                "confidence_interval": result.confidence_interval,
                "interpretation": result.interpretation,
                "metadata": result.metadata
            }
        
        return detailed
    
    def _generate_conclusions(self, statistical_results: List[StatisticalResult]) -> Dict[str, str]:
        """Generate conclusions from statistical analysis."""
        conclusions = {}
        
        significant_tests = [r for r in statistical_results if r.significant]
        
        if len(significant_tests) == 0:
            conclusions["overall"] = "No statistically significant spatial patterns detected."
        elif len(significant_tests) == len(statistical_results):
            conclusions["overall"] = "All statistical tests indicate significant spatial patterns."
        else:
            conclusions["overall"] = f"{len(significant_tests)} out of {len(statistical_results)} tests show significant patterns."
        
        # Test-specific conclusions
        for result in statistical_results:
            if result.interpretation:
                conclusions[result.statistic_name] = result.interpretation
        
        return conclusions
    
    def _generate_recommendations(self, statistical_results: List[StatisticalResult]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        significant_count = sum(1 for r in statistical_results if r.significant)
        total_count = len(statistical_results)
        
        if significant_count == 0:
            recommendations.extend([
                "Consider alternative analytical methods",
                "Verify data quality and completeness",
                "Review spatial scale and study area boundaries"
            ])
        elif significant_count < total_count // 2:
            recommendations.extend([
                "Mixed results suggest need for additional validation",
                "Consider larger sample sizes",
                "Apply alternative spatial analysis methods"
            ])
        else:
            recommendations.extend([
                "Strong statistical evidence supports pattern validity",
                "Consider replication with independent datasets",
                "Suitable for further analysis and publication"
            ])
        
        return recommendations
    
    def _create_significance_indicators(self, statistical_results: List[StatisticalResult]) -> List[Dict[str, Any]]:
        """Create significance indicators for dashboard display."""
        indicators = []
        
        for result in statistical_results:
            color = "green" if result.significant else "red"
            if result.p_value is not None:
                if result.p_value < 0.001:
                    level = "Very High"
                    color = "darkgreen"
                elif result.p_value < 0.01:
                    level = "High" 
                    color = "green"
                elif result.p_value < 0.05:
                    level = "Moderate"
                    color = "orange"
                else:
                    level = "Not Significant"
                    color = "red"
            else:
                level = "Unknown"
                color = "gray"
            
            indicators.append({
                "test_name": result.statistic_name,
                "significance_level": level,
                "p_value": result.p_value,
                "color": color,
                "significant": result.significant,
                "effect_size": result.effect_size
            })
        
        return indicators
    
    def _create_chart_data(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create chart data for visualizations."""
        charts = {}
        
        # P-value distribution chart
        if "statistical_results" in validation_results:
            p_values = [r.p_value for r in validation_results["statistical_results"] if r.p_value is not None]
            test_names = [r.statistic_name for r in validation_results["statistical_results"] if r.p_value is not None]
            
            charts["p_value_chart"] = {
                "type": "bar",
                "data": {
                    "labels": test_names,
                    "values": p_values
                },
                "title": "P-values by Statistical Test",
                "y_axis_label": "P-value"
            }
        
        # Significance levels chart
        if "significance_classification" in validation_results:
            sig_counts = validation_results["significance_classification"].get("significance_breakdown", {})
            
            charts["significance_levels"] = {
                "type": "pie",
                "data": {
                    "labels": list(sig_counts.keys()),
                    "values": [counts["count"] for counts in sig_counts.values()]
                },
                "title": "Tests by Significance Level"
            }
        
        return charts
    
    def _create_table_data(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create table data for detailed display."""
        tables = {}
        
        if "statistical_results" in validation_results:
            # Main results table
            results_data = []
            for result in validation_results["statistical_results"]:
                results_data.append({
                    "Test": result.statistic_name,
                    "Statistic": f"{result.statistic_value:.4f}" if result.statistic_value else "N/A",
                    "P-value": f"{result.p_value:.6f}" if result.p_value else "N/A", 
                    "Significant": "Yes" if result.significant else "No",
                    "Effect Size": f"{result.effect_size:.3f}" if result.effect_size else "N/A",
                    "Interpretation": result.interpretation or "N/A"
                })
            
            tables["main_results"] = {
                "title": "Statistical Test Results",
                "columns": ["Test", "Statistic", "P-value", "Significant", "Effect Size", "Interpretation"],
                "data": results_data
            }
        
        return tables
    
    def _generate_alerts(self, validation_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts for important findings."""
        alerts = []
        
        if "significance_classification" in validation_results:
            classification = validation_results["significance_classification"]["overall_classification"]
            reliability = validation_results["significance_classification"]["reliability_score"]
            
            if classification == "very_high":
                alerts.append({
                    "type": "success",
                    "message": "Very high statistical significance detected - pattern is highly reliable",
                    "priority": "high"
                })
            elif classification == "not_significant":
                alerts.append({
                    "type": "warning", 
                    "message": "No significant statistical patterns detected - review methodology",
                    "priority": "medium"
                })
            
            if reliability < 0.3:
                alerts.append({
                    "type": "error",
                    "message": f"Low reliability score ({reliability:.2f}) - results may not be trustworthy",
                    "priority": "high"
                })
        
        return alerts