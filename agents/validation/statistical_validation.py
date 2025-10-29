"""
A2A World Platform - Statistical Validation Framework

Comprehensive statistical validation framework implementing Moran's I spatial autocorrelation 
analysis, null hypothesis testing, and advanced statistical metrics for rigorous pattern validation.
This framework ensures discovered patterns are statistically significant and not due to random chance.
"""

import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import uuid
import warnings
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
import itertools


class SignificanceLevel(Enum):
    """Statistical significance levels for validation."""
    VERY_HIGH = 0.001  # α = 0.001
    HIGH = 0.01       # α = 0.01
    MODERATE = 0.05   # α = 0.05
    LOW = 0.10        # α = 0.10


class PatternSignificance(Enum):
    """Pattern significance classification."""
    HIGHLY_SIGNIFICANT = "highly_significant"
    SIGNIFICANT = "significant" 
    MARGINALLY_SIGNIFICANT = "marginally_significant"
    NOT_SIGNIFICANT = "not_significant"


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    statistic_name: str
    statistic_value: float
    p_value: float
    z_score: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None
    significant: bool = False
    significance_level: float = 0.05
    metadata: Optional[Dict[str, Any]] = None


class MoransIAnalyzer:
    """
    Advanced Moran's I spatial autocorrelation analyzer with global and local analysis.
    Implements proper statistical significance testing with multiple correction methods.
    """
    
    def __init__(self, significance_level: float = 0.05, n_permutations: int = 999):
        """
        Initialize Moran's I analyzer.
        
        Args:
            significance_level: Statistical significance threshold
            n_permutations: Number of permutations for Monte Carlo testing
        """
        self.significance_level = significance_level
        self.n_permutations = n_permutations
        self.logger = logging.getLogger(__name__)
        
    def calculate_global_morans_i(self, coordinates: np.ndarray, 
                                values: np.ndarray,
                                weights_method: str = "knn",
                                k_neighbors: int = 8) -> StatisticalResult:
        """
        Calculate Global Moran's I spatial autocorrelation statistic.
        
        Args:
            coordinates: Array of coordinate pairs (n_samples, 2)
            values: Array of values for spatial autocorrelation analysis
            weights_method: Method for calculating spatial weights ('knn', 'distance', 'queen')
            k_neighbors: Number of neighbors for k-nearest neighbor weights
            
        Returns:
            StatisticalResult with Moran's I analysis
        """
        try:
            n = len(coordinates)
            if n < 3:
                return self._create_insufficient_data_result("global_morans_i")
            
            # Calculate spatial weights matrix
            W = self._calculate_spatial_weights(coordinates, weights_method, k_neighbors)
            
            # Calculate Moran's I
            morans_i = self._compute_morans_i(values, W)
            
            # Calculate expected value and variance
            expected_i = -1.0 / (n - 1)
            variance_i = self._calculate_morans_i_variance(W, n)
            
            # Calculate z-score and p-value
            if variance_i > 0:
                z_score = (morans_i - expected_i) / np.sqrt(variance_i)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0.0
                p_value = 1.0
            
            # Monte Carlo significance test
            mc_p_value = self._monte_carlo_test(coordinates, values, W, morans_i)
            
            # Use more conservative p-value
            final_p_value = max(p_value, mc_p_value)
            
            # Calculate confidence interval
            if variance_i > 0:
                margin_error = stats.norm.ppf(1 - self.significance_level/2) * np.sqrt(variance_i)
                ci_lower = morans_i - margin_error
                ci_upper = morans_i + margin_error
                confidence_interval = (ci_lower, ci_upper)
            else:
                confidence_interval = None
            
            # Effect size (standardized Moran's I)
            effect_size = abs(z_score) if z_score is not None else 0.0
            
            return StatisticalResult(
                statistic_name="global_morans_i",
                statistic_value=morans_i,
                p_value=final_p_value,
                z_score=z_score,
                confidence_interval=confidence_interval,
                effect_size=effect_size,
                interpretation=self._interpret_morans_i(morans_i, final_p_value < self.significance_level),
                significant=final_p_value < self.significance_level,
                significance_level=self.significance_level,
                metadata={
                    "expected_value": expected_i,
                    "variance": variance_i,
                    "sample_size": n,
                    "weights_method": weights_method,
                    "k_neighbors": k_neighbors,
                    "monte_carlo_p_value": mc_p_value,
                    "analytical_p_value": p_value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Global Moran's I calculation failed: {e}")
            return self._create_error_result("global_morans_i", str(e))
    
    def calculate_local_morans_i(self, coordinates: np.ndarray,
                               values: np.ndarray,
                               weights_method: str = "knn",
                               k_neighbors: int = 8) -> Dict[str, Any]:
        """
        Calculate Local Indicators of Spatial Association (LISA) - Local Moran's I.
        
        Args:
            coordinates: Array of coordinate pairs
            values: Array of values for analysis
            weights_method: Method for calculating spatial weights
            k_neighbors: Number of neighbors for analysis
            
        Returns:
            Dictionary with local Moran's I results
        """
        try:
            n = len(coordinates)
            if n < 3:
                return {"error": "Insufficient data for local Moran's I analysis"}
            
            # Calculate spatial weights matrix
            W = self._calculate_spatial_weights(coordinates, weights_method, k_neighbors)
            
            # Standardize values
            values_std = (values - np.mean(values)) / np.std(values)
            
            # Calculate local Moran's I for each location
            local_stats = []
            for i in range(n):
                # Local Moran's I formula: Ii = zi * sum(wij * zj)
                neighbors_sum = np.sum(W[i, :] * values_std)
                local_i = values_std[i] * neighbors_sum
                
                # Calculate local p-value (simplified pseudo-significance)
                # In practice, this would use conditional permutation
                local_variance = np.sum(W[i, :]**2) * (n - 1) / (n - 2)
                if local_variance > 0:
                    local_z = local_i / np.sqrt(local_variance)
                    local_p = 2 * (1 - stats.norm.cdf(abs(local_z)))
                else:
                    local_z = 0.0
                    local_p = 1.0
                
                # Classify spatial association
                if values_std[i] > 0 and neighbors_sum > 0:
                    cluster_type = "HH"  # High-High
                elif values_std[i] < 0 and neighbors_sum < 0:
                    cluster_type = "LL"  # Low-Low
                elif values_std[i] > 0 and neighbors_sum < 0:
                    cluster_type = "HL"  # High-Low (spatial outlier)
                else:
                    cluster_type = "LH"  # Low-High (spatial outlier)
                
                local_stats.append({
                    "index": i,
                    "local_morans_i": float(local_i),
                    "z_score": float(local_z),
                    "p_value": float(local_p),
                    "cluster_type": cluster_type,
                    "significant": local_p < self.significance_level
                })
            
            # Summary statistics
            significant_count = sum(1 for stat in local_stats if stat["significant"])
            cluster_counts = {}
            for stat in local_stats:
                if stat["significant"]:
                    cluster_type = stat["cluster_type"]
                    cluster_counts[cluster_type] = cluster_counts.get(cluster_type, 0) + 1
            
            # Apply Bonferroni correction for multiple testing
            bonferroni_alpha = self.significance_level / n
            bonferroni_significant = sum(1 for stat in local_stats 
                                       if stat["p_value"] < bonferroni_alpha)
            
            return {
                "local_statistics": local_stats,
                "summary": {
                    "total_locations": n,
                    "significant_locations": significant_count,
                    "bonferroni_significant": bonferroni_significant,
                    "cluster_counts": cluster_counts,
                    "significance_rate": significant_count / n,
                    "bonferroni_alpha": bonferroni_alpha
                },
                "interpretation": self._interpret_local_morans_i(cluster_counts, significant_count)
            }
            
        except Exception as e:
            self.logger.error(f"Local Moran's I calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_spatial_weights(self, coordinates: np.ndarray,
                                 method: str = "knn",
                                 k_neighbors: int = 8) -> np.ndarray:
        """
        Calculate spatial weights matrix using specified method.
        
        Args:
            coordinates: Coordinate array
            method: Weights calculation method
            k_neighbors: Number of neighbors for k-nearest neighbor method
            
        Returns:
            Spatial weights matrix
        """
        n = coordinates.shape[0]
        
        if method == "knn":
            return self._knn_weights(coordinates, k_neighbors)
        elif method == "distance":
            return self._distance_weights(coordinates)
        elif method == "queen":
            return self._contiguity_weights(coordinates, "queen")
        else:
            # Default to k-nearest neighbors
            return self._knn_weights(coordinates, k_neighbors)
    
    def _knn_weights(self, coordinates: np.ndarray, k: int) -> np.ndarray:
        """Calculate k-nearest neighbor spatial weights."""
        n = coordinates.shape[0]
        k = min(k, n - 1)  # Ensure k doesn't exceed available neighbors
        
        # Use NearestNeighbors for efficiency
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        # Create weights matrix
        W = np.zeros((n, n))
        
        for i in range(n):
            for j in range(1, k + 1):  # Skip self (index 0)
                neighbor_idx = indices[i, j]
                # Use inverse distance weighting
                weight = 1.0 / (distances[i, j] + 1e-10)
                W[i, neighbor_idx] = weight
        
        # Row standardize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]
        
        return W
    
    def _distance_weights(self, coordinates: np.ndarray, 
                         threshold_percentile: float = 25) -> np.ndarray:
        """Calculate distance-based spatial weights."""
        n = coordinates.shape[0]
        
        # Calculate distance matrix
        distances = squareform(pdist(coordinates))
        
        # Set threshold as percentile of all distances
        threshold = np.percentile(distances[distances > 0], threshold_percentile)
        
        # Create weights matrix
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and distances[i, j] <= threshold:
                    W[i, j] = 1.0 / (distances[i, j] + 1e-10)
        
        # Row standardize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]
        
        return W
    
    def _contiguity_weights(self, coordinates: np.ndarray, 
                          contiguity_type: str = "queen") -> np.ndarray:
        """Calculate contiguity-based weights (simplified version)."""
        # This is a simplified implementation
        # In practice, would use proper spatial topology
        return self._distance_weights(coordinates, threshold_percentile=10)
    
    def _compute_morans_i(self, values: np.ndarray, W: np.ndarray) -> float:
        """Compute Moran's I statistic."""
        n = len(values)
        if n == 0:
            return 0.0
        
        # Center the values
        mean_val = np.mean(values)
        centered = values - mean_val
        
        # Calculate numerator and denominator
        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * centered[i] * centered[j]
        
        denominator = np.sum(centered**2)
        S0 = np.sum(W)  # Sum of all weights
        
        if denominator == 0 or S0 == 0:
            return 0.0
        
        morans_i = (n / S0) * (numerator / denominator)
        return morans_i
    
    def _calculate_morans_i_variance(self, W: np.ndarray, n: int) -> float:
        """Calculate variance of Moran's I under null hypothesis."""
        try:
            S0 = np.sum(W)
            S1 = 0.5 * np.sum((W + W.T)**2)
            S2 = np.sum((W.sum(axis=1))**2)
            
            b2 = n * np.sum((np.arange(n) - np.mean(np.arange(n)))**4) / (np.sum((np.arange(n) - np.mean(np.arange(n)))**2)**2)
            
            E_I = -1.0 / (n - 1)
            A = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
            B = b2 * ((n**2 - n) * S1 - 2*n*S2 + 6*S0**2)
            C = (n - 1) * (n - 2) * (n - 3) * S0**2
            
            if C > 0:
                var_I = (A - B) / C - E_I**2
                return max(var_I, 1e-10)  # Ensure positive variance
            else:
                return 1e-10
                
        except Exception:
            # Fallback to simplified variance calculation
            return (2.0 / ((n - 1) * (n - 2))) * (1.0 / n)
    
    def _monte_carlo_test(self, coordinates: np.ndarray, values: np.ndarray,
                         W: np.ndarray, observed_i: float) -> float:
        """Perform Monte Carlo permutation test."""
        try:
            n_more_extreme = 0
            
            for _ in range(self.n_permutations):
                # Randomly permute values while keeping coordinates fixed
                permuted_values = np.random.permutation(values)
                
                # Calculate Moran's I for permuted data
                permuted_i = self._compute_morans_i(permuted_values, W)
                
                # Count more extreme values
                if abs(permuted_i) >= abs(observed_i):
                    n_more_extreme += 1
            
            # Calculate pseudo p-value
            pseudo_p = (n_more_extreme + 1) / (self.n_permutations + 1)
            return pseudo_p
            
        except Exception as e:
            self.logger.warning(f"Monte Carlo test failed: {e}")
            return 1.0  # Conservative p-value
    
    def _interpret_morans_i(self, morans_i: float, significant: bool) -> str:
        """Interpret Moran's I statistic value."""
        if not significant:
            return "No significant spatial autocorrelation detected (random spatial pattern)"
        
        if morans_i > 0.5:
            return "Very strong positive spatial autocorrelation (highly clustered pattern)"
        elif morans_i > 0.3:
            return "Strong positive spatial autocorrelation (clustered pattern)"
        elif morans_i > 0.1:
            return "Moderate positive spatial autocorrelation (some clustering)"
        elif morans_i > -0.1:
            return "Weak spatial autocorrelation"
        elif morans_i > -0.3:
            return "Moderate negative spatial autocorrelation (dispersed pattern)"
        else:
            return "Strong negative spatial autocorrelation (highly dispersed pattern)"
    
    def _interpret_local_morans_i(self, cluster_counts: Dict[str, int], 
                                significant_count: int) -> str:
        """Interpret Local Moran's I results."""
        if significant_count == 0:
            return "No significant local spatial clusters detected"
        
        hh_count = cluster_counts.get("HH", 0)
        ll_count = cluster_counts.get("LL", 0)
        hl_count = cluster_counts.get("HL", 0)
        lh_count = cluster_counts.get("LH", 0)
        
        interpretation = f"Found {significant_count} significant local spatial associations: "
        
        if hh_count > 0:
            interpretation += f"{hh_count} high-value clusters, "
        if ll_count > 0:
            interpretation += f"{ll_count} low-value clusters, "
        if hl_count > 0:
            interpretation += f"{hl_count} high-value outliers, "
        if lh_count > 0:
            interpretation += f"{lh_count} low-value outliers, "
        
        return interpretation.rstrip(", ")
    
    def _create_insufficient_data_result(self, statistic_name: str) -> StatisticalResult:
        """Create result for insufficient data cases."""
        return StatisticalResult(
            statistic_name=statistic_name,
            statistic_value=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Insufficient data for analysis (n < 3)",
            metadata={"error": "insufficient_data"}
        )
    
    def _create_error_result(self, statistic_name: str, error_msg: str) -> StatisticalResult:
        """Create result for error cases."""
        return StatisticalResult(
            statistic_name=statistic_name,
            statistic_value=0.0,
            p_value=1.0,
            significant=False,
            interpretation=f"Analysis failed: {error_msg}",
            metadata={"error": error_msg}
        )


class NullHypothesisTests:
    """
    Comprehensive null hypothesis testing framework including Monte Carlo permutation tests,
    bootstrap resampling, Complete Spatial Randomness testing, and multiple comparison corrections.
    """
    
    def __init__(self, significance_level: float = 0.05, n_bootstrap: int = 1000, n_permutations: int = 999):
        """
        Initialize null hypothesis testing framework.
        
        Args:
            significance_level: Statistical significance threshold
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutation samples
        """
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.logger = logging.getLogger(__name__)
    
    def monte_carlo_permutation_test(self, coordinates: np.ndarray, 
                                   values: np.ndarray,
                                   test_statistic_func: callable,
                                   alternative: str = "two-sided") -> StatisticalResult:
        """
        Perform Monte Carlo permutation test for spatial pattern significance.
        
        Args:
            coordinates: Array of coordinate pairs
            values: Array of values to test
            test_statistic_func: Function to calculate test statistic
            alternative: Type of alternative hypothesis ('two-sided', 'greater', 'less')
            
        Returns:
            StatisticalResult with permutation test results
        """
        try:
            if len(coordinates) < 3:
                return self._create_insufficient_data_result("monte_carlo_permutation")
            
            # Calculate observed test statistic
            observed_stat = test_statistic_func(coordinates, values)
            
            # Generate null distribution through permutation
            null_distribution = []
            
            for _ in range(self.n_permutations):
                # Randomly permute values while keeping coordinates fixed
                permuted_values = np.random.permutation(values)
                
                # Calculate test statistic for permuted data
                permuted_stat = test_statistic_func(coordinates, permuted_values)
                null_distribution.append(permuted_stat)
            
            null_distribution = np.array(null_distribution)
            
            # Calculate p-value based on alternative hypothesis
            if alternative == "two-sided":
                n_more_extreme = np.sum(np.abs(null_distribution) >= np.abs(observed_stat))
            elif alternative == "greater":
                n_more_extreme = np.sum(null_distribution >= observed_stat)
            else:  # "less"
                n_more_extreme = np.sum(null_distribution <= observed_stat)
            
            # Pseudo p-value with continuity correction
            p_value = (n_more_extreme + 1) / (self.n_permutations + 1)
            
            # Calculate effect size (Cohen's d equivalent)
            null_std = np.std(null_distribution)
            if null_std > 0:
                effect_size = abs(observed_stat - np.mean(null_distribution)) / null_std
            else:
                effect_size = 0.0
            
            # Calculate confidence interval
            alpha = self.significance_level
            if alternative == "two-sided":
                ci_lower = np.percentile(null_distribution, 100 * alpha / 2)
                ci_upper = np.percentile(null_distribution, 100 * (1 - alpha / 2))
            else:
                ci_lower = np.percentile(null_distribution, 100 * alpha)
                ci_upper = np.percentile(null_distribution, 100 * (1 - alpha))
            
            return StatisticalResult(
                statistic_name="monte_carlo_permutation",
                statistic_value=observed_stat,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                interpretation=self._interpret_permutation_test(p_value, observed_stat, effect_size),
                significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                metadata={
                    "n_permutations": self.n_permutations,
                    "alternative": alternative,
                    "null_mean": np.mean(null_distribution),
                    "null_std": null_std,
                    "n_more_extreme": n_more_extreme
                }
            )
            
        except Exception as e:
            self.logger.error(f"Monte Carlo permutation test failed: {e}")
            return self._create_error_result("monte_carlo_permutation", str(e))
    
    def bootstrap_confidence_intervals(self, data: np.ndarray,
                                     statistic_func: callable,
                                     confidence_level: float = 0.95,
                                     method: str = "percentile") -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals for a statistic.
        
        Args:
            data: Input data array
            statistic_func: Function to calculate statistic
            confidence_level: Confidence level for intervals
            method: Bootstrap CI method ('percentile', 'bias-corrected', 'bca')
            
        Returns:
            Dictionary with bootstrap results
        """
        try:
            if len(data) < 3:
                return {"error": "Insufficient data for bootstrap analysis"}
            
            # Calculate observed statistic
            observed_stat = statistic_func(data)
            
            # Generate bootstrap samples
            bootstrap_stats = []
            
            for _ in range(self.n_bootstrap):
                # Bootstrap resample
                bootstrap_sample = resample(data, replace=True, n_samples=len(data))
                
                # Calculate statistic for bootstrap sample
                bootstrap_stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
            
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            
            if method == "percentile":
                ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
                ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            elif method == "bias-corrected":
                # Bias-corrected percentile method
                bias_correction = stats.norm.ppf((bootstrap_stats < observed_stat).mean())
                
                alpha1 = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(alpha / 2))
                alpha2 = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(1 - alpha / 2))
                
                ci_lower = np.percentile(bootstrap_stats, 100 * alpha1)
                ci_upper = np.percentile(bootstrap_stats, 100 * alpha2)
            else:
                # Default to percentile method
                ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
                ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            
            # Calculate bootstrap standard error
            bootstrap_se = np.std(bootstrap_stats)
            
            # Bias estimate
            bias = np.mean(bootstrap_stats) - observed_stat
            
            return {
                "observed_statistic": observed_stat,
                "bootstrap_mean": np.mean(bootstrap_stats),
                "bootstrap_std": np.std(bootstrap_stats),
                "bootstrap_se": bootstrap_se,
                "bias": bias,
                "confidence_interval": (ci_lower, ci_upper),
                "confidence_level": confidence_level,
                "n_bootstrap": self.n_bootstrap,
                "method": method,
                "bootstrap_distribution": bootstrap_stats
            }
            
        except Exception as e:
            self.logger.error(f"Bootstrap analysis failed: {e}")
            return {"error": str(e)}
    
    def complete_spatial_randomness_test(self, coordinates: np.ndarray,
                                       study_area: Optional[Tuple[float, float, float, float]] = None) -> StatisticalResult:
        """
        Test Complete Spatial Randomness (CSR) using Ripley's K function.
        
        Args:
            coordinates: Array of point coordinates
            study_area: Bounding box (min_x, min_y, max_x, max_y) for the study area
            
        Returns:
            StatisticalResult with CSR test results
        """
        try:
            n = len(coordinates)
            if n < 3:
                return self._create_insufficient_data_result("csr_test")
            
            # Define study area if not provided
            if study_area is None:
                min_x, min_y = coordinates.min(axis=0)
                max_x, max_y = coordinates.max(axis=0)
                # Add buffer to avoid edge effects
                buffer = 0.1 * max(max_x - min_x, max_y - min_y)
                study_area = (min_x - buffer, min_y - buffer, max_x + buffer, max_y + buffer)
            
            area = (study_area[2] - study_area[0]) * (study_area[3] - study_area[1])
            intensity = n / area  # Point intensity
            
            # Calculate Ripley's K function at multiple distance bands
            max_distance = min(
                (study_area[2] - study_area[0]) / 4,
                (study_area[3] - study_area[1]) / 4
            )
            distances = np.linspace(0.1 * max_distance, max_distance, 10)
            
            k_observed = []
            k_expected = []  # K(r) = πr² under CSR
            
            for r in distances:
                # Count pairs within distance r
                pair_count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                        if dist <= r:
                            pair_count += 1
                
                # Ripley's K estimator
                k_r = (area * 2 * pair_count) / (n * (n - 1))
                k_observed.append(k_r)
                
                # Expected K under CSR
                k_expected.append(np.pi * r**2)
            
            # L function: L(r) = sqrt(K(r)/π) - r
            l_observed = [np.sqrt(k / np.pi) - r for k, r in zip(k_observed, distances)]
            l_expected = [0.0] * len(distances)  # L(r) = 0 under CSR
            
            # Test statistic: maximum deviation from expected
            max_deviation = max(abs(l_obs) for l_obs in l_observed)
            
            # Monte Carlo test for significance
            mc_deviations = []
            for _ in range(min(self.n_permutations, 99)):  # Reduced for performance
                # Generate random points in study area
                random_x = np.random.uniform(study_area[0], study_area[2], n)
                random_y = np.random.uniform(study_area[1], study_area[3], n)
                random_coords = np.column_stack([random_x, random_y])
                
                # Calculate L function for random pattern
                random_l = []
                for r in distances:
                    pair_count = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            dist = np.sqrt(np.sum((random_coords[i] - random_coords[j])**2))
                            if dist <= r:
                                pair_count += 1
                    
                    k_r = (area * 2 * pair_count) / (n * (n - 1)) if n > 1 else 0
                    l_r = np.sqrt(k_r / np.pi) - r if k_r > 0 else -r
                    random_l.append(l_r)
                
                random_max_deviation = max(abs(l) for l in random_l)
                mc_deviations.append(random_max_deviation)
            
            # Calculate p-value
            n_more_extreme = sum(1 for dev in mc_deviations if dev >= max_deviation)
            p_value = (n_more_extreme + 1) / (len(mc_deviations) + 1)
            
            # Determine pattern type
            if max(l_observed) > 0.1:
                pattern_type = "clustered"
            elif min(l_observed) < -0.1:
                pattern_type = "dispersed"
            else:
                pattern_type = "random"
            
            return StatisticalResult(
                statistic_name="csr_ripley_k",
                statistic_value=max_deviation,
                p_value=p_value,
                interpretation=f"Spatial pattern appears {pattern_type} (max L-function deviation: {max_deviation:.3f})",
                significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                metadata={
                    "pattern_type": pattern_type,
                    "distances": distances.tolist(),
                    "k_observed": k_observed,
                    "k_expected": k_expected,
                    "l_observed": l_observed,
                    "study_area": study_area,
                    "intensity": intensity,
                    "n_monte_carlo": len(mc_deviations)
                }
            )
            
        except Exception as e:
            self.logger.error(f"CSR test failed: {e}")
            return self._create_error_result("csr_ripley_k", str(e))
    
    def nearest_neighbor_analysis(self, coordinates: np.ndarray,
                                study_area: Optional[Tuple[float, float, float, float]] = None) -> StatisticalResult:
        """
        Perform nearest neighbor analysis for spatial randomness assessment.
        
        Args:
            coordinates: Array of point coordinates
            study_area: Bounding box for the study area
            
        Returns:
            StatisticalResult with nearest neighbor analysis
        """
        try:
            n = len(coordinates)
            if n < 2:
                return self._create_insufficient_data_result("nearest_neighbor")
            
            # Calculate nearest neighbor distances
            nn_distances = []
            for i in range(n):
                distances = [np.linalg.norm(coordinates[i] - coordinates[j]) 
                           for j in range(n) if i != j]
                nn_distances.append(min(distances))
            
            # Observed mean nearest neighbor distance
            observed_mean = np.mean(nn_distances)
            
            # Calculate expected mean under CSR
            if study_area is None:
                min_x, min_y = coordinates.min(axis=0)
                max_x, max_y = coordinates.max(axis=0)
                area = (max_x - min_x) * (max_y - min_y)
            else:
                area = (study_area[2] - study_area[0]) * (study_area[3] - study_area[1])
            
            density = n / area if area > 0 else 1.0
            expected_mean = 1.0 / (2 * np.sqrt(density)) if density > 0 else 1.0
            
            # Nearest neighbor ratio
            nn_ratio = observed_mean / expected_mean if expected_mean > 0 else 1.0
            
            # Standard error and z-score
            se = 0.26136 / np.sqrt(n * density) if n > 0 and density > 0 else 1.0
            z_score = (observed_mean - expected_mean) / se if se > 0 else 0.0
            
            # P-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if z_score != 0 else 1.0
            
            # Interpretation
            if nn_ratio < 1.0:
                pattern_interpretation = f"Clustered pattern (R = {nn_ratio:.3f})"
            elif nn_ratio > 1.0:
                pattern_interpretation = f"Dispersed pattern (R = {nn_ratio:.3f})"
            else:
                pattern_interpretation = f"Random pattern (R = {nn_ratio:.3f})"
            
            return StatisticalResult(
                statistic_name="nearest_neighbor_ratio",
                statistic_value=nn_ratio,
                p_value=p_value,
                z_score=z_score,
                interpretation=pattern_interpretation,
                significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                metadata={
                    "observed_mean": observed_mean,
                    "expected_mean": expected_mean,
                    "standard_error": se,
                    "sample_size": n,
                    "study_area": area,
                    "density": density
                }
            )
            
        except Exception as e:
            self.logger.error(f"Nearest neighbor analysis failed: {e}")
            return self._create_error_result("nearest_neighbor_ratio", str(e))
    
    def multiple_comparison_correction(self, p_values: List[float],
                                     method: str = "bonferroni") -> Dict[str, Any]:
        """
        Apply multiple comparison corrections to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Dictionary with corrected p-values and results
        """
        try:
            if not p_values:
                return {"error": "No p-values provided"}
            
            p_array = np.array(p_values)
            n = len(p_array)
            
            if method == "bonferroni":
                corrected_p = np.minimum(p_array * n, 1.0)
                corrected_alpha = self.significance_level / n
            
            elif method == "holm":
                # Holm-Bonferroni method
                sorted_indices = np.argsort(p_array)
                corrected_p = np.zeros_like(p_array)
                
                for i, idx in enumerate(sorted_indices):
                    correction_factor = n - i
                    corrected_p[idx] = min(p_array[idx] * correction_factor, 1.0)
                
                corrected_alpha = self.significance_level
            
            elif method == "fdr_bh":
                # Benjamini-Hochberg FDR correction
                sorted_indices = np.argsort(p_array)
                corrected_p = np.zeros_like(p_array)
                
                for i in range(n):
                    idx = sorted_indices[i]
                    corrected_p[idx] = min(p_array[idx] * n / (i + 1), 1.0)
                
                corrected_alpha = self.significance_level
            
            else:
                # No correction
                corrected_p = p_array.copy()
                corrected_alpha = self.significance_level
            
            # Determine significance with correction
            significant = corrected_p < corrected_alpha
            
            return {
                "method": method,
                "original_p_values": p_values,
                "corrected_p_values": corrected_p.tolist(),
                "corrected_alpha": corrected_alpha,
                "significant": significant.tolist(),
                "n_significant_original": np.sum(np.array(p_values) < self.significance_level),
                "n_significant_corrected": np.sum(significant),
                "family_wise_error_rate": corrected_alpha if method == "bonferroni" else None
            }
            
        except Exception as e:
            self.logger.error(f"Multiple comparison correction failed: {e}")
            return {"error": str(e)}
    
    def _interpret_permutation_test(self, p_value: float, 
                                  observed_stat: float, 
                                  effect_size: float) -> str:
        """Interpret permutation test results."""
        significance = "significant" if p_value < self.significance_level else "not significant"
        
        if effect_size < 0.2:
            effect = "negligible"
        elif effect_size < 0.5:
            effect = "small"
        elif effect_size < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        return f"Pattern is {significance} (p = {p_value:.3f}) with {effect} effect size ({effect_size:.3f})"
    
    def _create_insufficient_data_result(self, statistic_name: str) -> StatisticalResult:
        """Create result for insufficient data cases."""
        return StatisticalResult(
            statistic_name=statistic_name,
            statistic_value=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Insufficient data for analysis",
            metadata={"error": "insufficient_data"}
        )
    
    def _create_error_result(self, statistic_name: str, error_msg: str) -> StatisticalResult:
        """Create result for error cases."""
        return StatisticalResult(
            statistic_name=statistic_name,
            statistic_value=0.0,
            p_value=1.0,
            significant=False,
            interpretation=f"Analysis failed: {error_msg}",
            metadata={"error": error_msg}
        )


# Continue with SpatialStatistics and other classes...
# (Due to length limits, I'll create this as a separate file or continue in the next part)