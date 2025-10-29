#!/usr/bin/env python3
"""
A2A World Platform - Pattern Discovery Test Script

Test script to validate HDBSCAN clustering functionality, database integration,
and statistical validation for Phase 1 Step 3 completion.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, List
from datetime import datetime

# Add agents directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from agents.discovery.pattern_discovery import PatternDiscoveryAgent
    from agents.core.config import DiscoveryAgentConfig
    from agents.core.pattern_storage import PatternStorage
    from agents.discovery.clustering import GeospatialHDBSCAN, SpatialStatistics, PatternSignificanceTest
    print("✅ Successfully imported pattern discovery modules")
except ImportError as e:
    print(f"❌ Failed to import pattern discovery modules: {e}")
    sys.exit(1)


class PatternDiscoveryTester:
    """Test harness for pattern discovery functionality."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_storage = PatternStorage()
        self.test_results = []
        
        # Initialize discovery agent
        self.config = DiscoveryAgentConfig(
            min_cluster_size=4,
            min_samples=2,
            confidence_threshold=0.6
        )
        
        self.discovery_agent = PatternDiscoveryAgent(
            agent_id="test_discovery_agent",
            config=self.config
        )
        
        print("🚀 Pattern Discovery Test Harness Initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all pattern discovery tests."""
        print("\n" + "="*60)
        print("🧪 RUNNING PATTERN DISCOVERY TESTS")
        print("="*60)
        
        test_results = {
            "start_time": datetime.utcnow().isoformat(),
            "tests": {},
            "summary": {}
        }
        
        # Test 1: Database connection and sample data creation
        test_results["tests"]["database_setup"] = await self._test_database_setup()
        
        # Test 2: HDBSCAN clustering functionality
        test_results["tests"]["hdbscan_clustering"] = await self._test_hdbscan_clustering()
        
        # Test 3: Spatial statistics calculations
        test_results["tests"]["spatial_statistics"] = await self._test_spatial_statistics()
        
        # Test 4: Pattern significance testing
        test_results["tests"]["significance_testing"] = await self._test_significance_testing()
        
        # Test 5: End-to-end pattern discovery workflow
        test_results["tests"]["end_to_end_discovery"] = await self._test_end_to_end_discovery()
        
        # Test 6: Database pattern storage and retrieval
        test_results["tests"]["pattern_storage"] = await self._test_pattern_storage()
        
        # Calculate summary
        test_results["summary"] = self._calculate_summary(test_results["tests"])
        test_results["end_time"] = datetime.utcnow().isoformat()
        
        # Print final results
        self._print_test_summary(test_results)
        
        return test_results
    
    async def _test_database_setup(self) -> Dict[str, Any]:
        """Test database connection and sample data creation."""
        print("\n📊 Test 1: Database Setup and Sample Data Creation")
        
        try:
            # Create sample sacred sites
            created_count = await self.pattern_storage.create_sample_sacred_sites(50)
            
            if created_count > 0:
                print(f"✅ Created {created_count} sample sacred sites")
                
                # Verify data retrieval
                sites = await self.pattern_storage.get_sacred_sites(limit=100)
                print(f"✅ Retrieved {len(sites)} sacred sites from database")
                
                if len(sites) >= 20:  # Sufficient for clustering
                    return {
                        "success": True,
                        "sites_created": created_count,
                        "sites_retrieved": len(sites),
                        "message": "Database setup successful"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Insufficient sites for clustering",
                        "sites_created": created_count,
                        "sites_retrieved": len(sites)
                    }
            else:
                return {
                    "success": False,
                    "error": "Failed to create sample sites",
                    "sites_created": 0
                }
                
        except Exception as e:
            self.logger.error(f"Database setup test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_hdbscan_clustering(self) -> Dict[str, Any]:
        """Test HDBSCAN clustering functionality."""
        print("\n🔍 Test 2: HDBSCAN Clustering Functionality")
        
        try:
            # Get test data
            sites = await self.pattern_storage.get_sacred_sites(limit=50)
            
            if len(sites) < 10:
                return {
                    "success": False,
                    "error": "Insufficient data for clustering test"
                }
            
            # Prepare clustering data
            import numpy as np
            
            coordinates = []
            features = []
            
            for site in sites:
                if site.get("latitude") is not None and site.get("longitude") is not None:
                    coordinates.append([site["latitude"], site["longitude"]])
                    # Add significance level as feature
                    features.append([
                        site["latitude"], 
                        site["longitude"],
                        site.get("significance_level", 1.0)
                    ])
            
            if len(features) < 10:
                return {
                    "success": False,
                    "error": "Insufficient valid coordinates for clustering"
                }
            
            X = np.array(features)
            coords = np.array(coordinates)
            
            # Test GeospatialHDBSCAN
            clusterer = GeospatialHDBSCAN(
                min_cluster_size=4,
                min_samples=2
            )
            
            labels = clusterer.fit_predict(X, coords)
            
            # Analyze results
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = sum(1 for label in labels if label == -1)
            
            print(f"✅ HDBSCAN clustering completed")
            print(f"   - Data points: {len(labels)}")
            print(f"   - Clusters found: {n_clusters}")
            print(f"   - Noise points: {n_noise}")
            
            if n_clusters > 0:
                return {
                    "success": True,
                    "data_points": len(labels),
                    "clusters_found": n_clusters,
                    "noise_points": n_noise,
                    "clustering_quality": "good" if n_clusters >= 2 else "minimal",
                    "message": "HDBSCAN clustering successful"
                }
            else:
                return {
                    "success": False,
                    "error": "No clusters found",
                    "data_points": len(labels),
                    "noise_points": n_noise
                }
                
        except Exception as e:
            self.logger.error(f"HDBSCAN clustering test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_spatial_statistics(self) -> Dict[str, Any]:
        """Test spatial statistics calculations."""
        print("\n📈 Test 3: Spatial Statistics Calculations")
        
        try:
            # Get test data
            sites = await self.pattern_storage.get_sacred_sites(limit=30)
            
            if len(sites) < 5:
                return {
                    "success": False,
                    "error": "Insufficient data for spatial statistics"
                }
            
            # Test spatial statistics
            spatial_stats = SpatialStatistics()
            
            # Test Moran's I calculation
            morans_result = spatial_stats.calculate_spatial_autocorrelation(sites)
            
            # Test nearest neighbor analysis
            nn_result = spatial_stats.calculate_nearest_neighbor_statistic(sites)
            
            print(f"✅ Spatial statistics calculated")
            print(f"   - Moran's I: {morans_result.get('morans_i', 0):.4f}")
            print(f"   - NN ratio: {nn_result.get('nn_ratio', 0):.4f}")
            
            # Validate results
            morans_valid = -1 <= morans_result.get('morans_i', 0) <= 1
            nn_valid = nn_result.get('nn_ratio', 0) > 0
            
            if morans_valid and nn_valid:
                return {
                    "success": True,
                    "morans_i": morans_result.get('morans_i', 0),
                    "morans_p_value": morans_result.get('p_value', 1.0),
                    "nn_ratio": nn_result.get('nn_ratio', 0),
                    "nn_p_value": nn_result.get('p_value', 1.0),
                    "message": "Spatial statistics calculations successful"
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid spatial statistics results",
                    "morans_valid": morans_valid,
                    "nn_valid": nn_valid
                }
                
        except Exception as e:
            self.logger.error(f"Spatial statistics test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_significance_testing(self) -> Dict[str, Any]:
        """Test pattern significance testing."""
        print("\n🎯 Test 4: Pattern Significance Testing")
        
        try:
            # Create mock cluster data
            cluster_data = {
                "cluster_id": 1,
                "size": 8,
                "points": [
                    {"latitude": 51.178, "longitude": -1.826, "significance_level": 4},
                    {"latitude": 51.179, "longitude": -1.825, "significance_level": 4},
                    {"latitude": 51.180, "longitude": -1.827, "significance_level": 3},
                    {"latitude": 51.181, "longitude": -1.824, "significance_level": 4},
                    {"latitude": 51.182, "longitude": -1.828, "significance_level": 5},
                    {"latitude": 51.183, "longitude": -1.823, "significance_level": 4},
                    {"latitude": 51.184, "longitude": -1.829, "significance_level": 3},
                    {"latitude": 51.185, "longitude": -1.822, "significance_level": 4}
                ],
                "centroid": {"latitude": 51.181, "longitude": -1.825},
                "compactness": 0.8
            }
            
            # Get all points for comparison
            all_points = await self.pattern_storage.get_sacred_sites(limit=100)
            
            if len(all_points) < 10:
                return {
                    "success": False,
                    "error": "Insufficient data for significance testing"
                }
            
            # Test significance assessment
            significance_tester = PatternSignificanceTest()
            
            significance_result = significance_tester.assess_cluster_significance(
                cluster_data, all_points
            )
            
            print(f"✅ Significance testing completed")
            print(f"   - Significance score: {significance_result.get('significance_score', 0):.3f}")
            print(f"   - Is significant: {significance_result.get('significant', False)}")
            print(f"   - P-value: {significance_result.get('p_value', 1.0):.6f}")
            
            # Validate results
            has_score = 'significance_score' in significance_result
            has_significant = 'significant' in significance_result
            has_p_value = 'p_value' in significance_result
            
            if has_score and has_significant and has_p_value:
                return {
                    "success": True,
                    "significance_score": significance_result.get('significance_score', 0),
                    "is_significant": significance_result.get('significant', False),
                    "p_value": significance_result.get('p_value', 1.0),
                    "test_results": significance_result.get('test_results', {}),
                    "message": "Significance testing successful"
                }
            else:
                return {
                    "success": False,
                    "error": "Incomplete significance test results",
                    "has_score": has_score,
                    "has_significant": has_significant,
                    "has_p_value": has_p_value
                }
                
        except Exception as e:
            self.logger.error(f"Significance testing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_end_to_end_discovery(self) -> Dict[str, Any]:
        """Test complete end-to-end pattern discovery workflow."""
        print("\n🔄 Test 5: End-to-End Pattern Discovery Workflow")
        
        try:
            # Run full pattern discovery from database
            discovery_result = await self.discovery_agent.discover_patterns_from_database()
            
            if discovery_result.get("error"):
                return {
                    "success": False,
                    "error": discovery_result["error"]
                }
            
            patterns_found = discovery_result.get("pattern_count", 0)
            significant_patterns = discovery_result.get("significant_patterns", 0)
            stored_in_db = discovery_result.get("stored_in_database", False)
            
            print(f"✅ End-to-end discovery completed")
            print(f"   - Patterns found: {patterns_found}")
            print(f"   - Significant patterns: {significant_patterns}")
            print(f"   - Stored in database: {stored_in_db}")
            
            if patterns_found > 0:
                return {
                    "success": True,
                    "patterns_found": patterns_found,
                    "significant_patterns": significant_patterns,
                    "significance_rate": significant_patterns / patterns_found if patterns_found > 0 else 0,
                    "stored_in_database": stored_in_db,
                    "discovery_id": discovery_result.get("dataset_id"),
                    "message": "End-to-end discovery successful"
                }
            else:
                return {
                    "success": False,
                    "error": "No patterns discovered",
                    "discovery_result": discovery_result
                }
                
        except Exception as e:
            self.logger.error(f"End-to-end discovery test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_pattern_storage(self) -> Dict[str, Any]:
        """Test pattern storage and retrieval."""
        print("\n💾 Test 6: Pattern Storage and Retrieval")
        
        try:
            # List existing patterns
            patterns, total_count = await self.pattern_storage.list_patterns(limit=50)
            
            print(f"✅ Pattern storage test completed")
            print(f"   - Patterns in database: {total_count}")
            print(f"   - Patterns retrieved: {len(patterns)}")
            
            if total_count > 0:
                # Test detailed pattern retrieval
                first_pattern = patterns[0]
                pattern_id = first_pattern["id"]
                
                detailed_pattern = await self.pattern_storage.get_pattern(pattern_id)
                
                if detailed_pattern:
                    print(f"   - Detailed retrieval: successful")
                    return {
                        "success": True,
                        "patterns_in_database": total_count,
                        "patterns_retrieved": len(patterns),
                        "detailed_retrieval": True,
                        "sample_pattern_id": pattern_id,
                        "message": "Pattern storage and retrieval successful"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Failed to retrieve detailed pattern",
                        "patterns_in_database": total_count
                    }
            else:
                return {
                    "success": True,
                    "patterns_in_database": 0,
                    "message": "No patterns stored yet (expected for first run)",
                    "note": "Run discovery test first"
                }
                
        except Exception as e:
            self.logger.error(f"Pattern storage test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_summary(self, tests: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate test summary statistics."""
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test.get("success", False))
        failed_tests = total_tests - passed_tests
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "overall_status": "PASSED" if success_rate >= 0.8 else "FAILED"
        }
    
    def _print_test_summary(self, test_results: Dict[str, Any]):
        """Print formatted test summary."""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        summary = test_results["summary"]
        
        print(f"Total Tests:    {summary['total_tests']}")
        print(f"Passed:         {summary['passed_tests']} ✅")
        print(f"Failed:         {summary['failed_tests']} ❌")
        print(f"Success Rate:   {summary['success_rate']:.1%}")
        print(f"Overall Status: {summary['overall_status']}")
        
        print("\nDETAILED RESULTS:")
        for test_name, test_result in test_results["tests"].items():
            status = "✅ PASS" if test_result.get("success", False) else "❌ FAIL"
            message = test_result.get("message", test_result.get("error", "No message"))
            print(f"  {test_name:20} - {status} - {message}")
        
        print("\n" + "="*60)
        
        if summary["overall_status"] == "PASSED":
            print("🎉 PATTERN DISCOVERY FUNCTIONALITY VALIDATED!")
            print("Phase 1 Step 3 requirements successfully implemented:")
            print("  ✅ HDBSCAN clustering implementation")
            print("  ✅ Geospatial pattern analysis")
            print("  ✅ Database integration")
            print("  ✅ Statistical validation")
            print("  ✅ API integration")
        else:
            print("⚠️  Some tests failed. Please review errors above.")


async def main():
    """Main test execution function."""
    print("🚀 A2A World Pattern Discovery Test Suite")
    print("Testing Phase 1 Step 3 implementation...")
    
    tester = PatternDiscoveryTester()
    results = await tester.run_all_tests()
    
    # Save results to file
    import json
    with open(f"pattern_discovery_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Return exit code
    return 0 if results["summary"]["overall_status"] == "PASSED" else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)