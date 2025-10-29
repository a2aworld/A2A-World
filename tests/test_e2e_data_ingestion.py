"""
End-to-End Tests for Enhanced Data Ingestion System

Tests the complete workflow from file upload through processing to database storage,
including performance benchmarks and load testing scenarios.
"""

import pytest
import asyncio
import time
import statistics
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import requests
import json

# Test configuration
TEST_CONFIG = {
    'api_base_url': 'http://localhost:8000/api/v1',
    'test_timeout': 300,  # 5 minutes
    'concurrent_uploads': 5,
    'large_file_threshold': 50 * 1024 * 1024,  # 50MB
}

# Sample test data
SAMPLE_KML_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>E2E Test Sites</name>
    <Placemark>
      <name>Test Site 1</name>
      <description>First test site for E2E testing</description>
      <Point>
        <coordinates>-74.0060,40.7128,0</coordinates>
      </Point>
    </Placemark>
    <Placemark>
      <name>Test Site 2</name>
      <description>Second test site for E2E testing</description>
      <Point>
        <coordinates>-73.9857,40.7489,0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>"""

SAMPLE_GEOJSON_CONTENT = {
    "type": "FeatureCollection",
    "name": "E2E Test Features",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "Test Feature 1",
                "description": "First test feature for E2E testing"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [-74.0060, 40.7128]
            }
        },
        {
            "type": "Feature", 
            "properties": {
                "name": "Test Feature 2",
                "description": "Second test feature for E2E testing"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [-73.9857, 40.7489]
            }
        }
    ]
}

SAMPLE_CSV_CONTENT = """name,latitude,longitude,description,culture
Test Site 1,40.7128,-74.0060,First CSV test site,Test Culture
Test Site 2,40.7489,-73.9857,Second CSV test site,Test Culture
Test Site 3,40.7614,-73.9776,Third CSV test site,Test Culture"""


class E2ETestRunner:
    """End-to-end test runner with performance monitoring."""
    
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.test_results = {}
        self.performance_metrics = {}
        
    def create_test_file(self, content: str, extension: str) -> str:
        """Create a temporary test file."""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=extension, 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(content)
            return f.name
    
    def upload_file(self, file_path: str, timeout: int = 60) -> Dict[str, Any]:
        """Upload a file and return the upload result."""
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/data/upload",
                files=files,
                timeout=timeout
            )
            upload_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result['upload_time'] = upload_time
                return result
            else:
                raise Exception(f"Upload failed: {response.status_code} - {response.text}")
    
    def poll_upload_status(self, upload_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Poll upload status until completion or timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.api_base_url}/data/upload/{upload_id}/status")
            
            if response.status_code == 200:
                status = response.json()
                
                if status['status'] in ['completed', 'error']:
                    status['total_processing_time'] = time.time() - start_time
                    return status
                
                # Wait before next poll
                time.sleep(2)
            else:
                raise Exception(f"Status check failed: {response.status_code}")
        
        raise TimeoutError(f"Upload processing timed out after {timeout} seconds")
    
    def test_file_upload_workflow(self, file_path: str, expected_features: int) -> Dict[str, Any]:
        """Test complete file upload workflow."""
        test_start = time.time()
        
        try:
            # Step 1: Upload file
            upload_result = self.upload_file(file_path)
            upload_id = upload_result['upload_id']
            
            # Step 2: Poll for completion
            final_status = self.poll_upload_status(upload_id)
            
            # Step 3: Validate results
            if final_status['status'] == 'completed':
                result = final_status['result']
                
                # Verify feature count
                actual_features = result.get('features_count', 0)
                if actual_features != expected_features:
                    raise AssertionError(
                        f"Expected {expected_features} features, got {actual_features}"
                    )
                
                # Check quality score if available
                quality_score = result.get('quality_score')
                if quality_score and quality_score < 0.5:
                    print(f"Warning: Low quality score {quality_score}")
                
                return {
                    'success': True,
                    'upload_time': upload_result['upload_time'],
                    'processing_time': final_status['total_processing_time'],
                    'total_time': time.time() - test_start,
                    'features_count': actual_features,
                    'quality_score': quality_score,
                    'file_size': os.path.getsize(file_path),
                    'result': result
                }
            else:
                raise Exception(f"Processing failed: {final_status.get('error', 'Unknown error')}")
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - test_start,
                'file_size': os.path.getsize(file_path)
            }


class PerformanceBenchmarks:
    """Performance benchmarking for the data ingestion system."""
    
    def __init__(self, test_runner: E2ETestRunner):
        self.test_runner = test_runner
        self.benchmarks = {}
    
    def benchmark_file_sizes(self) -> Dict[str, Any]:
        """Benchmark processing times for different file sizes."""
        file_sizes = [
            (1, "small"),      # 1 feature
            (10, "medium"),    # 10 features  
            (100, "large"),    # 100 features
            (1000, "xlarge")   # 1000 features
        ]
        
        results = {}
        
        for feature_count, size_label in file_sizes:
            print(f"Benchmarking {size_label} file ({feature_count} features)...")
            
            # Generate test file with specified number of features
            test_file = self._generate_kml_file(feature_count)
            
            try:
                result = self.test_runner.test_file_upload_workflow(test_file, feature_count)
                results[size_label] = {
                    'feature_count': feature_count,
                    'file_size': result['file_size'],
                    'processing_time': result.get('processing_time', 0),
                    'total_time': result.get('total_time', 0),
                    'success': result['success'],
                    'throughput_features_per_sec': (
                        feature_count / result['processing_time'] 
                        if result.get('processing_time') and result['processing_time'] > 0 
                        else 0
                    )
                }
            except Exception as e:
                results[size_label] = {
                    'feature_count': feature_count,
                    'error': str(e),
                    'success': False
                }
            finally:
                os.unlink(test_file)
        
        return results
    
    def benchmark_concurrent_uploads(self, num_concurrent: int = 5) -> Dict[str, Any]:
        """Benchmark concurrent file uploads."""
        print(f"Benchmarking {num_concurrent} concurrent uploads...")
        
        # Create test files
        test_files = []
        for i in range(num_concurrent):
            test_file = self._generate_kml_file(10, f"Concurrent Test {i+1}")
            test_files.append(test_file)
        
        start_time = time.time()
        results = []
        
        # Run concurrent uploads
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = []
            for test_file in test_files:
                future = executor.submit(
                    self.test_runner.test_file_upload_workflow, 
                    test_file, 
                    10
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    results.append({'success': False, 'error': str(e)})
        
        total_time = time.time() - start_time
        
        # Cleanup test files
        for test_file in test_files:
            try:
                os.unlink(test_file)
            except:
                pass
        
        successful_uploads = [r for r in results if r['success']]
        failed_uploads = [r for r in results if not r['success']]
        
        return {
            'total_time': total_time,
            'concurrent_uploads': num_concurrent,
            'successful_uploads': len(successful_uploads),
            'failed_uploads': len(failed_uploads),
            'success_rate': len(successful_uploads) / num_concurrent,
            'average_processing_time': (
                statistics.mean([r.get('processing_time', 0) for r in successful_uploads])
                if successful_uploads else 0
            ),
            'results': results
        }
    
    def benchmark_file_formats(self) -> Dict[str, Any]:
        """Benchmark processing times for different file formats."""
        formats = [
            ('kml', SAMPLE_KML_CONTENT, '.kml', 2),
            ('geojson', json.dumps(SAMPLE_GEOJSON_CONTENT), '.geojson', 2),
            ('csv', SAMPLE_CSV_CONTENT, '.csv', 3)
        ]
        
        results = {}
        
        for format_name, content, extension, expected_features in formats:
            print(f"Benchmarking {format_name} format...")
            
            test_file = self.test_runner.create_test_file(content, extension)
            
            try:
                result = self.test_runner.test_file_upload_workflow(test_file, expected_features)
                results[format_name] = {
                    'file_size': result['file_size'],
                    'processing_time': result.get('processing_time', 0),
                    'features_per_second': (
                        expected_features / result['processing_time']
                        if result.get('processing_time') and result['processing_time'] > 0
                        else 0
                    ),
                    'success': result['success']
                }
            except Exception as e:
                results[format_name] = {
                    'error': str(e),
                    'success': False
                }
            finally:
                os.unlink(test_file)
        
        return results
    
    def _generate_kml_file(self, feature_count: int, name_prefix: str = "Generated Site") -> str:
        """Generate a KML file with specified number of features."""
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Generated Test File - {feature_count} Features</name>"""
        
        for i in range(feature_count):
            # Generate random coordinates around NYC
            import random
            lat = 40.7 + (random.random() - 0.5) * 0.1  # ~40.65 to 40.75
            lon = -73.9 + (random.random() - 0.5) * 0.1  # ~-73.95 to -73.85
            
            kml_content += f"""
    <Placemark>
      <name>{name_prefix} {i+1}</name>
      <description>Generated test site number {i+1}</description>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>"""
        
        kml_content += """
  </Document>
</kml>"""
        
        return self.test_runner.create_test_file(kml_content, '.kml')


def run_comprehensive_e2e_tests():
    """Run comprehensive end-to-end tests."""
    print("Starting comprehensive E2E tests for Enhanced Data Ingestion System")
    print("=" * 80)
    
    # Initialize test runner
    test_runner = E2ETestRunner(TEST_CONFIG['api_base_url'])
    benchmarks = PerformanceBenchmarks(test_runner)
    
    all_results = {}
    
    # Test 1: Basic file format support
    print("\n1. Testing Basic File Format Support")
    print("-" * 40)
    
    format_tests = {}
    
    # KML test
    kml_file = test_runner.create_test_file(SAMPLE_KML_CONTENT, '.kml')
    try:
        kml_result = test_runner.test_file_upload_workflow(kml_file, 2)
        format_tests['kml'] = kml_result
        print(f"✓ KML: {kml_result['success']} - {kml_result.get('processing_time', 0):.2f}s")
    except Exception as e:
        format_tests['kml'] = {'success': False, 'error': str(e)}
        print(f"✗ KML: Failed - {e}")
    finally:
        os.unlink(kml_file)
    
    # GeoJSON test  
    geojson_file = test_runner.create_test_file(json.dumps(SAMPLE_GEOJSON_CONTENT), '.geojson')
    try:
        geojson_result = test_runner.test_file_upload_workflow(geojson_file, 2)
        format_tests['geojson'] = geojson_result
        print(f"✓ GeoJSON: {geojson_result['success']} - {geojson_result.get('processing_time', 0):.2f}s")
    except Exception as e:
        format_tests['geojson'] = {'success': False, 'error': str(e)}
        print(f"✗ GeoJSON: Failed - {e}")
    finally:
        os.unlink(geojson_file)
    
    # CSV test
    csv_file = test_runner.create_test_file(SAMPLE_CSV_CONTENT, '.csv')
    try:
        csv_result = test_runner.test_file_upload_workflow(csv_file, 3)
        format_tests['csv'] = csv_result
        print(f"✓ CSV: {csv_result['success']} - {csv_result.get('processing_time', 0):.2f}s")
    except Exception as e:
        format_tests['csv'] = {'success': False, 'error': str(e)}
        print(f"✗ CSV: Failed - {e}")
    finally:
        os.unlink(csv_file)
    
    all_results['format_support'] = format_tests
    
    # Test 2: Performance benchmarks
    print("\n2. Running Performance Benchmarks")
    print("-" * 40)
    
    try:
        size_benchmarks = benchmarks.benchmark_file_sizes()
        all_results['size_benchmarks'] = size_benchmarks
        
        for size, result in size_benchmarks.items():
            if result['success']:
                print(f"✓ {size.capitalize()}: {result['throughput_features_per_sec']:.1f} features/sec")
            else:
                print(f"✗ {size.capitalize()}: Failed")
    except Exception as e:
        print(f"✗ Size benchmarks failed: {e}")
        all_results['size_benchmarks'] = {'error': str(e)}
    
    # Test 3: Concurrent uploads
    print("\n3. Testing Concurrent Upload Handling")
    print("-" * 40)
    
    try:
        concurrent_results = benchmarks.benchmark_concurrent_uploads(TEST_CONFIG['concurrent_uploads'])
        all_results['concurrent_uploads'] = concurrent_results
        
        print(f"✓ Concurrent uploads: {concurrent_results['success_rate']:.1%} success rate")
        print(f"  - {concurrent_results['successful_uploads']}/{concurrent_results['concurrent_uploads']} succeeded")
        print(f"  - Average processing time: {concurrent_results['average_processing_time']:.2f}s")
    except Exception as e:
        print(f"✗ Concurrent upload test failed: {e}")
        all_results['concurrent_uploads'] = {'error': str(e)}
    
    # Test 4: Format-specific benchmarks
    print("\n4. Format-Specific Performance")
    print("-" * 40)
    
    try:
        format_benchmarks = benchmarks.benchmark_file_formats()
        all_results['format_benchmarks'] = format_benchmarks
        
        for fmt, result in format_benchmarks.items():
            if result['success']:
                print(f"✓ {fmt.upper()}: {result['features_per_second']:.1f} features/sec")
            else:
                print(f"✗ {fmt.upper()}: Failed")
    except Exception as e:
        print(f"✗ Format benchmarks failed: {e}")
        all_results['format_benchmarks'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("E2E Test Summary")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict) and 'error' not in results:
            for test_name, result in results.items():
                total_tests += 1
                if isinstance(result, dict) and result.get('success'):
                    passed_tests += 1
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}" if total_tests > 0 else "No tests run")
    
    # Save detailed results
    results_file = "e2e_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    # Check if API is available before running tests
    try:
        response = requests.get(f"{TEST_CONFIG['api_base_url']}/health", timeout=10)
        if response.status_code == 200:
            print("✓ API server is available")
            run_comprehensive_e2e_tests()
        else:
            print("✗ API server is not responding correctly")
            print("Please ensure the API server is running before executing E2E tests")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to API server: {e}")
        print("Please start the API server and try again")
        print(f"Expected URL: {TEST_CONFIG['api_base_url']}")