"""
A2A World Platform - Performance Optimization and Benchmarking

Advanced performance monitoring, optimization utilities, and benchmarking
tools for the A2A World REST API.
"""

import time
import asyncio
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    endpoint: str
    method: str
    response_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    requests_per_second: float
    success_rate: float
    error_count: int
    timestamp: datetime

@dataclass
class OptimizationReport:
    """Performance optimization report."""
    endpoint: str
    current_performance: PerformanceBenchmark
    optimizations_applied: List[str]
    improvement_percentage: float
    recommendations: List[str]

class PerformanceProfiler:
    """Advanced performance profiling utilities."""
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
        self.optimization_history = []
        
    @asynccontextmanager
    async def profile_request(self, request_id: str, endpoint: str):
        """Context manager for profiling individual requests."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            
            profile_data = {
                "request_id": request_id,
                "endpoint": endpoint,
                "duration_ms": (end_time - start_time) * 1000,
                "memory_delta_mb": end_memory - start_memory,
                "cpu_usage_percent": (start_cpu + end_cpu) / 2,
                "timestamp": datetime.utcnow()
            }
            
            self.profiles[request_id] = profile_data
            
            # Log slow requests
            if profile_data["duration_ms"] > 1000:
                logger.warning(
                    f"Slow request detected: {endpoint} took {profile_data['duration_ms']:.2f}ms"
                )
    
    def get_endpoint_statistics(self, endpoint: str) -> Dict[str, Any]:
        """Get performance statistics for a specific endpoint."""
        endpoint_profiles = [
            p for p in self.profiles.values() 
            if p.get("endpoint") == endpoint
        ]
        
        if not endpoint_profiles:
            return {"error": "No profile data available for endpoint"}
        
        durations = [p["duration_ms"] for p in endpoint_profiles]
        memory_deltas = [p["memory_delta_mb"] for p in endpoint_profiles]
        cpu_usages = [p["cpu_usage_percent"] for p in endpoint_profiles]
        
        return {
            "endpoint": endpoint,
            "total_requests": len(endpoint_profiles),
            "average_response_time_ms": sum(durations) / len(durations),
            "min_response_time_ms": min(durations),
            "max_response_time_ms": max(durations),
            "p95_response_time_ms": sorted(durations)[int(len(durations) * 0.95)],
            "average_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "average_cpu_percent": sum(cpu_usages) / len(cpu_usages),
            "slow_requests_count": len([d for d in durations if d > 1000]),
            "slow_requests_percentage": len([d for d in durations if d > 1000]) / len(durations) * 100
        }
    
    def cleanup_old_profiles(self, hours: int = 24):
        """Clean up old profile data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        old_profiles = [
            req_id for req_id, profile in self.profiles.items()
            if profile["timestamp"] < cutoff_time
        ]
        
        for req_id in old_profiles:
            del self.profiles[req_id]
        
        logger.info(f"Cleaned up {len(old_profiles)} old profiles")

class PerformanceBenchmarker:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self):
        self.benchmarks = {}
        self.baseline_metrics = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def run_endpoint_benchmark(
        self,
        endpoint_func: Callable,
        test_data: Any = None,
        concurrent_requests: int = 10,
        duration_seconds: int = 30
    ) -> PerformanceBenchmark:
        """Run comprehensive benchmark for an endpoint."""
        logger.info(f"Starting benchmark: {concurrent_requests} concurrent requests for {duration_seconds}s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        request_count = 0
        error_count = 0
        response_times = []
        
        # Monitor system resources
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_samples = []
        
        async def make_request():
            nonlocal request_count, error_count
            
            request_start = time.time()
            try:
                if asyncio.iscoroutinefunction(endpoint_func):
                    await endpoint_func(test_data) if test_data else await endpoint_func()
                else:
                    endpoint_func(test_data) if test_data else endpoint_func()
                request_count += 1
            except Exception as e:
                error_count += 1
                logger.debug(f"Benchmark request error: {e}")
            finally:
                response_times.append((time.time() - request_start) * 1000)
        
        # Run concurrent requests
        tasks = []
        while time.time() < end_time:
            # Start batch of concurrent requests
            batch_tasks = [make_request() for _ in range(concurrent_requests)]
            tasks.extend(batch_tasks)
            
            # Sample CPU usage
            cpu_samples.append(psutil.cpu_percent())
            
            # Wait a bit before next batch
            await asyncio.sleep(0.1)
        
        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        actual_duration = time.time() - start_time
        
        benchmark = PerformanceBenchmark(
            endpoint=endpoint_func.__name__,
            method="BENCHMARK",
            response_time_ms=sum(response_times) / len(response_times) if response_times else 0,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            requests_per_second=request_count / actual_duration,
            success_rate=(request_count / (request_count + error_count)) * 100 if (request_count + error_count) > 0 else 0,
            error_count=error_count,
            timestamp=datetime.utcnow()
        )
        
        self.benchmarks[endpoint_func.__name__] = benchmark
        
        logger.info(
            f"Benchmark completed: {benchmark.requests_per_second:.2f} req/s, "
            f"{benchmark.response_time_ms:.2f}ms avg, {benchmark.success_rate:.2f}% success"
        )
        
        return benchmark
    
    async def benchmark_all_endpoints(self, endpoint_registry: Dict[str, Callable]) -> Dict[str, PerformanceBenchmark]:
        """Benchmark all registered endpoints."""
        results = {}
        
        for name, endpoint_func in endpoint_registry.items():
            logger.info(f"Benchmarking endpoint: {name}")
            
            try:
                benchmark = await self.run_endpoint_benchmark(
                    endpoint_func,
                    concurrent_requests=5,  # Lighter load for full suite
                    duration_seconds=10
                )
                results[name] = benchmark
                
                # Brief pause between benchmarks
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Benchmark failed for {name}: {e}")
                continue
        
        return results
    
    def compare_benchmarks(
        self, 
        baseline: PerformanceBenchmark, 
        current: PerformanceBenchmark
    ) -> Dict[str, Any]:
        """Compare two benchmark results."""
        return {
            "endpoint": current.endpoint,
            "response_time_change_percent": (
                (current.response_time_ms - baseline.response_time_ms) / baseline.response_time_ms * 100
            ),
            "throughput_change_percent": (
                (current.requests_per_second - baseline.requests_per_second) / baseline.requests_per_second * 100
            ),
            "memory_change_percent": (
                (current.memory_usage_mb - baseline.memory_usage_mb) / baseline.memory_usage_mb * 100
                if baseline.memory_usage_mb > 0 else 0
            ),
            "success_rate_change": current.success_rate - baseline.success_rate,
            "performance_improved": (
                current.response_time_ms < baseline.response_time_ms and
                current.requests_per_second > baseline.requests_per_second
            )
        }

class PerformanceOptimizer:
    """Automated performance optimization system."""
    
    def __init__(self):
        self.optimization_rules = [
            self._optimize_database_queries,
            self._optimize_caching,
            self._optimize_memory_usage,
            self._optimize_response_compression
        ]
        self.applied_optimizations = {}
    
    async def analyze_performance(self, benchmark: PerformanceBenchmark) -> OptimizationReport:
        """Analyze performance and suggest optimizations."""
        recommendations = []
        potential_optimizations = []
        
        # Analyze response time
        if benchmark.response_time_ms > 1000:
            recommendations.append("Response time is slow (>1s). Consider caching or database optimization.")
            potential_optimizations.append("response_time_optimization")
        
        # Analyze memory usage
        if benchmark.memory_usage_mb > 100:
            recommendations.append("High memory usage detected. Consider memory optimization.")
            potential_optimizations.append("memory_optimization")
        
        # Analyze CPU usage
        if benchmark.cpu_percent > 80:
            recommendations.append("High CPU usage. Consider async operations or caching.")
            potential_optimizations.append("cpu_optimization")
        
        # Analyze throughput
        if benchmark.requests_per_second < 100:
            recommendations.append("Low throughput. Consider connection pooling or caching.")
            potential_optimizations.append("throughput_optimization")
        
        # Analyze error rate
        if benchmark.success_rate < 95:
            recommendations.append("High error rate detected. Review error handling and resilience.")
            potential_optimizations.append("error_handling_optimization")
        
        return OptimizationReport(
            endpoint=benchmark.endpoint,
            current_performance=benchmark,
            optimizations_applied=[],
            improvement_percentage=0.0,
            recommendations=recommendations
        )
    
    async def _optimize_database_queries(self, endpoint: str) -> List[str]:
        """Optimize database queries for endpoint."""
        optimizations = []
        
        # In a real implementation, this would:
        # 1. Analyze query patterns
        # 2. Add appropriate indexes
        # 3. Optimize query structure
        # 4. Implement connection pooling
        
        optimizations.append("Added database query optimization")
        logger.info(f"Applied database optimization for {endpoint}")
        
        return optimizations
    
    async def _optimize_caching(self, endpoint: str) -> List[str]:
        """Implement intelligent caching for endpoint."""
        optimizations = []
        
        # In a real implementation, this would:
        # 1. Identify cacheable responses
        # 2. Implement appropriate cache keys
        # 3. Set optimal TTL values
        # 4. Add cache warming strategies
        
        optimizations.append("Implemented intelligent caching")
        logger.info(f"Applied caching optimization for {endpoint}")
        
        return optimizations
    
    async def _optimize_memory_usage(self, endpoint: str) -> List[str]:
        """Optimize memory usage patterns."""
        optimizations = []
        
        # In a real implementation, this would:
        # 1. Implement object pooling
        # 2. Optimize data structures
        # 3. Add garbage collection tuning
        # 4. Implement streaming for large responses
        
        optimizations.append("Optimized memory usage patterns")
        logger.info(f"Applied memory optimization for {endpoint}")
        
        return optimizations
    
    async def _optimize_response_compression(self, endpoint: str) -> List[str]:
        """Implement response compression."""
        optimizations = []
        
        # In a real implementation, this would:
        # 1. Enable gzip compression
        # 2. Optimize JSON serialization
        # 3. Implement response streaming
        # 4. Add content negotiation
        
        optimizations.append("Enabled response compression")
        logger.info(f"Applied compression optimization for {endpoint}")
        
        return optimizations

class LoadTestRunner:
    """Advanced load testing system."""
    
    def __init__(self):
        self.test_scenarios = {}
        self.results = {}
    
    async def run_load_test(
        self,
        endpoint_func: Callable,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a comprehensive load test scenario."""
        scenario_name = scenario.get("name", "default")
        logger.info(f"Starting load test scenario: {scenario_name}")
        
        # Test configuration
        ramp_up_users = scenario.get("ramp_up_users", [1, 5, 10, 20, 50])
        duration_per_level = scenario.get("duration_per_level", 30)
        test_data = scenario.get("test_data")
        
        results = {
            "scenario": scenario_name,
            "levels": [],
            "summary": {}
        }
        
        for user_count in ramp_up_users:
            logger.info(f"Load test level: {user_count} concurrent users")
            
            # Run benchmark for this user level
            benchmark = await self.benchmarker.run_endpoint_benchmark(
                endpoint_func,
                test_data=test_data,
                concurrent_requests=user_count,
                duration_seconds=duration_per_level
            )
            
            level_result = {
                "concurrent_users": user_count,
                "requests_per_second": benchmark.requests_per_second,
                "average_response_time_ms": benchmark.response_time_ms,
                "success_rate": benchmark.success_rate,
                "memory_usage_mb": benchmark.memory_usage_mb,
                "cpu_percent": benchmark.cpu_percent
            }
            
            results["levels"].append(level_result)
            
            # Check for performance degradation
            if len(results["levels"]) > 1:
                previous = results["levels"][-2]
                current = level_result
                
                if current["requests_per_second"] < previous["requests_per_second"] * 0.8:
                    logger.warning(f"Performance degradation detected at {user_count} users")
                    results["performance_cliff"] = user_count
            
            # Brief pause between levels
            await asyncio.sleep(2)
        
        # Calculate summary statistics
        all_levels = results["levels"]
        results["summary"] = {
            "max_throughput": max(level["requests_per_second"] for level in all_levels),
            "min_response_time": min(level["average_response_time_ms"] for level in all_levels),
            "max_response_time": max(level["average_response_time_ms"] for level in all_levels),
            "optimal_user_count": max(
                all_levels,
                key=lambda x: x["requests_per_second"] / x["average_response_time_ms"]
            )["concurrent_users"]
        }
        
        self.results[scenario_name] = results
        logger.info(f"Load test completed: {scenario_name}")
        
        return results

# Performance monitoring decorators
def monitor_performance(
    track_memory: bool = True,
    track_cpu: bool = True,
    slow_threshold_ms: float = 1000
):
    """Decorator to monitor endpoint performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss if track_memory else 0
            start_cpu = psutil.cpu_percent() if track_cpu else 0
            
            try:
                result = await func(*args, **kwargs)
                
                # Record performance metrics
                duration_ms = (time.time() - start_time) * 1000
                
                if duration_ms > slow_threshold_ms:
                    logger.warning(
                        f"Slow endpoint {func.__name__}: {duration_ms:.2f}ms"
                    )
                
                # Log performance data
                perf_data = {
                    "endpoint": func.__name__,
                    "duration_ms": duration_ms,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if track_memory:
                    end_memory = psutil.Process().memory_info().rss
                    perf_data["memory_delta_mb"] = (end_memory - start_memory) / 1024 / 1024
                
                if track_cpu:
                    perf_data["cpu_percent"] = (start_cpu + psutil.cpu_percent()) / 2
                
                logger.debug(f"Performance: {json.dumps(perf_data)}")
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Endpoint {func.__name__} failed after {duration_ms:.2f}ms: {e}"
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                if duration_ms > slow_threshold_ms:
                    logger.warning(
                        f"Slow endpoint {func.__name__}: {duration_ms:.2f}ms"
                    )
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Endpoint {func.__name__} failed after {duration_ms:.2f}ms: {e}"
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Global instances
profiler = PerformanceProfiler()
benchmarker = PerformanceBenchmarker()
optimizer = PerformanceOptimizer()
load_tester = LoadTestRunner()
load_tester.benchmarker = benchmarker  # Inject benchmarker dependency

# Utility functions
async def run_performance_analysis(endpoint_registry: Dict[str, Callable]) -> Dict[str, Any]:
    """Run complete performance analysis on all endpoints."""
    logger.info("Starting comprehensive performance analysis")
    
    # Run benchmarks
    benchmarks = await benchmarker.benchmark_all_endpoints(endpoint_registry)
    
    # Generate optimization reports
    optimization_reports = {}
    for name, benchmark in benchmarks.items():
        report = await optimizer.analyze_performance(benchmark)
        optimization_reports[name] = report
    
    # Compile summary report
    analysis_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "benchmarks": {
            name: {
                "response_time_ms": b.response_time_ms,
                "requests_per_second": b.requests_per_second,
                "success_rate": b.success_rate,
                "memory_usage_mb": b.memory_usage_mb
            }
            for name, b in benchmarks.items()
        },
        "optimization_opportunities": {
            name: len(report.recommendations)
            for name, report in optimization_reports.items()
        },
        "performance_summary": {
            "fastest_endpoint": min(benchmarks.items(), key=lambda x: x[1].response_time_ms)[0],
            "highest_throughput": max(benchmarks.items(), key=lambda x: x[1].requests_per_second)[0],
            "most_reliable": max(benchmarks.items(), key=lambda x: x[1].success_rate)[0],
            "needs_attention": [
                name for name, report in optimization_reports.items()
                if len(report.recommendations) > 2
            ]
        }
    }
    
    logger.info("Performance analysis completed")
    return analysis_report

def get_system_resource_status() -> Dict[str, Any]:
    """Get current system resource utilization."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": {
            "percent": psutil.virtual_memory().percent,
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "used_gb": psutil.virtual_memory().used / (1024**3)
        },
        "disk": {
            "percent": psutil.disk_usage('/').percent,
            "free_gb": psutil.disk_usage('/').free / (1024**3)
        },
        "network": {
            "bytes_sent": psutil.net_io_counters().bytes_sent,
            "bytes_recv": psutil.net_io_counters().bytes_recv
        },
        "process": {
            "cpu_percent": psutil.Process().cpu_percent(),
            "memory_percent": psutil.Process().memory_percent(),
            "num_threads": psutil.Process().num_threads()
        }
    }