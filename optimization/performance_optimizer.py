# Real-time Performance Optimization for Harfang RL-LLM System
import asyncio
import time
import json
import numpy as np
import threading
from typing import Dict, Any, Optional, Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque, OrderedDict
import hashlib
import pickle


class SituationCache:
    """
    LRU cache for similar tactical situations to avoid redundant LLM calls
    """
    
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.95):
        """
        Initialize situation cache
        
        Args:
            max_size: Maximum number of cached situations
            similarity_threshold: Minimum similarity to consider a cache hit
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache = OrderedDict()  # LRU cache
        self.hit_count = 0
        self.miss_count = 0
        
        print(f"[CACHE] Situation cache initialized (max_size: {max_size})")
    
    def _calculate_situation_hash(self, features: Dict[str, Any]) -> str:
        """Calculate hash for tactical situation"""
        
        # Extract key tactical features for hashing
        key_features = {
            'distance_band': int(features.get('distance', 0) // 2000),  # 2km bands
            'engagement_phase': features.get('engagement_phase', 'UNKNOWN'),
            'threat_level_band': int(features.get('threat_level', 0) * 10),  # 0.1 bands
            'locked': features.get('locked', False),
            'energy_state': features.get('energy_state', 'MEDIUM'),
            'aspect_angle_band': int(features.get('aspect_angle', 0) // 15)  # 15° bands
        }
        
        # Create hash
        feature_string = json.dumps(key_features, sort_keys=True)
        return hashlib.md5(feature_string.encode()).hexdigest()[:12]
    
    def get(self, features: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Get cached response for similar situation"""
        
        situation_hash = self._calculate_situation_hash(features)
        
        if situation_hash in self.cache:
            # Move to end (most recently used)
            cached_data = self.cache.pop(situation_hash)
            self.cache[situation_hash] = cached_data
            
            self.hit_count += 1
            
            if self.hit_count % 100 == 0:  # Log every 100 hits
                hit_rate = self.hit_count / (self.hit_count + self.miss_count)
                print(f"[CACHE] Hit rate: {hit_rate:.1%} ({self.hit_count} hits)")
            
            return cached_data['response']
        
        self.miss_count += 1
        return None
    
    def put(self, features: Dict[str, Any], response: Tuple[float, Dict[str, Any]]):
        """Cache response for situation"""
        
        situation_hash = self._calculate_situation_hash(features)
        
        # Add to cache
        self.cache[situation_hash] = {
            'features': features,
            'response': response,
            'timestamp': time.time(),
            'access_count': 1
        }
        
        # Maintain max size (LRU eviction)
        if len(self.cache) > self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }


class AsyncLLMManager:
    """
    Asynchronous LLM manager for non-blocking tactical guidance
    """
    
    def __init__(self, llm_assistant, max_workers: int = 2, cache_size: int = 1000):
        """
        Initialize async LLM manager
        
        Args:
            llm_assistant: Base LLM assistant
            max_workers: Maximum concurrent LLM calls
            cache_size: Situation cache size
        """
        self.llm_assistant = llm_assistant
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = SituationCache(max_size=cache_size)
        
        # Async state management
        self.pending_requests = {}  # Dict[str, Future]
        self.last_response = (0.0, {"critique": "init"})
        self.response_queue = deque(maxlen=10)  # Recent responses
        
        # Performance tracking
        self.request_count = 0
        self.async_hit_count = 0
        self.average_response_time = 0.0
        
        print(f"[ASYNC LLM] Manager initialized with {max_workers} workers")
    
    def request_guidance_async(self, features: Dict[str, Any], 
                             step: int = 0) -> Tuple[float, Dict[str, Any]]:
        """
        Request tactical guidance with async processing and caching
        
        Args:
            features: Tactical features
            step: Current step number
        
        Returns:
            (shaping_delta, response_data) - may be cached or from previous request
        """
        
        # Check cache first
        cached_response = self.cache.get(features)
        if cached_response is not None:
            return cached_response
        
        # Generate request ID
        request_id = f"req_{step}_{int(time.time() * 1000) % 10000}"
        
        # Check if similar request is already pending
        if self._has_similar_pending_request(features):
            self.async_hit_count += 1
            return self.last_response  # Return last response immediately
        
        # Submit async request
        future = self.executor.submit(self._get_llm_guidance, features, step)
        self.pending_requests[request_id] = {
            'future': future,
            'features': features,
            'start_time': time.time()
        }
        
        # Return immediately with last response
        # The new response will be available for next call
        self.request_count += 1
        return self.last_response
    
    def _get_llm_guidance(self, features: Dict[str, Any], step: int) -> Tuple[float, Dict[str, Any]]:
        """Get LLM guidance (runs in background thread)"""
        
        try:
            start_time = time.time()
            
            # Use base assistant for guidance
            if hasattr(self.llm_assistant, 'request_shaping'):
                shaping_delta, response_data = self.llm_assistant.request_shaping(features, step)
            else:
                # Fallback
                shaping_delta, response_data = 0.0, {"critique": "async_fallback"}
            
            response_time = time.time() - start_time
            
            # Update performance tracking
            self.average_response_time = (
                self.average_response_time * 0.9 + response_time * 0.1
            )
            
            # Cache the response
            self.cache.put(features, (shaping_delta, response_data))
            
            # Update last response
            self.last_response = (shaping_delta, response_data)
            
            return (shaping_delta, response_data)
            
        except Exception as e:
            print(f"[ASYNC LLM] Error in background guidance: {e}")
            return (0.0, {"critique": f"async_error: {str(e)[:50]}", "error": True})
    
    def _has_similar_pending_request(self, features: Dict[str, Any]) -> bool:
        """Check if similar request is already being processed"""
        
        current_hash = self.cache._calculate_situation_hash(features)
        
        for request_data in self.pending_requests.values():
            if not request_data['future'].done():
                pending_hash = self.cache._calculate_situation_hash(request_data['features'])
                if current_hash == pending_hash:
                    return True
        
        return False
    
    def update_completed_requests(self):
        """Update completed async requests"""
        
        completed_requests = []
        
        for request_id, request_data in self.pending_requests.items():
            if request_data['future'].done():
                try:
                    # Get result
                    result = request_data['future'].result()
                    
                    # Update last response if this is recent
                    request_age = time.time() - request_data['start_time']
                    if request_age < 5.0:  # Use results from last 5 seconds
                        self.last_response = result
                    
                    completed_requests.append(request_id)
                    
                except Exception as e:
                    print(f"[ASYNC LLM] Request {request_id} failed: {e}")
                    completed_requests.append(request_id)
        
        # Remove completed requests
        for request_id in completed_requests:
            del self.pending_requests[request_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get async performance statistics"""
        
        cache_stats = self.cache.get_statistics()
        
        return {
            'total_requests': self.request_count,
            'async_hits': self.async_hit_count,
            'pending_requests': len(self.pending_requests),
            'average_response_time': self.average_response_time,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_size': cache_stats['cache_size']
        }


class OptimizedTacticalAssistant:
    """
    Performance-optimized tactical assistant with caching and async processing
    """
    
    def __init__(self, llm_assistant, enable_caching: bool = True,
                 enable_async: bool = True, max_rate_hz: float = 10.0):
        """
        Initialize optimized tactical assistant
        
        Args:
            llm_assistant: Base tactical assistant
            enable_caching: Enable situation caching
            enable_async: Enable async processing
            max_rate_hz: Maximum LLM call rate
        """
        self.base_assistant = llm_assistant
        self.enable_caching = enable_caching
        self.enable_async = enable_async
        self.max_rate_hz = max_rate_hz
        
        # Performance optimization components
        if enable_async:
            self.async_manager = AsyncLLMManager(llm_assistant, max_workers=2)
        else:
            self.async_manager = None
        
        if enable_caching:
            self.cache = SituationCache(max_size=1000)
        else:
            self.cache = None
        
        # Rate limiting
        self.last_call_time = 0.0
        self.min_interval = 1.0 / max_rate_hz
        
        # Performance monitoring
        self.performance_metrics = {
            'total_calls': 0,
            'cache_hits': 0,
            'async_responses': 0,
            'rate_limited_calls': 0,
            'average_response_time': 0.0
        }
        
        print(f"[OPTIMIZED ASSISTANT] Initialized with caching={enable_caching}, async={enable_async}")
    
    def request_shaping(self, features: Dict[str, Any], step: int = 0) -> Tuple[float, Dict[str, Any]]:
        """
        Optimized tactical shaping request with caching and async processing
        
        Args:
            features: Tactical features
            step: Current step number
        
        Returns:
            (shaping_delta, response_data)
        """
        
        self.performance_metrics['total_calls'] += 1
        start_time = time.time()
        
        # Rate limiting check
        now = time.time()
        if (now - self.last_call_time) < self.min_interval:
            self.performance_metrics['rate_limited_calls'] += 1
            # Return cached response
            if self.cache:
                cached = self.cache.get(features)
                if cached:
                    return cached
            return self.last_response if hasattr(self, 'last_response') else (0.0, {"critique": "rate_limited"})
        
        self.last_call_time = now
        
        # Try cache first
        if self.cache:
            cached_response = self.cache.get(features)
            if cached_response is not None:
                self.performance_metrics['cache_hits'] += 1
                return cached_response
        
        # Use async processing if enabled
        if self.async_manager:
            # Update any completed async requests first
            self.async_manager.update_completed_requests()
            
            # Get guidance (may be async)
            response = self.async_manager.request_guidance_async(features, step)
            self.performance_metrics['async_responses'] += 1
        else:
            # Synchronous fallback
            response = self.base_assistant.request_shaping(features)
        
        # Update performance metrics
        response_time = time.time() - start_time
        self.performance_metrics['average_response_time'] = (
            self.performance_metrics['average_response_time'] * 0.9 + response_time * 0.1
        )
        
        # Cache the response
        if self.cache:
            self.cache.put(features, response)
        
        self.last_response = response
        return response
    
    def extract_features(self, *args, **kwargs):
        """Extract features using base assistant"""
        return self.base_assistant.extract_features(*args, **kwargs)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics"""
        
        stats = self.performance_metrics.copy()
        
        # Add cache statistics
        if self.cache:
            cache_stats = self.cache.get_statistics()
            stats.update({f"cache_{k}": v for k, v in cache_stats.items()})
        
        # Add async statistics
        if self.async_manager:
            async_stats = self.async_manager.get_performance_stats()
            stats.update({f"async_{k}": v for k, v in async_stats.items()})
        
        # Calculate derived metrics
        if stats['total_calls'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calls']
            stats['rate_limit_rate'] = stats['rate_limited_calls'] / stats['total_calls']
        
        return stats
    
    def print_performance_summary(self):
        """Print performance optimization summary"""
        
        stats = self.get_optimization_stats()
        
        print(f"\n{'='*60}")
        print("PERFORMANCE OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total LLM calls: {stats['total_calls']}")
        print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
        print(f"Rate limited: {stats.get('rate_limit_rate', 0):.1%}")
        print(f"Avg response time: {stats['average_response_time']:.3f}s")
        
        if self.async_manager:
            print(f"Async responses: {stats['async_responses']}")
            print(f"Pending requests: {stats.get('async_pending_requests', 0)}")
        
        print(f"{'='*60}")


class BatchProcessor:
    """
    Batch processor for efficient handling of multiple simultaneous requests
    """
    
    def __init__(self, batch_size: int = 4, batch_timeout: float = 0.1):
        """
        Initialize batch processor
        
        Args:
            batch_size: Maximum batch size
            batch_timeout: Maximum time to wait for batch completion
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_batch = []
        self.last_batch_time = time.time()
        
        print(f"[BATCH PROCESSOR] Initialized (batch_size: {batch_size})")
    
    def add_request(self, features: Dict[str, Any], callback: Callable):
        """Add request to current batch"""
        
        self.pending_batch.append({
            'features': features,
            'callback': callback,
            'timestamp': time.time()
        })
        
        # Process batch if full or timeout reached
        if (len(self.pending_batch) >= self.batch_size or
            time.time() - self.last_batch_time > self.batch_timeout):
            self._process_batch()
    
    def _process_batch(self):
        """Process current batch of requests"""
        
        if not self.pending_batch:
            return
        
        print(f"[BATCH] Processing batch of {len(self.pending_batch)} requests")
        
        # Process all requests in batch (simplified - could be optimized further)
        for request in self.pending_batch:
            try:
                # Simulate batch processing (in real implementation, would optimize LLM calls)
                result = (0.0, {"critique": "batch_processed"})
                request['callback'](result)
            except Exception as e:
                print(f"[BATCH] Request failed: {e}")
        
        # Clear batch
        self.pending_batch = []
        self.last_batch_time = time.time()


class PerformanceMonitor:
    """
    Real-time performance monitoring for the RL-LLM system
    """
    
    def __init__(self, monitoring_window: int = 1000):
        """
        Initialize performance monitor
        
        Args:
            monitoring_window: Number of recent steps to monitor
        """
        self.monitoring_window = monitoring_window
        
        # Performance metrics
        self.step_times = deque(maxlen=monitoring_window)
        self.llm_times = deque(maxlen=monitoring_window)
        self.env_times = deque(maxlen=monitoring_window)
        self.total_times = deque(maxlen=monitoring_window)
        
        # System health metrics
        self.memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        
        self.start_time = time.time()
        
        print(f"[PERFORMANCE MONITOR] Initialized (window: {monitoring_window})")
    
    def record_step_timing(self, step_time: float, llm_time: float, env_time: float):
        """Record timing for a training step"""
        
        total_time = step_time
        
        self.step_times.append(step_time)
        self.llm_times.append(llm_time)
        self.env_times.append(env_time)
        self.total_times.append(total_time)
        
        # Log performance every 100 steps
        if len(self.step_times) % 100 == 0:
            self._log_performance_update()
    
    def _log_performance_update(self):
        """Log performance update"""
        
        if len(self.total_times) < 10:
            return
        
        # Calculate recent averages
        recent_steps = list(self.step_times)[-50:]  # Last 50 steps
        recent_llm = list(self.llm_times)[-50:]
        recent_env = list(self.env_times)[-50:]
        
        avg_step_time = np.mean(recent_steps)
        avg_llm_time = np.mean(recent_llm)
        avg_env_time = np.mean(recent_env)
        
        # Calculate steps per second
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        
        print(f"[PERF] Steps/sec: {steps_per_second:.1f} | "
              f"Step: {avg_step_time:.3f}s | LLM: {avg_llm_time:.3f}s | Env: {avg_env_time:.3f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        if not self.total_times:
            return {'status': 'no_data'}
        
        return {
            'total_steps_monitored': len(self.total_times),
            'average_step_time': np.mean(self.step_times),
            'average_llm_time': np.mean(self.llm_times),
            'average_env_time': np.mean(self.env_times),
            'steps_per_second': 1.0 / np.mean(self.step_times),
            'llm_time_percentage': np.mean(self.llm_times) / np.mean(self.total_times) * 100,
            'env_time_percentage': np.mean(self.env_times) / np.mean(self.total_times) * 100,
            'total_training_time': time.time() - self.start_time
        }


def create_optimized_training_loop(env, agent, tactical_assistant, total_timesteps: int,
                                 enable_optimizations: bool = True) -> Dict[str, Any]:
    """
    Create performance-optimized training loop
    
    Args:
        env: Training environment
        agent: RL agent
        tactical_assistant: Tactical assistant
        total_timesteps: Total training timesteps
        enable_optimizations: Enable performance optimizations
    
    Returns:
        Training results with performance metrics
    """
    
    print(f"[OPTIMIZED TRAINING] Starting optimized training loop")
    print(f"[OPTIMIZED TRAINING] Optimizations enabled: {enable_optimizations}")
    
    # Setup performance optimization
    if enable_optimizations:
        optimized_assistant = OptimizedTacticalAssistant(
            llm_assistant=tactical_assistant,
            enable_caching=True,
            enable_async=True,
            max_rate_hz=10.0
        )
        performance_monitor = PerformanceMonitor(monitoring_window=1000)
    else:
        optimized_assistant = tactical_assistant
        performance_monitor = None
    
    # Training loop with performance monitoring
    training_results = {
        'total_timesteps': total_timesteps,
        'optimization_enabled': enable_optimizations,
        'performance_metrics': {},
        'training_success': False
    }
    
    try:
        # Integrate optimized assistant with agent training
        if hasattr(agent, 'set_tactical_assistant'):
            agent.set_tactical_assistant(optimized_assistant)
        
        # Start training (this would integrate with actual RL training)
        print(f"[OPTIMIZED TRAINING] Training with optimizations...")
        
        # Simulate training steps for demonstration
        for step in range(min(100, total_timesteps // 1000)):  # Simulate 100 steps
            step_start = time.time()
            
            # Simulate environment step
            env_start = time.time()
            # env.step() would go here
            env_time = time.time() - env_start
            
            # Simulate LLM guidance
            llm_start = time.time()
            if enable_optimizations:
                mock_features = {'distance': 8000, 'threat_level': 0.3, 'step': step}
                optimized_assistant.request_shaping(mock_features, step)
            llm_time = time.time() - llm_start
            
            step_time = time.time() - step_start
            
            # Record performance
            if performance_monitor:
                performance_monitor.record_step_timing(step_time, llm_time, env_time)
        
        training_results['training_success'] = True
        
        # Get final performance metrics
        if enable_optimizations:
            training_results['performance_metrics'] = {
                'optimization_stats': optimized_assistant.get_optimization_stats(),
                'monitor_stats': performance_monitor.get_performance_summary() if performance_monitor else {}
            }
            
            # Print performance summary
            optimized_assistant.print_performance_summary()
        
    except Exception as e:
        print(f"[OPTIMIZED TRAINING] Training failed: {e}")
        training_results['error'] = str(e)
    
    return training_results


if __name__ == "__main__":
    print("Real-time Performance Optimization for Harfang RL-LLM")
    print("Provides caching, async processing, and performance monitoring")
    
    # Example usage
    print("\nExample: Create optimized assistant")
    print("optimized_assistant = OptimizedTacticalAssistant(base_assistant)")
    print("response = optimized_assistant.request_shaping(features)")
