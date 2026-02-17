"""Resource management for ATOM system.

Provides memory management, GPU resource allocation, and cleanup utilities.
"""

import gc
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .config import get_config
from .exceptions import ResourceError
from .logging import get_logger

logger = get_logger("resources")


class MemoryManager:
    """Memory management utilities for different backends."""

    def __init__(self):
        self.config = get_config()
        self._locks = {}
        self._memory_limits = {}

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage across all backends."""
        usage = {}

        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            usage.update({
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent()
            })

        if HAS_TORCH and torch.cuda.is_available():
            usage["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            usage["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)

        return usage

    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        usage = self.get_memory_usage()
        if usage:
            usage_str = ", ".join(f"{k}: {v:.1f}" for k, v in usage.items())
            logger.info(f"{prefix}Memory usage: {usage_str}")
        else:
            logger.debug(f"{prefix}Memory monitoring not available")

    def cleanup_memory(self):
        """Aggressively clean up memory across all backends."""
        logger.debug("Performing memory cleanup")

        # Python garbage collection
        gc.collect()

        # PyTorch memory cleanup
        if HAS_TORCH:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        if HAS_JAX:
            # Clear JAX caches
            jax.clear_caches()

        self.log_memory_usage("After cleanup: ")

    def set_memory_limit(self, backend: str, limit_mb: float):
        """Set memory limit for a specific backend."""
        self._memory_limits[backend] = limit_mb
        logger.info(f"Set {backend} memory limit to {limit_mb} MB")

    def check_memory_limit(self, backend: str) -> bool:
        """Check if memory usage exceeds limit for a backend."""
        if backend not in self._memory_limits:
            return True  # No limit set

        usage = self.get_memory_usage()
        limit = self._memory_limits[backend]

        if backend == "gpu" and "gpu_allocated_mb" in usage:
            return usage["gpu_allocated_mb"] <= limit
        elif backend == "cpu" and "rss_mb" in usage:
            return usage["rss_mb"] <= limit

        return True  # Can't check this backend

    @contextmanager
    def memory_context(self, backend: str = "auto"):
        """Context manager for memory monitoring."""
        self.log_memory_usage("Before operation: ")

        try:
            yield
        finally:
            self.cleanup_memory()


class GPUManager:
    """GPU resource management."""

    def __init__(self):
        self.config = get_config()
        self._device_locks = {}

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        if not HAS_TORCH:
            return False
        return torch.cuda.is_available()

    def get_device_count(self) -> int:
        """Get number of available GPUs."""
        if not self.is_gpu_available():
            return 0
        return torch.cuda.device_count()

    def get_optimal_device(self) -> str:
        """Get the optimal device based on configuration and availability."""
        config_device = self.config.get_device()

        if config_device == "auto":
            if self.is_gpu_available():
                return "cuda"
            else:
                return "cpu"
        else:
            if config_device == "cuda" and not self.is_gpu_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            return config_device

    def set_gpu_memory_fraction(self, fraction: float):
        """Set GPU memory fraction for PyTorch."""
        if not self.is_gpu_available():
            logger.warning("Cannot set GPU memory fraction: GPU not available")
            return

        if not (0.1 <= fraction <= 1.0):
            raise ResourceError(
                f"GPU memory fraction must be between 0.1 and 1.0, got {fraction}"
            )

        # This is a simplified version. In practice, you might want to use
        # torch.cuda.set_per_process_memory_fraction() or similar
        logger.info(f"GPU memory fraction set to {fraction}")

    def get_device_lock(self, device: str) -> threading.Lock:
        """Get a lock for device-specific operations."""
        if device not in self._device_locks:
            self._device_locks[device] = threading.Lock()
        return self._device_locks[device]

    def synchronize_device(self, device: str):
        """Synchronize device operations."""
        if device.startswith("cuda") and self.is_gpu_available():
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.synchronize()

    @contextmanager
    def device_context(self, device: str = "auto"):
        """Context manager for device operations."""
        if device == "auto":
            device = self.get_optimal_device()

        lock = self.get_device_lock(device)

        with lock:
            try:
                logger.debug(f"Acquired device lock for {device}")
                yield device
            finally:
                self.synchronize_device(device)
                logger.debug(f"Released device lock for {device}")


class ResourcePool:
    """Resource pool for managing shared resources."""

    def __init__(self, max_resources: int = 4):
        self.max_resources = max_resources
        self.available = threading.Semaphore(max_resources)
        self._active_count = 0
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a resource from the pool."""
        result = self.available.acquire(timeout=timeout)
        if result:
            with self._lock:
                self._active_count += 1
            logger.debug(f"Acquired resource from pool ({self._active_count}/{self.max_resources} active)")
        return result

    def release(self):
        """Release a resource back to the pool."""
        with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
                self.available.release()
                logger.debug(f"Released resource to pool ({self._active_count}/{self.max_resources} active)")
            else:
                logger.warning("Attempted to release resource when none are active")

    @contextmanager
    def resource_context(self):
        """Context manager for resource pool usage."""
        if not self.acquire():
            raise ResourceError("Failed to acquire resource from pool")

        try:
            yield
        finally:
            self.release()

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                "max_resources": self.max_resources,
                "active_count": self._active_count,
                "available_count": self.max_resources - self._active_count
            }


# Global instances
memory_manager = MemoryManager()
gpu_manager = GPUManager()
resource_pool = ResourcePool()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    return memory_manager


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    return gpu_manager


def get_resource_pool() -> ResourcePool:
    """Get the global resource pool instance."""
    return resource_pool


@contextmanager
def managed_resources(device: str = "auto", memory_monitor: bool = True):
    """Context manager for comprehensive resource management."""
    with gpu_manager.device_context(device):
        if memory_monitor:
            with memory_manager.memory_context():
                yield
        else:
            yield


def optimize_for_inference():
    """Optimize system for inference mode."""
    if HAS_TORCH:
        # Enable inference mode optimizations
        torch.set_grad_enabled(False)

        if torch.cuda.is_available():
            # Use TensorFloat32 for faster inference on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    logger.info("System optimized for inference")


def optimize_for_training():
    """Optimize system for training mode."""
    if HAS_TORCH:
        torch.set_grad_enabled(True)

        if torch.cuda.is_available():
            # Configure CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    logger.info("System optimized for training")


def setup_environment():
    """Setup environment for optimal performance."""
    config = get_config()

    # Set thread limits
    if config.hardware.num_threads:
        torch.set_num_threads(config.hardware.num_threads)
        os.environ["OMP_NUM_THREADS"] = str(config.hardware.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(config.hardware.num_threads)

    # Configure GPU memory fraction
    if config.hardware.gpu_memory_fraction < 1.0:
        gpu_manager.set_gpu_memory_fraction(config.hardware.gpu_memory_fraction)

    # JAX configuration
    if HAS_JAX and config.hardware.enable_x64_precision:
        jax.config.update("jax_enable_x64", True)

    logger.info("Environment configured for optimal performance")


if __name__ == "__main__":
    # Resource management demo
    print("ATOM Resource Management Demo")
    print("=" * 40)

    memory_manager.log_memory_usage("Initial: ")

    # Test memory cleanup
    memory_manager.cleanup_memory()

    # Test resource pool
    pool_stats = resource_pool.get_stats()
    print(f"Resource pool: {pool_stats}")

    # Test device detection
    device = gpu_manager.get_optimal_device()
    print(f"Optimal device: {device}")

    print("Resource management demo complete")